#[cfg(feature = "interp")]
use crate::discovery::interpreter_excluded;
use crate::discovery::{find_wast_files, should_skip_test};
use crate::summary::print_summary;
use crate::types::TestStats;
#[cfg(feature = "jit")]
use crate::wast_test_runner::{TestResult, WastTestRunner};
use log::{error, info, warn};
use sf_nano_core::{reset_native_runtime_state, target_has_simd, Config, Engine, Tier};
use std::{
    env,
    panic::AssertUnwindSafe,
    path::{Path, PathBuf},
    sync::Mutex,
    thread,
    time::Instant,
};
use structopt::StructOpt;

/// WebAssembly specification test runner for sf-nano
#[derive(StructOpt)]
#[structopt(
    name = "sf-nano-spectest",
    about = "Run WebAssembly specification tests (sf-nano)"
)]
struct Cli {
    /// Test name filters (exact match on filename without .wast extension)
    /// or file paths ending with .wast
    #[structopt()]
    filters: Vec<String>,

    /// Log level (trace, debug, info, warn, error)
    #[structopt(long = "log-level")]
    log_level: Option<String>,

    /// Backend to use for execution (auto, native, fusion, base)
    #[structopt(long = "backend")]
    backend: Option<String>,

    /// Override per-function compiler RAM budget, e.g. 0, 400K, 2M.
    #[structopt(long = "compiler-ram-budget")]
    compiler_ram_budget: Option<String>,

    /// Run the suite against the interpreter instead of the JIT.
    #[structopt(long = "interp")]
    interp: bool,
}

pub(crate) fn run() {
    let args = Cli::from_args();

    // `--interp` selects the engine, not a different suite. One runner
    // drives both, so a directive means the same thing on either and the
    // gap between them shows up as a failure rather than as a difference
    // of opinion between two test harnesses.
    let mut tier = Tier::DEFAULT;
    if args.interp {
        #[cfg(feature = "interp")]
        {
            tier = Tier::Interp;
        }
        #[cfg(not(feature = "interp"))]
        {
            eprintln!("--interp requires building with the 'interp' feature");
            std::process::exit(1);
        }
    }
    if let Some(requested) = &args.backend {
        tier = Tier::parse_str(requested).unwrap_or_else(|| {
            eprintln!(
                "engine '{}' is not in this build; it has: {}",
                requested,
                engine_names()
            );
            std::process::exit(1);
        });
    }
    let mut config = Config::new().tier(tier);
    match compiler_ram_budget(args.compiler_ram_budget.as_deref()) {
        Ok(Some(bytes)) => config = config.compiler_ram_budget_bytes(bytes),
        Ok(None) => {}
        Err(err) => {
            eprintln!("{}", err);
            std::process::exit(1);
        }
    }
    let engine = Engine::new(config).unwrap_or_else(|err| {
        eprintln!("{err}");
        std::process::exit(1);
    });
    print_engine(tier);

    // Initialize logging
    if let Some(level_str) = &args.log_level {
        let log_level = match level_str.to_lowercase().as_str() {
            "trace" => log::LevelFilter::Trace,
            "debug" => log::LevelFilter::Debug,
            "info" => log::LevelFilter::Info,
            "warn" => log::LevelFilter::Warn,
            "error" => log::LevelFilter::Error,
            _ => {
                eprintln!(
                    "Invalid log level '{}'. Valid options: trace, debug, info, warn, error",
                    level_str
                );
                std::process::exit(1);
            }
        };
        env_logger::Builder::new()
            .filter_level(log_level)
            .try_init()
            .ok();
    } else {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
            .try_init()
            .ok();
    }

    info!("Starting sf-nano WAST-based WebAssembly spec test runner");

    let testsuite_dir = match env::var("TESTSUITE_DIR") {
        Ok(dir) => {
            info!("Using TESTSUITE_DIR from environment: {}", dir);
            Path::new(&dir).to_path_buf()
        }
        Err(_) => {
            let manifest_dir = env!("CARGO_MANIFEST_DIR");
            let default_dir = Path::new(manifest_dir)
                .parent()
                .unwrap()
                .join("target")
                .join("webassembly-testsuite");

            warn!(
                "TESTSUITE_DIR not set, using default: {}",
                default_dir.display()
            );
            default_dir
        }
    };

    // Keep this aligned with discovery: read_dir validates the testsuite and
    // avoids qemu-riscv32's optimized Rust std statx metadata crash.
    if let Err(err) = std::fs::read_dir(&testsuite_dir) {
        error!(
            "Testsuite directory not readable at {}: {}",
            testsuite_dir.display(),
            err
        );
        std::process::exit(1);
    }

    info!("Using testsuite from: {}", testsuite_dir.display());

    #[cfg(feature = "jit")]
    if !run_wast_tests_with_large_stack(engine, testsuite_dir, args.filters) {
        std::process::exit(1);
    }
    #[cfg(not(feature = "jit"))]
    {
        let _ = (testsuite_dir, args.filters);
        eprintln!("the wast suite runs on the jit engine; pass --interp for the interpreter suite");
        std::process::exit(1);
    }
}

#[cfg(feature = "jit")]
fn run_wast_tests_with_large_stack(
    engine: Engine,
    testsuite_dir: PathBuf,
    filters: Vec<String>,
) -> bool {
    let stack_size = recommended_worker_stack_size();
    let builder = thread::Builder::new()
        .name("sf-nano-spectest".into())
        .stack_size(stack_size);
    let handle = builder
        .spawn(move || run_wast_tests(engine, &testsuite_dir, &filters))
        .unwrap_or_else(|err| panic!("failed to spawn spectest worker thread: {err}"));
    handle
        .join()
        .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
}

#[cfg(feature = "jit")]
/// The interpreter and JIT stacks are sized per engine now, but this
/// thread is spawned before one exists, so the sizing constant is local.
const WASM_STACK_BYTES_FOR_STACK_SIZING: usize = 2 * 1024 * 1024;

#[cfg(feature = "jit")]
fn recommended_worker_stack_size() -> usize {
    // Sized by sf-nano-core's configured Wasm operand/call stack —
    // the worker OS thread must be large enough to hold everything the
    // JIT might push. 4× headroom over the Wasm stack, with an 8 MiB
    // floor for deep-recursion spec tests.
    std::cmp::max(WASM_STACK_BYTES_FOR_STACK_SIZING * 4, 8 * 1024 * 1024)
}

/// The compiler RAM budget the flag or the environment asks for, if any.
fn compiler_ram_budget(arg: Option<&str>) -> Result<Option<u32>, String> {
    let value = match arg {
        Some(value) => Some(value.to_string()),
        None => match env::var("SF_NANO_COMPILER_RAM_BUDGET") {
            Ok(value) => Some(value),
            Err(env::VarError::NotPresent) => None,
            Err(err) => {
                return Err(format!(
                    "could not read SF_NANO_COMPILER_RAM_BUDGET: {}",
                    err
                ))
            }
        },
    };
    let Some(value) = value else {
        return Ok(None);
    };
    parse_byte_size(&value)
        .map(Some)
        .ok_or_else(|| format!("invalid compiler RAM budget '{}'", value))
}

fn parse_byte_size(text: &str) -> Option<u32> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    let (digits, suffix) = trimmed.split_at(
        trimmed
            .find(|ch: char| !ch.is_ascii_digit())
            .unwrap_or(trimmed.len()),
    );
    if digits.is_empty() {
        return None;
    }
    let value = digits.parse::<u64>().ok()?;
    let multiplier = match suffix.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1,
        "k" | "kb" | "kib" => 1024,
        "m" | "mb" | "mib" => 1024 * 1024,
        _ => return None,
    };
    value.checked_mul(multiplier)?.try_into().ok()
}

/// The engine names this build accepts, for error messages.
fn engine_names() -> String {
    let mut names: Vec<&str> = Tier::ALL.iter().map(|t| t.as_str()).collect();
    names.push("auto");
    names.join(", ")
}

fn print_engine(tier: Tier) {
    match tier {
        #[cfg(feature = "jit")]
        Tier::Jit => eprintln!(
            "[runtime] jit backend={}",
            sf_nano_core::active_native_backend_name()
        ),
        #[cfg(feature = "interp")]
        Tier::Interp => eprintln!("[runtime] interpreter (native dispatch)"),
    }
}

#[cfg(feature = "jit")]
enum WastInput {
    File(PathBuf),
    Embedded {
        path: &'static str,
        content: &'static str,
    },
}

#[cfg(feature = "jit")]
impl WastInput {
    fn path(&self) -> &Path {
        match self {
            Self::File(path) => path,
            Self::Embedded { path, .. } => Path::new(path),
        }
    }

    fn run(&self, runner: &mut WastTestRunner) -> TestResult {
        match self {
            Self::File(path) => runner.run_wast_file(path),
            Self::Embedded { content, .. } => runner.run_wast_content(content),
        }
    }
}

#[cfg(feature = "jit")]
fn run_wast_tests(engine: Engine, testsuite_dir: &Path, filters: &[String]) -> bool {
    let start_time = Instant::now();

    let mut wast_inputs: Vec<_> = find_wast_files(testsuite_dir)
        .into_iter()
        .map(WastInput::File)
        .collect();
    wast_inputs.push(WastInput::Embedded {
        path: "runtime_world_gc_funcref.wast",
        content: include_str!("../tests/runtime_world_gc_funcref.wast"),
    });
    wast_inputs.push(WastInput::Embedded {
        path: "regalloc_scratch_saturation.wast",
        content: include_str!("../tests/regalloc_scratch_saturation.wast"),
    });
    wast_inputs.sort_by(|left, right| left.path().cmp(right.path()));
    info!("Found {} WAST files", wast_inputs.len());

    // Separate filters into file paths (.wast) and test names
    let (file_path_filters, name_filters): (Vec<_>, Vec<_>) =
        filters.iter().partition(|f| f.ends_with(".wast"));

    let filtered_files: Vec<_> = if !filters.is_empty() {
        let mut files = Vec::new();

        for file_path in file_path_filters {
            let path = Path::new(file_path);
            let abs_path = if path.is_absolute() {
                path.to_path_buf()
            } else {
                std::env::current_dir()
                    .unwrap_or_else(|_| Path::new(".").to_path_buf())
                    .join(path)
            };

            if abs_path.exists() && abs_path.is_file() {
                files.push(WastInput::File(abs_path));
            } else {
                error!("File not found: {}", file_path);
            }
        }

        if !name_filters.is_empty() {
            let matched: Vec<_> = wast_inputs
                .into_iter()
                .filter(|input| {
                    if let Some(stem) = input.path().file_stem().and_then(|s| s.to_str()) {
                        name_filters.iter().any(|f| f.as_str() == stem)
                    } else {
                        false
                    }
                })
                .collect();
            files.extend(matched);
        }

        files
    } else {
        wast_inputs
    };

    let filter_desc = if !filters.is_empty() {
        format!(" (filtered by: {})", filters.join(", "))
    } else {
        String::new()
    };

    let simd_enabled = target_has_simd();

    info!("Running {} tests{}", filtered_files.len(), filter_desc);

    let mut stats = TestStats::new();

    // Install a panic hook that captures messages instead of printing to stderr.
    // This prevents expected panics (from assert_invalid/assert_unlinkable catch_unwind)
    // from cluttering the output, while still making messages available for unexpected panics.
    let last_panic: std::sync::Arc<Mutex<Option<String>>> = std::sync::Arc::new(Mutex::new(None));
    let last_panic_hook = last_panic.clone();
    std::panic::set_hook(Box::new(move |info| {
        let msg = if let Some(s) = info.payload().downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "<unknown>".to_string()
        };
        let location = info
            .location()
            .map(|l| format!(" at {}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_default();
        let full_msg = format!("{}{}", msg, location);
        // Print backtrace if RUST_BACKTRACE is set (for debugging)
        if std::env::var("RUST_BACKTRACE").is_ok() {
            eprintln!(
                "panic: {}\n{}",
                full_msg,
                std::backtrace::Backtrace::force_capture()
            );
        }
        *last_panic_hook.lock().unwrap() = Some(full_msg);
    }));

    for wast_input in filtered_files {
        let wast_file = wast_input.path();
        let test_name = wast_file
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown");

        let full_path = wast_file
            .strip_prefix(testsuite_dir)
            .unwrap_or(&wast_file)
            .to_string_lossy();
        let display_name = full_path.as_ref();

        // Out of scope rather than unsupported: post-3.0 proposals, the
        // superseded exception-handling spelling, and custom-section
        // annotations. They are not counted, because a total that mixes
        // "we do not do this" with "nobody targets this" cannot answer
        // the only question worth asking -- do we pass Wasm 3.0.
        if should_skip_test(test_name, simd_enabled) || should_skip_test(&full_path, simd_enabled) {
            continue;
        }
        // Out of scope for this ENGINE, rather than for the project. The
        // JIT implements SIMD and GC and runs every one of these, so the
        // check is tier-aware or it would cost the JIT its 100%.
        #[cfg(feature = "interp")]
        if engine.tier() == Tier::Interp && interpreter_excluded(test_name).is_some() {
            continue;
        }

        let test_start = Instant::now();

        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            reset_native_runtime_state();
            let mut runner = WastTestRunner::new(engine);
            wast_input.run(&mut runner)
        }));

        let duration = test_start.elapsed();

        // Convert panics to Error results
        let result = match result {
            Ok(r) => r,
            Err(_) => {
                let msg = last_panic
                    .lock()
                    .unwrap()
                    .take()
                    .unwrap_or_else(|| "<unknown>".to_string());
                TestResult::Error(format!("panic: {}", msg))
            }
        };

        match result {
            TestResult::Pass => {
                stats.passed += 1;
                info!("PASS {} ({:?})", display_name, duration);
            }
            TestResult::Fail(test_err) => {
                stats.failed += 1;
                error!("FAIL {} ({:?})", display_name, duration);
                error!("  File: {}", wast_file.display());
                match test_err.wasm_error() {
                    Some(wasm_err) => {
                        error!("  Expected: {}", test_err.context().unwrap_or("unknown"));
                        error!("  Actual:   wasmerror \"{}\"", wasm_err);
                    }
                    None => {
                        error!("  Error: {}", test_err);
                    }
                }
            }
            TestResult::Error(msg) => {
                stats.errored += 1;
                error!("ERROR {} ({:?})", display_name, duration);
                error!("  File: {}", wast_file.display());
                error!("  Error: {}", msg);
            }
        }
    }

    let total_duration = start_time.elapsed();
    print_summary(&stats, total_duration);
    stats.failed == 0 && stats.errored == 0
}
