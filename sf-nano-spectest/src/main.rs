mod discovery;
mod summary;
mod types;
mod wast_test_runner;

use discovery::{find_wast_files, should_skip_test};
use log::{error, info, warn};
use sf_nano_core::{
    active_runtime_engine, reset_native_runtime_state, set_backend_mode, set_reference_backend,
    set_reference_backend_mode, BackendMode, ReferenceBackendMode,
};
use std::{env, panic::AssertUnwindSafe, path::Path, sync::Mutex, time::Instant};
use structopt::StructOpt;
use summary::print_summary;
use types::TestStats;
use wast_test_runner::{TestResult, WastTestRunner};

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

    /// Use the debug-only native emulator backend
    #[structopt(long = "emu")]
    emu: bool,

    /// Use the debug-only native emulator backend with a 64-bit target profile
    #[structopt(long = "emu64", conflicts_with_all = &["emu", "emu32"])]
    emu64: bool,

    /// Use the debug-only native emulator backend with a 32-bit target profile
    #[structopt(long = "emu32", conflicts_with_all = &["emu", "emu64"])]
    emu32: bool,
}

fn main() {
    let args = Cli::from_args();

    if let Some(backend) = &args.backend {
        let mode = BackendMode::parse_str(backend).unwrap_or_else(|| {
            eprintln!(
                "invalid backend '{}'; expected one of: auto, native, fusion, base (compat: jit)",
                backend
            );
            std::process::exit(1);
        });
        set_backend_mode(mode);
    }
    let reference_mode = if args.emu || args.emu64 {
        ReferenceBackendMode::Emu64
    } else if args.emu32 {
        ReferenceBackendMode::Emu32
    } else {
        ReferenceBackendMode::Disabled
    };
    if let Err(err) = if args.emu {
        set_reference_backend(true)
    } else {
        set_reference_backend_mode(reference_mode)
    } {
        eprintln!("Error: {}", err);
        std::process::exit(1);
    }
    let runtime_engine = active_runtime_engine().unwrap_or_else(|err| {
        let requested = args
            .backend
            .as_deref()
            .and_then(BackendMode::parse_str)
            .unwrap_or(BackendMode::Native);
        eprintln!(
            "backend '{}' is unavailable in this build: {}",
            requested.as_str(),
            err
        );
        std::process::exit(1);
    });
    print_runtime_engine(runtime_engine);

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
                .parent()
                .unwrap()
                .join("target")
                .join("webassembly-testsuite-2.0");

            warn!(
                "TESTSUITE_DIR not set, using default: {}",
                default_dir.display()
            );
            default_dir
        }
    };

    if !testsuite_dir.exists() {
        error!(
            "Testsuite directory not found at: {}",
            testsuite_dir.display()
        );
        std::process::exit(1);
    }

    info!("Using testsuite from: {}", testsuite_dir.display());

    run_wast_tests(&testsuite_dir, &args.filters);
}

fn print_runtime_engine(engine: sf_nano_core::RuntimeEngine) {
    match engine {
        sf_nano_core::RuntimeEngine::Interpreter => eprintln!("[runtime] interpreter"),
        sf_nano_core::RuntimeEngine::MicroJit(backend) => {
            eprintln!("[runtime] micro-jit backend={backend}");
        }
    }
}

fn run_wast_tests(testsuite_dir: &Path, filters: &[String]) {
    let start_time = Instant::now();

    let wast_files = find_wast_files(testsuite_dir);
    info!("Found {} WAST files", wast_files.len());

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
                files.push(abs_path);
            } else {
                error!("File not found: {}", file_path);
            }
        }

        if !name_filters.is_empty() {
            let matched: Vec<_> = wast_files
                .into_iter()
                .filter(|path| {
                    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
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
        wast_files
    };

    let filter_desc = if !filters.is_empty() {
        format!(" (filtered by: {})", filters.join(", "))
    } else {
        String::new()
    };

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

    for wast_file in filtered_files {
        let test_name = wast_file
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown");

        let full_path = wast_file
            .strip_prefix(testsuite_dir)
            .unwrap_or(&wast_file)
            .to_string_lossy();

        if should_skip_test(test_name) || should_skip_test(&full_path) {
            stats.skipped += 1;
            warn!("SKIP {}: Feature not supported in sf-nano", test_name);
            continue;
        }

        let test_start = Instant::now();

        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            reset_native_runtime_state();
            let mut runner = WastTestRunner::new();
            runner.run_wast_file(&wast_file)
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
                info!("PASS {} ({:?})", test_name, duration);
            }
            TestResult::Fail(test_err) => {
                stats.failed += 1;
                error!("FAIL {} ({:?})", test_name, duration);
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
            TestResult::Skip(msg) => {
                stats.skipped += 1;
                info!("SKIP {}: {}", test_name, msg);
            }
            TestResult::Error(msg) => {
                stats.errored += 1;
                error!("ERROR {} ({:?})", test_name, duration);
                error!("  File: {}", wast_file.display());
                error!("  Error: {}", msg);
            }
        }
    }

    let total_duration = start_time.elapsed();
    print_summary(&stats, total_duration);
}
