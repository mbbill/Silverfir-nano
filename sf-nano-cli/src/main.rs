#[cfg(feature = "jit")]
use sf_nano_core::jit_stats_snapshot;
use sf_nano_core::wasi::{set_wasi_ctx, wasi_imports, WasiContextBuilder};
use sf_nano_core::{Config, Engine, Tier};

use std::path::PathBuf;
use std::{env, fs, process};

#[cfg(feature = "memprof")]
use memprof_report::Session as MemprofSession;
#[cfg(feature = "memprof")]
use std::alloc::System;
#[cfg(feature = "memprof")]
use tracked_alloc::TrackingAllocator;

#[cfg(feature = "memprof")]
#[global_allocator]
static GLOBAL_ALLOCATOR: TrackingAllocator<System> = TrackingAllocator::new(System);

fn main() {
    let args: Vec<String> = env::args().collect();
    process::exit(run_cli(&args));
}

fn run_cli(args: &[String]) -> i32 {
    if args.len() < 2 {
        print_usage(&args[0]);
        return 1;
    }
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        print_usage(&args[0]);
        return 0;
    }

    let mut preopens: Vec<(String, PathBuf)> = Vec::new();
    let mut tier = Tier::DEFAULT;
    #[cfg(feature = "interp")]
    let mut interp_stats = false;
    let mut compile_only = false;
    let mut compiler_ram_budget: Option<u32> = None;
    let mut parallel_compilation = true;
    let mut remaining_args: Vec<String> = Vec::new();
    #[cfg(feature = "memprof")]
    let mut memprof_enabled = false;
    #[cfg(feature = "memprof")]
    let mut memprof_report_path: Option<PathBuf> = None;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--" {
            remaining_args.extend(args[i + 1..].iter().cloned());
            break;
        } else if args[i] == "--memprof" {
            #[cfg(feature = "memprof")]
            {
                memprof_enabled = true;
            }
            #[cfg(not(feature = "memprof"))]
            {
                eprintln!("Error: --memprof requires building sf-nano-cli with --features memprof");
                return 1;
            }
        } else if args[i] == "--memprof-report" {
            i += 1;
            if i >= args.len() {
                eprintln!("Error: --memprof-report requires a path");
                return 1;
            }
            #[cfg(feature = "memprof")]
            {
                memprof_enabled = true;
                memprof_report_path = Some(PathBuf::from(&args[i]));
            }
            #[cfg(not(feature = "memprof"))]
            {
                eprintln!(
                    "Error: --memprof-report requires building sf-nano-cli with --features memprof"
                );
                return 1;
            }
        } else if args[i] == "--dir" {
            i += 1;
            if i < args.len() {
                let spec = &args[i];
                if let Some((guest, host)) = spec.split_once("::") {
                    if guest.is_empty() || host.is_empty() {
                        eprintln!("Error: --dir expects <guest::host> or <path>");
                        return 1;
                    }
                    preopens.push((guest.to_string(), PathBuf::from(host)));
                } else {
                    preopens.push((".".to_string(), PathBuf::from(spec)));
                }
            } else {
                eprintln!("Error: --dir requires <guest::host> or <path>");
                return 1;
            }
        } else if args[i] == "--engine" || args[i] == "--backend" {
            let flag = args[i].clone();
            i += 1;
            if i >= args.len() {
                eprintln!("Error: {} requires one of: {}", flag, engine_names());
                return 1;
            }
            let Some(parsed) = Tier::parse_str(&args[i]) else {
                eprintln!(
                    "Error: engine '{}' is not in this build; it has: {}",
                    args[i],
                    engine_names(),
                );
                return 1;
            };
            tier = parsed;
        } else if args[i] == "--interp" || args[i] == "--interp-stats" {
            #[cfg(feature = "interp")]
            {
                tier = Tier::Interp;
                if args[i] == "--interp-stats" {
                    interp_stats = true;
                }
            }
            #[cfg(not(feature = "interp"))]
            {
                eprintln!(
                    "Error: {} requires the interp feature; this build has: {}",
                    args[i],
                    engine_names()
                );
                return 1;
            }
        } else if args[i] == "--compile-only" || args[i] == "--no-run" {
            compile_only = true;
        } else if args[i] == "--no-parallel-compilation" {
            parallel_compilation = false;
        } else if args[i] == "--compiler-ram-budget" {
            i += 1;
            if i >= args.len() {
                eprintln!(
                    "Error: --compiler-ram-budget requires a byte count (for example 400K, 2M, or a raw number)"
                );
                return 1;
            }
            match parse_byte_size(&args[i]) {
                Some(bytes) => compiler_ram_budget = Some(bytes),
                None => {
                    eprintln!(
                        "Error: --compiler-ram-budget value '{}' is not a valid byte size",
                        args[i]
                    );
                    return 1;
                }
            }
        } else {
            remaining_args.push(args[i].clone());
        }
        i += 1;
    }

    #[cfg(feature = "memprof")]
    let memprof_session = if memprof_enabled {
        Some(MemprofSession::new(memprof_report_path.take(), args))
    } else {
        None
    };

    let exit_code = (|| {
        if remaining_args.is_empty() {
            eprintln!("Error: no wasm file specified");
            return 1;
        }

        if compiler_ram_budget.is_none() {
            match env::var("SF_NANO_COMPILER_RAM_BUDGET") {
                Ok(value) => match parse_byte_size(&value) {
                    Some(bytes) => compiler_ram_budget = Some(bytes),
                    None => {
                        eprintln!(
                            "Error: SF_NANO_COMPILER_RAM_BUDGET value '{}' is not a valid byte size",
                            value
                        );
                        return 1;
                    }
                },
                Err(env::VarError::NotPresent) => {}
                Err(err) => {
                    eprintln!("Error: could not read SF_NANO_COMPILER_RAM_BUDGET: {}", err);
                    return 1;
                }
            }
        }
        let mut config = Config::new()
            .tier(tier)
            .parallel_compilation(parallel_compilation);
        if let Some(bytes) = compiler_ram_budget {
            config = config.compiler_ram_budget_bytes(bytes);
        }
        let engine = match Engine::new(config) {
            Ok(engine) => engine,
            Err(err) => {
                eprintln!("Error: {err}");
                return 1;
            }
        };
        print_engine(tier);

        let path = PathBuf::from(&remaining_args[0]);
        let prog_args: Vec<String> = remaining_args[1..].to_vec();

        let data = match fs::read(&path) {
            Ok(data) => data,
            Err(err) => {
                eprintln!("Error reading '{}': {}", path.display(), err);
                return 1;
            }
        };

        let module_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("module");

        let mut wasi_args = vec![module_name.to_string()];
        wasi_args.extend(prog_args);

        let mut ctx_builder = WasiContextBuilder::new().args(&wasi_args);
        let wasi_test_only_specified =
            env::var("WASI_TEST_ONLY_SPECIFIED_VARS").ok().as_deref() == Some("1");
        if wasi_test_only_specified {
            for (key, value) in env::vars() {
                if key != "PATH" && key != "WASI_TEST_ONLY_SPECIFIED_VARS" {
                    ctx_builder = ctx_builder.env(&key, &value);
                }
            }
        } else {
            ctx_builder = ctx_builder.inherit_env();
        }
        if preopens.is_empty() {
            ctx_builder = ctx_builder.preopen_dir(".", std::path::Path::new("."));
        } else {
            for (guest, host) in &preopens {
                ctx_builder = ctx_builder.preopen_dir(guest, host.clone());
            }
        }
        let ctx = ctx_builder.build();
        set_wasi_ctx(ctx);

        run_module(
            &engine,
            &data,
            module_name,
            compile_only,
            #[cfg(feature = "interp")]
            interp_stats,
        )
    })();

    #[cfg(feature = "memprof")]
    if let Some(session) = memprof_session {
        match session.finish() {
            Ok(path) => eprintln!("[memprof] report={}", path.display()),
            Err(err) => {
                eprintln!("Error: memprof {}", err);
                if exit_code == 0 {
                    return 1;
                }
            }
        }
    }

    exit_code
}

/// Run the module on whichever engine was selected.
///
/// One path for both. The engine was chosen back in argument parsing and
/// `RuntimeWorld` honours it; nothing from here down knows or cares which one
/// is underneath.
fn run_module(
    engine: &Engine,
    data: &[u8],
    module_name: &str,
    compile_only: bool,
    #[cfg(feature = "interp")] interp_stats: bool,
) -> i32 {
    use sf_nano_core::{module::Module, RuntimeWorld};

    let module = match Module::new(module_name, data) {
        Ok(module) => module,
        Err(err) => {
            eprintln!("Error parsing module: {}", err);
            return 1;
        }
    };
    let imports = wasi_imports();
    let mut world = RuntimeWorld::new();
    let instance_id = match world.instantiate(engine, module, &imports) {
        Ok(instance_id) => instance_id,
        Err(err) => {
            eprintln!("Error instantiating module: {}", err.error());
            return 1;
        }
    };

    if compile_only {
        print_native_stats();
        return 0;
    }

    let instance = world
        .instance(instance_id)
        .expect("newly instantiated module missing from runtime world");
    let entry = if instance.has_function_export("_start") {
        "_start"
    } else {
        "main"
    };
    let result = world.invoke(instance_id, entry, &[]);

    print_native_stats();
    #[cfg(feature = "interp")]
    if interp_stats {
        let instance = world
            .instance(instance_id)
            .expect("invoked module missing from runtime world");
        print_interp_stats(instance);
    }

    match result {
        Ok(_) => 0,
        Err(err) => {
            if let Some(code) = err.exit_code() {
                return code;
            }
            eprintln!("Error: {}", err);
            1
        }
    }
}

/// Dispatch statistics the interpreter keeps and the JIT has no analogue
/// for -- the one place the CLI still asks which engine ran.
#[cfg(feature = "interp")]
fn print_interp_stats(instance: &sf_nano_core::Instance) {
    let Some(()) = instance.with_interp(|inst| {
        let native = inst.dispatch_count();
        if native > 0 && inst.dispatch_counting_enabled() {
            eprintln!("[interp] native dispatches: {native}");
        }
        let code_len = inst.engine_code_len();
        if code_len > 0 {
            eprintln!(
                "[interp] engine code: {code_len} bytes ({:.1} KB)",
                code_len as f64 / 1024.0
            );
        }
        let bigrams = inst.bigram_stats();
        if !bigrams.is_empty() {
            eprintln!("[interp] top fallthrough bigrams (static):");
            for ((a, b), n) in bigrams.iter().take(16) {
                eprintln!("[interp]   {a:?} -> {b:?}: {n}");
            }
        }
        let slow = inst.slow_exit_stats();
        if !slow.is_empty() {
            let total: u64 = slow.iter().map(|(_, n)| n).sum();
            eprintln!("[interp] slow exits: {total}");
            for (op, n) in slow.iter().take(12) {
                eprintln!("[interp]   {op:?}: {n}");
            }
        }
    }) else {
        eprintln!("[interp] --interp-stats: this module ran on another engine");
        return;
    };
}

fn print_usage(program_name: &str) {
    eprintln!("Silverfir-nano — WebAssembly interpreter");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!(
        "  {program_name} [--engine <name>] [--compile-only] [--no-parallel-compilation] [--dir <guest::host|path>] [--memprof] [--memprof-report <html>] <wasm-file> [args...]"
    );
    eprintln!();
    eprintln!("Run a WebAssembly module with WASI support.");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!(
        "  --engine <name>           Select the execution engine: {} (--backend is an alias).",
        engine_names()
    );
    eprintln!("  --interp                  Shorthand for --engine interp.");
    eprintln!(
        "  --interp-stats            Interpreter: print dispatch statistics (implies --interp)."
    );
    eprintln!("  --compile-only, --no-run  Compile/instantiate the module without invoking it.");
    eprintln!("  --no-parallel-compilation Force deterministic serial eager compilation.");
    eprintln!("  --compiler-ram-budget <n> Override the per-function compiler RAM budget.");
    eprintln!("  --dir <guest::host|path>  Preopen a host directory (repeatable).");
    #[cfg(feature = "memprof")]
    {
        eprintln!("  --memprof                 Record allocation events and emit an HTML report.");
        eprintln!(
            "  --memprof-report <html>   Write the report to this path. Default: /tmp/sf-nano-memprof-<pid>.html"
        );
        eprintln!();
        eprintln!("MEMPROF:");
        eprintln!("  Build/run:");
        eprintln!(
            "    cargo run --release -p sf-nano-cli --features memprof -- --memprof <wasm-file> [args...]"
        );
        eprintln!("  Shortcut:");
        eprintln!("    cargo memprof <wasm-file> [args...]");
        eprintln!("  Save report explicitly:");
        eprintln!("    cargo memprof --memprof-report /tmp/run.html <wasm-file> [args...]");
        eprintln!("  Example:");
        eprintln!(
            "    cargo memprof --memprof-report /tmp/lua.html benchmarks/wasi/lua/lua.wasm benchmarks/wasi/lua/fib_34.lua"
        );
        eprintln!();
        eprintln!("The HTML report shows the total usage curve. Click a point on the curve");
        eprintln!("to inspect the live allocation set at that time, grouped by type and size.");
    }
    #[cfg(not(feature = "memprof"))]
    {
        eprintln!(
            "  --memprof                 Requires building sf-nano-cli with --features memprof."
        );
        eprintln!(
            "  --memprof-report <html>   Requires building sf-nano-cli with --features memprof."
        );
    }
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

fn print_native_stats() {
    #[cfg(feature = "jit")]
    {
        let s = jit_stats_snapshot();
        if s.groups > 0 {
            let arch = if cfg!(target_arch = "aarch64") {
                "arm64"
            } else if cfg!(target_arch = "x86_64") {
                "x86_64"
            } else if cfg!(target_arch = "arm") {
                "armv7a"
            } else if cfg!(target_arch = "riscv32") {
                "riscv32"
            } else if cfg!(target_arch = "riscv64") {
                "riscv64"
            } else {
                "unknown"
            };
            eprintln!(
                "[{arch}] (func:{}, ssa:{}, mir:{}, code:{})",
                s.groups, s.ssa_ops, s.mir_ops, s.bytes_emitted
            );
        }
    }
}

/// The engine names this build accepts, for error messages.
fn engine_names() -> String {
    let mut names: Vec<&str> = Tier::ALL.iter().map(|t| t.as_str()).collect();
    names.push("auto");
    names.join(", ")
}

/// Announce the engine on stderr: stdout belongs to the guest program, which
/// the benchmark harness checksums.
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
