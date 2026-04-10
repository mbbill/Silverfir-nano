#[cfg(feature = "jit")]
use sf_nano_core::native_stats_snapshot;
use sf_nano_core::wasi::{set_wasi_ctx, wasi_imports, WasiContextBuilder};
use sf_nano_core::Instance;
use sf_nano_core::{
    active_runtime_engine, set_backend_mode, set_reference_backend_mode, BackendMode,
    ReferenceBackendMode,
};

#[cfg(feature = "memtrace")]
use std::alloc::System;
#[cfg(feature = "memtrace")]
use std::panic::{self, AssertUnwindSafe};
use std::path::PathBuf;
use std::{env, fs, process};

#[cfg(feature = "memtrace")]
#[global_allocator]
static ALLOCATOR: sf_nano_memtrace::TrackingAllocator<System> =
    sf_nano_memtrace::TrackingAllocator::new(System);

#[cfg(feature = "memtrace")]
#[derive(Debug)]
struct MemtraceOptions {
    enabled: bool,
    output_path: Option<PathBuf>,
}

#[cfg(feature = "memtrace")]
impl Default for MemtraceOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            output_path: None,
        }
    }
}

fn main() {
    #[cfg(feature = "memtrace")]
    {
        let original_args: Vec<String> = env::args().collect();
        let (memtrace_options, filtered_args) = match parse_memtrace_options(&original_args) {
            Ok(value) => value,
            Err(exit_code) => process::exit(exit_code),
        };
        let trace_output = if memtrace_options.enabled {
            match sf_nano_memtrace::configure_trace(
                original_args.clone(),
                memtrace_options.output_path.clone(),
            ) {
                Ok(path) => Some(path),
                Err(err) => {
                    eprintln!("Error: failed to open memtrace output: {}", err);
                    process::exit(1);
                }
            }
        } else {
            sf_nano_memtrace::disable_trace();
            None
        };

        let run_result = panic::catch_unwind(AssertUnwindSafe(|| run_cli(&filtered_args)));
        let exit_code = match run_result {
            Ok(code) => code,
            Err(payload) => {
                eprintln!("panic: {}", panic_payload_message(&payload));
                101
            }
        };
        finish_tracing(exit_code, trace_output);
    }

    #[cfg(not(feature = "memtrace"))]
    {
        let args: Vec<String> = env::args().collect();
        process::exit(run_cli(&args));
    }
}

#[cfg(feature = "memtrace")]
fn parse_memtrace_options(args: &[String]) -> Result<(MemtraceOptions, Vec<String>), i32> {
    let mut options = MemtraceOptions::default();
    let mut filtered_args = Vec::with_capacity(args.len());
    filtered_args.push(args[0].clone());

    let mut i = 1;
    while i < args.len() {
        let arg = &args[i];
        match arg.as_str() {
            "--" => {
                filtered_args.extend(args[i..].iter().cloned());
                break;
            }
            "--memtrace" => {
                options.enabled = true;
            }
            "--memtrace-output" => {
                i += 1;
                let Some(path) = args.get(i) else {
                    eprintln!("Error: --memtrace-output requires a path");
                    return Err(1);
                };
                options.enabled = true;
                options.output_path = Some(PathBuf::from(path));
            }
            "--memtrace-help" | "--help-memtrace" => {
                print_usage(&args[0]);
                return Err(0);
            }
            "--memtrace-top" | "--memtrace-json" => {
                eprintln!(
                    "Error: '{}' was removed; memtrace now emits only raw traces. Use --memtrace-output <path>.",
                    arg
                );
                return Err(1);
            }
            _ if arg.starts_with("--memtrace-") => {
                eprintln!("Error: unrecognized memtrace option '{}'", arg);
                return Err(1);
            }
            _ => filtered_args.push(arg.clone()),
        }
        i += 1;
    }

    Ok((options, filtered_args))
}

fn run_cli(args: &[String]) -> i32 {
    if args.len() < 2 {
        print_usage(&args[0]);
        return 1;
    }

    let mut dir: Option<PathBuf> = None;
    let mut backend_mode = BackendMode::Native;
    let mut reference_mode = ReferenceBackendMode::Disabled;
    let mut remaining_args: Vec<String> = Vec::new();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--" {
            remaining_args.extend(args[i + 1..].iter().cloned());
            break;
        } else if args[i] == "--dir" {
            i += 1;
            if i < args.len() {
                dir = Some(PathBuf::from(&args[i]));
            } else {
                eprintln!("Error: --dir requires a path");
                return 1;
            }
        } else if args[i] == "--backend" {
            i += 1;
            if i >= args.len() {
                eprintln!("Error: --backend requires one of: auto, native (alias: jit)");
                return 1;
            }
            let Some(parsed) = BackendMode::parse_str(&args[i]) else {
                eprintln!(
                    "Error: invalid backend '{}'; expected one of: auto, native (alias: jit)",
                    args[i]
                );
                return 1;
            };
            backend_mode = parsed;
        } else if args[i] == "--emu64" {
            if reference_mode == ReferenceBackendMode::Emu32 {
                eprintln!("Error: --emu32 conflicts with --emu64");
                return 1;
            }
            reference_mode = ReferenceBackendMode::Emu64;
        } else if args[i] == "--emu32" {
            if reference_mode == ReferenceBackendMode::Emu64 {
                eprintln!("Error: --emu32 conflicts with --emu64");
                return 1;
            }
            reference_mode = ReferenceBackendMode::Emu32;
        } else {
            remaining_args.push(args[i].clone());
        }
        i += 1;
    }

    if remaining_args.is_empty() {
        eprintln!("Error: no wasm file specified");
        return 1;
    }

    set_backend_mode(backend_mode);
    if let Err(err) = set_reference_backend_mode(reference_mode) {
        eprintln!("Error: {}", err);
        return 1;
    }
    let runtime_engine = match active_runtime_engine() {
        Ok(engine) => engine,
        Err(err) => {
            eprintln!(
                "Error: backend '{}' is unavailable in this build: {}",
                backend_mode.as_str(),
                err,
            );
            return 1;
        }
    };
    print_runtime_engine(runtime_engine);

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
    let preopen = dir.as_deref().unwrap_or_else(|| std::path::Path::new("."));
    ctx_builder = ctx_builder.preopen_dir(".", preopen);
    let ctx = ctx_builder.inherit_env().build();
    set_wasi_ctx(ctx);

    let imports = wasi_imports();
    let mut instance = match Instance::new(&data, &imports) {
        Ok(instance) => instance,
        Err(err) => {
            eprintln!("Error instantiating module: {}", err);
            return 1;
        }
    };

    let entry = if instance.has_function_export("_start") {
        "_start"
    } else {
        "main"
    };
    let result = instance.invoke(entry, &[]);

    #[cfg(feature = "jit")]
    {
        let s = native_stats_snapshot();
        if s.groups > 0 {
            let arch = if cfg!(target_arch = "aarch64") {
                "arm64"
            } else if cfg!(target_arch = "x86_64") {
                "x86_64"
            } else if cfg!(target_arch = "arm") {
                "armv7a"
            } else {
                "unknown"
            };
            eprintln!(
                "[{arch}] (func:{}, ssa:{}, mir:{}, code:{})",
                s.groups, s.ssa_ops, s.mir_ops, s.bytes_emitted
            );
        }
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

#[cfg(feature = "memtrace")]
fn finish_tracing(mut exit_code: i32, trace_output: Option<PathBuf>) -> ! {
    let printed_path = if trace_output.is_some() {
        sf_nano_memtrace::trace_output_path().or(trace_output.clone())
    } else {
        None
    };
    if trace_output.is_some() {
        if let Err(err) = sf_nano_memtrace::flush_trace() {
            eprintln!("Error flushing memtrace output: {}", err);
            if exit_code == 0 {
                exit_code = 1;
            }
        }
        sf_nano_memtrace::disable_trace();
        if let Some(path) = printed_path {
            eprintln!("[memtrace] raw trace={}", path.display());
        }
    }
    process::exit(exit_code);
}

#[cfg(feature = "memtrace")]
fn panic_payload_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

fn print_usage(program_name: &str) {
    eprintln!("Silverfir-nano — WebAssembly interpreter");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!(
        "  {program_name} [--backend <auto|native>] [--emu64|--emu32] [--dir <path>] <wasm-file> [args...]"
    );
    #[cfg(feature = "memtrace")]
    {
        eprintln!();
        eprintln!("MEMTRACE OPTIONS:");
        eprintln!("  --memtrace             Enable raw memtrace output");
        eprintln!("  --memtrace-output <path>  Write the raw trace log to <path>");
        eprintln!("  --memtrace-help        Show this help text");
    }
    eprintln!();
    eprintln!("Run a WebAssembly module with WASI support.");
}

fn print_runtime_engine(engine: sf_nano_core::RuntimeEngine) {
    match engine {
        sf_nano_core::RuntimeEngine::Jit(backend) => {
            eprintln!("[runtime] jit backend={backend}");
        }
    }
}
