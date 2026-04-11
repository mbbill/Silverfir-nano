#[cfg(feature = "jit")]
use sf_nano_core::native_stats_snapshot;
use sf_nano_core::wasi::{set_wasi_ctx, wasi_imports, WasiContextBuilder};
use sf_nano_core::Instance;
use sf_nano_core::{
    active_runtime_engine, set_backend_mode, set_reference_backend_mode, BackendMode,
    ReferenceBackendMode,
};

use std::path::PathBuf;
use std::{env, fs, process};

fn main() {
    let args: Vec<String> = env::args().collect();
    process::exit(run_cli(&args));
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

fn print_usage(program_name: &str) {
    eprintln!("Silverfir-nano — WebAssembly interpreter");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!(
        "  {program_name} [--backend <auto|native>] [--emu64|--emu32] [--dir <path>] <wasm-file> [args...]"
    );
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
