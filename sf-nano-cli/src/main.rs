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

    if args.len() < 2 {
        eprintln!("Silverfir-nano — WebAssembly interpreter");
        eprintln!();
        eprintln!("USAGE:");
        eprintln!(
            "  sf-nano-cli [--backend <auto|native>] [--emu64|--emu32] [--dir <path>] <wasm-file> [args...]"
        );
        eprintln!();
        eprintln!("Run a WebAssembly module with WASI support.");
        process::exit(1);
    }

    // Parse global runtime options.
    let mut dir: Option<PathBuf> = None;
    let mut backend_mode = BackendMode::Native;
    let mut reference_mode = ReferenceBackendMode::Disabled;
    let mut remaining_args: Vec<String> = Vec::new();
    {
        let mut i = 1;
        while i < args.len() {
            if args[i] == "--" {
                remaining_args.extend(args[i + 1..].iter().cloned());
                break;
            } else if args[i] == "--dir" {
                i += 1;
                if i < args.len() {
                    dir = Some(PathBuf::from(&args[i]));
                }
            } else if args[i] == "--backend" {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --backend requires one of: auto, native (alias: jit)");
                    process::exit(1);
                }
                backend_mode = BackendMode::parse_str(&args[i]).unwrap_or_else(|| {
                    eprintln!(
                        "Error: invalid backend '{}'; expected one of: auto, native (alias: jit)",
                        args[i]
                    );
                    process::exit(1);
                });
            } else if args[i] == "--emu64" {
                if reference_mode == ReferenceBackendMode::Emu32 {
                    eprintln!("Error: --emu32 conflicts with --emu64");
                    process::exit(1);
                }
                reference_mode = ReferenceBackendMode::Emu64;
            } else if args[i] == "--emu32" {
                if reference_mode == ReferenceBackendMode::Emu64 {
                    eprintln!("Error: --emu32 conflicts with --emu64");
                    process::exit(1);
                }
                reference_mode = ReferenceBackendMode::Emu32;
            } else {
                remaining_args.push(args[i].clone());
            }
            i += 1;
        }
    }

    if remaining_args.is_empty() {
        eprintln!("Error: no wasm file specified");
        process::exit(1);
    }

    set_backend_mode(backend_mode);
    if let Err(err) = set_reference_backend_mode(reference_mode) {
        eprintln!("Error: {}", err);
        process::exit(1);
    }
    let runtime_engine = active_runtime_engine().unwrap_or_else(|err| {
        eprintln!(
            "Error: backend '{}' is unavailable in this build: {}",
            backend_mode.as_str(),
            err,
        );
        process::exit(1);
    });
    print_runtime_engine(runtime_engine);

    let path = PathBuf::from(&remaining_args[0]);
    let prog_args: Vec<String> = remaining_args[1..].to_vec();

    // Read WASM binary
    let data = fs::read(&path).unwrap_or_else(|err| {
        eprintln!("Error reading '{}': {}", path.display(), err);
        process::exit(1);
    });

    let module_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("module");

    // Build WASI context
    let mut wasi_args = vec![module_name.to_string()];
    wasi_args.extend(prog_args);

    let mut ctx_builder = WasiContextBuilder::new().args(&wasi_args);
    // Only preopen a directory when explicitly requested
    let preopen = dir.as_deref().unwrap_or_else(|| std::path::Path::new("."));
    ctx_builder = ctx_builder.preopen_dir(".", preopen);
    let ctx = ctx_builder.inherit_env().build();
    set_wasi_ctx(ctx);

    // Create instance with WASI imports
    let imports = wasi_imports();
    let mut instance = Instance::new(&data, &imports).unwrap_or_else(|err| {
        eprintln!("Error instantiating module: {}", err);
        process::exit(1);
    });

    // Invoke _start, fallback to main
    let result = instance.invoke("_start", &[]);
    let result = match result {
        Err(ref err) if err.to_string().contains("not found") => instance.invoke("main", &[]),
        _ => result,
    };

    // Print native backend compile stats on exit
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
        Ok(_) => {}
        Err(err) => {
            if let Some(code) = err.exit_code() {
                process::exit(code);
            }
            eprintln!("Error: {}", err);
            process::exit(1);
        }
    }
}

fn print_runtime_engine(engine: sf_nano_core::RuntimeEngine) {
    match engine {
        sf_nano_core::RuntimeEngine::Jit(backend) => {
            eprintln!("[runtime] jit backend={backend}");
        }
    }
}
