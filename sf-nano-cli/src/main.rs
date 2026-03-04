#[cfg(feature = "profile")]
mod discover_fusion;

use sf_nano_core::wasi::{set_wasi_ctx, wasi_imports, WasiContextBuilder};
use sf_nano_core::{Instance, jit_stats_snapshot};

use std::path::PathBuf;
use std::{env, fs, process};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Silverfir-nano — WebAssembly interpreter");
        eprintln!();
        eprintln!("USAGE:");
        eprintln!("  sf-nano-cli <wasm-file> [args...]");
        #[cfg(feature = "profile")]
        {
            eprintln!("  sf-nano-cli discover-fusion [OPTIONS] <wasm-file>");
            eprintln!("  sf-nano-cli discover-fusion --help");
        }
        eprintln!();
        eprintln!("Run a WebAssembly module with WASI support.");
        #[cfg(feature = "profile")]
        eprintln!("Use 'discover-fusion' subcommand to profile and discover fusion patterns.");
        process::exit(1);
    }

    // Check for subcommands
    #[cfg(feature = "profile")]
    if args[1] == "discover-fusion" {
        discover_fusion::run_from_args(&args[2..]);
        return;
    }

    // Parse --dir option if present
    let mut dir: Option<PathBuf> = None;
    let mut remaining_args: Vec<String> = Vec::new();
    {
        let mut i = 1;
        while i < args.len() {
            if args[i] == "--dir" {
                i += 1;
                if i < args.len() {
                    dir = Some(PathBuf::from(&args[i]));
                }
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

    let mut ctx_builder = WasiContextBuilder::new()
        .args(&wasi_args);
    // Only preopen a directory when explicitly requested
    let preopen = dir.as_deref().unwrap_or_else(|| std::path::Path::new("."));
    ctx_builder = ctx_builder.preopen_dir(".", preopen);
    let ctx = ctx_builder
        .inherit_env()
        .build();
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
        Err(ref err) if err.to_string().contains("not found") => {
            instance.invoke("main", &[])
        }
        _ => result,
    };

    // Print JIT status on exit
    let s = jit_stats_snapshot();
    if s.groups > 0 || s.groups_skipped > 0 {
        let kb = s.bytes_emitted / 1024;
        if s.groups_skipped > 0 {
            eprintln!(
                "[jit] {} groups ({} ops), {}KB emitted | {} groups ({} ops) skipped (buffer full)",
                s.groups, s.ops, kb, s.groups_skipped, s.ops_skipped
            );
        } else {
            eprintln!("[jit] {} groups ({} ops), {}KB emitted", s.groups, s.ops, kb);
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
