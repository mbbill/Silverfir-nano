#[cfg(feature = "jit")]
use sf_nano_core::native_stats_snapshot;
use sf_nano_core::wasi::{set_wasi_ctx, wasi_imports, WasiContextBuilder};
use sf_nano_core::Instance;
use sf_nano_core::{
    active_runtime_engine, runtime_config, set_backend_mode, set_runtime_config, BackendMode,
    RuntimeConfig,
};

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
    let mut backend_mode = BackendMode::Native;
    let mut use_interp = false;
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
        } else if args[i] == "--interp" || args[i] == "--interp-stats" {
            use_interp = true;
            #[cfg(feature = "interp")]
            {
                if args[i] == "--interp-stats" {
                    interp_stats = true;
                }
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

        set_backend_mode(backend_mode);
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
        if compiler_ram_budget.is_some() || !parallel_compilation {
            let mut cfg: RuntimeConfig = *runtime_config();
            if let Some(bytes) = compiler_ram_budget {
                cfg.compiler_ram_budget_bytes = bytes;
            }
            cfg.parallel_compilation = parallel_compilation;
            if let Err(err) = set_runtime_config(cfg) {
                eprintln!(
                    "Error: compiler options could not install runtime config: {:?}",
                    err
                );
                return 1;
            }
        }
        if !use_interp {
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
        }

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

        if use_interp {
            #[cfg(feature = "interp")]
            {
                return run_interp(&data, module_name, interp_stats);
            }
            #[cfg(not(feature = "interp"))]
            {
                eprintln!("Error: --interp requires building sf-nano-cli with the interp feature");
                return 1;
            }
        }

        let imports = wasi_imports();
        let mut instance = match Instance::new(&data, &imports) {
            Ok(instance) => instance,
            Err(err) => {
                eprintln!("Error instantiating module: {}", err);
                return 1;
            }
        };

        if compile_only {
            print_native_stats();
            return 0;
        }

        let entry = if instance.has_function_export("_start") {
            "_start"
        } else {
            "main"
        };
        let result = instance.invoke(entry, &[]);

        print_native_stats();

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

/// Run the module through the interpreter, bridging its host-dispatch
/// interface to the same WASI import set the JIT path uses (the WASI
/// context installed by the caller applies unchanged).
#[cfg(feature = "interp")]
fn run_interp(data: &[u8], module_name: &str, stats: bool) -> i32 {
    use sf_nano_core::module::entities::FunctionDef;
    use sf_nano_core::module::Module;
    use sf_nano_core::value_type::ValueType;
    use sf_nano_core::{
        Caller, FunctionType, HostCallback, ImportValue, ImportedFunction, InterpInstance, Value,
        WasmError,
    };
    use std::collections::HashMap;

    fn raw_to_value(t: &ValueType, raw: u64) -> Option<Value> {
        match t {
            ValueType::I32 => Some(Value::I32(raw as u32 as i32)),
            ValueType::I64 => Some(Value::I64(raw as i64)),
            ValueType::F32 => Some(Value::F32(f32::from_bits(raw as u32))),
            ValueType::F64 => Some(Value::F64(f64::from_bits(raw))),
            _ => None,
        }
    }
    fn value_to_raw(v: &Value) -> Option<u64> {
        match v {
            Value::I32(x) => Some(*x as u32 as u64),
            Value::I64(x) => Some(*x as u64),
            Value::F32(x) => Some(x.to_bits() as u64),
            Value::F64(x) => Some(x.to_bits()),
            _ => None,
        }
    }

    let module = match Module::new(module_name, data) {
        Ok(m) => m,
        Err(err) => {
            eprintln!("Error parsing module: {}", err);
            return 1;
        }
    };

    // Host bridge tables: WASI callbacks by import name, and the module's
    // declared type for each imported function (drives raw<->Value
    // conversion).
    let mut host_fns: HashMap<(String, String), HostCallback> = HashMap::new();
    for imp in wasi_imports() {
        if let ImportValue::Func(ImportedFunction::Host { callback, .. }) = imp.value {
            host_fns.insert((imp.module, imp.name), callback);
        }
    }
    let mut import_types: HashMap<(String, String), FunctionType> = HashMap::new();
    for f in module.functions() {
        if let FunctionDef::Import {
            module: m, name: n, ..
        } = f.def()
        {
            import_types.insert((m.clone(), n.clone()), f.func_type().clone());
        }
    }

    let exit_code;
    {
        let mut inst = match InterpInstance::new(&module) {
            Ok(inst) => inst,
            Err(err) => {
                eprintln!("Error instantiating module: {}", err);
                return 1;
            }
        };
        println!("runtime engine: interpreter (native dispatch)");

        inst.set_host(Box::new(move |m, n, mem, args, results| {
            let key = (m.to_string(), n.to_string());
            let cb = host_fns
                .get(&key)
                .ok_or(WasmError::invalid("interp cli: unlinked host import"))?;
            let ft = import_types
                .get(&key)
                .ok_or(WasmError::invalid("interp cli: untyped host import"))?;
            let mut params = Vec::with_capacity(args.len());
            for (t, &raw) in ft.params().iter().zip(args.iter()) {
                params.push(raw_to_value(t, raw).ok_or(WasmError::invalid(
                    "interp cli: non-numeric host import parameter",
                ))?);
            }
            let mut vresults = Vec::with_capacity(results.len());
            for t in ft.results().iter() {
                vresults.push(raw_to_value(t, 0).ok_or(WasmError::invalid(
                    "interp cli: non-numeric host import result",
                ))?);
            }
            let mut caller = Caller::new(Some(&mut mem[..]));
            cb.call(&mut caller, &params, &mut vresults)?;
            for (dst, v) in results.iter_mut().zip(vresults.iter()) {
                *dst = value_to_raw(v).ok_or(WasmError::invalid(
                    "interp cli: non-numeric host import result",
                ))?;
            }
            Ok(())
        }));

        let entry = inst
            .find_export("_start")
            .or_else(|| inst.find_export("main"));
        let Some(entry) = entry else {
            eprintln!("Error: module exports neither _start nor main");
            return 1;
        };
        let (n_params, n_results) = inst.func_arity(entry).unwrap_or((0, 0));
        if n_params != 0 {
            eprintln!("Error: entry function takes parameters");
            return 1;
        }
        let mut results = vec![0u64; n_results];
        let result = inst.invoke(entry, &[], &mut results);

        if stats {
            let native = inst.dispatch_count();
            if native > 0 {
                eprintln!("[interp] native dispatches: {native}");
            }
            let slow = inst.slow_exit_stats();
            if !slow.is_empty() {
                let total: u64 = slow.iter().map(|(_, n)| n).sum();
                eprintln!("[interp] slow exits: {total}");
                for (op, n) in slow.iter().take(12) {
                    eprintln!("[interp]   {op:?}: {n}");
                }
            }
        }

        exit_code = match result {
            Ok(()) => 0,
            Err(err) => {
                if let Some(code) = err.exit_code() {
                    code
                } else {
                    eprintln!("Error: {}", err);
                    1
                }
            }
        };
    }
    exit_code
}

fn print_usage(program_name: &str) {
    eprintln!("Silverfir-nano — WebAssembly interpreter");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!(
        "  {program_name} [--backend <auto|native>] [--compile-only] [--no-parallel-compilation] [--dir <guest::host|path>] [--memprof] [--memprof-report <html>] <wasm-file> [args...]"
    );
    eprintln!();
    eprintln!("Run a WebAssembly module with WASI support.");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!("  --backend <auto|native>   Select the execution backend.");
    eprintln!("  --interp                  Run through the interpreter instead of the JIT.");
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
        let s = native_stats_snapshot();
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

fn print_runtime_engine(engine: sf_nano_core::RuntimeEngine) {
    match engine {
        sf_nano_core::RuntimeEngine::Jit(backend) => {
            eprintln!("[runtime] jit backend={backend}");
        }
    }
}
