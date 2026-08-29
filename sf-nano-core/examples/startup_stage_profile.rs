//! CI-only driver for the temporary eager interpreter startup profiler.

use sf_nano_core::module::entities::FunctionDef;
use sf_nano_core::module::Module;
use sf_nano_core::startup_profile::{self, Stage};
use sf_nano_core::{Caller, Engine, Import, Instance, Value, WasmError};
use std::env;
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

struct Case {
    name: String,
    wasm: Vec<u8>,
    imports: Vec<Import>,
}

fn inert_host(
    _caller: &mut Caller<'_>,
    _params: &[Value],
    _results: &mut [Value],
) -> Result<(), WasmError> {
    Ok(())
}

fn imports_for(name: &str, wasm: &[u8]) -> Result<Vec<Import>, String> {
    let module = Module::new(name, wasm).map_err(|error| error.to_string())?;
    Ok(module
        .functions()
        .iter()
        .filter_map(|function| match function.def() {
            FunctionDef::Import {
                module,
                name,
                func_type,
                ..
            } => Some(Import::func_typed(
                module.as_str(),
                name.as_str(),
                inert_host,
                func_type.as_ref().clone(),
            )),
            FunctionDef::Local(_) => None,
        })
        .collect::<Vec<_>>())
}

fn load_case(spec: &str) -> Result<Case, String> {
    let (name, path) = spec
        .split_once('=')
        .ok_or_else(|| format!("case must be NAME=PATH, got {spec:?}"))?;
    if name.is_empty() {
        return Err("case name cannot be empty".into());
    }
    let path = PathBuf::from(path);
    let wasm = fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    let imports = imports_for(name, &wasm)?;
    Ok(Case {
        name: name.into(),
        wasm,
        imports,
    })
}

fn run_once(engine: &Engine, case: &Case, emit: bool, iteration: usize) -> Result<(), String> {
    startup_profile::reset();
    let startup_begin = Instant::now();
    let instance = Instance::new(engine, &case.wasm, &case.imports)
        .map_err(|error| format!("{}: {error}", case.name))?;
    black_box(&instance);
    let drop_begin = Instant::now();
    drop(black_box(instance));
    startup_profile::record(Stage::Drop, drop_begin.elapsed());
    startup_profile::record(Stage::StartupTotal, startup_begin.elapsed());
    if !emit {
        return Ok(());
    }

    let snapshot = startup_profile::snapshot();
    print!(
        "{{\"case\":\"{}\",\"iteration\":{},\"stages\":{{",
        case.name, iteration
    );
    for (index, (stage, nanos, calls)) in snapshot.entries().enumerate() {
        if index != 0 {
            print!(",");
        }
        print!(
            "\"{}\":{{\"nanos\":{},\"calls\":{}}}",
            stage.name(),
            nanos,
            calls
        );
    }
    println!("}}}}");
    Ok(())
}

fn parse_count(value: Option<String>, label: &str) -> Result<usize, String> {
    value
        .ok_or_else(|| format!("missing {label}"))?
        .parse::<usize>()
        .map_err(|error| format!("invalid {label}: {error}"))
}

fn main() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let iterations = parse_count(args.next(), "iteration count")?;
    let warmups = parse_count(args.next(), "warmup count")?;
    if iterations == 0 {
        return Err("iteration count must be non-zero".into());
    }
    let cases = args
        .map(|spec| load_case(&spec))
        .collect::<Result<Vec<_>, _>>()?;
    if cases.is_empty() {
        return Err("at least one NAME=PATH case is required".into());
    }

    let engine = Engine::with_defaults();
    for case in &cases {
        for _ in 0..warmups {
            run_once(&engine, case, false, 0)?;
        }
        for iteration in 0..iterations {
            run_once(&engine, case, true, iteration)?;
        }
    }
    Ok(())
}
