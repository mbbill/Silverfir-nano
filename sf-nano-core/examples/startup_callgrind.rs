//! CI-only low-perturbation driver for eager interpreter startup profiling.
//!
//! Unlike `startup_stage_profile`, this binary does not put a clock in the
//! per-opcode path.  Callgrind attributes deterministic retired instructions
//! to the ordinary release build instead.

use sf_nano_core::module::entities::FunctionDef;
use sf_nano_core::module::Module;
use sf_nano_core::{Caller, Engine, Import, Instance, Value, WasmError};
use std::env;
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;

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
        .collect())
}

fn load_case(spec: &str) -> Result<Case, String> {
    let (name, path) = spec
        .split_once('=')
        .ok_or_else(|| format!("case must be NAME=PATH, got {spec:?}"))?;
    let path = PathBuf::from(path);
    let wasm = fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    let imports = imports_for(name, &wasm)?;
    Ok(Case {
        name: name.into(),
        wasm,
        imports,
    })
}

#[inline(never)]
fn measured_instantiate(engine: &Engine, case: &Case) -> Result<(), String> {
    let instance = Instance::new(engine, &case.wasm, &case.imports)
        .map_err(|error| format!("{}: {error}", case.name))?;
    black_box(&instance);
    drop(black_box(instance));
    Ok(())
}

fn main() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let iterations = args
        .next()
        .ok_or_else(|| "missing iteration count".to_string())?
        .parse::<usize>()
        .map_err(|error| format!("invalid iteration count: {error}"))?;
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
        for _ in 0..iterations {
            measured_instantiate(&engine, case)?;
        }
    }
    Ok(())
}
