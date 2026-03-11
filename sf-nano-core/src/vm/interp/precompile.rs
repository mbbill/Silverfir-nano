//! Interpreter module precompile entry.

use crate::error::WasmError;
use crate::vm::{
    backend::{active_backend_mode, BackendKind, BackendMode},
    entities::ModuleInst,
    lir::ir::LirProgram,
    store::Store,
};

use super::{
    build::{
        build_interpreter_function_for_spec, normalize_interpreter_backend,
        InterpreterBuildBundle,
    },
    finalizer::{self, InterpreterFinalized},
};

#[derive(Clone, Debug, Default)]
pub struct InterpreterPrecompileBundle {
    pub lir: LirProgram,
    pub finalized: Option<InterpreterFinalized>,
}

pub fn precompile_interpreter(
    bundle: InterpreterBuildBundle,
    module: &ModuleInst,
) -> Result<InterpreterPrecompileBundle, WasmError> {
    let finalized = finalizer::finalize_interpreter(&bundle, module)?;

    Ok(InterpreterPrecompileBundle {
        lir: bundle.lir,
        finalized: Some(finalized),
    })
}

pub fn interpreter_backend_mode() -> BackendMode {
    BackendMode::Base
}

pub fn precompile_module_two_pass(store: &Store) -> Result<(), WasmError> {
    let module = store.module();

    let all_compiled = module
        .functions
        .iter()
        .filter(|func| !func.is_external())
        .all(|func| func.spec().map(|spec| spec.has_fast_code()).unwrap_or(true));
    if all_compiled {
        return Ok(());
    }

    let requested = match active_backend_mode() {
        BackendMode::Native => BackendKind::Native,
        BackendMode::Fusion => BackendKind::Fusion,
        _ => BackendKind::Base,
    };
    let backend = normalize_interpreter_backend(requested);

    for (func_index, func) in module
        .functions
        .iter()
        .enumerate()
        .filter(|(_, func)| !func.is_external())
    {
        let Some(spec) = func.spec() else {
            continue;
        };
        if spec.has_fast_code() {
            continue;
        }

        let params_len = spec.func_type().params().len() as u16;
        let locals_len = spec.locals().len() as u16;
        let results_len = spec.func_type().results().len() as u16;
        let bundle = build_interpreter_function_for_spec(
            spec.code(),
            backend,
            store,
            module,
            params_len,
            params_len.saturating_add(locals_len),
            results_len,
        )
        .map_err(|err| {
            WasmError::internal(alloc::format!(
                "interpreter build failed for function {}: {}",
                func_index, err
            ))
        })?;
        let finalized = finalizer::finalize_interpreter(&bundle, module).map_err(|err| {
            WasmError::internal(alloc::format!(
                "interpreter finalization failed for function {}: {}",
                func_index, err
            ))
        })?;
        let cache = finalized.code.build_cache(
            params_len as usize,
            locals_len as usize,
            results_len as usize,
            bundle.planned.frame.frame_size as usize,
            finalized.stack_slots_used,
        );
        spec.set_fast_code(finalized.code, cache);
    }

    // Keep internal calls on the shared call_internal path until the call_local
    // specialization is rebuilt against the new LIR-driven lowering.
    Ok(())
}
