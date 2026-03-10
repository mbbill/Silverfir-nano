//! Fast backend precompile entry.

use crate::error::WasmError;
use crate::vm::{
    backend::{active_backend_mode, BackendKind, BackendMode},
    entities::ModuleInst,
    lir::legacy::lower::LirProgram,
    store::Store,
};

use super::{
    build::{build_fast_function_for_spec, FastBuildBundle},
    finalizer::{self, FastFinalized},
};

/// Fast backend precompile bundle.
#[derive(Clone, Debug, Default)]
pub struct FastPrecompileBundle {
    pub lir: LirProgram,
    pub finalized: Option<FastFinalized>,
}

pub fn precompile_fast(
    bundle: FastBuildBundle,
    module: &ModuleInst,
) -> Result<FastPrecompileBundle, WasmError> {
    let finalized = finalizer::finalize_fast(&bundle, module);

    Ok(FastPrecompileBundle {
        lir: bundle.lir,
        finalized: Some(finalized),
    })
}

pub fn fast_backend_mode() -> BackendMode {
    BackendMode::Fusion
}

pub fn precompile_module_two_pass(_store: &Store) -> Result<(), WasmError> {
    let module = _store.module();

    let all_compiled = module
        .functions
        .iter()
        .filter(|func| !func.is_external())
        .all(|func| func.spec().map(|spec| spec.has_fast_code()).unwrap_or(true));
    if all_compiled {
        return Ok(());
    }

    let backend = match active_backend_mode() {
        BackendMode::Fusion => BackendKind::Fusion,
        _ => BackendKind::Base,
    };

    for func in module.functions.iter().filter(|func| !func.is_external()) {
        let Some(spec) = func.spec() else {
            continue;
        };
        if spec.has_fast_code() {
            continue;
        }

        let params_len = spec.func_type().params().len() as u16;
        let locals_len = spec.locals().len() as u16;
        let results_len = spec.func_type().results().len() as u16;
        let bundle = build_fast_function_for_spec(
            spec.code(),
            backend,
            _store,
            module,
            params_len,
            params_len.saturating_add(locals_len),
            results_len,
        )?;
        let finalized = finalizer::finalize_fast(&bundle, module);
        let cache = finalized.code.build_cache(
            params_len as usize,
            locals_len as usize,
            results_len as usize,
        );
        spec.set_fast_code(finalized.code, cache);
    }

    optimize_internal_calls(_store);
    Ok(())
}

fn optimize_internal_calls(store: &Store) {
    let module = store.module();

    for func in module.functions.iter().filter(|func| !func.is_external()) {
        let Some(spec) = func.spec() else {
            continue;
        };
        let Some(fast_code) = spec.get_fast_code() else {
            continue;
        };

        for inst in fast_code.code() {
            if inst.handler as usize != super::handlers::full_set::op_call_internal as usize {
                continue;
            }

            let callee_ptr = inst.imm0 as *const crate::vm::entities::FunctionInst;
            let delta = inst.imm1 as u16;
            let callee = unsafe { &*callee_ptr };
            let Some(callee_spec) = callee.spec() else {
                continue;
            };
            if !callee_spec.has_fast_code() {
                continue;
            }

            let callee_entry = callee_spec.fast_cache().entry() as u64;
            let callee_params = callee_spec.func_type().params().len() as u16;
            let callee_locals = callee_spec.locals().len() as u16;

            let inst_ptr =
                inst as *const _ as *mut crate::vm::interp::fast::instruction::Instruction;
            unsafe {
                (*inst_ptr).handler = super::handlers::full_set::op_call_local;
                (*inst_ptr).imm0 = callee_entry;
                (*inst_ptr).imm1 = (delta as u64)
                    | ((callee_params as u64) << 16)
                    | ((callee_locals as u64) << 32);
                (*inst_ptr).imm2 = 0;
            }
        }
    }
}
