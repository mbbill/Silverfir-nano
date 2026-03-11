//! Native module/function precompile entry.

use crate::error::WasmError;
use crate::vm::native::{arch, code::NativeCode, ir::NativeProgram, lower};
use crate::vm::store::Store;

use super::{
    build::{build_native_function_for_spec, NativeBuildBundle},
    resolve,
};

/// Native precompile result.
#[derive(Debug, Default)]
pub struct NativePrecompiled {
    pub ir: Option<NativeProgram>,
    pub code: Option<NativeCode>,
}

pub fn precompile_native(bundle: NativeBuildBundle) -> Result<NativePrecompiled, WasmError> {
    let ir = lower::lower_native(&bundle.lir, &bundle.planned, bundle.backend_config);
    ir.validate()?;
    let resolved = resolve::resolve_native(&ir);
    let code = arch::compile_native(&ir, &resolved)?;

    Ok(NativePrecompiled {
        ir: Some(ir),
        code: Some(code),
    })
}

pub fn precompile_module(store: &Store) -> Result<(), WasmError> {
    let module = store.module();

    let all_compiled = module
        .functions
        .iter()
        .filter(|func| !func.is_external())
        .all(|func| {
            func.spec()
                .map(|spec| spec.has_native_code())
                .unwrap_or(true)
        });
    if all_compiled {
        return Ok(());
    }

    for (func_index, func) in module
        .functions
        .iter()
        .enumerate()
        .filter(|(_, func)| !func.is_external())
    {
        let Some(spec) = func.spec() else {
            continue;
        };
        if spec.has_native_code() {
            continue;
        }

        let params_len = spec.func_type().params().len() as u16;
        let locals_len = spec.locals().len() as u16;
        let results_len = spec.func_type().results().len() as u16;
        let bundle = build_native_function_for_spec(
            spec.code(),
            store,
            module,
            params_len,
            params_len.saturating_add(locals_len),
            results_len,
        )
        .map_err(|err| {
            WasmError::internal(alloc::format!(
                "native build failed for function {}: {}",
                func_index,
                err
            ))
        })?;
        let finalized = precompile_native(bundle).map_err(|err| {
            WasmError::internal(alloc::format!(
                "native codegen failed for function {}: {}",
                func_index,
                err
            ))
        })?;
        let code = finalized
            .code
            .ok_or_else(|| WasmError::internal("native precompile produced no code".into()))?;
        let cache = code.build_cache(
            params_len as usize,
            locals_len as usize,
            results_len as usize,
        );
        spec.set_native_code(code, cache);
    }

    Ok(())
}
