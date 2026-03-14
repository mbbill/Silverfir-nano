use alloc::{rc::Rc, vec::Vec};

use crate::{
    error::WasmError,
    vm::{
        native::{
            arch,
            code::{CompiledNativeModule, NativeCode, NativeCodeCache},
            ir::machine::MachineFuncId,
            lower::{lower_module, LowerFunctionInput, LowerModuleInput},
        },
        plan::{config::PlanConfig, prepare::PrepareInput, prepare_function, PreparedFunction},
        store::Store,
        wasm::{context::CompileContext, decode},
    },
};

#[inline]
pub const fn native_plan_config(backend: crate::vm::backend::BackendConfig) -> PlanConfig {
    PlanConfig::from_backend_config(backend, 3)
}

pub fn ensure_module_compiled(store: &Store) -> Result<(), WasmError> {
    let active_backend = arch::active_native_backend()
        .map_err(|err| WasmError::invalid(alloc::format!("native backend unavailable: {err}")))?;
    let backend = arch::compile_backend_config(active_backend);
    let module = store.module();
    let all_compiled = module
        .functions
        .iter()
        .filter_map(|func| func.spec())
        .all(|spec| {
            spec.get_native_code()
                .map(|code| {
                    code.compiled().backend_kind() == active_backend
                        && code.compiled().backend() == backend
                })
                .unwrap_or(false)
        });
    if all_compiled {
        return Ok(());
    }

    let mut lowered_inputs = Vec::new();
    let mut prepared_functions = Vec::new();
    for (func_idx, func) in module.functions.iter().enumerate() {
        let Some(spec) = func.spec() else {
            continue;
        };
        let params = spec.func_type().params().len() as u16;
        let local_count = params.saturating_add(spec.locals().len() as u16);
        let results = spec.func_type().results().len() as u16;
        let semantic = decode::decode_to_semantic_ir(
            spec.code(),
            CompileContext::new(&module.types, store, module, params, local_count, results),
        )
        .map_err(|err| {
            WasmError::internal(alloc::format!(
                "native decode failed for function {}: {}",
                func_idx,
                err
            ))
        })?;
        let prepared =
            prepare_function(PrepareInput { config: native_plan_config(backend) }, &semantic).map_err(
                |err| {
                    WasmError::internal(alloc::format!(
                        "native prepare failed for function {} type_idx={} params={} results={} max_stack={} ops={}: {}",
                        func_idx,
                        spec.type_index(),
                        params,
                        results,
                        semantic.max_stack_height,
                        semantic.ops.len(),
                        err
                    ))
                },
            )?;
        prepared_functions.push((MachineFuncId(func_idx as u32), prepared));
    }

    for (func_idx, (id, prepared)) in prepared_functions.iter().enumerate() {
        let result_count = module
            .functions
            .get(id.0 as usize)
            .and_then(|f| f.spec())
            .map(|s| s.func_type().results().len() as u16)
            .unwrap_or(0);
        lowered_inputs.push(LowerFunctionInput {
            id: *id,
            frame: prepared.frame,
            lir: &prepared.lir,
            result_count,
        });
    }

    let lowered = lower_module(LowerModuleInput {
        backend,
        functions: &lowered_inputs,
    })?;
    let compiled = Rc::new(CompiledNativeModule::new(
        active_backend,
        backend,
        lowered.module,
        lowered.runtime,
    )?);
    let arm64_entries = match active_backend {
        arch::NativeBackend::Arm64 => {
            Some(arch::arm64::compile::compile_module(module, &compiled)?)
        }
        #[cfg(debug_assertions)]
        arch::NativeBackend::Reference => None,
    };

    for (func_idx, func) in module.functions.iter().enumerate() {
        let Some(spec) = func.spec() else {
            continue;
        };
        let arm64_entry = arm64_entries
            .as_ref()
            .and_then(|entries| entries.get(func_idx).copied().flatten());
        spec.set_native_code(
            NativeCode::new(Rc::clone(&compiled), MachineFuncId(func_idx as u32)).with_arm64_entry(
                arm64_entry.map(|entry| entry.entry),
                arm64_entry.map(|entry| entry.root_return),
            ),
            NativeCodeCache::compiled(),
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::{boxed::Box, rc::Rc, string::String, vec};

    use super::ensure_module_compiled;
    use crate::{
        module::{entities::FunctionSpec, type_context::TypeContext, type_defs::FunctionType},
        value_type::ValueType,
        vm::{
            entities::{FunctionInst, ModuleInst},
            store::Store,
        },
    };

    #[test]
    fn compiles_all_local_functions_once() {
        let types = TypeContext::new(vec![
            Rc::new(FunctionType::new(vec![], vec![])),
            Rc::new(FunctionType::new(vec![ValueType::I32], vec![])),
        ]);
        let mut module = ModuleInst::new(String::from("m"), types);
        let mut spec0 = FunctionSpec::new(Rc::new(FunctionType::new(vec![], vec![])), 0);
        spec0.set_code((&[0x0b][..]).into());
        let mut spec1 =
            FunctionSpec::new(Rc::new(FunctionType::new(vec![ValueType::I32], vec![])), 1);
        spec1.set_code((&[0x20, 0x00, 0x1a, 0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec: spec0,
            type_index: 0,
        });
        module.functions.push(FunctionInst::Local {
            spec: spec1,
            type_index: 1,
        });
        let store = Box::new(Store::new(module));

        ensure_module_compiled(&store).expect("native compile should succeed");

        let first = store.module().functions[0]
            .spec()
            .and_then(|spec| spec.get_native_code())
            .expect("first native code");
        let second = store.module().functions[1]
            .spec()
            .and_then(|spec| spec.get_native_code())
            .expect("second native code");

        assert!(Rc::ptr_eq(first.compiled_rc(), second.compiled_rc()));
        assert_eq!(first.func_id().0, 0);
        assert_eq!(second.func_id().0, 1);
    }
}
