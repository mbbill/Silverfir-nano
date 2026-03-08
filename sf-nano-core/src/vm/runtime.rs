use crate::error::WasmError;
use crate::vm::backend::active_backend;
use crate::vm::entities::FunctionInst;
use crate::vm::store::Store;
use crate::vm::value::Value;

#[cfg(feature = "micro-jit")]
use crate::vm::backend::BackendKind;
#[cfg(feature = "micro-jit")]
use crate::vm::native;

pub fn eval(
    func_inst: &FunctionInst,
    store: &mut Store,
    args: &[Value],
) -> Result<crate::vm::interp::stack::InterpreterStack, WasmError> {
    #[cfg(feature = "micro-jit")]
    {
        if matches!(active_backend(), Ok(BackendKind::Native)) {
            if let FunctionInst::Local { spec, .. } = func_inst {
                if !spec.has_native_code() {
                    native::precompile::precompile_module(store)?;
                }
                if !spec.has_native_code() {
                    let module = store.module();
                    if let Some((func_idx, _)) = module
                        .functions
                        .iter()
                        .enumerate()
                        .find(|(_, candidate)| match candidate {
                            FunctionInst::Local { spec: candidate_spec, .. } => {
                                core::ptr::eq(candidate_spec, spec)
                            }
                            FunctionInst::External { .. } => false,
                        })
                    {
                        native::compiler::build_for_function(
                            spec,
                            Some(&module.types),
                            store,
                            module,
                            func_idx as u32,
                        )?;
                    }
                }
                if spec.has_native_code() {
                    return native::runtime::eval(func_inst, store, args);
                }
                return Err(WasmError::invalid(
                    "requested backend native is unavailable for this function".into(),
                ));
            }
        }
    }

    crate::vm::interp::fast::runtime::eval(func_inst, store, args)
}
