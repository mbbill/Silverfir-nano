use crate::error::WasmError;
use crate::vm::backend::backend_mode;
use crate::vm::entities::FunctionInst;
use crate::vm::store::Store;
use crate::vm::value::Value;

#[cfg(feature = "micro-jit")]
use crate::vm::backend::BackendMode;
#[cfg(feature = "micro-jit")]
use crate::vm::native;

pub fn eval(
    func_inst: &FunctionInst,
    store: &mut Store,
    args: &[Value],
) -> Result<crate::vm::interp::stack::InterpreterStack, WasmError> {
    #[cfg(feature = "micro-jit")]
    {
        if matches!(backend_mode(), BackendMode::Native) {
            if let FunctionInst::Local { spec, .. } = func_inst {
                if !spec.has_native_code() {
                    native::precompile::precompile_module(store)?;
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
