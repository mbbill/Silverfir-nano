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
            return native::runtime::eval(func_inst, store, args);
        }
    }

    crate::vm::interp::fast::runtime::eval(func_inst, store, args)
}
