use crate::{error::WasmError, vm::{entities::FunctionInst, stack::InterpreterStack, store::Store, value::Value}};

pub mod context;

pub fn eval(
    _func_inst: &FunctionInst,
    _store: &mut Store,
    _args: &[Value],
) -> Result<InterpreterStack, WasmError> {
    Err(WasmError::invalid(
        "native runtime execution is not wired to MachineIR yet".into(),
    ))
}
