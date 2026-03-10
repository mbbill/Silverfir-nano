//! Native runtime entry and direct execution flow.
//!
//! This runtime should jump between native entries directly, not through an
//! interpreter instruction stream.

use alloc::vec::Vec;

use crate::error::WasmError;
use crate::vm::{
    entities::FunctionInst, interp::stack::InterpreterStack, store::Store, value::Value,
};

use super::{code::NativeCode, context::NativeContext};

/// Native runtime launch bundle.
pub struct NativeRuntime<'a> {
    pub store: &'a mut Store,
    pub context: &'a mut NativeContext,
}

impl<'a> NativeRuntime<'a> {
    pub fn run_function(
        &mut self,
        _code: &NativeCode,
        _args: &[Value],
    ) -> Result<Vec<Value>, WasmError> {
        // TODO: Enter the direct native entry and return canonical values.
        Ok(Vec::new())
    }
}

pub fn run_function(
    _func: &FunctionInst,
    _store: &mut Store,
    _args: &[Value],
) -> Result<Vec<Value>, WasmError> {
    // TODO: Resolve the compiled native artifact for `_func`, construct
    // `NativeContext`, and tail into `NativeRuntime`.
    Ok(Vec::new())
}

pub fn eval(
    func: &FunctionInst,
    store: &mut Store,
    args: &[Value],
) -> Result<InterpreterStack, WasmError> {
    let results = run_function(func, store, args)?;
    let mut out = InterpreterStack::with_exact_capacity(results.len());
    for value in &results {
        out.push(value.to_raw());
    }
    Ok(out)
}
