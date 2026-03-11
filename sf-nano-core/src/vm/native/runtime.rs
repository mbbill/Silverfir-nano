//! Native runtime entry and direct execution flow.
//!
//! This file owns the live native backend runtime path:
//! - ensure native compilation exists for the requested function
//! - construct native runtime context
//! - enter through the native entry ABI
//! - return canonical Wasm values to the VM surface

use alloc::{vec, vec::Vec};

use crate::error::WasmError;
use crate::vm::{
    entities::{Caller, FunctionInst, MemInst},
    interp::stack::InterpreterStack,
    store::Store,
    value::Value,
};

use super::{
    arch::reference,
    code::NativeCode,
    context::NativeContext,
    precompile,
};

const MAX_SLOTS: usize = crate::constants::MAX_STACK_SIZE / core::mem::size_of::<u64>();

/// Native runtime launch bundle.
pub struct NativeRuntime<'a> {
    pub store: &'a mut Store,
}

impl<'a> NativeRuntime<'a> {
    pub fn run_function(
        &mut self,
        code: &NativeCode,
        args: &[Value],
        func: &FunctionInst,
    ) -> Result<Vec<Value>, WasmError> {
        let ft = func.func_type();
        let params_len = ft.params().len();
        if args.len() != params_len {
            return Err(WasmError::invalid(alloc::format!(
                "invalid argument count: got {}, expected {}",
                args.len(),
                params_len
            )));
        }

        let mut stack = vec![0u64; MAX_SLOTS];
        let stack_base = stack.as_mut_ptr();
        let stack_end = unsafe { stack_base.add(MAX_SLOTS) };
        for (index, arg) in args.iter().enumerate() {
            unsafe {
                *stack_base.add(index) = arg.to_raw();
            }
        }

        let results = run_local_function(
            code,
            self.store,
            stack_base,
            stack_end,
            params_len,
            ft.results().len(),
        )?;
        Ok(results
            .into_iter()
            .zip(ft.results().iter().copied())
            .map(|(raw, ty)| Value::from_raw(raw, ty))
            .collect())
    }
}

pub fn run_function(
    func: &FunctionInst,
    store: &mut Store,
    args: &[Value],
) -> Result<Vec<Value>, WasmError> {
    match func {
        FunctionInst::External {
            func_type,
            callback,
        } => {
            let params = func_type.params();
            let results = func_type.results();
            if args.len() != params.len() {
                return Err(WasmError::invalid(alloc::format!(
                    "invalid argument count: got {}, expected {}",
                    args.len(),
                    params.len()
                )));
            }

            let mut ret_vals = alloc::vec![Value::default(); results.len()];
            let mem_slice = if !store.module().memories.is_empty() {
                let mem = &store.module().memories[0] as *const MemInst as *mut MemInst;
                unsafe { Some((*mem).data.as_mut_slice()) }
            } else {
                None
            };
            let mut caller = Caller::new(mem_slice);
            callback(&mut caller, args, &mut ret_vals)?;
            Ok(ret_vals)
        }
        FunctionInst::Local { spec, .. } => {
            if !spec.has_native_code() {
                precompile::precompile_module(store)?;
            }
            let code = spec
                .get_native_code()
                .ok_or_else(|| WasmError::invalid("native backend unavailable for function".into()))?;
            NativeRuntime { store }.run_function(code, args, func)
        }
    }
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

fn run_local_function(
    code: &NativeCode,
    store: &mut Store,
    fp: *mut u64,
    stack_end: *mut u64,
    params_len: usize,
    results_len: usize,
) -> Result<Vec<u64>, WasmError> {
    let program = code
        .program()
        .ok_or_else(|| WasmError::internal("native code is missing finalized program".into()))?;
    let frame_slots_used = reference::frame_slots_used(program);
    let frame_end = unsafe { fp.add(frame_slots_used) };
    if frame_end > stack_end {
        return Err(WasmError::exhaustion("stack overflow".into()));
    }
    if frame_slots_used > params_len {
        unsafe {
            core::ptr::write_bytes(fp.add(params_len), 0, frame_slots_used - params_len);
        }
    }

    let mut ctx = NativeContext::new(store as *mut Store, stack_end, code as *const NativeCode);
    let entry = code
        .entry
        .ok_or_else(|| WasmError::internal("native code is missing entry".into()))?;

    unsafe {
        entry(&mut ctx, fp, 0, 0, 0, 0, 0, 0, 0);
    }

    if let Some(error) = ctx.error {
        return Err(error);
    }

    Ok((0..results_len)
        .map(|index| unsafe { *fp.add(index) })
        .collect())
}
