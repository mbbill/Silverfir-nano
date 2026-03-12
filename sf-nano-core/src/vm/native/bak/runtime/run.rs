//! Native runtime entry and direct execution flow.
//!
//! This file owns the live native backend runtime path:
//! - ensure native compilation exists for the requested function
//! - construct native runtime context
//! - enter through the native entry ABI
//! - return canonical Wasm values to the VM surface

use alloc::{vec, vec::Vec};

use crate::error::WasmError;
#[cfg(feature = "function-trace")]
use crate::vm::debug::function_trace;
use crate::vm::{
    entities::{Caller, FunctionInst, MemInst},
    interp::stack::InterpreterStack,
    store::Store,
    value::Value,
};

use crate::vm::native::{code::NativeCode, compile::precompile};

use super::{context::NativeContext, layout};

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
            func,
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
            let code = spec.get_native_code().ok_or_else(|| {
                WasmError::invalid("native backend unavailable for function".into())
            })?;
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
    func: &FunctionInst,
    store: &mut Store,
    fp: *mut u64,
    stack_end: *mut u64,
    params_len: usize,
    results_len: usize,
) -> Result<Vec<u64>, WasmError> {
    let program = code
        .program()
        .ok_or_else(|| WasmError::internal("native code is missing finalized program".into()))?;
    let frame_slots_used = layout::frame_slots_used(program);
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

    #[cfg(feature = "function-trace")]
    if let FunctionInst::Local { spec, .. } = func {
        function_trace::init_from_env();
        let backend = crate::vm::native::arch::active_native_backend()
            .map(|backend| backend.as_str())
            .unwrap_or("native");
        function_trace::native_root_entry(&mut ctx, spec, backend);
    }

    unsafe {
        entry(&mut ctx, fp, 0, 0, 0, 0, 0, 0, 0);
    }

    if let Some(error) = ctx.error {
        return Err(error);
    }

    #[cfg(feature = "function-trace")]
    if let FunctionInst::Local { spec, .. } = func {
        let results = unsafe { core::slice::from_raw_parts(fp, results_len) };
        function_trace::native_root_exit(&mut ctx, spec, results);
    }

    Ok((0..results_len)
        .map(|index| unsafe { *fp.add(index) })
        .collect())
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::run_function;
    use alloc::{rc::Rc, vec};
    use crate::{
        module::{
            entities::{Bytecode, FunctionSpec},
            type_context::TypeContext,
            type_defs::FunctionType,
        },
        utils::limits::Limits,
        value_type::ValueType,
        vm::{
            entities::{FunctionInst, ModuleInst, TableInst},
            native::arch::set_reference_backend,
            store::Store,
            value::{RefHandle, Value},
        },
    };

    fn build_call_indirect_store() -> Store {
        let sig = Rc::new(FunctionType::new(vec![], vec![ValueType::I32]));
        let types = TypeContext::new(vec![sig.clone()]);
        let mut module = ModuleInst::new("test".into(), types);
        module.functions.push(FunctionInst::Local {
            spec: FunctionSpec::new(sig.clone(), 0),
            type_index: 0,
        });
        module.functions.push(FunctionInst::Local {
            spec: FunctionSpec::new(sig, 0),
            type_index: 0,
        });
        module
            .tables
            .push(TableInst::new(Limits::new(1, Some(1)).expect("limits"), ValueType::funcref()));
        module.tables[0].elements[0] = RefHandle::new(0);

        let callee_spec = match &mut module.functions[0] {
            FunctionInst::Local { spec, .. } => spec,
            _ => unreachable!(),
        };
        callee_spec.set_code(Bytecode::from(&[0x41, 0xb2, 0x02, 0x0b][..]));

        let caller_spec = match &mut module.functions[1] {
            FunctionInst::Local { spec, .. } => spec,
            _ => unreachable!(),
        };
        caller_spec.set_code(Bytecode::from(&[
            0x41, 0x00, // i32.const 0
            0x11, 0x00, 0x00, // call_indirect type 0 table 0
            0x0b, // end
        ][..]));

        Store::new(module)
    }

    #[test]
    fn native_runtime_call_indirect_arm64() {
        set_reference_backend(false).expect("arm64 backend");
        let mut store = build_call_indirect_store();
        let func_ptr = &store.module().functions[1] as *const FunctionInst;
        let func = unsafe { &*func_ptr };
        let results = run_function(func, &mut store, &[]).expect("native run");
        assert_eq!(results, vec![Value::I32(306)]);
    }

    #[cfg(debug_assertions)]
    #[test]
    fn native_runtime_call_indirect_reference() {
        set_reference_backend(true).expect("reference backend");
        let mut store = build_call_indirect_store();
        let func_ptr = &store.module().functions[1] as *const FunctionInst;
        let func = unsafe { &*func_ptr };
        let results = run_function(func, &mut store, &[]).expect("reference run");
        assert_eq!(results, vec![Value::I32(306)]);
        set_reference_backend(false).expect("reset arm64 backend");
    }
}
