mod abi;
pub mod compile;
pub mod config;
mod emit;
mod enc;
mod reg;

use alloc::vec;

use crate::{
    constants::MAX_STACK_SIZE,
    error::WasmError,
    module::entities::FunctionSpec,
    vm::{
        native::{code::NativeCode, runtime::context::NativeContext},
        stack::InterpreterStack,
        store::Store,
        value::Value,
    },
};

#[cfg(feature = "function-trace")]
use crate::vm::debug::function_trace;

const MAX_STACK_SLOTS: usize = MAX_STACK_SIZE / core::mem::size_of::<u64>();

pub fn eval(
    spec: &FunctionSpec,
    code: &NativeCode,
    store: &mut Store,
    args: &[Value],
    backend: &'static str,
) -> Result<InterpreterStack, WasmError> {
    let func_type = spec.func_type();
    if args.len() != func_type.params().len() {
        return Err(WasmError::invalid(alloc::format!(
            "invalid argument count: got {}, expected {}",
            args.len(),
            func_type.params().len()
        )));
    }

    let compiled = code.compiled();
    let func_id = code.func_id();
    let runtime = compiled
        .runtime()
        .functions
        .get(func_id.0 as usize)
        .ok_or_else(|| {
            WasmError::internal("native entry function is missing runtime metadata".into())
        })?;
    let entry = code.arm64_entry().ok_or_else(|| {
        WasmError::internal("arm64 native entry is missing finalized code".into())
    })?;
    let root_return = code.arm64_root_return().ok_or_else(|| {
        WasmError::internal("arm64 native root return continuation is missing".into())
    })?;

    let mut stack = vec![0u64; MAX_STACK_SLOTS];
    let stack_base = stack.as_mut_ptr();
    let stack_end = unsafe { stack_base.add(MAX_STACK_SLOTS) };

    unsafe {
        for (index, arg) in args.iter().enumerate() {
            *stack_base.add(index) = arg.to_raw();
        }
        if runtime.frame_prefix_slots as usize > args.len() {
            core::ptr::write_bytes(
                stack_base.add(args.len()),
                0,
                runtime.frame_prefix_slots as usize - args.len(),
            );
        }
    }
    ensure_stack_capacity(stack_base, stack_end, runtime.total_frame_slots)?;

    let mut ctx = NativeContext::new(store as *mut Store, stack_end);
    seed_root_call_link(compiled, runtime, stack_base, root_return)?;
    #[cfg(feature = "function-trace")]
    {
        function_trace::init_from_env();
        function_trace::native_root_entry(&mut ctx, spec, backend);
    }

    #[cfg(has_guard_pages)]
    {
        use crate::vm::native::{runtime::context::ctx_offset, trap_signal};
        trap_signal::install_signal_handler();
        trap_signal::set_trap_kind_offset(ctx_offset::TRAP_KIND as usize);
        trap_signal::reset_debug_state();
        ctx.trap_kind = 0;
    }

    let status = unsafe { entry(&mut ctx, stack_base) };

    #[cfg(has_guard_pages)]
    if ctx.trap_kind != 0 {
        let error = WasmError::trap("out of bounds memory access".into());
        #[cfg(feature = "function-trace")]
        function_trace::native_trap_current(&mut ctx, &error);
        return Err(error);
    }

    if status != 0 {
        let error = ctx.error.take().unwrap_or_else(|| {
            WasmError::internal("arm64 root entry failed without setting an error".into())
        });
        #[cfg(feature = "function-trace")]
        function_trace::native_trap_current(&mut ctx, &error);
        return Err(error);
    }

    let results_len = func_type.results().len();
    let mut out = InterpreterStack::with_exact_capacity(results_len);
    unsafe {
        for index in 0..results_len {
            out.push(*stack_base.add(index));
        }
    }
    #[cfg(feature = "function-trace")]
    {
        let results = unsafe { core::slice::from_raw_parts(stack_base, results_len) };
        function_trace::native_root_exit(&mut ctx, spec, results);
    }
    Ok(out)
}

fn seed_root_call_link(
    compiled: &crate::vm::native::code::CompiledNativeModule,
    runtime: &crate::vm::native::ir::runtime::MachineFunctionRuntime,
    fp: *mut u64,
    root_return: *const u8,
) -> Result<(), WasmError> {
    let call_scratch = runtime.call_scratch.ok_or_else(|| {
        WasmError::internal("arm64 root entry requires call scratch for unified return".into())
    })?;
    let layout = compiled.runtime().call_link;
    unsafe {
        *fp.add(call_scratch.base_slot as usize + (layout.continuation_offset / 8) as usize) =
            root_return as u64;
        *fp.add(call_scratch.base_slot as usize + (layout.caller_frame_offset / 8) as usize) =
            fp as u64;
        *fp.add(
            call_scratch.base_slot as usize + (layout.caller_result_base_offset / 8) as usize,
        ) = 0;
    }
    Ok(())
}

pub(crate) fn ensure_stack_capacity(
    fp: *mut u64,
    stack_end: *mut u64,
    total_frame_slots: u16,
) -> Result<(), WasmError> {
    let end =
        (fp as usize).saturating_add(total_frame_slots as usize * core::mem::size_of::<u64>());
    if end > stack_end as usize {
        return Err(WasmError::exhaustion("stack overflow".into()));
    }
    Ok(())
}

pub(crate) unsafe extern "C" fn arm64_raise_trap(ctx: *mut NativeContext, kind: u64) -> u32 {
    let Some(ctx) = (unsafe { ctx.as_mut() }) else {
        return 1;
    };
    let error = match kind {
        0 => WasmError::trap("unreachable executed".into()),
        1 => WasmError::trap("out of bounds memory access".into()),
        2 => WasmError::trap("out of bounds table access".into()),
        3 => WasmError::trap("invalid function reference".into()),
        4 => WasmError::trap("indirect call type mismatch".into()),
        5 => WasmError::trap("integer divide by zero".into()),
        6 => WasmError::trap("integer overflow".into()),
        7 => WasmError::exhaustion("stack overflow".into()),
        _ => WasmError::trap("native helper failed".into()),
    };
    #[cfg(feature = "function-trace")]
    function_trace::native_trap_current(ctx, &error);
    ctx.error = Some(error);
    1
}

pub(crate) unsafe extern "C" fn arm64_raise_unsupported(
    ctx: *mut NativeContext,
    func_id: u64,
) -> u32 {
    let Some(ctx) = (unsafe { ctx.as_mut() }) else {
        return 1;
    };
    let error = WasmError::invalid(alloc::format!(
        "arm64 backend has not finalized machine function {} yet",
        func_id
    ));
    #[cfg(feature = "function-trace")]
    function_trace::native_trap_current(ctx, &error);
    ctx.error = Some(error);
    1
}

#[cfg(test)]
mod tests {
    use alloc::{boxed::Box, rc::Rc, string::String, vec};

    use super::eval;
    use crate::{
        module::{entities::FunctionSpec, type_context::TypeContext, type_defs::FunctionType},
        vm::{
            entities::{FunctionInst, ModuleInst},
            native::build::ensure_module_compiled,
            runtime,
            store::Store,
        },
    };

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn runtime_eval_executes_block_like_empty_function() {
        let ty = Rc::new(FunctionType::new(vec![], vec![]));
        let types = TypeContext::new(vec![Rc::clone(&ty)]);
        let mut module = ModuleInst::new(String::from("m"), types);

        let mut dummy = FunctionSpec::new(Rc::clone(&ty), 0);
        dummy.set_code((&[0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec: dummy,
            type_index: 0,
        });

        let mut empty = FunctionSpec::new(Rc::clone(&ty), 0);
        empty.set_code((&[0x02, 0x40, 0x0b, 0x02, 0x40, 0x0b, 0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec: empty,
            type_index: 0,
        });

        let mut store = Box::new(Store::new(module));
        ensure_module_compiled(&store).expect("native compile");

        let func_ptr = &store.module().functions[1] as *const FunctionInst;
        let FunctionInst::Local { spec, .. } = (unsafe { &*func_ptr }) else {
            panic!("expected local function");
        };
        let code = spec.get_native_code().expect("native code");
        let results = eval(spec, code, &mut store, &[], "arm64").expect("arm64 eval");
        assert_eq!(results.len(), 0);

        let func_ref = unsafe { &*func_ptr };
        let results = runtime::eval(func_ref, &mut store, &[]).expect("runtime eval");
        assert_eq!(results.len(), 0);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn runtime_eval_call_dummy_returns_result() {
        use crate::value_type::ValueType;
        // Types: 0 = () -> (), 1 = () -> (i32)
        let void_ty = Rc::new(FunctionType::new(vec![], vec![]));
        let i32_ty = Rc::new(FunctionType::new(vec![], vec![ValueType::I32]));
        let types = TypeContext::new(vec![Rc::clone(&void_ty), Rc::clone(&i32_ty)]);
        let mut module = ModuleInst::new(String::from("m"), types);

        // Function 0: $dummy = () -> () : [end]
        let mut dummy = FunctionSpec::new(Rc::clone(&void_ty), 0);
        dummy.set_code((&[0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec: dummy,
            type_index: 0,
        });

        // Function 1: () -> (i32) : [call 0, i32.const 42, end]
        let mut caller = FunctionSpec::new(Rc::clone(&i32_ty), 0);
        caller.set_code((&[0x10, 0x00, 0x41, 0x2a, 0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec: caller,
            type_index: 1,
        });

        let mut store = Box::new(Store::new(module));
        ensure_module_compiled(&store).expect("native compile");

        let func_ptr = &store.module().functions[1] as *const FunctionInst;
        let FunctionInst::Local { spec, .. } = (unsafe { &*func_ptr }) else {
            panic!("expected local function");
        };
        let code = spec.get_native_code().expect("native code");
        let results = eval(spec, code, &mut store, &[], "arm64").expect("arm64 eval");
        assert_eq!(results.len(), 1);
        assert_eq!(results.peek_at_index(0), 42);
    }
}
