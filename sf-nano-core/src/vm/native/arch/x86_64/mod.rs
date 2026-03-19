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
        native::{
            code::{CompiledNativeModule, NativeCode},
            runtime::context::NativeContext,
        },
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
    let entry = code.x86_64_entry().ok_or_else(|| {
        WasmError::internal("x86_64 native entry is missing finalized code".into())
    })?;
    let root_return = code.x86_64_root_return().ok_or_else(|| {
        WasmError::internal("x86_64 native root return continuation is missing".into())
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
            WasmError::internal("x86_64 root entry failed without setting an error".into())
        });
        #[cfg(feature = "function-trace")]
        function_trace::native_trap_current(&mut ctx, &error);
        return Err(error);
    }

    let results_len = func_type.results().len();
    let out = unsafe {
        crate::vm::native::runtime::collect_native_results_from_stack(
            stack_base,
            func_type.results(),
            compiled.backend().gp_unit_bytes,
        )
    };
    #[cfg(feature = "function-trace")]
    {
        let results = unsafe { core::slice::from_raw_parts(stack_base, results_len) };
        function_trace::native_root_exit(&mut ctx, spec, results);
    }
    Ok(out)
}

fn ensure_stack_capacity(
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

fn seed_root_call_link(
    compiled: &CompiledNativeModule,
    runtime: &crate::vm::native::ir::runtime::MachineFunctionRuntime,
    fp: *mut u64,
    root_return: *const u8,
) -> Result<(), WasmError> {
    let call_scratch = runtime.call_scratch.ok_or_else(|| {
        WasmError::internal("x86_64 root entry requires call scratch for unified return".into())
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

pub(crate) unsafe extern "C" fn x86_64_raise_trap(ctx: *mut NativeContext, kind: u64) -> u32 {
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
        7 => WasmError::exhaustion("call stack exhausted".into()),
        8 => WasmError::exhaustion("stack overflow".into()),
        _ => WasmError::trap("native helper failed".into()),
    };
    #[cfg(feature = "function-trace")]
    function_trace::native_trap_current(ctx, &error);
    ctx.error = Some(error);
    1
}

pub(crate) unsafe extern "C" fn x86_64_raise_unsupported(
    ctx: *mut NativeContext,
    func_id: u64,
) -> u32 {
    let Some(ctx) = (unsafe { ctx.as_mut() }) else {
        return 1;
    };
    let error = WasmError::invalid(alloc::format!(
        "x86_64 backend has not finalized machine function {} yet",
        func_id
    ));
    #[cfg(feature = "function-trace")]
    function_trace::native_trap_current(ctx, &error);
    ctx.error = Some(error);
    1
}
