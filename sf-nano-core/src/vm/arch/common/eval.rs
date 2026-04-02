//! Shared native eval: call a JIT-compiled function entry point.
//!
//! This is identical across all architectures since the entry signature
//! and call convention are the same.

use alloc::vec;

use crate::{
    constants::MAX_STACK_SIZE,
    error::WasmError,
    module::entities::FunctionSpec,
    vm::{
        result_buffer::ResultBuffer,
        runtime::{
            code::{CompiledNativeModule, NativeCode},
            context::NativeContext,
        },
        store::Store,
        value::Value,
    },
};

#[cfg(feature = "function-trace")]
use crate::vm::debug::function_trace;

const MAX_STACK_SLOTS: usize = MAX_STACK_SIZE / core::mem::size_of::<u64>();

pub(crate) fn eval(
    spec: &FunctionSpec,
    code: &NativeCode,
    store: &mut Store,
    args: &[Value],
    backend: &'static str,
) -> Result<ResultBuffer, WasmError> {
    let _ = backend;
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
        .abi()
        .functions
        .get(func_id.0 as usize)
        .ok_or_else(|| {
            WasmError::internal("native entry function is missing runtime metadata".into())
        })?;
    let entry = code
        .native_entry()
        .ok_or_else(|| WasmError::internal("native entry is missing finalized code".into()))?;
    let root_return = code
        .native_root_return()
        .ok_or_else(|| WasmError::internal("native root return continuation is missing".into()))?;

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
    ctx.seed_local_call_infos(compiled);
    seed_root_call_link(compiled, runtime, stack_base, root_return)?;
    #[cfg(feature = "function-trace")]
    {
        function_trace::init_from_env();
        function_trace::native_root_entry(&mut ctx, spec, backend);
    }

    #[cfg(has_guard_pages)]
    {
        use crate::vm::runtime::{context::ctx_offset, trap_signal};
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
            WasmError::internal("native root entry failed without setting an error".into())
        });
        #[cfg(feature = "function-trace")]
        function_trace::native_trap_current(&mut ctx, &error);
        return Err(error);
    }

    let out = unsafe {
        crate::vm::runtime::collect_native_results_from_stack(
            stack_base,
            func_type.results(),
            compiled.backend().gp_unit_bytes,
        )
    };
    #[cfg(feature = "function-trace")]
    {
        let results_len = func_type.results().len();
        let results = unsafe { core::slice::from_raw_parts(stack_base, results_len) };
        function_trace::native_root_exit(&mut ctx, spec, results);
    }
    Ok(out)
}

fn seed_root_call_link(
    compiled: &CompiledNativeModule,
    runtime: &crate::vm::machine::machine_ir::MachineFunctionAbi,
    fp: *mut u64,
    root_return: *const u8,
) -> Result<(), WasmError> {
    let call_scratch = runtime.call_scratch.ok_or_else(|| {
        WasmError::internal("native root entry requires call scratch for unified return".into())
    })?;
    let layout = compiled.abi().call_link;
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
