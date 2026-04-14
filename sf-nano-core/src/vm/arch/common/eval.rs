//! Shared native eval: call a JIT-compiled function entry point.
//!
//! This is identical across all architectures since the entry signature
//! and call convention are the same.

use crate::collections;

use crate::{
    constants::MAX_STACK_SIZE,
    error::WasmError,
    module::entities::FunctionSpec,
    vm::{
        result_buffer::ResultBuffer,
        runtime::{code::NativeCode, collect_native_results_from_stack, context::NativeContext},
        store::Store,
        value::Value,
    },
};

#[cfg(sf_call_trace)]
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
        return Err(WasmError::invalid("invalid argument count"));
    }

    let compiled = code.compiled();
    let func_id = code.func_id();
    let runtime = compiled
        .abi()
        .functions
        .get(func_id.0 as usize)
        .ok_or_else(|| WasmError::internal("native entry function is missing runtime metadata"))?;
    let entry = code
        .native_entry()
        .ok_or_else(|| WasmError::internal("native entry is missing finalized code"))?;

    let mut stack = collections::vec![0u64; MAX_STACK_SLOTS];
    let stack_base = stack.as_mut_ptr();
    let stack_end = unsafe { stack_base.add(MAX_STACK_SLOTS) };

    unsafe {
        for (index, arg) in args.iter().enumerate() {
            *stack_base.add(index) = arg.to_raw();
        }
        // Note: zero-init of non-param locals is performed by the callee
        // itself at function entry, only for slots flagged by the SsaProgram's
        // `local_slot_info.reads_before_write` analysis. The C entry path no
        // longer needs to pre-zero the frame prefix.
    }
    ensure_stack_capacity(stack_base, stack_end, runtime.total_frame_slots)?;

    let mut ctx = NativeContext::new(store as *mut Store, stack_end);
    ctx.seed_local_call_infos(compiled);
    // Under the new local-call ABI, no software call-link record is needed
    // at the root. The public-entry caller stub builds a backend-private
    // call record on the host stack itself, calls the internal entry, and
    // the body's unified Return copies results into the bytes at
    // `stack_base` (the root frame), which is exactly what
    // `collect_native_results_from_stack` reads from below.
    #[cfg(sf_call_trace)]
    {
        function_trace::init_from_env();
        function_trace::native_root_entry(&mut ctx, spec, backend);
    }

    #[cfg(sf_has_guard_pages)]
    {
        use crate::vm::runtime::{context::ctx_offset, trap_signal};
        trap_signal::install_signal_handler();
        trap_signal::set_trap_kind_offset(ctx_offset::TRAP_KIND as usize);
        trap_signal::reset_debug_state();
        ctx.trap_kind = 0;
    }

    let status = unsafe { entry(&mut ctx, stack_base) };

    #[cfg(sf_has_guard_pages)]
    if ctx.trap_kind != 0 {
        let error = WasmError::trap("out of bounds memory access");
        #[cfg(sf_call_trace)]
        function_trace::native_trap_current(&mut ctx, &error);
        return Err(error);
    }

    if status != 0 {
        let error = ctx.error.take().unwrap_or_else(|| {
            WasmError::internal("native root entry failed without setting an error")
        });
        #[cfg(sf_call_trace)]
        function_trace::native_trap_current(&mut ctx, &error);
        return Err(error);
    }

    let out = unsafe {
        collect_native_results_from_stack(
            stack_base,
            func_type.results(),
            compiled.backend().gp_unit_bytes,
        )
    };
    #[cfg(sf_call_trace)]
    {
        let results_len = func_type.results().len();
        let results = unsafe { core::slice::from_raw_parts(stack_base, results_len) };
        function_trace::native_root_exit(&mut ctx, spec, results);
    }
    Ok(out)
}

pub(crate) fn ensure_stack_capacity(
    fp: *mut u64,
    stack_end: *mut u64,
    total_frame_slots: u16,
) -> Result<(), WasmError> {
    let end =
        (fp as usize).saturating_add(total_frame_slots as usize * core::mem::size_of::<u64>());
    if end > stack_end as usize {
        return Err(WasmError::exhaustion("stack overflow"));
    }
    Ok(())
}
