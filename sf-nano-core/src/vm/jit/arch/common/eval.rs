//! Shared native eval: call a JIT-compiled function entry point.
//!
//! This is identical across all architectures since the entry signature
//! and call convention are the same.

#[cfg(not(sf_has_guard_pages))]
use crate::collections;

use crate::{
    error::WasmError,
    module::entities::FunctionSpec,
    vm::{
        jit::runtime::{
            code::NativeCode, collect_native_results_from_stack, context::NativeContext,
        },
        result_buffer::ResultBuffer,
        store::Store,
        value::Value,
        value_encoding::value_to_machine_raw_in_store,
    },
};

#[cfg(sf_has_guard_pages)]
use super::helpers::trap_error;
#[cfg(sf_has_guard_pages)]
use crate::vm::jit::machine::machine_ir::MachineTrapKind;
#[cfg(sf_has_guard_pages)]
use crate::vm::jit::runtime::trap_signal;

#[cfg(sf_call_trace)]
use crate::vm::jit::debug::function_trace;

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

    // Sized by runtime config. Hosted default matches the former
    // 2-MiB `constants::MAX_STACK_SIZE`; MCU embedders override via
    // `set_runtime_config(.. wasm_stack_bytes: N ..)`.
    let stack_bytes = crate::runtime_config().wasm_stack_bytes;
    if stack_bytes < core::mem::size_of::<u64>() {
        return Err(WasmError::runtime_not_configured());
    }
    #[cfg(sf_has_guard_pages)]
    let max_frame_bytes = compiled.max_frame_bytes();
    #[cfg(sf_has_guard_pages)]
    let stack = match store.take_native_stack_cache() {
        Some(stack) if stack.supports(stack_bytes, max_frame_bytes) => stack,
        _ => {
            crate::vm::jit::runtime::guard_pages::GuardPageStack::new(stack_bytes, max_frame_bytes)?
        }
    };
    #[cfg(sf_has_guard_pages)]
    let stack_base = stack.base();
    #[cfg(sf_has_guard_pages)]
    let stack_end = stack.end();
    #[cfg(sf_has_guard_pages)]
    let stack_guard_end = stack.guard_end();

    #[cfg(not(sf_has_guard_pages))]
    let mut stack = {
        let stack_slots = stack_bytes / core::mem::size_of::<u64>();
        let mut stack = store
            .take_native_stack_cache()
            .unwrap_or_else(|| collections::vec![0u64; stack_slots]);
        if stack.len() < stack_slots {
            stack.resize(stack_slots, 0);
        }
        stack
    };
    #[cfg(not(sf_has_guard_pages))]
    let stack_base = stack.as_mut_ptr();
    #[cfg(not(sf_has_guard_pages))]
    let stack_end = unsafe { stack_base.add(stack.len()) };

    unsafe {
        for (index, arg) in args.iter().enumerate() {
            *stack_base.add(index) =
                value_to_machine_raw_in_store(*arg, compiled.backend().gp_unit_bytes, store);
        }
        // Note: zero-init of non-param locals is performed by the callee
        // itself at function entry, only for slots flagged by the SsaProgram's
        // `cell_info.reads_before_write` analysis. The C entry path no
        // longer needs to pre-zero the frame prefix.
    }
    if let Err(error) = ensure_stack_capacity(stack_base, stack_end, runtime.total_frame_slots) {
        store.cache_native_stack(stack);
        return Err(error);
    }

    let n_globals = store.module().globals.len();
    let store_ptr = store as *mut Store;
    let mut ctx = if let Some(mut ctx) = store.take_native_context_cache() {
        ctx.prepare_for_invocation(store_ptr, stack_end);
        ctx
    } else {
        NativeContext::new(store_ptr, stack_end, n_globals)
    };
    #[cfg(sf_has_guard_pages)]
    {
        ctx.stack_guard_end = stack_guard_end;
    }
    ctx.seed_local_call_infos(compiled);
    // Under the caller-restored local-call ABI, no software call-link record
    // is needed at the root. The public-entry caller stub calls the internal
    // entry, and fallback returns are published in the root frame slots that
    // `collect_native_results_from_stack` reads below.
    #[cfg(sf_call_trace)]
    {
        function_trace::init_from_env();
        function_trace::native_root_entry(&mut ctx, spec, backend);
    }

    #[cfg(sf_has_guard_pages)]
    {
        use crate::vm::jit::runtime::context::ctx_offset;
        trap_signal::install_signal_handler();
        trap_signal::set_context_offsets(
            ctx_offset::TRAP_KIND as usize,
            ctx_offset::STACK_END as usize,
            ctx_offset::STACK_GUARD_END as usize,
        );
        trap_signal::reset_debug_state();
        ctx.trap_kind = 0;
    }

    let result = (|| {
        let status = unsafe { entry(&mut *ctx, stack_base) };

        #[cfg(sf_has_guard_pages)]
        if ctx.trap_kind != 0 {
            let error = match ctx.trap_kind {
                trap_signal::TRAP_STACK_OVERFLOW => trap_error(MachineTrapKind::StackOverflow),
                _ => trap_error(MachineTrapKind::MemoryOutOfBounds),
            };
            #[cfg(sf_call_trace)]
            function_trace::native_trap_current(&mut ctx, &error);
            return Err(error);
        }

        if status != 0 {
            let error = ctx
                .error
                .take()
                .or_else(|| core::mem::take(&mut ctx.pending_escape).into_error())
                .unwrap_or_else(|| {
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
                store,
            )
        }?;
        #[cfg(sf_call_trace)]
        {
            let results_len = func_type.results().len();
            let results = unsafe { core::slice::from_raw_parts(stack_base, results_len) };
            function_trace::native_root_exit(&mut ctx, spec, results);
        }
        Ok(out)
    })();
    store.cache_native_context(ctx);
    store.cache_native_stack(stack);
    result
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
