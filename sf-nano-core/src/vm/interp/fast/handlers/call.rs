//! Call operations.
//!
//! Handlers for: call_external, call_internal, call_indirect, call_ref
//!
//! Note: call_local is implemented in C (handlers_c/call.c)

use alloc::vec::Vec;

use super::common::*;
use super::trap_with;
use crate::vm::interp::fast::encoding::{call_external, call_indirect, call_ref};
use crate::vm::entities::{Caller, MemInst};
use crate::error::WasmError;
#[cfg(feature = "function-trace")]
use crate::vm::function_trace;

/// Read a value from the operand stack using fp-relative addressing.
/// operand_base = fp + operand_base_offset/8, slot at operand_base[index]
#[inline(always)]
fn operand_read(fp_pp: *mut *mut u64, operand_base_offset: usize, index: usize) -> u64 {
    unsafe {
        let fp = *fp_pp;
        let operand_base = fp.add(operand_base_offset / 8);
        *operand_base.add(index)
    }
}

/// Maximum call depth to prevent native stack overflow.
/// Make sure to change the same constant in vm_trampoline.h if modified here.
const MAX_CALL_DEPTH: u64 = 300;

// =============================================================================
// C Handler Helper Functions (called from C for slow paths)
// =============================================================================

/// Called by C impl_return when returning from a cross-module call.
/// Restores caller's module context and refreshes mem0.
///
/// # Safety
/// - `ctx` must be a valid pointer to Context
/// - `saved_module_ptr` must be a valid pointer to a live ModuleInst
/// - mem0 fields in ctx will be refreshed for the restored module
#[no_mangle]
pub unsafe extern "C" fn fast_return_cross_module(
    ctx: *mut Context,
    saved_module_ptr: u64,
) {
    let module = saved_module_ptr as *const ModuleInst;
    ctx_mut(ctx).current_module = module;
    let store = ctx_store(ctx);
    refresh_mem0_for_module(ctx, store, &*module);
}

// =============================================================================
// Unified Stack Helpers
// =============================================================================

/// Enter an internal callee using unified stack (no run_trampoline).
///
/// Sets up callee frame with 3 metadata slots, handles cross-module context,
/// and returns the callee's entry instruction for tail-call dispatch.
///
/// `delta` is the absolute frame offset where callee frame starts (fp + delta).
/// `pc` is the current call instruction (return_pc = pc + 1).
#[inline(always)]
fn enter_unified_callee(
    ctx: *mut Context,
    pc: *mut Instruction,
    fp_pp: *mut *mut u64,
    store_ref: &Store,
    callee: &FunctionInst,
    delta: usize,
) -> Result<*mut Instruction, WasmError> {
    // 1. Resolve callee — must be a local function
    let spec = match callee {
        FunctionInst::Local { spec, .. } => spec,
        FunctionInst::External { .. } => {
            return Err(WasmError::trap("unexpected external function".into()));
        }
    };

    // 2. Ensure callee has fast code (lazy compilation)
    if !spec.has_fast_code() {
        crate::vm::interp::fast::precompile::precompile_module_two_pass(
            store_ref,
        )?;
    }

    let cache = spec.fast_cache();
    let entry = cache.entry();

    // 3. Get callee function layout
    let func_type = spec.func_type();
    let params_count = func_type.params().len();
    let locals_count = spec.locals().len();
    let frame_size = params_count + locals_count;

    // 4. Compute callee_fp
    let callee_fp = unsafe { (*fp_pp).add(delta) };

    // 5. Stack overflow check (frame + 3 metadata slots)
    let new_stack_top = unsafe { callee_fp.add(frame_size + 3) };
    if new_stack_top > unsafe { (*ctx).hot.stack_end } {
        return Err(WasmError::exhaustion("stack overflow".into()));
    }

    // 6. Call depth check
    {
        let ctx_ref = ctx_mut(ctx);
        if ctx_ref.hot.call_depth >= MAX_CALL_DEPTH {
            return Err(WasmError::exhaustion("call stack exhausted".into()));
        }
        ctx_ref.hot.call_depth += 1;
    }

    // 7. Zero callee's locals
    if locals_count > 0 {
        unsafe {
            core::ptr::write_bytes(callee_fp.add(params_count), 0, locals_count);
        }
    }

    // 8. Push metadata (return_pc, saved_fp, saved_module)
    unsafe {
        *callee_fp.add(frame_size) = pc_fallthrough(pc) as u64; // return_pc
        *callee_fp.add(frame_size + 1) = *fp_pp as u64; // saved_fp
    }

    // 9. Single-module: no cross-module detection needed in sf-nano
    unsafe {
        *callee_fp.add(frame_size + 2) = 0;
    }

    // 10. Update fp to callee
    unsafe {
        *fp_pp = callee_fp;
    }

    #[cfg(feature = "function-trace")]
    unsafe {
        if function_trace::enabled() {
            function_trace::fast_function_trace_enter_entry(ctx, entry);
        }
    }

    // 11. Return callee entry for tail-call dispatch
    Ok(entry)
}

// =============================================================================
// Call Operations
// =============================================================================

/// Invoke an external (host/imported) function and write results back to the frame.
///
/// `delta` is the frame offset where callee arguments start (fp + delta).
#[inline(always)]
fn invoke_external_callee(
    callee: &FunctionInst,
    store_ref: &Store,
    _module_ref: &ModuleInst,
    fp_pp: *mut *mut u64,
    delta: usize,
) -> Result<(), WasmError> {
    let (func_type, callback) = match callee {
        FunctionInst::External { func_type, callback } => (func_type, callback),
        _ => return Err(WasmError::internal("expected external function".into())),
    };

    let params = func_type.params();
    let results = func_type.results();
    let results_len = results.len();

    // Build args from frame slots
    let args: Vec<ExtValue> = params
        .iter()
        .enumerate()
        .map(|(i, ty)| {
            let raw = frame_read(fp_pp, delta + i);
            raw_to_value(raw, *ty)
        })
        .collect();

    // Allocate results buffer
    let mut ret_vals = alloc::vec![ExtValue::default(); results_len];

    // Build Caller with memory access
    let mem_slice = if !store_ref.module().memories.is_empty() {
        // Safety: we need mutable access to memory for the host function.
        // The store is borrowed mutably via ctx_store_mut during the handler,
        // but here we only borrow the memory slice, not the whole store.
        let mem = &store_ref.module().memories[0] as *const MemInst as *mut MemInst;
        unsafe { Some((*mem).data.as_mut_slice()) }
    } else {
        None
    };
    let mut caller = Caller::new(mem_slice);

    // Call the external function
    callback(&mut caller, &args, &mut ret_vals)?;

    if ret_vals.len() != results_len {
        return Err(WasmError::internal("result count mismatch".into()));
    }

    // Results written to fp[delta..delta+results_len]
    for (i, v) in ret_vals.into_iter().enumerate() {
        frame_write(fp_pp, delta + i, value_to_raw(v));
    }

    Ok(())
}

/// External function call handler (imports, WASI, host functions).
///
/// Encoding: see encoding.toml "call_external" pattern
#[no_mangle]
#[inline(always)]
pub extern "C" fn impl_call_external(
    ctx: *mut Context,
    pc: *mut Instruction,
    fp_pp: *mut *mut u64,
    p_l0: *mut u64,
    p_l1: *mut u64,
    p_l2: *mut u64,
) -> *mut Instruction {
    let delta = call_external::decode_delta(pc) as usize;
    let func_idx = call_external::decode_func_idx(pc) as usize;

    // Spill l0/l1/l2 before external call
    l0_spill(fp_pp, p_l0);
    l1_spill(fp_pp, p_l1);
    l2_spill(fp_pp, p_l2);
    let store_ptr = ctx_store(ctx) as *const Store;

    let store_ref: &Store = ptr_ref(store_ptr);
    let callee: &FunctionInst = store_ref.function(func_idx);
    let module_ref = store_ref.module();

    if let Err(e) = invoke_external_callee(callee, store_ref, module_ref, fp_pp, delta) {
        return trap_with(ctx, e);
    }
    // Results are at fp[delta..delta+results_len], no sp update needed
    refresh_mem0_for_module(ctx, store_ref, module_ref);

    // Fill l0/l1/l2 after external call (fp unchanged, but host may have modified memory)
    l0_fill(fp_pp, p_l0);
    l1_fill(fp_pp, p_l1);
    l2_fill(fp_pp, p_l2);

    pc_fallthrough(pc)
}

// NOTE: impl_call_local is now implemented in C (handlers_c/call.c)

/// Optimized call handler for precompiled internal functions.
///
/// Uses unified stack model: sets up callee frame with metadata slots and
/// returns callee's entry instruction for tail-call dispatch (no run_trampoline).
///
/// Encoding: see encoding.toml "call_internal" pattern
#[no_mangle]
#[inline(always)]
pub extern "C" fn impl_call_internal(
    ctx: *mut Context,
    pc: *mut Instruction,
    fp_pp: *mut *mut u64,
    p_l0: *mut u64,
    p_l1: *mut u64,
    p_l2: *mut u64,
) -> *mut Instruction {
    use super::super::encoding::call_internal;

    let callee_func_ptr = call_internal::decode_callee_func(pc) as *const FunctionInst;
    let delta = call_internal::decode_delta(pc) as usize;

    // Spill l0/l1/l2 before frame setup
    l0_spill(fp_pp, p_l0);
    l1_spill(fp_pp, p_l1);
    l2_spill(fp_pp, p_l2);

    let callee: &FunctionInst = ptr_ref(callee_func_ptr);
    let store_ref = ctx_store(ctx);

    match enter_unified_callee(ctx, pc, fp_pp, store_ref, callee, delta)
    {
        Ok(entry) => entry,
        Err(e) => trap_with(ctx, e),
    }
}

/// Encoding: see encoding.toml "call_indirect" pattern
#[no_mangle]
#[inline(always)]
pub extern "C" fn impl_call_indirect(
    ctx: *mut Context,
    pc: *mut Instruction,
    fp_pp: *mut *mut u64,
    p_l0: *mut u64,
    p_l1: *mut u64,
    p_l2: *mut u64,
) -> *mut Instruction {
    let delta = call_indirect::decode_delta(pc) as usize;
    let type_idx = call_indirect::decode_type_idx(pc) as usize;
    let table_idx = call_indirect::decode_table_idx(pc) as usize;
    let operand_base_offset = call_indirect::decode_operand_base_offset(pc) as usize;
    let height = call_indirect::decode_height(pc) as usize;
    let store_ptr: *const Store = ctx_store(ctx) as *const Store;

    // Spill l0/l1/l2 before call
    l0_spill(fp_pp, p_l0);
    l1_spill(fp_pp, p_l1);
    l2_spill(fp_pp, p_l2);

    // Read element index from operand stack (at height - 1, as index is top of stack)
    let elem_index = operand_read(fp_pp, operand_base_offset, height - 1) as usize;

    // Get expected type from type_idx
    let store_ref: &Store = ptr_ref(store_ptr);
    let module_ref = store_ref.module();
    let expected_ty = match module_ref.get_type(type_idx as u32) {
        Some(t) => t.clone(),
        None => return trap_with(ctx, WasmError::trap("indirect call type error".into())),
    };

    // Get table and look up function reference
    let table = store_ref.table(table_idx);
    if elem_index >= table.elements.len() {
        return trap_with(ctx, WasmError::trap("undefined element".into()));
    }
    let func_ref = table.elements[elem_index];
    if func_ref.is_null() {
        return trap_with(ctx, WasmError::trap("uninitialized element".into()));
    }

    // Look up callee by function index (raw_value is the function index in sf-nano)
    let func_index = func_ref.raw_value();
    if func_index >= store_ref.module().functions.len() {
        return trap_with(ctx, WasmError::trap("uninitialized element".into()));
    }
    let callee: &FunctionInst = store_ref.function(func_index);

    // Check function type equivalence
    let actual_type = callee.func_type();
    if *actual_type != *expected_ty {
        let type_context = &module_ref.types;
        let mut types_equivalent = false;
        for (idx, ft) in type_context.as_slice().iter().enumerate() {
            if **ft == *actual_type {
                types_equivalent = type_context.types_equivalent(idx as u32, type_idx as u32);
                break;
            }
        }
        if !types_equivalent {
            return trap_with(ctx, WasmError::trap("indirect call type mismatch".into()));
        }
    }

    if callee.is_external() {
        if let Err(e) = invoke_external_callee(callee, store_ref, module_ref, fp_pp, delta) {
            return trap_with(ctx, e);
        }
        refresh_mem0_for_module(ctx, store_ref, module_ref);
        // Fill l0/l1/l2 after external call
        l0_fill(fp_pp, p_l0);
        l1_fill(fp_pp, p_l1);
        l2_fill(fp_pp, p_l2);
        return pc_fallthrough(pc);
    }

    // Internal call using unified stack (no run_trampoline).
    match enter_unified_callee(ctx, pc, fp_pp, store_ref, callee, delta) {
        Ok(entry) => entry,
        Err(e) => trap_with(ctx, e),
    }
}

/// Encoding: see encoding.toml "call_ref" pattern
#[no_mangle]
#[inline(always)]
pub extern "C" fn impl_call_ref(
    ctx: *mut Context,
    pc: *mut Instruction,
    fp_pp: *mut *mut u64,
    p_l0: *mut u64,
    p_l1: *mut u64,
    p_l2: *mut u64,
) -> *mut Instruction {
    let delta = call_ref::decode_delta(pc) as usize;
    let type_idx = call_ref::decode_type_idx(pc) as usize;
    let operand_base_offset = call_ref::decode_operand_base_offset(pc) as usize;
    let height = call_ref::decode_height(pc) as usize;
    let store_ptr = ctx_store(ctx) as *const Store;

    // Spill l0/l1/l2 before call
    l0_spill(fp_pp, p_l0);
    l1_spill(fp_pp, p_l1);
    l2_spill(fp_pp, p_l2);

    // Read function reference from operand stack (at height - 1, as ref is top of stack)
    let func_ref_raw = operand_read(fp_pp, operand_base_offset, height - 1) as usize;

    // Check for null reference
    if func_ref_raw == usize::MAX {
        return trap_with(ctx, WasmError::trap("null function reference".into()));
    }

    let func_ref = VmRefHandle::new(func_ref_raw);
    let func_index = func_ref.raw_value();

    // Look up the function
    let store_ref: &Store = ptr_ref(store_ptr);
    if func_index >= store_ref.module().functions.len() {
        return trap_with(ctx, WasmError::trap("invalid function reference".into()));
    }
    let callee: &FunctionInst = store_ref.function(func_index);

    // Verify type matches (for typed call_ref)
    if type_idx != usize::MAX {
        let module_ref = store_ref.module();
        if let Some(expected_ty) = module_ref.get_type(type_idx as u32) {
            let actual_type = callee.func_type();
            if *actual_type != **expected_ty {
                return trap_with(ctx, WasmError::trap("call_ref type mismatch".into()));
            }
        }
    }

    if callee.is_external() {
        let module_ref = store_ref.module();
        if let Err(e) = invoke_external_callee(callee, store_ref, module_ref, fp_pp, delta) {
            return trap_with(ctx, e);
        }
        refresh_mem0_for_module(ctx, store_ref, module_ref);
        // Fill l0/l1/l2 after external call
        l0_fill(fp_pp, p_l0);
        l1_fill(fp_pp, p_l1);
        l2_fill(fp_pp, p_l2);
        return pc_fallthrough(pc);
    }

    // Internal call using unified stack (no run_trampoline).
    match enter_unified_callee(ctx, pc, fp_pp, store_ref, callee, delta) {
        Ok(entry) => entry,
        Err(e) => trap_with(ctx, e),
    }
}
