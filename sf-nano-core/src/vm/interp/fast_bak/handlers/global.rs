//! Global variable operations.
//!
//! Handlers for: global.get, global.set
//!
//! Phase 3 TOS-only:
//! - global_get (pop0_push1): Write global value to p_dst
//! - global_set (pop1_push0): Read value from p_src, store to global

use super::common::*;
use super::trap_with;
use crate::vm::interp::fast::encoding::global;

// =============================================================================
// Global Operations (Phase 3 TOS-only)
// =============================================================================

/// global.get: read global value to TOS
/// Encoding: see encoding.toml "global" pattern
/// tos_pattern = { pop = 0, push = 1 } - writes to p_dst
#[no_mangle]
#[inline(always)]
pub extern "C" fn impl_global_get(
    ctx: *mut Context,
    pc: *mut Instruction,
    _fp_pp: *mut *mut u64,
    _p_l0: *mut u64,
    _p_l1: *mut u64,
    _p_l2: *mut u64,
    // Phase 3: Operand pointer for output
    p_dst: *mut u64,
) -> *mut Instruction {
    let idx = global::decode_global_idx(pc) as usize;
    let store_ref = ctx_store(ctx);

    let g = store_ref.global(idx);
    let v = value_to_raw(g.value);

    // Write result to TOS
    unsafe { *p_dst = v };

    pc_fallthrough(pc)
}

/// global.set: store TOS value to global
/// Encoding: see encoding.toml "global" pattern
/// tos_pattern = { pop = 1, push = 0 } - reads from p_src
#[no_mangle]
#[inline(always)]
pub extern "C" fn impl_global_set(
    ctx: *mut Context,
    pc: *mut Instruction,
    _fp_pp: *mut *mut u64,
    _p_l0: *mut u64,
    _p_l1: *mut u64,
    _p_l2: *mut u64,
    // Phase 3: Operand pointer for input
    p_src: *mut u64,
) -> *mut Instruction {
    let idx = global::decode_global_idx(pc) as usize;

    // Read value from TOS and store to global
    let raw = unsafe { *p_src };
    let store_mut = ctx_store_mut(ctx);
    let g = store_mut.global_mut(idx);
    let ty = g.value_type;
    let val = raw_to_value(raw, ty);
    g.value = val;

    pc_fallthrough(pc)
}
