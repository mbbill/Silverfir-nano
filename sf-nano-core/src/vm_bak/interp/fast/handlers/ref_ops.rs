//! Reference operations (Phase 3 TOS-only).
//!
//! Handlers for: ref.null, ref.is_null, ref.func, ref.as_non_null, ref.eq
//!
//! Phase 3 TOS-only: All handlers use TOS operand pointers (p_src, p_dst, etc.)
//! - pop0_push1: Writes output to p_dst
//! - pop1_push1: Reads from p_src, writes to p_dst
//! - pop2_push1: Reads from p_lhs/p_rhs, writes to p_dst

use super::common::*;
use super::trap_with;
use crate::vm::interp::fast::encoding::ref_func;
use crate::error::WasmError;

// =============================================================================
// Basic Reference Operations (Phase 3 TOS-only)
// =============================================================================

/// ref.null: push null reference (0→1)
/// tos_pattern = { pop = 0, push = 1 } - writes to p_dst
#[no_mangle]
#[inline(always)]
pub extern "C" fn impl_ref_null(
    _ctx: *mut Context,
    pc: *mut Instruction,
    _fp_pp: *mut *mut u64,
    _p_l0: *mut u64,
    _p_l1: *mut u64,
    _p_l2: *mut u64,
    // Phase 3: Operand pointer for output
    p_dst: *mut u64,
) -> *mut Instruction {
    let null_val = usize::MAX as u64;
    unsafe { *p_dst = null_val };
    pc_fallthrough(pc)
}

/// ref.is_null: check if ref is null (1→1)
/// tos_pattern = { pop = 1, push = 1 } - reads from p_src, writes to p_dst
#[no_mangle]
#[inline(always)]
pub extern "C" fn impl_ref_is_null(
    _ctx: *mut Context,
    pc: *mut Instruction,
    _fp_pp: *mut *mut u64,
    _p_l0: *mut u64,
    _p_l1: *mut u64,
    _p_l2: *mut u64,
    // Phase 3: Operand pointers
    p_src: *mut u64,
    p_dst: *mut u64,
) -> *mut Instruction {
    let v = unsafe { *p_src } as usize;
    let result = if v == usize::MAX { 1u64 } else { 0u64 };
    unsafe { *p_dst = result };
    pc_fallthrough(pc)
}

/// ref.func: push function reference (0→1)
/// tos_pattern = { pop = 0, push = 1 } - writes to p_dst
#[no_mangle]
#[inline(always)]
pub extern "C" fn impl_ref_func(
    _ctx: *mut Context,
    pc: *mut Instruction,
    _fp_pp: *mut *mut u64,
    _p_l0: *mut u64,
    _p_l1: *mut u64,
    _p_l2: *mut u64,
    // Phase 3: Operand pointer for output
    p_dst: *mut u64,
) -> *mut Instruction {
    let func_idx = ref_func::decode_func_idx(pc) as usize;

    // In sf-nano, function references are simply the function index
    let raw = func_idx;
    unsafe { *p_dst = raw as u64 };
    pc_fallthrough(pc)
}

/// ref.as_non_null: validate ref is not null (1→1)
/// tos_pattern = { pop = 1, push = 1 } - reads from p_src, writes to p_dst
#[no_mangle]
#[inline(always)]
pub extern "C" fn impl_ref_as_non_null(
    ctx: *mut Context,
    pc: *mut Instruction,
    _fp_pp: *mut *mut u64,
    _p_l0: *mut u64,
    _p_l1: *mut u64,
    _p_l2: *mut u64,
    // Phase 3: Operand pointers
    p_src: *mut u64,
    p_dst: *mut u64,
) -> *mut Instruction {
    let v = unsafe { *p_src };
    if v as usize == usize::MAX {
        return trap_with(ctx, WasmError::trap("null reference".into()));
    }
    unsafe { *p_dst = v };
    pc_fallthrough(pc)
}

/// ref.eq: compare two refs for equality (2→1)
/// tos_pattern = { pop = 2, push = 1 } - reads from p_lhs/p_rhs, writes to p_dst
#[no_mangle]
#[inline(always)]
pub extern "C" fn impl_ref_eq(
    _ctx: *mut Context,
    pc: *mut Instruction,
    _fp_pp: *mut *mut u64,
    _p_l0: *mut u64,
    _p_l1: *mut u64,
    _p_l2: *mut u64,
    // Phase 3: Operand pointers
    p_lhs: *mut u64,
    p_rhs: *mut u64,
    p_dst: *mut u64,
) -> *mut Instruction {
    let ref1 = unsafe { *p_lhs } as usize;
    let ref2 = unsafe { *p_rhs } as usize;
    let result = if ref1 == ref2 { 1u64 } else { 0u64 };
    unsafe { *p_dst = result };
    pc_fallthrough(pc)
}
