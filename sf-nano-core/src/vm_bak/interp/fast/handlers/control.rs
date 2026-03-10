//! Control flow operations.
//!
//! Handlers for: data (pseudo-instruction), term
//!
//! Note: Most control ops are implemented in C (handlers_c/control.c):
//! nop, drop, end, block, loop, select, if, else, unreachable, br, br_if, br_table
//! return is now also in C (handlers_c/call.c)

use super::common::{Context, Instruction};
use super::trap_with;
use crate::error::WasmError;

// NOTE: impl_return is now implemented in C (handlers_c/call.c)
// NOTE: impl_br_table is now implemented in C (handlers_c/control.c)

/// Terminal instruction handler - breaks the tail-call chain.
///
/// This mirrors xir_term: it simply returns NULL to exit the interpreter.
/// All frame management is handled by impl_return; impl_term just terminates.
/// Error state is stored in ctx.error and checked after trampoline returns.
#[no_mangle]
#[inline(always)]
pub extern "C" fn impl_term(
    _ctx: *mut Context,
    _pc: *mut Instruction,
    _fp_pp: *mut *mut u64,
    _p_l0: *mut u64,
    _p_l1: *mut u64,
    _p_l2: *mut u64,
) -> *mut Instruction {
    core::ptr::null_mut()
}

/// Data pseudo-instruction handler - traps if accidentally executed.
///
/// Data pseudo-instructions hold inline br_table entries. They should never
/// be reached during normal execution - br_table jumps over them to the target.
/// If executed, it indicates a bug in branch target calculation.
#[no_mangle]
#[inline(always)]
pub extern "C" fn impl_data(
    ctx: *mut Context,
    _pc: *mut Instruction,
    _fp_pp: *mut *mut u64,
    _p_l0: *mut u64,
    _p_l1: *mut u64,
    _p_l2: *mut u64,
) -> *mut Instruction {
    trap_with(ctx, WasmError::trap("executed data pseudo-instruction".into()))
}
