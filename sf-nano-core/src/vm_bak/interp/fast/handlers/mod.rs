// Allow #[inline(always)] on #[no_mangle] functions - needed for cross-language LTO
#![allow(unused_attributes)]

//! Handler implementations for the fast interpreter.
//!
//! This module provides the centralized handler declarations and organizes
//! handler implementations by category.
//!
//! ## Module Organization
//!
//! - `common` - Shared helper functions and re-exported types
//! - `control` - Control flow (nop, unreachable, return, br, br_if, br_table, if, else)
//! - `global` - Global variable access (global.get, global.set)
//! - `memory` - Memory operations (load, store, size, grow, bulk ops)
//! - `table` - Table operations (get, set, init, copy, grow, size, fill)
//! - `ref_ops` - Reference operations (ref.null, ref.func, ref.is_null, ref.as_non_null, ref.eq)
//! - `call` - Call operations (call, call_indirect, call_ref)
//!
//! ## Handler Signature
//!
//! All handlers have the same extern "C" signature (no-stack model):
//! ```ignore
//! pub extern "C" fn impl_<name>(
//!     ctx: *mut Context,
//!     pc: *mut Instruction,
//!     fp_pp: *mut *mut u64,
//! ) -> *mut Instruction
//! ```

use super::{context::Context, instruction::Instruction};
use crate::error::WasmError;
use core::ffi::{c_char, CStr};

/// Opaque next-handler type for preloaded dispatch.
/// Semantically equivalent to OpHandler, declared separately to avoid
/// recursive type alias. ABI-compatible: all function pointers are pointer-sized.
pub type NextHandler = unsafe extern "C" fn();

#[allow(improper_ctypes)]
extern "C" {
    /// C trampoline entry point for fast interpreter.
    /// Declared here for use by call handlers.
    /// The nh parameter is the preloaded handler for pc+1, computed by the caller.
    pub fn run_trampoline(
        ctx: *mut Context,
        pc: *mut Instruction,
        fp: *mut u64,
        l0: u64,
        l1: u64,
        l2: u64,
        t0: u64,
        t1: u64,
        t2: u64,
        t3: u64,
        nh: NextHandler,
    );
}

/// Handler function pointer type used by the fast interpreter.
/// Matches the C ABI signature in vm_trampoline.c.
/// The nh parameter carries a preloaded handler for next-handler dispatch.
pub type OpHandler = unsafe extern "C" fn(
    ctx: *mut Context,
    pc: *mut Instruction,
    fp: *mut u64,
    l0: u64,
    l1: u64,
    l2: u64,
    t0: u64,
    t1: u64,
    t2: u64,
    t3: u64,
    nh: NextHandler,
);

/// Full set of handler extern declarations, generated from handlers.def.
/// Import this module to access all op_* handler function pointers.
pub mod full_set {
    #[allow(unused_imports)]
    use super::*;

    // Generated extern "C" declarations for all handlers
    include!(concat!(env!("OUT_DIR"), "/fast_interp/fast_extern_decl.rs"));
}

// =============================================================================
// Terminal Instruction (static singleton)
// =============================================================================

/// Static termination instructions - used to exit the interpreter.
/// Two-element array: element [0] is the canonical terminal, element [1] is a
/// sentinel so that `pc_next(term())` reads valid memory. This is required for
/// next-handler preloading: every dispatch path reads `pc_next(np)->handler`
/// to prepare the `nh` parameter for the subsequent handler.
static mut TERM_INST: [Instruction; 2] = [
    Instruction {
        handler: full_set::op_term,
        imm0: 0,
        imm1: 0,
        imm2: 0,
    },
    Instruction {
        handler: full_set::op_term,
        imm0: 0,
        imm1: 0,
        imm2: 0,
    },
];

/// Return pointer to the static terminal instruction.
/// Used by impl_return and trap paths to unwind.
#[inline(always)]
pub fn term() -> *mut Instruction {
    // SAFETY: TERM_INST is read-only after initialization, safe to share
    unsafe { core::ptr::addr_of_mut!(TERM_INST[0]) }
}

/// Set error and return pointer to terminal instruction.
/// Used by handlers that need to trap.
#[inline(always)]
pub fn trap_with(ctx: *mut Context, error: WasmError) -> *mut Instruction {
    unsafe {
        if (*ctx).error.is_none() {
            (*ctx).error = Some(error);
        }
    }
    term()
}

/// C handler trap delegation - accepts null-terminated C string.
/// This allows C handlers to trap with a message without creating WasmError themselves.
///
/// # Safety
/// - `ctx` must be a valid pointer to Context
/// - `message` must be a valid null-terminated C string or NULL
/// - If `message` is non-NULL, it must point to valid UTF-8 (best effort conversion)
///
/// # Returns
/// Always returns a valid pointer to TERM_INST (never NULL).
#[no_mangle]
pub unsafe extern "C" fn fast_c_trap(ctx: *mut Context, message: *const c_char) -> *mut Instruction {
    use alloc::string::ToString;

    // Convert C string to Rust string
    let msg = if message.is_null() {
        "unknown trap"
    } else {
        CStr::from_ptr(message)
            .to_str()
            .unwrap_or("invalid UTF-8 in trap message")
    };

    // Create trap error and set in context
    let error = WasmError::trap(msg.to_string());
    if (*ctx).error.is_none() {
        (*ctx).error = Some(error);
    }

    // Return TERM_INST to cleanly exit tail-call chain
    term()
}

// Common helpers and re-exported types used by all handler modules
pub mod common;

// Handler categories
// Note: Many handlers are now implemented in C (handlers_c/*.c):
//   - arithmetic.c: i32/i64/f32/f64 add, sub, mul, div, rem
//   - bitwise.c: i32/i64 and, or, xor, shl, shr, rotl, rotr
//   - comparison.c: i32/i64/f32/f64 eqz, eq, ne, lt, gt, le, ge
//   - unary.c: i32/i64 clz, ctz, popcnt
//   - float_ops.c: f32/f64 min, max, abs, neg, ceil, floor, trunc, nearest, sqrt, copysign
//   - conversion.c: all wrap, extend, trunc, convert, demote, promote, reinterpret ops
//   - memory.c: all load/store ops
//   - control.c: nop, drop, end, block, loop, select, if, else, unreachable, br, br_if, copy, preserve_copy
pub mod call;
pub mod control;
pub mod global;
pub mod memory;
pub mod ref_ops;
pub mod table;
