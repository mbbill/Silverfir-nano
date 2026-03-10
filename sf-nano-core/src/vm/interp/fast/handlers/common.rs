//! Common helper functions used by all handler modules.
//!
//! These helpers provide safe abstractions for:
//! - Context access
//! - PC navigation (fallthrough, alt, immediates)
//! - Memory access
//! - Error handling

// Re-export commonly needed types for handler modules
pub use crate::{
    module::entities::ElementInit,
    module::type_context::TypeContext,
    utils::limits::Limitable,
    value_type::RefType,
    vm::{
        entities::{DataInst, ElementInst, FunctionInst, GlobalInst, MemInst, ModuleInst, TableInst},
        interp::fast::{context::Context, instruction::Instruction},
        store::Store,
        value::{RefHandle as VmRefHandle, Value as ExtValue},
    },
};
pub use core::ptr::NonNull;

// Re-export raw_value helpers for converting between ExtValue and u64
pub use crate::vm::interp::raw_value::{raw_to_value, value_to_raw};


// =============================================================================
// Context Access
// =============================================================================

#[inline(always)]
pub fn ctx_mut<'a>(ctx: *mut Context) -> &'a mut Context {
    unsafe { &mut *ctx }
}

#[inline(always)]
pub fn ctx_store<'a>(ctx: *mut Context) -> &'a Store {
    ctx_mut(ctx).store()
}

#[inline(always)]
pub fn ctx_store_mut<'a>(ctx: *mut Context) -> &'a mut Store {
    ctx_mut(ctx).store_mut()
}

#[inline(always)]
pub fn ctx_module<'a>(ctx: *mut Context) -> &'a ModuleInst {
    // SAFETY: execution guarantees current_module is valid during op handling
    ctx_mut(ctx).current_module().unwrap()
}

// =============================================================================
// Frame Slot Operations (for locals/params access)
// =============================================================================

/// Read a value from a frame slot at the given offset from fp (frame pointer).
/// `offset` is the absolute slot index (0 = first param, params+locals = operand base).
#[inline(always)]
pub fn frame_read(fp_pp: *mut *mut u64, offset: usize) -> u64 {
    unsafe {
        let fp = *fp_pp;
        *fp.add(offset)
    }
}

/// Write a value to a frame slot at the given offset from fp (frame pointer).
#[inline(always)]
pub fn frame_write(fp_pp: *mut *mut u64, offset: usize, value: u64) {
    unsafe {
        let fp = *fp_pp;
        *fp.add(offset) = value;
    }
}

/// Spill l0 register to fp[0] before frame changes (calls).
#[inline(always)]
pub fn l0_spill(fp_pp: *mut *mut u64, p_l0: *mut u64) {
    unsafe { *(*fp_pp) = *p_l0; }
}

/// Fill l0 register from fp[0] after frame restoration (returns, external calls).
#[inline(always)]
pub fn l0_fill(fp_pp: *mut *mut u64, p_l0: *mut u64) {
    unsafe { *p_l0 = *(*fp_pp); }
}

/// Spill l1 register to fp[1] before frame changes (calls).
#[inline(always)]
pub fn l1_spill(fp_pp: *mut *mut u64, p_l1: *mut u64) {
    unsafe { *((*fp_pp).add(1)) = *p_l1; }
}

/// Fill l1 register from fp[1] after frame restoration (returns, external calls).
#[inline(always)]
pub fn l1_fill(fp_pp: *mut *mut u64, p_l1: *mut u64) {
    unsafe { *p_l1 = *((*fp_pp).add(1)); }
}

/// Spill l2 register to fp[2] before frame changes (calls).
#[inline(always)]
pub fn l2_spill(fp_pp: *mut *mut u64, p_l2: *mut u64) {
    unsafe { *((*fp_pp).add(2)) = *p_l2; }
}

/// Fill l2 register from fp[2] after frame restoration (returns, external calls).
#[inline(always)]
pub fn l2_fill(fp_pp: *mut *mut u64, p_l2: *mut u64) {
    unsafe { *p_l2 = *((*fp_pp).add(2)); }
}

// =============================================================================
// PC Navigation
// =============================================================================

#[inline(always)]
pub fn pc_fallthrough(pc: *mut Instruction) -> *mut Instruction {
    unsafe { pc.add(1) }
}

/// Read branch target / error path pointer from imm0.
/// This is used for branches, conditionals, and error paths.
#[inline(always)]
pub fn pc_alt(pc: *mut Instruction) -> *mut Instruction {
    unsafe { (*pc).imm0 as *mut Instruction }
}

#[inline(always)]
pub fn pc_imm0(pc: *mut Instruction) -> u64 {
    unsafe { (*pc).imm0 }
}

#[inline(always)]
pub fn pc_imm1(pc: *mut Instruction) -> u64 {
    unsafe { (*pc).imm1 }
}

#[inline(always)]
pub fn pc_imm2(pc: *mut Instruction) -> u64 {
    unsafe { (*pc).imm2 }
}


// =============================================================================
// Pointer Helpers
// =============================================================================

#[inline(always)]
pub fn ptr_ref<'a, T>(ptr: *const T) -> &'a T {
    unsafe { &*ptr }
}


// =============================================================================
// Memory Helpers
// =============================================================================

#[inline(always)]
pub fn write_mem0(ctx: *mut Context, base: *mut u8, len: u64) {
    unsafe {
        (*ctx).hot.mem0_base = base;
        (*ctx).hot.mem0_size = len;
    }
}

#[inline(always)]
pub fn heap_info(ctx: *mut Context) -> (*mut u8, usize) {
    unsafe {
        let base = (*ctx).hot.mem0_base;
        let size = (*ctx).hot.mem0_size as usize;
        if base.is_null() || size == 0 {
            (core::ptr::null_mut(), 0)
        } else {
            (base, size)
        }
    }
}

// =============================================================================
// Branch Fixup (No-Stack Model)
// =============================================================================

/// Copy branch payload values between explicit fp-relative spans.
#[inline(always)]
pub fn branch_fixup_frame(
    fp_pp: *mut *mut u64,
    src_slot: usize,
    dst_slot: usize,
    count: usize,
) {
    if count == 0 {
        return;
    }
    unsafe {
        let fp = *fp_pp;
        for i in 0..count {
            let val = *fp.add(src_slot + i);
            *fp.add(dst_slot + i) = val;
        }
    }
}

// =============================================================================
// Module Memory Refresh
// =============================================================================

/// Refresh memory pointers for a given module (used after calls/returns).
/// In sf-nano, Store directly owns MemInst — no Rc<RefCell<>> indirection.
#[inline(always)]
pub fn refresh_mem0_for_module(
    ctx: *mut Context,
    store: &Store,
    _module: &ModuleInst,
) {
    // sf-nano: single-module store, memory accessed directly
    if store.module().memories.is_empty() {
        write_mem0(ctx, core::ptr::null_mut(), 0);
    } else {
        let mem = &store.module().memories[0];
        let ptr = mem.data.as_ptr() as *mut u8;
        let len = mem.data.len() as u64;
        write_mem0(ctx, ptr, len);
    }
}
