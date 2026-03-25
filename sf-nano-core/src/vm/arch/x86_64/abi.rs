//! ABI register plan for the x86_64 backend.
//!
//! This file defines:
//! - the mapping from MachineIR registers to physical registers
//! - scratch register arrays (private — only the pool sees them)
//! - C ABI boundary registers (entry args, return values)
//! - dynamic and callee-saved register sets
//!
//! It does NOT emit code and does NOT define role structs.

use crate::{
    error::WasmError,
    vm::machine::machine_ir::{
        MachineReg, MACHINE_CTX_REG, MACHINE_FIXED_REG_COUNT, MACHINE_FP_REG,
        MACHINE_MEM0_BASE_REG, MACHINE_MEM0_SIZE_REG,
    },
};

use super::reg::X86Reg;
use crate::vm::arch::common::scratch_pool::ScratchPool;

// ── C ABI boundary registers ─────────────────────────────────────────────────
// These are platform calling-convention facts, used only at the C↔JIT boundary
// (prologue, epilogue, helper calls). Not part of the JIT register plan.

pub(super) const C_ARG0: X86Reg = X86Reg::RDI;
pub(super) const C_ARG1: X86Reg = X86Reg::RSI;
pub(super) const C_ARG2: X86Reg = X86Reg::RDX;
pub(super) const C_RET0: X86Reg = X86Reg::RAX;

// ── Scratch pool construction ────────────────────────────────────────────────
// The arrays are private. The only way to get a scratch register is through
// the ScratchPool allocated in X86_64Backend::new().

const GP_SCRATCHES: [X86Reg; 2] = [X86Reg::RAX, X86Reg::R11];
const FP_SCRATCHES: [u32; 3] = [0, 1, 2]; // XMM0, XMM1, XMM2

pub(super) fn new_gp_scratch_pool() -> ScratchPool<X86Reg, 2> {
    ScratchPool::new(GP_SCRATCHES)
}

pub(super) fn new_fp_scratch_pool() -> ScratchPool<u32, 3> {
    ScratchPool::new(FP_SCRATCHES)
}

// ── Dynamic register arrays (MachineIR allocation) ───────────────────────────

/// GP registers available to the machine-reg file, ordered: local cache first,
/// then transients.
const DYNAMIC_GP: [X86Reg; 9] = [
    // GP local cache: callee-saved — free at helper calls
    X86Reg::R14,
    X86Reg::R15,
    // GP transient: caller-saved — dead at call boundaries
    X86Reg::RCX,
    X86Reg::RDX,
    X86Reg::RSI,
    X86Reg::RDI,
    X86Reg::R8,
    X86Reg::R9,
    X86Reg::R10,
];

/// FP transient XMM registers: XMM0-XMM5.
const DYNAMIC_FP_TRANSIENT: [u32; 6] = [0, 1, 2, 3, 4, 5];

/// FP cached local XMM registers: XMM6-XMM15.
/// All caller-saved on System V — must be spilled around helper calls.
const DYNAMIC_FP_LOCAL_CACHE: [u32; 10] = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

pub(super) const FP_MACHINE_REG_COUNT: usize =
    DYNAMIC_FP_TRANSIENT.len() + DYNAMIC_FP_LOCAL_CACHE.len();

// Compile-time check: transients + cache = 16 XMM regs.
const _: () = assert!(
    DYNAMIC_FP_TRANSIENT.len() + DYNAMIC_FP_LOCAL_CACHE.len() == 16,
    "FP register plan must account for all 16 XMM registers"
);

// ── Callee-saved sets ────────────────────────────────────────────────────────

/// Callee-saved GP registers we must push/pop in prologue/epilogue.
/// These are all the callee-saved regs we use: fixed + cached locals.
pub(super) const CALLEE_SAVED_GP: [X86Reg; 6] = [
    X86Reg::RBX, // MACHINE_CTX_REG
    X86Reg::RBP, // MACHINE_FP_REG
    X86Reg::R12, // MACHINE_MEM0_BASE_REG
    X86Reg::R13, // MACHINE_MEM0_SIZE_REG
    X86Reg::R14, // GP local cache
    X86Reg::R15, // GP local cache
];

// No callee-saved FP regs on System V AMD64.

pub(super) const STACK_ALIGNMENT_BYTES: u32 = 16;

// ── Capacity queries ─────────────────────────────────────────────────────────

#[inline]
pub(super) const fn max_gp_mapped_regs() -> usize {
    MACHINE_FIXED_REG_COUNT as usize + DYNAMIC_GP.len()
}

#[inline]
pub(super) const fn max_fp_machine_regs() -> usize {
    FP_MACHINE_REG_COUNT
}

#[inline]
pub(super) const fn max_total_machine_regs() -> usize {
    max_gp_mapped_regs() + max_fp_machine_regs()
}

// ── Mapping: MachineReg → physical register ──────────────────────────────────

#[inline]
pub(super) fn map_fixed_reg(reg: MachineReg) -> X86Reg {
    match reg {
        MACHINE_CTX_REG => X86Reg::RBX,
        MACHINE_FP_REG => X86Reg::RBP,
        MACHINE_MEM0_BASE_REG => X86Reg::R12,
        MACHINE_MEM0_SIZE_REG => X86Reg::R13,
        _ => unreachable!("not a fixed machine reg"),
    }
}

#[inline]
pub(super) fn map_reg(reg: MachineReg) -> Result<X86Reg, WasmError> {
    if reg.0 < MACHINE_FIXED_REG_COUNT {
        return Ok(map_fixed_reg(reg));
    }
    DYNAMIC_GP
        .get((reg.0 - MACHINE_FIXED_REG_COUNT) as usize)
        .copied()
        .ok_or_else(|| {
            WasmError::invalid(alloc::format!(
                "x86_64 has no GP mapping for machine reg {}",
                reg.0
            ))
        })
}

#[inline]
pub(super) fn fp_machine_reg(index: usize) -> Option<u32> {
    if index < DYNAMIC_FP_TRANSIENT.len() {
        return Some(DYNAMIC_FP_TRANSIENT[index]);
    }
    let local_index = index - DYNAMIC_FP_TRANSIENT.len();
    if local_index < DYNAMIC_FP_LOCAL_CACHE.len() {
        return Some(DYNAMIC_FP_LOCAL_CACHE[local_index]);
    }
    None
}
