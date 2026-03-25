//! ABI register plan for the ARM64 backend.
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

use super::reg::{Arm64FpReg, Arm64Reg};
use crate::vm::arch::common::scratch_pool::ScratchPool;

// ── C ABI boundary registers ─────────────────────────────────────────────────
// These are platform calling-convention facts, used only at the C↔JIT boundary
// (prologue, epilogue, helper calls). Not part of the JIT register plan.

pub(super) const C_ARG0: Arm64Reg = Arm64Reg::X0;
pub(super) const C_ARG1: Arm64Reg = Arm64Reg::X1;
pub(super) const C_ARG2: Arm64Reg = Arm64Reg::X2;
pub(super) const C_ARG3: Arm64Reg = Arm64Reg::X3;
pub(super) const C_RET0: Arm64Reg = Arm64Reg::X0;
pub(super) const C_RET1: Arm64Reg = Arm64Reg::X1;

// ── Scratch pool construction ────────────────────────────────────────────────
// The arrays are private. The only way to get a scratch register is through
// the ScratchPool allocated in Arm64Backend::new().

const GP_SCRATCHES: [Arm64Reg; 2] = [Arm64Reg::X16, Arm64Reg::X17];
const FP_SCRATCHES: [Arm64FpReg; 3] = [
    Arm64FpReg::new(0), // D0
    Arm64FpReg::new(1), // D1
    Arm64FpReg::new(2), // D2
];

pub(super) fn new_gp_scratch_pool() -> ScratchPool<Arm64Reg, 2> {
    ScratchPool::new(GP_SCRATCHES)
}

pub(super) fn new_fp_scratch_pool() -> ScratchPool<Arm64FpReg, 3> {
    ScratchPool::new(FP_SCRATCHES)
}

// ── Dynamic register arrays (MachineIR allocation) ───────────────────────────

const DYNAMIC_GP: [Arm64Reg; 19] = [
    Arm64Reg::X23, Arm64Reg::X24, Arm64Reg::X25, Arm64Reg::X26,
    Arm64Reg::X27, Arm64Reg::X28,
    Arm64Reg::X9, Arm64Reg::X10, Arm64Reg::X11, Arm64Reg::X12,
    Arm64Reg::X13, Arm64Reg::X14, Arm64Reg::X15,
    Arm64Reg::X3, Arm64Reg::X4, Arm64Reg::X5, Arm64Reg::X6,
    Arm64Reg::X7, Arm64Reg::X8,
];

const DYNAMIC_FP_TRANSIENT: [Arm64FpReg; 10] = [
    Arm64FpReg::new(3), Arm64FpReg::new(4), Arm64FpReg::new(5),
    Arm64FpReg::new(6), Arm64FpReg::new(7), Arm64FpReg::new(16),
    Arm64FpReg::new(17), Arm64FpReg::new(18), Arm64FpReg::new(19),
    Arm64FpReg::new(20),
];

const DYNAMIC_FP_LOCAL_CACHE: [Arm64FpReg; 19] = [
    Arm64FpReg::new(8), Arm64FpReg::new(9), Arm64FpReg::new(10),
    Arm64FpReg::new(11), Arm64FpReg::new(12), Arm64FpReg::new(13),
    Arm64FpReg::new(14), Arm64FpReg::new(15), Arm64FpReg::new(21),
    Arm64FpReg::new(22), Arm64FpReg::new(23), Arm64FpReg::new(24),
    Arm64FpReg::new(25), Arm64FpReg::new(26), Arm64FpReg::new(27),
    Arm64FpReg::new(28), Arm64FpReg::new(29), Arm64FpReg::new(30),
    Arm64FpReg::new(31),
];

pub(super) const FP_MACHINE_REG_COUNT: usize =
    DYNAMIC_FP_TRANSIENT.len() + DYNAMIC_FP_LOCAL_CACHE.len();

// ── Callee-saved sets ────────────────────────────────────────────────────────

pub(super) const CALLEE_SAVED_GP_PAIRS: [(Arm64Reg, Arm64Reg); 6] = [
    (Arm64Reg::X19, Arm64Reg::X20),
    (Arm64Reg::X21, Arm64Reg::X22),
    (Arm64Reg::X23, Arm64Reg::X24),
    (Arm64Reg::X25, Arm64Reg::X26),
    (Arm64Reg::X27, Arm64Reg::X28),
    (Arm64Reg::X29, Arm64Reg::X30),
];

pub(super) const CALLEE_SAVED_FP: [Arm64FpReg; 8] = [
    Arm64FpReg::new(8), Arm64FpReg::new(9), Arm64FpReg::new(10), Arm64FpReg::new(11),
    Arm64FpReg::new(12), Arm64FpReg::new(13), Arm64FpReg::new(14), Arm64FpReg::new(15),
];

pub(super) const STACK_ALIGNMENT_BYTES: u32 = 16;

const _: () = assert!(
    FP_MACHINE_REG_COUNT + 3 == 32,
    "FP register plan must account for all 32 D-registers"
);

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
pub(super) fn map_fixed_reg(reg: MachineReg) -> Arm64Reg {
    match reg {
        MACHINE_CTX_REG => Arm64Reg::X19,
        MACHINE_FP_REG => Arm64Reg::X20,
        MACHINE_MEM0_BASE_REG => Arm64Reg::X21,
        MACHINE_MEM0_SIZE_REG => Arm64Reg::X22,
        _ => unreachable!("not a fixed machine reg"),
    }
}

#[inline]
pub(super) fn map_reg(reg: MachineReg) -> Result<Arm64Reg, WasmError> {
    if reg.0 < MACHINE_FIXED_REG_COUNT {
        return Ok(map_fixed_reg(reg));
    }
    DYNAMIC_GP
        .get((reg.0 - MACHINE_FIXED_REG_COUNT) as usize)
        .copied()
        .ok_or_else(|| {
            WasmError::invalid(alloc::format!(
                "arm64 has no GP mapping for machine reg {}",
                reg.0
            ))
        })
}

#[inline]
pub(super) fn fp_machine_reg(index: usize) -> Option<Arm64FpReg> {
    if index < DYNAMIC_FP_TRANSIENT.len() {
        return Some(DYNAMIC_FP_TRANSIENT[index]);
    }
    let local_index = index - DYNAMIC_FP_TRANSIENT.len();
    if local_index < DYNAMIC_FP_LOCAL_CACHE.len() {
        return Some(DYNAMIC_FP_LOCAL_CACHE[local_index]);
    }
    None
}

