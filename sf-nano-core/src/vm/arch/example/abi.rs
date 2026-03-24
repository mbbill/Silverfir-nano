//! ABI register plan for the example backend.
//!
//! This file defines:
//! - the mapping from MachineIR registers to physical registers
//! - scratch register arrays (private — only the pool sees them)
//! - C ABI boundary registers (entry args, return values)
//! - dynamic and callee-saved register sets
//!
//! It does NOT emit code and does NOT define role structs.
//! The register layout follows abi.md:
//!   [fixed | gp_local_cache | gp_transient | fp_transient | fp_local_cache] + scratches

use crate::{
    error::WasmError,
    vm::machine::machine_ir::{
        MachineReg, MACHINE_CTX_REG, MACHINE_FIXED_REG_COUNT, MACHINE_FP_REG,
        MACHINE_MEM0_BASE_REG, MACHINE_MEM0_SIZE_REG,
    },
};

use super::reg::{fp, gp, FpReg, GpReg};
use crate::vm::arch::common::scratch_pool::ScratchPool;

// ── C ABI boundary registers ─────────────────────────────────────────────────
// These are platform calling-convention facts, used only at the C↔JIT boundary
// (prologue, epilogue, helper calls). Not part of the JIT register plan.

pub(super) const C_ARG0: GpReg = gp::X0;
pub(super) const C_ARG1: GpReg = gp::X1;
pub(super) const C_ARG2: GpReg = gp::X2;
pub(super) const C_RET0: GpReg = gp::X0;

// ── Scratch pool construction ────────────────────────────────────────────────
// The arrays are private. The only way to get a scratch register is through
// the ScratchPool allocated in Arm64Backend::new().

const GP_SCRATCHES: [GpReg; 2] = [gp::X9, gp::X10];
const FP_SCRATCHES: [FpReg; 3] = [fp::D0, fp::D1, fp::D2];

pub(super) fn new_gp_scratch_pool() -> ScratchPool<GpReg, 2> {
    ScratchPool::new(GP_SCRATCHES)
}

pub(super) fn new_fp_scratch_pool() -> ScratchPool<FpReg, 3> {
    ScratchPool::new(FP_SCRATCHES)
}

// ── Dynamic register arrays (MachineIR allocation) ───────────────────────────

const DYNAMIC_GP: [GpReg; 6] = [
    gp::X23, gp::X24, gp::X25, gp::X26, gp::X27, gp::X28,
];

const DYNAMIC_FP: [FpReg; 4] = [fp::D16, fp::D17, fp::D18, fp::D19];

// ── Callee-saved sets ────────────────────────────────────────────────────────

pub(super) const CALLEE_SAVED_GP: [GpReg; 12] = [
    gp::X19, gp::X20, gp::X21, gp::X22,
    gp::X23, gp::X24, gp::X25, gp::X26, gp::X27, gp::X28,
    gp::X29, gp::X30,
];

pub(super) const CALLEE_SAVED_FP: [FpReg; 8] = [
    fp::D8, fp::D9, fp::D10, fp::D11, fp::D12, fp::D13, fp::D14, fp::D15,
];

pub(super) const STACK_ALIGNMENT_BYTES: u32 = 16;

// ── Capacity queries ─────────────────────────────────────────────────────────

#[inline]
pub(super) const fn max_gp_mapped_regs() -> usize {
    MACHINE_FIXED_REG_COUNT as usize + DYNAMIC_GP.len()
}

#[inline]
pub(super) const fn max_fp_machine_regs() -> usize {
    DYNAMIC_FP.len()
}

#[inline]
pub(super) const fn max_total_machine_regs() -> usize {
    max_gp_mapped_regs() + max_fp_machine_regs()
}

// ── Mapping: MachineReg → physical register ──────────────────────────────────

#[inline]
pub(super) fn map_fixed_gp(reg: MachineReg) -> GpReg {
    match reg {
        MACHINE_CTX_REG => gp::X19,
        MACHINE_FP_REG => gp::X20,
        MACHINE_MEM0_BASE_REG => gp::X21,
        MACHINE_MEM0_SIZE_REG => gp::X22,
        _ => unreachable!("not a fixed machine reg"),
    }
}

#[inline]
pub(super) fn map_gp(reg: MachineReg) -> Result<GpReg, WasmError> {
    if reg.0 < MACHINE_FIXED_REG_COUNT {
        return Ok(map_fixed_gp(reg));
    }
    DYNAMIC_GP
        .get((reg.0 - MACHINE_FIXED_REG_COUNT) as usize)
        .copied()
        .ok_or_else(|| {
            WasmError::invalid(alloc::format!(
                "example backend has no GP mapping for machine reg {}",
                reg.0
            ))
        })
}

#[inline]
pub(super) fn map_fp(index: usize) -> Option<FpReg> {
    DYNAMIC_FP.get(index).copied()
}
