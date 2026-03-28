//! ABI register plan for the ARMv7-A backend.
//!
//! This file defines:
//! - the mapping from MachineIR registers to physical registers
//! - scratch register arrays (private — only the pool sees them)
//! - C ABI boundary registers (entry args, return values)
//! - dynamic and callee-saved register sets
//!
//! It does NOT emit code and does NOT define role structs.
//! The register layout follows:
//!   [fixed | gp_local_cache | gp_transient | fp_transient | fp_local_cache] + scratches
//!
//! # GP register plan (R0-R15)
//!
//! ```text
//! Reg    EABI             Role                    Count
//! ─────────────────────────────────────────────────────
//! R0-R2  caller-saved     GP transient                3
//! R3     caller-saved     GP transient                1
//! R4     callee-saved     fixed: MEM0_SIZE            1
//! R5-R7  callee-saved     GP local cache              3
//! R8     callee-saved     fixed: CTX                  1
//! R9     platform         GP local cache              1
//! R10    callee-saved     fixed: FP                   1
//! R11    callee-saved     fixed: MEM0_BASE            1
//! R12    caller-saved     scratch (IP)                1
//! R13    —                SP (reserved)               1
//! R14    —                LR (scratch)                1
//! R15    —                PC (reserved)               1
//! ─────────────────────────────────────────────────────
//!                         GP transient                4
//!                         GP local cache              4
//! ```
//!
//! # FP register plan (D0-D15, VFPv3-D16)
//!
//! ```text
//! Reg    EABI             Role                   Count
//! ─────────────────────────────────────────────────────
//! D0-D2  caller-saved     FP scratch                 3
//! D3-D7  caller-saved     FP transient (TOS)         5
//! D8-D15 callee-saved     FP local cache             8
//! ─────────────────────────────────────────────────────
//!                         FP transient               5
//!                         FP local cache             8
//! ```

use crate::{
    error::WasmError,
    vm::{
        backend::BackendConfig,
        machine::machine_ir::{
            MachineReg, MACHINE_CTX_REG, MACHINE_FIXED_REG_COUNT, MACHINE_FP_REG,
            MACHINE_MEM0_BASE_REG, MACHINE_MEM0_SIZE_REG,
            classify_gp_reg, classify_fp_reg,
        },
    },
};

use super::reg::{Arm32FpReg, Arm32Reg};
use crate::vm::arch::common::scratch_pool::ScratchPool;

// ── C ABI boundary registers ─────────────────────────────────────────────────
// These are platform calling-convention facts, used only at the C↔JIT boundary
// (prologue, epilogue, helper calls). Not part of the JIT register plan.

pub(super) const C_ARG0: Arm32Reg = Arm32Reg::R0;
pub(super) const C_ARG1: Arm32Reg = Arm32Reg::R1;
pub(super) const C_ARG2: Arm32Reg = Arm32Reg::R2;
pub(super) const C_ARG3: Arm32Reg = Arm32Reg::R3;
pub(super) const C_RET0: Arm32Reg = Arm32Reg::R0;
pub(super) const C_RET1: Arm32Reg = Arm32Reg::R1;

// ── Scratch pool construction ────────────────────────────────────────────────
// The arrays are private. The only way to get a scratch register is through
// the ScratchPool allocated in the backend.

const GP_SCRATCHES: [Arm32Reg; 2] = [Arm32Reg::R12, Arm32Reg::R14];
const FP_SCRATCHES: [Arm32FpReg; 3] = [
    Arm32FpReg::new(0), // D0
    Arm32FpReg::new(1), // D1
    Arm32FpReg::new(2), // D2
];

pub(super) fn new_gp_scratch_pool() -> ScratchPool<Arm32Reg, 2> {
    ScratchPool::new(GP_SCRATCHES)
}

pub(super) fn new_fp_scratch_pool() -> ScratchPool<Arm32FpReg, 3> {
    ScratchPool::new(FP_SCRATCHES)
}

// ── Dynamic register arrays (MachineIR allocation) ───────────────────────────
//
// These arrays are the single source of truth for register budgets.
// `config.rs` derives BackendConfig from their lengths.
// Ordering: local-cache first, then transient — must match MachineRegFile layout.

pub(super) const GP_UNIT_BYTES: u8 = 4;

/// GP local cache: preferred for cached locals. Shared lowering
/// synchronizes cached locals through frame slots across helper/call
/// boundaries, so this bank does not rely on every register being
/// callee-saved in the C ABI.
///
/// R9 is never used for fixed MachineIR state. It stays in the dynamic bank
/// only, so the fixed roles remain pinned to unquestionably preserved
/// registers.
pub(super) const GP_LOCAL_CACHE: [Arm32Reg; 4] = [
    Arm32Reg::R5,
    Arm32Reg::R6,
    Arm32Reg::R7,
    Arm32Reg::R9,
];

/// GP transient: dead at call boundaries.
pub(super) const GP_TRANSIENT: [Arm32Reg; 4] = [
    Arm32Reg::R3,
    Arm32Reg::R0,
    Arm32Reg::R1,
    Arm32Reg::R2,
];

/// Caller-saved FP registers reserved for transient SSA values (TOS lanes).
pub(super) const FP_TRANSIENT: [Arm32FpReg; 5] = [
    Arm32FpReg::new(3),
    Arm32FpReg::new(4),
    Arm32FpReg::new(5),
    Arm32FpReg::new(6),
    Arm32FpReg::new(7),
];

/// FP registers reserved for cached locals (D8-D15, callee-saved).
pub(super) const FP_LOCAL_CACHE: [Arm32FpReg; 8] = [
    Arm32FpReg::new(8),
    Arm32FpReg::new(9),
    Arm32FpReg::new(10),
    Arm32FpReg::new(11),
    Arm32FpReg::new(12),
    Arm32FpReg::new(13),
    Arm32FpReg::new(14),
    Arm32FpReg::new(15),
];

// Compile-time check: transients + cache + scratch must cover all 16 D-regs.
const _: () = assert!(
    FP_TRANSIENT.len() + FP_LOCAL_CACHE.len() + 3 == 16,
    "FP register plan must account for all 16 D-registers (VFPv3-D16)"
);

// ── Derived config ───────────────────────────────────────────────────────────

#[inline]
pub(crate) const fn compile_backend_config() -> BackendConfig {
    BackendConfig::new_with_gp_unit_bytes(
        GP_LOCAL_CACHE.len() as u8,
        GP_TRANSIENT.len() as u8,
        FP_LOCAL_CACHE.len() as u8,
        FP_TRANSIENT.len() as u8,
        GP_UNIT_BYTES,
    )
}

// ── Callee-saved sets ────────────────────────────────────────────────────────

/// Callee-saved GP registers to save/restore in prologue/epilogue.
/// R4-R11 are callee-saved in EABI; R14(LR) must also be saved since we call helpers.
pub(super) const CALLEE_SAVED_GP: [Arm32Reg; 9] = [
    Arm32Reg::R4,
    Arm32Reg::R5,
    Arm32Reg::R6,
    Arm32Reg::R7,
    Arm32Reg::R8,
    Arm32Reg::R9,
    Arm32Reg::R10,
    Arm32Reg::R11,
    Arm32Reg::R14, // LR
];

/// EABI callee-saved FP regs (D8-D15).
pub(super) const CALLEE_SAVED_FP_FIRST: u32 = 8;
pub(super) const CALLEE_SAVED_FP_COUNT: u32 = 8;

pub(super) const STACK_ALIGNMENT_BYTES: u32 = 8; // EABI requires 8-byte stack alignment

// ── Capacity queries ─────────────────────────────────────────────────────────

#[inline]
pub(super) const fn max_gp_mapped_regs() -> usize {
    MACHINE_FIXED_REG_COUNT as usize + GP_LOCAL_CACHE.len() + GP_TRANSIENT.len()
}

#[inline]
pub(super) const fn max_fp_machine_regs() -> usize {
    FP_TRANSIENT.len() + FP_LOCAL_CACHE.len()
}

#[inline]
pub(super) const fn max_total_machine_regs() -> usize {
    max_gp_mapped_regs() + max_fp_machine_regs()
}

// ── Mapping: MachineReg → physical register ──────────────────────────────────

#[inline]
pub(super) fn map_fixed_reg(reg: MachineReg) -> Arm32Reg {
    match reg {
        MACHINE_CTX_REG => Arm32Reg::R8,
        MACHINE_FP_REG => Arm32Reg::R10,
        MACHINE_MEM0_BASE_REG => Arm32Reg::R11,
        MACHINE_MEM0_SIZE_REG => Arm32Reg::R4,
        _ => unreachable!("not a fixed machine reg"),
    }
}

#[inline]
pub(super) fn map_reg(reg: MachineReg) -> Result<Arm32Reg, WasmError> {
    if reg.0 < MACHINE_FIXED_REG_COUNT {
        return Ok(map_fixed_reg(reg));
    }
    let config = compile_backend_config();
    match classify_gp_reg(reg, config) {
        Some((index, true)) => GP_LOCAL_CACHE.get(index).copied(),
        Some((index, false)) => GP_TRANSIENT.get(index).copied(),
        None => return Ok(map_fixed_reg(reg)),
    }
    .ok_or_else(|| {
        WasmError::invalid(alloc::format!(
            "armv7a MachineIR backend has no physical mapping for machine reg {}",
            reg.0
        ))
    })
}

#[inline]
pub(super) fn fp_machine_reg(index: usize) -> Option<Arm32FpReg> {
    let config = compile_backend_config();
    let (i, is_cache) = classify_fp_reg(index, config);
    if is_cache {
        FP_LOCAL_CACHE.get(i).copied()
    } else {
        FP_TRANSIENT.get(i).copied()
    }
}

#[inline]
pub(super) fn inv_map_reg(reg: Arm32Reg) -> MachineReg {
    match reg {
        Arm32Reg::R8 => MACHINE_CTX_REG,
        Arm32Reg::R10 => MACHINE_FP_REG,
        Arm32Reg::R11 => MACHINE_MEM0_BASE_REG,
        Arm32Reg::R4 => MACHINE_MEM0_SIZE_REG,
        other => {
            if let Some(i) = GP_LOCAL_CACHE.iter().position(|c| *c == other) {
                return MachineReg(MACHINE_FIXED_REG_COUNT + i as u16);
            }
            let i = GP_TRANSIENT
                .iter()
                .position(|c| *c == other)
                .expect("mapped reg must come from dynamic table");
            MachineReg(MACHINE_FIXED_REG_COUNT + GP_LOCAL_CACHE.len() as u16 + i as u16)
        }
    }
}
