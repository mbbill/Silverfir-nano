//! ARMv7-A physical register mapping and EABI-derived layout.
//!
//! # GP register plan (R0-R15)
//!
//! ```text
//! Reg    EABI             Role                    Count
//! ─────────────────────────────────────────────────────
//! R0-R2  caller-saved     GP transient                3
//! R3     caller-saved     GP transient                1
//! R4     callee-saved     fixed: MEM0_SIZE            1
//! R5-R6  callee-saved     GP local cache              2
//! R7-R8  callee-saved     GP transient                2
//! R9     platform         fixed: CTX                  1
//! R10    callee-saved     fixed: FP                   1
//! R11    callee-saved     fixed: MEM0_BASE            1
//! R12    caller-saved     scratch (IP)                1
//! R13    —                SP (reserved)               1
//! R14    —                LR (reserved)               1
//! R15    —                PC (reserved)               1
//! ─────────────────────────────────────────────────────
//!                         GP transient                6
//!                         GP local cache              2
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
    vm::native::ir::machine::{
        MachineReg, MACHINE_CTX_REG, MACHINE_FIXED_REG_COUNT, MACHINE_FP_REG,
        MACHINE_MEM0_BASE_REG, MACHINE_MEM0_SIZE_REG,
    },
};

use super::{emit::Arm32TextEmitter, enc, reg::Arm32Reg};

pub(super) const SCRATCH0: Arm32Reg = Arm32Reg::R12;
pub(super) const SCRATCH1: Arm32Reg = Arm32Reg::R14;

/// FP scratch registers (caller-saved, not used for values or parameters).
pub(super) const FP_SCRATCH0: u32 = 0; // D0
pub(super) const FP_SCRATCH1: u32 = 1; // D1
pub(super) const FP_SCRATCH2: u32 = 2; // D2

/// Caller-saved FP registers reserved for transient SSA values (TOS lanes).
const FP_TRANSIENT_REGS: [u32; 5] = [3, 4, 5, 6, 7];

/// FP registers reserved for cached locals (D8-D15, callee-saved).
const FP_LOCAL_CACHE_REGS: [u32; 8] = [8, 9, 10, 11, 12, 13, 14, 15];

/// Total FP machine-register capacity: transients first, then cached locals.
pub(super) const FP_MACHINE_REG_COUNT: usize = FP_TRANSIENT_REGS.len() + FP_LOCAL_CACHE_REGS.len();

// Compile-time check: transients + cache + scratch must cover all 16 D-regs.
const _: () = assert!(
    FP_TRANSIENT_REGS.len() + FP_LOCAL_CACHE_REGS.len() + 3 == 16,
    "FP register plan must account for all 16 D-registers (VFPv3-D16)"
);

/// GP registers available to the machine-reg file, ordered: local cache first,
/// then transients.
const DYNAMIC_REGS: [Arm32Reg; 8] = [
    // GP local cache: callee-saved — survive helper calls
    Arm32Reg::R5,
    Arm32Reg::R6,
    // GP transient: dead at call boundaries
    Arm32Reg::R3,
    Arm32Reg::R7,
    Arm32Reg::R8,
    Arm32Reg::R0,
    Arm32Reg::R1,
    Arm32Reg::R2,
];

/// Callee-saved GP registers to save/restore in prologue/epilogue.
/// R4-R11 are callee-saved in EABI; R14(LR) must also be saved since we call helpers.
const CALLEE_SAVED_GP_REGS: [Arm32Reg; 9] = [
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
const CALLEE_SAVED_FP_FIRST: u32 = 8;
const CALLEE_SAVED_FP_COUNT: u32 = 8;

const STACK_ALIGNMENT_BYTES: u32 = 8; // EABI requires 8-byte stack alignment

#[inline]
pub(super) const fn max_gp_mapped_regs() -> usize {
    MACHINE_FIXED_REG_COUNT as usize + DYNAMIC_REGS.len()
}

#[inline]
pub(super) const fn max_fp_machine_regs() -> usize {
    FP_MACHINE_REG_COUNT
}

#[inline]
pub(super) const fn max_total_machine_regs() -> usize {
    max_gp_mapped_regs() + max_fp_machine_regs()
}

#[inline]
pub(super) const fn fp_machine_reg(index: usize) -> Option<u32> {
    if index < FP_TRANSIENT_REGS.len() {
        return Some(FP_TRANSIENT_REGS[index]);
    }
    let local_index = index - FP_TRANSIENT_REGS.len();
    if local_index < FP_LOCAL_CACHE_REGS.len() {
        return Some(FP_LOCAL_CACHE_REGS[local_index]);
    }
    None
}

/// Returns true if this machine register is in the FP register file.
#[inline]
pub(super) const fn is_fp_machine_reg(reg: MachineReg) -> bool {
    reg.0 as usize >= max_gp_mapped_regs()
}

#[inline]
pub(super) fn map_fixed_reg(reg: MachineReg) -> Arm32Reg {
    match reg {
        MACHINE_CTX_REG => Arm32Reg::R9,
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
    DYNAMIC_REGS
        .get((reg.0 - MACHINE_FIXED_REG_COUNT) as usize)
        .copied()
        .ok_or_else(|| {
            WasmError::invalid(alloc::format!(
                "armv7a MachineIR backend has no physical mapping for machine reg {}",
                reg.0
            ))
        })
}

#[inline]
pub(super) fn inv_map_reg(reg: Arm32Reg) -> MachineReg {
    match reg {
        Arm32Reg::R9 => MACHINE_CTX_REG,
        Arm32Reg::R10 => MACHINE_FP_REG,
        Arm32Reg::R11 => MACHINE_MEM0_BASE_REG,
        Arm32Reg::R4 => MACHINE_MEM0_SIZE_REG,
        other => {
            let index = DYNAMIC_REGS
                .iter()
                .position(|candidate| *candidate == other)
                .expect("mapped reg must come from dynamic table");
            MachineReg(MACHINE_FIXED_REG_COUNT + index as u16)
        }
    }
}

/// Build the register mask for PUSH/POP from the callee-saved list.
fn callee_saved_gp_mask() -> u16 {
    let mut mask = 0u16;
    let mut i = 0;
    while i < CALLEE_SAVED_GP_REGS.len() {
        mask |= 1 << CALLEE_SAVED_GP_REGS[i].idx();
        i += 1;
    }
    mask
}

pub(super) fn emit_shared_prologue(text: &mut Arm32TextEmitter) {
    // PUSH {R4-R11, LR}
    text.emit_u32(enc::push(callee_saved_gp_mask()));
    // VPUSH {D8-D15}
    text.emit_u32(enc::vpush_d(CALLEE_SAVED_FP_FIRST, CALLEE_SAVED_FP_COUNT));
}

pub(super) fn emit_shared_epilogue(text: &mut Arm32TextEmitter) {
    // VPOP {D8-D15}
    text.emit_u32(enc::vpop_d(CALLEE_SAVED_FP_FIRST, CALLEE_SAVED_FP_COUNT));
    // POP {R4-R11, PC} — loading PC from the saved LR effectively returns
    let pop_mask =
        callee_saved_gp_mask() & !(1 << Arm32Reg::R14.idx()) | (1 << Arm32Reg::R15.idx());
    text.emit_u32(enc::pop(pop_mask));
}
