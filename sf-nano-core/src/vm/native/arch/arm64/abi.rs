//! ARM64 physical register mapping and ABI-derived layout.
//!
//! This module is the source of truth for facts imposed by the target ABI or
//! by our chosen physical register mapping:
//! - which hardware registers back fixed machine roles
//! - which GP/FP registers are available to the machine-reg file
//! - which FP registers are scratch vs machine-resident
//! - which physical registers must be preserved by the shared prologue/epilogue
//! - stack-frame layout derived from those saved-register sets
//!
//! It intentionally does *not* choose how much of that capacity the compiler
//! spends by default. That policy lives in `config.rs` as a backend budget
//! preset and must fit within the capacities described here.

use crate::{
    error::WasmError,
    vm::native::ir::machine::{
        MachineReg, MACHINE_CTX_REG, MACHINE_FIXED_REG_COUNT, MACHINE_FP_REG,
        MACHINE_MEM0_BASE_REG, MACHINE_MEM0_SIZE_REG,
    },
};

use super::{
    emit::Arm64TextEmitter,
    enc,
    reg::Arm64Reg,
};

pub(super) const SCRATCH0: Arm64Reg = Arm64Reg::X16;
pub(super) const SCRATCH1: Arm64Reg = Arm64Reg::X17;

/// FP scratch registers (caller-saved, not used for parameter passing in our ABI).
pub(super) const FP_SCRATCH0: u32 = 0; // D0/S0
pub(super) const FP_SCRATCH1: u32 = 1; // D1/S1
pub(super) const FP_SCRATCH2: u32 = 2; // D2/S2

/// All FP machine regs: transients first, then local cache.
///
/// We use D3-D10 for the machine FP bank. D8-D10 are callee-saved under the
/// platform ABI, so the shared prologue/epilogue preserves them explicitly.
pub(super) const FP_MACHINE_REGS: [u32; 8] = [3, 4, 5, 6, 7, 8, 9, 10];

const DYNAMIC_REGS: [Arm64Reg; 13] = [
    Arm64Reg::X9,
    Arm64Reg::X10,
    Arm64Reg::X11,
    Arm64Reg::X12,
    Arm64Reg::X13,
    Arm64Reg::X14,
    Arm64Reg::X15,
    Arm64Reg::X23,
    Arm64Reg::X24,
    Arm64Reg::X25,
    Arm64Reg::X26,
    Arm64Reg::X27,
    Arm64Reg::X28,
];

const CALLEE_SAVED_GP_PAIRS: [(Arm64Reg, Arm64Reg); 6] = [
    (Arm64Reg::X19, Arm64Reg::X20),
    (Arm64Reg::X21, Arm64Reg::X22),
    (Arm64Reg::X23, Arm64Reg::X24),
    (Arm64Reg::X25, Arm64Reg::X26),
    (Arm64Reg::X27, Arm64Reg::X28),
    (Arm64Reg::X29, Arm64Reg::X30),
];

const CALLEE_SAVED_FP_REGS: [u32; 3] = [8, 9, 10];
const STACK_SLOT_BYTES: u32 = core::mem::size_of::<u64>() as u32;
const STACK_ALIGNMENT_BYTES: u32 = 16;
const CALLEE_SAVED_GP_FRAME_SIZE: u32 =
    CALLEE_SAVED_GP_PAIRS.len() as u32 * (2 * STACK_SLOT_BYTES);
const CALLEE_SAVED_FP_FRAME_OFFSET: u32 = CALLEE_SAVED_GP_FRAME_SIZE;
const CALLEE_SAVED_FP_FRAME_SIZE: u32 = CALLEE_SAVED_FP_REGS.len() as u32 * STACK_SLOT_BYTES;
const CALLEE_SAVED_FRAME_SIZE: u32 = align_up_u32(
    CALLEE_SAVED_FP_FRAME_OFFSET + CALLEE_SAVED_FP_FRAME_SIZE,
    STACK_ALIGNMENT_BYTES,
);

#[inline]
pub(super) const fn max_gp_mapped_regs() -> usize {
    MACHINE_FIXED_REG_COUNT as usize + DYNAMIC_REGS.len()
}

#[inline]
pub(super) const fn max_fp_machine_regs() -> usize {
    FP_MACHINE_REGS.len()
}

#[inline]
pub(super) const fn max_total_machine_regs() -> usize {
    max_gp_mapped_regs() + max_fp_machine_regs()
}

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
    DYNAMIC_REGS
        .get((reg.0 - MACHINE_FIXED_REG_COUNT) as usize)
        .copied()
        .ok_or_else(|| {
            WasmError::invalid(alloc::format!(
                "arm64 MachineIR backend has no physical mapping for machine reg {}",
                reg.0
            ))
        })
}

#[inline]
pub(super) fn inv_map_reg(reg: Arm64Reg) -> MachineReg {
    match reg {
        Arm64Reg::X19 => MACHINE_CTX_REG,
        Arm64Reg::X20 => MACHINE_FP_REG,
        Arm64Reg::X21 => MACHINE_MEM0_BASE_REG,
        Arm64Reg::X22 => MACHINE_MEM0_SIZE_REG,
        other => {
            let index = DYNAMIC_REGS
                .iter()
                .position(|candidate| *candidate == other)
                .expect("mapped reg must come from dynamic table");
            MachineReg(MACHINE_FIXED_REG_COUNT + index as u16)
        }
    }
}

pub(super) fn emit_shared_prologue(text: &mut Arm64TextEmitter) {
    text.emit_u32(enc::sub_imm_64(
        Arm64Reg::SP,
        Arm64Reg::SP,
        CALLEE_SAVED_FRAME_SIZE,
    ));
    for (index, (lhs, rhs)) in CALLEE_SAVED_GP_PAIRS.iter().copied().enumerate() {
        text.emit_u32(enc::stp_64(
            lhs,
            rhs,
            Arm64Reg::SP,
            stack_pair_imm((index as u32) * 2 * STACK_SLOT_BYTES),
        ));
    }
    for (index, reg) in CALLEE_SAVED_FP_REGS.iter().copied().enumerate() {
        text.emit_u32(enc::str_d(
            reg,
            Arm64Reg::SP,
            stack_u64_slot(CALLEE_SAVED_FP_FRAME_OFFSET + index as u32 * STACK_SLOT_BYTES),
        ));
    }
}

pub(super) fn emit_shared_epilogue(text: &mut Arm64TextEmitter) {
    for (index, reg) in CALLEE_SAVED_FP_REGS.iter().copied().enumerate() {
        text.emit_u32(enc::ldr_d(
            reg,
            Arm64Reg::SP,
            stack_u64_slot(CALLEE_SAVED_FP_FRAME_OFFSET + index as u32 * STACK_SLOT_BYTES),
        ));
    }
    for (index, (lhs, rhs)) in CALLEE_SAVED_GP_PAIRS.iter().copied().enumerate() {
        text.emit_u32(enc::ldp_64(
            lhs,
            rhs,
            Arm64Reg::SP,
            stack_pair_imm((index as u32) * 2 * STACK_SLOT_BYTES),
        ));
    }
    text.emit_u32(enc::add_imm_64(
        Arm64Reg::SP,
        Arm64Reg::SP,
        CALLEE_SAVED_FRAME_SIZE,
    ));
    text.emit_u32(enc::ret());
}

const fn align_up_u32(value: u32, align: u32) -> u32 {
    value.div_ceil(align) * align
}

const fn stack_u64_slot(offset_bytes: u32) -> u32 {
    offset_bytes / STACK_SLOT_BYTES
}

const fn stack_pair_imm(offset_bytes: u32) -> i32 {
    (offset_bytes / STACK_SLOT_BYTES) as i32
}
