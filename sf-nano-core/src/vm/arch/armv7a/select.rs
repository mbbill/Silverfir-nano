//! Pure selector helpers.
//!
//! Every function here takes already-mapped physical registers or simple
//! constants and returns values. No MachineReg mapping, no scratch
//! allocation, no mutable state.

use crate::vm::machine::machine_ir::{
    MachineConvertOp, MachineTrapKind,
};

use super::enc::Cond;
use super::reg::Arm32Reg;

// ── Trap helpers ─────────────────────────────────────────────────────────────

pub(super) fn trap_kind_to_u32(kind: MachineTrapKind) -> u32 {
    match kind {
        MachineTrapKind::Unreachable => 0,
        MachineTrapKind::MemoryOutOfBounds => 1,
        MachineTrapKind::TableOutOfBounds => 2,
        MachineTrapKind::InvalidFunctionReference => 3,
        MachineTrapKind::IndirectCallTypeMismatch => 4,
        MachineTrapKind::IntegerDivideByZero => 5,
        MachineTrapKind::IntegerOverflow => 6,
        MachineTrapKind::StackOverflow => 7,
        MachineTrapKind::HelperFailure => 8,
    }
}

// ── Trap-site encoding ───────────────────────────────────────────────────────

pub(super) const TRAP_SITE_BLOCK_BITS: u32 = 12;
pub(super) const TRAP_SITE_BLOCK_MASK: u32 = (1 << TRAP_SITE_BLOCK_BITS) - 1;
pub(super) const TRAP_SITE_UNKNOWN_BLOCK: u32 = TRAP_SITE_BLOCK_MASK;

#[inline]
pub(super) fn encode_trap_site(func_idx: u32, block_idx: Option<u32>) -> u32 {
    let block = block_idx
        .filter(|&block| block < TRAP_SITE_UNKNOWN_BLOCK)
        .unwrap_or(TRAP_SITE_UNKNOWN_BLOCK);
    (func_idx << TRAP_SITE_BLOCK_BITS) | block
}

// ── Convert op code ──────────────────────────────────────────────────────────

pub(super) fn convert_op_code(op: MachineConvertOp) -> u32 {
    match op {
        MachineConvertOp::I32TruncF32S => 0,
        MachineConvertOp::I32TruncF32U => 1,
        MachineConvertOp::I32TruncF64S => 2,
        MachineConvertOp::I32TruncF64U => 3,
        MachineConvertOp::I64TruncF32S => 4,
        MachineConvertOp::I64TruncF32U => 5,
        MachineConvertOp::I64TruncF64S => 6,
        MachineConvertOp::I64TruncF64U => 7,
        MachineConvertOp::I32TruncSatF32S => 8,
        MachineConvertOp::I32TruncSatF32U => 9,
        MachineConvertOp::I32TruncSatF64S => 10,
        MachineConvertOp::I32TruncSatF64U => 11,
        MachineConvertOp::I64TruncSatF32S => 12,
        MachineConvertOp::I64TruncSatF32U => 13,
        MachineConvertOp::I64TruncSatF64S => 14,
        MachineConvertOp::I64TruncSatF64U => 15,
        _ => u32::MAX,
    }
}

// ── RBIT encoding ────────────────────────────────────────────────────────────

/// RBIT Rd, Rm (reverse bits, ARMv6T2+)
pub(super) fn rbit(dst: Arm32Reg, src: Arm32Reg) -> u32 {
    // RBIT: cond 0110 1111 1111 Rd 1111 0011 Rm
    super::enc::cond_bits(Cond::Al)
        | (0b01101111 << 20)
        | (0b1111 << 16)
        | ((dst.idx()) << 12)
        | (0b11110011 << 4)
        | src.idx()
}
