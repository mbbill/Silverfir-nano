//! ARM64 codegen fusion patterns: immediate selection and zero-store pair
//! fusion.
//!
//! These are pure functions — they inspect MachineIR and return optional
//! instruction encodings. No compiler state is accessed.

use crate::vm::machine::machine_ir::{
    MachineBlock, MachineInstKind, MachineIntBinaryOp, MachineIntWidth, MachineMemWidth,
    MachineReg, MachineValue,
};

use super::{
    enc::{self, Cond},
    reg::Arm64Reg,
};

// ── Immediate instruction selection ──────────────────────────────────────────

/// Try to select a reg+imm immediate form for a binary op.
/// Takes already-mapped physical registers — the caller handles MachineValue
/// matching and register mapping.
pub(super) fn int_binary_imm_inst(
    width: MachineIntWidth,
    op: MachineIntBinaryOp,
    dst: Arm64Reg,
    lhs: Arm64Reg,
    rhs: u64,
) -> Option<u32> {
    match (width, op) {
        (MachineIntWidth::I32, MachineIntBinaryOp::Add) => add_sub_imm_inst_32(true, dst, lhs, rhs),
        (MachineIntWidth::I64, MachineIntBinaryOp::Add) => add_sub_imm_inst_64(true, dst, lhs, rhs),
        (MachineIntWidth::I32, MachineIntBinaryOp::Sub) => {
            add_sub_imm_inst_32(false, dst, lhs, rhs)
        }
        (MachineIntWidth::I64, MachineIntBinaryOp::Sub) => {
            add_sub_imm_inst_64(false, dst, lhs, rhs)
        }
        (MachineIntWidth::I32, MachineIntBinaryOp::Mul) => mul_imm_inst_32(dst, lhs, rhs as u32),
        (MachineIntWidth::I64, MachineIntBinaryOp::Mul) => mul_imm_inst_64(dst, lhs, rhs),
        (
            MachineIntWidth::I32,
            op @ (MachineIntBinaryOp::And | MachineIntBinaryOp::Or | MachineIntBinaryOp::Xor),
        ) => logical_imm_inst_32(op, dst, lhs, rhs as u32),
        (
            MachineIntWidth::I64,
            op @ (MachineIntBinaryOp::And | MachineIntBinaryOp::Or | MachineIntBinaryOp::Xor),
        ) => logical_imm_inst_64(op, dst, lhs, rhs),
        (MachineIntWidth::I32, MachineIntBinaryOp::Shl) => {
            Some(enc::lsl_imm_32(dst, lhs, (rhs as u32) & 31))
        }
        (MachineIntWidth::I64, MachineIntBinaryOp::Shl) => {
            Some(enc::lsl_imm_64(dst, lhs, (rhs as u32) & 63))
        }
        (MachineIntWidth::I32, MachineIntBinaryOp::ShrU) => {
            Some(enc::lsr_imm_32(dst, lhs, (rhs as u32) & 31))
        }
        (MachineIntWidth::I64, MachineIntBinaryOp::ShrU) => {
            Some(enc::lsr_imm_64(dst, lhs, (rhs as u32) & 63))
        }
        (MachineIntWidth::I32, MachineIntBinaryOp::ShrS) => {
            Some(enc::asr_imm_32(dst, lhs, (rhs as u32) & 31))
        }
        (MachineIntWidth::I64, MachineIntBinaryOp::ShrS) => {
            Some(enc::asr_imm_64(dst, lhs, (rhs as u32) & 63))
        }
        _ => None,
    }
}

fn add_sub_imm_inst_32(is_add: bool, dst: Arm64Reg, lhs: Arm64Reg, imm: u64) -> Option<u32> {
    let imm = imm as u32;
    if let Some(imm12) = try_imm12_u32(imm) {
        return Some(if is_add {
            enc::add_imm_32(dst, lhs, imm12)
        } else {
            enc::sub_imm_32(dst, lhs, imm12)
        });
    }
    let neg = imm.wrapping_neg();
    try_imm12_u32(neg).map(|imm12| {
        if is_add {
            enc::sub_imm_32(dst, lhs, imm12)
        } else {
            enc::add_imm_32(dst, lhs, imm12)
        }
    })
}

fn add_sub_imm_inst_64(is_add: bool, dst: Arm64Reg, lhs: Arm64Reg, imm: u64) -> Option<u32> {
    if let Some(imm12) = try_imm12_u64(imm) {
        return Some(if is_add {
            enc::add_imm_64(dst, lhs, imm12)
        } else {
            enc::sub_imm_64(dst, lhs, imm12)
        });
    }
    let neg = imm.wrapping_neg();
    try_imm12_u64(neg).map(|imm12| {
        if is_add {
            enc::sub_imm_64(dst, lhs, imm12)
        } else {
            enc::add_imm_64(dst, lhs, imm12)
        }
    })
}

fn mul_imm_inst_32(dst: Arm64Reg, lhs: Arm64Reg, imm: u32) -> Option<u32> {
    if imm == 0 {
        return Some(enc::movz_32(dst, 0, 0));
    }
    if imm == 1 {
        return Some(enc::mov_reg_32(dst, lhs));
    }
    imm.is_power_of_two()
        .then(|| enc::lsl_imm_32(dst, lhs, imm.trailing_zeros()))
}

fn mul_imm_inst_64(dst: Arm64Reg, lhs: Arm64Reg, imm: u64) -> Option<u32> {
    if imm == 0 {
        return Some(enc::movz_64(dst, 0, 0));
    }
    if imm == 1 {
        return Some(enc::mov_reg_64(dst, lhs));
    }
    imm.is_power_of_two()
        .then(|| enc::lsl_imm_64(dst, lhs, imm.trailing_zeros()))
}

fn logical_imm_inst_32(
    op: MachineIntBinaryOp,
    dst: Arm64Reg,
    lhs: Arm64Reg,
    imm: u32,
) -> Option<u32> {
    match op {
        MachineIntBinaryOp::And => {
            if imm == 0 {
                Some(enc::movz_32(dst, 0, 0))
            } else if imm == u32::MAX {
                Some(enc::mov_reg_32(dst, lhs))
            } else {
                enc::and_imm_32(dst, lhs, imm)
            }
        }
        MachineIntBinaryOp::Or => {
            if imm == 0 {
                Some(enc::mov_reg_32(dst, lhs))
            } else {
                enc::orr_imm_32(dst, lhs, imm)
            }
        }
        MachineIntBinaryOp::Xor => {
            if imm == 0 {
                Some(enc::mov_reg_32(dst, lhs))
            } else if imm == u32::MAX {
                Some(enc::mvn_32(dst, lhs))
            } else {
                enc::eor_imm_32(dst, lhs, imm)
            }
        }
        _ => None,
    }
}

fn logical_imm_inst_64(
    op: MachineIntBinaryOp,
    dst: Arm64Reg,
    lhs: Arm64Reg,
    imm: u64,
) -> Option<u32> {
    match op {
        MachineIntBinaryOp::And => {
            if imm == 0 {
                Some(enc::movz_64(dst, 0, 0))
            } else if imm == u64::MAX {
                Some(enc::mov_reg_64(dst, lhs))
            } else {
                enc::and_imm_64(dst, lhs, imm)
            }
        }
        MachineIntBinaryOp::Or => {
            if imm == 0 {
                Some(enc::mov_reg_64(dst, lhs))
            } else {
                enc::orr_imm_64(dst, lhs, imm)
            }
        }
        MachineIntBinaryOp::Xor => {
            if imm == 0 {
                Some(enc::mov_reg_64(dst, lhs))
            } else if imm == u64::MAX {
                Some(enc::mvn_64(dst, lhs))
            } else {
                enc::eor_imm_64(dst, lhs, imm)
            }
        }
        _ => None,
    }
}

// ── Compare immediate selection ──────────────────────────────────────────────

pub(super) fn cmp_imm_inst(width: MachineIntWidth, lhs: Arm64Reg, rhs: u64) -> Option<u32> {
    match width {
        MachineIntWidth::I32 => try_imm12_u32(rhs as u32).map(|imm12| enc::cmp_imm_32(lhs, imm12)),
        MachineIntWidth::I64 => try_imm12_u64(rhs).map(|imm12| enc::cmp_imm_64(lhs, imm12)),
    }
}

pub(super) fn try_imm12_u32(value: u32) -> Option<u32> {
    (value < 4096).then_some(value)
}

pub(super) fn try_imm12_u64(value: u64) -> Option<u32> {
    (value < 4096).then_some(value as u32)
}

// ── Zero-store pair fusion ───────────────────────────────────────────────────

/// Detect consecutive `Store { src: Imm64(0), width: U64 }` pairs with the same
/// base register and adjacent 8-byte-aligned offsets → fuse into STP XZR, XZR.
pub(super) fn zero_store_pair_fusion(
    block: &MachineBlock,
    index: usize,
) -> Option<(MachineReg, i32)> {
    let a = block.ops.get(index)?;
    let b = block.ops.get(index + 1)?;
    let (
        MachineInstKind::Store {
            addr: addr_a,
            width: MachineMemWidth::U64,
            src: MachineValue::Imm64(0),
            ..
        },
        MachineInstKind::Store {
            addr: addr_b,
            width: MachineMemWidth::U64,
            src: MachineValue::Imm64(0),
            ..
        },
    ) = (&a.kind, &b.kind)
    else {
        return None;
    };
    if addr_a.base != addr_b.base {
        return None;
    }
    if addr_b.offset != addr_a.offset + 8 {
        return None;
    }
    let off_a = addr_a.offset as i64;
    if off_a < 0 || (off_a % 8) != 0 {
        return None;
    }
    let imm7 = (off_a / 8) as i32;
    if !(-64..=63).contains(&imm7) {
        return None;
    }
    Some((addr_a.base, imm7))
}

// ── Condition code mapping ───────────────────────────────────────────────────

use crate::vm::machine::machine_ir::{MachineCompareKind, MachineSign};

pub(super) fn map_int_cond(kind: MachineCompareKind, sign: MachineSign) -> Cond {
    match (kind, sign) {
        (MachineCompareKind::Eq, _) => Cond::Eq,
        (MachineCompareKind::Ne, _) => Cond::Ne,
        (MachineCompareKind::Lt, MachineSign::Signed) => Cond::Lt,
        (MachineCompareKind::Lt, MachineSign::Unsigned) => Cond::Lo,
        (MachineCompareKind::Gt, MachineSign::Signed) => Cond::Gt,
        (MachineCompareKind::Gt, MachineSign::Unsigned) => Cond::Hi,
        (MachineCompareKind::Le, MachineSign::Signed) => Cond::Le,
        (MachineCompareKind::Le, MachineSign::Unsigned) => Cond::Ls,
        (MachineCompareKind::Ge, MachineSign::Signed) => Cond::Ge,
        (MachineCompareKind::Ge, MachineSign::Unsigned) => Cond::Hs,
    }
}

pub(super) fn map_float_cond(kind: MachineCompareKind) -> Cond {
    match kind {
        MachineCompareKind::Eq => Cond::Eq,
        MachineCompareKind::Ne => Cond::Ne,
        MachineCompareKind::Lt => Cond::Mi,
        MachineCompareKind::Gt => Cond::Gt,
        MachineCompareKind::Le => Cond::Ls,
        MachineCompareKind::Ge => Cond::Ge,
    }
}
