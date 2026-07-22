//! ARM64 codegen fusion patterns: immediate selection and zero-store pair
//! fusion.
//!
//! These are pure functions — they inspect MachineIR and return optional
//! instruction encodings. No compiler state is accessed.

use crate::vm::machine::machine_ir::{
    MachineCompareKind, MachineInst, MachineInstKind, MachineIntBinaryOp, MachineIntWidth,
    MachineMemWidth, MachineReg, MachineSign, MachineStorageType, MachineValue,
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
        (MachineIntWidth::I32, MachineIntBinaryOp::Rotr) => {
            Some(enc::ror_imm_32(dst, lhs, (rhs as u32) & 31))
        }
        (MachineIntWidth::I64, MachineIntBinaryOp::Rotr) => {
            Some(enc::ror_imm_64(dst, lhs, (rhs as u32) & 63))
        }
        (MachineIntWidth::I32, MachineIntBinaryOp::Rotl) => {
            let shift = 32_u32.wrapping_sub(rhs as u32) & 31;
            Some(enc::ror_imm_32(dst, lhs, shift))
        }
        (MachineIntWidth::I64, MachineIntBinaryOp::Rotl) => {
            let shift = 64_u32.wrapping_sub(rhs as u32) & 63;
            Some(enc::ror_imm_64(dst, lhs, shift))
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
    a: &crate::vm::machine::machine_ir::MachineInst,
    b: &crate::vm::machine::machine_ir::MachineInst,
) -> Option<(MachineReg, i32)> {
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

// ── Bitwise NOT + AND fusion ─────────────────────────────────────────────

/// `(~not_rhs) & lhs`, represented by an XOR-all-ones followed by AND.
/// ARM64 lowers this to one flag-setting BICS instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BitClear {
    pub width: MachineIntWidth,
    pub dst: MachineReg,
    pub lhs: MachineReg,
    pub not_rhs: MachineReg,
    pub not_result: MachineReg,
}

pub(super) fn bit_clear_fusion(invert: &MachineInst, and: &MachineInst) -> Option<BitClear> {
    let MachineInstKind::IntBinary {
        width,
        op: MachineIntBinaryOp::Xor,
        dst: not_result,
        lhs: invert_lhs,
        rhs: invert_rhs,
    } = invert.kind
    else {
        return None;
    };
    let all_ones = match width {
        MachineIntWidth::I32 => u64::from(u32::MAX),
        MachineIntWidth::I64 => u64::MAX,
    };
    let not_rhs = match (invert_lhs, invert_rhs) {
        (MachineValue::Reg(src), MachineValue::Imm64(imm))
        | (MachineValue::Imm64(imm), MachineValue::Reg(src))
            if imm == all_ones || (width == MachineIntWidth::I32 && imm as u32 == u32::MAX) =>
        {
            src
        }
        _ => return None,
    };

    let MachineInstKind::IntBinary {
        width: and_width,
        op: MachineIntBinaryOp::And,
        dst,
        lhs: and_lhs,
        rhs: and_rhs,
    } = and.kind
    else {
        return None;
    };
    if and_width != width {
        return None;
    }
    let lhs = match (and_lhs, and_rhs) {
        (MachineValue::Reg(lhs), MachineValue::Reg(rhs)) if rhs == not_result => lhs,
        (MachineValue::Reg(lhs), MachineValue::Reg(rhs)) if lhs == not_result => rhs,
        _ => return None,
    };
    // If both AND operands are the temporary, eliminating its producer would
    // also eliminate the non-inverted input required by BICS.
    if lhs == not_result {
        return None;
    }
    Some(BitClear {
        width,
        dst,
        lhs,
        not_rhs,
        not_result,
    })
}

// ── Compare + select fusion ────────────────────────────────────────────────

/// Integer comparison whose materialized boolean is consumed by the next
/// select. If the boolean is dead afterward, ARM64 can emit `cmp + csel`
/// directly and omit the otherwise redundant CSET.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct IntCompareSelect {
    pub width: MachineIntWidth,
    pub kind: MachineCompareKind,
    pub sign: MachineSign,
    pub bool_reg: MachineReg,
    pub lhs: MachineValue,
    pub rhs: MachineValue,
    pub select_result: MachineReg,
}

pub(super) fn int_compare_select_fusion(
    compare: &MachineInst,
    select: &MachineInst,
) -> Option<IntCompareSelect> {
    let MachineInstKind::IntCompare {
        width,
        kind,
        sign,
        dst: bool_reg,
        lhs,
        rhs,
    } = compare.kind
    else {
        return None;
    };
    let MachineInstKind::Select {
        ty,
        dst: select_result,
        on_true,
        on_false,
        cond: MachineValue::Reg(cond),
    } = select.kind
    else {
        return None;
    };
    if ty == MachineStorageType::V128
        || cond != bool_reg
        || on_true == MachineValue::Reg(bool_reg)
        || on_false == MachineValue::Reg(bool_reg)
    {
        return None;
    }
    Some(IntCompareSelect {
        width,
        kind,
        sign,
        bool_reg,
        lhs,
        rhs,
        select_result,
    })
}

// ── Condition code mapping ───────────────────────────────────────────────────

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

#[cfg(test)]
mod tests {
    use super::*;

    fn compare_select() -> (MachineInst, MachineInst) {
        let compare = MachineInst {
            kind: MachineInstKind::IntCompare {
                width: MachineIntWidth::I32,
                kind: MachineCompareKind::Eq,
                sign: MachineSign::Unsigned,
                dst: MachineReg(15),
                lhs: MachineValue::Reg(MachineReg(21)),
                rhs: MachineValue::Reg(MachineReg(4)),
            },
        };
        let select = MachineInst {
            kind: MachineInstKind::Select {
                ty: MachineStorageType::GpWord,
                dst: MachineReg(4),
                on_true: MachineValue::Reg(MachineReg(13)),
                on_false: MachineValue::Reg(MachineReg(14)),
                cond: MachineValue::Reg(MachineReg(15)),
            },
        };
        (compare, select)
    }

    #[test]
    fn recognizes_dead_compare_result_consumed_by_select() {
        let (compare, select) = compare_select();
        assert_eq!(
            int_compare_select_fusion(&compare, &select),
            Some(IntCompareSelect {
                width: MachineIntWidth::I32,
                kind: MachineCompareKind::Eq,
                sign: MachineSign::Unsigned,
                bool_reg: MachineReg(15),
                lhs: MachineValue::Reg(MachineReg(21)),
                rhs: MachineValue::Reg(MachineReg(4)),
                select_result: MachineReg(4),
            })
        );
    }

    #[test]
    fn recognizes_not_and_as_bit_clear() {
        let invert = MachineInst {
            kind: MachineInstKind::IntBinary {
                width: MachineIntWidth::I64,
                op: MachineIntBinaryOp::Xor,
                dst: MachineReg(5),
                lhs: MachineValue::Reg(MachineReg(6)),
                rhs: MachineValue::Imm64(u64::MAX),
            },
        };
        let and = MachineInst {
            kind: MachineInstKind::IntBinary {
                width: MachineIntWidth::I64,
                op: MachineIntBinaryOp::And,
                dst: MachineReg(7),
                lhs: MachineValue::Reg(MachineReg(8)),
                rhs: MachineValue::Reg(MachineReg(5)),
            },
        };
        assert_eq!(
            bit_clear_fusion(&invert, &and),
            Some(BitClear {
                width: MachineIntWidth::I64,
                dst: MachineReg(7),
                lhs: MachineReg(8),
                not_rhs: MachineReg(6),
                not_result: MachineReg(5),
            })
        );
    }

    #[test]
    fn rejects_compare_boolean_used_as_a_select_value() {
        let (compare, mut select) = compare_select();
        let MachineInstKind::Select {
            ref mut on_true, ..
        } = select.kind
        else {
            unreachable!();
        };
        *on_true = MachineValue::Reg(MachineReg(15));
        assert_eq!(int_compare_select_fusion(&compare, &select), None);
    }

    #[test]
    fn selects_immediate_rotate_aliases() {
        let dst = Arm64Reg::from_raw(0);
        let src = Arm64Reg::from_raw(1);

        assert_eq!(enc::ror_imm_64(dst, src, 1), 0x93c1_0420);
        assert_eq!(enc::ror_imm_32(dst, src, 31), 0x1381_7c20);
        assert_eq!(
            int_binary_imm_inst(MachineIntWidth::I64, MachineIntBinaryOp::Rotr, dst, src, 65,),
            Some(enc::ror_imm_64(dst, src, 1)),
        );
        assert_eq!(
            int_binary_imm_inst(MachineIntWidth::I64, MachineIntBinaryOp::Rotl, dst, src, 1,),
            Some(enc::ror_imm_64(dst, src, 63)),
        );
        assert_eq!(
            int_binary_imm_inst(MachineIntWidth::I32, MachineIntBinaryOp::Rotl, dst, src, 33,),
            Some(enc::ror_imm_32(dst, src, 31)),
        );
    }
}
