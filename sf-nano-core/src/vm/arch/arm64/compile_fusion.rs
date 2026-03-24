//! Codegen fusion pattern functions: immediate selection,
//! float compare-branch fusion, and liveness analysis helpers.

use crate::error::WasmError;
use crate::vm::machine::machine_ir::{
    MachineBlock, MachineBlockId, MachineBranchCond,
    MachineInstKind, MachineIntBinaryOp, MachineIntWidth,
    MachineMemWidth, MachineReg, MachineTerminator, MachineValue,
};

use super::abi::map_reg;
use super::enc::{self, Cond};
use super::reg::Arm64Reg;
use super::compile_helpers::map_float_cond;

pub(super) fn int_binary_imm_inst(
    width: MachineIntWidth,
    op: MachineIntBinaryOp,
    dst: Arm64Reg,
    lhs: MachineValue,
    rhs: MachineValue,
) -> Result<Option<u32>, WasmError> {
    match (width, op, lhs, rhs) {
        (
            MachineIntWidth::I32,
            MachineIntBinaryOp::Add,
            MachineValue::Reg(lhs),
            MachineValue::Imm64(rhs),
        ) => {
            let lhs = map_reg(lhs)?;
            Ok(add_sub_imm_inst_32(true, dst, lhs, rhs))
        }
        (
            MachineIntWidth::I32,
            MachineIntBinaryOp::Add,
            MachineValue::Imm64(lhs),
            MachineValue::Reg(rhs),
        ) => {
            let rhs = map_reg(rhs)?;
            Ok(add_sub_imm_inst_32(true, dst, rhs, lhs))
        }
        (
            MachineIntWidth::I64,
            MachineIntBinaryOp::Add,
            MachineValue::Reg(lhs),
            MachineValue::Imm64(rhs),
        ) => {
            let lhs = map_reg(lhs)?;
            Ok(add_sub_imm_inst_64(true, dst, lhs, rhs))
        }
        (
            MachineIntWidth::I64,
            MachineIntBinaryOp::Add,
            MachineValue::Imm64(lhs),
            MachineValue::Reg(rhs),
        ) => {
            let rhs = map_reg(rhs)?;
            Ok(add_sub_imm_inst_64(true, dst, rhs, lhs))
        }
        (
            MachineIntWidth::I32,
            MachineIntBinaryOp::Sub,
            MachineValue::Reg(lhs),
            MachineValue::Imm64(rhs),
        ) => {
            let lhs = map_reg(lhs)?;
            Ok(add_sub_imm_inst_32(false, dst, lhs, rhs))
        }
        (
            MachineIntWidth::I64,
            MachineIntBinaryOp::Sub,
            MachineValue::Reg(lhs),
            MachineValue::Imm64(rhs),
        ) => {
            let lhs = map_reg(lhs)?;
            Ok(add_sub_imm_inst_64(false, dst, lhs, rhs))
        }
        (
            MachineIntWidth::I32,
            MachineIntBinaryOp::Mul,
            MachineValue::Reg(lhs),
            MachineValue::Imm64(rhs),
        ) => {
            let lhs = map_reg(lhs)?;
            Ok(mul_imm_inst_32(dst, lhs, rhs as u32))
        }
        (
            MachineIntWidth::I32,
            MachineIntBinaryOp::Mul,
            MachineValue::Imm64(lhs),
            MachineValue::Reg(rhs),
        ) => {
            let rhs = map_reg(rhs)?;
            Ok(mul_imm_inst_32(dst, rhs, lhs as u32))
        }
        (
            MachineIntWidth::I64,
            MachineIntBinaryOp::Mul,
            MachineValue::Reg(lhs),
            MachineValue::Imm64(rhs),
        ) => {
            let lhs = map_reg(lhs)?;
            Ok(mul_imm_inst_64(dst, lhs, rhs))
        }
        (
            MachineIntWidth::I64,
            MachineIntBinaryOp::Mul,
            MachineValue::Imm64(lhs),
            MachineValue::Reg(rhs),
        ) => {
            let rhs = map_reg(rhs)?;
            Ok(mul_imm_inst_64(dst, rhs, lhs))
        }
        (
            MachineIntWidth::I32,
            MachineIntBinaryOp::And,
            MachineValue::Reg(lhs),
            MachineValue::Imm64(rhs),
        ) => {
            let lhs = map_reg(lhs)?;
            Ok(logical_imm_inst_32(op, dst, lhs, rhs as u32))
        }
        (
            MachineIntWidth::I32,
            MachineIntBinaryOp::And,
            MachineValue::Imm64(lhs),
            MachineValue::Reg(rhs),
        ) => {
            let rhs = map_reg(rhs)?;
            Ok(logical_imm_inst_32(op, dst, rhs, lhs as u32))
        }
        (
            MachineIntWidth::I64,
            MachineIntBinaryOp::And,
            MachineValue::Reg(lhs),
            MachineValue::Imm64(rhs),
        ) => {
            let lhs = map_reg(lhs)?;
            Ok(logical_imm_inst_64(op, dst, lhs, rhs))
        }
        (
            MachineIntWidth::I64,
            MachineIntBinaryOp::And,
            MachineValue::Imm64(lhs),
            MachineValue::Reg(rhs),
        ) => {
            let rhs = map_reg(rhs)?;
            Ok(logical_imm_inst_64(op, dst, rhs, lhs))
        }
        (
            MachineIntWidth::I32,
            MachineIntBinaryOp::Or,
            MachineValue::Reg(lhs),
            MachineValue::Imm64(rhs),
        ) => {
            let lhs = map_reg(lhs)?;
            Ok(logical_imm_inst_32(op, dst, lhs, rhs as u32))
        }
        (
            MachineIntWidth::I32,
            MachineIntBinaryOp::Or,
            MachineValue::Imm64(lhs),
            MachineValue::Reg(rhs),
        ) => {
            let rhs = map_reg(rhs)?;
            Ok(logical_imm_inst_32(op, dst, rhs, lhs as u32))
        }
        (
            MachineIntWidth::I64,
            MachineIntBinaryOp::Or,
            MachineValue::Reg(lhs),
            MachineValue::Imm64(rhs),
        ) => {
            let lhs = map_reg(lhs)?;
            Ok(logical_imm_inst_64(op, dst, lhs, rhs))
        }
        (
            MachineIntWidth::I64,
            MachineIntBinaryOp::Or,
            MachineValue::Imm64(lhs),
            MachineValue::Reg(rhs),
        ) => {
            let rhs = map_reg(rhs)?;
            Ok(logical_imm_inst_64(op, dst, rhs, lhs))
        }
        (
            MachineIntWidth::I32,
            MachineIntBinaryOp::Xor,
            MachineValue::Reg(lhs),
            MachineValue::Imm64(rhs),
        ) => {
            let lhs = map_reg(lhs)?;
            Ok(logical_imm_inst_32(op, dst, lhs, rhs as u32))
        }
        (
            MachineIntWidth::I32,
            MachineIntBinaryOp::Xor,
            MachineValue::Imm64(lhs),
            MachineValue::Reg(rhs),
        ) => {
            let rhs = map_reg(rhs)?;
            Ok(logical_imm_inst_32(op, dst, rhs, lhs as u32))
        }
        (
            MachineIntWidth::I64,
            MachineIntBinaryOp::Xor,
            MachineValue::Reg(lhs),
            MachineValue::Imm64(rhs),
        ) => {
            let lhs = map_reg(lhs)?;
            Ok(logical_imm_inst_64(op, dst, lhs, rhs))
        }
        (
            MachineIntWidth::I64,
            MachineIntBinaryOp::Xor,
            MachineValue::Imm64(lhs),
            MachineValue::Reg(rhs),
        ) => {
            let rhs = map_reg(rhs)?;
            Ok(logical_imm_inst_64(op, dst, rhs, lhs))
        }
        (
            MachineIntWidth::I32,
            MachineIntBinaryOp::Shl,
            MachineValue::Reg(lhs),
            MachineValue::Imm64(rhs),
        ) => {
            let lhs = map_reg(lhs)?;
            Ok(Some(enc::lsl_imm_32(dst, lhs, (rhs as u32) & 31)))
        }
        (
            MachineIntWidth::I64,
            MachineIntBinaryOp::Shl,
            MachineValue::Reg(lhs),
            MachineValue::Imm64(rhs),
        ) => {
            let lhs = map_reg(lhs)?;
            Ok(Some(enc::lsl_imm_64(dst, lhs, (rhs as u32) & 63)))
        }
        (
            MachineIntWidth::I32,
            MachineIntBinaryOp::ShrU,
            MachineValue::Reg(lhs),
            MachineValue::Imm64(rhs),
        ) => {
            let lhs = map_reg(lhs)?;
            Ok(Some(enc::lsr_imm_32(dst, lhs, (rhs as u32) & 31)))
        }
        (
            MachineIntWidth::I64,
            MachineIntBinaryOp::ShrU,
            MachineValue::Reg(lhs),
            MachineValue::Imm64(rhs),
        ) => {
            let lhs = map_reg(lhs)?;
            Ok(Some(enc::lsr_imm_64(dst, lhs, (rhs as u32) & 63)))
        }
        (
            MachineIntWidth::I32,
            MachineIntBinaryOp::ShrS,
            MachineValue::Reg(lhs),
            MachineValue::Imm64(rhs),
        ) => {
            let lhs = map_reg(lhs)?;
            Ok(Some(enc::asr_imm_32(dst, lhs, (rhs as u32) & 31)))
        }
        (
            MachineIntWidth::I64,
            MachineIntBinaryOp::ShrS,
            MachineValue::Reg(lhs),
            MachineValue::Imm64(rhs),
        ) => {
            let lhs = map_reg(lhs)?;
            Ok(Some(enc::asr_imm_64(dst, lhs, (rhs as u32) & 63)))
        }
        _ => Ok(None),
    }
}

pub(super) fn add_sub_imm_inst_32(
    is_add: bool,
    dst: Arm64Reg,
    lhs: Arm64Reg,
    imm: u64,
) -> Option<u32> {
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

pub(super) fn add_sub_imm_inst_64(
    is_add: bool,
    dst: Arm64Reg,
    lhs: Arm64Reg,
    imm: u64,
) -> Option<u32> {
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

pub(super) fn mul_imm_inst_32(dst: Arm64Reg, lhs: Arm64Reg, imm: u32) -> Option<u32> {
    if imm == 0 {
        return Some(enc::movz_32(dst, 0, 0));
    }
    if imm == 1 {
        return Some(enc::mov_reg_32(dst, lhs));
    }
    imm.is_power_of_two()
        .then(|| enc::lsl_imm_32(dst, lhs, imm.trailing_zeros()))
}

pub(super) fn mul_imm_inst_64(dst: Arm64Reg, lhs: Arm64Reg, imm: u64) -> Option<u32> {
    if imm == 0 {
        return Some(enc::movz_64(dst, 0, 0));
    }
    if imm == 1 {
        return Some(enc::mov_reg_64(dst, lhs));
    }
    imm.is_power_of_two()
        .then(|| enc::lsl_imm_64(dst, lhs, imm.trailing_zeros()))
}

pub(super) fn logical_imm_inst_32(
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

pub(super) fn logical_imm_inst_64(
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

pub(super) fn cmp_imm_inst(
    width: MachineIntWidth,
    lhs: MachineValue,
    rhs: MachineValue,
) -> Result<Option<u32>, WasmError> {
    let (lhs, rhs) = match (lhs, rhs) {
        (MachineValue::Reg(lhs), MachineValue::Imm64(rhs)) => (lhs, rhs),
        _ => return Ok(None),
    };
    let lhs = map_reg(lhs)?;
    Ok(match width {
        MachineIntWidth::I32 => try_imm12_u32(rhs as u32).map(|imm12| enc::cmp_imm_32(lhs, imm12)),
        MachineIntWidth::I64 => try_imm12_u64(rhs).map(|imm12| enc::cmp_imm_64(lhs, imm12)),
    })
}

pub(super) fn try_imm12_u32(value: u32) -> Option<u32> {
    (value < 4096).then_some(value)
}

pub(super) fn try_imm12_u64(value: u64) -> Option<u32> {
    (value < 4096).then_some(value as u32)
}

/// Detect when the last instruction in a block is a FloatCompare whose result
/// register is only used by the branch terminator. Returns a fused ARM64 Cond.
pub(super) fn float_compare_branch_fusion(
    block: &MachineBlock,
    all_blocks: &[MachineBlock],
) -> Option<Cond> {
    let last = block.ops.last()?;
    let MachineTerminator::Branch {
        cond: MachineBranchCond::Value(MachineValue::Reg(cond_reg)),
        then_edge,
        else_edge,
    } = &block.terminator
    else {
        return None;
    };
    let MachineInstKind::FloatCompare { kind, dst, .. } = &last.kind else {
        return None;
    };
    if dst != cond_reg {
        return None;
    }
    if crate::vm::machine::peephole::reg_dead_at_block_entry(
        all_blocks,
        then_edge.target,
        *dst,
    ) && crate::vm::machine::peephole::reg_dead_at_block_entry(
        all_blocks,
        else_edge.target,
        *dst,
    ) {
        Some(map_float_cond(*kind))
    } else {
        None
    }
}

/// Detect consecutive `Store { src: Imm64(0), width: U64 }` pairs with the same
/// base register and adjacent 8-byte-aligned offsets.
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
    // STP signed imm7 range: -64..63
    if !(-64..=63).contains(&imm7) {
        return None;
    }
    Some((addr_a.base, imm7))
}

pub(super) fn is_fallthrough_edge(
    compiler: &super::compile::FunctionCompiler<'_>,
    target: MachineBlockId,
    args: &[MachineValue],
    fallthrough: Option<MachineBlockId>,
) -> bool {
    fallthrough == Some(target) && compiler.is_identity_edge(target, args)
}
