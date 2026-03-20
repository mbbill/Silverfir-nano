use super::types::MachineFloatWidth;
use alloc::vec::Vec;

use super::{
    types::{MachineBlockId, MachineConstId, MachineFuncId},
    MachineBranchCond, MachineEdge, MachineInstKind, MachineReg, MachineTerminator, MachineValue,
};
use crate::vm::native::ir::runtime::MachineExternBinding;

/// One read-only sidecar constant record referenced from machine IR.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MachineConstData {
    pub id: MachineConstId,
    pub align: u32,
    pub bytes: Vec<u8>,
}

/// Full machine program for one function.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MachineProgram {
    pub entry: MachineBlockId,
    /// Registers `[first_fp_reg, reg_count)` belong to the FP bank.
    pub first_fp_reg: u16,
    pub reg_count: u16,
    /// Number of FP-bank registers reserved for transient SSA values.
    pub fp_transient_count: u16,
    /// Initial semantic width for each FP-bank register, indexed by
    /// `reg - first_fp_reg`. FP cached-local regs use `Some(width)`;
    /// transient regs start as `None`.
    pub fp_reg_init_widths: Vec<Option<MachineFloatWidth>>,
    pub blocks: Vec<super::cfg::MachineBlock>,
}

impl MachineProgram {
    #[inline]
    pub fn is_fp_reg(&self, reg: super::MachineReg) -> bool {
        reg.0 >= self.first_fp_reg && reg.0 < self.reg_count
    }

    #[inline]
    pub fn is_gp_reg(&self, reg: super::MachineReg) -> bool {
        reg.0 < self.first_fp_reg
    }

    pub fn uses_reg(&self, target: MachineReg) -> bool {
        self.blocks.iter().any(|block| {
            block.params.iter().any(|param| param.reg == target)
                || block
                    .ops
                    .iter()
                    .any(|inst| inst_kind_uses_reg(&inst.kind, target))
                || term_uses_reg(&block.terminator, target)
        })
    }
}

fn value_uses_reg(value: MachineValue, target: MachineReg) -> bool {
    matches!(value, MachineValue::Reg(reg) if reg == target)
}

fn edge_uses_reg(edge: &MachineEdge, target: MachineReg) -> bool {
    edge.args.iter().any(|arg| value_uses_reg(*arg, target))
}

fn cond_uses_reg(cond: &MachineBranchCond, target: MachineReg) -> bool {
    match cond {
        MachineBranchCond::Value(value) => value_uses_reg(*value, target),
        MachineBranchCond::IntCompare { lhs, rhs, .. }
        | MachineBranchCond::FloatCompare { lhs, rhs, .. } => {
            value_uses_reg(*lhs, target) || value_uses_reg(*rhs, target)
        }
    }
}

fn term_uses_reg(term: &MachineTerminator, target: MachineReg) -> bool {
    match term {
        MachineTerminator::Return | MachineTerminator::Trap { .. } => false,
        MachineTerminator::Jump(edge) => edge_uses_reg(edge, target),
        MachineTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => {
            cond_uses_reg(cond, target)
                || edge_uses_reg(then_edge, target)
                || edge_uses_reg(else_edge, target)
        }
        MachineTerminator::JumpTable { index, entries } => {
            value_uses_reg(*index, target) || entries.iter().any(|edge| edge_uses_reg(edge, target))
        }
        MachineTerminator::CallDirect {
            callee_frame_base, ..
        } => *callee_frame_base == target,
        MachineTerminator::CallIndirect {
            callee_target,
            callee_frame_base,
            ..
        } => value_uses_reg(*callee_target, target) || *callee_frame_base == target,
    }
}

fn inst_kind_uses_reg(inst: &MachineInstKind, target: MachineReg) -> bool {
    match inst {
        MachineInstKind::Move { dst, src, .. } | MachineInstKind::Convert { dst, src, .. } => {
            *dst == target || value_uses_reg(*src, target)
        }
        MachineInstKind::FloatConst { dst, .. } => *dst == target,
        MachineInstKind::Lea { dst, addr } | MachineInstKind::Load { dst, addr, .. } => {
            *dst == target || addr.base == target
        }
        MachineInstKind::Store { addr, src, .. } => {
            addr.base == target || value_uses_reg(*src, target)
        }
        MachineInstKind::IntUnary { dst, src, .. }
        | MachineInstKind::FloatUnary { dst, src, .. } => {
            *dst == target || value_uses_reg(*src, target)
        }
        MachineInstKind::IntBinary { dst, lhs, rhs, .. }
        | MachineInstKind::IntCompare { dst, lhs, rhs, .. }
        | MachineInstKind::FloatBinary { dst, lhs, rhs, .. }
        | MachineInstKind::FloatCompare { dst, lhs, rhs, .. } => {
            *dst == target || value_uses_reg(*lhs, target) || value_uses_reg(*rhs, target)
        }
        MachineInstKind::IntMulWide {
            dst_lo,
            dst_hi,
            lhs,
            rhs,
            ..
        } => {
            *dst_lo == target
                || *dst_hi == target
                || value_uses_reg(*lhs, target)
                || value_uses_reg(*rhs, target)
        }
        MachineInstKind::Int64PairBinary {
            dst_lo,
            dst_hi,
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        }
        | MachineInstKind::Int64PairDivRem {
            dst_lo,
            dst_hi,
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        } => {
            *dst_lo == target
                || *dst_hi == target
                || value_uses_reg(*lhs_lo, target)
                || value_uses_reg(*lhs_hi, target)
                || value_uses_reg(*rhs_lo, target)
                || value_uses_reg(*rhs_hi, target)
        }
        MachineInstKind::Int64PairUnary {
            dst_lo,
            dst_hi,
            src_lo,
            src_hi,
            ..
        } => {
            *dst_lo == target
                || *dst_hi == target
                || value_uses_reg(*src_lo, target)
                || value_uses_reg(*src_hi, target)
        }
        MachineInstKind::Int64PairShift {
            dst_lo,
            dst_hi,
            lhs_lo,
            lhs_hi,
            rhs,
            ..
        } => {
            *dst_lo == target
                || *dst_hi == target
                || value_uses_reg(*lhs_lo, target)
                || value_uses_reg(*lhs_hi, target)
                || value_uses_reg(*rhs, target)
        }
        MachineInstKind::Int64PairCompare {
            dst,
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        } => {
            *dst == target
                || value_uses_reg(*lhs_lo, target)
                || value_uses_reg(*lhs_hi, target)
                || value_uses_reg(*rhs_lo, target)
                || value_uses_reg(*rhs_hi, target)
        }
        MachineInstKind::ConvertI64PairToFloat {
            dst,
            src_lo,
            src_hi,
            ..
        } => *dst == target || value_uses_reg(*src_lo, target) || value_uses_reg(*src_hi, target),
        MachineInstKind::ConvertFloatToI64Pair {
            dst_lo,
            dst_hi,
            src,
            ..
        }
        | MachineInstKind::ReinterpretF64ToI64Pair {
            dst_lo,
            dst_hi,
            src,
        } => *dst_lo == target || *dst_hi == target || value_uses_reg(*src, target),
        MachineInstKind::ReinterpretI64PairToF64 {
            dst,
            src_lo,
            src_hi,
        } => *dst == target || value_uses_reg(*src_lo, target) || value_uses_reg(*src_hi, target),
        MachineInstKind::Select {
            dst,
            on_true,
            on_false,
            cond,
            ..
        } => {
            *dst == target
                || value_uses_reg(*on_true, target)
                || value_uses_reg(*on_false, target)
                || value_uses_reg(*cond, target)
        }
        MachineInstKind::TrapIf { cond, .. } => cond_uses_reg(cond, target),
        MachineInstKind::CallHelper(_) => false,
    }
}

/// One machine function inside a machine module.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MachineFunction {
    pub id: MachineFuncId,
    pub program: MachineProgram,
}

/// One full machine module.
///
/// This is the shared allocation domain for function ids and sidecar constant
/// ids used by the machine IR, plus opaque external target ids used by helper
/// calls and other out-of-line native targets.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MachineModule {
    pub functions: Vec<MachineFunction>,
    pub consts: Vec<MachineConstData>,
    pub externs: Vec<MachineExternBinding>,
}

impl MachineModule {
    /// Run ISA-agnostic optimization passes on all functions.
    ///
    /// `first_transient` is the first GP transient register. FP transients are
    /// the prefix of the FP bank with length `fp_transient_count`.
    pub fn optimize(&mut self, first_transient: u16, gp_reg_width: u8) {
        for func in &mut self.functions {
            super::peephole::optimize(&mut func.program, first_transient, gp_reg_width);
        }
    }
}
