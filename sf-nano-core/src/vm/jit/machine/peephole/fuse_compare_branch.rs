//! Compare-and-branch fusion pass.
//!
//! Fuses `IntCompare { dst } + Branch { Reg(dst) }` into
//! `Branch { IntCompare { ... } }` when the compare result register is a
//! linear value and is dead in both successor blocks. Stores may sit between
//! the compare and the branch (a block often spills a value it compared, as
//! a function entry does with its parameters): a store defines no register,
//! so the compare's operands still hold the same values at the branch.
//!
//! This is a cross-block pass: it reads successor blocks to check liveness,
//! so it must run after the per-block optimizations are done.

use crate::vm::jit::backend::BackendConfig;
use crate::vm::jit::machine::machine_ir::{
    MachineBlock, MachineBlockId, MachineBranchCond, MachineEdge, MachineInstKind, MachineIntWidth,
    MachineReg, MachineRegOwner, MachineTerminator, MachineValue,
};

use super::helpers::{count_value_uses, inst_defines, terminator_uses_reg, value_is_reg};

pub(super) fn fuse_compare_branch(
    blocks: &mut [MachineBlock],
    gp_reg_width: u8,
    _config: BackendConfig,
) {
    for idx in 0..blocks.len() {
        // Terminator must be Branch { cond: Value(Reg(cond_reg)) }.
        let (cond_reg, then_target, else_target) = match &blocks[idx].terminator {
            MachineTerminator::Branch {
                cond: MachineBranchCond::Value(MachineValue::Reg(r)),
                then_edge,
                else_edge,
            } => (*r, then_edge.target, else_edge.target),
            _ => continue,
        };

        // The compare is the last op, or is followed only by stores that do
        // not read its result.
        let Some(compare_idx) = blocks[idx].ops.iter().rposition(|op| {
            !matches!(
                op.kind,
                MachineInstKind::Store { .. } | MachineInstKind::IndexedStore { .. }
            ) || count_value_uses(&op.kind, cond_reg) > 0
        }) else {
            continue;
        };
        let last_op = &blocks[idx].ops[compare_idx];

        // Build the fused branch condition, or skip.
        //
        // IntCompare and TestBits are fused here. FloatCompare is NOT fused
        // because on x86_64 Wasm float comparisons require multi-instruction
        // NaN handling (SETCC+SETNP+AND) that cannot be expressed as a single
        // conditional branch. ARM64's FCMP condition codes handle NaN
        // correctly with a single B.cond, but since this is a shared pass
        // it must be safe for all backends.
        let fused_cond = match &last_op.kind {
            MachineInstKind::IntCompare { width, .. }
                if *width == MachineIntWidth::I64 && gp_reg_width == 4 =>
            {
                continue
            }
            MachineInstKind::IntCompare {
                width,
                kind,
                sign,
                dst,
                lhs,
                rhs,
            } if *dst == cond_reg
                && last_op.kind.def_owner() == Some(MachineRegOwner::LinearValue) =>
            {
                MachineBranchCond::IntCompare {
                    width: *width,
                    kind: *kind,
                    sign: *sign,
                    lhs: *lhs,
                    rhs: *rhs,
                }
            }
            MachineInstKind::TestBits { width, .. }
                if *width == MachineIntWidth::I64 && gp_reg_width == 4 =>
            {
                continue
            }
            MachineInstKind::TestBits {
                width,
                kind,
                dst,
                src,
                mask,
            } if *dst == cond_reg
                && last_op.kind.def_owner() == Some(MachineRegOwner::LinearValue) =>
            {
                MachineBranchCond::TestBits {
                    width: *width,
                    kind: *kind,
                    src: MachineValue::Reg(*src),
                    mask: *mask,
                }
            }
            _ => continue,
        };

        // Reject if any edge passes dst as an arg.
        if term_edge_uses_value(&blocks[idx].terminator, cond_reg) {
            continue;
        }

        // Reject if dst is live-in to either successor.
        if !reg_dead_at_block_entry(blocks, then_target, cond_reg) {
            continue;
        }
        if !reg_dead_at_block_entry(blocks, else_target, cond_reg) {
            continue;
        }

        // Safe to fuse: remove the compare op and rewrite the terminator.
        blocks[idx].ops.remove(compare_idx);
        if let MachineTerminator::Branch { cond, .. } = &mut blocks[idx].terminator {
            *cond = fused_cond;
        }
    }
}

/// Check whether any edge arg in the terminator references `reg`.
fn term_edge_uses_value(term: &MachineTerminator, reg: MachineReg) -> bool {
    let check = |e: &MachineEdge| e.args.iter().any(|a| value_is_reg(a, reg));
    match term {
        MachineTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } => check(then_edge) || check(else_edge),
        MachineTerminator::Jump(edge) => check(edge),
        MachineTerminator::JumpTable { entries, .. } => entries.iter().any(|e| check(e)),
        _ => false,
    }
}

/// Returns true if `reg` is provably dead at the beginning of `target`:
/// either the block defines it before any use, the block has it as a
/// parameter, or the block never touches it.
fn reg_dead_at_block_entry(
    blocks: &[MachineBlock],
    target: MachineBlockId,
    reg: MachineReg,
) -> bool {
    let Some(block) = blocks.get(target.as_usize()) else {
        return false;
    };
    // If the target has reg as a param, it will be defined by the edge.
    if block.params.iter().any(|p| p.reg == reg) {
        return true;
    }
    // Scan ops: defined before used -> dead at entry.
    for op in &block.ops {
        if inst_defines(&op.kind, reg) {
            return true;
        }
        if count_value_uses(&op.kind, reg) > 0 {
            return false;
        }
    }
    // Reached terminator without touching reg.
    !terminator_uses_reg(&block.terminator, reg)
}
