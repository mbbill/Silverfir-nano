//! Jump-table index bounds from the predecessor's branch.
//!
//! A `JumpTable` sends any index above its last entry to that entry, the
//! default, so a backend clamps the index before indexing the table. A
//! `br_table` is often guarded by an explicit range check instead, which
//! LLVM emits for a `switch` after subtracting the lowest case:
//!
//! ```text
//! b7:  r5 <- ...; branch i32 r5 <=u 81 then b10 [r5] else b8
//! b10: jump_table r5 [82 cases, default]
//! ```
//!
//! When a jump table's block has one predecessor, that predecessor enters
//! it on one side of an unsigned i32 compare of the index's incoming value
//! against a constant, and the block does not redefine the index, the
//! compare bounds the index. If that bound is at most the default's
//! position, the table is marked `in_range` and the clamp can go.

use crate::vm::jit::machine::machine_ir::{
    MachineBlock, MachineBlockId, MachineBranchCond, MachineCompareKind, MachineIntWidth,
    MachineSign, MachineTerminator, MachineValue,
};

use super::forward_edge_stored_values::only_predecessors;
use super::helpers::inst_defines;

pub(super) fn bound_jump_table_indexes(blocks: &mut [MachineBlock]) {
    // Most functions have no jump table; skip the predecessor scan for them.
    if !blocks
        .iter()
        .any(|block| matches!(block.terminator, MachineTerminator::JumpTable { .. }))
    {
        return;
    }
    let predecessors = only_predecessors(blocks);
    for target in 0..blocks.len() {
        let block = &blocks[target];
        let MachineTerminator::JumpTable {
            index: MachineValue::Reg(index),
            entries,
            in_range: false,
        } = &block.terminator
        else {
            continue;
        };
        let (index, default) = (*index, entries.len().saturating_sub(1));
        if block.ops.iter().any(|op| inst_defines(&op.kind, index)) {
            continue;
        }
        let Some(param) = block.params.iter().position(|param| param.reg == index) else {
            continue;
        };
        let Some(predecessor) = predecessors[target] else {
            continue;
        };
        let Some(max) = index_bound(&blocks[predecessor].terminator, block.id, param) else {
            continue;
        };
        if u64::from(max) <= default as u64 {
            if let MachineTerminator::JumpTable { in_range, .. } = &mut blocks[target].terminator {
                *in_range = true;
            }
        }
    }
}

/// The largest unsigned i32 value that `terminator` can pass as argument
/// `param` on its only edge to `target`, when an i32 compare of that same
/// value against a constant decides the edge.
fn index_bound(
    terminator: &MachineTerminator,
    target: MachineBlockId,
    param: usize,
) -> Option<u32> {
    let MachineTerminator::Branch {
        cond:
            MachineBranchCond::IntCompare {
                width: MachineIntWidth::I32,
                kind,
                sign: MachineSign::Unsigned,
                lhs: MachineValue::Reg(compared),
                rhs: MachineValue::Imm64(constant),
            },
        then_edge,
        else_edge,
    } = terminator
    else {
        return None;
    };
    let (edge, taken) = match (then_edge.target == target, else_edge.target == target) {
        (true, false) => (then_edge, true),
        (false, true) => (else_edge, false),
        _ => return None,
    };
    if edge.args.get(param) != Some(&MachineValue::Reg(*compared)) {
        return None;
    }
    // An i32 compare reads the constant's low 32 bits.
    let constant = *constant as u32;
    match (kind, taken) {
        (MachineCompareKind::Le, true) | (MachineCompareKind::Gt, false) => Some(constant),
        (MachineCompareKind::Lt, true) | (MachineCompareKind::Ge, false) => constant.checked_sub(1),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::bound_jump_table_indexes;
    use crate::collections;
    use crate::vm::jit::machine::machine_ir::{
        MachineBlock, MachineBlockId, MachineBlockParam, MachineBranchCond, MachineCompareKind,
        MachineEdge, MachineInst, MachineInstKind, MachineIntBinaryOp, MachineIntWidth, MachineReg,
        MachineSign, MachineTerminator, MachineValue,
    };

    const INDEX: MachineReg = MachineReg(5);

    fn edge(target: u32, args: &[MachineValue]) -> MachineEdge {
        MachineEdge {
            target: MachineBlockId(target),
            args: args.iter().copied().collect(),
        }
    }

    /// b0 compares the index against `constant` and enters b1, a jump table
    /// with `cases` cases plus the default, on the `taken` side; the other
    /// side goes to b2.
    fn guarded_table(
        kind: MachineCompareKind,
        constant: u64,
        taken: bool,
        cases: usize,
    ) -> collections::Vec<MachineBlock> {
        let index_arg = [MachineValue::Reg(INDEX)];
        let (then_edge, else_edge) = if taken {
            (edge(1, &index_arg), edge(2, &[]))
        } else {
            (edge(2, &[]), edge(1, &index_arg))
        };
        collections::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: collections::vec![MachineBlockParam::gp_word(INDEX)],
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Branch {
                    cond: MachineBranchCond::IntCompare {
                        width: MachineIntWidth::I32,
                        kind,
                        sign: MachineSign::Unsigned,
                        lhs: MachineValue::Reg(INDEX),
                        rhs: MachineValue::Imm64(constant),
                    },
                    then_edge,
                    else_edge,
                },
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: collections::vec![MachineBlockParam::gp_word(INDEX)],
                ops: collections::Vec::new(),
                terminator: MachineTerminator::JumpTable {
                    index: MachineValue::Reg(INDEX),
                    entries: (0..=cases).map(|_| edge(2, &[])).collect(),
                    in_range: false,
                },
            },
            MachineBlock {
                id: MachineBlockId(2),
                params: collections::Vec::new(),
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Return,
            },
        ]
    }

    fn in_range(blocks: &[MachineBlock]) -> bool {
        matches!(
            blocks[1].terminator,
            MachineTerminator::JumpTable { in_range: true, .. }
        )
    }

    fn bounded(mut blocks: collections::Vec<MachineBlock>) -> bool {
        bound_jump_table_indexes(&mut blocks);
        in_range(&blocks)
    }

    #[test]
    fn a_guard_up_to_the_default_bounds_the_index() {
        // Four cases and the default: indexes 0..=4 need no clamp.
        assert!(bounded(guarded_table(MachineCompareKind::Le, 3, true, 4)));
        assert!(bounded(guarded_table(MachineCompareKind::Le, 4, true, 4)));
        assert!(bounded(guarded_table(MachineCompareKind::Lt, 5, true, 4)));
        assert!(bounded(guarded_table(MachineCompareKind::Gt, 4, false, 4)));
        assert!(bounded(guarded_table(MachineCompareKind::Ge, 5, false, 4)));
    }

    #[test]
    fn a_guard_past_the_default_leaves_the_clamp() {
        assert!(!bounded(guarded_table(MachineCompareKind::Le, 5, true, 4)));
        assert!(!bounded(guarded_table(MachineCompareKind::Lt, 6, true, 4)));
        assert!(!bounded(guarded_table(MachineCompareKind::Gt, 5, false, 4)));
        // The out-of-range side of the guard proves nothing.
        assert!(!bounded(guarded_table(MachineCompareKind::Le, 3, false, 4)));
        assert!(!bounded(guarded_table(MachineCompareKind::Gt, 3, true, 4)));
        // `index < 0` admits nothing; it is not a bound to rely on.
        assert!(!bounded(guarded_table(MachineCompareKind::Lt, 0, true, 4)));
    }

    #[test]
    fn the_index_must_reach_the_table_unchanged_from_the_guard() {
        let mut redefined = guarded_table(MachineCompareKind::Le, 3, true, 4);
        redefined[1].ops.push(MachineInst {
            kind: MachineInstKind::IntBinary {
                width: MachineIntWidth::I32,
                op: MachineIntBinaryOp::Add,
                dst: INDEX,
                lhs: MachineValue::Reg(INDEX),
                rhs: MachineValue::Imm64(1),
            },
        });
        assert!(!bounded(redefined));

        let mut other_value = guarded_table(MachineCompareKind::Le, 3, true, 4);
        if let MachineTerminator::Branch { then_edge, .. } = &mut other_value[0].terminator {
            then_edge.args = collections::vec![MachineValue::Reg(MachineReg(6))];
        }
        assert!(!bounded(other_value));
    }

    #[test]
    fn a_second_predecessor_leaves_the_clamp() {
        let mut blocks = guarded_table(MachineCompareKind::Le, 3, true, 4);
        blocks[2].params = collections::vec![MachineBlockParam::gp_word(INDEX)];
        blocks[2].terminator = MachineTerminator::Jump(edge(1, &[MachineValue::Reg(INDEX)]));
        for entry in match &mut blocks[1].terminator {
            MachineTerminator::JumpTable { entries, .. } => entries.iter_mut(),
            _ => unreachable!(),
        } {
            entry.target = MachineBlockId(3);
        }
        blocks.push(MachineBlock {
            id: MachineBlockId(3),
            params: collections::Vec::new(),
            ops: collections::Vec::new(),
            terminator: MachineTerminator::Return,
        });
        if let MachineTerminator::Branch { else_edge, .. } = &mut blocks[0].terminator {
            else_edge.args = collections::vec![MachineValue::Reg(INDEX)];
        }
        assert!(!bounded(blocks));
    }
}
