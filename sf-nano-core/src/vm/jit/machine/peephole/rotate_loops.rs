//! Loop rotation: a latch that jumps back to a header holding nothing but a
//! conditional branch takes that branch itself.
//!
//! A `while`-shaped loop tests at its header and closes with an
//! unconditional jump, so every iteration runs two branches. A self tail
//! call becomes exactly this shape:
//!
//! ```text
//! b2(r4, r5, r6): branch r4 == 0 then b0 [r5] else b1 [r4, r5, r6]
//! b1(r4, r5, r6): r7 <- r4 - 1; r8 <- r5 + r6; jump b2 [r7, r6, r8]
//! ```
//!
//! When the header has no instructions, the latch can bind the header's
//! parameters itself and end with a copy of the header's branch. That copy
//! reads the header's parameters, which now hold exactly the values the
//! jump would have passed:
//!
//! ```text
//! b1(r4, r5, r6): r7 <- r4 - 1; r8 <- r5 + r6
//!                 r4 <- r7; r5 <- r6; r6 <- r8
//!                 branch r4 == 0 then b0 [r5] else b1 [r4, r5, r6]
//! ```
//!
//! The latch then loops on itself through one conditional branch, and the
//! header runs only on loop entry.
//!
//! Only back edges are rotated, so straight-line code keeps its layout.
//! The parameter binding is a parallel copy written out as moves; only
//! acyclic copies of general-purpose values qualify, so no move needs a
//! temporary or an FP width.

use crate::collections;
use crate::vm::jit::machine::machine_ir::{
    MachineBlock, MachineBlockId, MachineBlockParam, MachineInst, MachineInstKind,
    MachineTerminator, MachineValue,
};

use super::hoist_loop_address_bases::{block_index_for_id, visit_edges};

pub(super) fn rotate_loops(blocks: &mut [MachineBlock], entry: MachineBlockId) {
    let candidates: collections::Vec<(usize, usize)> = (0..blocks.len())
        .filter_map(|latch| {
            let MachineTerminator::Jump(edge) = &blocks[latch].terminator else {
                return None;
            };
            let header = block_index_for_id(blocks, edge.target)?;
            let target = &blocks[header];
            (header != latch
                && target.ops.is_empty()
                && matches!(target.terminator, MachineTerminator::Branch { .. }))
            .then_some((latch, header))
        })
        .collect();
    if candidates.is_empty() {
        return;
    }
    let back_edge = back_edge_jumps(blocks, entry);
    for (latch, header) in candidates {
        if !back_edge[latch] {
            continue;
        }
        let MachineTerminator::Jump(edge) = &blocks[latch].terminator else {
            continue;
        };
        let Some(moves) = sequential_copies(&blocks[header].params, &edge.args) else {
            continue;
        };
        let branch = blocks[header].terminator.clone();
        let block = &mut blocks[latch];
        block.ops.extend(moves);
        block.terminator = branch;
    }
}

/// For each block, whether its `Jump` targets a block on the depth-first
/// stack from `entry`, which makes the jump a loop back edge.
fn back_edge_jumps(blocks: &[MachineBlock], entry: MachineBlockId) -> collections::Vec<bool> {
    #[derive(Clone, Copy, PartialEq)]
    enum State {
        New,
        OnStack,
        Done,
    }
    let successors: collections::Vec<collections::Vec<usize>> = blocks
        .iter()
        .map(|block| {
            let mut targets = collections::Vec::new();
            visit_edges(&block.terminator, |edge| {
                if let Some(target) = block_index_for_id(blocks, edge.target) {
                    targets.push(target);
                }
            });
            targets
        })
        .collect();
    let mut back_edge = collections::vec![false; blocks.len()];
    let mut state = collections::vec![State::New; blocks.len()];
    let Some(root) = block_index_for_id(blocks, entry) else {
        return back_edge;
    };
    let mut stack = collections::vec![(root, 0usize)];
    state[root] = State::OnStack;
    while let Some(&mut (block, ref mut next)) = stack.last_mut() {
        let Some(&target) = successors[block].get(*next) else {
            state[block] = State::Done;
            stack.pop();
            continue;
        };
        *next += 1;
        match state[target] {
            State::OnStack => {
                if matches!(blocks[block].terminator, MachineTerminator::Jump(_)) {
                    back_edge[block] = true;
                }
            }
            State::New => {
                state[target] = State::OnStack;
                stack.push((target, 0));
            }
            State::Done => {}
        }
    }
    back_edge
}

/// The parallel copy `params <- args` as ordered moves, or `None` when an
/// argument is a reserved cache register, a parameter is FP, or the copy is
/// cyclic.
fn sequential_copies(
    params: &[MachineBlockParam],
    args: &[MachineValue],
) -> Option<collections::Vec<MachineInst>> {
    if params.len() != args.len() {
        return None;
    }
    let mut pending: collections::Vec<(MachineBlockParam, MachineValue)> = collections::Vec::new();
    for (&param, &arg) in params.iter().zip(args) {
        if param.ty.is_fp() {
            return None;
        }
        match arg {
            MachineValue::Reg(reg) if reg == param.reg => {}
            MachineValue::Reg(_) | MachineValue::Imm64(_) => pending.push((param, arg)),
            MachineValue::ReservedReg(_) => return None,
        }
    }
    let mut moves = collections::Vec::with_capacity(pending.len());
    while !pending.is_empty() {
        // A move is ready once no other pending move still reads its
        // destination; none ready means the copy is a cycle.
        let ready = pending.iter().position(|(dst, _)| {
            !pending
                .iter()
                .any(|(_, src)| *src == MachineValue::Reg(dst.reg))
        })?;
        let (dst, src) = pending.remove(ready);
        moves.push(MachineInst {
            kind: MachineInstKind::Move {
                owner: dst.owner,
                ty: dst.ty,
                dst: dst.reg,
                src,
            },
        });
    }
    Some(moves)
}

#[cfg(test)]
mod tests {
    use super::rotate_loops;
    use crate::collections;
    use crate::vm::jit::machine::machine_ir::{
        MachineBlock, MachineBlockId, MachineBlockParam, MachineBranchCond, MachineCompareKind,
        MachineEdge, MachineInst, MachineInstKind, MachineIntBinaryOp, MachineIntWidth, MachineReg,
        MachineRegOwner, MachineSign, MachineStorageType, MachineTerminator, MachineValue,
    };

    fn reg(index: u16) -> MachineValue {
        MachineValue::Reg(MachineReg(index))
    }

    fn edge(target: u32, args: &[MachineValue]) -> MachineEdge {
        MachineEdge {
            target: MachineBlockId(target),
            args: args.iter().copied().collect(),
        }
    }

    fn params(regs: &[u16], ty: MachineStorageType) -> collections::Vec<MachineBlockParam> {
        regs.iter()
            .map(|&index| MachineBlockParam {
                reg: MachineReg(index),
                ty,
                owner: MachineRegOwner::CachedCell,
            })
            .collect()
    }

    fn int_binary(
        op: MachineIntBinaryOp,
        dst: u16,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> MachineInst {
        MachineInst {
            kind: MachineInstKind::IntBinary {
                width: MachineIntWidth::I64,
                op,
                dst: MachineReg(dst),
                lhs,
                rhs,
            },
        }
    }

    fn is_zero(lhs: MachineValue) -> MachineBranchCond {
        MachineBranchCond::IntCompare {
            width: MachineIntWidth::I64,
            kind: MachineCompareKind::Eq,
            sign: MachineSign::Unsigned,
            lhs,
            rhs: MachineValue::Imm64(0),
        }
    }

    /// The fibonacci-tail loop: b2 is the entry and header, b1 the latch
    /// jumping back with `latch_args`, b0 the exit.
    fn fib_loop(
        latch_args: &[MachineValue],
        ty: MachineStorageType,
    ) -> collections::Vec<MachineBlock> {
        collections::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: params(&[5], MachineStorageType::GpI64),
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Return,
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: params(&[4, 5, 6], ty),
                ops: collections::vec![
                    int_binary(MachineIntBinaryOp::Sub, 7, reg(4), MachineValue::Imm64(1)),
                    int_binary(MachineIntBinaryOp::Add, 8, reg(5), reg(6)),
                ],
                terminator: MachineTerminator::Jump(edge(2, latch_args)),
            },
            MachineBlock {
                id: MachineBlockId(2),
                params: params(&[4, 5, 6], ty),
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Branch {
                    cond: is_zero(reg(4)),
                    then_edge: edge(0, &[reg(5)]),
                    else_edge: edge(1, &[reg(4), reg(5), reg(6)]),
                },
            },
        ]
    }

    fn moves(block: &MachineBlock) -> collections::Vec<(u16, MachineValue)> {
        block
            .ops
            .iter()
            .filter_map(|op| match op.kind {
                MachineInstKind::Move { dst, src, .. } => Some((dst.0, src)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn latch_binds_header_params_in_order_and_takes_its_branch() {
        let mut blocks = fib_loop(&[reg(7), reg(6), reg(8)], MachineStorageType::GpI64);
        let header_branch = blocks[2].terminator.clone();
        rotate_loops(&mut blocks, MachineBlockId(2));
        // r5 must read r6 before r6 is overwritten.
        assert_eq!(moves(&blocks[1]), [(4, reg(7)), (5, reg(6)), (6, reg(8))]);
        assert_eq!(blocks[1].ops.len(), 5);
        assert_eq!(blocks[1].terminator, header_branch);
        // The header itself still serves loop entry.
        assert_eq!(blocks[2].terminator, header_branch);
    }

    #[test]
    fn reads_a_parameter_before_overwriting_it() {
        // r6 takes the old r4, so it must move before r4 is rebound; the
        // parameters' own order would overwrite r4 first. r5 is free from
        // the start, so it moves first.
        let mut blocks = fib_loop(&[reg(7), reg(8), reg(4)], MachineStorageType::GpI64);
        rotate_loops(&mut blocks, MachineBlockId(2));
        assert_eq!(moves(&blocks[1]), [(5, reg(8)), (6, reg(4)), (4, reg(7))]);
    }

    #[test]
    fn identity_and_immediate_arguments() {
        let mut blocks = fib_loop(
            &[reg(7), reg(5), MachineValue::Imm64(3)],
            MachineStorageType::GpI64,
        );
        rotate_loops(&mut blocks, MachineBlockId(2));
        assert_eq!(
            moves(&blocks[1]),
            [(4, reg(7)), (6, MachineValue::Imm64(3))]
        );
    }

    #[test]
    fn keeps_the_jump_for_a_cyclic_copy() {
        let mut blocks = fib_loop(&[reg(5), reg(4), reg(6)], MachineStorageType::GpI64);
        let before = blocks.clone();
        rotate_loops(&mut blocks, MachineBlockId(2));
        assert_eq!(blocks, before);
    }

    #[test]
    fn keeps_the_jump_for_fp_params() {
        let mut blocks = fib_loop(&[reg(7), reg(6), reg(8)], MachineStorageType::Fp64);
        let before = blocks.clone();
        rotate_loops(&mut blocks, MachineBlockId(2));
        assert_eq!(blocks, before);
    }

    #[test]
    fn keeps_the_jump_when_the_header_computes() {
        let mut blocks = fib_loop(&[reg(7), reg(6), reg(8)], MachineStorageType::GpI64);
        blocks[2]
            .ops
            .push(int_binary(MachineIntBinaryOp::Add, 9, reg(4), reg(5)));
        let before = blocks.clone();
        rotate_loops(&mut blocks, MachineBlockId(2));
        assert_eq!(blocks, before);
    }

    #[test]
    fn keeps_a_forward_jump_to_a_branch() {
        // Entering at b1 makes b1 -> b2 a forward edge: b2 is not on the
        // stack when b1 is explored, and b2 -> b1 is the back edge instead.
        let mut blocks = fib_loop(&[reg(7), reg(6), reg(8)], MachineStorageType::GpI64);
        let before = blocks.clone();
        rotate_loops(&mut blocks, MachineBlockId(1));
        assert_eq!(blocks, before);
    }
}
