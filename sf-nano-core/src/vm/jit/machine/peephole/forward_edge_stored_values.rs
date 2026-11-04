//! Store-to-load forwarding across a single-predecessor edge.
//!
//! A frame-homed local written at the end of one block and read at the
//! start of its only successor goes through memory:
//!
//! ```text
//! b7:  r5 <- ...; store [fp+8] <- r5; branch ... then b10 [r4] else b8 [r4]
//! b10: load r5 <- [fp+8]; jump_table r5 [...]
//! ```
//!
//! The block-local pass cannot see the pair. When the successor has exactly
//! one predecessor, and that predecessor leaves through a plain jump, branch
//! or jump table (no call clobbers registers on the edge), the register the
//! predecessor stored still holds the value when the successor starts. A
//! reload into that same register disappears, and the register travels
//! along the edge as an explicit block parameter. The frame store stays: traps and other blocks still read
//! the published value.
//!
//! Only frame slots are carried. Which stores are still valid at the end of
//! the predecessor, and which instructions in the successor invalidate
//! them, follow exactly the block-local pass's rules
//! (`track_store_effects`).

use crate::collections;

use crate::vm::jit::machine::machine_ir::{
    MachineAddr, MachineBlock, MachineBlockId, MachineBlockParam, MachineInstKind,
    MachineLoadExtension, MachineReg, MachineRegOwner, MachineTerminator, MachineValue,
    MACHINE_FIXED_REG_COUNT, MACHINE_FP_REG,
};

use super::forward_stored_values::track_store_effects;
use super::hoist_loop_address_bases::{block_index_for_id, visit_edges, visit_edges_mut};
use super::TrackedStore;

/// Reloads are looked for among a successor's first few instructions only:
/// across the benchmark modules almost every forwardable reload opens the
/// block (about 95% at the first two positions), and bounding the scan keeps
/// the pass cheap in large functions.
const RELOAD_WINDOW: usize = 4;

pub(super) fn forward_stored_values_across_edges(
    blocks: &mut [MachineBlock],
    entry: MachineBlockId,
) {
    let only_predecessor = only_predecessors(blocks);
    // Stores still valid at the end of each predecessor, computed once: a
    // jump-table block can be the only predecessor of many successors.
    let mut end_states: collections::Vec<Option<collections::Vec<TrackedStore>>> =
        collections::vec![None; blocks.len()];
    let mut tracked = collections::Vec::new();
    for successor in 0..blocks.len() {
        let Some(predecessor) = only_predecessor[successor] else {
            continue;
        };
        if predecessor == successor || blocks[successor].id == entry {
            continue;
        }
        if !matches!(
            blocks[predecessor].terminator,
            MachineTerminator::Jump(_)
                | MachineTerminator::Branch { .. }
                | MachineTerminator::JumpTable { .. }
        ) {
            continue;
        }
        let window = blocks[successor].ops.len().min(RELOAD_WINDOW);
        let reloads_frame = blocks[successor].ops[..window].iter().any(|inst| {
            matches!(
                inst.kind,
                MachineInstKind::Load {
                    addr: MachineAddr {
                        base: MACHINE_FP_REG,
                        ..
                    },
                    extension: MachineLoadExtension::None,
                    ..
                }
            )
        });
        if !reloads_frame {
            continue;
        }

        let end_state = end_states[predecessor].get_or_insert_with(|| {
            let mut state = collections::Vec::new();
            for inst in &blocks[predecessor].ops {
                track_store_effects(&inst.kind, &mut state);
            }
            state.retain(|entry| {
                entry.addr.base == MACHINE_FP_REG
                    && matches!(entry.src, MachineValue::Reg(reg) if reg.0 >= MACHINE_FIXED_REG_COUNT)
            });
            state
        });
        tracked.clear();
        tracked.extend(end_state.iter().copied().filter(|entry| match entry.src {
            MachineValue::Reg(reg) => {
                edges_preserve_reg(&blocks[predecessor], &blocks[successor], reg)
            }
            MachineValue::Imm64(_) | MachineValue::ReservedReg(_) => false,
        }));
        if tracked.is_empty() {
            continue;
        }

        let successor_id = blocks[successor].id;
        let block = &mut blocks[successor];
        let mut carried: collections::Vec<MachineBlockParam> = collections::Vec::new();
        let mut removed = [false; RELOAD_WINDOW];
        for (index, inst) in block.ops[..window].iter().enumerate() {
            if tracked.is_empty() {
                break;
            }
            if let MachineInstKind::Load {
                dst,
                addr,
                width,
                extension: MachineLoadExtension::None,
                ..
            } = inst.kind
            {
                // Only a reload into the very register that was stored goes
                // away outright. Rewriting a reload into another register as
                // a move would trade a load the backend could fold into an
                // ALU memory operand for a separate instruction.
                if let Some(entry) = tracked.iter().rev().find(|entry| {
                    entry.addr == addr
                        && entry.width == width
                        && entry.src == MachineValue::Reg(dst)
                }) {
                    if !carried.iter().any(|param| param.reg == dst) {
                        carried.push(MachineBlockParam {
                            reg: dst,
                            ty: entry.ty,
                            owner: MachineRegOwner::LinearValue,
                        });
                    }
                    // `dst` keeps the stored value, so the entry stays valid
                    // for later reloads in this block.
                    removed[index] = true;
                    continue;
                }
            }
            track_store_effects(&inst.kind, &mut tracked);
        }
        if carried.is_empty() {
            continue;
        }
        for index in (0..window).rev() {
            if removed[index] {
                block.ops.remove(index);
            }
        }
        carried.retain(|param| {
            !block
                .params
                .iter()
                .any(|existing| existing.reg == param.reg)
        });
        if carried.is_empty() {
            continue;
        }
        block.params.extend(carried.iter().copied());
        visit_edges_mut(&mut blocks[predecessor].terminator, |edge| {
            if edge.target == successor_id {
                edge.args
                    .extend(carried.iter().map(|param| MachineValue::Reg(param.reg)));
            }
        });
    }
}

/// The predecessor of each block that has exactly one, from the current
/// terminators. Earlier whole-program passes can add blocks (memmove
/// recognition does), so a loop graph built before them may be stale.
fn only_predecessors(blocks: &[MachineBlock]) -> collections::Vec<Option<usize>> {
    #[derive(Clone, Copy)]
    enum Incoming {
        None,
        One(usize),
        Many,
    }
    let mut incoming = collections::vec![Incoming::None; blocks.len()];
    for (source, block) in blocks.iter().enumerate() {
        visit_edges(&block.terminator, |edge| {
            let Some(target) = block_index_for_id(blocks, edge.target) else {
                return;
            };
            incoming[target] = match incoming[target] {
                Incoming::None => Incoming::One(source),
                Incoming::One(existing) if existing == source => Incoming::One(source),
                Incoming::One(_) | Incoming::Many => Incoming::Many,
            };
        });
    }
    incoming
        .into_iter()
        .map(|state| match state {
            Incoming::One(source) => Some(source),
            Incoming::None | Incoming::Many => None,
        })
        .collect()
}

/// `reg` reaches the successor unchanged: no edge from the predecessor binds
/// a successor parameter in `reg` to a different value.
fn edges_preserve_reg(
    predecessor: &MachineBlock,
    successor: &MachineBlock,
    reg: MachineReg,
) -> bool {
    let Some(position) = successor.params.iter().position(|param| param.reg == reg) else {
        return true;
    };
    let mut preserved = true;
    visit_edges(&predecessor.terminator, |edge| {
        if edge.target == successor.id && edge.args.get(position) != Some(&MachineValue::Reg(reg)) {
            preserved = false;
        }
    });
    preserved
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::jit::machine::machine_ir::{
        MachineAddr, MachineBranchCond, MachineEdge, MachineInst, MachineIntBinaryOp,
        MachineIntWidth, MachineMemWidth, MachineStorageType,
    };

    const STORED: MachineReg = MachineReg(5);
    const OTHER: MachineReg = MachineReg(6);
    const COND: MachineReg = MachineReg(7);
    const SLOT: MachineAddr = MachineAddr {
        base: MACHINE_FP_REG,
        offset: 8,
    };

    fn edge(target: u32, args: collections::Vec<MachineValue>) -> MachineEdge {
        MachineEdge {
            target: MachineBlockId(target),
            args,
        }
    }

    fn inst(kind: MachineInstKind) -> MachineInst {
        MachineInst { kind }
    }

    fn store(src: MachineReg) -> MachineInst {
        inst(MachineInstKind::Store {
            ty: MachineStorageType::GpWord,
            addr: SLOT,
            width: MachineMemWidth::U64,
            src: MachineValue::Reg(src),
        })
    }

    fn reload(dst: MachineReg) -> MachineInst {
        inst(MachineInstKind::Load {
            owner: MachineRegOwner::LinearValue,
            ty: MachineStorageType::GpWord,
            dst,
            addr: SLOT,
            width: MachineMemWidth::U64,
            extension: MachineLoadExtension::None,
        })
    }

    fn increment(dst: MachineReg, src: MachineReg) -> MachineInst {
        inst(MachineInstKind::IntBinary {
            width: MachineIntWidth::I64,
            op: MachineIntBinaryOp::Add,
            dst,
            lhs: MachineValue::Reg(src),
            rhs: MachineValue::Imm64(1),
        })
    }

    /// b0 (params OTHER, COND) runs `pred_ops`, then branches to b1 with
    /// `then_args` or to b2. b1 has `succ_params` and runs `succ_ops`.
    fn program(
        pred_ops: collections::Vec<MachineInst>,
        succ_params: collections::Vec<MachineBlockParam>,
        then_args: collections::Vec<MachineValue>,
        succ_ops: collections::Vec<MachineInst>,
    ) -> collections::Vec<MachineBlock> {
        collections::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: collections::vec![
                    MachineBlockParam::gp_word(OTHER),
                    MachineBlockParam::gp_word(COND),
                ],
                ops: pred_ops,
                terminator: MachineTerminator::Branch {
                    cond: MachineBranchCond::Value(MachineValue::Reg(COND)),
                    then_edge: edge(1, then_args),
                    else_edge: edge(2, collections::Vec::new()),
                },
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: succ_params,
                ops: succ_ops,
                terminator: MachineTerminator::Return,
            },
            MachineBlock {
                id: MachineBlockId(2),
                params: collections::Vec::new(),
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Return,
            },
        ]
    }

    fn run(blocks: &mut [MachineBlock]) {
        forward_stored_values_across_edges(blocks, MachineBlockId(0));
    }

    fn then_args(blocks: &[MachineBlock]) -> &[MachineValue] {
        let MachineTerminator::Branch { then_edge, .. } = &blocks[0].terminator else {
            unreachable!("test predecessor ends in a branch");
        };
        &then_edge.args
    }

    #[test]
    fn reload_of_the_stored_register_becomes_an_edge_parameter() {
        let mut blocks = program(
            collections::vec![increment(STORED, OTHER), store(STORED)],
            collections::Vec::new(),
            collections::Vec::new(),
            collections::vec![reload(STORED), increment(OTHER, STORED)],
        );
        run(&mut blocks);
        assert_eq!(blocks[1].ops, collections::vec![increment(OTHER, STORED)]);
        assert_eq!(
            blocks[1].params,
            collections::vec![MachineBlockParam {
                reg: STORED,
                ty: MachineStorageType::GpWord,
                owner: MachineRegOwner::LinearValue,
            }]
        );
        assert_eq!(then_args(&blocks), &[MachineValue::Reg(STORED)]);
        // The publishing store stays for traps and other readers.
        assert_eq!(blocks[0].ops[1], store(STORED));
    }

    #[test]
    fn register_already_passed_through_gains_no_second_parameter() {
        let mut blocks = program(
            collections::vec![store(STORED)],
            collections::vec![MachineBlockParam::gp_word(STORED)],
            collections::vec![MachineValue::Reg(STORED)],
            collections::vec![reload(STORED), increment(OTHER, STORED)],
        );
        run(&mut blocks);
        assert_eq!(blocks[1].ops, collections::vec![increment(OTHER, STORED)]);
        assert_eq!(blocks[1].params.len(), 1);
        assert_eq!(then_args(&blocks), &[MachineValue::Reg(STORED)]);
    }

    #[test]
    fn unchanged_when_the_value_cannot_be_proven_in_the_register() {
        let cases = [
            // The predecessor overwrites the register after storing it.
            program(
                collections::vec![store(STORED), increment(STORED, OTHER)],
                collections::Vec::new(),
                collections::Vec::new(),
                collections::vec![reload(STORED)],
            ),
            // The edge binds the register to a different value.
            program(
                collections::vec![store(STORED)],
                collections::vec![MachineBlockParam::gp_word(STORED)],
                collections::vec![MachineValue::Reg(OTHER)],
                collections::vec![reload(STORED)],
            ),
            // The successor overwrites the slot before reloading it.
            program(
                collections::vec![store(STORED)],
                collections::Vec::new(),
                collections::Vec::new(),
                collections::vec![store(OTHER), reload(STORED)],
            ),
            // The successor redefines the register before reloading it.
            program(
                collections::vec![store(STORED)],
                collections::Vec::new(),
                collections::Vec::new(),
                collections::vec![increment(STORED, OTHER), reload(STORED)],
            ),
            // A reload into a different register stays a load.
            program(
                collections::vec![store(STORED)],
                collections::Vec::new(),
                collections::Vec::new(),
                collections::vec![reload(OTHER)],
            ),
        ];
        for original in cases {
            let mut blocks = original.clone();
            run(&mut blocks);
            assert_eq!(blocks, original);
        }
    }

    #[test]
    fn unchanged_when_the_successor_has_another_predecessor() {
        let mut original = program(
            collections::vec![store(STORED)],
            collections::Vec::new(),
            collections::Vec::new(),
            collections::vec![reload(STORED)],
        );
        original[2].terminator = MachineTerminator::Jump(edge(1, collections::Vec::new()));
        let mut blocks = original.clone();
        run(&mut blocks);
        assert_eq!(blocks, original);
    }
}
