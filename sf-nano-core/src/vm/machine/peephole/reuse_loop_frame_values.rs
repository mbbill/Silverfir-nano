//! Carry unchanged frame values through natural loops.
//!
//! Cache planning can deliberately publish a local before a high-pressure
//! loop and reload it on each exit. If later SSA simplification removes that
//! pressure (for example by rematerializing a select literal), the reload's
//! physical lane is free throughout the loop. Carry the already-published
//! value in that lane as a block parameter and turn the dedicated exit reload
//! into an ordinary edge binding.
//!
//! The frame store remains in the preheader, preserving the interpreter/trap
//! contract. This pass only removes loads whose value is proven unchanged.

use crate::collections;
use crate::vm::machine::machine_ir::{
    MachineAddr, MachineBlock, MachineBlockParam, MachineInstKind, MachineLoadExtension,
    MachineMemWidth, MachineReg, MachineRegOwner, MachineStorageType, MachineTerminator,
    MachineValue, MACHINE_FP_REG,
};

use super::helpers::store_may_alias;
use super::hoist_loop_address_bases::{
    block_index_for_id, block_mentions_reg, natural_loop_nodes, visit_edges, visit_edges_mut,
    LoopGraph,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FrameLoad {
    owner: MachineRegOwner,
    ty: MachineStorageType,
    dst: MachineReg,
    addr: MachineAddr,
    width: MachineMemWidth,
    extension: MachineLoadExtension,
}

pub(super) fn reuse_loop_frame_values(blocks: &mut [MachineBlock], loop_graph: &LoopGraph) {
    if blocks.len() < 3 {
        return;
    }

    let latches_by_header = &loop_graph.latches_by_header;
    let predecessors = &loop_graph.predecessors;

    for header in (0..blocks.len()).rev() {
        if latches_by_header[header].is_empty() {
            continue;
        }
        let loop_nodes = natural_loop_nodes(header, &latches_by_header[header], predecessors);
        try_reuse_one_frame_value(blocks, header, &loop_nodes, predecessors);
    }
}

fn try_reuse_one_frame_value(
    blocks: &mut [MachineBlock],
    header: usize,
    loop_nodes: &[usize],
    predecessors: &[collections::Vec<usize>],
) {
    let mut in_loop = collections::vec![false; blocks.len()];
    for &block in loop_nodes {
        in_loop[block] = true;
    }

    let mut exit_targets = collections::Vec::new();
    for &source in loop_nodes {
        visit_edges(&blocks[source].terminator, |edge| {
            let Some(target) = block_index_for_id(blocks, edge.target) else {
                return;
            };
            if !in_loop[target] && !exit_targets.contains(&target) {
                exit_targets.push(target);
            }
        });
    }
    let Some(&first_exit) = exit_targets.first() else {
        return;
    };
    let Some(load) = blocks[first_exit].ops.first().and_then(frame_load) else {
        return;
    };
    if load.addr.base != MACHINE_FP_REG {
        return;
    }

    // Exit blocks must be dedicated to this loop, begin with the identical
    // reload, and not already bind the destination lane.
    for &target in &exit_targets {
        if predecessors[target].iter().any(|&source| !in_loop[source])
            || blocks[target]
                .params
                .iter()
                .any(|param| param.reg == load.dst)
            || blocks[target].ops.first().and_then(frame_load) != Some(load)
        {
            return;
        }
    }

    // The reload destination itself is the ideal carry lane: it introduces no
    // new register pressure after the loop and must be completely free within
    // the loop before we claim it.
    if loop_nodes
        .iter()
        .any(|&block| block_mentions_reg(&blocks[block], load.dst))
        || !loop_preserves_frame_value(blocks, loop_nodes, load)
    {
        return;
    }

    // Most loops do not have a reusable frame load on every dedicated exit.
    // Delay the whole-function entry-edge scan until the cheap, selective exit
    // checks above have found a viable candidate.
    let mut entry_sources = collections::Vec::<(usize, MachineValue)>::new();
    for source in 0..blocks.len() {
        if in_loop[source] {
            continue;
        }
        let mut enters = false;
        let mut only_header = true;
        visit_edges(&blocks[source].terminator, |edge| {
            let Some(target) = block_index_for_id(blocks, edge.target) else {
                return;
            };
            if in_loop[target] {
                enters = true;
                only_header &= target == header;
            }
        });
        if enters {
            if !only_header {
                return;
            }
            // The exact seed is checked after the common exit load is known.
            entry_sources.push((source, MachineValue::Imm64(0)));
        }
    }
    if entry_sources.is_empty() {
        return;
    }

    for (source, seed) in &mut entry_sources {
        let Some(value) = seed_store_value(&blocks[*source], load) else {
            return;
        };
        *seed = value;
    }

    let carry_param = MachineBlockParam {
        reg: load.dst,
        ty: load.ty,
        owner: load.owner,
    };
    for &block in loop_nodes {
        blocks[block].params.push(carry_param);
    }
    for &target in &exit_targets {
        blocks[target].ops.remove(0);
        blocks[target].params.push(carry_param);
    }

    let loop_ids: collections::Vec<_> = loop_nodes.iter().map(|&index| blocks[index].id).collect();
    let exit_ids: collections::Vec<_> =
        exit_targets.iter().map(|&index| blocks[index].id).collect();
    for source in 0..blocks.len() {
        let source_inside = in_loop[source];
        let entry_seed = entry_sources
            .iter()
            .find_map(|(entry, seed)| (*entry == source).then_some(*seed));
        visit_edges_mut(&mut blocks[source].terminator, |edge| {
            if loop_ids.contains(&edge.target) {
                edge.args.push(if source_inside {
                    MachineValue::Reg(load.dst)
                } else {
                    entry_seed.expect("validated natural-loop entry seed")
                });
            } else if source_inside && exit_ids.contains(&edge.target) {
                edge.args.push(MachineValue::Reg(load.dst));
            }
        });
    }
}

fn frame_load(inst: &crate::vm::machine::machine_ir::MachineInst) -> Option<FrameLoad> {
    let MachineInstKind::Load {
        owner,
        ty,
        dst,
        addr,
        width,
        extension,
    } = inst.kind
    else {
        return None;
    };
    Some(FrameLoad {
        owner,
        ty,
        dst,
        addr,
        width,
        extension,
    })
}

fn seed_store_value(block: &MachineBlock, load: FrameLoad) -> Option<MachineValue> {
    let MachineInstKind::Store {
        ty,
        addr,
        width,
        src,
    } = block.ops.last()?.kind
    else {
        return None;
    };
    (ty == load.ty && addr == load.addr && width == load.width).then_some(src)
}

fn loop_preserves_frame_value(
    blocks: &[MachineBlock],
    loop_nodes: &[usize],
    load: FrameLoad,
) -> bool {
    loop_nodes.iter().all(|&index| {
        let block = &blocks[index];
        block.ops.iter().all(|inst| match inst.kind {
            MachineInstKind::Store { addr, width, .. } => {
                !store_may_alias(load.addr, load.width, addr, width)
            }
            MachineInstKind::CallRuntime(_) => false,
            _ => true,
        }) && !matches!(
            block.terminator,
            MachineTerminator::Call { .. } | MachineTerminator::TailCall { .. }
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::machine::machine_ir::{
        MachineBranchCond, MachineEdge, MachineInst, MachineIntBinaryOp, MachineIntWidth,
    };

    fn edge(target: u32, args: &[MachineReg]) -> MachineEdge {
        MachineEdge {
            target: crate::vm::machine::machine_ir::MachineBlockId(target),
            args: args.iter().copied().map(MachineValue::Reg).collect(),
        }
    }

    #[test]
    fn carries_preheader_frame_store_through_multiblock_loop() {
        let source = MachineReg(6);
        let cond = MachineReg(7);
        let carried = MachineReg(5);
        let addr = MachineAddr {
            base: MACHINE_FP_REG,
            offset: 16,
        };
        let mut blocks = collections::vec![
            MachineBlock {
                id: crate::vm::machine::machine_ir::MachineBlockId(0),
                params: collections::vec![
                    MachineBlockParam::gp_i64(source),
                    MachineBlockParam::gp_word(cond),
                ],
                ops: collections::vec![MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpI64,
                        addr,
                        width: MachineMemWidth::U64,
                        src: MachineValue::Reg(source),
                    },
                }],
                terminator: MachineTerminator::Jump(edge(1, &[cond])),
            },
            MachineBlock {
                id: crate::vm::machine::machine_ir::MachineBlockId(1),
                params: collections::vec![MachineBlockParam::gp_word(cond)],
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Branch {
                    cond: MachineBranchCond::Value(MachineValue::Reg(cond)),
                    then_edge: edge(3, &[]),
                    else_edge: edge(2, &[cond]),
                },
            },
            MachineBlock {
                id: crate::vm::machine::machine_ir::MachineBlockId(2),
                params: collections::vec![MachineBlockParam::gp_word(cond)],
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Jump(edge(1, &[cond])),
            },
            MachineBlock {
                id: crate::vm::machine::machine_ir::MachineBlockId(3),
                params: collections::Vec::new(),
                ops: collections::vec![
                    MachineInst {
                        kind: MachineInstKind::Load {
                            owner: MachineRegOwner::CachedCell,
                            ty: MachineStorageType::GpI64,
                            dst: carried,
                            addr,
                            width: MachineMemWidth::U64,
                            extension: MachineLoadExtension::None,
                        },
                    },
                    MachineInst {
                        kind: MachineInstKind::IntBinary {
                            width: MachineIntWidth::I64,
                            op: MachineIntBinaryOp::Add,
                            dst: MachineReg(8),
                            lhs: MachineValue::Reg(carried),
                            rhs: MachineValue::Imm64(1),
                        },
                    },
                ],
                terminator: MachineTerminator::Return,
            },
        ];

        let loop_graph = crate::vm::machine::peephole::hoist_loop_address_bases::analyze_loop_graph(
            &blocks,
            crate::vm::machine::machine_ir::MachineBlockId(0),
        );
        reuse_loop_frame_values(&mut blocks, &loop_graph);

        assert_eq!(
            blocks[1].params.last().map(|param| param.reg),
            Some(carried)
        );
        assert_eq!(
            blocks[2].params.last().map(|param| param.reg),
            Some(carried)
        );
        assert_eq!(
            blocks[3].params.as_slice(),
            &[MachineBlockParam {
                reg: carried,
                ty: MachineStorageType::GpI64,
                owner: MachineRegOwner::CachedCell,
            }]
        );
        assert_eq!(blocks[3].ops.len(), 1, "the exit reload should be gone");

        let MachineTerminator::Jump(entry) = &blocks[0].terminator else {
            panic!("expected loop entry");
        };
        assert_eq!(entry.args.last(), Some(&MachineValue::Reg(source)));
        let MachineTerminator::Jump(backedge) = &blocks[2].terminator else {
            panic!("expected backedge");
        };
        assert_eq!(backedge.args.last(), Some(&MachineValue::Reg(carried)));
        let MachineTerminator::Branch { then_edge, .. } = &blocks[1].terminator else {
            panic!("expected loop branch");
        };
        assert_eq!(then_edge.args.last(), Some(&MachineValue::Reg(carried)));
    }
}
