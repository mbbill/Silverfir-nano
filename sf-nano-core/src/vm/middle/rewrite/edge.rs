//! Cross-block cached-local repair.
//!
//! Stack and SSA edge shape are already handled by the semantic CFG plus block
//! params. The only repair left here is cached-local set reconciliation.

use alloc::vec::Vec;

use crate::vm::middle::{
    cfg::CfgBlockId,
    frame::FrameSlot,
    joint_plan::{EdgeRepairQuery, JointPlanner},
    ssa_ir::{
        ir::{
            SsaBinding, SsaBlock, SsaEdge, SsaInst, SsaInstKind, SsaProgram, SsaTerminator,
            SsaValue,
        },
        target::SsaTarget,
    },
};

pub(super) fn insert_boundary_repair_blocks(
    program: &mut SsaProgram,
    exit_cached_slots: &[Vec<FrameSlot>],
    planner: &JointPlanner,
) {
    let original_len = program.blocks.len();
    let target_params = program
        .blocks
        .iter()
        .map(|block| block.params.clone())
        .collect::<Vec<_>>();
    let target_entries = program.block_entry_cached_slots.clone();
    let mut extra_blocks = Vec::new();

    for block_index in 0..original_len {
        let pred_exit = exit_cached_slots
            .get(block_index)
            .cloned()
            .unwrap_or_default();
        let terminator = &mut program.blocks[block_index].terminator;
        match terminator {
            SsaTerminator::Goto(edge) => maybe_repair_edge(
                edge,
                &pred_exit,
                &target_entries,
                &target_params,
                &mut extra_blocks,
                &mut program.block_entry_cached_slots,
                original_len,
                planner,
            ),
            SsaTerminator::Branch {
                then_edge,
                else_edge,
                ..
            } => {
                maybe_repair_edge(
                    then_edge,
                    &pred_exit,
                    &target_entries,
                    &target_params,
                    &mut extra_blocks,
                    &mut program.block_entry_cached_slots,
                    original_len,
                    planner,
                );
                maybe_repair_edge(
                    else_edge,
                    &pred_exit,
                    &target_entries,
                    &target_params,
                    &mut extra_blocks,
                    &mut program.block_entry_cached_slots,
                    original_len,
                    planner,
                );
            }
            SsaTerminator::BrTable { entries, .. } => {
                for edge in entries {
                    maybe_repair_edge(
                        edge,
                        &pred_exit,
                        &target_entries,
                        &target_params,
                        &mut extra_blocks,
                        &mut program.block_entry_cached_slots,
                        original_len,
                        planner,
                    );
                }
            }
            SsaTerminator::Return { .. } | SsaTerminator::TrapUnreachable => {}
        }
    }

    maybe_repair_entry(
        program,
        &target_entries,
        &target_params,
        &mut extra_blocks,
        original_len,
        planner,
    );
    program.blocks.extend(extra_blocks);
}

fn maybe_repair_edge(
    edge: &mut SsaEdge,
    pred_exit: &[FrameSlot],
    target_entries: &[Vec<FrameSlot>],
    target_params: &[Vec<SsaValue>],
    extra_blocks: &mut Vec<SsaBlock>,
    block_entry_cached_slots: &mut Vec<Vec<FrameSlot>>,
    original_len: usize,
    planner: &JointPlanner,
) {
    let target_id = edge.target.as_usize();
    let target_entry = target_entries.get(target_id).cloned().unwrap_or_default();
    let repair = planner.edge_repair(EdgeRepairQuery {
        succ_block: (target_id < original_len).then_some(CfgBlockId(target_id as u32)),
        pred_exit,
        succ_entry: &target_entry,
    });
    if repair.ensure_cached_locals.is_empty()
        && repair.reserve_cached_locals.is_empty()
        && repair.drop_cached_locals.is_empty()
    {
        return;
    }

    let repair_id = SsaTarget((original_len + extra_blocks.len()) as u32);
    let repair_params = target_params.get(target_id).cloned().unwrap_or_default();
    let mut ops = Vec::new();
    for slot in repair.drop_cached_locals {
        ops.push(SsaInst {
            kind: SsaInstKind::LocalDropCache { slot },
        });
    }
    for slot in repair.ensure_cached_locals {
        ops.push(SsaInst {
            kind: SsaInstKind::LocalEnsureCache { slot },
        });
    }
    for slot in repair.reserve_cached_locals {
        ops.push(SsaInst {
            kind: SsaInstKind::LocalReserveCache { slot },
        });
    }
    let repair_edge = SsaEdge {
        target: edge.target,
        bindings: repair_params
            .iter()
            .copied()
            .map(|param| SsaBinding {
                param,
                value: param,
            })
            .collect(),
    };
    extra_blocks.push(SsaBlock {
        id: repair_id,
        params: repair_params,
        ops,
        terminator: SsaTerminator::Goto(repair_edge),
    });
    block_entry_cached_slots.push(pred_exit.to_vec());
    edge.target = repair_id;
}

fn maybe_repair_entry(
    program: &mut SsaProgram,
    target_entries: &[Vec<FrameSlot>],
    target_params: &[Vec<SsaValue>],
    extra_blocks: &mut Vec<SsaBlock>,
    original_len: usize,
    planner: &JointPlanner,
) {
    let entry_target = program.entry.as_usize();
    let entry_cached = target_entries
        .get(entry_target)
        .cloned()
        .unwrap_or_default();
    let repair = planner.edge_repair(EdgeRepairQuery {
        succ_block: (entry_target < original_len).then_some(CfgBlockId(entry_target as u32)),
        pred_exit: &[],
        succ_entry: &entry_cached,
    });
    if repair.ensure_cached_locals.is_empty()
        && repair.reserve_cached_locals.is_empty()
        && repair.drop_cached_locals.is_empty()
    {
        return;
    }

    let repair_id = SsaTarget((original_len + extra_blocks.len()) as u32);
    let repair_params = target_params.get(entry_target).cloned().unwrap_or_default();
    let mut ops = Vec::new();
    for slot in repair.drop_cached_locals {
        ops.push(SsaInst {
            kind: SsaInstKind::LocalDropCache { slot },
        });
    }
    for slot in repair.ensure_cached_locals {
        ops.push(SsaInst {
            kind: SsaInstKind::LocalEnsureCache { slot },
        });
    }
    for slot in repair.reserve_cached_locals {
        ops.push(SsaInst {
            kind: SsaInstKind::LocalReserveCache { slot },
        });
    }
    let repair_edge = SsaEdge {
        target: program.entry,
        bindings: repair_params
            .iter()
            .copied()
            .map(|param| SsaBinding {
                param,
                value: param,
            })
            .collect(),
    };
    extra_blocks.push(SsaBlock {
        id: repair_id,
        params: repair_params,
        ops,
        terminator: SsaTerminator::Goto(repair_edge),
    });
    program.block_entry_cached_slots.push(Vec::new());
    program.entry = repair_id;
}
