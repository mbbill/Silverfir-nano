//! Cross-block cached-local repair.
//!
//! Stack and SSA edge shape are already handled by the semantic CFG plus block
//! params. The only repair left here is cached-local set reconciliation.

use alloc::vec::Vec;

use crate::vm::middle::{
    frame::FrameSlot,
    ssa_ir::{
        ir::{
            entry_cache_requirement, EntryCacheRequirement, SsaBinding, SsaBlock, SsaEdge, SsaInst,
            SsaInstKind, SsaProgram, SsaTerminator, SsaValue,
        },
        target::SsaTarget,
    },
};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct RepairActions {
    ensure_cached_locals: Vec<FrameSlot>,
    reserve_cached_locals: Vec<FrameSlot>,
    drop_cached_locals: Vec<FrameSlot>,
}

pub(super) fn insert_boundary_repair_blocks(
    program: &mut SsaProgram,
    exit_cached_slots: &[Vec<FrameSlot>],
) {
    let original_len = program.blocks.len();
    let target_blocks = program.blocks.clone();
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
                &target_blocks,
                &target_entries,
                &target_params,
                &mut extra_blocks,
                &mut program.block_entry_cached_slots,
                &mut program.block_cfg_origins,
                original_len,
            ),
            SsaTerminator::Branch {
                then_edge,
                else_edge,
                ..
            } => {
                maybe_repair_edge(
                    then_edge,
                    &pred_exit,
                    &target_blocks,
                    &target_entries,
                    &target_params,
                    &mut extra_blocks,
                    &mut program.block_entry_cached_slots,
                    &mut program.block_cfg_origins,
                    original_len,
                );
                maybe_repair_edge(
                    else_edge,
                    &pred_exit,
                    &target_blocks,
                    &target_entries,
                    &target_params,
                    &mut extra_blocks,
                    &mut program.block_entry_cached_slots,
                    &mut program.block_cfg_origins,
                    original_len,
                );
            }
            SsaTerminator::BrTable { entries, .. } => {
                for edge in entries {
                    maybe_repair_edge(
                        edge,
                        &pred_exit,
                        &target_blocks,
                        &target_entries,
                        &target_params,
                        &mut extra_blocks,
                        &mut program.block_entry_cached_slots,
                        &mut program.block_cfg_origins,
                        original_len,
                    );
                }
            }
            SsaTerminator::Return { .. } | SsaTerminator::TrapUnreachable => {}
        }
    }

    maybe_repair_entry(
        program,
        &target_blocks,
        &target_entries,
        &target_params,
        &mut extra_blocks,
        original_len,
    );
    program.blocks.extend(extra_blocks);
}

fn maybe_repair_edge(
    edge: &mut SsaEdge,
    pred_exit: &[FrameSlot],
    target_blocks: &[SsaBlock],
    target_entries: &[Vec<FrameSlot>],
    target_params: &[Vec<SsaValue>],
    extra_blocks: &mut Vec<SsaBlock>,
    block_entry_cached_slots: &mut Vec<Vec<FrameSlot>>,
    block_cfg_origins: &mut Vec<Vec<u32>>,
    original_len: usize,
) {
    let target_id = edge.target.as_usize();
    let target_entry = target_entries.get(target_id).cloned().unwrap_or_default();
    let target_ops = target_blocks
        .get(target_id)
        .map(|block| block.ops.as_slice())
        .unwrap_or(&[]);
    let repair = derive_edge_repair(pred_exit, &target_entry, target_ops);
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
    block_cfg_origins.push(Vec::new());
    edge.target = repair_id;
}

fn maybe_repair_entry(
    program: &mut SsaProgram,
    target_blocks: &[SsaBlock],
    target_entries: &[Vec<FrameSlot>],
    target_params: &[Vec<SsaValue>],
    extra_blocks: &mut Vec<SsaBlock>,
    original_len: usize,
) {
    let entry_target = program.entry.as_usize();
    let entry_cached = target_entries
        .get(entry_target)
        .cloned()
        .unwrap_or_default();
    let target_ops = target_blocks
        .get(entry_target)
        .map(|block| block.ops.as_slice())
        .unwrap_or(&[]);
    let repair = derive_edge_repair(&[], &entry_cached, target_ops);
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
    program.block_cfg_origins.push(Vec::new());
    program.entry = repair_id;
}

fn derive_edge_repair(
    pred_exit: &[FrameSlot],
    succ_entry: &[FrameSlot],
    target_ops: &[SsaInst],
) -> RepairActions {
    let mut repair = RepairActions::default();
    for &slot in pred_exit {
        if !succ_entry.contains(&slot) {
            repair.drop_cached_locals.push(slot);
        }
    }
    for &slot in succ_entry {
        if pred_exit.contains(&slot) {
            continue;
        }
        match entry_cache_requirement(target_ops, slot, succ_entry.contains(&slot)) {
            Some(EntryCacheRequirement::Ensure) => repair.ensure_cached_locals.push(slot),
            Some(EntryCacheRequirement::Reserve) => repair.reserve_cached_locals.push(slot),
            None => {}
        }
    }
    repair
}
