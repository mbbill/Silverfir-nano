//! Cross-block cached-local repair.
//!
//! Stack and SSA edge shape are already handled by the semantic CFG plus block
//! params. The only repair left here is cached-local set reconciliation.

use crate::collections;

use crate::vm::middle::{
    frame::FrameSlot,
    ssa_ir::{
        ir::{
            entry_cache_requirement, EntryCacheRequirement, SsaBinding, SsaBlock, SsaEdge, SsaInst,
            SsaProgram, SsaTerminator,
        },
        target::SsaTarget,
    },
};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct RepairActions {
    ensure_cached_locals: collections::Vec<FrameSlot>,
    reserve_cached_locals: collections::Vec<FrameSlot>,
    drop_cached_locals: collections::Vec<FrameSlot>,
}

impl RepairActions {
    fn is_empty(&self) -> bool {
        self.ensure_cached_locals.is_empty()
            && self.reserve_cached_locals.is_empty()
            && self.drop_cached_locals.is_empty()
    }
}

/// Identifies which outgoing edge of a terminator a repair applies to.
#[derive(Clone, Copy, Debug)]
enum EdgeSlot {
    Goto,
    BranchThen,
    BranchElse,
    BrTable(usize),
}

pub(super) fn insert_boundary_repair_blocks(
    program: &mut SsaProgram,
    exit_cached_slots: &[collections::Vec<FrameSlot>],
) {
    let original_len = program.blocks.len();

    // Scratch reused across blocks for edge enumeration; avoids allocating a
    // fresh Vec per block for the common 1- and 2-edge cases.
    let mut edge_slots: collections::Vec<(EdgeSlot, SsaTarget)> = collections::Vec::new();

    for block_index in 0..original_len {
        let pred_exit = exit_cached_slots
            .get(block_index)
            .cloned()
            .unwrap_or_default();

        edge_slots.clear();
        match &program.blocks[block_index].terminator {
            SsaTerminator::Goto(edge) => {
                edge_slots.push((EdgeSlot::Goto, edge.target));
            }
            SsaTerminator::Branch {
                then_edge,
                else_edge,
                ..
            } => {
                edge_slots.push((EdgeSlot::BranchThen, then_edge.target));
                edge_slots.push((EdgeSlot::BranchElse, else_edge.target));
            }
            SsaTerminator::BrTable { entries, .. } => {
                for (idx, edge) in entries.iter().enumerate() {
                    edge_slots.push((EdgeSlot::BrTable(idx), edge.target));
                }
            }
            SsaTerminator::Return { .. }
            | SsaTerminator::TailCallDirect { .. }
            | SsaTerminator::TailCallIndirect { .. }
            | SsaTerminator::TailCallRef { .. }
            | SsaTerminator::TrapUnreachable
            | SsaTerminator::EhThrow { .. }
            | SsaTerminator::EhThrowRef { .. } => {}
        }

        for i in 0..edge_slots.len() {
            let (slot, original_target) = edge_slots[i];
            if let Some(repair_id) = apply_edge_repair(program, &pred_exit, original_target) {
                retarget_edge(&mut program.blocks[block_index].terminator, slot, repair_id);
            }
        }
    }

    maybe_repair_entry(program);
}

/// Compute the repair for one outgoing edge (reads `program` immutably for
/// the target's ops / params) and, if a repair is needed, push the repair
/// block onto `program` and return its id.
fn apply_edge_repair(
    program: &mut SsaProgram,
    pred_exit: &[FrameSlot],
    original_target: SsaTarget,
) -> Option<SsaTarget> {
    let target_id = original_target.as_usize();

    // Read-only inspection of the original target block.
    let (repair, repair_params) = {
        let target_block = program.blocks.get(target_id);
        let target_ops = target_block.map(|b| b.ops.as_slice()).unwrap_or(&[]);
        let target_entry = program
            .block_entry_cached_slots
            .get(target_id)
            .map(|s| s.as_slice())
            .unwrap_or(&[]);
        let repair = derive_edge_repair(pred_exit, target_entry, target_ops);
        if repair.is_empty() {
            return None;
        }
        let repair_params = target_block.map(|b| b.params.clone()).unwrap_or_default();
        (repair, repair_params)
    };

    let repair_id = SsaTarget(program.blocks.len() as u32);
    let ops = build_repair_ops(&repair);
    let repair_edge = SsaEdge {
        target: original_target,
        bindings: repair_params
            .iter()
            .copied()
            .map(|param| SsaBinding {
                param,
                value: param,
            })
            .collect(),
    };
    program.blocks.push(SsaBlock {
        id: repair_id,
        params: repair_params,
        ops,
        extra_args: collections::Vec::new(),
        terminator: SsaTerminator::Goto(repair_edge),
    });
    program
        .block_entry_cached_slots
        .push(pred_exit.to_vec().into());
    if !program.block_cfg_origins.is_empty() {
        program.block_cfg_origins.push(collections::Vec::new());
    }
    Some(repair_id)
}

fn retarget_edge(terminator: &mut SsaTerminator, slot: EdgeSlot, repair_id: SsaTarget) {
    match (terminator, slot) {
        (SsaTerminator::Goto(edge), EdgeSlot::Goto) => edge.target = repair_id,
        (SsaTerminator::Branch { then_edge, .. }, EdgeSlot::BranchThen) => {
            then_edge.target = repair_id
        }
        (SsaTerminator::Branch { else_edge, .. }, EdgeSlot::BranchElse) => {
            else_edge.target = repair_id
        }
        (SsaTerminator::BrTable { entries, .. }, EdgeSlot::BrTable(idx)) => {
            if let Some(edge) = entries.get_mut(idx) {
                edge.target = repair_id;
            }
        }
        _ => debug_assert!(
            false,
            "retarget_edge: terminator shape changed between enumeration and mutation"
        ),
    }
}

fn maybe_repair_entry(program: &mut SsaProgram) {
    let entry_target = program.entry;
    let entry_id = entry_target.as_usize();

    let (repair, repair_params) = {
        let target_block = program.blocks.get(entry_id);
        let target_ops = target_block.map(|b| b.ops.as_slice()).unwrap_or(&[]);
        let target_entry = program
            .block_entry_cached_slots
            .get(entry_id)
            .map(|s| s.as_slice())
            .unwrap_or(&[]);
        let repair = derive_edge_repair(&[], target_entry, target_ops);
        if repair.is_empty() {
            return;
        }
        let repair_params = target_block.map(|b| b.params.clone()).unwrap_or_default();
        (repair, repair_params)
    };

    let repair_id = SsaTarget(program.blocks.len() as u32);
    let ops = build_repair_ops(&repair);
    let repair_edge = SsaEdge {
        target: entry_target,
        bindings: repair_params
            .iter()
            .copied()
            .map(|param| SsaBinding {
                param,
                value: param,
            })
            .collect(),
    };
    program.blocks.push(SsaBlock {
        id: repair_id,
        params: repair_params,
        ops,
        extra_args: collections::Vec::new(),
        terminator: SsaTerminator::Goto(repair_edge),
    });
    program
        .block_entry_cached_slots
        .push(collections::Vec::new());
    if !program.block_cfg_origins.is_empty() {
        program.block_cfg_origins.push(collections::Vec::new());
    }
    program.entry = repair_id;
}

fn build_repair_ops(repair: &RepairActions) -> collections::Vec<SsaInst> {
    let mut ops = collections::Vec::with_capacity(
        repair.drop_cached_locals.len()
            + repair.ensure_cached_locals.len()
            + repair.reserve_cached_locals.len(),
    );
    for &slot in &repair.drop_cached_locals {
        ops.push(SsaInst::local_drop_cache(slot));
    }
    for &slot in &repair.ensure_cached_locals {
        ops.push(SsaInst::local_ensure_cache(slot));
    }
    for &slot in &repair.reserve_cached_locals {
        ops.push(SsaInst::local_reserve_cache(slot));
    }
    ops
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
