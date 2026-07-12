//! Cross-block cached-local repair.
//!
//! Stack and SSA edge shape are already handled by the semantic CFG plus block
//! params. The only repair left here is cached-local set reconciliation.

use crate::collections;
use tracked_alloc::collections::BTreeMap;

use crate::vm::middle::{
    frame::FrameSlot,
    joint_plan::facts::{self, RepairActions},
    ssa_ir::{
        ir::{
            entry_cache_requirement, SsaBinding, SsaBlock, SsaEdge, SsaInst, SsaProgram,
            SsaTerminator,
        },
        target::SsaTarget,
    },
};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct RepairBlockKey {
    target: SsaTarget,
    pred_exit: collections::Vec<FrameSlot>,
    repair: RepairActions,
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
    let repair_count = count_boundary_repairs(program, exit_cached_slots, original_len);
    program.blocks.reserve_exact(repair_count);
    program.block_entry_cached_slots.reserve_exact(repair_count);

    // Scratch reused across blocks for edge enumeration; avoids allocating a
    // fresh Vec per block for the common 1- and 2-edge cases.
    let mut edge_slots: collections::Vec<(EdgeSlot, SsaTarget)> = collections::Vec::new();
    let mut repair_blocks: BTreeMap<RepairBlockKey, SsaTarget> = BTreeMap::new();

    for block_index in 0..original_len {
        let pred_exit = exit_cached_slots
            .get(block_index)
            .map(|slots| slots.as_slice())
            .unwrap_or(&[]);

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
            | SsaTerminator::ReturnScalar { .. }
            | SsaTerminator::TailCallDirect { .. }
            | SsaTerminator::TailCallIndirect { .. }
            | SsaTerminator::TailCallRef { .. }
            | SsaTerminator::TrapUnreachable
            | SsaTerminator::EhThrow { .. }
            | SsaTerminator::EhThrowRef { .. } => {}
        }

        for i in 0..edge_slots.len() {
            let (slot, original_target) = edge_slots[i];
            if let Some(repair_id) =
                apply_edge_repair(program, &pred_exit, original_target, &mut repair_blocks)
            {
                retarget_edge(&mut program.blocks[block_index].terminator, slot, repair_id);
            }
        }
    }

    maybe_repair_entry(program);
}

fn count_boundary_repairs(
    program: &SsaProgram,
    exit_cached_slots: &[collections::Vec<FrameSlot>],
    original_len: usize,
) -> usize {
    let mut count = 0;
    let mut edge_slots: collections::Vec<(EdgeSlot, SsaTarget)> = collections::Vec::new();

    for block_index in 0..original_len {
        let pred_exit = exit_cached_slots
            .get(block_index)
            .map(|slots| slots.as_slice())
            .unwrap_or(&[]);

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
            | SsaTerminator::ReturnScalar { .. }
            | SsaTerminator::TailCallDirect { .. }
            | SsaTerminator::TailCallIndirect { .. }
            | SsaTerminator::TailCallRef { .. }
            | SsaTerminator::TrapUnreachable
            | SsaTerminator::EhThrow { .. }
            | SsaTerminator::EhThrowRef { .. } => {}
        }

        for i in 0..edge_slots.len() {
            let (_, original_target) = edge_slots[i];
            if edge_repair_needed(program, pred_exit, original_target) {
                count += 1;
            }
        }
    }

    if edge_repair_needed(program, &[], program.entry) {
        count += 1;
    }

    count
}

fn edge_repair_needed(
    program: &SsaProgram,
    pred_exit: &[FrameSlot],
    original_target: SsaTarget,
) -> bool {
    let target_id = original_target.as_usize();
    let target_block = program.blocks.get(target_id);
    let target_ops = target_block.map(|b| b.ops.as_slice()).unwrap_or(&[]);
    let target_entry = program
        .block_entry_cached_slots
        .get(target_id)
        .map(|s| s.as_slice())
        .unwrap_or(&[]);

    for &slot in pred_exit {
        if !target_entry.contains(&slot) {
            return true;
        }
    }
    for &slot in target_entry {
        if pred_exit.contains(&slot) {
            continue;
        }
        if entry_cache_requirement(target_ops, slot, target_entry.contains(&slot)).is_some() {
            return true;
        }
    }
    false
}

/// Compute the repair for one outgoing edge (reads `program` immutably for
/// the target's ops / params) and, if a repair is needed, push the repair
/// block onto `program` and return its id.
fn apply_edge_repair(
    program: &mut SsaProgram,
    pred_exit: &[FrameSlot],
    original_target: SsaTarget,
    repair_blocks: &mut BTreeMap<RepairBlockKey, SsaTarget>,
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

    let key = RepairBlockKey {
        target: original_target,
        pred_exit: pred_exit.to_vec().into(),
        repair: repair.clone(),
    };
    if let Some(&repair_id) = repair_blocks.get(&key) {
        return Some(repair_id);
    }

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
    repair_blocks.insert(key, repair_id);
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

/// Emit-side edge-repair derivation: reads the successor's first-use
/// requirement straight from its lowered ops. Kept for synthesized bridge-block
/// edges (which never enter pass D's plan) and shared with pass D's action
/// computation through [`facts::derive_edge_repair`] so the logic exists once.
pub(super) fn derive_edge_repair(
    pred_exit: &[FrameSlot],
    succ_entry: &[FrameSlot],
    target_ops: &[SsaInst],
) -> RepairActions {
    facts::derive_edge_repair(pred_exit, succ_entry, |slot| {
        entry_cache_requirement(target_ops, slot, succ_entry.contains(&slot))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value_type::ValueType;
    use crate::vm::middle::ssa_ir::{
        ir::{SsaOp, SsaOperand, SsaValue},
        validate::validate_program,
    };
    use crate::vm::wasm::primitive_op::PrimitiveOpKind;

    #[test]
    fn br_table_edges_share_identical_boundary_repair_block() {
        let slot0 = FrameSlot(0);
        let slot1 = FrameSlot(1);
        let index = SsaValue(0);
        let cached = SsaValue(1);

        let mut program = SsaProgram {
            entry: SsaTarget(0),
            blocks: collections::Vec::new(),
            local_slot_types: collections::vec![ValueType::I32, ValueType::I32],
            result_types: collections::Vec::new(),
            local_slot_info: collections::vec![Default::default(), Default::default()],
            block_entry_cached_slots: collections::vec![
                collections::Vec::new(),
                collections::vec![slot0]
            ],
            value_types: collections::vec![ValueType::I32, ValueType::I32],
            value_sink_local: collections::vec![None, None],
            const_pool: collections::Vec::new(),
            primitive_pool: collections::Vec::new(),
            call_ops: collections::Vec::new(),
        };
        let pool_idx = program
            .intern_primitive(PrimitiveOpKind::I32Const { value: 0 })
            .unwrap();
        let index_inst =
            SsaInst::primitive(pool_idx, index, [SsaOperand::NONE, SsaOperand::NONE], 0);
        let to_target = SsaEdge {
            target: SsaTarget(1),
            bindings: collections::Vec::new(),
        };
        program.blocks = collections::vec![
            SsaBlock {
                id: SsaTarget(0),
                params: collections::Vec::new(),
                ops: collections::vec![index_inst],
                extra_args: collections::Vec::new(),
                terminator: SsaTerminator::BrTable {
                    index,
                    entries: collections::vec![to_target.clone(), to_target.clone(), to_target],
                },
            },
            SsaBlock {
                id: SsaTarget(1),
                params: collections::Vec::new(),
                ops: collections::vec![SsaInst::local_get_cache(slot0, cached)],
                extra_args: collections::Vec::new(),
                terminator: SsaTerminator::Return { results: None },
            },
        ];
        let exit_cached_slots =
            collections::vec![collections::vec![slot1], collections::vec![slot0]];

        insert_boundary_repair_blocks(&mut program, &exit_cached_slots);

        assert_eq!(
            program.blocks.len(),
            3,
            "repeated br_table edges should be retargeted to one shared repair block"
        );
        let repair_target = match &program.blocks[0].terminator {
            SsaTerminator::BrTable { entries, .. } => {
                assert!(entries.iter().all(|edge| edge.target == entries[0].target));
                entries[0].target
            }
            other => panic!("expected br_table, got {other:?}"),
        };
        assert_eq!(repair_target, SsaTarget(2));
        let repair_block = &program.blocks[repair_target.as_usize()];
        assert_eq!(
            program.block_entry_cached_slots[repair_target.as_usize()],
            collections::vec![slot1]
        );
        assert_eq!(
            repair_block
                .ops
                .iter()
                .map(|inst| (inst.op, FrameSlot(inst.meta)))
                .collect::<collections::Vec<_>>(),
            collections::vec![
                (SsaOp::LOCAL_DROP_CACHE, slot1),
                (SsaOp::LOCAL_ENSURE_CACHE, slot0)
            ]
        );
        assert!(matches!(
            &repair_block.terminator,
            SsaTerminator::Goto(edge) if edge.target == SsaTarget(1)
        ));
        validate_program(&program).unwrap();
    }
}
