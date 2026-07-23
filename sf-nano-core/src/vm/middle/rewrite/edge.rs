//! Cross-block cached-local repair.
//!
//! Stack and SSA edge shape are already handled by the semantic CFG plus block
//! params. The only repair left here is cached-local set reconciliation.

use crate::collections;
use tracked_alloc::collections::BTreeMap;

use crate::vm::middle::{
    cell::CellId,
    joint_plan::facts::{self, RepairActions, RepairActionsSpans, RowSpan, NO_REPAIR},
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
    pred_exit: collections::Vec<CellId>,
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

/// Optional debug-plan edge-repair data. Release rewrite passes empty slices
/// and derives every repair from emitted SSA; the indexed representation stays
/// available to cross-check the debug oracle.
#[derive(Clone, Copy)]
pub(super) struct PlanRepairs<'a> {
    pub pool: &'a [RepairActionsSpans],
    pub slot_arena: &'a [CellId],
    pub index_arena: &'a [u32],
    pub spans: &'a [RowSpan],
}

pub(super) fn insert_boundary_repair_blocks(
    program: &mut SsaProgram,
    exit_cached_slots: &[collections::Vec<CellId>],
    plan: PlanRepairs<'_>,
    semantic_count: usize,
) {
    let original_len = program.blocks.len();
    let repair_count = count_boundary_repairs(
        program,
        exit_cached_slots,
        plan,
        semantic_count,
        original_len,
    );
    program.blocks.reserve_exact(repair_count);
    program.block_entry_cached_cells.reserve_exact(repair_count);

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
        collect_edge_slots(&program.blocks[block_index].terminator, &mut edge_slots);

        for &(slot, original_target) in &edge_slots {
            let repair = resolve_edge_repair(
                program,
                plan,
                semantic_count,
                block_index,
                edge_slot_ordinal(slot),
                original_target,
                pred_exit,
            );
            if repair.is_empty() {
                continue;
            }
            if let Some(repair_id) = materialize_repair_block(
                program,
                original_target,
                pred_exit,
                repair,
                &mut repair_blocks,
            ) {
                retarget_edge(&mut program.blocks[block_index].terminator, slot, repair_id);
            }
        }
    }

    maybe_repair_entry(program);
}

/// Enumerate a terminator's outgoing edges in the fixed edge order
/// (Goto | BranchThen, BranchElse | BrTable(idx)) — the same order pass D keys
/// its per-edge repair on.
fn collect_edge_slots(
    terminator: &SsaTerminator,
    edge_slots: &mut collections::Vec<(EdgeSlot, SsaTarget)>,
) {
    match terminator {
        SsaTerminator::Goto(edge) => edge_slots.push((EdgeSlot::Goto, edge.target)),
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
}

#[inline]
fn edge_slot_ordinal(slot: EdgeSlot) -> usize {
    match slot {
        EdgeSlot::Goto | EdgeSlot::BranchThen => 0,
        EdgeSlot::BranchElse => 1,
        EdgeSlot::BrTable(idx) => idx,
    }
}

/// The repair actions for one outgoing edge. When optional debug-plan data is
/// supplied, semantic-to-semantic edges can read it; release rewrite derives
/// every edge from the target's lowered ops.
fn resolve_edge_repair(
    program: &SsaProgram,
    plan: PlanRepairs<'_>,
    semantic_count: usize,
    block_index: usize,
    ordinal: usize,
    target: SsaTarget,
    pred_exit: &[CellId],
) -> RepairActions {
    if block_index < semantic_count && target.as_usize() < semantic_count {
        let index = plan_edge_index(plan, block_index, ordinal);
        return if index == NO_REPAIR {
            RepairActions::default()
        } else {
            // Rehydrate the flattened pool entry into the transient owned form
            // this half consumes (build_repair_ops + the RepairBlockKey dedup).
            let spans = plan.pool[index as usize];
            RepairActions {
                ensure_cached_cells: plan.slot_arena[spans.ensure.range()].to_vec().into(),
                reserve_cached_cells: plan.slot_arena[spans.reserve.range()].to_vec().into(),
                drop_cached_cells: plan.slot_arena[spans.drop.range()].to_vec().into(),
            }
        };
    }
    let target_id = target.as_usize();
    let target_ops = program
        .blocks
        .get(target_id)
        .map(|b| b.ops.as_slice())
        .unwrap_or(&[]);
    let target_entry = program
        .block_entry_cached_cells
        .get(target_id)
        .map(|s| s.as_slice())
        .unwrap_or(&[]);
    derive_edge_repair(pred_exit, target_entry, target_ops)
}

/// The plan's repair-pool index for semantic block `block_index`'s edge at
/// `ordinal` (`NO_REPAIR` = no repair, or out of range).
#[inline]
fn plan_edge_index(plan: PlanRepairs<'_>, block_index: usize, ordinal: usize) -> u32 {
    plan.spans
        .get(block_index)
        .and_then(|span| plan.index_arena.get(span.offset as usize + ordinal))
        .copied()
        .unwrap_or(NO_REPAIR)
}

fn count_boundary_repairs(
    program: &SsaProgram,
    exit_cached_slots: &[collections::Vec<CellId>],
    plan: PlanRepairs<'_>,
    semantic_count: usize,
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
        collect_edge_slots(&program.blocks[block_index].terminator, &mut edge_slots);

        for &(slot, original_target) in &edge_slots {
            let needed =
                if block_index < semantic_count && original_target.as_usize() < semantic_count {
                    plan_edge_index(plan, block_index, edge_slot_ordinal(slot)) != NO_REPAIR
                } else {
                    edge_repair_needed(program, pred_exit, original_target)
                };
            if needed {
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
    pred_exit: &[CellId],
    original_target: SsaTarget,
) -> bool {
    let target_id = original_target.as_usize();
    let target_block = program.blocks.get(target_id);
    let target_ops = target_block.map(|b| b.ops.as_slice()).unwrap_or(&[]);
    let target_entry = program
        .block_entry_cached_cells
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

/// Materialize (or dedup to an existing) repair block for `repair` on the edge
/// into `original_target`, returning the repair block's id. `None` when the
/// repair is empty. Repair derivation now happens up front in
/// [`resolve_edge_repair`]; this half owns only block synthesis, the
/// identity-forwarding param bindings, and the `RepairBlockKey` dedup.
fn materialize_repair_block(
    program: &mut SsaProgram,
    original_target: SsaTarget,
    pred_exit: &[CellId],
    repair: RepairActions,
    repair_blocks: &mut BTreeMap<RepairBlockKey, SsaTarget>,
) -> Option<SsaTarget> {
    if repair.is_empty() {
        return None;
    }
    let target_id = original_target.as_usize();
    let repair_params = program
        .blocks
        .get(target_id)
        .map(|b| b.params.clone())
        .unwrap_or_default();

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
        .block_entry_cached_cells
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
            .block_entry_cached_cells
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
        .block_entry_cached_cells
        .push(collections::Vec::new());
    program.entry = repair_id;
}

fn build_repair_ops(repair: &RepairActions) -> collections::Vec<SsaInst> {
    let mut ops = collections::Vec::with_capacity(
        repair.drop_cached_cells.len()
            + repair.ensure_cached_cells.len()
            + repair.reserve_cached_cells.len(),
    );
    for &slot in &repair.drop_cached_cells {
        ops.push(SsaInst::cell_drop_cache(slot));
    }
    for &slot in &repair.ensure_cached_cells {
        ops.push(SsaInst::cell_ensure_cache(slot));
    }
    for &slot in &repair.reserve_cached_cells {
        ops.push(SsaInst::cell_reserve_cache(slot));
    }
    ops
}

/// Emit-side edge-repair derivation: reads the successor's first-use
/// requirement straight from its lowered ops. Kept for synthesized bridge-block
/// edges (which never enter pass D's plan) and shared with pass D's action
/// computation through [`facts::derive_edge_repair`] so the logic exists once.
pub(super) fn derive_edge_repair(
    pred_exit: &[CellId],
    succ_entry: &[CellId],
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
        let slot0 = CellId(0);
        let slot1 = CellId(1);
        let index = SsaValue(0);
        let cached = SsaValue(1);

        let mut program = SsaProgram {
            cell_homes: collections::Vec::new(),
            entry: SsaTarget(0),
            blocks: collections::Vec::new(),
            cell_types: collections::vec![ValueType::I32, ValueType::I32],
            result_types: collections::Vec::new(),
            cell_info: collections::vec![Default::default(), Default::default()],
            block_entry_cached_cells: collections::vec![
                collections::Vec::new(),
                collections::vec![slot0]
            ],
            // Requirement rows are intentionally absent until final SSA.
            block_entry_cache_requirements: collections::Vec::new(),
            preferred_preserved: collections::Vec::new(),
            value_types: collections::vec![ValueType::I32, ValueType::I32],
            value_sink_cell: collections::vec![None, None],
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
                ops: collections::vec![SsaInst::cell_get_cache(slot0, cached)],
                extra_args: collections::Vec::new(),
                terminator: SsaTerminator::Return { results: None },
            },
        ];
        let exit_cached_slots =
            collections::vec![collections::vec![slot1], collections::vec![slot0]];

        // semantic_count = 0 routes every edge through the emit-side derive
        // path this test exercises.
        insert_boundary_repair_blocks(
            &mut program,
            &exit_cached_slots,
            PlanRepairs {
                pool: &[],
                slot_arena: &[],
                index_arena: &[],
                spans: &[],
            },
            0,
        );

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
            program.block_entry_cached_cells[repair_target.as_usize()],
            collections::vec![slot1]
        );
        assert_eq!(
            repair_block
                .ops
                .iter()
                .map(|inst| (inst.op, CellId(inst.meta)))
                .collect::<collections::Vec<_>>(),
            collections::vec![
                (SsaOp::CELL_DROP_CACHE, slot1),
                (SsaOp::CELL_ENSURE_CACHE, slot0)
            ]
        );
        assert!(matches!(
            &repair_block.terminator,
            SsaTerminator::Goto(edge) if edge.target == SsaTarget(1)
        ));
        validate_program(&program).unwrap();
    }
}
