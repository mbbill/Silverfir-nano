//! Lower planned program into backend-facing LIR.
//!
//! This file must not reintroduce `pre_height` or backend deduction from stack
//! height. Any required frame slot or rotation information should already be
//! explicit by the time it is lowered here.

use alloc::vec::Vec;

use crate::error::WasmError;
use crate::vm::plan::{
    frame::{FrameLayoutPlan, FrameSpan},
    group::{GroupId, GroupRange, GroupTerminator},
    plan::{
        PlannedBrTableEntry, PlannedBranchKind, PlannedMarkerKind, PlannedOp, PlannedOpKind,
        PlannedProgram,
    },
};

use super::ir::{LirBrTableEntry, LirFrameCopy, LirMarkerKind, LirOp, LirOpKind};
use super::target::LirTarget;

/// Lowered LIR program plus shared group mapping.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LirProgram {
    pub ops: alloc::vec::Vec<LirOp>,
    pub groups: alloc::vec::Vec<LirGroup>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LirGroup {
    pub id: GroupId,
    pub lir_range: core::ops::Range<usize>,
    pub terminator: Option<GroupTerminator>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct LoweringMap {
    semantic_to_lir_start: alloc::vec::Vec<usize>,
}

impl LoweringMap {
    fn from_planned_ops(planned_ops: &[PlannedOp]) -> Self {
        let semantic_to_lir_start = planned_ops
            .iter()
            .enumerate()
            .filter_map(|(lir_idx, op)| {
                if is_synthetic_planned_op(&op.kind) {
                    None
                } else {
                    Some(lir_idx)
                }
            })
            .collect();

        Self {
            semantic_to_lir_start,
        }
    }

    fn target_for_semantic_index(&self, semantic_index: usize) -> LirTarget {
        let lir_index = self.semantic_to_lir_start[semantic_index];
        LirTarget(lir_index as u32)
    }
}

pub fn lower_to_lir(planned: PlannedProgram) -> Result<LirProgram, WasmError> {
    let lowering_map = LoweringMap::from_planned_ops(&planned.ops);
    let ops = planned
        .ops
        .iter()
        .map(|op| lower_planned_op(op, &lowering_map, planned.frame))
        .collect::<Vec<_>>();

    Ok(LirProgram {
        groups: map_group_ranges(&planned.groups.groups, &ops),
        ops,
    })
}

/// Map planning-stage groups into LIR ranges.
pub fn map_group_ranges(
    planned_groups: &[GroupRange],
    _lir: &[LirOp],
) -> alloc::vec::Vec<LirGroup> {
    planned_groups
        .iter()
        .map(|group| LirGroup {
            id: group.id,
            lir_range: group.start as usize..group.end as usize,
            terminator: group.terminator,
        })
        .collect()
}

fn lower_planned_op(
    op: &PlannedOp,
    lowering_map: &LoweringMap,
    frame: FrameLayoutPlan,
) -> LirOp {
    let kind = match &op.kind {
        PlannedOpKind::Core(kind) => LirOpKind::Core(kind.clone()),
        PlannedOpKind::InitLocals { l0, l1, l2 } => LirOpKind::InitLocals {
            l0: *l0,
            l1: *l1,
            l2: *l2,
        },
        PlannedOpKind::Spill(artifact) => LirOpKind::CacheTransfer(*artifact),
        PlannedOpKind::LocalGet { local } => match local {
            crate::vm::plan::plan::PlannedLocal::Hot(reg) => LirOpKind::LocalGetHot { reg: *reg },
            crate::vm::plan::plan::PlannedLocal::Frame(frame_slot) => {
                LirOpKind::LocalGetFrame {
                    frame_slot: *frame_slot,
                }
            }
        },
        PlannedOpKind::LocalSet { local } => match local {
            crate::vm::plan::plan::PlannedLocal::Hot(reg) => LirOpKind::LocalSetHot { reg: *reg },
            crate::vm::plan::plan::PlannedLocal::Frame(frame_slot) => {
                LirOpKind::LocalSetFrame {
                    frame_slot: *frame_slot,
                }
            }
        },
        PlannedOpKind::LocalTee { local } => match local {
            crate::vm::plan::plan::PlannedLocal::Hot(reg) => LirOpKind::LocalTeeHot { reg: *reg },
            crate::vm::plan::plan::PlannedLocal::Frame(frame_slot) => {
                LirOpKind::LocalTeeFrame {
                    frame_slot: *frame_slot,
                }
            }
        },
        PlannedOpKind::Marker(kind) => LirOpKind::Marker(lower_marker(*kind)),
        PlannedOpKind::Branch {
            kind,
            payload,
            target,
            ..
        } => LirOpKind::Branch {
            kind: *kind,
            fixup: lower_branch_fixup(frame, op.height, *kind, *payload),
            target: target.map(|target| lower_target(target, lowering_map)),
        },
        PlannedOpKind::BrTable { entries, .. } => LirOpKind::BrTable {
            entries: entries
                .iter()
                .copied()
                .map(|entry| lower_br_table_entry(entry, lowering_map, frame, op.height))
                .collect(),
        },
        PlannedOpKind::CallExternal {
            func_idx,
            args,
            results,
        } => LirOpKind::CallExternal {
            func_idx: *func_idx,
            args: lower_call_args(frame, op.height, args.count, 0),
            results: lower_call_results(frame, op.height, args.count, 0, results.count),
        },
        PlannedOpKind::CallInternal {
            callee,
            args,
            results,
        } => LirOpKind::CallInternal {
            callee: *callee,
            args: lower_call_args(frame, op.height, args.count, 0),
            results: lower_call_results(frame, op.height, args.count, 0, results.count),
        },
        PlannedOpKind::CallIndirect {
            type_idx,
            table_idx,
            args,
            results,
            ..
        } => LirOpKind::CallIndirect {
            type_idx: *type_idx,
            table_idx: *table_idx,
            index_slot: lower_operand_slot(frame, op.height.saturating_sub(1)),
            args: lower_call_args(frame, op.height, args.count, 1),
            results: lower_call_results(frame, op.height, args.count, 1, results.count),
        },
        PlannedOpKind::Return { results } => LirOpKind::Return {
            results: lower_return_results(frame, op.height, *results),
        },
    };

    LirOp {
        kind,
        window: op.rotation,
        alt: op.alt.map(|target| lower_target(target, lowering_map)),
    }
}

fn lower_marker(kind: PlannedMarkerKind) -> LirMarkerKind {
    match kind {
        PlannedMarkerKind::Block { params, results } => LirMarkerKind::Block { params, results },
        PlannedMarkerKind::Loop { params, results } => LirMarkerKind::Loop { params, results },
        PlannedMarkerKind::If { params, results } => LirMarkerKind::If { params, results },
        PlannedMarkerKind::Else => LirMarkerKind::Else,
        PlannedMarkerKind::End => LirMarkerKind::End,
    }
}

fn lower_br_table_entry(
    entry: PlannedBrTableEntry,
    lowering_map: &LoweringMap,
    frame: FrameLayoutPlan,
    pre_height: u16,
) -> LirBrTableEntry {
    LirBrTableEntry {
        target: entry.target.map(|target| lower_target(target, lowering_map)),
        fixup: lower_payload_fixup(frame, pre_height.saturating_sub(1), entry.payload),
    }
}

fn lower_branch_fixup(
    frame: FrameLayoutPlan,
    pre_height: u16,
    kind: PlannedBranchKind,
    payload_dst: Option<FrameSpan>,
) -> Option<LirFrameCopy> {
    let payload_height = match kind {
        PlannedBranchKind::Br => pre_height,
        PlannedBranchKind::BrIf => pre_height.saturating_sub(1),
    };
    lower_payload_fixup(frame, payload_height, payload_dst)
}

fn lower_payload_fixup(
    frame: FrameLayoutPlan,
    payload_height: u16,
    payload_desc: Option<FrameSpan>,
) -> Option<LirFrameCopy> {
    let payload = payload_desc?;
    let stack_drop = payload.start.0.saturating_sub(frame.operand_base);
    let arity = payload.count;
    let src_start = payload_height.saturating_sub(arity);
    let dst_start = payload_height
        .saturating_sub(stack_drop)
        .saturating_sub(arity);
    Some(LirFrameCopy {
        src: lower_operand_span(frame, src_start, arity),
        dst: lower_operand_span(frame, dst_start, arity),
    })
}

fn lower_call_args(
    frame: FrameLayoutPlan,
    pre_height: u16,
    arg_count: u16,
    extra_pops: u16,
) -> FrameSpan {
    let start = pre_height.saturating_sub(arg_count.saturating_add(extra_pops));
    lower_operand_span(frame, start, arg_count)
}

fn lower_call_results(
    frame: FrameLayoutPlan,
    pre_height: u16,
    arg_count: u16,
    extra_pops: u16,
    result_count: u16,
) -> FrameSpan {
    let start = pre_height.saturating_sub(arg_count.saturating_add(extra_pops));
    lower_operand_span(frame, start, result_count)
}

fn lower_return_results(
    frame: FrameLayoutPlan,
    pre_height: u16,
    results: Option<FrameSpan>,
) -> Option<FrameSpan> {
    let shape = results?;
    Some(lower_operand_span(
        frame,
        pre_height.saturating_sub(shape.count),
        shape.count,
    ))
}

fn lower_operand_span(frame: FrameLayoutPlan, start: u16, count: u16) -> FrameSpan {
    FrameSpan::new(lower_operand_slot(frame, start), count)
}

fn lower_operand_slot(frame: FrameLayoutPlan, idx: u16) -> crate::vm::plan::frame::FrameSlot {
    frame.operand_slot(idx)
}

fn lower_target(
    target: crate::vm::wasm::common::SemanticTarget,
    lowering_map: &LoweringMap,
) -> LirTarget {
    lowering_map.target_for_semantic_index(target.index().as_usize())
}

fn is_synthetic_planned_op(kind: &PlannedOpKind) -> bool {
    matches!(kind, PlannedOpKind::Spill(_) | PlannedOpKind::InitLocals { .. })
}
