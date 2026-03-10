//! Semantic-to-planned placement.

use alloc::vec::Vec;

use crate::vm::wasm::{
    common::SemanticTarget,
    semantic_ir::{SemanticOp, SemanticOpKind, SemanticProgram},
};

use super::{
    frame::{FrameLayoutPlan, FrameSpan},
    group::{self, GroupInputOp},
    hot_local::HotLocalPlan,
    policy::PlanPolicy,
    types::{
        PlannedBrTableEntry, PlannedBranchKind, PlannedLocal, PlannedMarkerKind, PlannedOpKind,
    },
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SeedOp {
    pub(crate) kind: PlannedOpKind,
    pub(crate) alt: Option<SemanticTarget>,
}

pub(crate) struct PlacementResult {
    pub(crate) ops: Vec<SeedOp>,
    pub(crate) group_inputs: Vec<GroupInputOp>,
}

pub(crate) fn place_semantic_ops(
    semantic: &SemanticProgram,
    frame: FrameLayoutPlan,
    hot_locals: Option<&HotLocalPlan>,
    policy: &PlanPolicy,
) -> PlacementResult {
    let mut ops = Vec::with_capacity(semantic.ops.len());
    let mut group_inputs = Vec::with_capacity(semantic.ops.len());

    for semantic_op in &semantic.ops {
        let kind = place_semantic_op(semantic_op, frame, hot_locals);
        group_inputs.push(group::build_group_input(policy, semantic_op, &kind));
        ops.push(SeedOp {
            kind,
            alt: semantic_op.alt,
        });
    }

    PlacementResult { ops, group_inputs }
}

fn place_semantic_op(
    op: &SemanticOp,
    frame: FrameLayoutPlan,
    hot_locals: Option<&HotLocalPlan>,
) -> PlannedOpKind {
    match &op.kind {
        SemanticOpKind::Primitive(kind) => PlannedOpKind::Primitive(kind.clone()),
        SemanticOpKind::LocalGet { idx } => PlannedOpKind::LocalGet {
            local: place_local(*idx, frame, hot_locals),
        },
        SemanticOpKind::LocalSet { idx } => PlannedOpKind::LocalSet {
            local: place_local(*idx, frame, hot_locals),
        },
        SemanticOpKind::LocalTee { idx } => PlannedOpKind::LocalTee {
            local: place_local(*idx, frame, hot_locals),
        },
        SemanticOpKind::Block { params, results } => {
            PlannedOpKind::Marker(PlannedMarkerKind::Block {
                params: *params,
                results: *results,
            })
        }
        SemanticOpKind::Loop { params, results } => {
            PlannedOpKind::Marker(PlannedMarkerKind::Loop {
                params: *params,
                results: *results,
            })
        }
        SemanticOpKind::If { params, results } => PlannedOpKind::Marker(PlannedMarkerKind::If {
            params: *params,
            results: *results,
        }),
        SemanticOpKind::Else => PlannedOpKind::Marker(PlannedMarkerKind::Else),
        SemanticOpKind::End => PlannedOpKind::Marker(PlannedMarkerKind::End),
        SemanticOpKind::Br { stack_drop, arity } => PlannedOpKind::Branch {
            kind: PlannedBranchKind::Br,
            condition_slot: None,
            payload: branch_payload(frame, *stack_drop, *arity),
            target: op.alt,
        },
        SemanticOpKind::BrIf { stack_drop, arity } => PlannedOpKind::Branch {
            kind: PlannedBranchKind::BrIf,
            condition_slot: Some(frame.operand_slot(0)),
            payload: branch_payload(frame, *stack_drop, *arity),
            target: op.alt,
        },
        SemanticOpKind::BrTable { entries } => PlannedOpKind::BrTable {
            index_slot: frame.operand_slot(0),
            entries: entries
                .iter()
                .map(|entry| PlannedBrTableEntry {
                    target: entry.target,
                    payload: branch_payload(frame, entry.stack_drop, entry.arity),
                })
                .collect(),
        },
        SemanticOpKind::CallExternal {
            func_idx,
            params,
            results,
        } => PlannedOpKind::CallExternal {
            func_idx: *func_idx,
            args: call_args(frame, *params),
            results: call_results(frame, *results),
        },
        SemanticOpKind::CallInternal {
            callee,
            params,
            results,
        } => PlannedOpKind::CallInternal {
            callee: *callee,
            args: call_args(frame, *params),
            results: call_results(frame, *results),
        },
        SemanticOpKind::CallIndirect {
            type_idx,
            table_idx,
            params,
            results,
        } => PlannedOpKind::CallIndirect {
            type_idx: *type_idx,
            table_idx: *table_idx,
            index_slot: frame.operand_slot(0),
            args: call_args(frame, *params),
            results: call_results(frame, *results),
        },
        SemanticOpKind::ReturnVoid => PlannedOpKind::Return { results: None },
        SemanticOpKind::ReturnOne => PlannedOpKind::Return {
            results: Some(FrameSpan::single(frame.operand_slot(0))),
        },
        SemanticOpKind::Return { arity } => PlannedOpKind::Return {
            results: Some(call_results(frame, *arity)),
        },
    }
}

fn place_local(
    idx: u16,
    frame: FrameLayoutPlan,
    hot_locals: Option<&HotLocalPlan>,
) -> PlannedLocal {
    if let Some(hot) = hot_locals.and_then(|plan| plan.resolve(idx as u32)) {
        PlannedLocal::Hot(hot)
    } else {
        PlannedLocal::Frame(frame.local_slot(idx))
    }
}

fn branch_payload(frame: FrameLayoutPlan, stack_drop: u32, arity: u16) -> Option<FrameSpan> {
    if arity == 0 {
        None
    } else {
        Some(FrameSpan::new(frame.operand_slot(stack_drop as u16), arity))
    }
}

fn call_args(frame: FrameLayoutPlan, params: u16) -> FrameSpan {
    FrameSpan::new(frame.operand_slot(0), params)
}

fn call_results(frame: FrameLayoutPlan, results: u16) -> FrameSpan {
    FrameSpan::new(frame.operand_slot(0), results)
}
