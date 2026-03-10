//! Planning-stage grouping.
//!
//! Grouping belongs here, before backend-facing IR.
//! Backends consume the result; they do not rediscover groups later.

use alloc::vec::Vec;

use crate::vm::wasm::{
    common::SemanticTarget,
    semantic_ir::{SemanticOp, SemanticOpKind},
};

use super::{
    policy::{GroupPolicy, PlanPolicy},
    types::{PlannedBranchKind, PlannedMarkerKind, PlannedOpKind},
};

/// Planned group identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GroupId(pub u32);

impl GroupId {
    #[inline]
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

/// One grouped region in planned order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupRange {
    pub id: GroupId,
    pub start: u32,
    pub end: u32,
    pub terminator: Option<GroupTerminator>,
}

/// Full grouping plan for one function.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GroupPlan {
    pub groups: alloc::vec::Vec<GroupRange>,
}

/// Planning-stage group terminator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupTerminator {
    Br,
    BrIf,
    BrTable,
    If,
    ReturnVoid,
    ReturnOne,
    Return,
    Unreachable,
}

/// Planning-stage grouping input.
///
/// By the time grouping runs, semantic decode and planning policy should
/// already have decided:
/// - whether an op can start/extend a group
/// - whether it is a terminator
/// - which semantic targets jump into it
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GroupInputOp {
    pub can_start: bool,
    pub can_extend: bool,
    pub terminator: Option<GroupTerminator>,
    pub alt: Option<SemanticTarget>,
    pub br_table_targets: Vec<SemanticTarget>,
}

pub fn build_group_input(
    policy: &PlanPolicy,
    semantic_op: &SemanticOp,
    planned_kind: &PlannedOpKind,
) -> GroupInputOp {
    GroupInputOp {
        can_start: policy.can_start(semantic_op) && groupable_kind(planned_kind),
        can_extend: policy.can_extend(0, semantic_op) && groupable_kind(planned_kind),
        terminator: terminator_for(policy, semantic_op, planned_kind),
        alt: semantic_op.alt,
        br_table_targets: collect_br_table_targets(semantic_op),
    }
}

fn groupable_kind(kind: &PlannedOpKind) -> bool {
    !matches!(
        kind,
        PlannedOpKind::Marker(_) | PlannedOpKind::Spill(_) | PlannedOpKind::InitHotLocals { .. }
    )
}

fn terminator_for(
    policy: &PlanPolicy,
    semantic_op: &SemanticOp,
    planned_kind: &PlannedOpKind,
) -> Option<GroupTerminator> {
    if !policy.is_terminator(semantic_op) {
        return None;
    }

    Some(match planned_kind {
        PlannedOpKind::Branch {
            kind: PlannedBranchKind::Br,
            ..
        } => GroupTerminator::Br,
        PlannedOpKind::Branch {
            kind: PlannedBranchKind::BrIf,
            ..
        } => GroupTerminator::BrIf,
        PlannedOpKind::BrTable { .. } => GroupTerminator::BrTable,
        PlannedOpKind::Marker(PlannedMarkerKind::If { .. }) => GroupTerminator::If,
        PlannedOpKind::Return { results: None } => GroupTerminator::ReturnVoid,
        PlannedOpKind::Return {
            results: Some(span),
        } if span.count == 1 => GroupTerminator::ReturnOne,
        PlannedOpKind::Return { .. } => GroupTerminator::Return,
        _ => GroupTerminator::Unreachable,
    })
}

fn collect_br_table_targets(op: &SemanticOp) -> Vec<SemanticTarget> {
    match &op.kind {
        SemanticOpKind::BrTable { entries } => {
            entries.iter().filter_map(|entry| entry.target).collect()
        }
        _ => Vec::new(),
    }
}

fn incoming_targets(ops: &[GroupInputOp]) -> Vec<bool> {
    let mut incoming = alloc::vec![false; ops.len()];

    for op in ops {
        if let Some(target) = op.alt {
            let idx = target.index().as_usize();
            if idx < incoming.len() {
                incoming[idx] = true;
            }
        }

        for target in &op.br_table_targets {
            let idx = target.index().as_usize();
            if idx < incoming.len() {
                incoming[idx] = true;
            }
        }
    }

    incoming
}

/// Shared planning-stage grouping.
pub fn plan_groups(ops: &[GroupInputOp]) -> GroupPlan {
    let branch_targets = incoming_targets(ops);
    let mut groups = Vec::new();
    let mut i = 0usize;
    let mut next_id = 0u32;

    while i < ops.len() {
        if !ops[i].can_start {
            i += 1;
            continue;
        }

        let start = i;
        let mut terminator = ops[i].terminator;
        i += 1;

        while terminator.is_none() && i < ops.len() {
            if branch_targets[i] || !ops[i].can_extend {
                break;
            }
            terminator = ops[i].terminator;
            i += 1;
        }

        groups.push(GroupRange {
            id: GroupId(next_id),
            start: start as u32,
            end: i as u32,
            terminator,
        });
        next_id += 1;
    }

    GroupPlan { groups }
}
