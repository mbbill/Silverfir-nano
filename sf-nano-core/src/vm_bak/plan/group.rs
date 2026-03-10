use alloc::vec::Vec;

use crate::vm::wasm::{
    common::OpIndex,
    semantic_ir::{SemanticOp, SemanticOpKind},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupTerminator {
    Br,
    BrIf,
    If,
    ReturnVoid,
    ReturnOne,
    Return,
    Unreachable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupRange {
    pub semantic_start: usize,
    pub semantic_end: usize,
    pub terminator: Option<GroupTerminator>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GroupPlan {
    pub ranges: Vec<GroupRange>,
}

pub trait SemanticGroupPolicy {
    fn can_group(&self, op: &SemanticOp) -> bool;
    fn is_redirect_marker(&self, op: &SemanticOp) -> bool;
    fn terminator(&self, op: &SemanticOp) -> Option<GroupTerminator>;
}

fn incoming_targets(ops: &[SemanticOp]) -> Vec<bool> {
    let mut incoming = alloc::vec![false; ops.len()];

    for op in ops {
        if let Some(target) = op.alt_target {
            if target.as_usize() < incoming.len() {
                incoming[target.as_usize()] = true;
            }
        }

        if let SemanticOpKind::BrTable { entries } = &op.kind {
            for entry in entries {
                if let Some(target) = entry.target_idx {
                    if target.as_usize() < incoming.len() {
                        incoming[target.as_usize()] = true;
                    }
                }
            }
        }
    }

    incoming
}

pub fn plan_groups<P: SemanticGroupPolicy>(ops: &[SemanticOp], policy: &P) -> GroupPlan {
    let branch_targets = incoming_targets(ops);
    let mut ranges = Vec::new();
    let mut i = 0usize;

    while i < ops.len() {
        if !policy.can_group(&ops[i]) {
            i += 1;
            continue;
        }

        let start = i;
        let mut terminator = policy.terminator(&ops[i]);
        i += 1;

        while terminator.is_none() && i < ops.len() {
            if branch_targets[i] || policy.is_redirect_marker(&ops[i]) || !policy.can_group(&ops[i]) {
                break;
            }

            terminator = policy.terminator(&ops[i]);
            i += 1;
            if terminator.is_some() {
                break;
            }
        }

        ranges.push(GroupRange {
            semantic_start: start,
            semantic_end: i,
            terminator,
        });
    }

    GroupPlan { ranges }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::wasm::semantic_ir::SemanticOp;

    struct DummyPolicy;

    impl SemanticGroupPolicy for DummyPolicy {
        fn can_group(&self, op: &SemanticOp) -> bool {
            !matches!(op.kind, SemanticOpKind::Else | SemanticOpKind::End)
        }

        fn is_redirect_marker(&self, op: &SemanticOp) -> bool {
            matches!(op.kind, SemanticOpKind::Else | SemanticOpKind::End)
        }

        fn terminator(&self, op: &SemanticOp) -> Option<GroupTerminator> {
            match op.kind {
                SemanticOpKind::If { .. } => Some(GroupTerminator::If),
                SemanticOpKind::ReturnOne => Some(GroupTerminator::ReturnOne),
                _ => None,
            }
        }
    }

    fn op(kind: SemanticOpKind) -> SemanticOp {
        SemanticOp {
            kind,
            window: 0,
            pre_height: 0,
            fallthrough: None,
            alt_target: None,
            has_target: false,
        }
    }

    #[test]
    fn test_plan_groups_stops_at_redirect_markers() {
        let ops = alloc::vec![
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            op(SemanticOpKind::Else),
            op(SemanticOpKind::LocalGet { idx: 1 }),
        ];

        let plan = plan_groups(&ops, &DummyPolicy);
        assert_eq!(plan.ranges.len(), 2);
        assert_eq!(plan.ranges[0].semantic_start, 0);
        assert_eq!(plan.ranges[0].semantic_end, 2);
        assert_eq!(plan.ranges[1].semantic_start, 3);
        assert_eq!(plan.ranges[1].semantic_end, 4);
    }

    #[test]
    fn test_plan_groups_breaks_at_incoming_targets() {
        let mut branch = op(SemanticOpKind::If { params: 0, results: 0 });
        branch.alt_target = Some(OpIndex::new(2));
        let ops = alloc::vec![
            op(SemanticOpKind::LocalGet { idx: 0 }),
            branch,
            op(SemanticOpKind::LocalGet { idx: 1 }),
        ];

        let plan = plan_groups(&ops, &DummyPolicy);
        assert_eq!(plan.ranges.len(), 2);
        assert_eq!(plan.ranges[0].semantic_start, 0);
        assert_eq!(plan.ranges[0].semantic_end, 2);
        assert_eq!(plan.ranges[0].terminator, Some(GroupTerminator::If));
        assert_eq!(plan.ranges[1].semantic_start, 2);
        assert_eq!(plan.ranges[1].semantic_end, 3);
    }
}
