//! Backend-supplied grouping and planning policy.

use crate::vm::{
    backend::BackendKind,
    wasm::semantic_ir::{SemanticOp, SemanticOpKind},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupingMode {
    Ignore,
    PatternDriven,
    Maximal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanPolicy {
    pub backend: BackendKind,
    pub grouping: GroupingMode,
}

impl PlanPolicy {
    #[inline]
    pub const fn for_backend(backend: BackendKind) -> Self {
        let grouping = backend.capabilities().grouping;
        Self { backend, grouping }
    }
}

/// Grouping policy over planned/semantic ops.
pub trait GroupPolicy {
    fn can_start(&self, _op: &SemanticOp) -> bool;
    fn can_extend(&self, _current_len: usize, _next: &SemanticOp) -> bool;
    fn is_terminator(&self, _op: &SemanticOp) -> bool;
}

impl GroupPolicy for PlanPolicy {
    fn can_start(&self, op: &SemanticOp) -> bool {
        match self.grouping {
            GroupingMode::Ignore => false,
            GroupingMode::PatternDriven | GroupingMode::Maximal => {
                !matches!(
                    op.kind,
                    SemanticOpKind::Block { .. }
                        | SemanticOpKind::Loop { .. }
                        | SemanticOpKind::Else
                        | SemanticOpKind::End
                )
            }
        }
    }

    fn can_extend(&self, _current_len: usize, next: &SemanticOp) -> bool {
        self.can_start(next)
    }

    fn is_terminator(&self, op: &SemanticOp) -> bool {
        matches!(
            op.kind,
            SemanticOpKind::Br { .. }
                | SemanticOpKind::BrIf { .. }
                | SemanticOpKind::BrTable { .. }
                | SemanticOpKind::If { .. }
                | SemanticOpKind::ReturnVoid
                | SemanticOpKind::ReturnOne
                | SemanticOpKind::Return { .. }
        )
    }
}
