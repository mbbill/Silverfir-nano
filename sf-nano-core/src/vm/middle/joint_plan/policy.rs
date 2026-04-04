//! Shared planner policy helpers.

use crate::vm::wasm::{primitive_op::PrimitiveOpKind, semantic_ir::SemanticOpKind};

#[inline]
pub(super) fn op_must_drop_all_caches(kind: &SemanticOpKind) -> bool {
    matches!(
        kind,
        SemanticOpKind::If { .. }
            | SemanticOpKind::Else { .. }
            | SemanticOpKind::End
            | SemanticOpKind::Br { .. }
            | SemanticOpKind::BrIf { .. }
            | SemanticOpKind::BrTable { .. }
            | SemanticOpKind::CallDirect { .. }
            | SemanticOpKind::CallIndirect { .. }
            | SemanticOpKind::ReturnVoid
            | SemanticOpKind::ReturnOne
            | SemanticOpKind::Return { .. }
    )
}

#[inline]
pub(super) fn clears_cache_region(kind: &SemanticOpKind) -> bool {
    op_must_drop_all_caches(kind)
        || matches!(
            kind,
            SemanticOpKind::Loop { .. } | SemanticOpKind::Primitive(PrimitiveOpKind::Unreachable)
        )
}
