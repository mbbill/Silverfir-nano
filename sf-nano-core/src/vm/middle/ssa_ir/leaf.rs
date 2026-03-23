//! SSA-IR leaf-op vocabulary.
//!
//! This wraps `PrimitiveOpKind` but excludes operations that are designated as
//! true slot-based boundary ops in prepared SSA-IR.

use crate::vm::wasm::primitive_op::PrimitiveOpKind;

/// One local, non-runtime leaf op.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SsaLeafOp(PrimitiveOpKind);

impl SsaLeafOp {
    #[inline]
    pub(crate) fn from_primitive(kind: PrimitiveOpKind) -> Option<Self> {
        (!is_boundary_primitive(&kind)).then_some(Self(kind))
    }

    #[inline]
    pub(crate) fn primitive(&self) -> &PrimitiveOpKind {
        &self.0
    }
}

#[inline]
pub(crate) fn is_boundary_primitive(kind: &PrimitiveOpKind) -> bool {
    matches!(
        kind,
        PrimitiveOpKind::MemoryGrow { .. }
            | PrimitiveOpKind::MemoryFill { .. }
            | PrimitiveOpKind::MemoryCopy { .. }
            | PrimitiveOpKind::TableGrow { .. }
            | PrimitiveOpKind::TableFill { .. }
            | PrimitiveOpKind::TableCopy { .. }
            | PrimitiveOpKind::MemoryInit { .. }
            | PrimitiveOpKind::DataDrop { .. }
            | PrimitiveOpKind::TableInit { .. }
            | PrimitiveOpKind::ElemDrop { .. }
    )
}
