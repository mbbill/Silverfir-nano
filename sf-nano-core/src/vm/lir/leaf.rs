//! LIR-local leaf-op vocabulary.
//!
//! This wraps `PrimitiveOpKind` but excludes operations that are designated as
//! true slot-based boundary ops in prepared LIR.

use crate::vm::wasm::primitive_op::PrimitiveOpKind;

/// One local, non-runtime leaf op.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LirLeafOp(PrimitiveOpKind);

impl LirLeafOp {
    #[inline]
    pub fn from_primitive(kind: PrimitiveOpKind) -> Option<Self> {
        (!is_runtime_boundary_primitive(&kind)).then_some(Self(kind))
    }

    #[inline]
    pub fn primitive(&self) -> &PrimitiveOpKind {
        &self.0
    }
}

#[inline]
pub fn is_runtime_boundary_primitive(kind: &PrimitiveOpKind) -> bool {
    matches!(
        kind,
        PrimitiveOpKind::MemoryGrow { .. }
            | PrimitiveOpKind::TableGrow { .. }
            | PrimitiveOpKind::MemoryInit { .. }
            | PrimitiveOpKind::DataDrop { .. }
            | PrimitiveOpKind::TableInit { .. }
            | PrimitiveOpKind::ElemDrop { .. }
    )
}
