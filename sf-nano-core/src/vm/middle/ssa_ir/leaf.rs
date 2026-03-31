//! SSA-IR leaf-op vocabulary.
//!
//! This wraps `PrimitiveOpKind` for use in prepared SSA-IR leaf instructions.

use crate::vm::wasm::primitive_op::PrimitiveOpKind;

/// One local, non-runtime leaf op.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SsaLeafOp(PrimitiveOpKind);

impl SsaLeafOp {
    #[inline]
    pub(crate) fn from_primitive(kind: PrimitiveOpKind) -> Option<Self> {
        Some(Self(kind))
    }

    #[inline]
    pub(crate) fn primitive(&self) -> &PrimitiveOpKind {
        &self.0
    }
}
