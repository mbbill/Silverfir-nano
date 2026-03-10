//! Semantic/planned alignment for LIR lowering.

use alloc::vec::Vec;

use crate::error::WasmError;
use crate::vm::{
    plan::plan::{PlannedOp, PlannedOpKind, PlannedProgram},
    wasm::semantic_ir::{SemanticOp, SemanticProgram},
};

#[derive(Clone, Debug)]
pub(super) struct SemanticPlannedOp<'a> {
    pub(super) semantic: &'a SemanticOp,
    pub(super) planned: &'a PlannedOp,
}

pub(super) fn map_semantic_to_planned<'a>(
    semantic: &'a SemanticProgram,
    planned: &'a PlannedProgram,
) -> Result<Vec<SemanticPlannedOp<'a>>, WasmError> {
    let planned_semantic = planned
        .ops
        .iter()
        .filter(|op| !is_synthetic_planned_op(&op.kind))
        .collect::<Vec<_>>();
    if planned_semantic.len() != semantic.ops.len() {
        return Err(WasmError::internal(
            "planned/semantic op count mismatch during CFG LIR lowering".into(),
        ));
    }
    Ok(semantic
        .ops
        .iter()
        .zip(planned_semantic)
        .map(|(semantic, planned)| SemanticPlannedOp { semantic, planned })
        .collect())
}

fn is_synthetic_planned_op(kind: &PlannedOpKind) -> bool {
    matches!(
        kind,
        PlannedOpKind::Spill(_) | PlannedOpKind::InitLocals { .. }
    )
}
