//! Semantic/planned alignment for LIR lowering.

use alloc::vec::Vec;

use crate::error::WasmError;
use crate::vm::{
    plan::{PlannedOp, PlannedOpKind, PlannedProgram},
    wasm::semantic_ir::{SemanticOp, SemanticOpKind, SemanticProgram},
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
    let mut mapped = Vec::with_capacity(semantic.ops.len());
    let mut semantic_index = 0usize;

    for planned_op in &planned.ops {
        if is_synthetic_planned_op(&planned_op.kind) {
            continue;
        }

        let semantic_op = semantic.ops.get(semantic_index).ok_or_else(|| {
            WasmError::internal("planned/semantic op count mismatch during CFG LIR lowering".into())
        })?;

        mapped.push(SemanticPlannedOp {
            semantic: semantic_op,
            planned: planned_op,
        });
        semantic_index += 1;
    }

    if semantic_index != semantic.ops.len() {
        return Err(WasmError::internal(
            "planned/semantic op count mismatch during CFG LIR lowering".into(),
        ));
    }

    Ok(mapped)
}

fn is_synthetic_planned_op(kind: &PlannedOpKind) -> bool {
    matches!(
        kind,
        PlannedOpKind::Spill(_) | PlannedOpKind::InitHotLocals { .. }
    )
}

pub(super) fn resolve_local_index(op: &SemanticPlannedOp<'_>) -> Result<u32, WasmError> {
    match op.semantic.kind {
        SemanticOpKind::LocalGet { idx }
        | SemanticOpKind::LocalSet { idx }
        | SemanticOpKind::LocalTee { idx } => Ok(idx),
        _ => Err(WasmError::internal(
            "LIR local lowering expected semantic local op".into(),
        )),
    }
}
