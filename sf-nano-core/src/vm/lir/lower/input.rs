//! Semantic/planned alignment for LIR lowering.

use alloc::vec::Vec;
use core::mem;

use crate::error::WasmError;
use crate::vm::{
    plan::{PlannedOp, PlannedOpKind, PlannedProgram},
    wasm::semantic_ir::{SemanticOp, SemanticProgram},
};

#[derive(Clone, Debug)]
pub(super) struct SemanticPlannedOp<'a> {
    pub(super) semantic: &'a SemanticOp,
    pub(super) planned: &'a PlannedOp,
    pub(super) prefix: Vec<&'a PlannedOp>,
}

pub(super) fn map_semantic_to_planned<'a>(
    semantic: &'a SemanticProgram,
    planned: &'a PlannedProgram,
) -> Result<Vec<SemanticPlannedOp<'a>>, WasmError> {
    let mut mapped = Vec::with_capacity(semantic.ops.len());
    let mut semantic_index = 0usize;
    let mut prefix = Vec::new();

    for planned_op in &planned.ops {
        if is_synthetic_planned_op(&planned_op.kind) {
            prefix.push(planned_op);
            continue;
        }

        let semantic_op = semantic.ops.get(semantic_index).ok_or_else(|| {
            WasmError::internal("planned/semantic op count mismatch during CFG LIR lowering".into())
        })?;

        mapped.push(SemanticPlannedOp {
            semantic: semantic_op,
            planned: planned_op,
            prefix: mem::take(&mut prefix),
        });
        semantic_index += 1;
    }

    if semantic_index != semantic.ops.len() || !prefix.is_empty() {
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
