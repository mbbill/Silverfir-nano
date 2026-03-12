//! Semantic/planned alignment for prepared LIR lowering.

use alloc::vec::Vec;

use crate::error::WasmError;
use crate::vm::{
    lir::ir::LirLocalCachePrefs,
    plan::{hot_local::HotLocalPlan, frame::FrameLayoutPlan, PlannedOp, PlannedOpKind, PlannedProgram},
    wasm::semantic_ir::{SemanticOp, SemanticOpKind, SemanticProgram},
};

#[derive(Clone, Debug)]
pub(super) struct AlignedOp<'a> {
    pub(super) semantic: &'a SemanticOp,
    pub(super) planned: &'a PlannedOp,
    /// Synthetic transient-window prep emitted immediately before the semantic op.
    pub(super) prefix: Vec<&'a PlannedOp>,
}

pub(super) fn align_semantic_to_planned<'a>(
    semantic: &'a SemanticProgram,
    planned: &'a PlannedProgram,
) -> Result<Vec<AlignedOp<'a>>, WasmError> {
    let mut aligned = Vec::with_capacity(semantic.ops.len());
    let mut semantic_index = 0usize;
    let mut prefix = Vec::new();

    for planned_op in &planned.ops {
        match &planned_op.kind {
            PlannedOpKind::InitHotLocals { .. } => {}
            PlannedOpKind::Spill(_) => prefix.push(planned_op),
            _ => {
                let semantic_op = semantic.ops.get(semantic_index).ok_or_else(|| {
                    WasmError::internal(
                        "planned/semantic op count mismatch during prepared LIR lowering".into(),
                    )
                })?;
                aligned.push(AlignedOp {
                    semantic: semantic_op,
                    planned: planned_op,
                    prefix: core::mem::take(&mut prefix),
                });
                semantic_index += 1;
            }
        }
    }

    if semantic_index != semantic.ops.len() {
        return Err(WasmError::internal(
            "planned/semantic op count mismatch during prepared LIR lowering".into(),
        ));
    }
    if !prefix.is_empty() {
        return Err(WasmError::internal(
            "dangling synthetic spill/fill ops at end of planned program".into(),
        ));
    }

    Ok(aligned)
}

pub(super) fn resolve_local_index(op: &AlignedOp<'_>) -> Result<u32, WasmError> {
    match op.semantic.kind {
        SemanticOpKind::LocalGet { idx }
        | SemanticOpKind::LocalSet { idx }
        | SemanticOpKind::LocalTee { idx } => Ok(idx as u32),
        _ => Err(WasmError::internal(
            "prepared LIR local lowering expected semantic local op".into(),
        )),
    }
}

pub(super) fn lower_local_cache_plan(
    hot_locals: Option<&HotLocalPlan>,
    frame: FrameLayoutPlan,
) -> LirLocalCachePrefs {
    let preferred_slots = hot_locals
        .map(|plan| {
            plan.mapping()
                .iter()
                .filter_map(|local_idx| local_idx.map(|idx| frame.local_slot(idx as u16)))
                .collect()
        })
        .unwrap_or_default();
    LirLocalCachePrefs { preferred_slots }
}
