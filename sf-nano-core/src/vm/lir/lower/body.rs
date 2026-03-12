//! Straight-line block-body lowering.

use crate::error::WasmError;
use crate::vm::{plan::frame::FrameLayoutPlan, wasm::semantic_ir::SemanticOpKind};

use super::{
    input::{resolve_local_index, SemanticPlannedOp},
    ops::{
        lower_call_external, lower_call_indirect, lower_call_internal, lower_local_get,
        lower_local_set, lower_local_tee, lower_primitive,
    },
    state::{BlockState, ValueAlloc},
};

pub(super) fn lower_block_body_op(
    op: &SemanticPlannedOp<'_>,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    values: &mut ValueAlloc,
) -> Result<(), WasmError> {
    match &op.semantic.kind {
        SemanticOpKind::Primitive(kind)
            if matches!(
                kind,
                crate::vm::wasm::primitive_op::PrimitiveOpKind::Unreachable
            ) =>
        {
            Err(WasmError::internal(
                "unreachable must end an LIR block, not appear in the body".into(),
            ))
        }
        SemanticOpKind::Primitive(kind) => {
            lower_primitive(kind, state, values)
        }
        SemanticOpKind::LocalGet { .. } => {
            lower_local_get(resolve_local_index(op)?, state, frame, values)
        }
        SemanticOpKind::LocalSet { .. } => {
            lower_local_set(resolve_local_index(op)?, state, frame)
        }
        SemanticOpKind::LocalTee { .. } => {
            lower_local_tee(resolve_local_index(op)?, state, frame)
        }
        SemanticOpKind::Block { .. } | SemanticOpKind::Loop { .. } | SemanticOpKind::End => Ok(()),
        SemanticOpKind::Else => Err(WasmError::internal(
            "else must end an LIR block, not appear in the body".into(),
        )),
        SemanticOpKind::CallExternal {
            func_idx,
            params,
            results,
        } => {
            lower_call_external(*func_idx, *params, *results, state, values)
        }
        SemanticOpKind::CallInternal {
            callee,
            params,
            results,
        } => {
            lower_call_internal(*callee, *params, *results, state, values)
        }
        SemanticOpKind::CallIndirect {
            type_idx,
            table_idx,
            params,
            results,
        } => {
            lower_call_indirect(*type_idx, *table_idx, *params, *results, state, values)
        }
        SemanticOpKind::If { .. }
        | SemanticOpKind::Br { .. }
        | SemanticOpKind::BrIf { .. }
        | SemanticOpKind::BrTable { .. }
        | SemanticOpKind::ReturnVoid
        | SemanticOpKind::ReturnOne
        | SemanticOpKind::Return { .. } => Err(WasmError::internal(
            "control-flow terminators must end an LIR block".into(),
        )),
    }
}
