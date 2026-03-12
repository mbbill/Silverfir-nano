//! Straight-line block-body lowering.

use crate::error::WasmError;
use crate::vm::{
    plan::{frame::FrameLayoutPlan, PlannedOpKind},
    wasm::semantic_ir::SemanticOpKind,
};

use super::{
    input::{resolve_local_index, AlignedOp},
    ops::{
        lower_call_external, lower_call_indirect, lower_call_internal, lower_local_get,
        lower_local_set, lower_local_tee, lower_prefix_ops, lower_primitive,
    },
    state::{BlockState, ValueAlloc},
};

pub(super) fn lower_block_body_op(
    op: &AlignedOp<'_>,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    values: &mut ValueAlloc,
) -> Result<(), WasmError> {
    lower_prefix_ops(op, state, values)?;

    match (&op.semantic.kind, &op.planned.kind) {
        (SemanticOpKind::Primitive(kind), PlannedOpKind::Primitive(_))
            if matches!(
            kind,
            crate::vm::wasm::primitive_op::PrimitiveOpKind::Unreachable
        ) =>
        {
            Err(WasmError::internal(
                "unreachable must end a prepared LIR block, not appear in the body".into(),
            ))
        }
        (SemanticOpKind::Primitive(kind), PlannedOpKind::Primitive(_)) => {
            lower_primitive(kind, state, values)
        }
        (SemanticOpKind::LocalGet { .. }, PlannedOpKind::LocalGet { .. }) => {
            lower_local_get(resolve_local_index(op)?, state, frame, values)
        }
        (SemanticOpKind::LocalSet { .. }, PlannedOpKind::LocalSet { .. }) => {
            lower_local_set(resolve_local_index(op)?, state, frame)
        }
        (SemanticOpKind::LocalTee { .. }, PlannedOpKind::LocalTee { .. }) => {
            lower_local_tee(resolve_local_index(op)?, state, frame)
        }
        (
            SemanticOpKind::CallExternal { .. },
            PlannedOpKind::CallExternal {
                func_idx,
                args,
                results,
            },
        ) => {
            lower_call_external(*func_idx, *args, *results, state);
            Ok(())
        }
        (
            SemanticOpKind::CallInternal { .. },
            PlannedOpKind::CallInternal {
                callee,
                args,
                results,
            },
        ) => {
            lower_call_internal(*callee, *args, *results, state);
            Ok(())
        }
        (
            SemanticOpKind::CallIndirect { .. },
            PlannedOpKind::CallIndirect {
                type_idx,
                table_idx,
                index_slot,
                args,
                results,
            },
        ) => {
            lower_call_indirect(*type_idx, *table_idx, *index_slot, *args, *results, state);
            Ok(())
        }
        (SemanticOpKind::Block { .. }, PlannedOpKind::Marker(_))
        | (SemanticOpKind::Loop { .. }, PlannedOpKind::Marker(_))
        | (SemanticOpKind::End, PlannedOpKind::Marker(_)) => Ok(()),
        (SemanticOpKind::Else, PlannedOpKind::Marker(_)) => Err(
            WasmError::internal("else must end a prepared LIR block, not appear in the body".into()),
        ),
        (
            SemanticOpKind::If { .. }
            | SemanticOpKind::Br { .. }
            | SemanticOpKind::BrIf { .. }
            | SemanticOpKind::BrTable { .. }
            | SemanticOpKind::ReturnVoid
            | SemanticOpKind::ReturnOne
            | SemanticOpKind::Return { .. },
            _,
        ) => Err(WasmError::internal(
            "control-flow terminators must end a prepared LIR block".into(),
        )),
        _ => Err(WasmError::internal(
            "semantic/planned mismatch during prepared LIR body lowering".into(),
        )),
    }
}
