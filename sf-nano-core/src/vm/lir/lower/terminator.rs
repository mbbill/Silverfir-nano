//! Terminator lowering and successor selection.

use alloc::vec::Vec;

use crate::error::WasmError;
use crate::vm::{
    lir::{ir::LirTerminator, target::LirTarget},
    plan::{frame::FrameLayoutPlan, PlannedOpKind},
    wasm::{common::SemanticTarget, semantic_ir::SemanticOpKind},
};

use super::{
    edge::{br_table_edge, edge_to_target, goto_next, next_edge, EdgeMapping},
    input::{resolve_local_index, AlignedOp},
    ops::{
        lower_call_external, lower_call_indirect, lower_call_internal, lower_local_get,
        lower_local_set, lower_local_tee, lower_prefix_ops, lower_primitive,
    },
    state::{BlockState, EntryState, ValueAlloc},
};

pub(super) fn lower_block_terminator(
    op: &AlignedOp<'_>,
    aligned: &[AlignedOp<'_>],
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    semantic_to_block: &[LirTarget],
    entry_states: &[EntryState],
    values: &mut ValueAlloc,
) -> Result<LirTerminator, WasmError> {
    lower_prefix_ops(op, state, values)?;

    match (&op.semantic.kind, &op.planned.kind) {
        (SemanticOpKind::Primitive(kind), PlannedOpKind::Primitive(_))
            if matches!(
            kind,
            crate::vm::wasm::primitive_op::PrimitiveOpKind::Unreachable
        ) =>
        {
            Ok(LirTerminator::TrapUnreachable)
        }
        (SemanticOpKind::Primitive(kind), PlannedOpKind::Primitive(_)) => {
            lower_primitive(kind, state, values)?;
            goto_next(op.semantic, state, semantic_to_block, entry_states)
        }
        (SemanticOpKind::LocalGet { .. }, PlannedOpKind::LocalGet { .. }) => {
            lower_local_get(resolve_local_index(op)?, state, frame, values)?;
            goto_next(op.semantic, state, semantic_to_block, entry_states)
        }
        (SemanticOpKind::LocalSet { .. }, PlannedOpKind::LocalSet { .. }) => {
            lower_local_set(resolve_local_index(op)?, state, frame)?;
            goto_next(op.semantic, state, semantic_to_block, entry_states)
        }
        (SemanticOpKind::LocalTee { .. }, PlannedOpKind::LocalTee { .. }) => {
            lower_local_tee(resolve_local_index(op)?, state, frame)?;
            goto_next(op.semantic, state, semantic_to_block, entry_states)
        }
        (SemanticOpKind::Block { .. }, PlannedOpKind::Marker(_))
        | (SemanticOpKind::Loop { .. }, PlannedOpKind::Marker(_)) => {
            goto_next(op.semantic, state, semantic_to_block, entry_states)
        }
        (SemanticOpKind::If { .. }, PlannedOpKind::Marker(_)) => {
            let cond = state.pop_one()?;
            let then_edge = next_edge(op.semantic, state, semantic_to_block, entry_states)?;
            let else_edge = edge_to_target(
                resolve_if_else_target(op, aligned)?,
                state,
                EdgeMapping::Identity,
                semantic_to_block,
                entry_states,
            )?;
            Ok(LirTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            })
        }
        (SemanticOpKind::Else, PlannedOpKind::Marker(_)) => {
            if let Some(target) = op.semantic.alt {
                Ok(LirTerminator::Goto(edge_to_target(
                    target,
                    state,
                    EdgeMapping::Identity,
                    semantic_to_block,
                    entry_states,
                )?))
            } else {
                goto_next(op.semantic, state, semantic_to_block, entry_states)
            }
        }
        (SemanticOpKind::End, PlannedOpKind::Marker(_)) => {
            goto_next(op.semantic, state, semantic_to_block, entry_states)
        }
        (SemanticOpKind::Br { stack_drop, .. }, PlannedOpKind::Branch { .. }) => Ok(
            LirTerminator::Goto(edge_to_target(
            op.semantic
                .alt
                .ok_or_else(|| WasmError::invalid("semantic br missing target".into()))?,
            state,
            EdgeMapping::TakenBranch {
                stack_drop: *stack_drop,
                payload: branch_payload(op),
            },
            semantic_to_block,
            entry_states,
        )?)),
        (SemanticOpKind::BrIf { stack_drop, .. }, PlannedOpKind::Branch { .. }) => {
            let cond = state.pop_one()?;
            let then_edge = edge_to_target(
                op.semantic
                .alt
                    .ok_or_else(|| WasmError::invalid("semantic br_if missing target".into()))?,
                state,
                EdgeMapping::TakenBranch {
                    stack_drop: *stack_drop,
                    payload: branch_payload(op),
                },
                semantic_to_block,
                entry_states,
            )?;
            let else_edge = next_edge(op.semantic, state, semantic_to_block, entry_states)?;
            Ok(LirTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            })
        }
        (
            SemanticOpKind::BrTable { entries },
            PlannedOpKind::BrTable {
                entries: planned_entries,
                ..
            },
        ) => {
            let index = state.pop_one()?;
            if entries.len() != planned_entries.len() {
                return Err(WasmError::internal(
                    "semantic/planned br_table entry count mismatch during prepared LIR lowering"
                        .into(),
                ));
            }
            let entries = entries
                .iter()
                .zip(planned_entries.iter())
                .map(|(entry, planned_entry)| {
                    br_table_edge(
                        entry,
                        planned_entry.payload,
                        state,
                        semantic_to_block,
                        entry_states,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(LirTerminator::BrTable { index, entries })
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
            goto_next(op.semantic, state, semantic_to_block, entry_states)
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
            goto_next(op.semantic, state, semantic_to_block, entry_states)
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
            goto_next(op.semantic, state, semantic_to_block, entry_states)
        }
        (
            SemanticOpKind::ReturnVoid,
            PlannedOpKind::Return { results },
        )
        | (
            SemanticOpKind::ReturnOne,
            PlannedOpKind::Return { results },
        )
        | (
            SemanticOpKind::Return { .. },
            PlannedOpKind::Return { results },
        ) => Ok(LirTerminator::Return { results: *results }),
        _ => Err(WasmError::internal(
            "semantic/planned mismatch during prepared LIR terminator lowering".into(),
        )),
    }
}

fn resolve_if_else_target(
    op: &AlignedOp<'_>,
    aligned: &[AlignedOp<'_>],
) -> Result<SemanticTarget, WasmError> {
    let target = op
        .semantic
        .alt
        .ok_or_else(|| WasmError::invalid("semantic if missing alt target".into()))?;
    let Some(target_op) = aligned.get(target.index().as_usize()) else {
        return Err(WasmError::invalid(
            "semantic if alt target out of range".into(),
        ));
    };

    if matches!(target_op.semantic.kind, SemanticOpKind::Else) {
        target_op
            .semantic
            .next
            .map(|next| SemanticTarget::new(next.as_usize()))
            .ok_or_else(|| WasmError::invalid("semantic else missing body target".into()))
    } else {
        Ok(target)
    }
}

fn branch_payload(op: &AlignedOp<'_>) -> Option<crate::vm::plan::frame::FrameSpan> {
    match &op.planned.kind {
        PlannedOpKind::Branch { payload, .. } => *payload,
        _ => None,
    }
}
