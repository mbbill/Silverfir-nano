//! Terminator lowering and successor selection.

use alloc::vec::Vec;

use crate::error::WasmError;
use crate::vm::{
    lir::{ir::LirTerminator, target::LirTarget},
    plan::frame::FrameLayoutPlan,
    wasm::{common::SemanticTarget, semantic_ir::SemanticOpKind},
};

use super::{
    edge::{br_table_edge, edge_to_target, goto_next, next_edge, EdgeMapping},
    input::{resolve_local_index, SemanticPlannedOp},
    ops::{
        lower_call_external, lower_call_indirect, lower_call_internal, lower_local_get,
        lower_local_set, lower_local_tee, lower_primitive,
    },
    state::{BlockState, ValueAlloc},
};

pub(super) fn lower_block_terminator(
    op: &SemanticPlannedOp<'_>,
    mapped: &[SemanticPlannedOp<'_>],
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    semantic_to_block: &[LirTarget],
    values: &mut ValueAlloc,
) -> Result<LirTerminator, WasmError> {
    match &op.semantic.kind {
        SemanticOpKind::Primitive(kind)
            if matches!(
                kind,
                crate::vm::wasm::primitive_op::PrimitiveOpKind::Unreachable
            ) =>
        {
            Ok(LirTerminator::TrapUnreachable)
        }
        SemanticOpKind::Primitive(kind) => {
            lower_primitive(kind, state, values)?;
            goto_next(op.semantic, state, semantic_to_block)
        }
        SemanticOpKind::LocalGet { .. } => {
            lower_local_get(resolve_local_index(op)?, state, frame, values)?;
            goto_next(op.semantic, state, semantic_to_block)
        }
        SemanticOpKind::LocalSet { .. } => {
            lower_local_set(resolve_local_index(op)?, state, frame)?;
            goto_next(op.semantic, state, semantic_to_block)
        }
        SemanticOpKind::LocalTee { .. } => {
            lower_local_tee(resolve_local_index(op)?, state, frame)?;
            goto_next(op.semantic, state, semantic_to_block)
        }
        SemanticOpKind::Block { .. } | SemanticOpKind::Loop { .. } => {
            goto_next(op.semantic, state, semantic_to_block)
        }
        SemanticOpKind::If { .. } => {
            let cond = state.pop_one()?;
            let then_edge = next_edge(op.semantic, state, semantic_to_block)?;
            let else_edge = edge_to_target(
                resolve_if_else_target(op, mapped)?,
                state,
                EdgeMapping::Identity,
                semantic_to_block,
            )?;
            Ok(LirTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            })
        }
        SemanticOpKind::Else => {
            if let Some(target) = op.semantic.alt {
                Ok(LirTerminator::Goto(edge_to_target(
                    target,
                    state,
                    EdgeMapping::Identity,
                    semantic_to_block,
                )?))
            } else {
                goto_next(op.semantic, state, semantic_to_block)
            }
        }
        SemanticOpKind::End => goto_next(op.semantic, state, semantic_to_block),
        SemanticOpKind::Br { stack_drop, arity } => Ok(LirTerminator::Goto(edge_to_target(
            op.semantic
                .alt
                .ok_or_else(|| WasmError::invalid("semantic br missing target".into()))?,
            state,
            EdgeMapping::Branch {
                stack_drop: *stack_drop,
                arity: *arity,
            },
            semantic_to_block,
        )?)),
        SemanticOpKind::BrIf { stack_drop, arity } => {
            let cond = state.pop_one()?;
            let then_edge = edge_to_target(
                op.semantic
                    .alt
                    .ok_or_else(|| WasmError::invalid("semantic br_if missing target".into()))?,
                state,
                EdgeMapping::Branch {
                    stack_drop: *stack_drop,
                    arity: *arity,
                },
                semantic_to_block,
            )?;
            let else_edge = next_edge(op.semantic, state, semantic_to_block)?;
            Ok(LirTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            })
        }
        SemanticOpKind::BrTable { entries } => {
            let index = state.pop_one()?;
            let entries = entries
                .iter()
                .map(|entry| br_table_edge(entry, state, semantic_to_block))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(LirTerminator::BrTable { index, entries })
        }
        SemanticOpKind::CallExternal {
            func_idx,
            params,
            results,
        } => {
            lower_call_external(*func_idx, *params, *results, state, values)?;
            goto_next(op.semantic, state, semantic_to_block)
        }
        SemanticOpKind::CallInternal {
            callee,
            params,
            results,
        } => {
            lower_call_internal(*callee, *params, *results, state, values)?;
            goto_next(op.semantic, state, semantic_to_block)
        }
        SemanticOpKind::CallIndirect {
            type_idx,
            table_idx,
            params,
            results,
        } => {
            lower_call_indirect(*type_idx, *table_idx, *params, *results, state, values)?;
            goto_next(op.semantic, state, semantic_to_block)
        }
        SemanticOpKind::ReturnVoid => Ok(LirTerminator::Return { values: Vec::new() }),
        SemanticOpKind::ReturnOne => Ok(LirTerminator::Return {
            values: state.top_values(1)?,
        }),
        SemanticOpKind::Return { arity } => Ok(LirTerminator::Return {
            values: state.top_values(*arity as usize)?,
        }),
    }
}

fn resolve_if_else_target(
    op: &SemanticPlannedOp<'_>,
    mapped: &[SemanticPlannedOp<'_>],
) -> Result<SemanticTarget, WasmError> {
    let target = op
        .semantic
        .alt
        .ok_or_else(|| WasmError::invalid("semantic if missing alt target".into()))?;
    let Some(target_op) = mapped.get(target.index().as_usize()) else {
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
