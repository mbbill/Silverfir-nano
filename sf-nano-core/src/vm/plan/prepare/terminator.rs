//! Terminator lowering and successor selection.

use alloc::vec::Vec;

use crate::{
    error::WasmError,
    vm::{
        lir::ir::LirTerminator,
        plan::frame::FrameLayoutPlan,
        wasm::semantic_ir::SemanticOpKind,
    },
};

use super::{
    edge::{br_table_edge, edge_to_target, goto_next, next_edge, EdgeMapping},
    ops::{
        branch_payload, lower_boundary_primitive, lower_call_external, lower_call_indirect,
        lower_call_internal, lower_local_get, lower_local_set, lower_local_tee,
        lower_prefix_actions, lower_primitive, return_results,
    },
    state::{BlockState, EntryState, ValueAlloc},
    steps::PreparedOp,
};

pub(super) fn lower_block_terminator(
    semantic_index: usize,
    op: &PreparedOp<'_>,
    semantic_len: usize,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    semantic_to_block: &[crate::vm::lir::target::LirTarget],
    block_params: &[Vec<crate::vm::lir::ir::LirValue>],
    entry_states: &[EntryState],
    values: &mut ValueAlloc,
) -> Result<LirTerminator, WasmError> {
    lower_prefix_actions(op, state, values)?;

    match &op.semantic.kind {
        SemanticOpKind::Primitive(kind)
            if matches!(kind, crate::vm::wasm::primitive_op::PrimitiveOpKind::Unreachable) =>
        {
            Ok(LirTerminator::TrapUnreachable)
        }
        SemanticOpKind::Primitive(kind) => {
            if crate::vm::lir::leaf::is_boundary_primitive(kind) {
                lower_boundary_primitive(kind, state, frame)?;
            } else {
                lower_primitive(kind, state, values)?;
            }
            goto_next(
                semantic_index,
                semantic_len,
                state,
                semantic_to_block,
                block_params,
                entry_states,
            )
        }
        SemanticOpKind::LocalGet { idx } => {
            lower_local_get(*idx, state, frame, values)?;
            goto_next(
                semantic_index,
                semantic_len,
                state,
                semantic_to_block,
                block_params,
                entry_states,
            )
        }
        SemanticOpKind::LocalSet { idx } => {
            lower_local_set(*idx, state, frame)?;
            goto_next(
                semantic_index,
                semantic_len,
                state,
                semantic_to_block,
                block_params,
                entry_states,
            )
        }
        SemanticOpKind::LocalTee { idx } => {
            lower_local_tee(*idx, state, frame)?;
            goto_next(
                semantic_index,
                semantic_len,
                state,
                semantic_to_block,
                block_params,
                entry_states,
            )
        }
        SemanticOpKind::Block { .. } | SemanticOpKind::Loop { .. } => {
            goto_next(
                semantic_index,
                semantic_len,
                state,
                semantic_to_block,
                block_params,
                entry_states,
            )
        }
        SemanticOpKind::If { else_target, .. } => {
            let cond = state.pop_one()?;
            let then_edge = next_edge(
                semantic_index,
                semantic_len,
                state,
                semantic_to_block,
                block_params,
                entry_states,
            )?;
            let else_edge = edge_to_target(
                *else_target,
                state,
                EdgeMapping::Identity,
                semantic_to_block,
                block_params,
                entry_states,
            )?;
            Ok(LirTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            })
        }
        SemanticOpKind::Else { end_target } => Ok(LirTerminator::Goto(edge_to_target(
            *end_target,
            state,
            EdgeMapping::Identity,
            semantic_to_block,
            block_params,
            entry_states,
        )?)),
        SemanticOpKind::End => {
            goto_next(
                semantic_index,
                semantic_len,
                state,
                semantic_to_block,
                block_params,
                entry_states,
            )
        }
        SemanticOpKind::Br {
            stack_drop,
            arity,
            target,
        } => Ok(LirTerminator::Goto(edge_to_target(
            *target,
            state,
            EdgeMapping::TakenBranch {
                stack_drop: *stack_drop,
                payload: branch_payload(frame, *stack_drop, *arity),
            },
            semantic_to_block,
            block_params,
            entry_states,
        )?)),
        SemanticOpKind::BrIf {
            stack_drop,
            arity,
            target,
        } => {
            let cond = state.pop_one()?;
            let then_edge = edge_to_target(
                *target,
                state,
                EdgeMapping::TakenBranch {
                    stack_drop: *stack_drop,
                    payload: branch_payload(frame, *stack_drop, *arity),
                },
                semantic_to_block,
                block_params,
                entry_states,
            )?;
            let else_edge = next_edge(
                semantic_index,
                semantic_len,
                state,
                semantic_to_block,
                block_params,
                entry_states,
            )?;
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
                .map(|entry| {
                    br_table_edge(
                        entry,
                        branch_payload(frame, entry.stack_drop, entry.arity),
                        state,
                        semantic_to_block,
                        block_params,
                        entry_states,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(LirTerminator::BrTable { index, entries })
        }
        SemanticOpKind::CallExternal {
            func_idx,
            params,
            results,
        } => {
            lower_call_external(*func_idx, *params, *results, frame, state);
            goto_next(
                semantic_index,
                semantic_len,
                state,
                semantic_to_block,
                block_params,
                entry_states,
            )
        }
        SemanticOpKind::CallInternal {
            callee,
            params,
            results,
        } => {
            lower_call_internal(*callee, *params, *results, frame, state);
            goto_next(
                semantic_index,
                semantic_len,
                state,
                semantic_to_block,
                block_params,
                entry_states,
            )
        }
        SemanticOpKind::CallIndirect {
            type_idx,
            table_idx,
            params,
            results,
        } => {
            lower_call_indirect(*type_idx, *table_idx, *params, *results, frame, state);
            goto_next(
                semantic_index,
                semantic_len,
                state,
                semantic_to_block,
                block_params,
                entry_states,
            )
        }
        SemanticOpKind::ReturnVoid => Ok(LirTerminator::Return { results: None }),
        SemanticOpKind::ReturnOne => Ok(LirTerminator::Return {
            results: return_results(frame, 1),
        }),
        SemanticOpKind::Return { arity } => Ok(LirTerminator::Return {
            results: return_results(frame, *arity),
        }),
    }
}
