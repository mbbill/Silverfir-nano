//! LIR block lowering and terminator construction.

use alloc::vec::Vec;

use crate::error::WasmError;
use crate::vm::{
    lir::{
        ir::{LirInst, LirInstKind, LirTerminator},
        leaf::LirLeafOp,
        target::LirTarget,
    },
    plan::{config::PlanConfig, PlannedProgram},
    wasm::semantic_ir::SemanticOpKind,
};

use super::{
    edge::{br_table_edge, edge_to_target, goto_next, next_edge, EdgeMapping},
    input::SemanticPlannedOp,
    ops::{
        lower_call_external, lower_call_indirect, lower_call_internal, lower_local_get,
        lower_local_set, lower_local_tee, lower_primitive, lower_synthetic_prefix_op,
    },
    stack::{materialize_top_values, pop_one},
    state::{BlockState, ValueAlloc},
};

#[derive(Clone, Debug)]
pub(super) struct LoweredBlock {
    pub(super) ops: Vec<LirInst>,
    pub(super) terminator: LirTerminator,
}

pub(super) fn lower_block_range(
    semantic_range: core::ops::Range<usize>,
    mut state: BlockState,
    mapped: &[SemanticPlannedOp<'_>],
    planned: &PlannedProgram,
    config: PlanConfig,
    semantic_to_block: &[LirTarget],
    values: &mut ValueAlloc,
) -> Result<LoweredBlock, WasmError> {
    let last_index = semantic_range
        .end
        .checked_sub(1)
        .ok_or_else(|| WasmError::internal("LIR block cannot be empty".into()))?;

    for semantic_index in semantic_range.start..last_index {
        lower_block_body_op(&mapped[semantic_index], &mut state, planned, config, values)?;
    }

    let terminator = lower_block_end_op(
        &mapped[last_index],
        &mut state,
        planned,
        config,
        semantic_to_block,
        values,
    )?;

    Ok(LoweredBlock {
        ops: state.ops,
        terminator,
    })
}

fn lower_block_body_op(
    op: &SemanticPlannedOp<'_>,
    state: &mut BlockState,
    planned: &PlannedProgram,
    config: PlanConfig,
    values: &mut ValueAlloc,
) -> Result<(), WasmError> {
    for prefix_op in &op.prefix {
        lower_synthetic_prefix_op(prefix_op, state, planned.frame, values)?;
    }

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
            lower_primitive(kind, state, planned.frame, values);
            Ok(())
        }
        SemanticOpKind::LocalGet { .. } => {
            lower_local_get(&op.planned.kind, state, planned.frame, values)
        }
        SemanticOpKind::LocalSet { .. } => {
            lower_local_set(&op.planned.kind, state, planned.frame, values)
        }
        SemanticOpKind::LocalTee { .. } => {
            lower_local_tee(&op.planned.kind, state, planned.frame, values)
        }
        SemanticOpKind::Block { .. }
        | SemanticOpKind::Loop { .. }
        | SemanticOpKind::Else
        | SemanticOpKind::End => Ok(()),
        SemanticOpKind::CallExternal {
            func_idx,
            params,
            results,
        } => {
            lower_call_external(*func_idx, *params, *results, state, planned.frame, values);
            Ok(())
        }
        SemanticOpKind::CallInternal {
            callee,
            params,
            results,
        } => {
            lower_call_internal(*callee, *params, *results, state, planned.frame, values);
            Ok(())
        }
        SemanticOpKind::CallIndirect {
            type_idx,
            table_idx,
            params,
            results,
        } => {
            lower_call_indirect(
                *type_idx,
                *table_idx,
                *params,
                *results,
                state,
                planned.frame,
                values,
            );
            Ok(())
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

fn lower_block_end_op(
    op: &SemanticPlannedOp<'_>,
    state: &mut BlockState,
    planned: &PlannedProgram,
    config: PlanConfig,
    semantic_to_block: &[LirTarget],
    values: &mut ValueAlloc,
) -> Result<LirTerminator, WasmError> {
    for planned_op in &op.prefix {
        lower_synthetic_prefix_op(planned_op, state, planned.frame, values)?;
    }

    match &op.semantic.kind {
        SemanticOpKind::Primitive(kind)
            if matches!(
                kind,
                crate::vm::wasm::primitive_op::PrimitiveOpKind::Unreachable
            ) =>
        {
            state.ops.push(LirInst {
                kind: LirInstKind::Leaf {
                    op: LirLeafOp::from(kind.clone()),
                    args: Vec::new(),
                    results: Vec::new(),
                },
            });
            Ok(LirTerminator::TrapUnreachable)
        }
        SemanticOpKind::Primitive(kind) => {
            lower_primitive(kind, state, planned.frame, values);
            goto_next(
                op.semantic,
                state,
                planned,
                config,
                semantic_to_block,
                values,
            )
        }
        SemanticOpKind::LocalGet { .. } => {
            lower_local_get(&op.planned.kind, state, planned.frame, values)?;
            goto_next(
                op.semantic,
                state,
                planned,
                config,
                semantic_to_block,
                values,
            )
        }
        SemanticOpKind::LocalSet { .. } => {
            lower_local_set(&op.planned.kind, state, planned.frame, values)?;
            goto_next(
                op.semantic,
                state,
                planned,
                config,
                semantic_to_block,
                values,
            )
        }
        SemanticOpKind::LocalTee { .. } => {
            lower_local_tee(&op.planned.kind, state, planned.frame, values)?;
            goto_next(
                op.semantic,
                state,
                planned,
                config,
                semantic_to_block,
                values,
            )
        }
        SemanticOpKind::Block { .. } | SemanticOpKind::Loop { .. } => goto_next(
            op.semantic,
            state,
            planned,
            config,
            semantic_to_block,
            values,
        ),
        SemanticOpKind::If { .. } => {
            let cond = pop_one(state, planned.frame, values);
            let then_edge = next_edge(
                op.semantic,
                state,
                planned,
                config,
                semantic_to_block,
                values,
            )?;
            let else_edge = edge_to_target(
                op.semantic
                    .alt
                    .ok_or_else(|| WasmError::invalid("semantic if missing alt target".into()))?,
                state,
                EdgeMapping::Identity,
                planned,
                config,
                semantic_to_block,
                values,
            )?;
            Ok(LirTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            })
        }
        SemanticOpKind::Else | SemanticOpKind::End => goto_next(
            op.semantic,
            state,
            planned,
            config,
            semantic_to_block,
            values,
        ),
        SemanticOpKind::Br { stack_drop, arity } => Ok(LirTerminator::Goto(edge_to_target(
            op.semantic
                .alt
                .ok_or_else(|| WasmError::invalid("semantic br missing target".into()))?,
            state,
            EdgeMapping::Branch {
                stack_drop: *stack_drop,
                arity: *arity,
            },
            planned,
            config,
            semantic_to_block,
            values,
        )?)),
        SemanticOpKind::BrIf { stack_drop, arity } => {
            let cond = pop_one(state, planned.frame, values);
            let then_edge = edge_to_target(
                op.semantic
                    .alt
                    .ok_or_else(|| WasmError::invalid("semantic br_if missing target".into()))?,
                state,
                EdgeMapping::Branch {
                    stack_drop: *stack_drop,
                    arity: *arity,
                },
                planned,
                config,
                semantic_to_block,
                values,
            )?;
            let else_edge = next_edge(
                op.semantic,
                state,
                planned,
                config,
                semantic_to_block,
                values,
            )?;
            Ok(LirTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            })
        }
        SemanticOpKind::BrTable { entries } => {
            let index = pop_one(state, planned.frame, values);
            let entries = entries
                .iter()
                .map(|entry| {
                    br_table_edge(entry, state, planned, config, semantic_to_block, values)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(LirTerminator::BrTable { index, entries })
        }
        SemanticOpKind::CallExternal {
            func_idx,
            params,
            results,
        } => {
            lower_call_external(*func_idx, *params, *results, state, planned.frame, values);
            goto_next(
                op.semantic,
                state,
                planned,
                config,
                semantic_to_block,
                values,
            )
        }
        SemanticOpKind::CallInternal {
            callee,
            params,
            results,
        } => {
            lower_call_internal(*callee, *params, *results, state, planned.frame, values);
            goto_next(
                op.semantic,
                state,
                planned,
                config,
                semantic_to_block,
                values,
            )
        }
        SemanticOpKind::CallIndirect {
            type_idx,
            table_idx,
            params,
            results,
        } => {
            lower_call_indirect(
                *type_idx,
                *table_idx,
                *params,
                *results,
                state,
                planned.frame,
                values,
            );
            goto_next(
                op.semantic,
                state,
                planned,
                config,
                semantic_to_block,
                values,
            )
        }
        SemanticOpKind::ReturnVoid => Ok(LirTerminator::Return { values: Vec::new() }),
        SemanticOpKind::ReturnOne => Ok(LirTerminator::Return {
            values: materialize_top_values(state, 1, planned.frame, values),
        }),
        SemanticOpKind::Return { arity } => Ok(LirTerminator::Return {
            values: materialize_top_values(state, *arity as usize, planned.frame, values),
        }),
    }
}
