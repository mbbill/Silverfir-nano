//! Terminator lowering and successor selection.

use alloc::vec::Vec;

use crate::{
    error::WasmError,
    vm::{
        lir::ir::{LirBlock, LirEdge, LirInst, LirInstKind, LirTerminator},
        plan::frame::FrameLayoutPlan,
        wasm::{common::SemanticTarget, semantic_ir::SemanticOpKind},
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
    original_block_count: usize,
    extra_blocks_len: usize,
) -> Result<LoweredTerminator, WasmError> {
    lower_prefix_actions(op, state, values)?;

    match &op.semantic.kind {
        SemanticOpKind::Primitive(kind)
            if matches!(
                kind,
                crate::vm::wasm::primitive_op::PrimitiveOpKind::Unreachable
            ) =>
        {
            Ok(LoweredTerminator::new(LirTerminator::TrapUnreachable))
        }
        SemanticOpKind::Primitive(kind) => {
            if crate::vm::lir::leaf::is_boundary_primitive(kind) {
                lower_boundary_primitive(kind, state, frame)?;
            } else {
                lower_primitive(kind, state, values)?;
            }
            maybe_publish_live_window_for_targets(
                &[fallthrough_target(semantic_index, semantic_len)?],
                state,
                frame,
                entry_states,
            );
            Ok(LoweredTerminator::new(goto_next(
                semantic_index,
                semantic_len,
                state,
                semantic_to_block,
                block_params,
                entry_states,
            )?))
        }
        SemanticOpKind::LocalGet { idx } => {
            lower_local_get(*idx, state, frame, values)?;
            maybe_publish_live_window_for_targets(
                &[fallthrough_target(semantic_index, semantic_len)?],
                state,
                frame,
                entry_states,
            );
            Ok(LoweredTerminator::new(goto_next(
                semantic_index,
                semantic_len,
                state,
                semantic_to_block,
                block_params,
                entry_states,
            )?))
        }
        SemanticOpKind::LocalSet { idx } => {
            lower_local_set(*idx, state, frame)?;
            maybe_publish_live_window_for_targets(
                &[fallthrough_target(semantic_index, semantic_len)?],
                state,
                frame,
                entry_states,
            );
            Ok(LoweredTerminator::new(goto_next(
                semantic_index,
                semantic_len,
                state,
                semantic_to_block,
                block_params,
                entry_states,
            )?))
        }
        SemanticOpKind::LocalTee { idx } => {
            lower_local_tee(*idx, state, frame)?;
            maybe_publish_live_window_for_targets(
                &[fallthrough_target(semantic_index, semantic_len)?],
                state,
                frame,
                entry_states,
            );
            Ok(LoweredTerminator::new(goto_next(
                semantic_index,
                semantic_len,
                state,
                semantic_to_block,
                block_params,
                entry_states,
            )?))
        }
        SemanticOpKind::Block { .. } | SemanticOpKind::Loop { .. } => {
            maybe_publish_live_window_for_targets(
                &[fallthrough_target(semantic_index, semantic_len)?],
                state,
                frame,
                entry_states,
            );
            Ok(LoweredTerminator::new(goto_next(
                semantic_index,
                semantic_len,
                state,
                semantic_to_block,
                block_params,
                entry_states,
            )?))
        }
        SemanticOpKind::If { else_target, .. } => {
            let cond = state.pop_one()?;
            maybe_publish_live_window_for_targets(
                &[
                    fallthrough_target(semantic_index, semantic_len)?,
                    *else_target,
                ],
                state,
                frame,
                entry_states,
            );
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
            Ok(LoweredTerminator::new(LirTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            }))
        }
        SemanticOpKind::Else { end_target } => {
            maybe_publish_live_window_for_targets(&[*end_target], state, frame, entry_states);
            Ok(LoweredTerminator::new(LirTerminator::Goto(edge_to_target(
                *end_target,
                state,
                EdgeMapping::Identity,
                semantic_to_block,
                block_params,
                entry_states,
            )?)))
        }
        SemanticOpKind::End => {
            maybe_publish_live_window_for_targets(
                &[fallthrough_target(semantic_index, semantic_len)?],
                state,
                frame,
                entry_states,
            );
            Ok(LoweredTerminator::new(goto_next(
                semantic_index,
                semantic_len,
                state,
                semantic_to_block,
                block_params,
                entry_states,
            )?))
        }
        SemanticOpKind::Br {
            stack_drop,
            arity,
            target,
        } => {
            if target_expects_canonical_payload(*target, *stack_drop, state, entry_states)? {
                publish_taken_branch_payload_at(*stack_drop, *arity, state, frame)?;
            }
            Ok(LoweredTerminator::new(LirTerminator::Goto(edge_to_target(
                *target,
                state,
                EdgeMapping::TakenBranch {
                    stack_drop: *stack_drop,
                    payload: branch_payload(frame, state.height(), *stack_drop, *arity),
                },
                semantic_to_block,
                block_params,
                entry_states,
            )?)))
        }
        SemanticOpKind::BrIf {
            stack_drop,
            arity,
            target,
        } => {
            let cond = state.pop_one()?;
            let fallthrough = fallthrough_target(semantic_index, semantic_len)?;
            let needs_then_bridge =
                target_expects_canonical_payload(*target, *stack_drop, state, entry_states)?
                    && *arity != 0;
            if needs_then_bridge {
                maybe_publish_live_window_for_targets(&[fallthrough], state, frame, entry_states);
                let payload = state.top_values(*arity as usize).map_err(|err| {
                    WasmError::internal(alloc::format!(
                        "taken br_if could not bind {} payload values for synthetic then block: {}",
                        arity,
                        err
                    ))
                })?;
                let then_block_id = crate::vm::lir::target::LirTarget(
                    (original_block_count + extra_blocks_len) as u32,
                );
                let then_params = values.many(*arity as usize);
                let payload_span = branch_payload(frame, state.height(), *stack_drop, *arity)
                    .ok_or_else(|| {
                        WasmError::internal(
                            "taken br_if with nonzero arity must produce a branch payload span"
                                .into(),
                        )
                    })?;
                let target_block = *semantic_to_block
                    .get(target.index().as_usize())
                    .ok_or_else(|| WasmError::invalid("edge target out of range".into()))?;
                let target_params = block_params
                    .get(target_block.as_usize())
                    .ok_or_else(|| WasmError::invalid("edge target out of range".into()))?;
                if !target_params.is_empty() {
                    return Err(WasmError::internal(
                        "synthetic br_if then bridge requires a canonical-only branch target"
                            .into(),
                    ));
                }
                let mut then_ops = Vec::with_capacity(*arity as usize);
                for (offset, param) in then_params.iter().copied().enumerate() {
                    then_ops.push(LirInst {
                        kind: LirInstKind::StoreSlot {
                            slot: payload_span.start.advance(offset as u16),
                            src: param,
                        },
                    });
                }
                let then_edge = LirEdge {
                    target: then_block_id,
                    bindings: then_params
                        .iter()
                        .copied()
                        .zip(payload.into_iter())
                        .map(|(param, value)| crate::vm::lir::ir::LirBinding { param, value })
                        .collect(),
                };
                let bridge_target = edge_to_target(
                    *target,
                    state,
                    EdgeMapping::TakenBranch {
                        stack_drop: *stack_drop,
                        payload: None,
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
                let bridge_block = LirBlock {
                    id: then_block_id,
                    params: then_params,
                    ops: then_ops,
                    terminator: LirTerminator::Goto(bridge_target),
                };
                Ok(LoweredTerminator {
                    terminator: LirTerminator::Branch {
                        cond,
                        then_edge,
                        else_edge,
                    },
                    extra_blocks: alloc::vec![bridge_block],
                })
            } else {
                if target_expects_canonical_payload(*target, *stack_drop, state, entry_states)? {
                    publish_taken_branch_payload_at(*stack_drop, *arity, state, frame)?;
                }
                maybe_publish_live_window_for_targets(&[fallthrough], state, frame, entry_states);
                let then_edge = edge_to_target(
                    *target,
                    state,
                    EdgeMapping::TakenBranch {
                        stack_drop: *stack_drop,
                        payload: branch_payload(frame, state.height(), *stack_drop, *arity),
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
                Ok(LoweredTerminator::new(LirTerminator::Branch {
                    cond,
                    then_edge,
                    else_edge,
                }))
            }
        }
        SemanticOpKind::BrTable { entries } => {
            let index = state.pop_one()?;
            maybe_publish_taken_branch_payloads(entries, state, frame, entry_states)?;
            let entries = entries
                .iter()
                .map(|entry| {
                    br_table_edge(
                        entry,
                        branch_payload(frame, state.height(), entry.stack_drop, entry.arity),
                        state,
                        semantic_to_block,
                        block_params,
                        entry_states,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(LoweredTerminator::new(LirTerminator::BrTable {
                index,
                entries,
            }))
        }
        SemanticOpKind::CallExternal {
            func_idx,
            params,
            results,
        } => {
            lower_call_external(*func_idx, *params, *results, frame, state);
            maybe_publish_live_window_for_targets(
                &[fallthrough_target(semantic_index, semantic_len)?],
                state,
                frame,
                entry_states,
            );
            Ok(LoweredTerminator::new(goto_next(
                semantic_index,
                semantic_len,
                state,
                semantic_to_block,
                block_params,
                entry_states,
            )?))
        }
        SemanticOpKind::CallInternal {
            callee,
            params,
            results,
        } => {
            lower_call_internal(*callee, *params, *results, frame, state);
            maybe_publish_live_window_for_targets(
                &[fallthrough_target(semantic_index, semantic_len)?],
                state,
                frame,
                entry_states,
            );
            Ok(LoweredTerminator::new(goto_next(
                semantic_index,
                semantic_len,
                state,
                semantic_to_block,
                block_params,
                entry_states,
            )?))
        }
        SemanticOpKind::CallIndirect {
            type_idx,
            table_idx,
            params,
            results,
        } => {
            lower_call_indirect(*type_idx, *table_idx, *params, *results, frame, state);
            maybe_publish_live_window_for_targets(
                &[fallthrough_target(semantic_index, semantic_len)?],
                state,
                frame,
                entry_states,
            );
            Ok(LoweredTerminator::new(goto_next(
                semantic_index,
                semantic_len,
                state,
                semantic_to_block,
                block_params,
                entry_states,
            )?))
        }
        SemanticOpKind::ReturnVoid => Ok(LoweredTerminator::new(LirTerminator::Return {
            results: None,
        })),
        SemanticOpKind::ReturnOne => Ok(LoweredTerminator::new(LirTerminator::Return {
            results: {
                canonicalize_return_results(state, frame, values, 1);
                return_results(frame, 1)
            },
        })),
        SemanticOpKind::Return { arity } => Ok(LoweredTerminator::new(LirTerminator::Return {
            results: {
                canonicalize_return_results(state, frame, values, *arity);
                return_results(frame, *arity)
            },
        })),
    }
}

pub(super) struct LoweredTerminator {
    pub(super) terminator: LirTerminator,
    pub(super) extra_blocks: Vec<LirBlock>,
}

impl LoweredTerminator {
    fn new(terminator: LirTerminator) -> Self {
        Self {
            terminator,
            extra_blocks: Vec::new(),
        }
    }
}

pub(super) fn fallthrough_target(
    semantic_index: usize,
    semantic_len: usize,
) -> Result<SemanticTarget, WasmError> {
    let next = semantic_index
        .checked_add(1)
        .filter(|next| *next < semantic_len)
        .ok_or_else(|| WasmError::invalid("missing fallthrough target".into()))?;
    Ok(SemanticTarget::new(next))
}

pub(super) fn maybe_publish_live_window_for_targets(
    targets: &[SemanticTarget],
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    entry_states: &[EntryState],
) {
    if state.live().is_empty() {
        return;
    }

    let max_target_spill_depth = targets
        .iter()
        .filter_map(|target| entry_states.get(target.index().as_usize()))
        .filter(|entry| entry.stack_height == state.height())
        .map(|entry| entry.spill_depth)
        .max()
        .unwrap_or(state.spill_depth());
    if max_target_spill_depth <= state.spill_depth() {
        return;
    }

    let publish_count = max_target_spill_depth.saturating_sub(state.spill_depth()) as usize;
    let base_slot = frame.operand_slot(state.spill_depth());
    let prefix_values = state.live()[..publish_count].to_vec();
    for (offset, value) in prefix_values.into_iter().enumerate() {
        state.ops.push(LirInst {
            kind: LirInstKind::StoreSlot {
                slot: base_slot.advance(offset as u16),
                src: value,
            },
        });
    }
}

pub(super) fn canonicalize_live_window_for_target(
    target: SemanticTarget,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    entry_states: &[EntryState],
) -> Result<(), WasmError> {
    let target_entry = *entry_states
        .get(target.index().as_usize())
        .ok_or_else(|| WasmError::invalid("edge target out of range".into()))?;
    if target_entry.stack_height != state.height() || target_entry.spill_depth <= state.spill_depth()
    {
        return Ok(());
    }

    let publish_count = target_entry.spill_depth.saturating_sub(state.spill_depth());
    let base_slot = frame.operand_slot(state.spill_depth());
    let spilled = state.spill_prefix(publish_count)?;
    for (offset, value) in spilled.into_iter().enumerate() {
        state.ops.push(LirInst {
            kind: LirInstKind::StoreSlot {
                slot: base_slot.advance(offset as u16),
                src: value,
            },
        });
    }
    Ok(())
}

fn target_expects_canonical_payload(
    target: SemanticTarget,
    stack_drop: u32,
    state: &BlockState,
    entry_states: &[EntryState],
) -> Result<bool, WasmError> {
    let Some(entry) = entry_states.get(target.index().as_usize()) else {
        return Err(WasmError::invalid(
            "taken branch target out of range".into(),
        ));
    };
    Ok(entry.live_value_count() == 0
        && entry.stack_height == state.height().saturating_sub(stack_drop as u16))
}

fn publish_taken_branch_payload_at(
    stack_drop: u32,
    arity: u16,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
) -> Result<(), WasmError> {
    if arity == 0 {
        return Ok(());
    }
    let payload = state.top_values(arity as usize).map_err(|err| {
        WasmError::internal(alloc::format!(
            "taken branch could not publish {} payload values: {}",
            arity,
            err
        ))
    })?;
    let base_slot = frame.operand_slot(
        state
            .height()
            .saturating_sub(stack_drop as u16)
            .saturating_sub(arity),
    );
    for (offset, value) in payload.into_iter().enumerate() {
        state.ops.push(LirInst {
            kind: LirInstKind::StoreSlot {
                slot: base_slot.advance(offset as u16),
                src: value,
            },
        });
    }
    Ok(())
}

fn maybe_publish_taken_branch_payloads(
    entries: &[crate::vm::wasm::common::BrTableEntry],
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    entry_states: &[EntryState],
) -> Result<(), WasmError> {
    let mut published = alloc::vec::Vec::<u32>::new();
    for entry in entries {
        if !target_expects_canonical_payload(entry.target, entry.stack_drop, state, entry_states)? {
            continue;
        }
        if published.contains(&entry.stack_drop) {
            continue;
        }
        publish_taken_branch_payload_at(entry.stack_drop, entry.arity, state, frame)?;
        published.push(entry.stack_drop);
    }
    if published.is_empty() {
        return Ok(());
    }
    Ok(())
}

fn canonicalize_return_results(
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    values: &mut ValueAlloc,
    arity: u16,
) {
    if arity == 0 {
        return;
    }

    let src = frame.operand_slot(state.height().saturating_sub(arity));
    let dst = frame.operand_slot(0);
    if src == dst {
        return;
    }

    for offset in 0..arity as usize {
        let value = values.fresh();
        state.ops.push(crate::vm::lir::ir::LirInst {
            kind: crate::vm::lir::ir::LirInstKind::LoadSlot {
                slot: src.advance(offset as u16),
                dst: value,
            },
        });
        state.ops.push(crate::vm::lir::ir::LirInst {
            kind: crate::vm::lir::ir::LirInstKind::StoreSlot {
                slot: dst.advance(offset as u16),
                src: value,
            },
        });
    }
}
