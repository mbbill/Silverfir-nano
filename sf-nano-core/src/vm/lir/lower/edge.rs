//! CFG edge construction and successor-argument mapping.

use crate::error::WasmError;
use crate::vm::{
    lir::{
        ir::{LirEdge, LirTerminator},
        target::LirTarget,
    },
    plan::{config::PlanConfig, PlannedProgram},
    wasm::{
        common::{BrTableEntry, SemanticTarget},
        semantic_ir::SemanticOp,
    },
};

use super::{
    stack::materialize_stack_index,
    state::{BlockState, ValueAlloc},
};

pub(super) fn goto_next(
    semantic_op: &SemanticOp,
    state: &mut BlockState,
    planned: &PlannedProgram,
    config: PlanConfig,
    semantic_to_block: &[LirTarget],
    values: &mut ValueAlloc,
) -> Result<LirTerminator, WasmError> {
    Ok(LirTerminator::Goto(next_edge(
        semantic_op,
        state,
        planned,
        config,
        semantic_to_block,
        values,
    )?))
}

pub(super) fn next_edge(
    semantic_op: &SemanticOp,
    state: &mut BlockState,
    planned: &PlannedProgram,
    config: PlanConfig,
    semantic_to_block: &[LirTarget],
    values: &mut ValueAlloc,
) -> Result<LirEdge, WasmError> {
    let next = semantic_op
        .next
        .ok_or_else(|| WasmError::invalid("missing fallthrough target".into()))?;
    edge_to_target(
        SemanticTarget::new(next.as_usize()),
        state,
        EdgeMapping::Identity,
        planned,
        config,
        semantic_to_block,
        values,
    )
}

pub(super) fn edge_to_target(
    target: SemanticTarget,
    state: &mut BlockState,
    mapping: EdgeMapping,
    planned: &PlannedProgram,
    config: PlanConfig,
    semantic_to_block: &[LirTarget],
    values: &mut ValueAlloc,
) -> Result<LirEdge, WasmError> {
    let target_height = match mapping {
        EdgeMapping::Identity => state.height(),
        EdgeMapping::Branch { stack_drop, .. } => state.height().saturating_sub(stack_drop as u16),
    };
    let target_cached = core::cmp::min(target_height as usize, config.tos_register_count as usize);
    let start = target_height as usize - target_cached;
    let tos = (start..target_height as usize)
        .map(|target_index| match mapping {
            EdgeMapping::Identity => {
                materialize_stack_index(state, planned.frame, values, target_index as u16)
            }
            EdgeMapping::Branch { stack_drop, arity } => {
                let payload_start = target_height as usize - arity as usize;
                let current_index = if target_index < payload_start {
                    target_index as u16
                } else {
                    target_index.saturating_add(stack_drop as usize) as u16
                };
                materialize_stack_index(state, planned.frame, values, current_index)
            }
        })
        .collect();

    Ok(LirEdge {
        target: *semantic_to_block
            .get(target.index().as_usize())
            .ok_or_else(|| WasmError::invalid("edge target out of range".into()))?,
        tos,
    })
}

pub(super) fn br_table_edge(
    entry: &BrTableEntry,
    state: &mut BlockState,
    planned: &PlannedProgram,
    config: PlanConfig,
    semantic_to_block: &[LirTarget],
    values: &mut ValueAlloc,
) -> Result<LirEdge, WasmError> {
    edge_to_target(
        entry
            .target
            .ok_or_else(|| WasmError::invalid("br_table entry missing target".into()))?,
        state,
        EdgeMapping::Branch {
            stack_drop: entry.stack_drop,
            arity: entry.arity,
        },
        planned,
        config,
        semantic_to_block,
        values,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EdgeMapping {
    Identity,
    Branch { stack_drop: u32, arity: u16 },
}
