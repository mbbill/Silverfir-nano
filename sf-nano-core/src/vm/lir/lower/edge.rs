//! CFG edge construction and successor-argument mapping.

use alloc::vec::Vec;

use crate::error::WasmError;
use crate::vm::{
    lir::{
        ir::{LirEdge, LirTerminator},
        target::LirTarget,
    },
    wasm::{
        common::{BrTableEntry, SemanticTarget},
        semantic_ir::SemanticOp,
    },
};

use super::state::BlockState;

pub(super) fn goto_next(
    semantic_op: &SemanticOp,
    state: &BlockState,
    semantic_to_block: &[LirTarget],
) -> Result<LirTerminator, WasmError> {
    Ok(LirTerminator::Goto(next_edge(semantic_op, state, semantic_to_block)?))
}

pub(super) fn next_edge(
    semantic_op: &SemanticOp,
    state: &BlockState,
    semantic_to_block: &[LirTarget],
) -> Result<LirEdge, WasmError> {
    let next = semantic_op
        .next
        .ok_or_else(|| WasmError::invalid("missing fallthrough target".into()))?;
    edge_to_target(
        SemanticTarget::new(next.as_usize()),
        state,
        EdgeMapping::Identity,
        semantic_to_block,
    )
}

pub(super) fn edge_to_target(
    target: SemanticTarget,
    state: &BlockState,
    mapping: EdgeMapping,
    semantic_to_block: &[LirTarget],
) -> Result<LirEdge, WasmError> {
    let target_height = match mapping {
        EdgeMapping::Identity => state.height(),
        EdgeMapping::Branch { stack_drop, .. } => state.height().saturating_sub(stack_drop as u16),
    };
    let args = (0..target_height as usize)
        .map(|target_index| match mapping {
            EdgeMapping::Identity => state.value_at(target_index),
            EdgeMapping::Branch { stack_drop, arity } => {
                let payload_start = target_height as usize - arity as usize;
                let current_index = if target_index < payload_start {
                    target_index
                } else {
                    target_index.saturating_add(stack_drop as usize)
                };
                state.value_at(current_index)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(LirEdge {
        target: *semantic_to_block
            .get(target.index().as_usize())
            .ok_or_else(|| WasmError::invalid("edge target out of range".into()))?,
        args,
    })
}

pub(super) fn br_table_edge(
    entry: &BrTableEntry,
    state: &BlockState,
    semantic_to_block: &[LirTarget],
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
        semantic_to_block,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EdgeMapping {
    Identity,
    Branch { stack_drop: u32, arity: u16 },
}
