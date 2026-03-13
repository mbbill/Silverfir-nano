//! CFG edge construction.

use alloc::vec::Vec;

use crate::{
    error::WasmError,
    vm::{
        lir::{
            ir::{LirBinding, LirEdge, LirTerminator, LirValue},
            target::LirTarget,
        },
        plan::frame::FrameSpan,
        wasm::common::{BrTableEntry, SemanticTarget},
    },
};

use super::state::{BlockState, EntryState};

pub(super) fn goto_next(
    semantic_index: usize,
    semantic_len: usize,
    state: &BlockState,
    semantic_to_block: &[LirTarget],
    block_params: &[Vec<LirValue>],
    entry_states: &[EntryState],
) -> Result<LirTerminator, WasmError> {
    Ok(LirTerminator::Goto(next_edge(
        semantic_index,
        semantic_len,
        state,
        semantic_to_block,
        block_params,
        entry_states,
    )?))
}

pub(super) fn next_edge(
    semantic_index: usize,
    semantic_len: usize,
    state: &BlockState,
    semantic_to_block: &[LirTarget],
    block_params: &[Vec<LirValue>],
    entry_states: &[EntryState],
) -> Result<LirEdge, WasmError> {
    let next = semantic_index
        .checked_add(1)
        .filter(|next| *next < semantic_len)
        .ok_or_else(|| WasmError::invalid("missing fallthrough target".into()))?;
    edge_to_target(
        SemanticTarget::new(next),
        state,
        EdgeMapping::Identity,
        semantic_to_block,
        block_params,
        entry_states,
    )
}

pub(super) fn edge_to_target(
    target: SemanticTarget,
    state: &BlockState,
    mapping: EdgeMapping,
    semantic_to_block: &[LirTarget],
    block_params: &[Vec<LirValue>],
    entry_states: &[EntryState],
) -> Result<LirEdge, WasmError> {
    let target_entry = *entry_states
        .get(target.index().as_usize())
        .ok_or_else(|| WasmError::invalid("edge target out of range".into()))?;
    let target_block = *semantic_to_block
        .get(target.index().as_usize())
        .ok_or_else(|| WasmError::invalid("edge target out of range".into()))?;
    let target_params = block_params
        .get(target_block.as_usize())
        .ok_or_else(|| WasmError::invalid("edge target out of range".into()))?;

    let mapped_height = match mapping {
        EdgeMapping::Identity => state.height(),
        EdgeMapping::TakenBranch { stack_drop, .. } => {
            state.height().saturating_sub(stack_drop as u16)
        }
    };
    if mapped_height != target_entry.stack_height {
        return Err(WasmError::internal(alloc::format!(
            "prepared LIR edge to semantic op {} computes stack height {}, but target expects {}",
            target.index().as_usize(),
            mapped_height,
            target_entry.stack_height,
        )));
    }

    let bindings = match mapping {
        EdgeMapping::Identity => {
            let live_values = state.top_values(target_entry.live_value_count() as usize)?;
            bind_values(target_params, &live_values)?
        }
        EdgeMapping::TakenBranch { payload, .. } => {
            if target_entry.live_value_count() != 0 {
                return Err(WasmError::internal(alloc::format!(
                    "taken branch to semantic op {} must enter with canonical frame payload only, but target expects {} live params (payload_slots={})",
                    target.index().as_usize(),
                    target_entry.live_value_count(),
                    payload.map_or(0, |span| span.count),
                )));
            }
            if !target_params.is_empty() {
                return Err(WasmError::internal(alloc::format!(
                    "taken branch to semantic op {} must not require live params when payload is canonical frame state",
                    target.index().as_usize(),
                )));
            }
            Vec::new()
        }
    };

    Ok(LirEdge {
        target: target_block,
        bindings,
    })
}

pub(super) fn br_table_edge(
    entry: &BrTableEntry,
    payload: Option<FrameSpan>,
    state: &BlockState,
    semantic_to_block: &[LirTarget],
    block_params: &[Vec<LirValue>],
    entry_states: &[EntryState],
) -> Result<LirEdge, WasmError> {
    edge_to_target(
        entry.target,
        state,
        EdgeMapping::TakenBranch {
            stack_drop: entry.stack_drop,
            payload,
        },
        semantic_to_block,
        block_params,
        entry_states,
    )
}

fn bind_values(target_params: &[LirValue], values: &[LirValue]) -> Result<Vec<LirBinding>, WasmError> {
    if target_params.len() != values.len() {
        return Err(WasmError::internal(alloc::format!(
            "prepared LIR edge binding mismatch: target expects {} params but source provides {} values",
            target_params.len(),
            values.len(),
        )));
    }

    Ok(target_params
        .iter()
        .zip(values.iter())
        .map(|(param, value)| LirBinding {
            param: *param,
            value: *value,
        })
        .collect())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EdgeMapping {
    Identity,
    TakenBranch {
        stack_drop: u32,
        payload: Option<FrameSpan>,
    },
}
