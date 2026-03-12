//! CFG edge construction and successor-window mapping.

use alloc::vec::Vec;

use crate::error::WasmError;
use crate::vm::{
    lir::{
        ir::{LirEdge, LirTerminator},
        target::LirTarget,
    },
    plan::frame::FrameSpan,
    wasm::{
        common::{BrTableEntry, SemanticTarget},
        semantic_ir::SemanticOp,
    },
};

use super::state::{BlockState, EntryState};

pub(super) fn goto_next(
    semantic_op: &SemanticOp,
    state: &BlockState,
    semantic_to_block: &[LirTarget],
    entry_states: &[EntryState],
) -> Result<LirTerminator, WasmError> {
    Ok(LirTerminator::Goto(next_edge(
        semantic_op,
        state,
        semantic_to_block,
        entry_states,
    )?))
}

pub(super) fn next_edge(
    semantic_op: &SemanticOp,
    state: &BlockState,
    semantic_to_block: &[LirTarget],
    entry_states: &[EntryState],
) -> Result<LirEdge, WasmError> {
    let next = semantic_op
        .next
        .ok_or_else(|| WasmError::invalid("missing fallthrough target".into()))?;
    edge_to_target(
        SemanticTarget::new(next.as_usize()),
        state,
        EdgeMapping::Identity,
        semantic_to_block,
        entry_states,
    )
}

pub(super) fn edge_to_target(
    target: SemanticTarget,
    state: &BlockState,
    mapping: EdgeMapping,
    semantic_to_block: &[LirTarget],
    entry_states: &[EntryState],
) -> Result<LirEdge, WasmError> {
    let target_entry = *entry_states
        .get(target.index().as_usize())
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

    let tos = match mapping {
        EdgeMapping::Identity => state.top_values(target_entry.live_tos_count() as usize)?,
        EdgeMapping::TakenBranch { payload, .. } => {
            if target_entry.live_tos_count() != 0 {
                return Err(WasmError::internal(alloc::format!(
                    "taken branch to semantic op {} must enter with canonical frame payload only, but target expects {} live TOS values (payload_slots={})",
                    target.index().as_usize(),
                    target_entry.live_tos_count(),
                    payload.map_or(0, |span| span.count),
                )));
            }
            Vec::new()
        }
    };

    Ok(LirEdge {
        target: *semantic_to_block
            .get(target.index().as_usize())
            .ok_or_else(|| WasmError::invalid("edge target out of range".into()))?,
        stack_height: mapped_height,
        tos,
    })
}

pub(super) fn br_table_edge(
    entry: &BrTableEntry,
    payload: Option<FrameSpan>,
    state: &BlockState,
    semantic_to_block: &[LirTarget],
    entry_states: &[EntryState],
) -> Result<LirEdge, WasmError> {
    edge_to_target(
        entry
            .target
            .ok_or_else(|| WasmError::invalid("br_table entry missing target".into()))?,
        state,
        EdgeMapping::TakenBranch {
            stack_drop: entry.stack_drop,
            payload,
        },
        semantic_to_block,
        entry_states,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EdgeMapping {
    Identity,
    TakenBranch {
        stack_drop: u32,
        payload: Option<FrameSpan>,
    },
}
