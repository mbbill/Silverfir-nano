//! LIR block orchestration.

use alloc::vec::Vec;

use crate::error::WasmError;
use crate::vm::{
    lir::{
        ir::{LirInst, LirTerminator},
        target::LirTarget,
    },
    plan::frame::FrameLayoutPlan,
};

use super::{
    body::lower_block_body_op,
    input::AlignedOp,
    state::{BlockState, EntryState, ValueAlloc},
    terminator::lower_block_terminator,
};

#[derive(Clone, Debug)]
pub(super) struct LoweredBlock {
    pub(super) ops: Vec<LirInst>,
    pub(super) terminator: LirTerminator,
}

pub(super) fn lower_block_range(
    semantic_range: core::ops::Range<usize>,
    mut state: BlockState,
    aligned: &[AlignedOp<'_>],
    frame: FrameLayoutPlan,
    semantic_to_block: &[LirTarget],
    entry_states: &[EntryState],
    values: &mut ValueAlloc,
) -> Result<LoweredBlock, WasmError> {
    let last_index = semantic_range
        .end
        .checked_sub(1)
        .ok_or_else(|| WasmError::internal("LIR block cannot be empty".into()))?;

    for semantic_index in semantic_range.start..last_index {
        lower_block_body_op(&aligned[semantic_index], &mut state, frame, values)?;
        state.validate_tos_fit("block body")?;
    }

    let terminator = lower_block_terminator(
        &aligned[last_index],
        aligned,
        &mut state,
        frame,
        semantic_to_block,
        entry_states,
        values,
    )?;
    state.validate_tos_fit("block terminator")?;

    Ok(LoweredBlock {
        ops: state.ops,
        terminator,
    })
}
