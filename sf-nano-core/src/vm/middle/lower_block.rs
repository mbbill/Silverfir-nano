//! Block lowering orchestration.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::{
    error::WasmError,
    value_type::ValueType,
    vm::{
        middle::{
            frame::FrameLayoutPlan,
            ssa_ir::{
                ir::{SsaBlock, SsaInst, SsaTerminator},
                target::SsaTarget,
            },
        },
        wasm::semantic_ir::SemanticOpKind,
    },
};

use super::{
    lower_ops::lower_block_body_op,
    state::{BlockState, EntryState, ValueAlloc},
    spill_plan::PreparedOp,
    lower_term::{
        canonicalize_live_window_for_target, fallthrough_target, lower_block_terminator,
    },
};

#[derive(Clone, Debug)]
pub(super) struct LoweredBlock {
    pub(super) ops: Vec<SsaInst>,
    pub(super) terminator: SsaTerminator,
    pub(super) extra_blocks: Vec<SsaBlock>,
}

pub(super) fn lower_block_range(
    semantic_range: core::ops::Range<usize>,
    mut state: BlockState,
    prepared: &[PreparedOp<'_>],
    frame: FrameLayoutPlan,
    semantic_to_block: &[SsaTarget],
    block_params: &[Vec<crate::vm::middle::ssa_ir::ir::SsaValue>],
    entry_states: &[EntryState],
    values: &mut ValueAlloc,
    original_block_count: usize,
    extra_blocks_len: usize,
    local_types: &[ValueType],
    result_types: &[ValueType],
    op_result_types: &BTreeMap<usize, Vec<ValueType>>,
    skip_reload_iter: &mut dyn Iterator<Item = Vec<bool>>,
) -> Result<LoweredBlock, WasmError> {
    let last_index = semantic_range
        .end
        .checked_sub(1)
        .ok_or_else(|| WasmError::internal("SSA-IR block cannot be empty".into()))?;

    for semantic_index in semantic_range.start..last_index {
        // End ops in the body need their publication logic — the same
        // reconciliation that lower_block_terminator does for End when
        // it's the last op in a block.
        if matches!(prepared[semantic_index].semantic.kind, SemanticOpKind::End) {
            let target = fallthrough_target(semantic_index, prepared.len())?;
            canonicalize_live_window_for_target(target, &mut state, frame, entry_states)?;
        }
        lower_block_body_op(
            &prepared[semantic_index],
            semantic_index,
            &mut state,
            frame,
            values,
            local_types,
            op_result_types,
            skip_reload_iter,
        )?;
        state.validate_live_fit("block body")?;
    }

    let terminator = lower_block_terminator(
        last_index,
        &prepared[last_index],
        prepared.len(),
        &mut state,
        frame,
        semantic_to_block,
        block_params,
        entry_states,
        values,
        original_block_count,
        extra_blocks_len,
        local_types,
        result_types,
        op_result_types,
        skip_reload_iter,
    )?;
    state.validate_live_fit("block terminator")?;

    Ok(LoweredBlock {
        ops: state.ops,
        terminator: terminator.terminator,
        extra_blocks: terminator.extra_blocks,
    })
}
