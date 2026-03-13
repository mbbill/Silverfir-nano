//! Direct preparation from decoded Wasm semantics into prepared LIR.

mod block;
mod cfg;
mod edge;
mod ops;
mod state;
mod steps;
mod terminator;

use alloc::vec::Vec;

use crate::{
    error::WasmError,
    vm::{
        lir::{
            ir::{LirBlock, LirProgram},
            target::LirTarget,
            validate::validate_program,
        },
        wasm::semantic_ir::SemanticProgram,
    },
};

use super::{
    analyze_local_cache_prefs,
    config::PlanConfig,
    frame::{plan_frame_layout, FrameLayoutPlan},
};
use self::{
    block::lower_block_range,
    cfg::{build_block_ranges, build_semantic_to_block_map, retain_reachable_blocks},
    state::{BlockState, EntryState, ValueAlloc},
    steps::prepare_semantic_ops,
};

/// Preparation input bundle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrepareInput {
    pub config: PlanConfig,
}

/// Shared frontend output consumed by interpreter and native backends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedFunction {
    pub frame: FrameLayoutPlan,
    pub lir: LirProgram,
}

pub fn prepare_function(
    input: PrepareInput,
    semantic: &SemanticProgram,
) -> Result<PreparedFunction, WasmError> {
    semantic.validate()?;

    let frame = plan_frame_layout(
        semantic.local_count,
        semantic.max_stack_height,
        input.config.call_scratch_slots,
    );
    let local_cache = analyze_local_cache_prefs(semantic, input.config.cached_locals, frame);
    let prepared = prepare_semantic_ops(semantic, frame, input.config)?;

    if semantic.ops.is_empty() {
        return Ok(PreparedFunction {
            frame,
            lir: LirProgram {
                entry: LirTarget(0),
                local_cache,
                blocks: Vec::new(),
            },
        });
    }

    let block_ranges = retain_reachable_blocks(semantic, build_block_ranges(semantic));
    let semantic_to_block = build_semantic_to_block_map(semantic.ops.len(), &block_ranges);
    let mut values = ValueAlloc::default();
    let block_params = block_ranges
        .iter()
        .map(|range| make_block_params(prepared.entry_states[range.start], &mut values))
        .collect::<Vec<_>>();

    let mut blocks = Vec::with_capacity(block_ranges.len());
    for (block_index, semantic_range) in block_ranges.into_iter().enumerate() {
        let params = block_params[block_index].clone();
        let state = BlockState::from_entry(prepared.entry_states[semantic_range.start], &params, input.config.tos_lanes)?;
        let block = lower_block_range(
            semantic_range.clone(),
            state,
            &prepared.ops,
            frame,
            &semantic_to_block,
            &block_params,
            &prepared.entry_states,
            &mut values,
        )?;
        blocks.push(LirBlock {
            id: LirTarget(block_index as u32),
            params,
            ops: block.ops,
            terminator: block.terminator,
        });
    }

    let lir = LirProgram {
        entry: semantic_to_block[0],
        local_cache,
        blocks,
    };
    validate_program(&lir)?;

    Ok(PreparedFunction { frame, lir })
}

#[inline]
fn make_block_params(
    entry: EntryState,
    values: &mut ValueAlloc,
) -> alloc::vec::Vec<crate::vm::lir::ir::LirValue> {
    values.many(entry.live_value_count() as usize)
}
