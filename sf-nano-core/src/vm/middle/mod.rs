//! Middle layer: SIR → LIR.
//!
//! This layer sits between the Wasm semantic frontend (`wasm/`) and the
//! machine lowering layer (`machine/`). It transforms decoded semantic IR
//! into prepared LIR with explicit spill/fill and canonical frame layout.
//!
//! - `lir/` — the LIR contract definitions consumed by `machine/`
//! - `config.rs` — backend budget configuration
//! - `frame.rs` — shared frame types and layout planning
//! - `local_cache.rs` — cached local analysis
//! - `spill_plan.rs` — spill/fill planning pass
//! - `lower_*.rs` — SIR → LIR lowering steps
//! - `state.rs` — preparation state (value allocation, budget tracking)
//! - `optimize.rs` — post-construction LIR optimization

pub mod config;
pub mod frame;
pub mod lir;

mod local_cache;
mod lower_block;
mod lower_cfg;
mod lower_edge;
mod lower_ops;
mod lower_term;
mod optimize;
mod spill_plan;
mod state;

#[cfg(test)]
mod tests;

use alloc::vec::Vec;

use crate::{
    error::WasmError,
    vm::{
        middle::lir::{
            ir::{LirBlock, LirProgram},
            target::LirTarget,
            validate::validate_program,
        },
        wasm::semantic_ir::SemanticProgram,
    },
};

use self::{
    lower_block::lower_block_range,
    lower_cfg::{build_block_ranges, build_semantic_to_block_map, retain_reachable_blocks},
    state::{BlockState, EntryState, ValueAlloc},
    spill_plan::prepare_semantic_ops,
};
use self::{
    config::PlanConfig,
    frame::{plan_frame_layout, FrameLayoutPlan},
};

pub use local_cache::analyze_local_cache_prefs;

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
    let (local_cache, continuation_skip_reload) = analyze_local_cache_prefs(
        semantic,
        input.config.gp_unit_bytes,
        input.config.gp_local_cache_budget,
        input.config.fp_local_cache_budget,
        frame,
    );
    let prepared = prepare_semantic_ops(semantic, frame, input.config)?;

    if semantic.ops.is_empty() {
        return Ok(PreparedFunction {
            frame,
            lir: LirProgram {
                entry: LirTarget(0),
                local_cache,
                blocks: Vec::new(),
                value_types: Vec::new(),
            },
        });
    }

    let block_ranges = retain_reachable_blocks(semantic, build_block_ranges(semantic));
    let semantic_to_block = build_semantic_to_block_map(semantic.ops.len(), &block_ranges);
    let mut values = ValueAlloc::default();
    let local_types = &semantic.local_types;
    let op_result_types = &semantic.op_result_types;
    let block_params = block_ranges
        .iter()
        .map(|range| make_block_params(&prepared.entry_states[range.start], &mut values))
        .collect::<Vec<_>>();

    let original_block_count = block_ranges.len();
    let mut blocks = Vec::with_capacity(block_ranges.len());
    let mut extra_blocks = Vec::new();
    let mut skip_reload_iter = continuation_skip_reload.into_iter();
    for (block_index, semantic_range) in block_ranges.into_iter().enumerate() {
        let params = block_params[block_index].clone();
        let state = BlockState::from_entry(
            &prepared.entry_states[semantic_range.start],
            &params,
            input.config.gp_unit_bytes,
            input.config.gp_transient_budget,
            input.config.fp_transient_budget,
        )?;
        let block = lower_block_range(
            semantic_range.clone(),
            state,
            &prepared.ops,
            frame,
            &semantic_to_block,
            &block_params,
            &prepared.entry_states,
            &mut values,
            original_block_count,
            extra_blocks.len(),
            local_types,
            &semantic.result_types,
            op_result_types,
            &mut skip_reload_iter,
        )?;
        blocks.push(LirBlock {
            id: LirTarget(block_index as u32),
            params,
            ops: block.ops,
            terminator: block.terminator,
        });
        extra_blocks.extend(block.extra_blocks);
    }
    blocks.extend(extra_blocks);

    let lir = LirProgram {
        entry: semantic_to_block[0],
        local_cache,
        blocks,
        value_types: values.take_types(),
    };
    let mut lir = lir;
    optimize::optimize_lir(&mut lir, frame);
    validate_program(&lir)?;

    Ok(PreparedFunction { frame, lir })
}

#[inline]
fn make_block_params(
    entry: &EntryState,
    values: &mut ValueAlloc,
) -> Vec<lir::ir::LirValue> {
    debug_assert_eq!(
        entry.live_types.len(),
        entry.live_value_count() as usize,
        "EntryState.live_types length must match live_value_count",
    );
    values.many_typed(&entry.live_types)
}
