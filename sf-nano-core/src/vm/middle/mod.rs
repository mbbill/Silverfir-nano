//! Middle layer: SIR → SSA-IR.
//!
//! This layer sits between the Wasm semantic frontend (`wasm/`) and the
//! machine lowering layer (`machine/`). It transforms decoded semantic IR
//! into prepared SSA-IR with explicit spill/fill and canonical frame layout.
//!
//! - `ssa_ir/` — the SSA-IR contract definitions consumed by `machine/`
//! - `frame.rs` — shared frame types and layout planning
//! - `local_cache.rs` — cached local analysis
//! - `spill_plan.rs` — spill/fill planning pass
//! - `lower_*.rs` — SIR → SSA-IR lowering steps
//! - `state.rs` — preparation state (value allocation, budget tracking)
//! - `optimize.rs` — post-construction SSA-IR optimization

pub(crate) mod frame;
pub(crate) mod ssa_ir;

mod local_cache;
mod lower_block;
mod lower_cfg;
mod lower_edge;
mod lower_ops;
mod lower_term;
mod optimize;
mod resource_plan;
mod sink_plan;
mod spill_plan;
mod state;
mod thread_jumps;

#[cfg(test)]
mod tests;

use alloc::{vec, vec::Vec};

use crate::{
    error::WasmError,
    vm::{
        backend::BackendConfig,
        middle::ssa_ir::{
            ir::{SsaBlock, SsaProgram},
            target::SsaTarget,
            validate::validate_program,
        },
        wasm::semantic_ir::SemanticProgram,
    },
};

use self::frame::{plan_frame_layout, FrameLayoutPlan};
use self::{
    lower_block::lower_block_range,
    lower_cfg::{build_block_ranges, build_semantic_to_block_map, retain_reachable_blocks},
    spill_plan::prepare_semantic_ops,
    state::{BlockState, EntryState, ValueAlloc},
};

use local_cache::analyze_local_cache_prefs;

/// Preparation input bundle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PrepareInput {
    pub config: BackendConfig,
}

/// Shared frontend output consumed by interpreter and native backends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PreparedFunction {
    pub frame: FrameLayoutPlan,
    pub ssa: SsaProgram,
}

pub(crate) fn prepare_function(
    input: PrepareInput,
    semantic: &SemanticProgram,
) -> Result<PreparedFunction, WasmError> {
    semantic.validate()?;

    let gp_dynamic_budget = input
        .config
        .gp_transient_budget
        .saturating_add(input.config.gp_local_cache_budget);
    let fp_dynamic_budget = input
        .config
        .fp_transient_budget
        .saturating_add(input.config.fp_local_cache_budget);

    let frame = plan_frame_layout(
        semantic.local_count,
        semantic.max_stack_height,
        input.config.call_scratch_slots,
    );
    let local_cache = analyze_local_cache_prefs(
        semantic,
        input.config.gp_unit_bytes,
        input.config.gp_local_cache_budget,
        input.config.fp_local_cache_budget,
        frame,
    );
    let prepared = prepare_semantic_ops(
        semantic,
        frame,
        input.config,
        &local_cache,
        gp_dynamic_budget,
        fp_dynamic_budget,
    )?;

    if semantic.ops.is_empty() {
        return Ok(PreparedFunction {
            frame,
            ssa: SsaProgram {
                entry: SsaTarget(0),
                local_cache,
                blocks: Vec::new(),
                block_entry_cached_slots: Vec::new(),
                value_types: Vec::new(),
                value_sink_local: Vec::new(),
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
    for (block_index, semantic_range) in block_ranges.into_iter().enumerate() {
        let params = block_params[block_index].clone();
        let state = BlockState::from_entry(
            &prepared.entry_states[semantic_range.start],
            &params,
            input.config.gp_unit_bytes,
            gp_dynamic_budget,
            fp_dynamic_budget,
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
        )?;
        blocks.push(SsaBlock {
            id: SsaTarget(block_index as u32),
            params,
            ops: block.ops,
            terminator: block.terminator,
        });
        extra_blocks.extend(block.extra_blocks);
    }
    blocks.extend(extra_blocks);

    let ssa = SsaProgram {
        entry: semantic_to_block[0],
        local_cache,
        block_entry_cached_slots: vec![Vec::new(); blocks.len()],
        blocks,
        value_types: values.take_types(),
        value_sink_local: Vec::new(),
    };
    let mut ssa = ssa;
    thread_jumps::simplify_cfg(&mut ssa);
    optimize::optimize_ssa(&mut ssa, frame);
    resource_plan::plan_resources(
        &mut ssa,
        input.config.gp_unit_bytes,
        gp_dynamic_budget,
        fp_dynamic_budget,
    );
    sink_plan::plan_sinks(&mut ssa);
    validate_program(&ssa)?;

    Ok(PreparedFunction { frame, ssa })
}

#[inline]
fn make_block_params(entry: &EntryState, values: &mut ValueAlloc) -> Vec<ssa_ir::ir::SsaValue> {
    debug_assert_eq!(
        entry.live_types.len(),
        entry.live_value_count() as usize,
        "EntryState.live_types length must match live_value_count",
    );
    values.many_typed(&entry.live_types)
}
