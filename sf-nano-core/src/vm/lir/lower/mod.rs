//! Lower semantic/planning data into prepared LIR.
//!
//! The lowering pipeline is intentionally split into:
//! 1. semantic/planned alignment with synthetic spill/fill prefixes
//! 2. block-entry stack-window shaping
//! 3. straight-line body lowering
//! 4. terminator lowering

mod body;
mod boundary;
mod block;
mod cfg;
mod edge;
mod input;
mod ops;
mod state;
mod terminator;

use alloc::vec::Vec;

use crate::error::WasmError;
use crate::vm::{
    lir::{
        ir::{LirBlock, LirProgram},
        target::LirTarget,
    },
    plan::{config::PlanConfig, PlannedProgram},
    wasm::semantic_ir::SemanticProgram,
};

use self::{
    block::lower_block_range,
    boundary::{compute_entry_states, make_block_params},
    cfg::{build_block_ranges, build_semantic_to_block_map, retain_reachable_blocks},
    input::{align_semantic_to_planned, lower_local_cache_plan},
    state::{BlockState, ValueAlloc},
};

pub fn lower_to_lir(
    semantic: &SemanticProgram,
    planned: &PlannedProgram,
    config: PlanConfig,
) -> Result<LirProgram, WasmError> {
    semantic.validate()?;
    planned.validate(semantic, config)?;

    if semantic.ops.is_empty() {
        return Ok(LirProgram {
            entry: LirTarget(0),
            local_cache: lower_local_cache_plan(planned.hot_locals.as_ref(), planned.frame),
            blocks: Vec::new(),
        });
    }

    let aligned = align_semantic_to_planned(semantic, planned)?;
    if aligned.is_empty() {
        return Ok(LirProgram {
            entry: LirTarget(0),
            local_cache: lower_local_cache_plan(planned.hot_locals.as_ref(), planned.frame),
            blocks: Vec::new(),
        });
    }

    let block_ranges = retain_reachable_blocks(semantic, build_block_ranges(semantic, planned));
    let semantic_to_block = build_semantic_to_block_map(aligned.len(), &block_ranges);
    let entry_states = compute_entry_states(semantic, &aligned);
    let mut values = ValueAlloc::default();

    let mut blocks = Vec::with_capacity(block_ranges.len());
    for (block_index, semantic_range) in block_ranges.into_iter().enumerate() {
        let params = make_block_params(entry_states[semantic_range.start], &mut values);
        let state = BlockState::from_params(&params, config.tos_register_count)?;
        let block = lower_block_range(
            semantic_range.clone(),
            state,
            &aligned,
            planned.frame,
            &semantic_to_block,
            &entry_states,
            &mut values,
        )?;
        blocks.push(LirBlock {
            id: LirTarget(block_index as u32),
            params,
            ops: block.ops,
            terminator: block.terminator,
        });
    }

    let program = LirProgram {
        entry: semantic_to_block[0],
        local_cache: lower_local_cache_plan(planned.hot_locals.as_ref(), planned.frame),
        blocks,
    };
    program.validate()?;
    Ok(program)
}
