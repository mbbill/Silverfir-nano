//! Middle layer: semantic Wasm IR to prepared SSA-IR.
//!
//! This module is being rebuilt around a simpler ownership model:
//! - `cfg.rs` builds the explicit semantic CFG
//! - `slot_ssa.rs` will build slot-only SSA
//! - `joint_plan/` will decide cache and spill policy
//! - `rewrite/` will materialize the final SSA-IR

pub(crate) mod frame;
pub(crate) mod ssa_ir;

mod budget;
mod cfg;
mod cleanup;
mod joint_plan;
mod optimize;
mod rewrite;
mod sink_plan;
#[cfg(test)]
mod slot_ssa;

#[cfg(test)]
mod tests;

use crate::collections::{self, phase_span_with_function};

use crate::{
    error::WasmError,
    vm::{backend::BackendConfig, wasm::semantic_ir::SemanticProgram},
};

use self::{
    frame::{plan_frame_layout, FrameLayoutPlan},
    joint_plan::JointPlanner,
    ssa_ir::{ir::SsaProgram, target::SsaTarget, validate::validate_program},
};

/// Preparation input bundle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PrepareInput {
    pub config: BackendConfig,
    pub function_index: Option<u32>,
}

/// Shared frontend output consumed by interpreter and native backends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PreparedFunction {
    pub frame: FrameLayoutPlan,
    pub ssa: SsaProgram,
}

pub(crate) fn prepare_function(
    input: PrepareInput,
    semantic: SemanticProgram,
) -> Result<PreparedFunction, WasmError> {
    semantic.validate()?;
    semantic.ensure_prepare_supported()?;

    let frame = plan_frame_layout(
        semantic.local_count,
        semantic.max_stack_height,
        input.config.call_scratch_slots,
    );

    if semantic.ops.is_empty() {
        return Ok(PreparedFunction {
            frame,
            ssa: empty_program(&semantic),
        });
    }

    let cfg_phase = phase_span_with_function("cfg_lower", input.function_index);
    let semantic_cfg = cfg::build_semantic_cfg(&semantic);
    drop(cfg_phase);

    // The old slot-only SSA pass was retained only as a block-count sanity
    // check. The semantic CFG is the source of block structure for the current
    // rewriter, so avoid materializing a second transient IR just to re-count
    // the same blocks.
    let slot_block_count = semantic_cfg.blocks.len();

    let joint_plan_phase = phase_span_with_function("joint_plan", input.function_index);
    let planner = JointPlanner::build(
        &semantic,
        &semantic_cfg,
        slot_block_count,
        frame,
        input.config,
    )?;
    drop(joint_plan_phase);

    let rewrite_cfg = rewrite::RewriteCfg::from_semantic_cfg(&semantic_cfg);
    drop(semantic_cfg);

    let ssa_emit_phase = phase_span_with_function("ssa_emit", input.function_index);
    let mut ssa = rewrite::rewrite_function(semantic, &rewrite_cfg, planner, frame)?;
    drop(ssa_emit_phase);
    // Nothing past rewrite_function reads semantic_cfg, so release the compact
    // rewrite CFG before running cleanup / optimize / sink / validate.
    drop(rewrite_cfg);

    let ssa_cleanup_phase = phase_span_with_function("ssa_cleanup", input.function_index);
    cleanup::cleanup_program(&mut ssa);
    drop(ssa_cleanup_phase);

    let ssa_opt_phase = phase_span_with_function("ssa_opt", input.function_index);
    optimize::optimize_program(&mut ssa);
    drop(ssa_opt_phase);

    let ssa_sink_phase = phase_span_with_function("ssa_sink", input.function_index);
    sink_plan::plan_sinks(&mut ssa);
    drop(ssa_sink_phase);

    shrink_prepared_ssa_storage(&mut ssa);

    let ssa_validate_phase = phase_span_with_function("ssa_validate", input.function_index);
    validate_program(&ssa)?;
    drop(ssa_validate_phase);

    Ok(PreparedFunction { frame, ssa })
}

fn shrink_prepared_ssa_storage(ssa: &mut SsaProgram) {
    ssa.blocks.shrink_to_fit();
    for block in &mut ssa.blocks {
        block.params.shrink_to_fit();
        block.ops.shrink_to_fit();
        block.extra_args.shrink_to_fit();
    }
    ssa.local_slot_types.shrink_to_fit();
    ssa.result_types.shrink_to_fit();
    ssa.local_slot_info.shrink_to_fit();
    ssa.block_entry_cached_slots.shrink_to_fit();
    for slots in &mut ssa.block_entry_cached_slots {
        slots.shrink_to_fit();
    }
    ssa.block_cfg_origins.shrink_to_fit();
    for origins in &mut ssa.block_cfg_origins {
        origins.shrink_to_fit();
    }
    ssa.value_types.shrink_to_fit();
    ssa.value_sink_local.shrink_to_fit();
    ssa.const_pool.shrink_to_fit();
    ssa.primitive_pool.shrink_to_fit();
    ssa.call_ops.shrink_to_fit();
}

fn empty_program(semantic: &SemanticProgram) -> SsaProgram {
    SsaProgram {
        entry: SsaTarget(0),
        blocks: collections::Vec::new(),
        local_slot_types: semantic.local_types.clone(),
        result_types: semantic.result_types.clone(),
        local_slot_info: collections::vec![Default::default(); semantic.local_count as usize],
        block_entry_cached_slots: collections::Vec::new(),
        block_cfg_origins: collections::Vec::new(),
        value_types: collections::Vec::new(),
        value_sink_local: collections::Vec::new(),
        const_pool: collections::Vec::new(),
        primitive_pool: collections::Vec::new(),
        call_ops: collections::Vec::new(),
    }
}
