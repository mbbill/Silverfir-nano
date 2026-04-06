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
mod slot_ssa;

#[cfg(test)]
mod tests;

use alloc::vec::Vec;

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

    let frame = plan_frame_layout(
        semantic.local_count,
        semantic.max_stack_height,
        input.config.call_scratch_slots,
    );

    if semantic.ops.is_empty() {
        return Ok(PreparedFunction {
            frame,
            ssa: empty_program(semantic),
        });
    }

    let semantic_cfg = cfg::build_semantic_cfg(semantic);
    let slot_program = slot_ssa::lower_slot_only_ssa(semantic, &semantic_cfg, frame)?;
    let planner = JointPlanner::build(semantic, &semantic_cfg, &slot_program, frame, input.config)?;
    let mut ssa =
        rewrite::rewrite_function(semantic, &semantic_cfg, &slot_program, &planner, frame)?;
    cleanup::cleanup_program(&mut ssa);
    optimize::optimize_program(&mut ssa);
    sink_plan::plan_sinks(&mut ssa);
    validate_program(&ssa)?;

    Ok(PreparedFunction { frame, ssa })
}

fn empty_program(semantic: &SemanticProgram) -> SsaProgram {
    SsaProgram {
        entry: SsaTarget(0),
        blocks: Vec::new(),
        local_slot_types: semantic.local_types.clone(),
        local_slot_info: alloc::vec![Default::default(); semantic.local_count as usize],
        block_entry_cached_slots: Vec::new(),
        block_cfg_origins: Vec::new(),
        value_types: Vec::new(),
        value_sink_local: Vec::new(),
    }
}
