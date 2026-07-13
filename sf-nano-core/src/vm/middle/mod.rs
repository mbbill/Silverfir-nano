//! Middle layer: semantic Wasm IR to prepared SSA-IR.
//!
//! Ownership model:
//! - `cfg.rs` builds the explicit semantic CFG
//! - `joint_plan/` decides cache and spill policy
//! - `rewrite/` materializes the final SSA-IR

pub(crate) mod frame;
pub(crate) mod ssa_ir;

mod budget;
mod cfg;
mod cleanup;
mod discipline;
mod final_signals;
mod joint_plan;
mod optimize;
mod rewrite;
mod sink_plan;

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

/// Module-level facts the middle-end reads but semantic decode does not carry.
/// Threaded into [`prepare_function`] for the residency solver's call-class
/// split: a direct call to a local JIT body is survivable for preserved-class
/// nominated residents, everything else is a cache barrier.
///
/// INVARIANT (design §5.b): the prepared SSA is DERIVED from module facts and
/// must be RECOMPUTED, never cached, when those facts change (`instance.rs`
/// clears native code on a table-mutation revert; the recompile re-runs
/// `prepare_function`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct ModuleFacts<'a> {
    /// Per-function-index locality: is callee `i` a locally-JIT-compiled
    /// function (vs an imported/host function)?
    pub is_local_func: &'a [bool],
}

/// Shared frontend output consumed by interpreter and native backends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PreparedFunction {
    pub frame: FrameLayoutPlan,
    pub ssa: SsaProgram,
}

pub(crate) fn prepare_function(
    input: PrepareInput,
    facts: ModuleFacts<'_>,
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

    let joint_plan_phase = phase_span_with_function("joint_plan", input.function_index);
    let planner = JointPlanner::build(
        &semantic,
        &semantic_cfg,
        frame,
        input.config,
        facts.is_local_func,
    )?;
    let preferred_preserved: collections::Vec<bool> =
        planner.preferred_preserved().iter().copied().collect();
    drop(joint_plan_phase);

    let rewrite_cfg = rewrite::RewriteCfg::from_semantic_cfg(&semantic_cfg);
    drop(semantic_cfg);

    let ssa_emit_phase = phase_span_with_function("ssa_emit", input.function_index);
    let mut ssa = rewrite::rewrite_function(semantic, &rewrite_cfg, planner, frame, input.config)?;
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

    // Derive the machine-facing entry requirement rows over the FINAL SSA (the
    // exact program the machine lowers, after cleanup merges + optimize +
    // sink). Deriving them at plan time is unfaithful: a goto-successor merge
    // folds a slot's first-touch across the block boundary a per-plan-block
    // row cannot see. The preserved-class bit is different: it is the solver's
    // own nomination (the residency was priced with it and the plan's call
    // survival obeys it), so it is published from the plan, not re-derived.
    ssa.block_entry_cache_requirements = final_signals::derive_entry_cache_requirements(&ssa);
    ssa.preferred_preserved = preferred_preserved;

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
    ssa.block_entry_cache_requirements.shrink_to_fit();
    for reqs in &mut ssa.block_entry_cache_requirements {
        reqs.shrink_to_fit();
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
        block_entry_cache_requirements: collections::Vec::new(),
        preferred_preserved: collections::Vec::new(),
        value_types: collections::Vec::new(),
        value_sink_local: collections::Vec::new(),
        const_pool: collections::Vec::new(),
        primitive_pool: collections::Vec::new(),
        call_ops: collections::Vec::new(),
    }
}
