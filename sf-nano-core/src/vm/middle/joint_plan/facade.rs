//! Public planner facade used by the rewriter.
//!
//! This file is intentionally thin. It holds the immutable plan plus a few
//! precomputed semantic flags, then delegates each rewrite consultation to the
//! policy-focused helper modules.

use crate::collections;

use crate::{
    error::WasmError,
    vm::{
        backend::BackendConfig,
        middle::{
            cfg::{CfgBlockId, SemanticCfg},
            frame::{FrameLayoutPlan, FrameSlot},
        },
        wasm::semantic_ir::SemanticProgram,
    },
};

use super::{
    block_open::{block_open_decision, finalize_block_entry_cached_locals, target_entry_decision},
    build, build_uniform,
    facts::FunctionPlan,
    interface::{
        BlockOpenDecision, FunctionSetupDecision, LocalAccessDecision, LocalAccessQuery,
        TargetEntryDecision,
    },
    local_access::decide_local_access,
    validate,
};

/// Planner facade used by `rewrite/`.
#[derive(Clone, Debug)]
pub(crate) struct JointPlanner {
    plan: FunctionPlan,
}

impl JointPlanner {
    pub(crate) fn build(
        semantic: &SemanticProgram,
        cfg: &SemanticCfg,
        slot_block_count: usize,
        frame: FrameLayoutPlan,
        config: BackendConfig,
    ) -> Result<Self, WasmError> {
        let plan = build::build_plan(semantic, cfg, frame, config)?;
        validate::validate_plan(cfg, slot_block_count, &plan)?;
        Ok(Self { plan })
    }

    /// Build a uniform-boundary plan: every block has the same cached-local
    /// set and the same dynamic-register split between transient and cache.
    /// See `STREAMING_MIDDLE.md` for the full contract.
    pub(crate) fn build_uniform_boundary(
        semantic: &SemanticProgram,
        cfg: &SemanticCfg,
        slot_block_count: usize,
        frame: FrameLayoutPlan,
        config: BackendConfig,
    ) -> Result<Self, WasmError> {
        let plan = build_uniform::build_uniform_plan(semantic, cfg, frame, config)?;
        validate::validate_plan(cfg, slot_block_count, &plan)?;
        Ok(Self { plan })
    }

    #[inline]
    pub(crate) fn is_uniform_boundary(&self) -> bool {
        self.plan.uniform_boundary
    }

    #[inline]
    pub(crate) fn function_setup(&self) -> FunctionSetupDecision {
        FunctionSetupDecision {
            gp_unit_bytes: self.plan.gp_unit_bytes,
            gp_dynamic_budget: self.plan.gp_dynamic_budget,
            fp_dynamic_budget: self.plan.fp_dynamic_budget,
        }
    }

    #[inline]
    pub(crate) fn block_open(&self, block: CfgBlockId) -> BlockOpenDecision<'_> {
        block_open_decision(&self.plan, block)
    }

    #[inline]
    pub(crate) fn target_entry(&self, block: CfgBlockId) -> TargetEntryDecision {
        target_entry_decision(&self.plan, block)
    }

    #[inline]
    pub(crate) fn local_access(&self, query: LocalAccessQuery<'_>) -> LocalAccessDecision {
        decide_local_access(&self.plan, query)
    }

    #[inline]
    pub(crate) fn finalize_block_entry(
        &self,
        block: CfgBlockId,
        actual_exit: &[FrameSlot],
    ) -> collections::Vec<FrameSlot> {
        finalize_block_entry_cached_locals(&self.plan, block, actual_exit)
    }
}
