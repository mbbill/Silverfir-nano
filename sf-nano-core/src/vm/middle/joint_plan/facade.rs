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
    build,
    facts::{FunctionPlan, RepairActions},
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
        frame: FrameLayoutPlan,
        config: BackendConfig,
    ) -> Result<Self, WasmError> {
        let plan = build::build_plan(semantic, cfg, frame, config)?;
        validate::validate_plan(cfg, &plan)?;
        Ok(Self { plan })
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

    /// Pass D exact entry cache row for `block` (rewrite-time drift assert).
    #[inline]
    pub(crate) fn exact_entry(&self, block: CfgBlockId) -> &[FrameSlot] {
        &self.plan.blocks[block.as_usize()].exact_entry
    }

    /// Pass D exact exit cache row for `block` (rewrite-time drift assert).
    #[inline]
    pub(crate) fn exact_exit(&self, block: CfgBlockId) -> &[FrameSlot] {
        &self.plan.blocks[block.as_usize()].exact_exit
    }

    /// Pass D repair actions for one out-edge of `block` (edge order
    /// Goto | BranchThen, BranchElse | BrTable(idx)); `None` when the edge
    /// needs no repair. Used by the rewrite-time drift assert.
    #[inline]
    pub(crate) fn exact_edge_repair(
        &self,
        block: CfgBlockId,
        edge_ordinal: usize,
    ) -> Option<&RepairActions> {
        self.plan.blocks[block.as_usize()]
            .repair
            .get(edge_ordinal)
            .copied()
            .flatten()
            .map(|idx| &self.plan.repair_pool[idx as usize])
    }
}
