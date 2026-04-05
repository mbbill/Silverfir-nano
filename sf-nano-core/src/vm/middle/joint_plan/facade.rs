//! Public planner facade used by the rewriter.
//!
//! This file is intentionally thin. It holds the immutable plan plus a few
//! precomputed semantic flags, then delegates each rewrite consultation to the
//! policy-focused helper modules.

use alloc::vec::Vec;

use crate::{
    error::WasmError,
    vm::{
        backend::BackendConfig,
        middle::{
            cfg::{CfgBlockId, SemanticCfg},
            frame::FrameLayoutPlan,
            slot_ssa::SlotSsaProgram,
        },
        wasm::semantic_ir::SemanticProgram,
    },
};

use super::{
    block_open::{
        before_op_decision, block_open_decision, finalize_block_entry_cached_locals,
        target_entry_decision,
    },
    build,
    facts::FunctionPlan,
    interface::{
        BeforeOpDecision, BeforeOpQuery, BlockOpenDecision, EdgeRepairDecision, EdgeRepairQuery,
        FunctionSetupDecision, LocalAccessDecision, LocalAccessQuery, TargetEntryDecision,
    },
    local_access::decide_local_access,
    policy::op_must_drop_all_caches,
    repair::derive_edge_repair,
    validate,
};

/// Planner facade used by `rewrite/`.
#[derive(Clone, Debug)]
pub(crate) struct JointPlanner {
    plan: FunctionPlan,
    op_force_drop_all_caches: Vec<bool>,
}

impl JointPlanner {
    pub(crate) fn build(
        semantic: &SemanticProgram,
        cfg: &SemanticCfg,
        slot_program: &SlotSsaProgram,
        frame: FrameLayoutPlan,
        config: BackendConfig,
    ) -> Result<Self, WasmError> {
        let plan = build::build_plan(semantic, cfg, slot_program, frame, config)?;
        validate::validate_plan(cfg, slot_program, &plan)?;
        Ok(Self {
            plan,
            op_force_drop_all_caches: semantic
                .ops
                .iter()
                .map(|op| op_must_drop_all_caches(&op.kind))
                .collect(),
        })
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
    pub(crate) fn target_entry(&self, semantic_index: usize) -> TargetEntryDecision<'_> {
        target_entry_decision(&self.plan, semantic_index)
    }

    #[inline]
    pub(crate) fn before_op<'a>(&'a self, query: BeforeOpQuery<'_>) -> BeforeOpDecision<'a> {
        before_op_decision(&self.plan, &self.op_force_drop_all_caches, query)
    }

    #[inline]
    pub(crate) fn local_access(&self, query: LocalAccessQuery<'_>) -> LocalAccessDecision {
        decide_local_access(&self.plan, query)
    }

    #[inline]
    pub(crate) fn finalize_block_entry(
        &self,
        block: CfgBlockId,
        actual_exit: &[crate::vm::middle::frame::FrameSlot],
    ) -> Vec<crate::vm::middle::frame::FrameSlot> {
        finalize_block_entry_cached_locals(&self.plan, block, actual_exit)
    }

    pub(crate) fn edge_repair(&self, query: EdgeRepairQuery<'_>) -> EdgeRepairDecision {
        derive_edge_repair(&self.plan, query)
    }
}
