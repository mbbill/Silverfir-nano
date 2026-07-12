//! Public planner facade used by the rewriter.
//!
//! This file is intentionally thin. It holds the immutable plan plus a few
//! precomputed semantic flags, then delegates each rewrite consultation to the
//! policy-focused helper modules.

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
    block_open::{block_open_decision, target_entry_decision},
    build,
    facts::{FunctionPlan, RepairActionsSpans, RowSpan},
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

    /// `block`'s exact entry cache row — the authoritative published entry set,
    /// and the seed the rewriter opens the block with. A slice into the plan's
    /// flat `row_arena`; rewrite copies it into the program's published row.
    #[inline]
    pub(crate) fn exact_entry(&self, block: CfgBlockId) -> &[FrameSlot] {
        &self.plan.row_arena[self.plan.blocks[block.as_usize()].exact_entry.range()]
    }

    /// `block`'s exact exit cache row — the authoritative published exit set,
    /// checked against lowered reality by the standing guard. Slice into
    /// `row_arena`.
    #[inline]
    pub(crate) fn exact_exit(&self, block: CfgBlockId) -> &[FrameSlot] {
        &self.plan.row_arena[self.plan.blocks[block.as_usize()].exact_exit.range()]
    }

    /// The content-deduped repair-action pool `rewrite/edge.rs` indexes for
    /// semantic edges. Each entry's slot groups live in [`Self::repair_slot_arena`].
    #[inline]
    pub(crate) fn repair_pool(&self) -> &[RepairActionsSpans] {
        &self.plan.repair_pool
    }

    /// The flat arena holding the repair pool's drop/ensure/reserve slot groups.
    #[inline]
    pub(crate) fn repair_slot_arena(&self) -> &[FrameSlot] {
        &self.plan.repair_slot_arena
    }

    /// The flat per-edge repair index arena (`NO_REPAIR` = no repair); a block's
    /// slice is at its [`Self::repair_span`].
    #[inline]
    pub(crate) fn repair_index_arena(&self) -> &[u32] {
        &self.plan.repair_index_arena
    }

    /// `block`'s span into [`Self::repair_index_arena`] (one entry per out-edge,
    /// terminator edge order).
    #[inline]
    pub(crate) fn repair_span(&self, block: CfgBlockId) -> RowSpan {
        self.plan.blocks[block.as_usize()].repair
    }
}
