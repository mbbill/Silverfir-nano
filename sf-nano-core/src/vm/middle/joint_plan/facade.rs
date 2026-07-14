//! Public planner facade used by the rewriter.
//!
//! This file is intentionally thin. It holds the immutable plan plus a few
//! precomputed semantic flags, then delegates each rewrite consultation to the
//! policy-focused helper modules.

use crate::{
    collections,
    error::WasmError,
    vm::{
        backend::BackendConfig,
        middle::{
            cell::CellId,
            cfg::{CfgBlockId, SemanticCfg},
            frame::FrameLayoutPlan,
        },
        wasm::semantic_ir::SemanticProgram,
    },
};

use super::{
    block_open::{block_open_decision, target_entry_decision},
    build,
    cell_access::decide_local_access,
    facts::{FunctionPlan, RepairActionsSpans, RowSpan},
    interface::{
        BlockOpenDecision, CellAccessDecision, CellAccessQuery, FunctionSetupDecision,
        TargetEntryDecision,
    },
    validate,
};

/// Planner facade used by `rewrite/`.
#[derive(Clone, Debug)]
pub(crate) struct JointPlanner {
    plan: FunctionPlan,
    /// Module-fact copy: which function indices are local JIT bodies. A direct
    /// call to one is survivable — preserved-class nominated residents ride it
    /// out in their callee-saved registers.
    is_local_func: collections::Vec<bool>,
}

impl JointPlanner {
    pub(crate) fn build(
        semantic: &SemanticProgram,
        cfg: &SemanticCfg,
        frame: FrameLayoutPlan,
        config: BackendConfig,
        is_local_func: &[bool],
    ) -> Result<Self, WasmError> {
        let plan = build::build_plan(semantic, cfg, frame, config, is_local_func)?;
        validate::validate_plan(cfg, &plan)?;
        Ok(Self {
            plan,
            is_local_func: is_local_func.iter().copied().collect(),
        })
    }

    /// Whether a direct call to `callee` is survivable for preserved-class
    /// nominated residents (the callee is a local JIT body).
    #[inline]
    pub(crate) fn direct_call_survivable(&self, callee: u32) -> bool {
        self.is_local_func
            .get(callee as usize)
            .copied()
            .unwrap_or(false)
    }

    /// The solver's function-scope preserved-class nomination, per local slot.
    /// Published to the machine as `SsaProgram::preferred_preserved`.
    #[inline]
    pub(crate) fn preferred_preserved(&self) -> &[bool] {
        &self.plan.preferred_preserved
    }

    /// Whether `slot` is preserved-class nominated: such a resident survives a
    /// survivable (direct local-JIT) call instead of being killed by it.
    #[inline]
    pub(crate) fn is_preserved_nominated(&self, slot: CellId) -> bool {
        self.plan.is_preserved_nominated(slot)
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
    pub(crate) fn cell_access(&self, query: CellAccessQuery<'_>) -> CellAccessDecision {
        decide_local_access(&self.plan, query)
    }

    /// `block`'s exact entry cache row — the authoritative published entry set,
    /// and the seed the rewriter opens the block with. A slice into the plan's
    /// flat `row_arena`; rewrite copies it into the program's published row.
    #[inline]
    pub(crate) fn exact_entry(&self, block: CfgBlockId) -> &[CellId] {
        &self.plan.row_arena[self.plan.blocks[block.as_usize()].exact_entry.range()]
    }

    /// `block`'s exact exit cache row — the authoritative published exit set,
    /// checked against lowered reality by the standing guard. Slice into
    /// `row_arena`.
    #[inline]
    pub(crate) fn exact_exit(&self, block: CfgBlockId) -> &[CellId] {
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
    pub(crate) fn repair_slot_arena(&self) -> &[CellId] {
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
