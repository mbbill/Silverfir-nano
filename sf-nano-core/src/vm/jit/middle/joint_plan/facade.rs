//! Public planner facade used by the rewriter.
//!
//! This file is intentionally thin. It holds the immutable plan plus a few
//! precomputed semantic flags, then delegates each rewrite consultation to the
//! policy-focused helper modules.

use crate::{
    collections,
    error::WasmError,
    vm::{
        jit::backend::BackendConfig,
        jit::middle::{
            cell::CellId,
            cfg::{CfgBlockId, SemanticCfg},
            frame::FrameLayoutPlan,
        },
        jit::wasm::semantic_ir::SemanticProgram,
    },
};

use super::{
    block_open::{block_open_decision, target_entry_decision},
    build,
    cell_access::decide_local_access,
    facts::FunctionPlan,
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

    /// The region solver's public resident set. This is the rewrite seed and
    /// the admission policy; the rewriter trims it to the entry cells that the
    /// emitted block actually needs before publishing the final SSA row.
    #[inline]
    pub(crate) fn planned_residents(&self, block: CfgBlockId) -> &[CellId] {
        self.plan.planned_residents(block.as_usize())
    }

    /// `block`'s debug-oracle entry cache row. Release builds do not compute
    /// this data; debug/test rewrite compares it with the emit-derived row.
    #[inline]
    pub(crate) fn exact_entry(&self, block: CfgBlockId) -> &[CellId] {
        &self.plan.row_arena[self.plan.blocks[block.as_usize()].exact_entry.range()]
    }

    /// `block`'s debug-oracle exit cache row, checked against lowered reality.
    /// Release builds do not compute this data.
    #[inline]
    pub(crate) fn exact_exit(&self, block: CfgBlockId) -> &[CellId] {
        &self.plan.row_arena[self.plan.blocks[block.as_usize()].exact_exit.range()]
    }
}
