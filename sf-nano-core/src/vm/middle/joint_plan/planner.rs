//! Public planner facade used by the rewriter.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::{
    error::WasmError,
    value_type::ValueType,
    vm::{
        backend::BackendConfig,
        middle::{
            cfg::{CfgBlockId, SemanticCfg},
            frame::{FrameLayoutPlan, FrameSlot},
            slot_ssa::SlotSsaProgram,
        },
        wasm::semantic_ir::SemanticProgram,
    },
};

use super::{
    builder,
    policy::op_must_drop_all_caches,
    types::{EntryState, FunctionPlan, PreparedLocalAccess},
    validate,
};

/// Planner facade used by `rewrite.rs`.
#[derive(Clone, Debug)]
pub(crate) struct JointPlanner {
    plan: FunctionPlan,
    op_force_drop_all_caches: Vec<bool>,
}

/// Function-wide setup facts needed by the rewriter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FunctionSetupDecision {
    pub gp_unit_bytes: u8,
    pub gp_dynamic_budget: u8,
    pub fp_dynamic_budget: u8,
}

/// Exact transient stack contract at one rewrite boundary.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TransientContract<'a> {
    pub stack_height: u16,
    pub spill_depth: u16,
    pub live_types: &'a [ValueType],
}

impl TransientContract<'_> {
    #[inline]
    pub(crate) const fn live_value_count(&self) -> u16 {
        self.stack_height.saturating_sub(self.spill_depth)
    }
}

/// Chosen block-open boundary.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BlockOpenDecision<'a> {
    pub transient: TransientContract<'a>,
    pub cached_locals: &'a [FrameSlot],
}

/// Target boundary facts for control-flow canonicalization.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TargetEntryDecision<'a> {
    pub transient: TransientContract<'a>,
}

/// Authoritative planner decision before one semantic op executes.
#[derive(Clone, Debug)]
pub(crate) struct BeforeOpDecision<'a> {
    pub transient: TransientContract<'a>,
    pub drop_cached_locals: Vec<FrameSlot>,
}

/// Query for the planner-owned pre-op boundary transition.
#[derive(Clone, Debug)]
pub(crate) struct BeforeOpQuery<'a> {
    pub semantic_index: usize,
    pub resident_cache: &'a BTreeSet<FrameSlot>,
}

/// Query for slot-vs-cache lowering of a local op.
#[derive(Clone, Debug)]
pub(crate) struct LocalAccessQuery<'a> {
    pub semantic_index: usize,
    pub slot: FrameSlot,
    pub resident_cache: &'a BTreeSet<FrameSlot>,
}

/// Planner decision for a local op lowering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LocalAccessDecision {
    Slot,
    Cache,
}

/// Chosen block-exit boundary.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BlockExitDecision<'a> {
    pub cached_locals: &'a [FrameSlot],
}

/// Query for repairing a block edge.
#[derive(Clone, Debug)]
pub(crate) struct EdgeRepairQuery<'a> {
    pub pred_exit: &'a [FrameSlot],
    pub succ_entry: &'a [FrameSlot],
}

/// Required repair actions for a mismatching edge.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct EdgeRepairDecision {
    pub ensure_cached_locals: Vec<FrameSlot>,
    pub drop_cached_locals: Vec<FrameSlot>,
}

impl JointPlanner {
    pub(crate) fn build(
        semantic: &SemanticProgram,
        cfg: &SemanticCfg,
        slot_program: &SlotSsaProgram,
        frame: FrameLayoutPlan,
        config: BackendConfig,
    ) -> Result<Self, WasmError> {
        let plan = builder::build_plan(semantic, cfg, slot_program, frame, config)?;
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
        let block_plan = &self.plan.blocks[block.as_usize()];
        BlockOpenDecision {
            transient: transient_contract(&block_plan.entry),
            cached_locals: &block_plan.entry_cached_locals,
        }
    }

    #[inline]
    pub(crate) fn target_entry(&self, semantic_index: usize) -> TargetEntryDecision<'_> {
        TargetEntryDecision {
            transient: transient_contract(&self.plan.entry_states[semantic_index]),
        }
    }

    #[inline]
    pub(crate) fn before_op(&self, query: BeforeOpQuery<'_>) -> BeforeOpDecision<'_> {
        let op_plan = &self.plan.op_plans[query.semantic_index];
        let mut drop_cached_locals = op_plan
            .drop_cached_locals
            .iter()
            .copied()
            .filter(|slot| query.resident_cache.contains(slot))
            .collect::<Vec<_>>();
        if self
            .op_force_drop_all_caches
            .get(query.semantic_index)
            .copied()
            .unwrap_or(false)
        {
            drop_cached_locals = query.resident_cache.iter().copied().collect();
        }
        BeforeOpDecision {
            transient: transient_contract(&op_plan.before),
            drop_cached_locals,
        }
    }

    #[inline]
    pub(crate) fn local_access(&self, query: LocalAccessQuery<'_>) -> LocalAccessDecision {
        if query.resident_cache.contains(&query.slot) {
            return LocalAccessDecision::Cache;
        }
        match self.plan.op_plans[query.semantic_index].local_access {
            PreparedLocalAccess::Slot => LocalAccessDecision::Slot,
            PreparedLocalAccess::Cache => LocalAccessDecision::Cache,
        }
    }

    #[inline]
    pub(crate) fn block_exit(&self, block: CfgBlockId) -> BlockExitDecision<'_> {
        BlockExitDecision {
            cached_locals: &self.plan.blocks[block.as_usize()].exit_cached_locals,
        }
    }

    pub(crate) fn edge_repair(&self, query: EdgeRepairQuery<'_>) -> EdgeRepairDecision {
        let mut decision = EdgeRepairDecision::default();
        for &slot in query.pred_exit {
            if !query.succ_entry.contains(&slot) {
                decision.drop_cached_locals.push(slot);
            }
        }
        for &slot in query.succ_entry {
            if !query.pred_exit.contains(&slot) {
                decision.ensure_cached_locals.push(slot);
            }
        }
        decision
    }
}

#[inline]
fn transient_contract(entry: &EntryState) -> TransientContract<'_> {
    TransientContract {
        stack_height: entry.stack_height,
        spill_depth: entry.spill_depth,
        live_types: &entry.live_types,
    }
}
