//! Core joint-plan data types.

use alloc::vec::Vec;

use crate::value_type::ValueType;
use crate::vm::middle::{
    cfg::CfgBlockId,
    frame::{FrameSlot, FrameSpan},
    ssa_ir::ir::SsaValue,
};

/// Straight-line local access choice for one semantic local op.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum PreparedLocalAccess {
    #[default]
    Slot,
    Cache,
}

/// Prefix actions chosen before one semantic op executes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PrepAction {
    Spill(FrameSpan),
    Fill(FrameSpan, Vec<ValueType>),
    DropCache(FrameSlot),
}

/// Exact transient entry state at one semantic program point.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct EntryState {
    pub stack_height: u16,
    pub spill_depth: u16,
    pub live_types: Vec<ValueType>,
}

impl EntryState {
    #[inline]
    pub(crate) const fn live_value_count(&self) -> u16 {
        self.stack_height.saturating_sub(self.spill_depth)
    }
}

/// Planned straight-line behavior for one semantic op.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct OpPlan {
    pub prefix: Vec<PrepAction>,
    pub local_access: PreparedLocalAccess,
}

/// Explicit block-boundary state shared between the planner and the rewriter.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct BoundaryState {
    pub cached_locals: Vec<FrameSlot>,
    pub stack_values: Vec<SsaValue>,
}

/// Planned cache repair for one non-canonical incoming edge.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct EdgePlan {
    pub source: CfgBlockId,
    pub target: CfgBlockId,
    pub ensure_cached_locals: Vec<FrameSlot>,
    pub drop_cached_locals: Vec<FrameSlot>,
}

/// Planned boundary state for one block.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct BlockPlan {
    pub canonical_predecessor: Option<CfgBlockId>,
    pub entry: BoundaryState,
    pub exit: BoundaryState,
}

/// Whole-function joint plan.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct FunctionPlan {
    pub gp_unit_bytes: u8,
    pub gp_dynamic_budget: u8,
    pub fp_dynamic_budget: u8,
    pub op_plans: Vec<OpPlan>,
    pub entry_states: Vec<EntryState>,
    pub blocks: Vec<BlockPlan>,
    pub repairs: Vec<EdgePlan>,
}
