//! Core joint-plan data types.

use crate::collections;

use crate::value_type::ValueType;
use crate::vm::middle::frame::FrameSlot;
/// Exact transient entry state at one CFG block entry.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct EntryState {
    pub stack_height: u16,
    pub spill_depth: u16,
    /// Full semantic stack typing at this program point, from bottom to top.
    pub stack_types: collections::Vec<ValueType>,
    /// Rewrite-facing resident suffix.
    pub live_types: collections::Vec<ValueType>,
}

/// Compact per-block local summary retained for the public-cache solver.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct BlockLocalSummary {
    pub slot_scores: collections::Vec<LocalSlotScore>,
}

/// Compact per-slot access count for one CFG block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LocalSlotScore {
    pub slot: FrameSlot,
    pub access_count: u16,
}

/// Planned block boundary state used by the planner facade.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct BlockPlan {
    pub entry: EntryState,
    /// Cached locals the block may assume at open while lowering once.
    ///
    /// This is intentionally tentative. Rewrite observes the actual exit and
    /// finalizes the public block entry afterward by trimming only useless
    /// carried-in locals.
    pub tentative_entry_cached_locals: collections::Vec<FrameSlot>,
}

/// Whole-function joint plan.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct FunctionPlan {
    pub gp_unit_bytes: u8,
    pub gp_dynamic_budget: u8,
    pub fp_dynamic_budget: u8,
    pub blocks: collections::Vec<BlockPlan>,
}
