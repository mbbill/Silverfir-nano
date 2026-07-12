//! Core joint-plan data types.

use crate::collections;

use crate::value_type::ValueType;
use crate::vm::middle::frame::FrameSlot;
use crate::vm::middle::ssa_ir::ir::EntryCacheRequirement;
/// Exact transient entry state at one CFG block entry.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct EntryState {
    pub stack_height: u16,
    pub spill_depth: u16,
    /// Full semantic stack typing at this program point, from bottom to top.
    pub stack_types: collections::Vec<ValueType>,
}

impl EntryState {
    /// Rewrite-facing resident suffix: the live values above the spilled
    /// prefix. This is exactly `stack_types[spill_depth..]`; the materialized
    /// field it replaces was always that slice, given the construction-time
    /// invariant `stack_types.len() == stack_height`.
    #[inline]
    pub(crate) fn live_types(&self) -> &[ValueType] {
        &self.stack_types[self.spill_depth as usize..]
    }
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

/// Cached-local boundary repair actions for one control-flow edge.
///
/// Three ordered groups the target block runs on entry: drops (locals the
/// predecessor exit still holds cached but the successor does not), then
/// ensures (successor residents that must be reloaded), then reserves
/// (successor residents whose first use is a write). Within each group the
/// slots stay slot-ascending. `build_repair_ops` emits them in that order.
///
/// The plan (pass D) produces these for semantic edges; `rewrite/edge.rs`
/// still derives them emit-side for synthesized bridge-block edges, sharing
/// [`derive_edge_repair`] so the logic exists once.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct RepairActions {
    pub ensure_cached_locals: collections::Vec<FrameSlot>,
    pub reserve_cached_locals: collections::Vec<FrameSlot>,
    pub drop_cached_locals: collections::Vec<FrameSlot>,
}

impl RepairActions {
    pub(crate) fn is_empty(&self) -> bool {
        self.ensure_cached_locals.is_empty()
            && self.reserve_cached_locals.is_empty()
            && self.drop_cached_locals.is_empty()
    }
}

/// Derive the boundary repair for one edge from a predecessor's exit cache set
/// and a successor's entry cache set.
///
/// `classify` reports each successor-entry slot's first-use requirement — pass
/// D reads it from the walker's recorded cache-event stream, emit-side bridge
/// derivation reads it from the target block's lowered ops. Both call this one
/// function so the drop/ensure/reserve grouping cannot drift.
pub(crate) fn derive_edge_repair(
    pred_exit: &[FrameSlot],
    succ_entry: &[FrameSlot],
    mut classify: impl FnMut(FrameSlot) -> Option<EntryCacheRequirement>,
) -> RepairActions {
    let mut repair = RepairActions::default();
    for &slot in pred_exit {
        if !succ_entry.contains(&slot) {
            repair.drop_cached_locals.push(slot);
        }
    }
    for &slot in succ_entry {
        if pred_exit.contains(&slot) {
            continue;
        }
        match classify(slot) {
            Some(EntryCacheRequirement::Ensure) => repair.ensure_cached_locals.push(slot),
            Some(EntryCacheRequirement::Reserve) => repair.reserve_cached_locals.push(slot),
            None => {}
        }
    }
    repair
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
    /// Pass D exact entry cache set (slot-ascending): the tentative set trimmed
    /// to the locals the block actually requires cached on entry.
    pub exact_entry: collections::Vec<FrameSlot>,
    /// Pass D exact exit cache set (slot-ascending): the residents live when
    /// the block hands off to its successors.
    pub exact_exit: collections::Vec<FrameSlot>,
    /// Per out-edge index into [`FunctionPlan::repair_pool`], in terminator
    /// edge order (Goto | BranchThen, BranchElse | BrTable(idx)). `None` means
    /// the edge needs no repair.
    pub repair: collections::Vec<Option<u32>>,
}

/// Whole-function joint plan.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct FunctionPlan {
    pub gp_unit_bytes: u8,
    pub gp_dynamic_budget: u8,
    pub fp_dynamic_budget: u8,
    pub blocks: collections::Vec<BlockPlan>,
    /// Content-deduped boundary-repair action lists (pass D), indexed by
    /// [`BlockPlan::repair`]. Consumed only by rewrite.
    pub repair_pool: collections::Vec<RepairActions>,
}
