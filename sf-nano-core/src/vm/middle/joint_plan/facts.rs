//! Core joint-plan data types.

use crate::collections;

use crate::value_type::ValueType;
use crate::vm::middle::frame::FrameSlot;
/// Exact transient entry state at one semantic program point.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct EntryState {
    pub stack_height: u16,
    pub spill_depth: u16,
    /// Full semantic stack typing at this program point, from bottom to top.
    ///
    /// `live_types` is still the rewrite-facing resident suffix. The full stack
    /// shape is planner-only metadata used for entry-stack ranking.
    pub stack_types: collections::Vec<ValueType>,
    pub live_types: collections::Vec<ValueType>,
}

/// Compact per-op entry state for target branch queries.
///
/// The rewriter only needs `stack_height` and `spill_depth` when checking
/// branch target contracts. The full `EntryState` with heap-allocated
/// `stack_types` and `live_types` vecs is not needed for this purpose.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CompactEntryPoint {
    pub stack_height: u16,
    pub spill_depth: u16,
}

/// First meaningful access shape for one local inside a block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FirstAccessKind {
    ReadFirst,
    WriteFirst,
}

/// Block-local use summary for one local slot.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct BlockLocalInfo {
    pub slot: FrameSlot,
    pub entry_first_access_kind: Option<FirstAccessKind>,
    pub entry_first_read_distance: Option<u16>,
    pub entry_first_write_distance: Option<u16>,
    pub entry_read_count: u16,
    pub entry_write_count: u16,
    pub entry_hot_score: i32,
    pub first_access_kind: Option<FirstAccessKind>,
    pub first_read_distance: Option<u16>,
    pub first_write_distance: Option<u16>,
    pub read_count: u16,
    pub write_count: u16,
    pub access_offsets: collections::Vec<u16>,
    pub hot_score: i32,
}

/// Compact per-block local summary retained for all blocks.
///
/// This stores only the data needed for cross-block queries (successor bonus
/// and block-open planning), not the full per-op access offsets.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct BlockLocalSummary {
    pub ranked_slots: collections::Vec<FrameSlot>,
    pub slot_scores: collections::Vec<LocalSlotScore>,
}

/// Compact per-slot score for cross-block queries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LocalSlotScore {
    pub slot: FrameSlot,
    pub entry_hot_score: i32,
    pub entry_first_access_kind: Option<FirstAccessKind>,
    pub used_anywhere: bool,
    pub read_count: u16,
    pub write_count: u16,
}

impl BlockLocalSummary {
    #[cfg(test)]
    #[inline]
    pub(crate) fn slot_score(&self, slot: FrameSlot) -> Option<&LocalSlotScore> {
        self.slot_scores.iter().find(|s| s.slot == slot)
    }
}

/// Entry-stack use summary for one original block-entry stack value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BlockStackValueInfo {
    pub stack_index: u16,
    pub ty: ValueType,
    pub touched_before_barrier: bool,
    pub first_touch_distance: Option<u16>,
    pub touch_count: u16,
    pub hot_score: i32,
}

/// Entry-stack facts used by block-open planning.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct BlockEntryStackRegion {
    pub entry_stack_height: u16,
    pub values: collections::Vec<BlockStackValueInfo>,
}

/// Whole-block use summary for one transient stack symbol.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct TransientSymbolInfo {
    pub first_touch_distance: Option<u16>,
    pub touch_count: u16,
    pub access_offsets: collections::Vec<u16>,
    pub hot_score: i32,
}

impl TransientSymbolInfo {
    #[inline]
    pub(crate) fn used_anywhere(&self) -> bool {
        !self.access_offsets.is_empty()
    }
}

/// Whole-block transient-symbol facts for one CFG block.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct BlockTransientRegion {
    pub entry_stack_height: u16,
    pub symbols: collections::Vec<TransientSymbolInfo>,
}

/// Local semantic op kind used by the planner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LocalOpKind {
    Get,
    Set,
    Tee,
}

/// Per-op planner lookup facts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct OpInfo {
    pub block_index: u32,
    pub block_offset: u16,
    pub is_block_start: bool,
    pub local_op: Option<(FrameSlot, LocalOpKind)>,
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
    pub local_slot_types: collections::Vec<ValueType>,
    /// Compact per-op entry points for target branch queries.
    ///
    /// Only `stack_height` and `spill_depth` — the rewriter never reads
    /// `live_types` or `stack_types` from these entries.
    pub compact_entries: collections::Vec<CompactEntryPoint>,
    pub op_info: collections::Vec<OpInfo>,
    pub block_local_summaries: collections::Vec<BlockLocalSummary>,
    pub block_stack_regions: collections::Vec<BlockEntryStackRegion>,
    pub block_transient_regions: collections::Vec<BlockTransientRegion>,
    pub blocks: collections::Vec<BlockPlan>,
}
