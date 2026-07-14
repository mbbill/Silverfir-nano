//! Core joint-plan data types.

use crate::collections;

use crate::value_type::ValueType;
use crate::vm::middle::cell::CellId;
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
pub(crate) struct BlockCellSummary {
    pub slot_scores: collections::Vec<CellScore>,
}

/// Compact per-slot access count for one CFG block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CellScore {
    pub slot: CellId,
    pub access_count: u16,
}

/// Cached-local boundary repair actions for one control-flow edge — the
/// transient derivation/consumption form (three owned slot groups).
///
/// Three ordered groups the target block runs on entry: drops (locals the
/// predecessor exit still holds cached but the successor does not), then
/// ensures (successor residents that must be reloaded), then reserves
/// (successor residents whose first use is a write). Within each group the
/// slots stay slot-ascending. `build_repair_ops` emits them in that order.
///
/// This owned form is used only transiently while deriving or consuming one
/// edge. The RETAINED plan pool stores the flattened [`RepairActionsSpans`]
/// into [`FunctionPlan::repair_slot_arena`] instead — POD spans, no per-entry
/// `Vec` headers (the a3a7a102 lesson). `rewrite/edge.rs` derives this owned
/// form emit-side for synthesized bridge-block edges, sharing
/// [`derive_edge_repair`] so the logic exists once.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct RepairActions {
    pub ensure_cached_cells: collections::Vec<CellId>,
    pub reserve_cached_cells: collections::Vec<CellId>,
    pub drop_cached_cells: collections::Vec<CellId>,
}

impl RepairActions {
    pub(crate) fn is_empty(&self) -> bool {
        self.ensure_cached_cells.is_empty()
            && self.reserve_cached_cells.is_empty()
            && self.drop_cached_cells.is_empty()
    }
}

/// Flattened repair actions: spans into [`FunctionPlan::repair_slot_arena`], one
/// span per slot group, in the same drop/ensure/reserve grouping
/// [`RepairActions`] uses. POD (`Copy`) so the retained `repair_pool` carries no
/// per-entry `Vec` headers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RepairActionsSpans {
    pub drop: RowSpan,
    pub ensure: RowSpan,
    pub reserve: RowSpan,
}

/// Derive the boundary repair for one edge from a predecessor's exit cache set
/// and a successor's entry cache set.
///
/// `classify` reports each successor-entry slot's first-use requirement — pass
/// D reads it from the walker's recorded cache-event stream, emit-side bridge
/// derivation reads it from the target block's lowered ops. Both call this one
/// function so the drop/ensure/reserve grouping cannot drift.
pub(crate) fn derive_edge_repair(
    pred_exit: &[CellId],
    succ_entry: &[CellId],
    mut classify: impl FnMut(CellId) -> Option<EntryCacheRequirement>,
) -> RepairActions {
    let mut repair = RepairActions::default();
    for &slot in pred_exit {
        if !succ_entry.contains(&slot) {
            repair.drop_cached_cells.push(slot);
        }
    }
    for &slot in succ_entry {
        if pred_exit.contains(&slot) {
            continue;
        }
        match classify(slot) {
            Some(EntryCacheRequirement::Ensure) => repair.ensure_cached_cells.push(slot),
            Some(EntryCacheRequirement::Reserve) => repair.reserve_cached_cells.push(slot),
            None => {}
        }
    }
    repair
}

/// An `(offset, len)` slice of a flat plan arena. Pass D stores per-block cache
/// rows as spans into function-level arenas rather than a small `Vec` per block:
/// on multi-thousand-block functions the per-block `Vec` headers alone dominate
/// planner memory (the a3a7a102 lesson — per-block materialization).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RowSpan {
    pub offset: u32,
    pub len: u32,
}

impl RowSpan {
    #[inline]
    pub(crate) fn range(self) -> core::ops::Range<usize> {
        self.offset as usize..self.offset as usize + self.len as usize
    }
}

/// Sentinel in [`FunctionPlan::repair_index_arena`] meaning "this edge needs no
/// repair" (in place of `None`).
pub(crate) const NO_REPAIR: u32 = u32::MAX;

/// Planned block boundary state used by the planner facade.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct BlockPlan {
    pub entry: EntryState,
    /// The block's planned cache residents (the region solver's public set), as
    /// a span into [`FunctionPlan::resident_arena`].
    ///
    /// This is the ADMISSION policy, not the published entry row. `cell_access`
    /// admits a local to cache exactly when it is a planned resident, so a hot
    /// local re-accessed after a call is re-cached even though the exact entry
    /// row trimmed it — the call invalidated entry residency but not admission.
    /// Lowering seeds each block from its exact-entry row; this set only governs
    /// mid-block re-admission.
    pub planned_residents: RowSpan,
    /// Pass D exact entry cache set (slot-ascending), as a span into
    /// [`FunctionPlan::row_arena`]: the planned residents trimmed to the locals
    /// the block actually requires cached on entry.
    pub exact_entry: RowSpan,
    /// Pass D exact exit cache set (slot-ascending), as a span into
    /// [`FunctionPlan::row_arena`]: the residents live when the block hands off.
    pub exact_exit: RowSpan,
    /// Per out-edge repair, as a span into [`FunctionPlan::repair_index_arena`]
    /// in terminator edge order (Goto | BranchThen, BranchElse | BrTable(idx)).
    /// Each entry is a [`FunctionPlan::repair_pool`] index, or [`NO_REPAIR`].
    pub repair: RowSpan,
}

/// Whole-function joint plan.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct FunctionPlan {
    pub gp_unit_bytes: u8,
    pub gp_dynamic_budget: u8,
    pub fp_dynamic_budget: u8,
    pub blocks: collections::Vec<BlockPlan>,
    /// Flat arena for every block's planned-resident set; blocks index it via
    /// their `planned_residents` span. Built in `build_plan` before pass D (the
    /// walker reads it), so it is a separate arena from `row_arena`.
    pub resident_arena: collections::Vec<CellId>,
    /// Content-deduped boundary-repair action lists (pass D), indexed by the
    /// entries of `repair_index_arena`. Each entry's slot groups live in
    /// `repair_slot_arena`. Consumed only by rewrite.
    pub repair_pool: collections::Vec<RepairActionsSpans>,
    /// Flat arena of the repair pool's drop/ensure/reserve slot groups; pool
    /// entries index it via their [`RepairActionsSpans`] spans.
    pub repair_slot_arena: collections::Vec<CellId>,
    /// Flat arena for every block's exact entry/exit cache rows; blocks index it
    /// via their `exact_entry` / `exact_exit` spans.
    pub row_arena: collections::Vec<CellId>,
    /// Flat arena of per-edge repair-pool indices (or [`NO_REPAIR`]); blocks
    /// index it via their `repair` span.
    pub repair_index_arena: collections::Vec<u32>,
    /// Function-scope preserved-class nomination, per local slot (indexed by
    /// `CellId.0`): `true` when the residency solver priced the local's
    /// call tax at the preserved rate. The walker and rewriter keep nominated
    /// residents alive across survivable (direct local-JIT) calls, and the
    /// machine reads the same bit as its preserved-lane placement preference.
    pub preferred_preserved: collections::Vec<bool>,
}

impl FunctionPlan {
    /// `block`'s planned-resident set (admission policy): a slice into
    /// [`Self::resident_arena`].
    #[inline]
    pub(crate) fn planned_residents(&self, block_index: usize) -> &[CellId] {
        &self.resident_arena[self.blocks[block_index].planned_residents.range()]
    }

    /// Whether `slot` is preserved-class nominated: such a resident survives a
    /// survivable (direct local-JIT) call instead of being killed by it.
    #[inline]
    pub(crate) fn is_preserved_nominated(&self, slot: CellId) -> bool {
        self.preferred_preserved
            .get(slot.0 as usize)
            .copied()
            .unwrap_or(false)
    }
}
