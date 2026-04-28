//! Rewrite-facing block and op boundary decisions.
//!
//! Under `ALGORITHM4`, cached-local public residency is solved before rewrite.
//! Every block simply opens with its solved public set, and cached-local
//! membership no longer changes as a side effect of local accesses.

use crate::collections;

use crate::vm::middle::cfg::CfgBlockId;
use crate::vm::middle::frame::FrameSlot;

use super::{
    facts::{EntryState, FunctionPlan},
    interface::{BlockOpenDecision, TargetEntryDecision, TransientContract},
};

#[inline]
pub(crate) fn block_open_decision(plan: &FunctionPlan, block: CfgBlockId) -> BlockOpenDecision<'_> {
    let block_plan = &plan.blocks[block.as_usize()];
    BlockOpenDecision {
        transient: transient_contract(&block_plan.entry),
        cached_locals: &block_plan.tentative_entry_cached_locals,
        stack_types: &block_plan.entry.stack_types,
    }
}

#[inline]
pub(crate) fn target_entry_decision(
    plan: &FunctionPlan,
    semantic_index: usize,
) -> TargetEntryDecision {
    let compact = &plan.compact_entries[semantic_index];
    TargetEntryDecision {
        stack_height: compact.stack_height,
        spill_depth: compact.spill_depth,
    }
}

#[inline]
pub(crate) fn finalize_block_entry_cached_locals(
    plan: &FunctionPlan,
    block: CfgBlockId,
    _actual_exit: &[FrameSlot],
) -> collections::Vec<FrameSlot> {
    plan.blocks[block.as_usize()]
        .tentative_entry_cached_locals
        .clone()
}

#[inline]
fn transient_contract(entry: &EntryState) -> TransientContract<'_> {
    TransientContract {
        stack_height: entry.stack_height,
        spill_depth: entry.spill_depth,
        live_types: &entry.live_types,
    }
}
