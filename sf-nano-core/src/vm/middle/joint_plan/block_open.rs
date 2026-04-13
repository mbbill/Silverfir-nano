//! Rewrite-facing block and op boundary decisions.
//!
//! Under `ALGORITHM4`, cached-local public residency is solved before rewrite.
//! Every block simply opens with its solved public set, and cached-local
//! membership no longer changes as a side effect of local accesses.

use crate::collections;

use crate::vm::middle::cfg::CfgBlockId;

use super::{
    facts::{EntryState, FunctionPlan},
    interface::{
        BeforeOpDecision, BeforeOpQuery, BlockOpenDecision, TargetEntryDecision, TransientContract,
    },
};

#[inline]
pub(crate) fn block_open_decision(plan: &FunctionPlan, block: CfgBlockId) -> BlockOpenDecision<'_> {
    let block_plan = &plan.blocks[block.as_usize()];
    BlockOpenDecision {
        transient: transient_contract(&block_plan.entry),
        cached_locals: &block_plan.tentative_entry_cached_locals,
    }
}

#[inline]
pub(crate) fn target_entry_decision(
    plan: &FunctionPlan,
    semantic_index: usize,
) -> TargetEntryDecision<'_> {
    let entry = if plan
        .op_info
        .get(semantic_index)
        .map(|info| info.is_block_start)
        .unwrap_or(false)
    {
        &plan.blocks[plan.op_info[semantic_index].block_index as usize].entry
    } else {
        &plan.entry_states[semantic_index]
    };
    TargetEntryDecision {
        transient: transient_contract(entry),
    }
}

#[inline]
pub(crate) fn before_op_decision<'plan>(
    plan: &'plan FunctionPlan,
    query: BeforeOpQuery<'_>,
) -> BeforeOpDecision<'plan> {
    let _ = query.resident_cache;
    BeforeOpDecision {
        transient: transient_contract(&plan.op_plans[query.semantic_index].before),
        drop_cached_locals: collections::Vec::new(),
    }
}

#[inline]
pub(crate) fn finalize_block_entry_cached_locals(
    plan: &FunctionPlan,
    block: CfgBlockId,
    _actual_exit: &[crate::vm::middle::frame::FrameSlot],
) -> collections::Vec<crate::vm::middle::frame::FrameSlot> {
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
