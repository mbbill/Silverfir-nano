//! Rewrite-facing block and op boundary decisions.
//!
//! Under `ALGORITHM4`, cached-local public residency is solved before rewrite.
//! Every block simply opens with its solved public set, and cached-local
//! membership no longer changes as a side effect of local accesses.

use crate::vm::jit::middle::cfg::CfgBlockId;

use super::{
    facts::{EntryState, FunctionPlan},
    interface::{BlockOpenDecision, TargetEntryDecision, TransientContract},
};

#[inline]
pub(crate) fn block_open_decision(plan: &FunctionPlan, block: CfgBlockId) -> BlockOpenDecision<'_> {
    let block_plan = &plan.blocks[block.as_usize()];
    BlockOpenDecision {
        transient: transient_contract(&block_plan.entry),
        stack_types: &block_plan.entry.stack_types,
    }
}

#[inline]
pub(crate) fn target_entry_decision(plan: &FunctionPlan, block: CfgBlockId) -> TargetEntryDecision {
    let entry = &plan.blocks[block.as_usize()].entry;
    TargetEntryDecision {
        stack_height: entry.stack_height,
        spill_depth: entry.spill_depth,
    }
}

#[inline]
fn transient_contract(entry: &EntryState) -> TransientContract<'_> {
    TransientContract {
        stack_height: entry.stack_height,
        spill_depth: entry.spill_depth,
        live_types: entry.live_types(),
    }
}
