//! Boundary-shaping helpers.
//!
//! Cached-local entry is the real boundary policy problem in the current
//! middle-end. Stack/SSA edge shape is already mostly solved by Wasm semantics
//! plus block params. The planner chooses a tentative cached-local entry set
//! per block, rewrite lowers once from it, and then finalizes the public block
//! entry from actual use/exit feedback.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::vm::middle::cfg::CfgBlockId;

use super::{
    facts::{EntryState, FunctionPlan, LocalOpKind},
    interface::{
        BeforeOpDecision, BeforeOpQuery, BlockOpenDecision, TargetEntryDecision,
        TransientContract,
    },
    local_access::decide_local_access,
    pressure::{
        drop_until_fits, fits_with_extra_cached_local, keep_key, slot_bank,
        target_keep_key_after_local_access, weakest_cached_local,
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

pub(crate) fn before_op_decision<'plan>(
    plan: &'plan FunctionPlan,
    op_force_drop_all_caches: &[bool],
    query: BeforeOpQuery<'_>,
) -> BeforeOpDecision<'plan> {
    let mut working = query
        .resident_cache
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut drop_cached_locals = Vec::new();

    // Hard barriers flush all cached locals unconditionally.
    if op_force_drop_all_caches
        .get(query.semantic_index)
        .copied()
        .unwrap_or(false)
    {
        drop_cached_locals.extend(working.iter().copied());
        working.clear();
    }

    // For get/tee, decide whether the target local deserves one extra cache
    // slot after the op. If yes and budgets are tight, drop the weakest cached
    // local in the same bank before lowering the op.
    if let Some((slot, kind)) = plan.op_info[query.semantic_index].local_op {
        if matches!(kind, LocalOpKind::Get | LocalOpKind::Tee) && !working.contains(&slot) {
            let access = decide_local_access(
                plan,
                super::interface::LocalAccessQuery {
                    semantic_index: query.semantic_index,
                    slot,
                    resident_cache: &working,
                },
            );
            if matches!(access, super::interface::LocalAccessDecision::Cache)
                && !fits_with_extra_cached_local(plan, query.semantic_index, &working, slot)
            {
                let target_keep =
                    target_keep_key_after_local_access(plan, query.semantic_index, slot);
                while !fits_with_extra_cached_local(plan, query.semantic_index, &working, slot) {
                    let Some(victim) = weakest_cached_local(
                        plan,
                        query.semantic_index,
                        &working,
                        slot_bank(&plan.local_slot_types, slot),
                        None,
                    ) else {
                        break;
                    };
                    if keep_key(plan, query.semantic_index, victim, None, false) >= target_keep {
                        break;
                    }
                    working.remove(&victim);
                    drop_cached_locals.push(victim);
                }
            }
        }
    }

    // After optional admission drops, trim any cached locals that still make
    // the current transient live window overflow the dynamic bank budgets.
    for slot in drop_until_fits(plan, query.semantic_index, &working, None, None) {
        if working.remove(&slot) {
            drop_cached_locals.push(slot);
        }
    }

    BeforeOpDecision {
        transient: transient_contract(before_op_entry_state(plan, query.semantic_index)),
        drop_cached_locals,
    }
}

/// Finalize a public block entry after lowering once from the tentative entry.
///
/// The only legal post-lowering trim is removing locals that were carried into
/// the tentative entry, were never used by the block, and did not survive the
/// actual exit. Used locals stay, and unused locals that naturally carried
/// through stay too.
pub(crate) fn finalize_block_entry_cached_locals(
    plan: &FunctionPlan,
    block: CfgBlockId,
    actual_exit: &[crate::vm::middle::frame::FrameSlot],
) -> Vec<crate::vm::middle::frame::FrameSlot> {
    let block_index = block.as_usize();
    let tentative_entry = &plan.blocks[block_index].tentative_entry_cached_locals;
    let region = &plan.block_regions[block_index];
    tentative_entry
        .iter()
        .copied()
        .filter(|slot| {
            region
                .info(*slot)
                .map(|info| info.used_anywhere())
                .unwrap_or(false)
                || actual_exit.contains(slot)
        })
        .collect()
}

#[inline]
fn transient_contract(entry: &EntryState) -> TransientContract<'_> {
    TransientContract {
        stack_height: entry.stack_height,
        spill_depth: entry.spill_depth,
        live_types: &entry.live_types,
    }
}

#[inline]
fn before_op_entry_state(plan: &FunctionPlan, semantic_index: usize) -> &EntryState {
    let info = &plan.op_info[semantic_index];
    if info.is_block_start {
        &plan.blocks[info.block_index as usize].entry
    } else {
        &plan.op_plans[semantic_index].before
    }
}
