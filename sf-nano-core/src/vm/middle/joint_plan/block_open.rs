//! Rewrite-facing block and op boundary decisions.
//!
//! Under `ALGORITHM4`, cached-local public residency is solved before rewrite.
//! Every block simply opens with its solved public set, and cached-local
//! membership no longer changes as a side effect of local accesses.

use alloc::vec::Vec;

use crate::vm::middle::cfg::CfgBlockId;
use crate::vm::middle::frame::FrameSlot;
use crate::vm::middle::{
    budget::count_live_bank_budget_units,
    joint_plan::pressure::{keep_key, slot_bank, slot_cost, weakest_cached_local, CacheBank},
};

use super::{
    facts::{EntryState, FunctionPlan},
    interface::{
        BeforeOpDecision, BeforeOpQuery, BlockOpenDecision, PressureFallbackQuery,
        TargetEntryDecision, TransientContract,
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
        drop_cached_locals: Vec::new(),
    }
}

#[inline]
pub(crate) fn finalize_block_entry_cached_locals(
    plan: &FunctionPlan,
    block: CfgBlockId,
    _actual_exit: &[crate::vm::middle::frame::FrameSlot],
) -> Vec<crate::vm::middle::frame::FrameSlot> {
    plan.blocks[block.as_usize()]
        .tentative_entry_cached_locals
        .clone()
}

pub(crate) fn pressure_fallback_drops(
    plan: &FunctionPlan,
    query: PressureFallbackQuery<'_>,
) -> Vec<FrameSlot> {
    let effective_live_types = query
        .live_types
        .iter()
        .zip(query.live_aliases.iter())
        .filter_map(|(ty, alias)| {
            alias
                .and_then(|slot| query.resident_cache.contains(&slot).then_some(()))
                .is_none()
                .then_some(*ty)
        })
        .collect::<Vec<_>>();
    let (gp_live, fp_live) =
        count_live_bank_budget_units(&effective_live_types, plan.gp_unit_bytes);
    let mut gp_cache = 0usize;
    let mut fp_cache = 0usize;
    for &slot in query.resident_cache {
        match slot_bank(&plan.local_slot_types, slot) {
            CacheBank::Gp => {
                gp_cache += slot_cost(&plan.local_slot_types, plan.gp_unit_bytes, slot)
            }
            CacheBank::Fp => {
                fp_cache += slot_cost(&plan.local_slot_types, plan.gp_unit_bytes, slot)
            }
        }
    }

    let mut working = query
        .resident_cache
        .iter()
        .copied()
        .collect::<alloc::collections::BTreeSet<_>>();
    let mut dropped = Vec::new();

    while gp_live + gp_cache > plan.gp_dynamic_budget as usize
        || fp_live + fp_cache > plan.fp_dynamic_budget as usize
    {
        let need_gp = gp_live + gp_cache > plan.gp_dynamic_budget as usize;
        let need_fp = fp_live + fp_cache > plan.fp_dynamic_budget as usize;
        let gp_victim = need_gp
            .then(|| {
                weakest_cached_local(plan, query.semantic_index, &working, CacheBank::Gp, None)
            })
            .flatten();
        let fp_victim = need_fp
            .then(|| {
                weakest_cached_local(plan, query.semantic_index, &working, CacheBank::Fp, None)
            })
            .flatten();

        let victim = match (gp_victim, fp_victim) {
            (Some(gp_slot), Some(fp_slot)) => {
                let gp_keep = keep_key(plan, query.semantic_index, gp_slot, None, false);
                let fp_keep = keep_key(plan, query.semantic_index, fp_slot, None, false);
                if gp_keep <= fp_keep {
                    gp_slot
                } else {
                    fp_slot
                }
            }
            (Some(slot), None) | (None, Some(slot)) => slot,
            (None, None) => break,
        };
        working.remove(&victim);
        match slot_bank(&plan.local_slot_types, victim) {
            CacheBank::Gp => {
                gp_cache = gp_cache.saturating_sub(slot_cost(
                    &plan.local_slot_types,
                    plan.gp_unit_bytes,
                    victim,
                ))
            }
            CacheBank::Fp => {
                fp_cache = fp_cache.saturating_sub(slot_cost(
                    &plan.local_slot_types,
                    plan.gp_unit_bytes,
                    victim,
                ))
            }
        }
        dropped.push(victim);
    }

    dropped
}

#[inline]
fn transient_contract(entry: &EntryState) -> TransientContract<'_> {
    TransientContract {
        stack_height: entry.stack_height,
        spill_depth: entry.spill_depth,
        live_types: &entry.live_types,
    }
}
