//! Rewrite-facing block and op boundary decisions.
//!
//! Under `ALGORITHM4`, cached-local public residency is solved before rewrite.
//! Every block simply opens with its solved public set, and cached-local
//! membership no longer changes as a side effect of local accesses.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::value_type::ValueType;
use crate::vm::middle::cfg::CfgBlockId;
use crate::vm::middle::frame::FrameSlot;
use crate::vm::middle::{
    budget::gp_value_budget_units,
    joint_plan::pressure::{keep_key, slot_bank, slot_cost, CacheBank},
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

pub(crate) fn pressure_fallback_drops_into(
    plan: &FunctionPlan,
    query: PressureFallbackQuery<'_>,
    working: &mut Vec<FrameSlot>,
    dropped: &mut Vec<FrameSlot>,
) {
    let (gp_live, fp_live) = count_effective_live_bank_budget_units(
        query.live_types,
        query.live_aliases,
        query.resident_cache,
        plan.gp_unit_bytes,
    );
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

    working.clear();
    working.extend(query.resident_cache.iter().copied());
    dropped.clear();

    while gp_live + gp_cache > plan.gp_dynamic_budget as usize
        || fp_live + fp_cache > plan.fp_dynamic_budget as usize
    {
        let need_gp = gp_live + gp_cache > plan.gp_dynamic_budget as usize;
        let need_fp = fp_live + fp_cache > plan.fp_dynamic_budget as usize;
        let gp_victim = if need_gp {
            weakest_cached_local_in_working_set(
                plan,
                query.semantic_index,
                &working,
                CacheBank::Gp,
                None,
            )
        } else {
            None
        };
        let fp_victim = if need_fp {
            weakest_cached_local_in_working_set(
                plan,
                query.semantic_index,
                &working,
                CacheBank::Fp,
                None,
            )
        } else {
            None
        };

        let victim = match (gp_victim, fp_victim) {
            (Some((gp_index, gp_slot)), Some((fp_index, fp_slot))) => {
                let gp_keep = keep_key(plan, query.semantic_index, gp_slot, None, false);
                let fp_keep = keep_key(plan, query.semantic_index, fp_slot, None, false);
                if gp_keep <= fp_keep {
                    (gp_index, gp_slot)
                } else {
                    (fp_index, fp_slot)
                }
            }
            (Some(victim), None) | (None, Some(victim)) => victim,
            (None, None) => break,
        };
        let victim = working.swap_remove(victim.0);
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
}

fn count_effective_live_bank_budget_units(
    live_types: &[ValueType],
    live_aliases: &[Option<FrameSlot>],
    resident_cache: &BTreeSet<FrameSlot>,
    gp_unit_bytes: u8,
) -> (usize, usize) {
    let mut gp = 0usize;
    let mut fp = 0usize;
    for (&ty, alias) in live_types.iter().zip(live_aliases.iter()) {
        if alias
            .and_then(|slot| resident_cache.contains(&slot).then_some(()))
            .is_some()
        {
            continue;
        }
        match ty {
            ValueType::F32 | ValueType::F64 => fp += 1,
            _ => gp += gp_value_budget_units(ty, gp_unit_bytes),
        }
    }
    (gp, fp)
}

fn weakest_cached_local_in_working_set(
    plan: &FunctionPlan,
    semantic_index: usize,
    resident_cache: &[FrameSlot],
    bank: CacheBank,
    used_now: Option<FrameSlot>,
) -> Option<(usize, FrameSlot)> {
    let mut best = None;
    for (index, &slot) in resident_cache.iter().enumerate() {
        if slot_bank(&plan.local_slot_types, slot) != bank {
            continue;
        }
        let key = keep_key(plan, semantic_index, slot, used_now, false);
        if best.map(|(_, _, current)| key < current).unwrap_or(true) {
            best = Some((index, slot, key));
        }
    }
    best.map(|(index, slot, _)| (index, slot))
}

#[inline]
fn transient_contract(entry: &EntryState) -> TransientContract<'_> {
    TransientContract {
        stack_height: entry.stack_height,
        spill_depth: entry.spill_depth,
        live_types: &entry.live_types,
    }
}
