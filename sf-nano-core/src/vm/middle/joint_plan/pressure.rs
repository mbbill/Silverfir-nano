//! Mid-block cached-local pressure policy.
//!
//! The full long-term plan ranks locals and stack values together. The current
//! implementation here focuses on cached locals, while stack transients still
//! use the separate transient contract. Even with that limitation, the keep-key
//! ordering below gives us the intended cached-local behavior:
//! - values unused in the block are evicted first
//! - then values dead after the current point
//! - then values still used later
//! - values required by the current local op shape are effectively pinned by
//!   giving them the strongest keep key

use alloc::collections::BTreeSet;

#[cfg(test)]
use crate::vm::middle::budget::count_live_bank_budget_units;
use crate::{
    value_type::ValueType,
    vm::middle::{budget::gp_value_budget_units, frame::FrameSlot},
};

use super::facts::FunctionPlan;

const CARRY_BONUS: i32 = 1024;
const REMAINING_USE_WEIGHT: i32 = 64;
const NEXT_USE_DISTANCE_PENALTY: i32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct KeepKey {
    class: u8,
    score: i32,
}

impl KeepKey {
    #[inline]
    pub(crate) const fn weakest() -> Self {
        Self { class: 0, score: 0 }
    }

    #[inline]
    pub(crate) const fn strongest() -> Self {
        Self {
            class: 3,
            score: i32::MAX,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CacheBank {
    Gp,
    Fp,
}

#[inline]
pub(crate) fn slot_bank(local_slot_types: &[ValueType], slot: FrameSlot) -> CacheBank {
    let ty = local_slot_types
        .get(slot.0 as usize)
        .copied()
        .unwrap_or(ValueType::I64);
    if ty.is_float() {
        CacheBank::Fp
    } else {
        CacheBank::Gp
    }
}

#[inline]
pub(crate) fn slot_cost(
    local_slot_types: &[ValueType],
    gp_unit_bytes: u8,
    slot: FrameSlot,
) -> usize {
    let ty = local_slot_types
        .get(slot.0 as usize)
        .copied()
        .unwrap_or(ValueType::I64);
    if ty.is_float() {
        1
    } else {
        gp_value_budget_units(ty, gp_unit_bytes)
    }
}

#[cfg(test)]
pub(crate) fn fits_with_cached_locals(
    plan: &FunctionPlan,
    semantic_index: usize,
    resident_cache: &BTreeSet<FrameSlot>,
) -> bool {
    fits_with_live_types(
        plan,
        current_live_types(plan, semantic_index),
        resident_cache,
    ) && fits_with_live_types(
        plan,
        semantic_successor_live_types(plan, semantic_index),
        resident_cache,
    )
}

pub(crate) fn weakest_cached_local(
    plan: &FunctionPlan,
    semantic_index: usize,
    resident_cache: &BTreeSet<FrameSlot>,
    bank: CacheBank,
    used_now: Option<FrameSlot>,
) -> Option<FrameSlot> {
    let mut best: Option<(FrameSlot, KeepKey)> = None;
    for &slot in resident_cache {
        if slot_bank(&plan.local_slot_types, slot) != bank {
            continue;
        }
        let key = keep_key(plan, semantic_index, slot, used_now, false);
        if best.map(|(_, current)| key < current).unwrap_or(true) {
            best = Some((slot, key));
        }
    }
    best.map(|(slot, _)| slot)
}

pub(crate) fn keep_key(
    plan: &FunctionPlan,
    semantic_index: usize,
    slot: FrameSlot,
    used_now: Option<FrameSlot>,
    carried_bonus: bool,
) -> KeepKey {
    if used_now == Some(slot) {
        return KeepKey::strongest();
    }

    let op_info = &plan.op_info[semantic_index];
    let block = &plan.block_regions[op_info.block_index as usize];
    let Some(info) = block.info(slot) else {
        return KeepKey::weakest();
    };

    if info.used_after(op_info.block_offset) {
        let next_distance = i32::from(info.next_use_distance(op_info.block_offset).unwrap_or(0));
        let remaining = i32::from(info.remaining_use_count(op_info.block_offset));
        return KeepKey {
            class: 2,
            score: info.hot_score + remaining * REMAINING_USE_WEIGHT
                - next_distance * NEXT_USE_DISTANCE_PENALTY
                + if carried_bonus { CARRY_BONUS } else { 0 },
        };
    }

    if info.used_anywhere() {
        return KeepKey {
            class: 1,
            score: info.hot_score + if carried_bonus { CARRY_BONUS } else { 0 },
        };
    }

    KeepKey::weakest()
}

#[cfg(test)]
fn count_cache_units(
    local_slot_types: &[ValueType],
    gp_unit_bytes: u8,
    slots: impl IntoIterator<Item = FrameSlot>,
) -> (usize, usize) {
    let mut gp = 0usize;
    let mut fp = 0usize;
    for slot in slots {
        match slot_bank(local_slot_types, slot) {
            CacheBank::Gp => gp += slot_cost(local_slot_types, gp_unit_bytes, slot),
            CacheBank::Fp => fp += slot_cost(local_slot_types, gp_unit_bytes, slot),
        }
    }
    (gp, fp)
}

#[cfg(test)]
fn fits_with_live_types(
    plan: &FunctionPlan,
    live_types: &[ValueType],
    resident_cache: &BTreeSet<FrameSlot>,
) -> bool {
    let (gp_live, fp_live) = count_live_bank_budget_units(live_types, plan.gp_unit_bytes);
    let (gp_cache, fp_cache) = count_cache_units(
        &plan.local_slot_types,
        plan.gp_unit_bytes,
        resident_cache.iter().copied(),
    );
    gp_live + gp_cache <= plan.gp_dynamic_budget as usize
        && fp_live + fp_cache <= plan.fp_dynamic_budget as usize
}

#[cfg(test)]
fn semantic_successor_live_types(plan: &FunctionPlan, semantic_index: usize) -> &[ValueType] {
    let Some(next_index) = semantic_index.checked_add(1) else {
        return current_live_types(plan, semantic_index);
    };
    let Some(next_info) = plan.op_info.get(next_index) else {
        return current_live_types(plan, semantic_index);
    };
    let current_spill_delta = current_additional_spill_delta(plan, semantic_index);
    let Some(next_state) = (if next_info.is_block_start {
        plan.blocks
            .get(next_info.block_index as usize)
            .map(|block| &block.entry)
    } else {
        plan.entry_states.get(next_index)
    }) else {
        return current_live_types(plan, semantic_index);
    };

    let spill_depth = next_state
        .spill_depth
        .saturating_add(current_spill_delta)
        .min(next_state.stack_height) as usize;
    &next_state.stack_types[spill_depth..]
}

#[cfg(test)]
fn current_live_types(plan: &FunctionPlan, semantic_index: usize) -> &[ValueType] {
    let Some(info) = plan.op_info.get(semantic_index) else {
        return &plan.op_plans[semantic_index].before.live_types;
    };
    if info.is_block_start {
        return plan
            .blocks
            .get(info.block_index as usize)
            .map(|block| block.entry.live_types.as_slice())
            .unwrap_or(&plan.op_plans[semantic_index].before.live_types);
    }
    &plan.op_plans[semantic_index].before.live_types
}

#[cfg(test)]
fn current_additional_spill_delta(plan: &FunctionPlan, semantic_index: usize) -> u16 {
    let Some(info) = plan.op_info.get(semantic_index) else {
        return 0;
    };
    if !info.is_block_start {
        return 0;
    }
    let Some(block_entry) = plan
        .blocks
        .get(info.block_index as usize)
        .map(|block| &block.entry)
    else {
        return 0;
    };
    let Some(semantic_entry) = plan.entry_states.get(semantic_index) else {
        return 0;
    };
    block_entry
        .spill_depth
        .saturating_sub(semantic_entry.spill_depth)
}

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeSet;
    use alloc::vec::Vec;

    use super::*;
    use crate::vm::middle::joint_plan::facts::{
        BlockEntryStackRegion, BlockLocalRegion, BlockPlan, EntryState, OpInfo, OpPlan,
    };

    #[test]
    fn block_start_fit_uses_finalized_block_entry_transient_state() {
        let plan = FunctionPlan {
            gp_unit_bytes: 8,
            gp_dynamic_budget: 1,
            fp_dynamic_budget: 1,
            local_slot_types: alloc::vec![ValueType::I32],
            op_plans: alloc::vec![OpPlan {
                before: EntryState {
                    stack_height: 1,
                    spill_depth: 0,
                    stack_types: alloc::vec![ValueType::I32],
                    live_types: alloc::vec![ValueType::I32],
                },
                after: EntryState {
                    stack_height: 1,
                    spill_depth: 0,
                    stack_types: alloc::vec![ValueType::I32],
                    live_types: alloc::vec![ValueType::I32],
                },
            }],
            entry_states: alloc::vec![EntryState {
                stack_height: 1,
                spill_depth: 0,
                stack_types: alloc::vec![ValueType::I32],
                live_types: alloc::vec![ValueType::I32],
            }],
            op_info: alloc::vec![OpInfo {
                block_index: 0,
                block_offset: 0,
                is_block_start: true,
                local_op: None,
            }],
            block_regions: alloc::vec![BlockLocalRegion::default()],
            block_stack_regions: alloc::vec![BlockEntryStackRegion::default()],
            block_transient_regions: alloc::vec![
                crate::vm::middle::joint_plan::facts::BlockTransientRegion::default()
            ],
            blocks: alloc::vec![BlockPlan {
                entry: EntryState {
                    stack_height: 1,
                    spill_depth: 1,
                    stack_types: alloc::vec![ValueType::I32],
                    live_types: Vec::new(),
                },
                ..Default::default()
            }],
        };

        let resident_cache = BTreeSet::from([FrameSlot(0)]);
        assert!(
            fits_with_cached_locals(&plan, 0, &resident_cache),
            "block-start fit should use the finalized block entry live window, not the stale precomputed one"
        );
    }
}
