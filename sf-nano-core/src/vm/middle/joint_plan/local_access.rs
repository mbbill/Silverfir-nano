//! Local access policy helpers.
//!
//! The policy implemented here follows the simplified rules from `PLAN.md`:
//! - `local.set` always prefers cache form
//! - `local.get` and `local.tee` always need their value result, but only cache
//!   the local if the extra residency is better than the weakest current cache
//! - if the local is already resident, use the cache form directly

use super::{
    facts::{FunctionPlan, LocalOpKind},
    interface::{LocalAccessDecision, LocalAccessQuery},
    pressure::{
        fits_with_extra_cached_local, keep_key, slot_bank, target_keep_key_after_local_access,
        weakest_cached_local,
    },
};

#[inline]
pub(crate) fn decide_local_access(
    plan: &FunctionPlan,
    query: LocalAccessQuery<'_>,
) -> LocalAccessDecision {
    if query.resident_cache.contains(&query.slot) {
        return LocalAccessDecision::Cache;
    }

    let Some((slot, kind)) = plan.op_info[query.semantic_index].local_op else {
        return LocalAccessDecision::Slot;
    };
    debug_assert_eq!(slot, query.slot);

    match kind {
        LocalOpKind::Set => LocalAccessDecision::Cache,
        LocalOpKind::Get | LocalOpKind::Tee => {
            if fits_with_extra_cached_local(
                plan,
                query.semantic_index,
                query.resident_cache,
                query.slot,
            ) {
                return LocalAccessDecision::Cache;
            }

            let target_keep =
                target_keep_key_after_local_access(plan, query.semantic_index, query.slot);
            let Some(victim) = weakest_cached_local(
                plan,
                query.semantic_index,
                query.resident_cache,
                slot_bank(&plan.local_slot_types, query.slot),
                None,
            ) else {
                return LocalAccessDecision::Slot;
            };

            if target_keep > keep_key(plan, query.semantic_index, victim, None, false) {
                LocalAccessDecision::Cache
            } else {
                LocalAccessDecision::Slot
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeSet;

    use super::*;
    use crate::{
        value_type::ValueType,
        vm::middle::{
            frame::FrameSlot,
            joint_plan::facts::{
                BlockEntryStackRegion, BlockLocalInfo, BlockLocalRegion, BlockPlan, EntryState,
                FirstAccessKind, OpInfo, OpPlan,
            },
        },
    };

    fn plan_for_local_access(
        target_kind: LocalOpKind,
        target_hot: i32,
        target_offsets: &[u16],
        victim_hot: i32,
        victim_offsets: &[u16],
    ) -> FunctionPlan {
        FunctionPlan {
            gp_unit_bytes: 8,
            gp_dynamic_budget: 2,
            fp_dynamic_budget: 2,
            local_slot_types: alloc::vec![ValueType::I32, ValueType::I32],
            op_plans: alloc::vec![OpPlan {
                before: EntryState {
                    stack_height: 1,
                    spill_depth: 0,
                    stack_types: alloc::vec![ValueType::I32],
                    live_types: alloc::vec![ValueType::I32],
                },
                ..Default::default()
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
                local_op: Some((FrameSlot(0), target_kind)),
            }],
            block_regions: alloc::vec![BlockLocalRegion {
                ranked_slots: alloc::vec![FrameSlot(0), FrameSlot(1)],
                locals: alloc::vec![
                    Some(BlockLocalInfo {
                        slot: FrameSlot(0),
                        entry_first_access_kind: Some(match target_kind {
                            LocalOpKind::Set | LocalOpKind::Tee => FirstAccessKind::WriteFirst,
                            LocalOpKind::Get => FirstAccessKind::ReadFirst,
                        }),
                        entry_first_read_distance: Some(0),
                        entry_first_write_distance: Some(0),
                        entry_read_count: 2,
                        entry_write_count: 1,
                        entry_hot_score: target_hot,
                        first_access_kind: Some(match target_kind {
                            LocalOpKind::Set | LocalOpKind::Tee => FirstAccessKind::WriteFirst,
                            LocalOpKind::Get => FirstAccessKind::ReadFirst,
                        }),
                        first_read_distance: Some(0),
                        first_write_distance: Some(0),
                        read_count: 2,
                        write_count: 1,
                        access_offsets: target_offsets.to_vec(),
                        hot_score: target_hot,
                    }),
                    Some(BlockLocalInfo {
                        slot: FrameSlot(1),
                        entry_first_access_kind: Some(FirstAccessKind::ReadFirst),
                        entry_first_read_distance: Some(0),
                        entry_first_write_distance: None,
                        entry_read_count: 1,
                        entry_write_count: 0,
                        entry_hot_score: victim_hot,
                        first_access_kind: Some(FirstAccessKind::ReadFirst),
                        first_read_distance: Some(0),
                        first_write_distance: None,
                        read_count: 1,
                        write_count: 0,
                        access_offsets: victim_offsets.to_vec(),
                        hot_score: victim_hot,
                    }),
                ],
            }],
            block_stack_regions: alloc::vec![BlockEntryStackRegion::default()],
            block_transient_regions: alloc::vec![
                crate::vm::middle::joint_plan::facts::BlockTransientRegion::default()
            ],
            blocks: alloc::vec![BlockPlan {
                entry: EntryState {
                    stack_height: 1,
                    spill_depth: 0,
                    stack_types: alloc::vec![ValueType::I32],
                    live_types: alloc::vec![ValueType::I32],
                },
                ..Default::default()
            }],
        }
    }

    #[test]
    fn local_set_always_prefers_cache() {
        let plan = plan_for_local_access(LocalOpKind::Set, 10, &[0], 10, &[0]);
        let resident_cache = BTreeSet::new();
        let decision = decide_local_access(
            &plan,
            LocalAccessQuery {
                semantic_index: 0,
                slot: FrameSlot(0),
                resident_cache: &resident_cache,
            },
        );
        assert_eq!(decision, LocalAccessDecision::Cache);
    }

    #[test]
    fn local_get_caches_when_target_outranks_weakest_victim() {
        let plan = plan_for_local_access(LocalOpKind::Get, 400, &[0, 1], 10, &[0]);
        let resident_cache = BTreeSet::from([FrameSlot(1)]);
        let decision = decide_local_access(
            &plan,
            LocalAccessQuery {
                semantic_index: 0,
                slot: FrameSlot(0),
                resident_cache: &resident_cache,
            },
        );
        assert_eq!(decision, LocalAccessDecision::Cache);
    }

    #[test]
    fn local_get_uses_slot_when_target_is_colder_than_resident_cache() {
        let plan = plan_for_local_access(LocalOpKind::Get, 5, &[0, 1], 200, &[0, 1]);
        let resident_cache = BTreeSet::from([FrameSlot(1)]);
        let decision = decide_local_access(
            &plan,
            LocalAccessQuery {
                semantic_index: 0,
                slot: FrameSlot(0),
                resident_cache: &resident_cache,
            },
        );
        assert_eq!(decision, LocalAccessDecision::Slot);
    }
}
