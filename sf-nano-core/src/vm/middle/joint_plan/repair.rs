//! Cached-local edge repair.
//!
//! Under the current Wasm/SSA shape, stack repair is mostly solved already.
//! What remains here is just the cached-local set difference between a
//! predecessor exit and a finalized successor entry.

use super::{
    facts::{FirstAccessKind, FunctionPlan},
    interface::{EdgeRepairDecision, EdgeRepairQuery},
};

pub(crate) fn derive_edge_repair(
    plan: &FunctionPlan,
    query: EdgeRepairQuery<'_>,
) -> EdgeRepairDecision {
    let mut decision = EdgeRepairDecision::default();
    for &slot in query.pred_exit {
        if !query.succ_entry.contains(&slot) {
            decision.drop_cached_locals.push(slot);
        }
    }
    for &slot in query.succ_entry {
        if !query.pred_exit.contains(&slot) {
            let materialization = query
                .succ_block
                .and_then(|block| plan.block_regions.get(block.as_usize()))
                .and_then(|region| region.info(slot))
                .and_then(|info| info.entry_first_access_kind);
            if matches!(materialization, Some(FirstAccessKind::WriteFirst)) {
                decision.reserve_cached_locals.push(slot);
            } else {
                decision.ensure_cached_locals.push(slot);
            }
        }
    }
    decision
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::middle::{
        cfg::CfgBlockId,
        frame::FrameSlot,
        joint_plan::facts::{
            BlockLocalInfo, BlockLocalRegion, EntryState, FirstAccessKind, FunctionPlan,
        },
    };
    use alloc::vec::Vec;

    fn test_plan() -> FunctionPlan {
        let mut locals = alloc::vec![None; 4];
        locals[2] = Some(BlockLocalInfo {
            slot: FrameSlot(2),
            entry_first_access_kind: Some(FirstAccessKind::ReadFirst),
            ..BlockLocalInfo::default()
        });
        locals[3] = Some(BlockLocalInfo {
            slot: FrameSlot(3),
            entry_first_access_kind: Some(FirstAccessKind::WriteFirst),
            ..BlockLocalInfo::default()
        });
        FunctionPlan {
            gp_unit_bytes: 8,
            gp_dynamic_budget: 4,
            fp_dynamic_budget: 4,
            local_slot_types: alloc::vec![],
            op_plans: alloc::vec![],
            entry_states: alloc::vec![EntryState::default()],
            op_info: alloc::vec![],
            block_regions: alloc::vec![BlockLocalRegion {
                ranked_slots: Vec::new(),
                locals,
            }],
            block_stack_regions: alloc::vec![],
            block_transient_regions: alloc::vec![],
            blocks: alloc::vec![],
        }
    }

    #[test]
    fn derives_drop_ensure_and_reserve_from_final_entry_diff() {
        let plan = test_plan();
        let decision = derive_edge_repair(
            &plan,
            EdgeRepairQuery {
                succ_block: Some(CfgBlockId(0)),
                pred_exit: &[FrameSlot(0), FrameSlot(1)],
                succ_entry: &[FrameSlot(1), FrameSlot(2), FrameSlot(3)],
            },
        );
        assert_eq!(decision.drop_cached_locals, alloc::vec![FrameSlot(0)]);
        assert_eq!(decision.ensure_cached_locals, alloc::vec![FrameSlot(2)]);
        assert_eq!(decision.reserve_cached_locals, alloc::vec![FrameSlot(3)]);
    }

    #[test]
    fn defaults_to_ensure_without_successor_block_region_facts() {
        let plan = test_plan();
        let decision = derive_edge_repair(
            &plan,
            EdgeRepairQuery {
                succ_block: None,
                pred_exit: &[],
                succ_entry: &[FrameSlot(3)],
            },
        );
        assert_eq!(decision.ensure_cached_locals, alloc::vec![FrameSlot(3)]);
        assert!(decision.reserve_cached_locals.is_empty());
        assert!(decision.drop_cached_locals.is_empty());
    }
}
