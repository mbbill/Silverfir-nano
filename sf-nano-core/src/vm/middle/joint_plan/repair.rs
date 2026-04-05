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
