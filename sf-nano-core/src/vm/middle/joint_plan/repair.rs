//! Cached-local edge repair.
//!
//! Under the current Wasm/SSA shape, stack repair is mostly solved already.
//! What remains here is just the cached-local set difference between a
//! predecessor exit and a finalized successor entry.

use super::interface::{EdgeRepairDecision, EdgeRepairQuery};

pub(crate) fn derive_edge_repair(query: EdgeRepairQuery<'_>) -> EdgeRepairDecision {
    let mut decision = EdgeRepairDecision::default();
    for &slot in query.pred_exit {
        if !query.succ_entry.contains(&slot) {
            decision.drop_cached_locals.push(slot);
        }
    }
    for &slot in query.succ_entry {
        if !query.pred_exit.contains(&slot) {
            decision.ensure_cached_locals.push(slot);
        }
    }
    decision
}
