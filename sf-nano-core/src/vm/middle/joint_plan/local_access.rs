//! Local slot-vs-cache lowering decisions.
//!
//! `ALGORITHM4` chooses one public cached-local set per block owner region.
//! Rewrite should use cache form exactly for those public residents. Private
//! flex promotion can be layered on later, but v1 intentionally keeps the
//! membership policy simple and explicit.

use super::{
    facts::FunctionPlan,
    interface::{LocalAccessDecision, LocalAccessQuery},
};

#[inline]
pub(crate) fn decide_local_access(
    plan: &FunctionPlan,
    query: LocalAccessQuery<'_>,
) -> LocalAccessDecision {
    if query.resident_cache.contains(&query.slot) {
        return LocalAccessDecision::Cache;
    }

    let block_index = plan.op_info[query.semantic_index].block_index as usize;
    if plan.blocks[block_index]
        .tentative_entry_cached_locals
        .contains(&query.slot)
    {
        LocalAccessDecision::Cache
    } else {
        LocalAccessDecision::Slot
    }
}
