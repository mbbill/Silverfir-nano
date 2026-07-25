//! Local slot-vs-cache lowering decisions.
//!
//! `ALGORITHM4` chooses one public cached-local set per block owner region.
//! Rewrite should use cache form exactly for those public residents. Private
//! flex promotion can be layered on later, but v1 intentionally keeps the
//! membership policy simple and explicit.

use super::{
    facts::FunctionPlan,
    interface::{CellAccessDecision, CellAccessQuery},
};

#[inline]
pub(crate) fn decide_local_access(
    plan: &FunctionPlan,
    query: CellAccessQuery<'_>,
) -> CellAccessDecision {
    if query.resident_cache.contains(&query.slot) {
        return CellAccessDecision::Cache;
    }

    if plan
        .planned_residents(query.block.as_usize())
        .contains(&query.slot)
    {
        CellAccessDecision::Cache
    } else {
        CellAccessDecision::Slot
    }
}
