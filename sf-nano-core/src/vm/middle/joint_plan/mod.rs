//! Joint transient/cache planning.
//!
//! This module exposes the single planner facade used by `rewrite.rs`.

#[allow(unused_imports)]
pub(crate) use planner::{
    BeforeOpDecision, BeforeOpQuery, BlockExitDecision, BlockOpenDecision, EdgeRepairDecision,
    EdgeRepairQuery, FunctionSetupDecision, JointPlanner, LocalAccessDecision, LocalAccessQuery,
    TargetEntryDecision, TransientContract,
};
pub(super) mod builder;
pub(super) mod canonical;
mod planner;
mod policy;
pub(super) mod scan;
pub(super) mod types;
pub(super) mod validate;
