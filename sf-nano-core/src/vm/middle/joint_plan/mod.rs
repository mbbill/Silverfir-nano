//! Joint transient/cache planning.
//!
//! This module exposes the single planner facade used by `rewrite/`.

#[allow(unused_imports)]
pub(crate) use facade::JointPlanner;
#[allow(unused_imports)]
pub(crate) use interface::{
    BeforeOpDecision, BeforeOpQuery, BlockExitDecision, BlockOpenDecision, EdgeRepairDecision,
    EdgeRepairQuery, FunctionSetupDecision, LocalAccessDecision, LocalAccessQuery,
    TargetEntryDecision, TransientContract,
};
mod block_open;
pub(super) mod build;
pub(super) mod canonical;
pub(super) mod entry_region;
mod facade;
pub(super) mod facts;
mod interface;
mod local_access;
mod policy;
mod pressure;
mod repair;
pub(super) mod validate;
