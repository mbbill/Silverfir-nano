//! Joint transient/cache planning.
//!
//! This module exposes the single planner facade used by `rewrite/`.

pub(crate) use facade::JointPlanner;
pub(crate) use interface::{LocalAccessDecision, LocalAccessQuery, TransientContract};
mod block_open;
pub(super) mod build;
pub(super) mod entry_region;
mod exact;
mod facade;
pub(super) mod facts;
pub(super) mod init_locals;
mod interface;
mod local_access;
pub(super) mod region_solver;
pub(super) mod validate;
