//! Joint transient/cache planning.
//!
//! This module exposes the single planner facade used by `rewrite/`.

pub(crate) use facade::JointPlanner;
pub(crate) use interface::{CellAccessDecision, CellAccessQuery, TransientContract};
mod block_open;
pub(super) mod build;
mod cell_access;
pub(super) mod entry_region;
mod exact;
mod facade;
pub(super) mod facts;
pub(super) mod init_locals;
mod interface;
pub(super) mod region_solver;
pub(super) mod validate;
