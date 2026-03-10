//! Stack-aware planning layer.
//!
//! This is the last layer allowed to reason about:
//! - rotating T-window semantics
//! - spill/fill insertion
//! - hot-local policy
//! - grouping

pub mod config;
pub mod frame;
pub mod group;
pub mod hot_local;
pub mod place;
pub mod plan;
pub mod policy;
pub mod tos;
pub mod types;

pub use plan::build_planned_program;
pub use types::{
    PlannedBrTableEntry, PlannedBranchKind, PlannedHotLocalInit, PlannedLocal, PlannedMarkerKind,
    PlannedOp, PlannedOpKind, PlannedProgram, PlanningInput,
};
