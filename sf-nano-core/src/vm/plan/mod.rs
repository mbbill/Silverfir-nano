//! Frontend preparation between decoded Wasm semantics and prepared LIR.
//!
//! Responsibilities:
//! - canonical frame layout
//! - local-cache preference analysis
//! - semantic-to-LIR preparation with explicit spill/fill
//! - prepared-LIR grouping side metadata

pub mod config;
pub mod frame;
pub mod group;
pub mod policy;
pub mod prepare;

mod local_cache;

pub use group::{GroupId, GroupPlan, GroupRange, GroupTerminator};
pub use local_cache::analyze_local_cache_prefs;
pub use policy::{GroupingMode, PlanPolicy};
pub use prepare::{prepare_function, PrepareInput, PreparedFunction};
