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
pub mod plan;
pub mod policy;
pub mod spill;
pub mod tos;
