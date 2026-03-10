//! Standalone native backend.
//!
//! Important design intent:
//! - no `Instruction`
//! - no `NativeInst`
//! - no generic `tos_slots`
//! - no `read_t0` / `read_topN` / `write_tN` helper entry families
//! - no backend-side rediscovery of grouping
//!
//! Native consumes:
//! - shared planning-stage groups
//! - backend-facing LIR
//! and emits direct native entries plus uniform cold wrappers.

pub mod arm64;
pub mod bridge;
pub mod build;
pub mod code;
pub mod code_buf;
pub mod context;
pub mod dump;
pub mod entry;
pub mod finalizer;
pub mod helper;
pub mod helper_meta;
pub mod jitdump;
pub mod lower;
pub mod map;
pub mod precompile;
pub mod resolve;
pub mod runtime;
pub mod stats;

pub use entry::NativeEntry;
pub use stats::{native_capacity_skips, native_stats, native_stats_snapshot, NativeStatsSnapshot};
