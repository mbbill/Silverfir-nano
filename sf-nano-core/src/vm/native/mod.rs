//! Standalone native backend.
//!
//! Native consumes backend-facing CFG + SSA LIR and lowers it into one
//! target-independent native IR before ISA-specific emission.
//!
//! It must not drift back toward:
//! - interpreter-style instruction streams
//! - backend-side stack reconstruction
//! - legacy LIR/window semantics
//! - ISA-specific semantic optimization

pub mod arch;
pub mod bridge;
pub mod build;
pub mod code;
pub mod code_buf;
pub mod context;
pub mod dump;
pub mod entry;
pub mod executor;
pub mod finalizer;
pub mod helper;
pub mod helper_meta;
pub mod ir;
pub mod jitdump;
pub mod lower;
pub mod map;
pub mod precompile;
pub mod resolve;
pub mod runtime;
pub mod stats;

pub use entry::NativeEntry;
pub use stats::{native_capacity_skips, native_stats, native_stats_snapshot, NativeStatsSnapshot};
