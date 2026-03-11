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

pub mod abi;
pub mod arch;
pub mod code;
pub mod code_buf;
pub mod compile;
pub mod helper;
pub mod ir;
mod jitdump;
pub mod lower;
mod map;
pub mod runtime;
pub mod stats;
mod symbols;

pub use compile::{build, precompile, resolve};
pub use helper::{bridge, meta as helper_meta};
pub use runtime::entry::NativeEntry;
pub use runtime::{context, entry, layout};
pub use stats::{native_capacity_skips, native_stats, native_stats_snapshot, NativeStatsSnapshot};
