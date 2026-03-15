//! Active native backend surface.
//!
//! Only the new machine-facing IR and its lowering pipeline are active here.
//! Old implementation artifacts stay under `native/bak/`.

pub mod arch;
pub mod build;
pub mod code;
pub mod code_buf;
#[cfg(feature = "guard-pages")]
pub mod guard_pages;
mod helper;
pub mod ir;
pub mod ir_dump;
pub mod lower;
pub mod profiler;
pub mod runtime;
#[cfg(feature = "guard-pages")]
pub mod trap_signal;

/// Minimal native stats surface kept for CLI/debug compatibility while the new
/// backend is being rebuilt.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeStatsSnapshot {
    pub groups: usize,
    pub ops: usize,
    pub bytes_emitted: usize,
    pub groups_skipped: usize,
    pub ops_skipped: usize,
}

#[inline]
pub const fn native_stats_snapshot() -> NativeStatsSnapshot {
    NativeStatsSnapshot {
        groups: 0,
        ops: 0,
        bytes_emitted: 0,
        groups_skipped: 0,
        ops_skipped: 0,
    }
}

#[inline]
pub const fn native_stats() -> (usize, usize) {
    (0, 0)
}

#[inline]
pub const fn native_capacity_skips() -> (usize, usize) {
    (0, 0)
}
