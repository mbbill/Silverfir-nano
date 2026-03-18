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

/// Minimal native stats surface for CLI/debug output.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeStatsSnapshot {
    pub groups: usize,
    pub ops: usize,
    pub bytes_emitted: usize,
    pub groups_skipped: usize,
    pub ops_skipped: usize,
}

use core::sync::atomic::{AtomicUsize, Ordering};

static STATS_GROUPS: AtomicUsize = AtomicUsize::new(0);
static STATS_OPS: AtomicUsize = AtomicUsize::new(0);
static STATS_BYTES: AtomicUsize = AtomicUsize::new(0);

pub fn set_native_stats(groups: usize, ops: usize, bytes_emitted: usize) {
    STATS_GROUPS.store(groups, Ordering::Relaxed);
    STATS_OPS.store(ops, Ordering::Relaxed);
    STATS_BYTES.store(bytes_emitted, Ordering::Relaxed);
}

#[inline]
pub fn native_stats_snapshot() -> NativeStatsSnapshot {
    NativeStatsSnapshot {
        groups: STATS_GROUPS.load(Ordering::Relaxed),
        ops: STATS_OPS.load(Ordering::Relaxed),
        bytes_emitted: STATS_BYTES.load(Ordering::Relaxed),
        groups_skipped: 0,
        ops_skipped: 0,
    }
}

#[inline]
pub fn native_stats() -> (usize, usize) {
    (STATS_GROUPS.load(Ordering::Relaxed), STATS_OPS.load(Ordering::Relaxed))
}

#[inline]
pub const fn native_capacity_skips() -> (usize, usize) {
    (0, 0)
}
