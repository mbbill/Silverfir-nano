//! Lightweight native backend stats.
//!
//! These are compatibility-facing counters for CLI/reporting. They do not
//! reintroduce old descriptor-stream design.

use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};

pub struct NativeStats {
    pub groups: AtomicUsize,
    pub ops: AtomicUsize,
    pub bytes_emitted: AtomicUsize,
    pub groups_skipped_capacity: AtomicUsize,
    pub ops_skipped_capacity: AtomicUsize,
}

pub static NATIVE_STATS: NativeStats = NativeStats {
    groups: AtomicUsize::new(0),
    ops: AtomicUsize::new(0),
    bytes_emitted: AtomicUsize::new(0),
    groups_skipped_capacity: AtomicUsize::new(0),
    ops_skipped_capacity: AtomicUsize::new(0),
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeStatsSnapshot {
    pub groups: usize,
    pub ops: usize,
    pub bytes_emitted: usize,
    pub groups_skipped: usize,
    pub ops_skipped: usize,
}

#[inline]
pub fn native_stats_snapshot() -> NativeStatsSnapshot {
    NativeStatsSnapshot {
        groups: NATIVE_STATS.groups.load(Relaxed),
        ops: NATIVE_STATS.ops.load(Relaxed),
        bytes_emitted: NATIVE_STATS.bytes_emitted.load(Relaxed),
        groups_skipped: NATIVE_STATS.groups_skipped_capacity.load(Relaxed),
        ops_skipped: NATIVE_STATS.ops_skipped_capacity.load(Relaxed),
    }
}

#[inline]
pub fn native_stats() -> (usize, usize) {
    (
        NATIVE_STATS.groups.load(Relaxed),
        NATIVE_STATS.ops.load(Relaxed),
    )
}

#[inline]
pub fn native_capacity_skips() -> (usize, usize) {
    (
        NATIVE_STATS.groups_skipped_capacity.load(Relaxed),
        NATIVE_STATS.ops_skipped_capacity.load(Relaxed),
    )
}
