//! Machine layer: LIR → MIR lowering and transforms.
//!
//! This layer sits between the middle layer (`middle/`) and the architecture
//! backends (`arch/`). It owns:
//! - `mir/` — the MachineIR contract definitions consumed by `arch/`
//! - `lower/` — the lowering passes that transform LIR into MachineIR
//! - MachineIR transforms (peephole optimization, validation)

pub mod ir_dump;
pub mod lower;
pub mod mir;
pub mod peephole;
pub mod validate;

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
    (
        STATS_GROUPS.load(Ordering::Relaxed),
        STATS_OPS.load(Ordering::Relaxed),
    )
}

#[inline]
pub const fn native_capacity_skips() -> (usize, usize) {
    (0, 0)
}
