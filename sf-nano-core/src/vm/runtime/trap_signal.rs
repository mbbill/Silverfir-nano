//! Generic trap-table and install latch for the guard-page signal handler.
//!
//! This module is the OS-agnostic half of the guard-page trap mechanism.
//! It owns:
//!
//! - the registry of JIT code ranges → error-return addresses,
//! - the signal-storm debug counter,
//! - the `TRAP_KIND_OFFSET` that tells the platform handler where to
//!   record the trap kind on the active [`NativeContext`],
//! - the one-shot install latch for the OS-specific signal handler.
//!
//! The per-(arch × os) ucontext parsing, register surgery, and sigaction
//! wiring live under [`crate::vm::runtime::os::signal`]. Each platform
//! module there exports a single `install_platform_handler()` and reads
//! back into this module through the two `pub(in crate::vm::runtime)`
//! accessors below:
//!
//! - [`signal_count_inc_and_check`] — bumps the storm counter, returns
//!   `true` if the caller should abort.
//! - [`try_resolve_trap`] — looks up a faulting PC in the trap table and
//!   returns `(error_ret, trap_kind_offset)` when the PC lies in JIT code.
//!
//! The handler is async-signal-safe: it does not allocate. It sets a
//! `trap_kind` flag in `NativeContext` (reachable via the architecture's
//! context register) and rewrites the signal frame's PC to the function's
//! `return_error_label`. The Rust caller (`eval()`) reads `trap_kind` after
//! JIT returns and creates the `WasmError`.
//!
//! This module is gated on `#[cfg(sf_has_guard_pages)]`.

use crate::collections;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use super::os::signal::install_platform_handler;

/// Debug counter: aborts after too many consecutive signals (infinite loop detection).
static SIGNAL_COUNT: AtomicUsize = AtomicUsize::new(0);

/// A registered JIT code range with its error-return address.
#[derive(Clone, Copy)]
struct JitCodeRange {
    code_start: usize,
    code_end: usize,
    return_error: usize,
}

/// Global trap table. Sorted by `code_start` for binary search.
///
/// Access is synchronized by a simple spinlock. The signal handler acquires
/// a read-only view; registration happens during compilation (rare).
static mut TRAP_TABLE: Option<collections::Vec<JitCodeRange>> = None;
static TRAP_TABLE_LOCK: AtomicBool = AtomicBool::new(false);
static HANDLER_INSTALLED: AtomicBool = AtomicBool::new(false);

/// Offset of the `trap_kind` field within `NativeContext`, set once at init.
static TRAP_KIND_OFFSET: AtomicUsize = AtomicUsize::new(0);

fn lock_trap_table() {
    while TRAP_TABLE_LOCK
        .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        core::hint::spin_loop();
    }
}

fn unlock_trap_table() {
    TRAP_TABLE_LOCK.store(false, Ordering::Release);
}

/// Register JIT code ranges in the global trap table.
pub(crate) fn register_jit_ranges(ranges: &[(usize, usize, usize)]) {
    lock_trap_table();
    unsafe {
        let table = &raw mut TRAP_TABLE;
        let table = (*table).get_or_insert_with(collections::Vec::new);
        for &(start, end, error_ret) in ranges {
            table.push(JitCodeRange {
                code_start: start,
                code_end: end,
                return_error: error_ret,
            });
        }
        table.sort_unstable_by_key(|r| r.code_start);
    }
    unlock_trap_table();
}

/// Remove JIT ranges that belong to an executable code buffer.
///
/// Code buffers own the machine code backing these ranges. Once a buffer is
/// reset or dropped, any existing fault-to-error-tail mapping inside that
/// address interval is stale even if the OS later reuses the same virtual
/// address for another module.
pub(crate) fn unregister_jit_ranges_in(code_start: usize, code_end: usize) {
    lock_trap_table();
    unsafe {
        let table = &raw mut TRAP_TABLE;
        if let Some(table) = (*table).as_mut() {
            table.retain(|range| range.code_end <= code_start || range.code_start >= code_end);
        }
    }
    unlock_trap_table();
}

/// Look up the error-return address for a faulting PC.
///
/// Must be called with the trap table lock held (or from the signal handler
/// where concurrent modification is not expected on the same thread).
unsafe fn lookup_return_error(pc: usize) -> Option<usize> {
    let Some(table) = (unsafe { &*(&raw const TRAP_TABLE) }).as_ref() else {
        return None;
    };
    let idx = table.partition_point(|r| r.code_start <= pc);
    if idx == 0 {
        return None;
    }
    let entry = &table[idx - 1];
    if pc < entry.code_end {
        Some(entry.return_error)
    } else {
        None
    }
}

/// Bump the signal-storm counter and report whether the caller should abort.
///
/// Called by every `os::signal::*` platform handler at entry. Returns
/// `true` once the counter exceeds 100 consecutive signals — a crude
/// infinite-loop detector that prevents a bad trap-table from producing
/// an unbounded re-entry loop.
#[inline]
pub(in crate::vm::runtime) fn signal_count_inc_and_check() -> bool {
    SIGNAL_COUNT.fetch_add(1, Ordering::Relaxed) > 100
}

/// Resolve a faulting PC to its trap-handling parameters.
///
/// Returns `(error_ret, trap_kind_offset)` when `pc` lies inside a
/// registered JIT code range. The caller (a platform-specific signal
/// handler) uses `error_ret` as the new PC and `trap_kind_offset` to
/// locate the `trap_kind` field inside `NativeContext`.
///
/// Returns `None` when the fault did not happen in JIT code; the caller
/// should abort because we cannot chain to another handler safely.
///
/// # Safety
///
/// Must be called from a signal context where concurrent mutation of
/// `TRAP_TABLE` on the same thread is not possible. The trap table is
/// append-only during compilation and stable during execution, so this
/// holds as long as signals only fire during `eval()`.
#[inline]
pub(in crate::vm::runtime) unsafe fn try_resolve_trap(pc: usize) -> Option<(usize, usize)> {
    let error_ret = unsafe { lookup_return_error(pc) }?;
    let trap_kind_offset = TRAP_KIND_OFFSET.load(Ordering::Relaxed);
    Some((error_ret, trap_kind_offset))
}

/// Install the signal handler (idempotent).
pub(crate) fn install_signal_handler() {
    if HANDLER_INSTALLED.load(Ordering::Relaxed) {
        return;
    }
    if HANDLER_INSTALLED
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
        .is_err()
    {
        return;
    }
    unsafe {
        install_platform_handler();
    }
}

/// Reset debug state tracked by the signal handler.
///
/// The infinite-loop detector is only meaningful within a single native entry:
/// one wasm invocation should either complete or trap after the first fault.
/// Resetting here prevents expected trapping test cases from accumulating
/// counts across many independent invocations.
pub(crate) fn reset_debug_state() {
    SIGNAL_COUNT.store(0, Ordering::Relaxed);
}

/// Drop all registered JIT ranges.
///
/// Callers must only use this when no compiled native frames from the old
/// ranges can still fault, otherwise a later trap would be unable to resolve
/// back to its owning function.
pub(crate) fn clear_registered_jit_ranges() {
    lock_trap_table();
    unsafe {
        let table = &raw mut TRAP_TABLE;
        if let Some(table) = (*table).as_mut() {
            table.clear();
        }
    }
    unlock_trap_table();
}

/// Set the byte offset of `NativeContext::trap_kind` so the signal handler
/// can write it without knowing the struct layout at compile time.
pub(crate) fn set_trap_kind_offset(offset: usize) {
    TRAP_KIND_OFFSET.store(offset, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clear_registered_jit_ranges_drops_stale_entries() {
        clear_registered_jit_ranges();
        register_jit_ranges(&[(0x1000, 0x1100, 0x2000)]);

        unsafe {
            assert_eq!(lookup_return_error(0x1080), Some(0x2000));
        }

        clear_registered_jit_ranges();

        unsafe {
            assert_eq!(lookup_return_error(0x1080), None);
        }
    }

    #[test]
    fn unregister_jit_ranges_in_drops_only_overlapping_entries() {
        clear_registered_jit_ranges();
        register_jit_ranges(&[
            (0x1000, 0x1100, 0x2000),
            (0x1200, 0x1300, 0x2200),
            (0x2000, 0x2100, 0x3000),
        ]);

        unregister_jit_ranges_in(0x1000, 0x1800);

        unsafe {
            assert_eq!(lookup_return_error(0x1080), None);
            assert_eq!(lookup_return_error(0x1280), None);
            assert_eq!(lookup_return_error(0x2080), Some(0x3000));
        }

        clear_registered_jit_ranges();
    }

    #[test]
    fn reset_debug_state_clears_signal_counter() {
        SIGNAL_COUNT.store(17, Ordering::Relaxed);
        reset_debug_state();
        assert_eq!(SIGNAL_COUNT.load(Ordering::Relaxed), 0);
    }
}
