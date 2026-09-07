//! Generic trap-table and install latch for the guard-page signal handler.
//!
//! This module is the OS-agnostic half of the guard-page trap mechanism.
//! It owns:
//!
//! - the registry of JIT code ranges → error-return addresses,
//! - the signal-storm debug counter,
//! - the context offsets that tell the platform handler where to record the
//!   trap kind and how to classify stack-guard faults,
//! - the one-shot install latch for the OS-specific signal handler.
//!
//! The per-(arch × os) ucontext parsing, register surgery, and sigaction
//! wiring live under [`crate::vm::jit::runtime::os::signal`]. Each platform
//! module there exports a single `install_platform_handler()` and reads
//! back into this module through the two `pub(in crate::vm::jit::runtime)`
//! accessors below:
//!
//! - [`signal_count_inc_and_check`] — bumps the storm counter, returns
//!   `true` if the caller should abort.
//! - [`try_resolve_trap`] — looks up a faulting PC in the trap table and
//!   returns the handler metadata when the PC lies in JIT code.
//!
//! The handler is async-signal-safe: it does not allocate. It sets a
//! `trap_kind` flag in `NativeContext` (reachable via the architecture's
//! context register) and rewrites the signal frame's PC to the function's
//! `body_local_error_label`. The Rust caller (`eval()`) reads `trap_kind` after
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
static STACK_END_OFFSET: AtomicUsize = AtomicUsize::new(0);
static STACK_GUARD_END_OFFSET: AtomicUsize = AtomicUsize::new(0);

pub(crate) const TRAP_MEMORY_OUT_OF_BOUNDS: u32 = 1;
pub(crate) const TRAP_STACK_OVERFLOW: u32 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::vm::jit::runtime) struct TrapResolution {
    pub error_ret: usize,
    pub trap_kind_offset: usize,
    pub stack_end_offset: usize,
    pub stack_guard_end_offset: usize,
}

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
/// Must be called with the trap table lock held, including in a signal handler.
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
pub(in crate::vm::jit::runtime) fn signal_count_inc_and_check() -> bool {
    SIGNAL_COUNT.fetch_add(1, Ordering::Relaxed) > 100
}

/// Resolve a faulting PC to its trap-handling parameters.
///
/// Returns handler metadata when `pc` lies inside a registered JIT code range.
/// The caller (a platform-specific signal handler) uses `error_ret` as the new
/// PC, `trap_kind_offset` to locate the `trap_kind` field inside
/// `NativeContext`, and the stack offsets to distinguish guarded-stack faults
/// from guarded-linear-memory faults.
///
/// Returns `None` when the fault did not happen in JIT code; the caller
/// should abort because we cannot chain to another handler safely.
///
/// # Safety
///
/// The interrupted thread must not hold the trap-table lock. Guard faults
/// originate in generated code, outside registration and code-buffer teardown.
/// Other threads may compile or drop code concurrently; the lookup takes the
/// same lock as those writers before borrowing any vector storage.
#[inline]
pub(in crate::vm::jit::runtime) unsafe fn try_resolve_trap(pc: usize) -> Option<TrapResolution> {
    lock_trap_table();
    let error_ret = unsafe { lookup_return_error(pc) };
    unlock_trap_table();
    let error_ret = error_ret?;
    let trap_kind_offset = TRAP_KIND_OFFSET.load(Ordering::Relaxed);
    let stack_end_offset = STACK_END_OFFSET.load(Ordering::Relaxed);
    let stack_guard_end_offset = STACK_GUARD_END_OFFSET.load(Ordering::Relaxed);
    Some(TrapResolution {
        error_ret,
        trap_kind_offset,
        stack_end_offset,
        stack_guard_end_offset,
    })
}

/// Classify one JIT-attributed guard-page fault.
///
/// The platform handler already knows the fault came from JIT code. If the
/// faulting address lands in the wasm-stack guard range, report stack
/// overflow; otherwise keep the historical guarded-memory classification.
#[inline]
pub(in crate::vm::jit::runtime) unsafe fn classify_trap_kind(
    ctx_ptr: *mut u8,
    fault_addr: usize,
    resolution: TrapResolution,
) -> u32 {
    let stack_end = unsafe { *(ctx_ptr.add(resolution.stack_end_offset) as *const *mut u8) };
    let stack_guard_end =
        unsafe { *(ctx_ptr.add(resolution.stack_guard_end_offset) as *const *mut u8) };
    let stack_end = stack_end as usize;
    let stack_guard_end = stack_guard_end as usize;
    if fault_addr >= stack_end && fault_addr < stack_guard_end {
        TRAP_STACK_OVERFLOW
    } else {
        TRAP_MEMORY_OUT_OF_BOUNDS
    }
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
pub(crate) fn set_context_offsets(
    trap_kind_offset: usize,
    stack_end_offset: usize,
    stack_guard_end_offset: usize,
) {
    TRAP_KIND_OFFSET.store(trap_kind_offset, Ordering::Relaxed);
    STACK_END_OFFSET.store(stack_end_offset, Ordering::Relaxed);
    STACK_GUARD_END_OFFSET.store(stack_guard_end_offset, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static TEST_TRAP_TABLE_LOCK: Mutex<()> = Mutex::new(());

    fn resolve_error(pc: usize) -> Option<usize> {
        // These test calls do not run inside a table mutation.
        unsafe { try_resolve_trap(pc) }.map(|resolution| resolution.error_ret)
    }

    #[test]
    fn clear_registered_jit_ranges_drops_stale_entries() {
        let _guard = TEST_TRAP_TABLE_LOCK.lock().unwrap();

        clear_registered_jit_ranges();
        register_jit_ranges(&[(0x1000, 0x1100, 0x2000)]);

        assert_eq!(resolve_error(0x1080), Some(0x2000));

        clear_registered_jit_ranges();

        assert_eq!(resolve_error(0x1080), None);
    }

    #[test]
    fn unregister_jit_ranges_in_drops_only_overlapping_entries() {
        let _guard = TEST_TRAP_TABLE_LOCK.lock().unwrap();

        clear_registered_jit_ranges();
        register_jit_ranges(&[
            (0x1000, 0x1100, 0x2000),
            (0x1200, 0x1300, 0x2200),
            (0x2000, 0x2100, 0x3000),
        ]);

        unregister_jit_ranges_in(0x1000, 0x1800);

        assert_eq!(resolve_error(0x1080), None);
        assert_eq!(resolve_error(0x1280), None);
        assert_eq!(resolve_error(0x2080), Some(0x3000));

        clear_registered_jit_ranges();
    }

    #[test]
    fn trap_lookup_stays_valid_while_other_threads_register_and_remove_code() {
        let _guard = TEST_TRAP_TABLE_LOCK.lock().unwrap();
        register_jit_ranges(&[(0x1000, 0x1100, 0x2000)]);
        let start = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let ranges: collections::Vec<_> = (0..128)
                    .map(|i| (0x3000 + i * 0x100, 0x3100 + i * 0x100, 0x9000))
                    .collect();
                start.wait();
                for _ in 0..1000 {
                    register_jit_ranges(&ranges);
                    unregister_jit_ranges_in(0x3000, 0xb000);
                }
            });
            start.wait();
            for _ in 0..50_000 {
                assert_eq!(resolve_error(0x1080), Some(0x2000));
                assert_eq!(resolve_error(0x2000), None);
                assert_eq!(resolve_error(0xc000), None);
            }
        });
        unregister_jit_ranges_in(0x1000, 0x1100);
    }

    #[test]
    fn reset_debug_state_clears_signal_counter() {
        SIGNAL_COUNT.store(17, Ordering::Relaxed);
        reset_debug_state();
        assert_eq!(SIGNAL_COUNT.load(Ordering::Relaxed), 0);
    }
}
