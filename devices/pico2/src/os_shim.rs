//! Bare-metal OS shims for sf-nano-core.
//!
//! sf-nano-core's `runtime/os/none.rs` declares these four symbols
//! `extern "C"`; the embedder (this firmware) resolves them at link
//! time. Milestone 2 provides **null** implementations that never
//! actually allocate — just enough to satisfy the linker and confirm
//! the crate compiles and links against the firmware.
//!
//! Any runtime call into these will return null / be a no-op, so the
//! JIT will fail cleanly (`CodeBuffer::new()` returns
//! `Err("sf_os_alloc_executable returned null")`) the first time it
//! tries to allocate code memory. Milestone 3 replaces these with a
//! real static RWX arena.

#![allow(clippy::missing_safety_doc)]

/// Reserve `capacity` bytes of memory suitable for holding JIT code.
/// Null stub: returns null so the JIT fails fast if it ever tries.
#[unsafe(no_mangle)]
pub extern "C" fn sf_os_alloc_executable(_capacity: usize) -> *mut u8 {
    core::ptr::null_mut()
}

/// Release a region previously returned by `sf_os_alloc_executable`.
#[unsafe(no_mangle)]
pub extern "C" fn sf_os_free_executable(_base: *mut u8, _capacity: usize) {}

/// Make the region writable so the JIT can emit machine code.
#[unsafe(no_mangle)]
pub extern "C" fn sf_os_begin_write_executable(_base: *mut u8, _capacity: usize) {}

/// Make the region executable and flush the I-cache over the written
/// range. Null stub is a no-op — real impl needs `dsb; isb` on Cortex-M.
#[unsafe(no_mangle)]
pub extern "C" fn sf_os_finish_write_executable(
    _base: *mut u8,
    _capacity: usize,
    _written_start: usize,
    _written_len: usize,
) {
}
