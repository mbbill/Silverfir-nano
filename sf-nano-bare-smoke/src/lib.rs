//! Bare-metal smoke test for sf-nano-core on `target_os = "none"`.
//!
//! sf-nano-core's `runtime/os/none.rs` declares the extern "C" symbols
//! that a bare-metal embedder must provide in order to allocate and
//! manage executable JIT code memory without a hosted OS. This crate
//! provides stub implementations of those symbols and links against
//! sf-nano-core with `--no-default-features --features jit`.
//!
//! `cargo check` inside this crate compiles sf-nano-core against
//! `aarch64-unknown-none` (see `.cargo/config.toml`) which activates
//! `sf_os_none` and pulls in `runtime/os/none.rs`. If the extern symbol
//! contract drifts on either side — Rust frontend signature, attribute,
//! or visibility — this crate stops compiling.
//!
//! The stubs here return trivial placeholder values. They are not a
//! working allocator: a real embedder would back them with a static
//! RWX pool, an MMU-driven allocator, or whatever their runtime has.
//! Calling `CodeBuffer::new()` against these stubs would return
//! `Err("sf_os_alloc_executable returned null")` — which is expected:
//! this crate proves the *contract* compiles, not that the *runtime*
//! functions.

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]

extern crate alloc;

use sf_nano_core as _;

/// Reserve `capacity` bytes of memory suitable for holding JIT-emitted
/// machine code. A real embedder would return a pointer into an RWX
/// region (or a region that can be toggled RW/RX on demand); the smoke
/// stub returns null so that any attempt to use it fails early.
#[unsafe(no_mangle)]
pub extern "C" fn sf_os_alloc_executable(_capacity: usize) -> *mut u8 {
    core::ptr::null_mut()
}

/// Release a region previously returned by `sf_os_alloc_executable`.
#[unsafe(no_mangle)]
pub extern "C" fn sf_os_free_executable(_base: *mut u8, _capacity: usize) {}

/// Make the region writable so the JIT can emit machine code into it.
/// On a target with W^X the embedder would toggle page permissions or
/// equivalent; the smoke stub is a no-op.
#[unsafe(no_mangle)]
pub extern "C" fn sf_os_begin_write_executable(_base: *mut u8, _capacity: usize) {}

/// Mark the region executable, and flush the instruction cache over the
/// range `[written_start, written_start + written_len)` so the newly
/// emitted machine code becomes observable to the CPU's fetch unit.
#[unsafe(no_mangle)]
pub extern "C" fn sf_os_finish_write_executable(
    _base: *mut u8,
    _capacity: usize,
    _written_start: usize,
    _written_len: usize,
) {
}
