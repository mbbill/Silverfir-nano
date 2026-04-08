//! OS abstraction layer for the native JIT runtime.
//!
//! This module is the *only* place in the crate that talks to host OS
//! primitives (mmap, VirtualAlloc, sigaction, …). Everything else in the
//! runtime layer — `code_buf`, `guard_pages`, `trap_signal` — must call
//! through this API instead of invoking syscalls directly.
//!
//! # Structure
//!
//! Exactly one OS submodule is active for a given build, chosen by the
//! `sf_os_*` cfgs emitted from the crate's `build.rs`:
//!
//! - `sf_os_linux`   → `linux` (shares POSIX primitives with macOS via `posix`)
//! - `sf_os_macos`   → `macos` (shares POSIX primitives with Linux via `posix`)
//! - `sf_os_windows` → `windows`
//! - `sf_os_none`    → `none` (bare-metal; embedder provides extern shims)
//!
//! The active module re-exports the following free functions, which form
//! the full API surface the rest of the runtime is allowed to use.
//!
//! ## Executable code buffer primitives
//!
//! - `alloc_executable(capacity)`  → reserves `capacity` bytes suitable for
//!   holding JIT-emitted machine code. The returned region must be writable
//!   after `begin_write_executable` and executable after
//!   `finish_write_executable`.
//! - `free_executable(base, capacity)` → releases the region.
//! - `begin_write_executable(base, capacity)` → make the region writable.
//!   On macOS this is a per-thread W^X toggle
//!   (`pthread_jit_write_protect_np`); on Linux/Windows it is a page
//!   permission change.
//! - `finish_write_executable(base, capacity, written_start, written_len)`
//!   → flip back to executable and flush the instruction cache over the
//!   actually-modified range.
//!
//! ## Guarded linear memory primitives
//!
//! Compiled only when `sf_has_guard_pages` is set (see `build.rs`).
//!
//! - `reserve_guarded(total)` → reserves `total` bytes as unmapped / no-access
//!   backing for a wasm linear memory with guard pages. No pages are
//!   immediately usable — they must be committed via `commit_guarded`.
//! - `commit_guarded(base, offset, len)` → upgrade `[offset, offset + len)`
//!   within the reservation to readable+writable.
//! - `release_guarded(base, total)` → release the whole reservation.
//!
//! # Adding a new target
//!
//! 1. Add the `target_os = "..."` → `sf_os_*` mapping in `build.rs`.
//! 2. Add a module here that exports the functions listed above.
//! 3. Gate it below with `#[cfg(sf_os_*)]`.
//!
//! No other file in the crate should need to change.

#[cfg(sf_has_posix)]
mod posix;

// Per-(arch × os) signal handler modules live here. Only compiled when
// `sf_has_guard_pages` is set, because they are the only consumers of
// `trap_signal`. Exactly one platform submodule is active for a given
// build; see `signal/mod.rs`.
#[cfg(sf_has_guard_pages)]
pub(in crate::vm::runtime) mod signal;

#[cfg(sf_os_linux)]
mod linux;
#[cfg(sf_os_linux)]
pub(crate) use linux::*;

#[cfg(sf_os_macos)]
mod macos;
#[cfg(sf_os_macos)]
pub(crate) use macos::*;

#[cfg(sf_os_windows)]
mod windows;
#[cfg(sf_os_windows)]
pub(crate) use windows::*;

#[cfg(sf_os_none)]
mod none;
#[cfg(sf_os_none)]
pub(crate) use none::*;
