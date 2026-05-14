//! Per-(arch × os) signal handler implementations.
//!
//! Each submodule wires up a SIGSEGV/SIGBUS handler that converts
//! guard-page faults into wasm traps. Exactly one submodule is active
//! per build, selected by the combination of `sf_os_*` and `sf_backend_*`
//! cfgs. All modules export a single entry point:
//!
//! ```ignore
//! pub(in crate::vm::runtime) unsafe fn install_platform_handler();
//! ```
//!
//! They cooperate with the generic trap table in
//! [`crate::vm::runtime::trap_signal`] through two small accessors:
//!
//! - `trap_signal::signal_count_inc_and_check()` — abort-on-storm guard
//! - `trap_signal::try_resolve_trap(pc)` — returns the JIT error-return
//!   target plus the `NativeContext` offsets needed to record and classify the
//!   trap kind
//!
//! The platform module owns the ucontext layout and the register
//! surgery; `trap_signal.rs` owns the trap table, the install
//! idempotency latch, and the lifetime of the registered JIT ranges.
//!
//! This module is only compiled when `sf_has_guard_pages` is set.

#[cfg(all(sf_os_macos, sf_backend_arm64))]
mod arm64_macos;
#[cfg(all(sf_os_macos, sf_backend_arm64))]
pub(in crate::vm::runtime) use arm64_macos::install_platform_handler;

#[cfg(all(sf_os_linux, sf_backend_arm64))]
mod arm64_linux;
#[cfg(all(sf_os_linux, sf_backend_arm64))]
pub(in crate::vm::runtime) use arm64_linux::install_platform_handler;

#[cfg(all(sf_os_macos, sf_backend_x64))]
mod x64_macos;
#[cfg(all(sf_os_macos, sf_backend_x64))]
pub(in crate::vm::runtime) use x64_macos::install_platform_handler;

#[cfg(all(sf_os_linux, sf_backend_x64))]
mod x64_linux;
#[cfg(all(sf_os_linux, sf_backend_x64))]
pub(in crate::vm::runtime) use x64_linux::install_platform_handler;

#[cfg(all(sf_os_windows, sf_backend_x64))]
mod x64_windows;
#[cfg(all(sf_os_windows, sf_backend_x64))]
pub(in crate::vm::runtime) use x64_windows::install_platform_handler;

#[cfg(all(sf_os_linux, sf_backend_armv7a))]
mod armv7a_linux;
#[cfg(all(sf_os_linux, sf_backend_armv7a))]
pub(in crate::vm::runtime) use armv7a_linux::install_platform_handler;
