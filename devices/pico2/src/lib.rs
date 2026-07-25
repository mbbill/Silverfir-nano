//! Shared library for the `devices/pico2` binaries.
//!
//! Carries the `#[global_allocator]`, the bare-metal `sf_os_*` shims,
//! and the per-board sf-nano-core configuration so the bins share a
//! single source of truth. Binaries call [`init`] once at the top of
//! `main` before any allocation or JIT work.
//!
//! Builds for both Cortex-M33 (ARM) and Hazard3 (RV32IMAC) — all
//! arch-specific glue is concentrated in [`arch`]; everything else
//! (board, display, heap, kernels, os_shim) is single-source.

#![no_std]

extern crate alloc;

pub mod arch;
pub mod board;
pub mod config;
pub mod display;
pub mod heap;
pub mod kernels;
pub mod os_shim;

/// Install the heap for this board. Must be called exactly once at
/// startup, before any `Box`/`Vec` is constructed. The engine is built
/// separately with [`config::engine`], because it is a value now rather
/// than a process-wide setting.
pub fn init() {
    heap::init();
}
