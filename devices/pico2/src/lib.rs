//! Shared library for the `devices/pico2` binaries.
//!
//! Carries the `#[global_allocator]`, the bare-metal `sf_os_*` shims,
//! and the per-board sf-nano-core configuration so `sf-nano-pico2`
//! and `lcd_demo` share a single source of truth. Binaries call
//! [`init`] once at the top of `main` before any allocation or JIT
//! work.

#![no_std]

extern crate alloc;

pub mod board;
pub mod config;
pub mod display;
pub mod heap;
pub mod mandelbrot_kernel;
pub mod os_shim;

/// Install the heap + runtime configuration for this board. Must be
/// called exactly once at startup, before any `Box`/`Vec` is
/// constructed and before sf-nano-core instantiates any module.
pub fn init() {
    heap::init();
    config::init();
}
