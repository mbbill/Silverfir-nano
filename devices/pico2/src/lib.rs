//! Shared library for the `devices/pico2` binaries.
//!
//! Carries the `#[global_allocator]` and the bare-metal `sf_os_*` shims
//! so `sf-nano-pico2` and `lcd_demo` share a single source of truth.
//! Binaries call `sf_nano_pico2::heap::init()` at the top of `main`
//! before any allocation happens.

#![no_std]

extern crate alloc;

pub mod heap;
pub mod os_shim;
