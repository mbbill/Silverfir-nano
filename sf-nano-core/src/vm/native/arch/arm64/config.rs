//! ARM64 backend budget preset.
//!
//! These numbers are policy, not ABI facts. They choose how much of the
//! physical capacity described in `abi.rs` the compiler should spend by
//! default on:
//! - GP cached locals
//! - GP transient lanes
//! - FP cached locals
//! - FP transient lanes
//!
//! Keep this separate from `abi.rs`: changing the physical register mapping or
//! save/restore layout should not silently change compiler policy, and tuning
//! the default budget should not require touching the ABI description.

use crate::vm::backend::BackendConfig;

#[inline]
pub const fn compile_backend_config() -> BackendConfig {
    // Policy choice within the ARM64 ABI capacity from `abi.rs`:
    // 3 GP cached locals (X9-X11), 4 GP transients (X12-X15),
    // 7 FP local cache (D8-D14), 6 FP transients (D3-D7, D16).
    BackendConfig::new(3, 4, 7, 6)
}
