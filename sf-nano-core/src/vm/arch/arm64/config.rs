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
    // 13 GP cached locals (X23-X28 callee-saved + X9-X15 caller-saved),
    //  6 GP transients    (X3-X8),
    // 19 FP local cache   (D8-D15 callee-saved + D21-D31 caller-saved),
    // 10 FP transients    (D3-D7, D16-D20).
    BackendConfig::new_with_gp_unit_bytes(13, 6, 19, 10, 8)
}
