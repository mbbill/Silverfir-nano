//! x86_64 backend budget preset.
//!
//! Policy: how much of the physical register capacity described in `abi.rs`
//! the compiler should spend by default. Keep separate from `abi.rs` so
//! changing the physical register mapping does not silently change compiler
//! policy.

use crate::vm::backend::BackendConfig;

#[inline]
pub const fn compile_backend_config() -> BackendConfig {
    // Policy choice within the x86_64 System V ABI capacity from `abi.rs`:
    //  2 GP cached locals  (R14, R15) — callee-saved, free at helper calls
    //  7 GP transients     (RCX, RDX, RSI, RDI, R8, R9, R10)
    // 10 FP cached locals  (XMM6-XMM15) — caller-saved, save/restore at calls
    //  6 FP transients     (XMM0-XMM5)
    BackendConfig::new_with_gp_unit_bytes(2, 7, 10, 6, 8)
}
