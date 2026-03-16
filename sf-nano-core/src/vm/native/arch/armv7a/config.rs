//! ARMv7-A backend budget preset.
//!
//! With 16 GP registers (R0-R15) and VFPv3-D16 (D0-D15), budgets are:
//! - GP cached locals: 2 (R5-R6, callee-saved)
//! - GP transient lanes: 6 (R3, R7, R8, R0, R1, R2)
//! - FP cached locals: 8 (D8-D15, callee-saved)
//! - FP transient lanes: 5 (D3-D7)

use crate::vm::backend::BackendConfig;

#[inline]
pub const fn compile_backend_config() -> BackendConfig {
    BackendConfig::new(2, 6, 8, 5)
}
