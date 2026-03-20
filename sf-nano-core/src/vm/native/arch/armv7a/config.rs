//! ARMv7-A backend budget preset.
//!
//! With 16 GP registers (R0-R15) and VFPv3-D16 (D0-D15), budgets are:
//! - GP cached locals: 4 (R5-R7 and R9)
//! - GP transient lanes: 4
//! - FP cached locals: 8 (D8-D15, callee-saved)
//! - FP transient lanes: 5 (D3-D7)

use crate::vm::backend::BackendConfig;

#[inline]
pub const fn compile_backend_config() -> BackendConfig {
    BackendConfig::new_with_gp_unit_bytes(4, 4, 8, 5, 4)
}
