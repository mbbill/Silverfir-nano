//! Planning configuration derived from the backend.

use crate::vm::backend::{BackendConfig, BackendKind};

/// Planning configuration derived from the selected backend.
///
/// This is intentionally a small, explicit resource budget. Backend codegen
/// should consume the results of planning, not keep reaching back into this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanConfig {
    pub backend: BackendKind,
    pub hot_local_count: u8,
    pub tos_register_count: u8,
}

impl PlanConfig {
    #[inline]
    pub const fn for_backend(backend: BackendKind, value: BackendConfig) -> Self {
        Self {
            backend,
            hot_local_count: value.hot_local_count,
            tos_register_count: value.tos_register_count,
        }
    }

    #[inline]
    pub const fn operand_base(self, frame_size: u16) -> u16 {
        frame_size
    }
}

impl From<(BackendKind, BackendConfig)> for PlanConfig {
    #[inline]
    fn from(value: (BackendKind, BackendConfig)) -> Self {
        Self::for_backend(value.0, value.1)
    }
}
