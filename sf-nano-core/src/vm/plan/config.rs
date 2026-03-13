//! Preparation configuration supplied by the backend.

use crate::vm::backend::BackendConfig;

/// Backend-facing preparation budgets.
///
/// These values shape only the shared frontend handoff:
/// - `cached_locals` selects how many canonical locals may be mirrored later
/// - `tos_lanes` bounds the transient live SSA window in prepared LIR
/// - `call_scratch_slots` reserves native-frame scratch for call-link/helper use
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanConfig {
    pub cached_locals: u8,
    pub tos_lanes: u8,
    pub call_scratch_slots: u16,
}

impl PlanConfig {
    #[inline]
    pub const fn new(cached_locals: u8, tos_lanes: u8, call_scratch_slots: u16) -> Self {
        Self {
            cached_locals,
            tos_lanes,
            call_scratch_slots,
        }
    }

    #[inline]
    pub const fn from_backend_config(value: BackendConfig, call_scratch_slots: u16) -> Self {
        Self::new(
            value.hot_local_count,
            value.tos_register_count,
            call_scratch_slots,
        )
    }
}
