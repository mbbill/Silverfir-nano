//! Preparation configuration supplied by the backend.

use crate::vm::backend::BackendConfig;

/// Backend-facing preparation budgets.
///
/// These values shape only the shared frontend handoff:
/// - `gp_cached_locals`/`fp_cached_locals` select how many canonical locals may be cached per bank
/// - `gp_lir_lanes`/`fp_lir_lanes` bound the transient live SSA window per bank
/// - `call_scratch_slots` reserves native-frame scratch for call-link/helper use
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanConfig {
    pub gp_cached_locals: u8,
    pub gp_lir_lanes: u8,
    pub fp_cached_locals: u8,
    pub fp_lir_lanes: u8,
    pub call_scratch_slots: u16,
}

impl PlanConfig {
    #[inline]
    pub const fn new(
        gp_cached_locals: u8,
        gp_lir_lanes: u8,
        fp_cached_locals: u8,
        fp_lir_lanes: u8,
        call_scratch_slots: u16,
    ) -> Self {
        Self {
            gp_cached_locals,
            gp_lir_lanes,
            fp_cached_locals,
            fp_lir_lanes,
            call_scratch_slots,
        }
    }

    #[inline]
    pub const fn from_backend_config(value: BackendConfig, call_scratch_slots: u16) -> Self {
        Self {
            gp_cached_locals: value.gp_local_cache_count,
            gp_lir_lanes: value.gp_lane_count,
            fp_cached_locals: value.fp_local_cache_count,
            fp_lir_lanes: value.fp_lane_count,
            call_scratch_slots,
        }
    }
}
