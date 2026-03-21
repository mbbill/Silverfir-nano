//! Preparation configuration supplied by the backend.

use crate::vm::backend::BackendConfig;

/// Backend-facing preparation budgets.
///
/// These values shape only the shared frontend handoff:
/// - `gp_unit_bytes` describes the size of one GP budget unit
/// - `gp_local_cache_budget`/`fp_local_cache_budget` select the cached-local
///   budget per bank
/// - `gp_transient_budget`/`fp_transient_budget` bound the transient live SSA
///   window per bank
/// - `call_scratch_slots` reserves native-frame scratch for call-link/helper use
///
/// There is intentionally no `fp_unit_bytes` today: all current backends
/// charge both `f32` and `f64` as one FP budget unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlanConfig {
    pub gp_unit_bytes: u8,
    pub gp_local_cache_budget: u8,
    pub gp_transient_budget: u8,
    pub fp_local_cache_budget: u8,
    pub fp_transient_budget: u8,
    pub call_scratch_slots: u16,
}

impl PlanConfig {
    #[inline]
    pub(crate) const fn new(
        gp_local_cache_budget: u8,
        gp_transient_budget: u8,
        fp_local_cache_budget: u8,
        fp_transient_budget: u8,
        call_scratch_slots: u16,
    ) -> Self {
        Self::new_with_gp_unit_bytes(
            gp_local_cache_budget,
            gp_transient_budget,
            fp_local_cache_budget,
            fp_transient_budget,
            call_scratch_slots,
            core::mem::size_of::<usize>() as u8,
        )
    }

    #[inline]
    pub(crate) const fn new_with_gp_unit_bytes(
        gp_local_cache_budget: u8,
        gp_transient_budget: u8,
        fp_local_cache_budget: u8,
        fp_transient_budget: u8,
        call_scratch_slots: u16,
        gp_unit_bytes: u8,
    ) -> Self {
        Self {
            gp_unit_bytes,
            gp_local_cache_budget,
            gp_transient_budget,
            fp_local_cache_budget,
            fp_transient_budget,
            call_scratch_slots,
        }
    }

    #[inline]
    pub(crate) const fn from_backend_config(value: BackendConfig, call_scratch_slots: u16) -> Self {
        Self {
            gp_unit_bytes: value.gp_unit_bytes,
            gp_local_cache_budget: value.gp_local_cache_budget,
            gp_transient_budget: value.gp_transient_budget,
            fp_local_cache_budget: value.fp_local_cache_budget,
            fp_transient_budget: value.fp_transient_budget,
            call_scratch_slots,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PlanConfig;
    use crate::vm::backend::BackendConfig;

    #[test]
    fn plan_config_defaults_gp_unit_bytes_to_host_word_size() {
        let config = PlanConfig::new(1, 2, 3, 4, 5);
        assert_eq!(config.gp_unit_bytes as usize, core::mem::size_of::<usize>());
    }

    #[test]
    fn plan_config_propagates_backend_gp_unit_bytes() {
        let backend = BackendConfig::new_with_gp_unit_bytes(1, 2, 3, 4, 4);
        let plan = PlanConfig::from_backend_config(backend, 5);
        assert_eq!(plan.gp_unit_bytes, 4);
    }
}
