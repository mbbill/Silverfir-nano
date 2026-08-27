//! The register ABI the shared compiler plans against.
//!
//! This is the JIT's own contract between its middle-end and its backends,
//! not a VM-wide concept: which engine runs a module is [`crate::vm::engine`],
//! and which ISA the JIT targets is fixed by the target. "Backend" means
//! the ISA here.

/// Planning-time backend configuration.
///
/// This carries the abstract MachineIR register ABI chosen by the backend.
/// It deliberately does not expose physical registers. The shared compiler
/// sees only counts and the abstract layout:
///
/// ```text
/// [fixed | gp_volatile | gp_preserved | gp_internal_scratch | fp_volatile | fp_preserved]
/// ```
///
/// The backend's `abi.rs` owns the mapping from these abstract dynamic lanes to
/// concrete target registers. Middle and MachineIR may use the volatility
/// counts to decide which values can die at a local call and which values may
/// rely on callee preservation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BackendConfig {
    /// Size in bytes of one GP budget unit on the target backend.
    ///
    /// This is separate from the frame-slot contract. Wasm values still
    /// occupy one canonical 8-byte slot in the native frame. The planner uses
    /// this to turn semantic values into GP budget-unit costs.
    ///
    /// There is intentionally no matching `fp_unit_bytes` field today:
    /// across the currently supported backends, both `f32` and `f64` consume
    /// exactly one FP register budget unit.
    pub gp_unit_bytes: u8,
    pub gp_volatile_dynamic: u8,
    pub gp_preserved_dynamic: u8,
    pub gp_internal_scratch: u8,
    pub gp_dynamic_budget: u8,
    pub fp_volatile_dynamic: u8,
    pub fp_preserved_dynamic: u8,
    pub fp_dynamic_budget: u8,
    pub gp_arg_lanes: u8,
    pub fp_arg_lanes: u8,
    pub scalar_return_lanes: bool,
    pub call_scratch_slots: u16,
    /// Estimated per-lane cost, in native instructions, of nominating one
    /// preserved dynamic lane: the body's lazy save at entry plus restores at
    /// its return paths. The residency solver weighs this one-time cost
    /// against the trip-weighted call-crossing relief a preserved-class
    /// resident earns. Backends whose prologue bulk-saves the physical
    /// registers regardless (everyone but arm64 today) would price this 0.
    pub preserved_lane_save_overhead: u8,
    /// Every 32-bit integer instruction writes a zero-extended destination
    /// (x86_64 r32 and AArch64 W-register semantics). Lets the shared
    /// peephole relax redundant `ZeroExtend32` index obligations on indexed
    /// memory ops.
    pub gp32_defs_zero_extend: bool,
}

impl BackendConfig {
    /// Constructor for backends whose ABI does not split the dynamic bank
    /// by volatility (riscv), and for synthetic test configurations on any
    /// host. Arm targets and x86-64 reach `BackendConfig` only through
    /// [`Self::with_volatility`].
    #[inline]
    #[cfg(any(sf_backend_riscv32, sf_backend_riscv64, test))]
    pub(crate) const fn new(
        gp_unit_bytes: u8,
        gp_dynamic_budget: u8,
        fp_dynamic_budget: u8,
        call_scratch_slots: u16,
    ) -> Self {
        let gp_internal_scratch = default_gp_internal_scratch(gp_unit_bytes, gp_dynamic_budget);
        let gp_volatile_dynamic = gp_dynamic_budget.saturating_sub(gp_internal_scratch);
        Self::with_volatility(
            gp_unit_bytes,
            gp_volatile_dynamic,
            0,
            gp_internal_scratch,
            fp_dynamic_budget,
            0,
            default_gp_arg_lanes(gp_volatile_dynamic),
            default_fp_arg_lanes(fp_dynamic_budget),
            false,
            call_scratch_slots,
        )
    }

    #[inline]
    pub(crate) const fn with_volatility(
        gp_unit_bytes: u8,
        gp_volatile_dynamic: u8,
        gp_preserved_dynamic: u8,
        gp_internal_scratch: u8,
        fp_volatile_dynamic: u8,
        fp_preserved_dynamic: u8,
        gp_arg_lanes: u8,
        fp_arg_lanes: u8,
        scalar_return_lanes: bool,
        call_scratch_slots: u16,
    ) -> Self {
        let gp_dynamic_budget = gp_volatile_dynamic
            .saturating_add(gp_preserved_dynamic)
            .saturating_add(gp_internal_scratch);
        let fp_dynamic_budget = fp_volatile_dynamic.saturating_add(fp_preserved_dynamic);
        Self {
            gp_unit_bytes,
            gp_volatile_dynamic,
            gp_preserved_dynamic,
            gp_internal_scratch,
            gp_dynamic_budget,
            fp_volatile_dynamic,
            fp_preserved_dynamic,
            fp_dynamic_budget,
            gp_arg_lanes: min_u8(gp_arg_lanes, gp_volatile_dynamic),
            fp_arg_lanes: min_u8(fp_arg_lanes, fp_volatile_dynamic),
            scalar_return_lanes,
            call_scratch_slots,
            preserved_lane_save_overhead: 0,
            gp32_defs_zero_extend: false,
        }
    }

    /// Declare that every 32-bit integer instruction writes a zero-extended
    /// destination (x86_64 r32 and AArch64 W-register semantics), letting the
    /// shared peephole relax redundant `ZeroExtend32` index obligations.
    /// Leave off for backends whose 32-bit ops sign-extend (riscv64).
    #[inline]
    #[cfg(any(sf_backend_arm64, sf_backend_x64, test))]
    pub(crate) const fn with_gp32_zero_extending_defs(mut self) -> Self {
        self.gp32_defs_zero_extend = true;
        self
    }

    /// Override the estimated per-lane preserved-nomination cost (see the
    /// field docs). Chainable after `with_volatility` by backends with a lazy
    /// per-body preserved save.
    #[inline]
    #[cfg(any(
        sf_backend_arm64,
        sf_backend_armv7a,
        sf_backend_thumbm,
        sf_backend_x64,
        test
    ))]
    pub(crate) const fn with_preserved_lane_save_overhead(mut self, overhead: u8) -> Self {
        self.preserved_lane_save_overhead = overhead;
        self
    }

    #[inline]
    pub(crate) const fn is_32bit_gp_target(self) -> bool {
        self.gp_unit_bytes == 4
    }

    /// GP dynamic budget exposed to the middle-end and normal native
    /// allocation. The remaining tail, if any, is reserved for lowering-only
    /// scratch borrowing.
    #[inline]
    pub(crate) const fn allocatable_gp_dynamic_budget(self) -> u8 {
        self.gp_volatile_dynamic
            .saturating_add(self.gp_preserved_dynamic)
    }

    #[inline]
    pub(crate) const fn allocatable_fp_dynamic_budget(self) -> u8 {
        self.fp_volatile_dynamic
            .saturating_add(self.fp_preserved_dynamic)
    }

    // ── Register layout helpers ──────────────────────────────────────────
    //
    // Layout: [fixed(4) | gp_dynamic | fp_dynamic]
    //
    // These derive the bank boundaries from the unified dynamic budgets so no
    // call-site needs to recompute them manually.

    /// Number of fixed MachineIR registers (ctx, fp, mem0_base, mem0_size).
    pub(crate) const FIXED: u16 = 4;

    /// First FP MachineReg ID (= first reg after all GP regs).
    #[inline]
    pub(crate) const fn first_fp_reg(self) -> u16 {
        Self::FIXED + self.gp_dynamic_budget as u16
    }

    /// Total MachineReg count across all dynamic banks.
    #[inline]
    pub(crate) const fn total_reg_count(self) -> u16 {
        self.first_fp_reg() + self.fp_dynamic_budget as u16
    }
}

#[inline]
const fn min_u8(lhs: u8, rhs: u8) -> u8 {
    if lhs < rhs {
        lhs
    } else {
        rhs
    }
}

#[inline]
#[cfg(any(sf_backend_riscv32, sf_backend_riscv64, test))]
const fn default_gp_internal_scratch(gp_unit_bytes: u8, gp_dynamic_budget: u8) -> u8 {
    let preferred = if gp_unit_bytes == 4 { 2 } else { 1 };
    let max_reserve = gp_dynamic_budget.saturating_sub(1);
    if preferred > max_reserve {
        max_reserve
    } else {
        preferred
    }
}

#[inline]
#[cfg(any(sf_backend_riscv32, sf_backend_riscv64, test))]
const fn default_gp_arg_lanes(gp_volatile_dynamic: u8) -> u8 {
    min_u8(gp_volatile_dynamic, 4)
}

#[inline]
#[cfg(any(sf_backend_riscv32, sf_backend_riscv64, test))]
const fn default_fp_arg_lanes(fp_volatile_dynamic: u8) -> u8 {
    min_u8(fp_volatile_dynamic, 4)
}

#[cfg(test)]
mod tests {
    use super::BackendConfig;

    #[test]
    fn backend_config_keeps_explicit_gp_unit_bytes() {
        let config = BackendConfig::new(4, 3, 7, 8);
        assert_eq!(config.gp_unit_bytes, 4);
        assert_eq!(config.call_scratch_slots, 8);
        assert_eq!(config.gp_internal_scratch, 2);
        assert_eq!(config.gp_volatile_dynamic, 1);
        assert_eq!(config.gp_preserved_dynamic, 0);
    }

    #[test]
    fn backend_config_detects_32bit_gp_targets() {
        assert!(BackendConfig::new(4, 3, 7, 8).is_32bit_gp_target());
        assert!(!BackendConfig::new(8, 3, 7, 3).is_32bit_gp_target());
    }

    #[test]
    fn backend_config_allows_explicit_call_scratch_slots() {
        let config = BackendConfig::new(8, 3, 7, 9);
        assert_eq!(config.call_scratch_slots, 9);
    }

    #[test]
    fn backend_config_publishes_dynamic_volatility_layout() {
        let config = BackendConfig::with_volatility(8, 6, 4, 1, 5, 3, 4, 2, true, 3);
        assert_eq!(config.gp_dynamic_budget, 11);
        assert_eq!(config.fp_dynamic_budget, 8);
        assert_eq!(config.allocatable_gp_dynamic_budget(), 10);
        assert_eq!(config.allocatable_fp_dynamic_budget(), 8);
        assert_eq!(config.gp_arg_lanes, 4);
        assert_eq!(config.fp_arg_lanes, 2);
        assert!(config.scalar_return_lanes);
        assert!(!config.gp32_defs_zero_extend);
    }

    #[test]
    fn backend_config_gp32_zero_extending_defs_is_opt_in() {
        let config = BackendConfig::with_volatility(8, 6, 4, 1, 5, 3, 4, 2, true, 3)
            .with_gp32_zero_extending_defs();
        assert!(config.gp32_defs_zero_extend);
    }
}
