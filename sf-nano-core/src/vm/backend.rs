//! Backend selection.
//!
//! This module carries only backend identity and runtime backend selection.
//! Backend-specific planning configuration belongs to the backend
//! implementation, not to shared VM code.
use core::sync::atomic::{AtomicU8, Ordering};

/// High-level execution backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Native,
}

/// Planning-time backend configuration.
///
/// This carries only the flexible register budget that the backend chooses to
/// spend above the fixed machine ABI roles (`ctx`, `fp`, and the pinned mem0
/// view regs).
///
/// Different layers interpret this budget differently:
/// - `middle/` treats it as one unified dynamic bank per register class
/// - native lowering maps that bank onto concrete machine registers
/// - the frontend frame planner reserves `call_scratch_slots` in the native
///   frame prefix for call-link and helper scratch state
/// - backend-only temporaries must be modeled explicitly through scratch pools
///   or lowering helpers rather than by ad hoc reach-through into these
///   dynamic banks
///
/// It is *not* the place to describe fixed machine roles or runtime stack
/// state.
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
    pub gp_dynamic_budget: u8,
    pub fp_dynamic_budget: u8,
    pub call_scratch_slots: u16,
}

impl BackendConfig {
    #[inline]
    pub(crate) const fn new(
        gp_dynamic_budget: u8,
        fp_dynamic_budget: u8,
        gp_unit_bytes: u8,
        call_scratch_slots: u16,
    ) -> Self {
        Self {
            gp_unit_bytes,
            gp_dynamic_budget,
            fp_dynamic_budget,
            call_scratch_slots,
        }
    }

    #[inline]
    pub(crate) const fn is_32bit_gp_target(self) -> bool {
        self.gp_unit_bytes == 4
    }

    /// GP dynamic lanes kept out of normal value/cache placement so machine
    /// lowering still has a small internal scratch tail when a helper needs an
    /// extra temporary beyond the semantic live set.
    #[inline]
    pub(crate) const fn gp_internal_scratch_reserve(self) -> u8 {
        let preferred = if self.is_32bit_gp_target() { 2 } else { 1 };
        let max_reserve = self.gp_dynamic_budget.saturating_sub(1);
        if preferred > max_reserve {
            max_reserve
        } else {
            preferred
        }
    }

    /// GP dynamic budget exposed to the middle-end and normal native
    /// allocation. The remaining tail, if any, is reserved for lowering-only
    /// scratch borrowing.
    #[inline]
    pub(crate) const fn allocatable_gp_dynamic_budget(self) -> u8 {
        self.gp_dynamic_budget
            .saturating_sub(self.gp_internal_scratch_reserve())
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

#[cfg(test)]
mod tests {
    use super::BackendConfig;

    #[test]
    fn backend_config_keeps_explicit_gp_unit_bytes() {
        let config = BackendConfig::new(3, 7, 4, 8);
        assert_eq!(config.gp_unit_bytes, 4);
        assert_eq!(config.call_scratch_slots, 8);
    }

    #[test]
    fn backend_config_detects_32bit_gp_targets() {
        assert!(BackendConfig::new(3, 7, 4, 8).is_32bit_gp_target());
        assert!(!BackendConfig::new(3, 7, 8, 3).is_32bit_gp_target());
    }

    #[test]
    fn backend_config_allows_explicit_call_scratch_slots() {
        let config = BackendConfig::new(3, 7, 8, 9);
        assert_eq!(config.call_scratch_slots, 9);
    }
}

impl BackendKind {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
        }
    }
}

/// Runtime backend selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendMode {
    Auto,
    Native,
}

impl BackendMode {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Native => "native",
        }
    }

    #[inline]
    pub fn parse_str(name: &str) -> Option<Self> {
        match name {
            "auto" => Some(Self::Auto),
            "native" | "jit" => Some(Self::Native),
            _ => None,
        }
    }
}

static ACTIVE_BACKEND_MODE: AtomicU8 = AtomicU8::new(BackendMode::Native as u8);

pub fn set_backend_mode(mode: BackendMode) {
    ACTIVE_BACKEND_MODE.store(mode as u8, Ordering::Relaxed);
}

pub(crate) fn active_backend_mode() -> BackendMode {
    match ACTIVE_BACKEND_MODE.load(Ordering::Relaxed) {
        x if x == BackendMode::Native as u8 => BackendMode::Native,
        _ => BackendMode::Auto,
    }
}

pub fn backend_mode() -> BackendMode {
    active_backend_mode()
}

pub(crate) fn resolve_backend_mode(mode: BackendMode) -> Result<BackendKind, &'static str> {
    match mode {
        BackendMode::Native | BackendMode::Auto => {
            #[cfg(sf_jit)]
            {
                Ok(BackendKind::Native)
            }
            #[cfg(not(sf_jit))]
            {
                Err("native backend not compiled in")
            }
        }
    }
}

pub fn active_backend() -> Result<BackendKind, &'static str> {
    resolve_backend_mode(active_backend_mode())
}
