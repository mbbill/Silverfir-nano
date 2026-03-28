//! Backend selection.
//!
//! This module carries only backend identity and runtime backend selection.
//! Backend-specific planning configuration belongs to the backend
//! implementation, not to shared VM code.
use core::sync::atomic::{AtomicU8, Ordering};

/// High-level execution backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Base,
    Fusion,
    Native,
}

/// Planning-time backend configuration.
///
/// This carries only the flexible register budget that the backend chooses to
/// spend above the fixed machine ABI roles (`ctx`, `fp`, and the pinned mem0
/// view regs).
///
/// Different layers consume different subsets of this budget:
/// - planning/SSA-IR shapes the live transient window from
///   `gp_transient_budget`/`fp_transient_budget`
/// - native lowering maps `gp_local_cache_budget`/`fp_local_cache_budget`
///   onto cache regs
/// - backends may also repurpose cache or transient regs for other temporary
///   work
///   when they can prove the owning values are not live
///
/// It is *not* the place to describe fixed machine roles or runtime stack
/// state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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
    pub gp_local_cache_budget: u8,
    pub gp_transient_budget: u8,
    pub fp_local_cache_budget: u8,
    pub fp_transient_budget: u8,
}

impl BackendConfig {
    #[cfg(any(test, feature = "interp"))]
    #[inline]
    pub(crate) const fn new(
        gp_local_cache_budget: u8,
        gp_transient_budget: u8,
        fp_local_cache_budget: u8,
        fp_transient_budget: u8,
    ) -> Self {
        Self::new_with_gp_unit_bytes(
            gp_local_cache_budget,
            gp_transient_budget,
            fp_local_cache_budget,
            fp_transient_budget,
            core::mem::size_of::<usize>() as u8,
        )
    }

    #[inline]
    pub(crate) const fn new_with_gp_unit_bytes(
        gp_local_cache_budget: u8,
        gp_transient_budget: u8,
        fp_local_cache_budget: u8,
        fp_transient_budget: u8,
        gp_unit_bytes: u8,
    ) -> Self {
        // Minimum GP transient budget: the worst-case simultaneous GP
        // unit pressure comes from `select(i64, i64, i32)` — two i64
        // operands (each 8/gp_unit_bytes units) plus one i32 condition
        // (1 unit).  No wasm instruction takes more than 3 operands.
        debug_assert!(
            gp_transient_budget as usize >= (8 / gp_unit_bytes as usize) * 2 + 1,
            "GP transient budget too small for target GP unit width"
        );
        Self {
            gp_unit_bytes,
            gp_local_cache_budget,
            gp_transient_budget,
            fp_local_cache_budget,
            fp_transient_budget,
        }
    }

    #[inline]
    pub(crate) const fn is_32bit_gp_target(self) -> bool {
        self.gp_unit_bytes == 4
    }

    // ── Register layout helpers ──────────────────────────────────────────
    //
    // Layout: [fixed(4) | gp_cache | gp_trans | fp_trans | fp_cache]
    //
    // These derive partition boundaries from the budget so that no
    // call-site needs to recompute them manually.

    /// Number of fixed MachineIR registers (ctx, fp, mem0_base, mem0_size).
    pub(crate) const FIXED: u16 = 4;

    /// First GP transient MachineReg ID (= first reg after GP cache).
    #[inline]
    pub(crate) const fn first_gp_transient(self) -> u16 {
        Self::FIXED + self.gp_local_cache_budget as u16
    }

    /// First FP MachineReg ID (= first reg after all GP regs).
    #[inline]
    pub(crate) const fn first_fp_reg(self) -> u16 {
        Self::FIXED + self.gp_local_cache_budget as u16 + self.gp_transient_budget as u16
    }

    /// Total MachineReg count across all partitions.
    #[inline]
    pub(crate) const fn total_reg_count(self) -> u16 {
        self.first_fp_reg()
            + self.fp_transient_budget as u16
            + self.fp_local_cache_budget as u16
    }
}

#[cfg(test)]
mod tests {
    use super::BackendConfig;

    #[test]
    fn backend_config_defaults_gp_unit_bytes_to_host_word_size() {
        let config = BackendConfig::new(1, 2, 3, 4);
        assert_eq!(config.gp_unit_bytes as usize, core::mem::size_of::<usize>());
    }

    #[test]
    fn backend_config_keeps_explicit_gp_unit_bytes() {
        let config = BackendConfig::new_with_gp_unit_bytes(1, 2, 3, 4, 4);
        assert_eq!(config.gp_unit_bytes, 4);
    }

    #[test]
    fn backend_config_detects_32bit_gp_targets() {
        assert!(BackendConfig::new_with_gp_unit_bytes(1, 2, 3, 4, 4).is_32bit_gp_target());
        assert!(!BackendConfig::new_with_gp_unit_bytes(1, 2, 3, 4, 8).is_32bit_gp_target());
    }
}

impl BackendKind {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Base => "base",
            Self::Fusion => "fusion",
            Self::Native => "native",
        }
    }
}

/// Runtime backend selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendMode {
    Auto,
    Base,
    Fusion,
    Native,
}

impl BackendMode {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Base => "base",
            Self::Fusion => "fusion",
            Self::Native => "native",
        }
    }

    #[inline]
    pub fn parse_str(name: &str) -> Option<Self> {
        match name {
            "auto" => Some(Self::Auto),
            "base" => Some(Self::Base),
            "fusion" => Some(Self::Fusion),
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
        x if x == BackendMode::Base as u8 => BackendMode::Base,
        x if x == BackendMode::Fusion as u8 => BackendMode::Fusion,
        x if x == BackendMode::Native as u8 => BackendMode::Native,
        _ => BackendMode::Auto,
    }
}

pub fn backend_mode() -> BackendMode {
    active_backend_mode()
}

pub(crate) fn resolve_backend_mode(mode: BackendMode) -> Result<BackendKind, &'static str> {
    match mode {
        BackendMode::Base => {
            #[cfg(feature = "interp")]
            {
                Ok(BackendKind::Base)
            }
            #[cfg(not(feature = "interp"))]
            {
                Err("interpreter backend not compiled in")
            }
        }
        BackendMode::Fusion => {
            #[cfg(all(feature = "interp", feature = "fusion"))]
            {
                Ok(BackendKind::Fusion)
            }
            #[cfg(all(feature = "interp", not(feature = "fusion")))]
            {
                Err("fusion backend not compiled in")
            }
            #[cfg(not(feature = "interp"))]
            {
                Err("interpreter backend not compiled in")
            }
        }
        BackendMode::Native => {
            #[cfg(feature = "micro-jit")]
            {
                Ok(BackendKind::Native)
            }
            #[cfg(not(feature = "micro-jit"))]
            {
                Err("native backend not compiled in")
            }
        }
        BackendMode::Auto => {
            #[cfg(feature = "micro-jit")]
            {
                return Ok(BackendKind::Native);
            }
            #[cfg(all(not(feature = "micro-jit"), feature = "interp"))]
            {
                return Ok(BackendKind::Base);
            }
            #[cfg(not(any(feature = "micro-jit", feature = "interp")))]
            {
                Err("no execution backend compiled in")
            }
        }
    }
}

pub fn active_backend() -> Result<BackendKind, &'static str> {
    resolve_backend_mode(active_backend_mode())
}
