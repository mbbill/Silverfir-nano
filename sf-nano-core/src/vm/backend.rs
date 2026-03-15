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
/// - planning/LIR shapes the live transient window from `gp_lane_count`/`fp_lane_count`
/// - native lowering maps `gp_local_cache_count`/`fp_local_cache_count` onto cache regs
/// - backends may also repurpose cache or lane regs for other temporary work
///   when they can prove the owning values are not live
///
/// It is *not* the place to describe fixed machine roles or runtime stack
/// state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendConfig {
    pub gp_local_cache_count: u8,
    pub gp_lane_count: u8,
    pub fp_local_cache_count: u8,
    pub fp_lane_count: u8,
}

impl BackendConfig {
    #[inline]
    pub const fn new(
        gp_local_cache_count: u8,
        gp_lane_count: u8,
        fp_local_cache_count: u8,
        fp_lane_count: u8,
    ) -> Self {
        Self {
            gp_local_cache_count,
            gp_lane_count,
            fp_local_cache_count,
            fp_lane_count,
        }
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

    #[inline]
    pub const fn resolve(self) -> BackendKind {
        match self {
            Self::Auto => {
                #[cfg(feature = "micro-jit")]
                {
                    return BackendKind::Native;
                }
                #[cfg(all(not(feature = "micro-jit"), feature = "interp", feature = "fusion"))]
                {
                    return BackendKind::Fusion;
                }
                BackendKind::Base
            }
            Self::Base => BackendKind::Base,
            Self::Fusion => BackendKind::Fusion,
            Self::Native => BackendKind::Native,
        }
    }
}

static ACTIVE_BACKEND_MODE: AtomicU8 = AtomicU8::new(BackendMode::Native as u8);

pub fn set_backend_mode(mode: BackendMode) {
    ACTIVE_BACKEND_MODE.store(mode as u8, Ordering::Relaxed);
}

pub fn active_backend_mode() -> BackendMode {
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

pub fn resolve_backend_mode(mode: BackendMode) -> Result<BackendKind, &'static str> {
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
            #[cfg(feature = "interp")]
            {
                return Ok(BackendKind::Base);
            }
            Err("no execution backend compiled in")
        }
    }
}

pub fn active_backend() -> Result<BackendKind, &'static str> {
    resolve_backend_mode(active_backend_mode())
}
