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
/// Shared shape for backend-specific register budgets.
/// The values themselves must be chosen by the backend implementation.
///
/// Different layers consume different subsets of this budget:
/// - planning/LIR shape around TOS lanes and hot locals
/// - native lowering later consumes temps plus reserved VM roles like ctx/fp
///
/// It is *not* the place to leak runtime stack-state into codegen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendConfig {
    pub ctx_register_count: u8,
    pub fp_register_count: u8,
    pub tmp_register_count: u8,
    pub hot_local_count: u8,
    pub tos_register_count: u8,
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
                #[cfg(all(not(feature = "micro-jit"), feature = "fusion"))]
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

static ACTIVE_BACKEND_MODE: AtomicU8 = AtomicU8::new(BackendMode::Auto as u8);

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
        BackendMode::Base => Ok(BackendKind::Base),
        BackendMode::Fusion => {
            #[cfg(feature = "fusion")]
            {
                Ok(BackendKind::Fusion)
            }
            #[cfg(not(feature = "fusion"))]
            {
                Err("fusion backend not compiled in")
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
            // Keep `auto` on the proven base pipeline until fusion is brought
            // back up on top of the refactored frontend/backend boundary.
            Ok(BackendKind::Base)
        }
    }
}

pub fn active_backend() -> Result<BackendKind, &'static str> {
    resolve_backend_mode(active_backend_mode())
}
