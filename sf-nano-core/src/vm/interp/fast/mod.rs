//! Fast interpreter — minimal no_std port from sf-core.
//!
//! Layout:
//! - `runtime`: Entry point and stack management.
//! - `instruction`: 32-byte instruction header & arena.
//! - `context`: Hot state + opaque context container.
//! - `fast_code`: FastCode storage and FastCodeCache.
//! - `handlers/`: Handler implementations organized by category.
//! - `builder/`: Modular IR builder components.
//! - `encoding`: Generated instruction encoding/decoding.

use core::sync::atomic::{AtomicU8, Ordering::Relaxed};

/// Number of TOS (Top-of-Stack) registers in the fast interpreter.
pub const TOS_REGISTER_COUNT: usize = 4;

/// Requested backend policy for future fast-code compilation.
///
/// `Auto` prefers JIT, then fusion, then base depending on which backends
/// are compiled in and available at runtime.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendMode {
    Auto = 0,
    Base = 1,
    Fusion = 2,
    Jit = 3,
}

impl BackendMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Base => "base",
            Self::Fusion => "fusion",
            Self::Jit => "jit",
        }
    }
}

/// Concrete backend selected after feature-based availability resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Base,
    Fusion,
    Jit,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Base => "base",
            Self::Fusion => "fusion",
            Self::Jit => "jit",
        }
    }
}

static BACKEND_MODE: AtomicU8 = AtomicU8::new(BackendMode::Auto as u8);

/// Set the backend policy for future fast-code compilation.
pub fn set_backend_mode(mode: BackendMode) {
    BACKEND_MODE.store(mode as u8, Relaxed);
}

/// Get the current backend policy.
pub fn backend_mode() -> BackendMode {
    match BACKEND_MODE.load(Relaxed) {
        1 => BackendMode::Base,
        2 => BackendMode::Fusion,
        3 => BackendMode::Jit,
        _ => BackendMode::Auto,
    }
}

/// Resolve the current backend policy against compiled feature availability.
///
/// `Auto` returns the highest-priority compiled backend: JIT, then fusion,
/// then base. This does not account for runtime failures such as JIT arena
/// allocation; auto mode may still fall back further during compilation.
pub fn active_backend() -> Result<BackendKind, &'static str> {
    resolve_backend_mode(backend_mode())
}

/// Resolve a backend policy against compiled feature availability.
pub fn resolve_backend_mode(mode: BackendMode) -> Result<BackendKind, &'static str> {
    match mode {
        BackendMode::Auto => {
            #[cfg(feature = "micro-jit")]
            {
                return Ok(BackendKind::Jit);
            }
            #[cfg(all(not(feature = "micro-jit"), feature = "fusion"))]
            {
                return Ok(BackendKind::Fusion);
            }
            #[cfg(all(not(feature = "micro-jit"), not(feature = "fusion")))]
            {
                return Ok(BackendKind::Base);
            }
        }
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
        BackendMode::Jit => {
            #[cfg(feature = "micro-jit")]
            {
                Ok(BackendKind::Jit)
            }
            #[cfg(not(feature = "micro-jit"))]
            {
                Err("micro-jit backend not compiled in")
            }
        }
    }
}

pub mod builder;
pub mod context;
pub mod encoding;
pub mod fast_code;
pub mod frame_layout;
pub mod handlers;

/// Generated handler variant lookup tables.
#[allow(dead_code)]
pub mod handler_lookup {
    include!(concat!(env!("OUT_DIR"), "/fast_interp/fast_handler_lookup.rs"));
}

pub mod instruction;
pub mod precompile;
pub mod runtime;

pub mod fusion;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_backend_always_available() {
        assert_eq!(resolve_backend_mode(BackendMode::Base), Ok(BackendKind::Base));
    }

    #[test]
    fn test_auto_backend_prefers_best_compiled_backend() {
        let expected = if cfg!(feature = "micro-jit") {
            BackendKind::Jit
        } else if cfg!(feature = "fusion") {
            BackendKind::Fusion
        } else {
            BackendKind::Base
        };
        assert_eq!(resolve_backend_mode(BackendMode::Auto), Ok(expected));
    }

    #[cfg(not(feature = "micro-jit"))]
    #[test]
    fn test_jit_backend_rejected_when_not_compiled() {
        assert_eq!(
            resolve_backend_mode(BackendMode::Jit),
            Err("micro-jit backend not compiled in"),
        );
    }

    #[cfg(not(feature = "fusion"))]
    #[test]
    fn test_fusion_backend_rejected_when_not_compiled() {
        assert_eq!(
            resolve_backend_mode(BackendMode::Fusion),
            Err("fusion backend not compiled in"),
        );
    }
}
