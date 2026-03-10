use core::sync::atomic::{AtomicU8, Ordering::Relaxed};
use crate::vm::plan::{CompileConfig, FAST_COMPILE_CONFIG};

/// Requested backend policy for future fast-code compilation.
///
/// `Auto` prefers the native backend, then fusion, then base depending on
/// which backends are compiled in and available at runtime.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendMode {
    Auto = 0,
    Base = 1,
    Fusion = 2,
    Native = 3,
}

impl BackendMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Base => "base",
            Self::Fusion => "fusion",
            Self::Native => "native",
        }
    }

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

/// Concrete backend selected after feature-based availability resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Base,
    Fusion,
    Native,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Base => "base",
            Self::Fusion => "fusion",
            Self::Native => "native",
        }
    }

    pub fn compile_config(self) -> CompileConfig {
        match self {
            Self::Base | Self::Fusion | Self::Native => FAST_COMPILE_CONFIG,
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
        3 => BackendMode::Native,
        _ => BackendMode::Auto,
    }
}

/// Resolve the current backend policy against compiled feature availability.
///
/// `Auto` returns the highest-priority compiled backend: native, then fusion,
/// then base. This does not account for runtime failures such as native arena
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
                return Ok(BackendKind::Native);
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
        BackendMode::Native => {
            #[cfg(feature = "micro-jit")]
            {
                Ok(BackendKind::Native)
            }
            #[cfg(not(feature = "micro-jit"))]
            {
                Err("native backend not compiled in (requires micro-jit feature)")
            }
        }
    }
}

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
            BackendKind::Native
        } else if cfg!(feature = "fusion") {
            BackendKind::Fusion
        } else {
            BackendKind::Base
        };
        assert_eq!(resolve_backend_mode(BackendMode::Auto), Ok(expected));
    }

    #[cfg(not(feature = "micro-jit"))]
    #[test]
    fn test_native_backend_rejected_when_not_compiled() {
        assert_eq!(
            resolve_backend_mode(BackendMode::Native),
            Err("native backend not compiled in (requires micro-jit feature)"),
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
