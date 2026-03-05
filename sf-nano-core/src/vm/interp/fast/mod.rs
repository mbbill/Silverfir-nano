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

#[cfg(all(feature = "micro-jit", feature = "fusion"))]
compile_error!("Features `micro-jit` and `fusion` are mutually exclusive. Enable only one.");

/// Number of TOS (Top-of-Stack) registers in the fast interpreter.
pub const TOS_REGISTER_COUNT: usize = 4;

/// Runtime override to disable fusion (used by discover-fusion profiling).
#[cfg(feature = "fusion")]
static FUSION_DISABLED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Disable fusion at runtime (e.g., for profiling the raw instruction stream).
#[cfg(feature = "fusion")]
pub fn set_fusion_disabled(disabled: bool) {
    FUSION_DISABLED.store(disabled, core::sync::atomic::Ordering::Relaxed);
}

/// Check whether instruction fusion is currently disabled.
#[cfg(feature = "fusion")]
pub fn is_fusion_disabled() -> bool {
    FUSION_DISABLED.load(core::sync::atomic::Ordering::Relaxed)
}

#[cfg(not(feature = "fusion"))]
pub fn set_fusion_disabled(_disabled: bool) {}

#[cfg(not(feature = "fusion"))]
pub fn is_fusion_disabled() -> bool {
    true
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

#[cfg(feature = "profile")]
pub mod profiler;
#[cfg(feature = "profile")]
pub mod pattern_trie;
#[cfg(feature = "profile")]
pub mod fusion_discovery;

#[cfg(feature = "micro-jit")]
pub mod jit;
