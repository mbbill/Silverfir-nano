use core::sync::atomic::{AtomicBool, Ordering};

use crate::vm::backend::BackendConfig;

pub mod arm64;
pub mod emulator;

/// Active native backend implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeBackend {
    Arm64,
    #[cfg(debug_assertions)]
    Reference,
}

#[inline]
pub const fn compile_backend_config(backend: NativeBackend) -> BackendConfig {
    match backend {
        // Each backend returns an explicit budget preset. Physical register
        // mapping and ABI constraints stay in the backend-specific ABI/layout
        // code; this function selects policy, not hardware facts.
        NativeBackend::Arm64 => arm64::config::compile_backend_config(),
        #[cfg(debug_assertions)]
        NativeBackend::Reference => emulator::config::compile_backend_config(),
    }
}

impl NativeBackend {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Arm64 => "arm64",
            #[cfg(debug_assertions)]
            Self::Reference => "emulator",
        }
    }
}

#[cfg(debug_assertions)]
static FORCE_REFERENCE: AtomicBool = AtomicBool::new(false);

#[inline]
fn host_native_backend() -> Option<NativeBackend> {
    #[cfg(target_arch = "aarch64")]
    {
        return Some(NativeBackend::Arm64);
    }

    None
}

pub fn active_native_backend() -> Result<NativeBackend, &'static str> {
    #[cfg(debug_assertions)]
    {
        if FORCE_REFERENCE.load(Ordering::Relaxed) {
            return Ok(NativeBackend::Reference);
        }
    }

    if let Some(backend) = host_native_backend() {
        return Ok(backend);
    }

    #[cfg(debug_assertions)]
    {
        return Ok(NativeBackend::Reference);
    }

    #[allow(unreachable_code)]
    Err("native backend is unavailable on this target")
}

#[inline]
pub fn active_backend_config() -> Result<BackendConfig, &'static str> {
    active_native_backend().map(compile_backend_config)
}

pub fn set_reference_backend(enabled: bool) -> Result<(), &'static str> {
    #[cfg(debug_assertions)]
    {
        FORCE_REFERENCE.store(enabled, Ordering::Relaxed);
        return Ok(());
    }

    if enabled {
        Err("reference backend is only available in debug builds")
    } else {
        Ok(())
    }
}
