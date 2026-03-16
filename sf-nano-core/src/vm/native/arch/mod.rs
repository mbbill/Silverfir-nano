use core::sync::atomic::{AtomicBool, Ordering};

use crate::vm::backend::BackendConfig;

#[cfg(target_arch = "aarch64")]
pub mod arm64;
#[cfg(target_arch = "arm")]
pub mod armv7a;
pub mod emulator;
#[cfg(target_arch = "x86_64")]
pub mod x86_64;

/// Macro for cfg-gating items that require the emulator/reference backend.
///
/// The emulator is available in debug builds (any target) and on targets
/// without a native host backend (e.g. ARM32) so that `--emu` works.
macro_rules! cfg_has_reference {
    ($($item:item)*) => {
        $(
            #[cfg(any(
                debug_assertions,
                not(any(target_arch = "aarch64", target_arch = "x86_64"))
            ))]
            $item
        )*
    };
}

/// Active native backend implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeBackend {
    #[cfg(target_arch = "aarch64")]
    Arm64,
    #[cfg(target_arch = "arm")]
    Armv7a,
    #[cfg(target_arch = "x86_64")]
    X86_64,
    #[cfg(any(
        debug_assertions,
        not(any(target_arch = "aarch64", target_arch = "x86_64"))
    ))]
    Reference,
}

#[inline]
pub const fn compile_backend_config(backend: NativeBackend) -> BackendConfig {
    match backend {
        // Each backend returns an explicit budget preset. Physical register
        // mapping and ABI constraints stay in the backend-specific ABI/layout
        // code; this function selects policy, not hardware facts.
        #[cfg(target_arch = "aarch64")]
        NativeBackend::Arm64 => arm64::config::compile_backend_config(),
        #[cfg(target_arch = "arm")]
        NativeBackend::Armv7a => armv7a::config::compile_backend_config(),
        #[cfg(target_arch = "x86_64")]
        NativeBackend::X86_64 => x86_64::config::compile_backend_config(),
        #[cfg(any(
            debug_assertions,
            not(any(target_arch = "aarch64", target_arch = "x86_64"))
        ))]
        NativeBackend::Reference => emulator::config::compile_backend_config(),
    }
}

impl NativeBackend {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            #[cfg(target_arch = "aarch64")]
            Self::Arm64 => "arm64",
            #[cfg(target_arch = "arm")]
            Self::Armv7a => "armv7a",
            #[cfg(target_arch = "x86_64")]
            Self::X86_64 => "x86_64",
            #[cfg(any(
                debug_assertions,
                not(any(target_arch = "aarch64", target_arch = "x86_64"))
            ))]
            Self::Reference => "emulator",
        }
    }
}

cfg_has_reference! {
    static FORCE_REFERENCE: AtomicBool = AtomicBool::new(false);
}

#[inline]
fn host_native_backend() -> Option<NativeBackend> {
    #[cfg(target_arch = "aarch64")]
    {
        return Some(NativeBackend::Arm64);
    }

    #[cfg(target_arch = "arm")]
    {
        return Some(NativeBackend::Armv7a);
    }

    #[cfg(target_arch = "x86_64")]
    {
        return Some(NativeBackend::X86_64);
    }

    #[allow(unreachable_code)]
    None
}

pub fn active_native_backend() -> Result<NativeBackend, &'static str> {
    #[cfg(any(
        debug_assertions,
        not(any(target_arch = "aarch64", target_arch = "x86_64"))
    ))]
    {
        if FORCE_REFERENCE.load(Ordering::Relaxed) {
            return Ok(NativeBackend::Reference);
        }
    }

    if let Some(backend) = host_native_backend() {
        return Ok(backend);
    }

    #[cfg(any(
        debug_assertions,
        not(any(target_arch = "aarch64", target_arch = "x86_64"))
    ))]
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
    #[cfg(any(
        debug_assertions,
        not(any(target_arch = "aarch64", target_arch = "x86_64"))
    ))]
    {
        FORCE_REFERENCE.store(enabled, Ordering::Relaxed);
        return Ok(());
    }

    #[allow(unreachable_code)]
    if enabled {
        Err("reference backend is only available in debug builds")
    } else {
        Ok(())
    }
}
