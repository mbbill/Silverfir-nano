use core::sync::atomic::{AtomicU8, Ordering};

use crate::vm::backend::BackendConfig;

pub(crate) mod common;
pub(crate) mod emulator;

#[cfg(target_arch = "aarch64")]
pub(crate) mod arm64;
#[cfg(target_arch = "arm")]
pub(crate) mod armv7a;
#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(debug_assertions)]
#[allow(dead_code)]
mod example;

/// Macro for cfg-gating items that require the emulator/reference backend.
///
/// The emulator is available on all targets and build profiles so that
/// `--emu64` / `--emu32` always work, including in release builds.
macro_rules! cfg_has_reference {
    ($($item:item)*) => {
        $($item)*
    };
}

/// Process-global reference backend target mode.
///
/// `Disabled` means "do not force the emulator backend on a host that has a
/// real native backend". When the emulator is the only available backend on a
/// target, `Disabled` still falls back to the default `Emu64` profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ReferenceBackendMode {
    Disabled = 0,
    Emu64 = 1,
    Emu32 = 2,
}

impl ReferenceBackendMode {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Emu64 => "emu64",
            Self::Emu32 => "emu32",
        }
    }

    #[inline]
    pub const fn is_enabled(self) -> bool {
        !matches!(self, Self::Disabled)
    }
}

/// Active native backend implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NativeBackend {
    #[cfg(target_arch = "aarch64")]
    Arm64,
    #[cfg(target_arch = "arm")]
    Armv7a,
    #[cfg(target_arch = "x86_64")]
    X86_64,
    Reference,
}

#[inline]
pub(crate) fn compile_backend_config(backend: NativeBackend) -> BackendConfig {
    match backend {
        // Each backend returns an explicit budget preset. Physical register
        // mapping and ABI constraints stay in the backend-specific ABI/layout
        // code; this function selects policy, not hardware facts.
        #[cfg(target_arch = "aarch64")]
        NativeBackend::Arm64 => arm64::abi::compile_backend_config(),
        #[cfg(target_arch = "arm")]
        NativeBackend::Armv7a => armv7a::abi::compile_backend_config(),
        #[cfg(target_arch = "x86_64")]
        NativeBackend::X86_64 => x86_64::abi::compile_backend_config(),
        NativeBackend::Reference => {
            emulator::config::compile_backend_config(effective_reference_backend_mode())
        }
    }
}

impl NativeBackend {
    #[inline]
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            #[cfg(target_arch = "aarch64")]
            Self::Arm64 => "arm64",
            #[cfg(target_arch = "arm")]
            Self::Armv7a => "armv7a",
            #[cfg(target_arch = "x86_64")]
            Self::X86_64 => "x86_64",
            Self::Reference => "emulator",
        }
    }
}

cfg_has_reference! {
    static REFERENCE_BACKEND_MODE: AtomicU8 =
        AtomicU8::new(ReferenceBackendMode::Disabled as u8);
}

#[inline]
fn selected_reference_backend_mode() -> ReferenceBackendMode {
    match REFERENCE_BACKEND_MODE.load(Ordering::Relaxed) {
        x if x == ReferenceBackendMode::Disabled as u8 => ReferenceBackendMode::Disabled,
        x if x == ReferenceBackendMode::Emu64 as u8 => ReferenceBackendMode::Emu64,
        x if x == ReferenceBackendMode::Emu32 as u8 => ReferenceBackendMode::Emu32,
        _ => ReferenceBackendMode::Disabled,
    }
}

#[inline]
fn effective_reference_backend_mode() -> ReferenceBackendMode {
    match selected_reference_backend_mode() {
        ReferenceBackendMode::Disabled => ReferenceBackendMode::Emu64,
        mode => mode,
    }
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

pub(crate) fn active_native_backend() -> Result<NativeBackend, &'static str> {
    if selected_reference_backend_mode().is_enabled() {
        return Ok(NativeBackend::Reference);
    }

    if let Some(backend) = host_native_backend() {
        return Ok(backend);
    }

    Ok(NativeBackend::Reference)
}

#[inline]
pub(crate) fn active_backend_config() -> Result<BackendConfig, &'static str> {
    active_native_backend().map(compile_backend_config)
}

#[inline]
pub(crate) fn backend_display_name(backend: NativeBackend) -> &'static str {
    match backend {
        NativeBackend::Reference => effective_reference_backend_mode().as_str(),
        _ => backend.as_str(),
    }
}

#[inline]
pub(crate) fn active_native_backend_name() -> Result<&'static str, &'static str> {
    active_native_backend().map(backend_display_name)
}

pub(crate) fn set_reference_backend_mode(mode: ReferenceBackendMode) -> Result<(), &'static str> {
    REFERENCE_BACKEND_MODE.store(mode as u8, Ordering::Relaxed);
    Ok(())
}

pub(crate) fn set_reference_backend(enabled: bool) -> Result<(), &'static str> {
    set_reference_backend_mode(if enabled {
        ReferenceBackendMode::Emu64
    } else {
        ReferenceBackendMode::Disabled
    })
}

#[cfg(test)]
pub(crate) fn backend_mode_test_lock() -> &'static std::sync::Mutex<()> {
    use std::sync::{Mutex, OnceLock};

    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[cfg(test)]
mod tests {
    use super::{
        backend_display_name, backend_mode_test_lock, compile_backend_config,
        set_reference_backend_mode, NativeBackend, ReferenceBackendMode,
    };

    #[test]
    fn reference_backend_mode_bool_alias_defaults_to_emu64_budget() {
        let _lock = backend_mode_test_lock()
            .lock()
            .expect("backend mode test lock");
        set_reference_backend_mode(ReferenceBackendMode::Disabled).expect("reset reference mode");
        let config = compile_backend_config(NativeBackend::Reference);
        assert_eq!(config.gp_unit_bytes, 8);
        assert_eq!(config.gp_transient_budget, 4);
    }

    #[test]
    fn reference_backend_mode_emu32_uses_its_own_32bit_budget() {
        let _lock = backend_mode_test_lock()
            .lock()
            .expect("backend mode test lock");
        set_reference_backend_mode(ReferenceBackendMode::Emu32).expect("enable emu32");
        let config = compile_backend_config(NativeBackend::Reference);
        assert_eq!(config.gp_unit_bytes, 4);
        assert_eq!(config.gp_local_cache_budget, 4);
        assert_eq!(config.gp_transient_budget, 5);
        assert_eq!(config.fp_local_cache_budget, 8);
        assert_eq!(config.fp_transient_budget, 5);
        assert_eq!(backend_display_name(NativeBackend::Reference), "emu32");
        set_reference_backend_mode(ReferenceBackendMode::Disabled).expect("reset reference mode");
    }
}
