use core::sync::atomic::{AtomicBool, Ordering};

/// Active native backend implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeBackend {
    Arm64,
    #[cfg(debug_assertions)]
    Reference,
}

impl NativeBackend {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Arm64 => "arm64",
            #[cfg(debug_assertions)]
            Self::Reference => "reference",
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
