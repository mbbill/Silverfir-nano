//! ISA-specific native backend implementations.
//!
//! This layer lowers target-independent native IR into encoded machine code.
//! It must not own Wasm/LIR semantics or backend-wide optimization policy.

use core::sync::atomic::{AtomicBool, Ordering};

use crate::{
    error::WasmError,
    vm::native::{code::NativeCode, ir::NativeProgram, resolve::ResolvedNativeEntry},
};

pub mod arm64;
#[cfg(debug_assertions)]
pub mod reference;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeBackend {
    Arm64,
    Reference,
}

impl NativeBackend {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Arm64 => "arm64",
            Self::Reference => "reference",
        }
    }
}

#[cfg(debug_assertions)]
static FORCE_REFERENCE: AtomicBool = AtomicBool::new(false);

pub fn set_reference_backend(enabled: bool) -> Result<(), &'static str> {
    #[cfg(debug_assertions)]
    {
        FORCE_REFERENCE.store(enabled, Ordering::Relaxed);
        Ok(())
    }

    #[cfg(not(debug_assertions))]
    {
        if enabled {
            Err("reference backend is only available in debug builds")
        } else {
            Ok(())
        }
    }
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

    Err("native backend is unavailable on this target")
}

#[inline]
fn host_native_backend() -> Option<NativeBackend> {
    #[cfg(target_arch = "aarch64")]
    {
        return Some(NativeBackend::Arm64);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        None
    }
}

pub fn compile_native(
    program: &NativeProgram,
    resolved: &[ResolvedNativeEntry],
) -> Result<NativeCode, WasmError> {
    match active_native_backend()
        .map_err(|err| WasmError::invalid(alloc::format!("native backend unavailable: {err}")))?
    {
        NativeBackend::Arm64 => arm64::compile_program(program, resolved),
        NativeBackend::Reference => {
            #[cfg(debug_assertions)]
            {
                reference::compile_program(program, resolved)
            }

            #[cfg(not(debug_assertions))]
            {
                let _ = (program, resolved);
                Err(WasmError::internal(
                    "reference backend is not built in release mode".into(),
                ))
            }
        }
    }
}
