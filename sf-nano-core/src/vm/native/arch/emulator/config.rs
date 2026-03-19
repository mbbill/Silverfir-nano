use crate::vm::{backend::BackendConfig, native::arch::ReferenceBackendMode};

#[inline]
pub const fn compile_backend_config(mode: ReferenceBackendMode) -> BackendConfig {
    match mode {
        ReferenceBackendMode::Disabled | ReferenceBackendMode::Emu64 => {
            super::super::budget_presets::emulator64_backend_config()
        }
        ReferenceBackendMode::Emu32 => super::super::budget_presets::armv7a_backend_config(),
    }
}
