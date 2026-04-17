use crate::vm::{arch::ReferenceBackendMode, backend::BackendConfig};

#[inline]
pub(crate) fn compile_backend_config(mode: ReferenceBackendMode) -> BackendConfig {
    debug_assert!(
        mode.is_enabled(),
        "compile_backend_config called with Disabled mode"
    );
    match mode {
        ReferenceBackendMode::Emu64 => BackendConfig::new(12, 13, 8, 3),
        ReferenceBackendMode::Emu32 => BackendConfig::new(9, 13, 4, 8),
        ReferenceBackendMode::Disabled => BackendConfig::new(12, 13, 8, 3),
    }
}
