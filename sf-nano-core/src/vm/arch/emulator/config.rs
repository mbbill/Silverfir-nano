use crate::vm::{arch::ReferenceBackendMode, backend::BackendConfig};

#[inline]
pub(crate) fn compile_backend_config(mode: ReferenceBackendMode) -> BackendConfig {
    debug_assert!(
        mode.is_enabled(),
        "compile_backend_config called with Disabled mode"
    );
    match mode {
        ReferenceBackendMode::Emu64 => BackendConfig::new(3, 4, 7, 6, 8, 3),
        ReferenceBackendMode::Emu32 => BackendConfig::new(4, 5, 8, 5, 4, 8),
        ReferenceBackendMode::Disabled => BackendConfig::new(3, 4, 7, 6, 8, 3),
    }
}
