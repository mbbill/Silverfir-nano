use crate::vm::{backend::BackendConfig, arch::ReferenceBackendMode};

#[inline]
pub(crate) fn compile_backend_config(mode: ReferenceBackendMode) -> BackendConfig {
    debug_assert!(
        mode.is_enabled(),
        "compile_backend_config called with Disabled mode"
    );
    match mode {
        ReferenceBackendMode::Emu64 => BackendConfig::new_with_gp_unit_bytes(3, 4, 7, 6, 8),
        ReferenceBackendMode::Emu32 => BackendConfig::new_with_gp_unit_bytes(4, 5, 8, 5, 4),
        ReferenceBackendMode::Disabled => BackendConfig::new_with_gp_unit_bytes(3, 4, 7, 6, 8),
    }
}
