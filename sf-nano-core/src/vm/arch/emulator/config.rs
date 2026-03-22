use crate::vm::{backend::BackendConfig, arch::ReferenceBackendMode};

#[inline]
pub(crate) const fn compile_backend_config(mode: ReferenceBackendMode) -> BackendConfig {
    match mode {
        // Emulator profiles are reference-target policies, not aliases for a
        // concrete ISA backend. Their numbers may happen to match a native
        // backend during bring-up, but they are owned here so later 32-bit
        // ports can diverge without changing emulator semantics implicitly.
        ReferenceBackendMode::Disabled | ReferenceBackendMode::Emu64 => {
            BackendConfig::new_with_gp_unit_bytes(3, 4, 7, 6, 8)
        }
        ReferenceBackendMode::Emu32 => BackendConfig::new_with_gp_unit_bytes(4, 4, 8, 5, 4),
    }
}
