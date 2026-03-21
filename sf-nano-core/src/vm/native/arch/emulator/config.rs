use crate::vm::{backend::BackendConfig, native::arch::EmulatorMode};

#[inline]
pub const fn compile_backend_config(mode: EmulatorMode) -> BackendConfig {
    match mode {
        // Emulator profiles are reference-target policies, not aliases for a
        // concrete ISA backend. Their numbers may happen to match a native
        // backend during bring-up, but they are owned here so later 32-bit
        // ports can diverge without changing emulator semantics implicitly.
        EmulatorMode::Disabled | EmulatorMode::Emu64 => {
            BackendConfig::new_with_gp_unit_bytes(3, 4, 7, 6, 8)
        }
        EmulatorMode::Emu32 => BackendConfig::new_with_gp_unit_bytes(4, 4, 8, 5, 4),
    }
}
