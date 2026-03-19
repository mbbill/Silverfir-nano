use crate::vm::backend::BackendConfig;

#[inline]
pub const fn armv7a_backend_config() -> BackendConfig {
    BackendConfig::new_with_gp_unit_bytes(2, 6, 8, 5, 4)
}

#[inline]
pub const fn emulator64_backend_config() -> BackendConfig {
    BackendConfig::new_with_gp_unit_bytes(3, 4, 7, 6, 8)
}
