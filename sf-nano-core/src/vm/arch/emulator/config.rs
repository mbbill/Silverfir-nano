use crate::vm::backend::BackendConfig;

#[cfg(sf_backend_emu64)]
#[inline]
pub(crate) const fn compile_backend_config() -> BackendConfig {
    BackendConfig::new(8, 12, 13, 3)
}

#[cfg(sf_backend_emu32)]
#[inline]
pub(crate) const fn compile_backend_config() -> BackendConfig {
    BackendConfig::new(4, 9, 13, 8)
}
