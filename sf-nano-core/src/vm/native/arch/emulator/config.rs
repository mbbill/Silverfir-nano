use crate::vm::backend::BackendConfig;

#[inline]
pub const fn compile_backend_config() -> BackendConfig {
    BackendConfig::new(3, 4, 2, 6)
}
