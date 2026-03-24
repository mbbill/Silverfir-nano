//! Example backend budget policy.

use crate::vm::backend::BackendConfig;

#[inline]
pub(crate) const fn compile_backend_config() -> BackendConfig {
    BackendConfig::new_with_gp_unit_bytes(4, 4, 4, 2, 8)
}
