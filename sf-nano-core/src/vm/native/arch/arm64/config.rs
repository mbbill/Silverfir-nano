use crate::vm::backend::BackendConfig;

#[inline]
pub const fn compile_backend_config() -> BackendConfig {
    // 3 GP cached locals (X9-X11), 4 GP transients (X12-X15),
    // 2 FP local cache (D5-D6), 2 FP transients (D3-D4)
    BackendConfig::new(3, 4, 2, 2)
}
