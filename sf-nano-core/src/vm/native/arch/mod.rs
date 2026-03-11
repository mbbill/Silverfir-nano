//! ISA-specific native backend implementations.
//!
//! This layer lowers target-independent native IR into encoded machine code.
//! It must not own Wasm/LIR semantics or backend-wide optimization policy.

use crate::{
    error::WasmError,
    vm::native::{code::NativeCode, ir::NativeProgram, resolve::ResolvedNativeEntry},
};

pub mod arm64;
#[cfg(any(debug_assertions, test))]
pub mod reference;

pub fn compile_native(
    program: &NativeProgram,
    resolved: &[ResolvedNativeEntry],
) -> Result<NativeCode, WasmError> {
    #[cfg(any(debug_assertions, test))]
    if option_env!("SF_NATIVE_FORCE_REFERENCE").is_some() {
        return reference::compile_program(program, resolved);
    }

    #[cfg(target_arch = "aarch64")]
    {
        return arm64::compile_program(program, resolved);
    }

    #[cfg(all(not(target_arch = "aarch64"), any(debug_assertions, test)))]
    {
        return reference::compile_program(program, resolved);
    }

    #[cfg(all(not(target_arch = "aarch64"), not(any(debug_assertions, test))))]
    {
        Err(WasmError::internal(
            "native backend is unavailable on this architecture in release builds".into(),
        ))
    }
}
