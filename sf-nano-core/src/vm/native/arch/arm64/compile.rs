//! ARM64 code packaging.
//!
//! This owns only executable allocation and entry selection for ARM64. Native
//! semantics remain in placed `NativeIR` and the shared bring-up entry.

use alloc::vec::Vec;

use crate::{
    error::WasmError,
    vm::native::{
        code::{DirectCallPatch, NativeCode},
        code_buf::CodeBuffer,
        helper_meta::HelperMetadataArena,
        ir::NativeProgram,
        resolve::ResolvedNativeEntry,
    },
};

use super::lower::lower_arm64;

const MIN_CODE_BUFFER_CAPACITY: usize = 4096;

pub fn compile_program(
    program: &NativeProgram,
    resolved: &[ResolvedNativeEntry],
) -> Result<NativeCode, WasmError> {
    let _ = resolved;

    let lowered = lower_arm64(program)?;
    let mut executable =
        CodeBuffer::with_capacity(lowered.text.len().max(MIN_CODE_BUFFER_CAPACITY))
            .map_err(|err| WasmError::internal(err.into()))?;

    executable.begin_write();
    executable.emit_bytes(&lowered.text);
    executable.finish_write(0, lowered.text.len());

    let entry = Some(unsafe { executable.fn_ptr::<crate::vm::native::entry::NativeEntry>(0) });

    Ok(NativeCode::from_parts(
        entry,
        lowered.text,
        Some(executable),
        HelperMetadataArena::new(),
        Vec::<DirectCallPatch>::new(),
        Some(program.clone()),
    ))
}
