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

use super::lower::{lower_arm64, supports_direct_lowering_base};
use super::lower::lower_shared_entry;

const MIN_CODE_BUFFER_CAPACITY: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm64CompileMode {
    Auto,
    Direct,
    SharedEntry,
}

pub fn compile_program(
    program: &NativeProgram,
    resolved: &[ResolvedNativeEntry],
) -> Result<NativeCode, WasmError> {
    compile_program_with_mode(program, resolved, Arm64CompileMode::Auto)
}

pub fn compile_program_with_mode(
    program: &NativeProgram,
    resolved: &[ResolvedNativeEntry],
    mode: Arm64CompileMode,
) -> Result<NativeCode, WasmError> {
    let _ = resolved;

    let lowered = match mode {
        Arm64CompileMode::Direct => lower_arm64(program)?,
        Arm64CompileMode::Auto => {
            if supports_direct_lowering_base(program) {
                lower_arm64(program)?
            } else {
                lower_shared_entry()
            }
        }
        Arm64CompileMode::SharedEntry => lower_shared_entry(),
    };
    let mut executable =
        CodeBuffer::with_capacity(lowered.text.len().max(MIN_CODE_BUFFER_CAPACITY))
            .map_err(|err| WasmError::internal(err.into()))?;

    executable.begin_write();
    executable.emit_bytes(&lowered.text);
    for patch in &lowered.local_literal_patches {
        let target = unsafe { executable.base_ptr().add(patch.target_offset as usize) as usize as u64 };
        executable.patch_u64(patch.literal_offset as usize, target);
    }
    executable.finish_write(0, lowered.text.len());

    let entry = Some(unsafe {
        executable.fn_ptr::<crate::vm::native::entry::NativeEntry>(lowered.entry_offset as usize)
    });
    let internal_entry = lowered.internal_entry_offset.map(|offset| unsafe {
        executable.fn_ptr::<crate::vm::native::entry::NativeEntry>(offset as usize)
    });

    Ok(NativeCode::from_parts(
        entry,
        internal_entry,
        lowered.text,
        Some(executable),
        HelperMetadataArena::new(),
        lowered.direct_call_patches,
        Some(program.clone()),
    ))
}
