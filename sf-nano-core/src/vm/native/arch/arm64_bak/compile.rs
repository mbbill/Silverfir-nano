//! ARM64 code packaging.
//!
//! This owns only executable allocation and entry selection for ARM64. Native
//! semantics remain in placed `NativeIR` and the shared bring-up entry.

use alloc::vec::Vec;

use crate::{
    error::WasmError,
    vm::native::{
        code::{DirectCallPatch, NativeCode, NativeCodeRegion},
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
    compile_program_with_mode(program, resolved, Arm64CompileMode::Auto, None)
}

pub fn compile_program_with_mode(
    program: &NativeProgram,
    resolved: &[ResolvedNativeEntry],
    mode: Arm64CompileMode,
    callee_may_change_views: Option<&[bool]>,
) -> Result<NativeCode, WasmError> {
    let lowered = lower_program(program, resolved, mode, callee_may_change_views)?;
    let mut executable =
        CodeBuffer::with_capacity(lowered.text.len().max(MIN_CODE_BUFFER_CAPACITY))
            .map_err(|err| WasmError::internal(err.into()))?;
    executable.begin_write();
    let code = install_lowered(program, lowered, &mut executable)?;
    if let Some(region) = code.region() {
        executable.finish_write(region.start as usize, region.len as usize);
    } else {
        executable.finish_write(0, 0);
    }
    Ok(NativeCode {
        executable: Some(executable),
        ..code
    })
}

fn lower_program(
    program: &NativeProgram,
    resolved: &[ResolvedNativeEntry],
    mode: Arm64CompileMode,
    callee_may_change_views: Option<&[bool]>,
) -> Result<super::lower::Arm64LoweredFunction, WasmError> {
    let _ = resolved;

    Ok(match mode {
        Arm64CompileMode::Direct => lower_arm64(program, callee_may_change_views)?,
        Arm64CompileMode::Auto => {
            if supports_direct_lowering_base(program) {
                lower_arm64(program, callee_may_change_views)?
            } else {
                lower_shared_entry()
            }
        }
        Arm64CompileMode::SharedEntry => lower_shared_entry(),
    })
}

fn install_lowered(
    program: &NativeProgram,
    lowered: super::lower::Arm64LoweredFunction,
    executable: &mut CodeBuffer,
) -> Result<NativeCode, WasmError> {
    let code_len = lowered.text.len() as u32;
    let code_start = executable.emit_bytes(&lowered.text);
    for patch in &lowered.local_literal_patches {
        let target =
            unsafe { executable.base_ptr().add(code_start + patch.target_offset as usize) as usize as u64 };
        executable.patch_u64(code_start + patch.literal_offset as usize, target);
    }

    let entry = Some(unsafe {
        executable.fn_ptr::<crate::vm::native::entry::NativeEntry>(
            code_start + lowered.entry_offset as usize,
        )
    });
    let internal_entry = lowered.internal_entry_offset.map(|offset| unsafe {
        executable.fn_ptr::<crate::vm::native::entry::NativeEntry>(code_start + offset as usize)
    });

    Ok(NativeCode::from_parts(
        entry,
        internal_entry,
        lowered.text,
        Some(NativeCodeRegion {
            start: code_start as u32,
            len: code_len,
        }),
        lowered.debug_regions,
        None,
        HelperMetadataArena::new(),
        lowered.direct_call_patches,
        Some(program.clone()),
    ))
}
