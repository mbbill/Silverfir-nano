//! ARM64 native entry metadata.
//!
//! This file should stay limited to ARM64 entry/patch representation, not
//! frontend semantics.

use crate::error::WasmError;
use crate::vm::native::ir::NativeBlockId;
#[cfg(any(debug_assertions, test))]
use crate::vm::native::{arch::reference, context::NativeContext, ir::NativeProgram};

/// One unresolved ARM64 block-entry patch site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arm64EntryPatch {
    pub code_offset: u32,
    pub target: NativeBlockId,
}

/// Shared Rust target for the bring-up ARM64 entry trampoline.
///
/// The generated ARM64 stub tail-branches here with the native entry ABI
/// intact. This keeps the ISA layer real while the per-op ARM64 emitter is
/// still being brought up.
#[cfg(any(debug_assertions, test))]
pub(super) unsafe extern "C" fn shared_native_entry(
    ctx: *mut NativeContext,
    fp: *mut u64,
    _l0: u64,
    _l1: u64,
    _l2: u64,
    _t0: u64,
    _t1: u64,
    _t2: u64,
    _t3: u64,
) {
    let Some(ctx) = (unsafe { ctx.as_mut() }) else {
        return;
    };
    let Some(program) = current_program(ctx as *mut NativeContext) else {
        return;
    };
    let program = unsafe { &*program };

    if let Err(error) = reference::execute_program(ctx, program, fp) {
        ctx.error = Some(error);
    }
}

#[cfg(any(debug_assertions, test))]
fn current_program(ctx: *mut NativeContext) -> Option<*const NativeProgram> {
    let Some(ctx_ref) = (unsafe { ctx.as_mut() }) else {
        return None;
    };
    let Some(code) = (unsafe { ctx_ref.current_code.as_ref() }) else {
        ctx_ref.error = Some(WasmError::internal(
            "native entry called without current code".into(),
        ));
        return None;
    };
    let Some(program) = code.program() else {
        ctx_ref.error = Some(WasmError::internal(
            "native code is missing finalized program".into(),
        ));
        return None;
    };
    Some(program as *const NativeProgram)
}
