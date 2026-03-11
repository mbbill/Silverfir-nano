//! Runtime entry glue for the reference native backend.
//!
//! The reference backend must accept the same finalized `NativeProgram` input
//! shape as a real ISA backend. This entry therefore owns only ABI bridging
//! and delegates execution semantics to `machine.rs`.

use crate::error::WasmError;

use crate::vm::native::{context::NativeContext, ir::NativeProgram};

use super::machine;

/// Shared runtime entry for the reference native backend during bring-up.
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

    if let Err(error) = machine::execute_program(ctx, program, fp) {
        ctx.error = Some(error);
    }
}

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
