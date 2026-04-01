use crate::error::WasmError;

use super::{common::{set_ctx_error, NativeCallStatus}, context::NativeContext};

/// Trap entry point called from generated code. Sets `ctx.error` to the
/// appropriate trap error and returns nonzero.
///
/// # Safety
///
/// `ctx` must point to a valid `NativeContext`.
pub(crate) unsafe extern "C" fn raise_trap(ctx: *mut NativeContext, kind: u64) -> u32 {
    let Some(ctx) = (unsafe { ctx.as_mut() }) else {
        return NativeCallStatus::Error as u32;
    };
    let error = match kind {
        0 => WasmError::trap("unreachable executed".into()),
        1 => WasmError::trap("out of bounds memory access".into()),
        2 => WasmError::trap("out of bounds table access".into()),
        3 => WasmError::trap("invalid function reference".into()),
        4 => WasmError::trap("indirect call type mismatch".into()),
        5 => WasmError::trap("integer divide by zero".into()),
        6 => WasmError::trap("integer overflow".into()),
        7 => WasmError::trap("invalid conversion to integer".into()),
        8 => WasmError::exhaustion("stack overflow".into()),
        _ => WasmError::trap("native call failed".into()),
    };
    set_ctx_error(ctx, error);
    NativeCallStatus::Error as u32
}
