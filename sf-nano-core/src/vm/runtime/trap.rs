use crate::error::WasmError;
use crate::vm::arch::common::helpers::trap_error;
use crate::vm::machine::machine_ir::MachineTrapKind;

use super::{
    common::{set_ctx_error, NativeCallStatus},
    context::NativeContext,
};

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
        0 => trap_error(MachineTrapKind::Unreachable),
        1 => trap_error(MachineTrapKind::MemoryOutOfBounds),
        2 => trap_error(MachineTrapKind::TableOutOfBounds),
        3 => trap_error(MachineTrapKind::InvalidFunctionReference),
        4 => trap_error(MachineTrapKind::IndirectCallTypeMismatch),
        5 => trap_error(MachineTrapKind::IntegerDivideByZero),
        6 => trap_error(MachineTrapKind::IntegerOverflow),
        7 => trap_error(MachineTrapKind::InvalidConversion),
        8 => trap_error(MachineTrapKind::StackOverflow),
        _ => WasmError::trap("native call failed"),
    };
    set_ctx_error(ctx, error);
    NativeCallStatus::Error as u32
}
