use core::mem::align_of;

use crate::error::WasmError;
use crate::value_type::ValueType;
use crate::vm::value::Value;

use super::context::NativeContext;

/// Shared status code used by runtime boundary entrypoints.
///
/// Wire values are pinned: every arch post-call site treats nonzero as trap,
/// so `Error == 1` must not change. `Thrown == 2` is only consumed by
/// EH-capable call sites; 2-way callers collapse it into `Error`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum NativeCallStatus {
    Ok = 0,
    Error = 1,
    Thrown = 2,
}

#[inline]
pub(crate) fn set_ctx_error(ctx: &mut NativeContext, error: WasmError) {
    #[cfg(sf_call_trace)]
    crate::vm::debug::function_trace::native_trap_current(ctx, &error);
    ctx.error = Some(error);
}

#[inline]
pub(crate) fn internal_error(message: &'static str) -> WasmError {
    WasmError::internal(message)
}

#[inline]
pub(crate) fn trap_error(message: &'static str) -> WasmError {
    WasmError::trap(message)
}

#[inline]
pub(crate) fn value_matches_value_type(value: &Value, ty: ValueType) -> bool {
    match (value, ty) {
        (Value::I32(_), ValueType::I32) => true,
        (Value::I64(_), ValueType::I64) => true,
        (Value::F32(_), ValueType::F32) => true,
        (Value::F64(_), ValueType::F64) => true,
        #[cfg(sf_has_simd)]
        (Value::V128(_), ValueType::V128) => true,
        (Value::Ref(_, _), ValueType::Ref(_)) => true,
        _ => false,
    }
}

#[inline]
pub(crate) unsafe fn decode_metadata<'a, T>(metadata: *const u8) -> Result<&'a T, WasmError> {
    if metadata.is_null() {
        return Err(internal_error(
            "native call entry received null metadata pointer",
        ));
    }
    if (metadata as usize) % align_of::<T>() != 0 {
        return Err(internal_error(
            "native call entry received misaligned metadata pointer",
        ));
    }
    Ok(unsafe { &*metadata.cast::<T>() })
}

/// Run a frame-backed runtime call. The body may signal a thrown wasm
/// exception by returning `Ok(NativeCallStatus::Thrown)` after installing
/// `PendingEscape::Throw { .. }` on the context. Any returned
/// `Err(WasmError::HostThrow { .. })` is treated as a programmer bug (the
/// entry must consume and convert it before returning) and collapsed into
/// `Error` with a debug_assert.
#[inline]
pub(crate) fn run_frame_call_with_status<T>(
    ctx: *mut NativeContext,
    frame: *mut u64,
    metadata: *const u8,
    body: impl FnOnce(&mut NativeContext, *mut u64, &T) -> Result<NativeCallStatus, WasmError>,
) -> u32 {
    let Some(ctx) = (unsafe { ctx.as_mut() }) else {
        return NativeCallStatus::Error as u32;
    };

    ctx.error = None;

    let result = if frame.is_null() {
        Err(internal_error(
            "native call entry received null frame pointer",
        ))
    } else {
        unsafe { decode_metadata::<T>(metadata) }.and_then(|meta| body(ctx, frame, meta))
    };

    match result {
        Ok(status) => status as u32,
        Err(WasmError::HostThrow { .. }) => {
            debug_assert!(
                false,
                "HostThrow must be consumed by the runtime-call entry before reaching run_frame_call_with_status"
            );
            set_ctx_error(
                ctx,
                WasmError::trap("host throw leaked past runtime-call entry"),
            );
            NativeCallStatus::Error as u32
        }
        Err(error) => {
            set_ctx_error(ctx, error);
            NativeCallStatus::Error as u32
        }
    }
}
