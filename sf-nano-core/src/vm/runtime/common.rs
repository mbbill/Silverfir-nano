use core::mem::align_of;

use crate::error::WasmError;

use super::context::NativeContext;

/// Shared status code used by runtime boundary entrypoints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum NativeCallStatus {
    Ok = 0,
    Error = 1,
}

#[inline]
pub(crate) fn set_ctx_error(ctx: &mut NativeContext, error: WasmError) {
    #[cfg(feature = "function-trace")]
    crate::vm::debug::function_trace::native_trap_current(ctx, &error);
    ctx.error = Some(error);
}

#[inline]
pub(crate) fn internal_error(message: &str) -> WasmError {
    WasmError::internal(message.into())
}

#[inline]
pub(crate) fn trap_error(message: &str) -> WasmError {
    WasmError::trap(message.into())
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

#[inline]
pub(crate) fn run_frame_call<T>(
    ctx: *mut NativeContext,
    frame: *mut u64,
    metadata: *const u8,
    body: impl FnOnce(&mut NativeContext, *mut u64, &T) -> Result<(), WasmError>,
) -> u32 {
    let Some(ctx) = (unsafe { ctx.as_mut() }) else {
        return NativeCallStatus::Error as u32;
    };

    ctx.error = None;

    let result = if frame.is_null() {
        Err(internal_error("native call entry received null frame pointer"))
    } else {
        unsafe { decode_metadata::<T>(metadata) }.and_then(|meta| body(ctx, frame, meta))
    };

    match result {
        Ok(()) => NativeCallStatus::Ok as u32,
        Err(error) => {
            set_ctx_error(ctx, error);
            NativeCallStatus::Error as u32
        }
    }
}
