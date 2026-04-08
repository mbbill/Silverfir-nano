//! x86_64 runtime helpers called from generated code.

use crate::error::WasmError;
use crate::vm::raw_value::{as_f32, as_f64, from_i32, from_i64};
use crate::vm::runtime::context::NativeContext;

#[cfg(sf_call_trace)]
use crate::vm::debug::function_trace;

// ── raise_trap ───────────────────────────────────────────────────────────────

pub(crate) unsafe extern "C" fn x86_64_raise_trap(ctx: *mut NativeContext, kind: u64) -> u32 {
    let Some(ctx) = (unsafe { ctx.as_mut() }) else {
        return 1;
    };
    let error = match kind {
        0 => WasmError::trap("unreachable executed".into()),
        1 => WasmError::trap("out of bounds memory access".into()),
        2 => WasmError::trap("out of bounds table access".into()),
        3 => WasmError::trap("invalid function reference".into()),
        4 => WasmError::trap("indirect call type mismatch".into()),
        5 => WasmError::trap("integer divide by zero".into()),
        6 => WasmError::trap("integer overflow".into()),
        7 => WasmError::exhaustion("stack overflow".into()),
        _ => WasmError::trap("native helper failed".into()),
    };
    #[cfg(sf_call_trace)]
    function_trace::native_trap_current(ctx, &error);
    ctx.error = Some(error);
    1
}

pub(crate) unsafe extern "C" fn x86_64_raise_unsupported(
    ctx: *mut NativeContext,
    func_id: u64,
) -> u32 {
    let Some(ctx) = (unsafe { ctx.as_mut() }) else {
        return 1;
    };
    let error = WasmError::invalid(alloc::format!(
        "x86_64 backend has not finalized machine function {} yet",
        func_id
    ));
    #[cfg(sf_call_trace)]
    function_trace::native_trap_current(ctx, &error);
    ctx.error = Some(error);
    1
}

// ── Trapping truncation ──────────────────────────────────────────────────────

/// Return type for trapping truncation helpers.
/// On System V AMD64, a 2-field repr(C) struct of u64s is returned in RAX and RDX.
#[repr(C)]
pub(crate) struct TruncResult {
    pub status: u64,
    pub value: u64,
}

/// Trapping truncation helper called from generated code.
/// Returns status in RAX (0 = ok) and result in RDX via struct return.
pub(crate) unsafe extern "C" fn x86_64_trapping_trunc(
    ctx: *mut NativeContext,
    src_bits: u64,
    op_code: u64,
) -> TruncResult {
    let result = match op_code {
        0 => trunc_f32_to_i32_s(src_bits as u32),
        1 => trunc_f32_to_i32_u(src_bits as u32),
        2 => trunc_f64_to_i32_s(src_bits),
        3 => trunc_f64_to_i32_u(src_bits),
        4 => trunc_f32_to_i64_s(src_bits as u32),
        5 => trunc_f32_to_i64_u(src_bits as u32),
        6 => trunc_f64_to_i64_s(src_bits),
        7 => trunc_f64_to_i64_u(src_bits),
        _ => Err(WasmError::trap("invalid trunc op".into())),
    };
    match result {
        Ok(value) => TruncResult { status: 0, value },
        Err(err) => {
            if let Some(ctx) = unsafe { ctx.as_mut() } {
                ctx.error = Some(err);
            }
            TruncResult {
                status: 1,
                value: 0,
            }
        }
    }
}

/// Saturating truncation helper called from generated code.
/// Returns result in RAX (no error possible for sat).

#[cfg(sf_os_windows)]
pub(crate) unsafe extern "C" fn x86_64_trapping_trunc_win(
    ctx: *mut NativeContext,
    src_bits: u64,
    op_code: u64,
    out_value: *mut u64,
) -> u32 {
    let result = match op_code {
        0 => trunc_f32_to_i32_s(src_bits as u32),
        1 => trunc_f32_to_i32_u(src_bits as u32),
        2 => trunc_f64_to_i32_s(src_bits),
        3 => trunc_f64_to_i32_u(src_bits),
        4 => trunc_f32_to_i64_s(src_bits as u32),
        5 => trunc_f32_to_i64_u(src_bits as u32),
        6 => trunc_f64_to_i64_s(src_bits),
        7 => trunc_f64_to_i64_u(src_bits),
        _ => Err(WasmError::trap("invalid trunc op".into())),
    };
    match result {
        Ok(value) => {
            unsafe { *out_value = value };
            0
        }
        Err(err) => {
            if let Some(ctx) = unsafe { ctx.as_mut() } {
                ctx.error = Some(err);
            }
            1
        }
    }
}

pub(crate) unsafe extern "C" fn x86_64_saturating_trunc(src_bits: u64, op_code: u64) -> u64 {
    match op_code {
        8 => trunc_sat_f32_to_i32_s(src_bits as u32),
        9 => trunc_sat_f32_to_i32_u(src_bits as u32),
        10 => trunc_sat_f64_to_i32_s(src_bits),
        11 => trunc_sat_f64_to_i32_u(src_bits),
        12 => trunc_sat_f32_to_i64_s(src_bits as u32),
        13 => trunc_sat_f32_to_i64_u(src_bits as u32),
        14 => trunc_sat_f64_to_i64_s(src_bits),
        15 => trunc_sat_f64_to_i64_u(src_bits),
        _ => 0,
    }
}

// Trapping truncation implementations (matching Wasm spec)

fn trunc_f32_to_i32_s(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(bits as u64);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value >= 2147483648.0_f32 || value < -2147483648.0_f32 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(from_i32(value as i32))
}

fn trunc_f32_to_i32_u(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(bits as u64);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value >= 4294967296.0_f32 || value <= -1.0_f32 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(u64::from(value as u32))
}

fn trunc_f64_to_i32_s(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value >= 2147483648.0 || value <= -2147483649.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(from_i32(value as i32))
}

fn trunc_f64_to_i32_u(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value >= 4294967296.0 || value <= -1.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(u64::from(value as u32))
}

fn trunc_f32_to_i64_s(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(bits as u64);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value >= 9223372036854775808.0_f32 || value < -9223372036854775808.0_f32 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(from_i64(value as i64))
}

fn trunc_f32_to_i64_u(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(bits as u64);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value >= 18446744073709551616.0_f32 || value <= -1.0_f32 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(value as u64)
}

fn trunc_f64_to_i64_s(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value >= 9223372036854775808.0 || value < -9223372036854775808.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(from_i64(value as i64))
}

fn trunc_f64_to_i64_u(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value >= 18446744073709551616.0 || value <= -1.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(value as u64)
}

// Saturating truncation implementations

fn trunc_sat_f32_to_i32_s(bits: u32) -> u64 {
    let value = as_f32(bits as u64);
    if value.is_nan() {
        return 0;
    }
    if value >= 2147483648.0_f32 {
        return from_i32(i32::MAX);
    }
    if value < -2147483648.0_f32 {
        return from_i32(i32::MIN);
    }
    from_i32(value as i32)
}

fn trunc_sat_f32_to_i32_u(bits: u32) -> u64 {
    let value = as_f32(bits as u64);
    if value.is_nan() || value <= -1.0_f32 {
        return 0;
    }
    if value >= 4294967296.0_f32 {
        return u64::from(u32::MAX);
    }
    u64::from(value as u32)
}

fn trunc_sat_f64_to_i32_s(bits: u64) -> u64 {
    let value = as_f64(bits);
    if value.is_nan() {
        return 0;
    }
    if value >= 2147483648.0 {
        return from_i32(i32::MAX);
    }
    if value <= -2147483649.0 {
        return from_i32(i32::MIN);
    }
    from_i32(value as i32)
}

fn trunc_sat_f64_to_i32_u(bits: u64) -> u64 {
    let value = as_f64(bits);
    if value.is_nan() || value <= -1.0 {
        return 0;
    }
    if value >= 4294967296.0 {
        return u64::from(u32::MAX);
    }
    u64::from(value as u32)
}

fn trunc_sat_f32_to_i64_s(bits: u32) -> u64 {
    let value = as_f32(bits as u64);
    if value.is_nan() {
        return 0;
    }
    if value >= 9223372036854775808.0_f32 {
        return from_i64(i64::MAX);
    }
    if value < -9223372036854775808.0_f32 {
        return from_i64(i64::MIN);
    }
    from_i64(value as i64)
}

fn trunc_sat_f32_to_i64_u(bits: u32) -> u64 {
    let value = as_f32(bits as u64);
    if value.is_nan() || value <= -1.0_f32 {
        return 0;
    }
    if value >= 18446744073709551616.0_f32 {
        return u64::MAX;
    }
    value as u64
}

fn trunc_sat_f64_to_i64_s(bits: u64) -> u64 {
    let value = as_f64(bits);
    if value.is_nan() {
        return 0;
    }
    if value >= 9223372036854775808.0 {
        return from_i64(i64::MAX);
    }
    if value < -9223372036854775808.0 {
        return from_i64(i64::MIN);
    }
    from_i64(value as i64)
}

fn trunc_sat_f64_to_i64_u(bits: u64) -> u64 {
    let value = as_f64(bits);
    if value.is_nan() || value <= -1.0 {
        return 0;
    }
    if value >= 18446744073709551616.0 {
        return u64::MAX;
    }
    value as u64
}
