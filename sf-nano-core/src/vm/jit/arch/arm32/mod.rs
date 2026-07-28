pub(super) mod abi;
pub(crate) mod backend;
pub(crate) mod compile;
mod control;
// Encoder selection: `enc` resolves to either the A32 encoder (`enc_a32.rs`)
// or the Thumb-2 encoder (`enc_t2.rs`) based on `sf_arm32_isa_thumb`. Set by
// build.rs for sf_backend_thumbm, and also for sf_backend_armv7a when the
// `thumb2-test` cargo feature is on (see that feature's comment).
#[cfg(not(sf_arm32_isa_thumb))]
#[path = "enc_a32.rs"]
mod enc;
#[cfg(sf_arm32_isa_thumb)]
#[path = "enc_t2.rs"]
mod enc;
mod inst;
mod operands;
mod preserved;
mod reg;
mod select;
mod template;

/// Mark a code address as a Thumb-mode function entry for ARM/Thumb
/// interworking. On Thumb-2 builds (`sf_arm32_isa_thumb`), any address that
/// will be used as a branch target — function entries, direct-call patches,
/// indirect branch destinations — must have LSB=1 set so that `BX` / `BLX`
/// (or Rust-side function-pointer calls via `blx reg`) switch the CPU to
/// Thumb mode on entry. Without this, the processor would try to decode
/// Thumb-2 bytes as A32 and fault immediately.
///
/// On A32 builds the bit is meaningless (A32 instructions must be word-
/// aligned, so LSB=0 always) and this helper is a no-op.
#[inline]
pub(crate) const fn thumb_interworking_bit(addr: usize) -> usize {
    #[cfg(sf_arm32_isa_thumb)]
    {
        addr | 1
    }
    #[cfg(not(sf_arm32_isa_thumb))]
    {
        addr
    }
}

use crate::{
    error::WasmError,
    vm::{
        jit::runtime::{context::NativeContext, trap::raise_trap},
        jit::value_encoding::{as_f32, as_f64, from_f32, from_f64, from_i32, from_i64},
    },
};

unsafe extern "C" {
    fn ceilf(x: f32) -> f32;
    fn floorf(x: f32) -> f32;
    fn truncf(x: f32) -> f32;
    fn ceil(x: f64) -> f64;
    fn floor(x: f64) -> f64;
    fn trunc(x: f64) -> f64;
}

// `eval` is now provided by the shared `arch::common::eval::eval`. The
// `dispatch_eval` central switch in `arch/mod.rs` routes arm32 callers
// directly to the common entry point — no per-arch wrapper is required.

/// AAPCS-friendly shim around the canonical `runtime::trap::raise_trap`
/// helper. The canonical entry takes `kind: u64`, which AAPCS would force
/// into register pair `r2/r3` (skipping `r1`) — annoying for the JIT trap
/// stub to set up. This shim accepts `kind: u32` so we can pass it cleanly
/// in `r1`, then widens it before delegating.
///
/// # Safety
/// `ctx` must point to a valid `NativeContext` for the duration of the call.
pub(crate) unsafe extern "C" fn arm32_raise_trap(ctx: *mut NativeContext, kind: u32) -> u32 {
    unsafe { raise_trap(ctx, u64::from(kind)) }
}

#[inline]
fn join_u64(lo: u32, hi: u32) -> u64 {
    u64::from(lo) | (u64::from(hi) << 32)
}

/// Rotate a 64-bit value left by the low 6 bits of `count`.
///
/// Retained for the register-count Rotl/Rotr fallback in the arm32 backend.
/// The constant-count path and the Shl/ShrS/ShrU register-count paths are
/// fully inlined; only these two rotations keep the helper fallback.
pub(crate) extern "C" fn arm32_i64_rotl(lo: u32, hi: u32, count: u32) -> u64 {
    join_u64(lo, hi).rotate_left(count & 63)
}

/// Rotate a 64-bit value right by the low 6 bits of `count`. See
/// [`arm32_i64_rotl`] for why this helper is retained.
pub(crate) extern "C" fn arm32_i64_rotr(lo: u32, hi: u32, count: u32) -> u64 {
    join_u64(lo, hi).rotate_right(count & 63)
}

/// Signed 64-bit division over split lo/hi halves.
pub(crate) extern "C" fn arm32_i64_div_s(
    lhs_lo: u32,
    lhs_hi: u32,
    rhs_lo: u32,
    rhs_hi: u32,
) -> u64 {
    let lhs = join_u64(lhs_lo, lhs_hi) as i64;
    let rhs = join_u64(rhs_lo, rhs_hi) as i64;
    lhs.wrapping_div(rhs) as u64
}

/// Unsigned 64-bit division over split lo/hi halves.
pub(crate) extern "C" fn arm32_i64_div_u(
    lhs_lo: u32,
    lhs_hi: u32,
    rhs_lo: u32,
    rhs_hi: u32,
) -> u64 {
    join_u64(lhs_lo, lhs_hi) / join_u64(rhs_lo, rhs_hi)
}

/// Signed 64-bit remainder over split lo/hi halves.
pub(crate) extern "C" fn arm32_i64_rem_s(
    lhs_lo: u32,
    lhs_hi: u32,
    rhs_lo: u32,
    rhs_hi: u32,
) -> u64 {
    let lhs = join_u64(lhs_lo, lhs_hi) as i64;
    let rhs = join_u64(rhs_lo, rhs_hi) as i64;
    lhs.wrapping_rem(rhs) as u64
}

/// Unsigned 64-bit remainder over split lo/hi halves.
pub(crate) extern "C" fn arm32_i64_rem_u(
    lhs_lo: u32,
    lhs_hi: u32,
    rhs_lo: u32,
    rhs_hi: u32,
) -> u64 {
    join_u64(lhs_lo, lhs_hi) % join_u64(rhs_lo, rhs_hi)
}

// ─── i64 ↔ float conversion helpers ─────────────────────────────────────────

/// Convert signed i64 (passed as lo/hi halves) to f64.
pub(crate) extern "C" fn arm32_i64s_to_f64(lo: u32, hi: u32) -> f64 {
    let val = (lo as u64) | ((hi as u64) << 32);
    (val as i64) as f64
}

/// Convert unsigned i64 (passed as lo/hi halves) to f64.
pub(crate) extern "C" fn arm32_i64u_to_f64(lo: u32, hi: u32) -> f64 {
    let val = (lo as u64) | ((hi as u64) << 32);
    val as f64
}

/// Convert signed i64 (passed as lo/hi halves) to f32.
pub(crate) extern "C" fn arm32_i64s_to_f32(lo: u32, hi: u32) -> f32 {
    let val = (lo as u64) | ((hi as u64) << 32);
    (val as i64) as f32
}

/// Convert unsigned i64 (passed as lo/hi halves) to f32.
pub(crate) extern "C" fn arm32_i64u_to_f32(lo: u32, hi: u32) -> f32 {
    let val = (lo as u64) | ((hi as u64) << 32);
    val as f32
}

/// Saturating float-to-i64 conversion helper for legalized pair results.
pub(crate) extern "C" fn arm32_saturating_trunc(src_bits: u64, op_code: u32) -> u64 {
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

/// Trapping float-to-i64 conversion helper for legalized pair results.
pub(crate) unsafe extern "C" fn arm32_trapping_trunc(
    src_bits: u64,
    op_code: u32,
    ctx: *mut NativeContext,
    out: *mut u64,
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
        _ => Err(WasmError::trap("invalid trunc op")),
    };
    match result {
        Ok(value) => {
            if let Some(out) = unsafe { out.as_mut() } {
                *out = value;
            }
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

pub(crate) extern "C" fn arm32_f32_ceil(val: f32) -> f32 {
    ceil_f32(val)
}

pub(crate) extern "C" fn arm32_f32_floor(val: f32) -> f32 {
    floor_f32(val)
}

pub(crate) extern "C" fn arm32_f32_trunc(val: f32) -> f32 {
    trunc_f32(val)
}

pub(crate) extern "C" fn arm32_f32_nearest_bits(bits: u32) -> u32 {
    wasm_f32_nearest_bits(bits) as u32
}

pub(crate) extern "C" fn arm32_f64_ceil(val: f64) -> f64 {
    ceil_f64(val)
}

pub(crate) extern "C" fn arm32_f64_floor(val: f64) -> f64 {
    floor_f64(val)
}

pub(crate) extern "C" fn arm32_f64_trunc(val: f64) -> f64 {
    trunc_f64(val)
}

pub(crate) extern "C" fn arm32_f64_nearest_bits(bits: u64) -> u64 {
    wasm_f64_nearest_bits(bits)
}

#[inline]
fn ceil_f32(value: f32) -> f32 {
    unsafe { ceilf(value) }
}

#[inline]
fn floor_f32(value: f32) -> f32 {
    unsafe { floorf(value) }
}

#[inline]
fn trunc_f32(value: f32) -> f32 {
    unsafe { truncf(value) }
}

#[inline]
fn ceil_f64(value: f64) -> f64 {
    unsafe { ceil(value) }
}

#[inline]
fn floor_f64(value: f64) -> f64 {
    unsafe { floor(value) }
}

#[inline]
fn trunc_f64(value: f64) -> f64 {
    unsafe { trunc(value) }
}

fn wasm_f32_nearest_bits(bits: u32) -> u64 {
    let value = as_f32(bits as u64);
    if !value.is_finite() {
        return u64::from(bits);
    }
    let floor = floor_f32(value);
    let diff = value - floor;
    let rounded = if diff < 0.5 {
        floor
    } else if diff > 0.5 {
        floor + 1.0
    } else if (floor as i64) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    };
    from_f32(rounded)
}

fn wasm_f64_nearest_bits(bits: u64) -> u64 {
    let value = as_f64(bits);
    if !value.is_finite() {
        return bits;
    }
    let floor = floor_f64(value);
    let diff = value - floor;
    let rounded = if diff < 0.5 {
        floor
    } else if diff > 0.5 {
        floor + 1.0
    } else if (floor as i64) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    };
    from_f64(rounded)
}

fn trunc_f32_to_i32_s(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(bits as u64);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer"));
    }
    if value >= 2147483648.0_f32 || value < -2147483648.0_f32 {
        return Err(WasmError::trap("integer overflow"));
    }
    Ok(from_i32(value as i32))
}

fn trunc_f32_to_i32_u(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(bits as u64);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer"));
    }
    if value >= 4294967296.0_f32 || value <= -1.0_f32 {
        return Err(WasmError::trap("integer overflow"));
    }
    Ok(u64::from(value as u32))
}

fn trunc_f64_to_i32_s(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer"));
    }
    if value >= 2147483648.0 || value <= -2147483649.0 {
        return Err(WasmError::trap("integer overflow"));
    }
    Ok(from_i32(value as i32))
}

fn trunc_f64_to_i32_u(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer"));
    }
    if value >= 4294967296.0 || value <= -1.0 {
        return Err(WasmError::trap("integer overflow"));
    }
    Ok(u64::from(value as u32))
}

fn trunc_f32_to_i64_s(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(bits as u64) as f64;
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer"));
    }
    if value.is_infinite() || value <= -9223372036854777856.0 || value >= 9223372036854775808.0 {
        return Err(WasmError::trap("integer overflow"));
    }
    Ok(from_i64(value as i64))
}

fn trunc_f32_to_i64_u(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(bits as u64) as f64;
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer"));
    }
    if value.is_infinite() || value <= -1.0 || value >= 18446744073709551616.0 {
        return Err(WasmError::trap("integer overflow"));
    }
    Ok(value as u64)
}

fn trunc_f64_to_i64_s(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer"));
    }
    if value.is_infinite() || value <= -9223372036854777856.0 || value >= 9223372036854775808.0 {
        return Err(WasmError::trap("integer overflow"));
    }
    Ok(from_i64(value as i64))
}

fn trunc_f64_to_i64_u(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer"));
    }
    if value.is_infinite() || value <= -1.0 || value >= 18446744073709551616.0 {
        return Err(WasmError::trap("integer overflow"));
    }
    Ok(value as u64)
}

fn trunc_sat_f32_to_i64_s(bits: u32) -> u64 {
    let value = as_f32(bits as u64) as f64;
    if value.is_nan() {
        0
    } else if value <= i64::MIN as f64 {
        from_i64(i64::MIN)
    } else if value >= i64::MAX as f64 {
        from_i64(i64::MAX)
    } else {
        from_i64(value as i64)
    }
}

fn trunc_sat_f32_to_i64_u(bits: u32) -> u64 {
    let value = as_f32(bits as u64) as f64;
    if value.is_nan() || value <= 0.0 {
        0
    } else if value >= u64::MAX as f64 {
        u64::MAX
    } else {
        value as u64
    }
}

fn trunc_sat_f64_to_i64_s(bits: u64) -> u64 {
    let value = as_f64(bits);
    if value.is_nan() {
        0
    } else if value <= i64::MIN as f64 {
        from_i64(i64::MIN)
    } else if value >= i64::MAX as f64 {
        from_i64(i64::MAX)
    } else {
        from_i64(value as i64)
    }
}

fn trunc_sat_f64_to_i64_u(bits: u64) -> u64 {
    let value = as_f64(bits);
    if value.is_nan() || value <= 0.0 {
        0
    } else if value >= u64::MAX as f64 {
        u64::MAX
    } else {
        value as u64
    }
}

fn trunc_sat_f32_to_i32_s(bits: u32) -> u64 {
    let value = as_f32(bits as u64);
    if value.is_nan() {
        0
    } else if value <= i32::MIN as f32 {
        from_i32(i32::MIN)
    } else if value >= i32::MAX as f32 {
        from_i32(i32::MAX)
    } else {
        from_i32(value as i32)
    }
}

fn trunc_sat_f32_to_i32_u(bits: u32) -> u64 {
    let value = as_f32(bits as u64);
    if value.is_nan() || value <= 0.0 {
        0
    } else if value >= u32::MAX as f32 {
        u64::from(u32::MAX)
    } else {
        u64::from(value as u32)
    }
}

fn trunc_sat_f64_to_i32_s(bits: u64) -> u64 {
    let value = as_f64(bits);
    if value.is_nan() {
        0
    } else if value <= i32::MIN as f64 {
        from_i32(i32::MIN)
    } else if value >= i32::MAX as f64 {
        from_i32(i32::MAX)
    } else {
        from_i32(value as i32)
    }
}

fn trunc_sat_f64_to_i32_u(bits: u64) -> u64 {
    let value = as_f64(bits);
    if value.is_nan() || value <= 0.0 {
        0
    } else if value >= u32::MAX as f64 {
        u64::from(u32::MAX)
    } else {
        u64::from(value as u32)
    }
}
