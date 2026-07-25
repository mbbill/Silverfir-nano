//! RV32 backend.
//!
//! RV32 shares RISC-V register identities and instruction encoders with RV64,
//! but uses the 32-bit MachineIR path: pointer-sized calls and local-call
//! metadata are 32-bit, while Wasm i64 values are lowered as lo/hi GP pairs.

pub(super) mod abi;
pub(crate) mod backend;
pub(crate) mod compile;
mod preserved;
mod template;

use crate::{
    error::WasmError,
    vm::{
        runtime::context::NativeContext,
        value_encoding::{as_f32, as_f64, from_f32, from_f64, from_i64},
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

#[inline]
fn join_u64(lo: u32, hi: u32) -> u64 {
    u64::from(lo) | (u64::from(hi) << 32)
}

pub(crate) extern "C" fn rv32_i64_div_s(lhs_lo: u32, lhs_hi: u32, rhs_lo: u32, rhs_hi: u32) -> u64 {
    let lhs = join_u64(lhs_lo, lhs_hi) as i64;
    let rhs = join_u64(rhs_lo, rhs_hi) as i64;
    lhs.wrapping_div(rhs) as u64
}

pub(crate) extern "C" fn rv32_i64_div_u(lhs_lo: u32, lhs_hi: u32, rhs_lo: u32, rhs_hi: u32) -> u64 {
    join_u64(lhs_lo, lhs_hi) / join_u64(rhs_lo, rhs_hi)
}

pub(crate) extern "C" fn rv32_i64_rem_s(lhs_lo: u32, lhs_hi: u32, rhs_lo: u32, rhs_hi: u32) -> u64 {
    let lhs = join_u64(lhs_lo, lhs_hi) as i64;
    let rhs = join_u64(rhs_lo, rhs_hi) as i64;
    lhs.wrapping_rem(rhs) as u64
}

pub(crate) extern "C" fn rv32_i64_rem_u(lhs_lo: u32, lhs_hi: u32, rhs_lo: u32, rhs_hi: u32) -> u64 {
    join_u64(lhs_lo, lhs_hi) % join_u64(rhs_lo, rhs_hi)
}

pub(crate) extern "C" fn rv32_i64s_to_f32_bits(lo: u32, hi: u32) -> u32 {
    ((join_u64(lo, hi) as i64) as f32).to_bits()
}

pub(crate) extern "C" fn rv32_i64u_to_f32_bits(lo: u32, hi: u32) -> u32 {
    (join_u64(lo, hi) as f32).to_bits()
}

pub(crate) extern "C" fn rv32_i64s_to_f64_bits(lo: u32, hi: u32) -> u64 {
    ((join_u64(lo, hi) as i64) as f64).to_bits()
}

pub(crate) extern "C" fn rv32_i64u_to_f64_bits(lo: u32, hi: u32) -> u64 {
    (join_u64(lo, hi) as f64).to_bits()
}

pub(crate) unsafe extern "C" fn rv32_float_to_i64_pair(
    ctx: *mut NativeContext,
    bits_lo: u32,
    bits_hi: u32,
    op_code: u32,
    out: *mut u64,
) -> u32 {
    let bits = join_u64(bits_lo, bits_hi);
    let result = match op_code {
        4 => trunc_f32_to_i64_s(bits_lo),
        5 => trunc_f32_to_i64_u(bits_lo),
        6 => trunc_f64_to_i64_s(bits),
        7 => trunc_f64_to_i64_u(bits),
        12 => Ok(trunc_sat_f32_to_i64_s(bits_lo)),
        13 => Ok(trunc_sat_f32_to_i64_u(bits_lo)),
        14 => Ok(trunc_sat_f64_to_i64_s(bits)),
        15 => Ok(trunc_sat_f64_to_i64_u(bits)),
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

pub(crate) extern "C" fn rv32_f32_ceil_bits(bits: u32) -> u32 {
    unsafe { ceilf(as_f32(u64::from(bits))).to_bits() }
}

pub(crate) extern "C" fn rv32_f32_floor_bits(bits: u32) -> u32 {
    unsafe { floorf(as_f32(u64::from(bits))).to_bits() }
}

pub(crate) extern "C" fn rv32_f32_trunc_bits(bits: u32) -> u32 {
    unsafe { truncf(as_f32(u64::from(bits))).to_bits() }
}

pub(crate) extern "C" fn rv32_f32_nearest_bits(bits: u32) -> u32 {
    wasm_f32_nearest_bits(bits) as u32
}

pub(crate) extern "C" fn rv32_f64_ceil_bits(bits: u64) -> u64 {
    unsafe { ceil(as_f64(bits)).to_bits() }
}

pub(crate) extern "C" fn rv32_f64_floor_bits(bits: u64) -> u64 {
    unsafe { floor(as_f64(bits)).to_bits() }
}

pub(crate) extern "C" fn rv32_f64_trunc_bits(bits: u64) -> u64 {
    unsafe { trunc(as_f64(bits)).to_bits() }
}

pub(crate) extern "C" fn rv32_f64_nearest_bits(bits: u64) -> u64 {
    wasm_f64_nearest_bits(bits)
}

fn wasm_f32_nearest_bits(bits: u32) -> u64 {
    let value = as_f32(u64::from(bits));
    if !value.is_finite() {
        return u64::from(bits);
    }
    let floor = unsafe { floorf(value) };
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
    let floor = unsafe { floor(value) };
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

fn trunc_f32_to_i64_s(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(u64::from(bits)) as f64;
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer"));
    }
    if value.is_infinite() || value <= -9223372036854777856.0 || value >= 9223372036854775808.0 {
        return Err(WasmError::trap("integer overflow"));
    }
    Ok(from_i64(value as i64))
}

fn trunc_f32_to_i64_u(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(u64::from(bits)) as f64;
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
    let value = as_f32(u64::from(bits)) as f64;
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
    let value = as_f32(u64::from(bits)) as f64;
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
