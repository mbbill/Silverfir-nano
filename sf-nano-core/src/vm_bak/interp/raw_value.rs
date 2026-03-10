//! Internal raw value system for WebAssembly interpreter.
//!
//! Provides efficient 64-bit aligned storage and zero-cost conversions
//! for maximum interpreter performance. INTERNAL ONLY — not for external APIs.

use crate::value_type::ValueType;
use crate::vm::value::{RefHandle, Value};

/// Raw value type for efficient runtime storage.
/// All WebAssembly values are stored as 64-bit unsigned integers.
/// Types are not tracked at runtime since validation guarantees correctness.
pub type RawValue = u64;

/// Convert i32 to raw value (zero-extended to 64 bits)
#[inline(always)]
pub const fn from_i32(val: i32) -> RawValue {
    val as u32 as u64
}

/// Convert i64 to raw value
#[inline(always)]
pub const fn from_i64(val: i64) -> RawValue {
    val as u64
}

/// Convert f32 to raw value (stored in lower 32 bits, zero-extended)
#[inline(always)]
pub const fn from_f32(val: f32) -> RawValue {
    val.to_bits() as u64
}

/// Convert f64 to raw value
#[inline(always)]
pub const fn from_f64(val: f64) -> RawValue {
    val.to_bits()
}

/// Convert reference to raw value
#[inline(always)]
pub const fn from_ref(val: RefHandle) -> RawValue {
    val.0 as u64
}

/// Extract i32 from raw value
#[inline(always)]
pub const fn as_i32(val: RawValue) -> i32 {
    val as u32 as i32
}

/// Extract i64 from raw value
#[inline(always)]
pub const fn as_i64(val: RawValue) -> i64 {
    val as i64
}

/// Extract f32 from raw value
#[inline(always)]
pub const fn as_f32(val: RawValue) -> f32 {
    f32::from_bits(val as u32)
}

/// Extract f64 from raw value
#[inline(always)]
pub const fn as_f64(val: RawValue) -> f64 {
    f64::from_bits(val)
}

/// Extract reference from raw value
#[inline(always)]
pub const fn as_ref(val: RawValue) -> RefHandle {
    RefHandle(val as usize)
}

/// Extract u32 from raw value (for unsigned operations)
#[inline(always)]
pub const fn as_u32(val: RawValue) -> u32 {
    val as u32
}

/// Extract u64 from raw value (for unsigned operations)
#[inline(always)]
pub const fn as_u64(val: RawValue) -> u64 {
    val
}

/// Convert Value (public enum) to RawValue (internal u64)
#[inline]
pub fn value_to_raw(val: Value) -> RawValue {
    match val {
        Value::I32(v) => from_i32(v),
        Value::I64(v) => from_i64(v),
        Value::F32(v) => from_f32(v),
        Value::F64(v) => from_f64(v),
        Value::Ref(r, _) => from_ref(r),
        Value::Unknown => 0,
    }
}

/// Convert RawValue (internal u64) to Value (public enum).
/// Requires type information to reconstruct the correct variant.
#[inline]
pub fn raw_to_value(raw: RawValue, value_type: ValueType) -> Value {
    match value_type {
        ValueType::I32 => Value::I32(as_i32(raw)),
        ValueType::I64 => Value::I64(as_i64(raw)),
        ValueType::F32 => Value::F32(as_f32(raw)),
        ValueType::F64 => Value::F64(as_f64(raw)),
        ValueType::V128 => Value::I64(as_i64(raw)), // TODO: Proper V128 support
        ValueType::Ref(ref_type) => Value::Ref(as_ref(raw), ref_type),
        _ => Value::Unknown,
    }
}
