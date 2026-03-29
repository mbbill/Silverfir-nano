//! Shared raw value utilities for VM execution stacks.

use crate::value_type::ValueType;
use crate::vm::value::{RefHandle, Value};

pub(crate) type RawValue = u64;

#[inline(always)]
pub(crate) const fn from_i32(val: i32) -> RawValue {
    val as u32 as u64
}

#[inline(always)]
pub(crate) const fn from_i64(val: i64) -> RawValue {
    val as u64
}

#[inline(always)]
pub(crate) const fn from_f32(val: f32) -> RawValue {
    val.to_bits() as u64
}

#[inline(always)]
pub(crate) const fn from_f64(val: f64) -> RawValue {
    val.to_bits()
}

#[inline(always)]
pub(crate) const fn from_ref(val: RefHandle) -> RawValue {
    val.0 as u64
}

#[inline(always)]
pub(crate) const fn as_i32(val: RawValue) -> i32 {
    val as u32 as i32
}

#[inline(always)]
pub(crate) const fn as_i64(val: RawValue) -> i64 {
    val as i64
}

#[inline(always)]
pub(crate) const fn as_f32(val: RawValue) -> f32 {
    f32::from_bits(val as u32)
}

#[inline(always)]
pub(crate) const fn as_f64(val: RawValue) -> f64 {
    f64::from_bits(val)
}

#[inline(always)]
pub(crate) const fn as_ref(val: RawValue) -> RefHandle {
    RefHandle::new(val as usize)
}

#[inline(always)]
pub(crate) const fn as_u32(val: RawValue) -> u32 {
    val as u32
}

#[inline(always)]
pub(crate) const fn as_u64(val: RawValue) -> u64 {
    val
}

#[inline]
pub(crate) fn value_to_raw(val: Value) -> RawValue {
    match val {
        Value::I32(v) => from_i32(v),
        Value::I64(v) => from_i64(v),
        Value::F32(v) => from_f32(v),
        Value::F64(v) => from_f64(v),
        Value::Ref(r, _) => from_ref(r),
        Value::Unknown => 0,
    }
}

#[inline]
pub(crate) fn raw_to_value(raw: RawValue, value_type: ValueType) -> Value {
    match value_type {
        ValueType::I32 => Value::I32(as_i32(raw)),
        ValueType::I64 => Value::I64(as_i64(raw)),
        ValueType::F32 => Value::F32(as_f32(raw)),
        ValueType::F64 => Value::F64(as_f64(raw)),
        ValueType::V128 => Value::I64(as_i64(raw)),
        ValueType::Ref(ref_type) => Value::Ref(as_ref(raw), ref_type),
        _ => Value::Unknown,
    }
}
