//! Shared value encoding utilities for VM execution stacks.

use crate::value_type::ValueType;
use crate::vm::value::{RefHandle, Value};

pub(crate) type RawValue = u64;

// When generated code uses 32-bit GP slots on a wider host, refs need a
// compact wire encoding that preserves the special/extern/pooled split instead
// of reusing the host `RefHandle` bit layout directly.
const TARGET32_REF_SPECIAL_TAG: u32 = 1 << 28;
const TARGET32_REF_EXTERN_TAG: u32 = 1 << 29;
const TARGET32_REF_POOL_TAG: u32 = 1 << 27;
const TARGET32_REF_PAYLOAD_MASK: u32 = TARGET32_REF_SPECIAL_TAG - 1;
const TARGET32_REF_HOST_PAYLOAD_MASK: u32 = TARGET32_REF_POOL_TAG - 1;

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
    val.encoded() as u64
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
pub(crate) fn ref_to_machine_raw(handle: RefHandle, gp_unit_bytes: u8) -> RawValue {
    if gp_unit_bytes != 4 {
        return handle.encoded() as u64;
    }

    if handle.is_null() {
        return u32::MAX as u64;
    }

    if !handle.is_special() {
        return (handle.encoded() as u32 & TARGET32_REF_PAYLOAD_MASK) as u64;
    }

    let payload = (handle.payload() as u32) & TARGET32_REF_HOST_PAYLOAD_MASK;
    let mut raw = TARGET32_REF_SPECIAL_TAG | payload;
    if handle.is_pooled() {
        raw |= TARGET32_REF_POOL_TAG;
    }
    if handle.is_extern() {
        raw |= TARGET32_REF_EXTERN_TAG;
    }
    raw as u64
}

#[inline(always)]
pub(crate) fn machine_raw_to_ref(raw: RawValue, gp_unit_bytes: u8) -> RefHandle {
    if gp_unit_bytes != 4 {
        return RefHandle::new(raw as usize);
    }

    let raw = raw as u32;
    if raw == u32::MAX {
        return RefHandle::null();
    }
    if (raw & TARGET32_REF_SPECIAL_TAG) == 0 {
        return RefHandle::new((raw & TARGET32_REF_PAYLOAD_MASK) as usize);
    }

    let payload = (raw & TARGET32_REF_HOST_PAYLOAD_MASK) as usize;
    let is_extern = (raw & TARGET32_REF_EXTERN_TAG) != 0;
    if (raw & TARGET32_REF_POOL_TAG) != 0 {
        let mut handle = RefHandle::from_pool_index(payload);
        if is_extern {
            handle = handle.to_extern().expect("pooled refs are special");
        }
        handle
    } else if is_extern {
        RefHandle::externref(payload)
    } else {
        RefHandle::hostref(payload)
    }
}

#[cfg(sf_emulator)]
#[inline(always)]
pub(crate) const fn as_u32(val: RawValue) -> u32 {
    val as u32
}

#[cfg(sf_emulator)]
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

#[inline]
pub(crate) fn value_to_machine_raw(val: Value, gp_unit_bytes: u8) -> RawValue {
    match val {
        Value::Ref(r, _) => ref_to_machine_raw(r, gp_unit_bytes),
        _ => value_to_raw(val),
    }
}

#[inline]
pub(crate) fn machine_raw_to_value(
    raw: RawValue,
    value_type: ValueType,
    gp_unit_bytes: u8,
) -> Value {
    match value_type {
        ValueType::Ref(ref_type) => Value::Ref(machine_raw_to_ref(raw, gp_unit_bytes), ref_type),
        _ => raw_to_value(raw, value_type),
    }
}

#[inline]
pub(crate) fn normalize_machine_raw(
    raw: RawValue,
    value_type: ValueType,
    gp_unit_bytes: u8,
) -> RawValue {
    machine_raw_to_value(raw, value_type, gp_unit_bytes).to_raw()
}
