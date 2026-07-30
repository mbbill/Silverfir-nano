//! Shared value encoding utilities for VM execution stacks.

use crate::error::WasmError;
use crate::value_type::ValueType;
use crate::vm::jit::store::Store;
use crate::vm::value::{RefHandle, Value, FUNCADDR_TOP};

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

/// Convert a function reference read from `frame_owner`'s local frame form
/// into the instance-independent absolute form.
///
/// This is total: null, special references, and values already outside this
/// instance's local-index range pass through unchanged. Resolution uses only
/// the per-instance local-index map and never checks out an instance slot.
#[inline]
pub(crate) fn absolutize(frame_owner: &Store, handle: RefHandle) -> RefHandle {
    if handle.is_null() || handle.is_special() {
        return handle;
    }
    let local_index = handle.encoded();
    if local_index >= frame_owner.module().functions.len() {
        return handle;
    }
    let Some(funcaddr) = frame_owner.funcaddr_for_local_index(local_index) else {
        return handle;
    };
    RefHandle::new(
        FUNCADDR_TOP
            .checked_sub(funcaddr)
            .expect("registered funcaddr exceeds the encoding range"),
    )
}

/// Convert a function reference written into `frame_owner`'s frame to that
/// instance's local form when it names one of its own functions.
///
/// This is total: null, special references, local-range values, malformed
/// out-of-range values, and another instance's absolute references pass
/// through unchanged. Resolution uses only the shared function arena and
/// never checks out an instance slot.
#[inline]
pub(crate) fn localize(frame_owner: &Store, handle: RefHandle) -> RefHandle {
    if handle.is_null() || handle.is_special() {
        return handle;
    }
    let encoded = handle.encoded();
    if encoded < frame_owner.module().functions.len() {
        return handle;
    }
    let Some(funcaddr) = FUNCADDR_TOP.checked_sub(encoded) else {
        return handle;
    };
    let Some(entry) = frame_owner.function_entry_for_funcaddr(funcaddr) else {
        return handle;
    };
    if entry.owner == frame_owner.instance_handle().self_id() {
        RefHandle::new(entry.local_index as usize)
    } else {
        handle
    }
}

/// Put a reference into the form required by a container owned by `store`.
///
/// Reachable containers hold the absolute form. A statically private
/// container may hold the owning instance's local form. Both conversions are
/// total, so foreign, null, and non-function references pass through.
#[inline]
pub(crate) fn retag_for_container(store: &Store, handle: RefHandle, reachable: bool) -> RefHandle {
    if reachable {
        absolutize(store, handle)
    } else {
        localize(store, handle)
    }
}

#[inline]
pub(crate) fn value_to_raw_in_store(val: Value, frame_owner: &mut Store) -> RawValue {
    match val {
        #[cfg(sf_has_simd)]
        Value::V128(value) => frame_owner.intern_v128(value),
        Value::I32(v) => from_i32(v),
        Value::I64(v) => from_i64(v),
        Value::F32(v) => from_f32(v),
        Value::F64(v) => from_f64(v),
        Value::Ref(r, _) => from_ref(localize(frame_owner, r)),
        Value::Unknown => 0,
    }
}

/// Convert an absolute [`Value`] for storage in a container owned by `store`.
#[inline]
pub(crate) fn value_to_container_raw_in_store(
    val: Value,
    reachable: bool,
    store: &mut Store,
) -> RawValue {
    match val {
        Value::Ref(r, _) => from_ref(retag_for_container(store, r, reachable)),
        other => value_to_raw_in_store(other, store),
    }
}

#[inline]
pub(crate) fn value_to_machine_raw_in_store(
    val: Value,
    gp_unit_bytes: u8,
    frame_owner: &mut Store,
) -> RawValue {
    match val {
        Value::Ref(r, _) => ref_to_machine_raw(localize(frame_owner, r), gp_unit_bytes),
        #[cfg(sf_has_simd)]
        Value::V128(value) => frame_owner.intern_v128(value),
        Value::I32(v) => from_i32(v),
        Value::I64(v) => from_i64(v),
        Value::F32(v) => from_f32(v),
        Value::F64(v) => from_f64(v),
        Value::Unknown => 0,
    }
}

#[inline]
#[cfg(sf_has_simd)]
fn invalid_v128_raw_error() -> WasmError {
    WasmError::internal("invalid v128 raw value")
}

#[inline]
pub(crate) fn try_machine_raw_to_value_in_store(
    raw: RawValue,
    value_type: ValueType,
    gp_unit_bytes: u8,
    frame_owner: &Store,
) -> Result<Value, WasmError> {
    Ok(match value_type {
        ValueType::Ref(ref_type) => Value::Ref(
            absolutize(frame_owner, machine_raw_to_ref(raw, gp_unit_bytes)),
            ref_type,
        ),
        ValueType::I32 => Value::I32(as_i32(raw)),
        ValueType::I64 => Value::I64(as_i64(raw)),
        ValueType::F32 => Value::F32(as_f32(raw)),
        ValueType::F64 => Value::F64(as_f64(raw)),
        #[cfg(sf_has_simd)]
        ValueType::V128 => Value::V128(
            frame_owner
                .get_v128(raw)
                .ok_or_else(invalid_v128_raw_error)?,
        ),
        #[cfg(not(sf_has_simd))]
        ValueType::V128 => Value::Unknown,
        _ => Value::Unknown,
    })
}

pub(crate) fn try_raw_to_value_in_store(
    raw: RawValue,
    value_type: ValueType,
    frame_owner: &Store,
) -> Result<Value, WasmError> {
    Ok(match value_type {
        ValueType::I32 => Value::I32(as_i32(raw)),
        ValueType::I64 => Value::I64(as_i64(raw)),
        ValueType::F32 => Value::F32(as_f32(raw)),
        ValueType::F64 => Value::F64(as_f64(raw)),
        #[cfg(sf_has_simd)]
        ValueType::V128 => Value::V128(
            frame_owner
                .get_v128(raw)
                .ok_or_else(invalid_v128_raw_error)?,
        ),
        #[cfg(not(sf_has_simd))]
        ValueType::V128 => Value::Unknown,
        ValueType::Ref(ref_type) => Value::Ref(absolutize(frame_owner, as_ref(raw)), ref_type),
        _ => Value::Unknown,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn function_address_range_endpoints_round_trip_at_both_gp_widths() {
        for gp_unit_bytes in [4, 8] {
            for encoded in [0, FUNCADDR_TOP] {
                let handle = RefHandle::new(encoded);
                assert!(!handle.is_special());
                let raw = ref_to_machine_raw(handle, gp_unit_bytes);
                let decoded = machine_raw_to_ref(raw, gp_unit_bytes);
                assert_eq!(decoded.raw(), encoded);
                assert!(!decoded.is_special());
            }
        }
    }
}
