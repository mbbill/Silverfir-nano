//! Public WebAssembly value API.
//!
//! This module provides the public interface for WebAssembly values,
//! used for function arguments, return values, and host API interactions.

use crate::value_type::{RefType, ValueType};
use core::fmt::Display;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RefHandle(pub(crate) usize);

impl RefHandle {
    #[cfg(target_pointer_width = "64")]
    const SPECIAL_TAG: usize = 1 << 60;
    #[cfg(target_pointer_width = "64")]
    const EXTERN_TAG: usize = 1 << 61;
    #[cfg(target_pointer_width = "64")]
    const PAYLOAD_BITS: u32 = 60;

    #[cfg(target_pointer_width = "32")]
    const SPECIAL_TAG: usize = 1 << 28;
    #[cfg(target_pointer_width = "32")]
    const EXTERN_TAG: usize = 1 << 29;
    #[cfg(target_pointer_width = "32")]
    const PAYLOAD_BITS: u32 = 28;

    const fn payload_mask() -> usize {
        Self::SPECIAL_TAG - 1
    }

    const fn pool_payload_tag() -> usize {
        1usize << (Self::PAYLOAD_BITS - 1)
    }

    const fn host_payload_mask() -> usize {
        Self::pool_payload_tag() - 1
    }

    pub const fn new(value: usize) -> Self {
        Self(value)
    }

    pub const fn null() -> Self {
        Self(usize::MAX)
    }

    pub fn is_null(&self) -> bool {
        self.0 == usize::MAX
    }

    pub fn hostref(index: usize) -> Self {
        Self(Self::SPECIAL_TAG | (index & Self::host_payload_mask()))
    }

    pub fn externref(index: usize) -> Self {
        Self(Self::SPECIAL_TAG | Self::EXTERN_TAG | (index & Self::host_payload_mask()))
    }

    pub fn is_host(&self) -> bool {
        self.is_special() && !self.is_pooled()
    }

    pub fn is_special(&self) -> bool {
        !self.is_null() && (self.0 & Self::SPECIAL_TAG) != 0
    }

    pub fn is_extern(&self) -> bool {
        if self.is_null() {
            false
        } else {
            (self.0 & Self::EXTERN_TAG) != 0
        }
    }

    pub(crate) fn is_pooled(&self) -> bool {
        self.is_special() && (self.0 & Self::pool_payload_tag()) != 0
    }

    pub(crate) fn pooled_index(&self) -> Option<usize> {
        self.is_pooled()
            .then_some(self.0 & Self::host_payload_mask())
    }

    pub fn host_index(&self) -> Option<usize> {
        self.is_host().then_some(self.0 & Self::host_payload_mask())
    }

    pub(crate) fn from_pool_index(index: usize) -> Self {
        Self(Self::SPECIAL_TAG | Self::pool_payload_tag() | (index & Self::host_payload_mask()))
    }

    pub fn to_any(self) -> Result<Self, ()> {
        if self.is_null() {
            Ok(self)
        } else if self.is_special() && self.is_extern() {
            Ok(Self(self.0 & !Self::EXTERN_TAG))
        } else {
            Err(())
        }
    }

    pub fn to_extern(self) -> Result<Self, ()> {
        if self.is_null() {
            Ok(self)
        } else if self.is_special() {
            Ok(Self(self.0 | Self::EXTERN_TAG))
        } else {
            Err(())
        }
    }

    #[inline]
    pub const fn encoded(self) -> usize {
        self.0
    }

    pub fn payload(&self) -> usize {
        if self.is_null() {
            return usize::MAX;
        }
        if self.is_pooled() {
            self.0 & Self::host_payload_mask()
        } else if self.is_special() {
            self.0 & Self::host_payload_mask()
        } else {
            self.0 & Self::payload_mask()
        }
    }
}

impl From<RefHandle> for usize {
    fn from(val: RefHandle) -> Self {
        val.0
    }
}

impl Display for RefHandle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Value {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    #[cfg(sf_has_simd)]
    V128([u8; 16]),
    Ref(RefHandle, RefType),
    #[default]
    Unknown,
}

impl From<i32> for Value {
    fn from(val: i32) -> Self {
        Value::I32(val)
    }
}

impl From<i64> for Value {
    fn from(val: i64) -> Self {
        Value::I64(val)
    }
}

impl From<f32> for Value {
    fn from(val: f32) -> Self {
        Value::F32(val)
    }
}

impl From<f64> for Value {
    fn from(val: f64) -> Self {
        Value::F64(val)
    }
}

#[cfg(sf_has_simd)]
impl From<[u8; 16]> for Value {
    fn from(val: [u8; 16]) -> Self {
        Value::V128(val)
    }
}

impl From<Value> for i8 {
    fn from(val: Value) -> Self {
        match val {
            Value::I32(val) => val as i8,
            _ => panic!("Value is not an i32"),
        }
    }
}

impl From<Value> for u8 {
    fn from(val: Value) -> Self {
        match val {
            Value::I32(val) => val as u8,
            _ => panic!("Value is not an i32"),
        }
    }
}

impl From<Value> for i16 {
    fn from(val: Value) -> Self {
        match val {
            Value::I32(val) => val as i16,
            _ => panic!("Value is not an i32"),
        }
    }
}

impl From<Value> for u16 {
    fn from(val: Value) -> Self {
        match val {
            Value::I32(val) => val as u16,
            _ => panic!("Value is not an i32"),
        }
    }
}

impl From<Value> for i32 {
    fn from(val: Value) -> Self {
        match val {
            Value::I32(val) => val,
            _ => panic!("Value is not an i32"),
        }
    }
}

impl From<Value> for u32 {
    fn from(val: Value) -> Self {
        match val {
            Value::I32(val) => val as u32,
            _ => panic!("Value is not an i32"),
        }
    }
}

impl From<Value> for i64 {
    fn from(val: Value) -> Self {
        match val {
            Value::I64(val) => val,
            _ => panic!("Value is not an i64"),
        }
    }
}

impl From<Value> for u64 {
    fn from(val: Value) -> Self {
        match val {
            Value::I64(val) => val as u64,
            _ => panic!("Value is not an i64"),
        }
    }
}

impl From<Value> for usize {
    fn from(val: Value) -> Self {
        match val {
            Value::I32(val) => val as u32 as usize,
            _ => panic!("Value is not an i32"),
        }
    }
}

impl From<Value> for f32 {
    fn from(val: Value) -> Self {
        match val {
            Value::F32(val) => val,
            _ => panic!("Value is not an f32"),
        }
    }
}

impl From<Value> for f64 {
    fn from(val: Value) -> Self {
        match val {
            Value::F64(val) => val,
            _ => panic!("Value is not an f64"),
        }
    }
}

impl From<Value> for RefHandle {
    fn from(val: Value) -> Self {
        match val {
            Value::Ref(r, _) => r,
            _ => panic!("Value is not a reference"),
        }
    }
}

#[cfg(sf_has_simd)]
impl From<Value> for [u8; 16] {
    fn from(val: Value) -> Self {
        match val {
            Value::V128(val) => val,
            _ => panic!("Value is not a v128"),
        }
    }
}

impl Value {
    #[inline]
    pub fn from_v128_bytes(_bytes: [u8; 16]) -> Self {
        #[cfg(sf_has_simd)]
        {
            Self::V128(_bytes)
        }
        #[cfg(not(sf_has_simd))]
        {
            Self::Unknown
        }
    }

    #[inline]
    pub fn as_v128_bytes(&self) -> Option<[u8; 16]> {
        #[cfg(sf_has_simd)]
        {
            if let Self::V128(bytes) = self {
                return Some(*bytes);
            }
        }
        None
    }

    pub fn value_type(&self) -> ValueType {
        match self {
            Value::I32(_) => ValueType::I32,
            Value::I64(_) => ValueType::I64,
            Value::F32(_) => ValueType::F32,
            Value::F64(_) => ValueType::F64,
            #[cfg(sf_has_simd)]
            Value::V128(_) => ValueType::V128,
            Value::Ref(_, ref_type) => ValueType::Ref(*ref_type),
            Value::Unknown => ValueType::Unknown,
        }
    }

    pub fn default_for_type(value_type: ValueType) -> Self {
        match value_type {
            ValueType::I32 => Value::I32(0),
            ValueType::I64 => Value::I64(0),
            ValueType::F32 => Value::F32(0.0),
            ValueType::F64 => Value::F64(0.0),
            #[cfg(sf_has_simd)]
            ValueType::V128 => Value::V128([0; 16]),
            #[cfg(not(sf_has_simd))]
            ValueType::V128 => Value::Unknown,
            ValueType::Ref(ref_type) => Value::Ref(RefHandle::null(), ref_type),
            ValueType::Unknown => Value::Unknown,
        }
    }

    #[inline]
    pub fn to_raw(&self) -> u64 {
        match *self {
            Value::I32(v) => v as u32 as u64,
            Value::I64(v) => v as u64,
            Value::F32(v) => f32::to_bits(v) as u64,
            Value::F64(v) => f64::to_bits(v),
            #[cfg(sf_has_simd)]
            Value::V128(_) => panic!("v128 cannot be encoded as a scalar raw value"),
            Value::Ref(r, _) => r.encoded() as u64,
            Value::Unknown => 0,
        }
    }

    #[inline]
    pub fn from_raw(raw: u64, ty: ValueType) -> Self {
        match ty {
            ValueType::I32 => Value::I32(raw as i32),
            ValueType::I64 => Value::I64(raw as i64),
            ValueType::F32 => Value::F32(f32::from_bits(raw as u32)),
            ValueType::F64 => Value::F64(f64::from_bits(raw)),
            ValueType::Ref(ref_type) => Value::Ref(RefHandle::new(raw as usize), ref_type),
            #[cfg(sf_has_simd)]
            ValueType::V128 => panic!("v128 cannot be decoded from a scalar raw value"),
            _ => Value::Unknown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Value;
    use crate::value_type::ValueType;

    #[test]
    fn default_scalar_values_match_their_types() {
        assert_eq!(Value::default_for_type(ValueType::I32), Value::I32(0));
        assert_eq!(Value::default_for_type(ValueType::I64), Value::I64(0));
        assert_eq!(Value::default_for_type(ValueType::F32), Value::F32(0.0));
        assert_eq!(Value::default_for_type(ValueType::F64), Value::F64(0.0));
    }

    #[cfg(sf_has_simd)]
    #[test]
    fn v128_values_report_their_type_and_default() {
        let bytes = [0xAB; 16];
        let value = Value::from(bytes);

        assert_eq!(value.value_type(), ValueType::V128);
        assert_eq!(<[u8; 16]>::from(value), bytes);
        assert_eq!(
            Value::default_for_type(ValueType::V128),
            Value::V128([0; 16])
        );
    }
}
