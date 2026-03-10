//! External WebAssembly value API.
//!
//! This module provides the public interface for WebAssembly values,
//! used for function arguments, return values, and external API interactions.

use crate::value_type::{RefType, ValueType};
use core::fmt::Display;

/// A tagged reference handle for WebAssembly references.
/// Uses tag bits to distinguish between function references and extern references.
///
/// ## Tag Bit Layout (bits 60-61):
/// - Bit 61: Extern hierarchy marker (for externref values)
/// - Bit 60: Host reference marker (opaque host values)
///
/// ## Tag Combinations:
/// - `0b00`: Plain reference (funcref or untagged)
/// - `0b01` (bit 60): Host value
/// - `0b10` (bit 61): Externref wrapping a plain reference
/// - `0b11` (bits 60-61): Externref wrapping a host value
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RefHandle(pub(crate) usize);

impl RefHandle {
    // Tag bit constants
    const HOST_TAG: usize = 1 << 60;
    const EXTERN_TAG: usize = 1 << 61;
    const TAG_MASK: usize = 0x3 << 60; // Bits 60-61

    pub fn new(value: usize) -> Self {
        Self(value)
    }

    /// Null reference — uses usize::MAX as a sentinel value since it is
    /// guaranteed to be an invalid index.
    pub const fn null() -> Self {
        Self(usize::MAX)
    }

    pub fn is_null(&self) -> bool {
        self.0 == usize::MAX
    }

    /// Create an externref from a raw index (for host/test harness use).
    /// Tag: bit 61 = 1 for extern, bit 60 = 1 for host value.
    pub fn externref(index: usize) -> Self {
        Self(Self::EXTERN_TAG | Self::HOST_TAG | (index & 0xFFF_FFFF))
    }

    /// Check if this is a host reference (opaque host value)
    pub fn is_host(&self) -> bool {
        !self.is_null() && (self.0 & Self::HOST_TAG) != 0
    }

    /// Check if this reference is in the extern hierarchy
    pub fn is_extern(&self) -> bool {
        if self.is_null() {
            false
        } else {
            (self.0 & Self::EXTERN_TAG) != 0
        }
    }

    /// Get the raw payload value without any tag bits.
    pub fn raw_value(&self) -> usize {
        if self.is_null() {
            return usize::MAX;
        }
        self.0 & !Self::TAG_MASK
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

/// WebAssembly value for external API interactions.
/// This represents values passed into and out of WebAssembly functions.
/// Always carries both value and type information.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Value {
    /// 32-bit integer
    I32(i32),
    /// 64-bit integer
    I64(i64),
    /// 32-bit float
    F32(f32),
    /// 64-bit float
    F64(f64),
    /// Reference type (funcref, externref)
    /// Carries both the reference handle and its type information
    Ref(RefHandle, RefType),
    /// Unknown/uninitialized value
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

impl Value {
    /// Get the ValueType of this Value
    pub fn value_type(&self) -> ValueType {
        match self {
            Value::I32(_) => ValueType::I32,
            Value::I64(_) => ValueType::I64,
            Value::F32(_) => ValueType::F32,
            Value::F64(_) => ValueType::F64,
            Value::Ref(_, ref_type) => ValueType::Ref(*ref_type),
            Value::Unknown => ValueType::Unknown,
        }
    }

    /// Create a default value for the given type.
    /// For reference types, this returns a null reference with the appropriate type.
    pub fn default_for_type(value_type: ValueType) -> Self {
        match value_type {
            ValueType::I32 => Value::I32(0),
            ValueType::I64 => Value::I64(0),
            ValueType::F32 => Value::F32(0.0),
            ValueType::F64 => Value::F64(0.0),
            ValueType::V128 => Value::Unknown,
            ValueType::Ref(ref_type) => Value::Ref(RefHandle::null(), ref_type),
            ValueType::Unknown => Value::Unknown,
        }
    }

    /// Convert value to raw u64 representation for the interpreter stack.
    #[inline]
    pub fn to_raw(&self) -> u64 {
        match *self {
            Value::I32(v) => v as u32 as u64,
            Value::I64(v) => v as u64,
            Value::F32(v) => f32::to_bits(v) as u64,
            Value::F64(v) => f64::to_bits(v),
            Value::Ref(r, _) => r.raw_value() as u64,
            Value::Unknown => 0,
        }
    }

    /// Create a Value from raw u64 and a ValueType.
    #[inline]
    pub fn from_raw(raw: u64, ty: ValueType) -> Self {
        match ty {
            ValueType::I32 => Value::I32(raw as i32),
            ValueType::I64 => Value::I64(raw as i64),
            ValueType::F32 => Value::F32(f32::from_bits(raw as u32)),
            ValueType::F64 => Value::F64(f64::from_bits(raw)),
            ValueType::Ref(ref_type) => Value::Ref(RefHandle::new(raw as usize), ref_type),
            _ => Value::Unknown,
        }
    }
}
