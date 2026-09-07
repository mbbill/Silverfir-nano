//! Values at the embedding boundary. Engine slot encodings remain private.
use crate::value_type::{RefType, ValueType};
use crate::vm::value::{RefValue as RawRef, Value as RawValue};

/// Temporary embedding conversions. Keep common small signatures off the
/// allocator without changing the engine's slot layout or limiting arities.
pub(crate) enum ValueBuffer<T: Copy> {
    Empty,
    Inline { values: [T; 4], len: usize },
    Heap(alloc::vec::Vec<T>),
}

impl<T: Copy> ValueBuffer<T> {
    pub(crate) fn new(len: usize, initial: T) -> Self {
        if len == 0 {
            Self::Empty
        } else if len <= 4 {
            Self::Inline {
                values: [initial; 4],
                len,
            }
        } else {
            Self::Heap(alloc::vec![initial; len])
        }
    }
}

impl<T: Copy> core::ops::Deref for ValueBuffer<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match self {
            Self::Empty => &[],
            Self::Inline { values, len } => &values[..*len],
            Self::Heap(values) => values,
        }
    }
}

impl<T: Copy> core::ops::DerefMut for ValueBuffer<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        match self {
            Self::Empty => &mut [],
            Self::Inline { values, len } => &mut values[..*len],
            Self::Heap(values) => values,
        }
    }
}

/// An opaque reference obtained from Wasm or constructed as a host label.
/// Engine references belong to one runtime world. Copying a handle does not
/// keep an instance alive; using a handle after its owner is freed fails.
/// Raw engine encodings cannot be constructed through the safe public API:
///
/// ```compile_fail
/// let forged = sf_nano_core::RefValue::new(0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RefValue {
    pub(crate) raw: RawRef,
    pub(crate) world: usize,
}

impl RefValue {
    pub const fn null() -> Self {
        Self {
            raw: RawRef::null(),
            world: 0,
        }
    }
    pub fn is_null(self) -> bool {
        self.raw.is_null()
    }

    /// A portable host label in the `any` hierarchy. The index must fit in
    /// 27 bits on every supported target; larger indices panic.
    pub fn hostref(index: usize) -> Self {
        assert!(index < (1 << 27), "host reference index exceeds 27 bits");
        Self {
            raw: RawRef::hostref(index),
            world: 0,
        }
    }

    /// A portable host label in the `extern` hierarchy, with the same index
    /// bound as [`Self::hostref`].
    pub fn externref(index: usize) -> Self {
        assert!(index < (1 << 27), "host reference index exceeds 27 bits");
        Self {
            raw: RawRef::externref(index),
            world: 0,
        }
    }

    /// The label supplied to `hostref`/`externref`. Engine references and
    /// nulls have no host label, including GC objects converted to externref.
    pub fn host_id(self) -> Option<usize> {
        self.raw.is_host().then(|| self.raw.payload())
    }
    pub fn is_extern(self) -> bool {
        self.raw.is_extern()
    }
    pub fn to_any(self) -> Result<Self, ()> {
        Ok(Self {
            raw: self.raw.to_any()?,
            ..self
        })
    }
    pub fn to_extern(self) -> Result<Self, ()> {
        Ok(Self {
            raw: self.raw.to_extern()?,
            ..self
        })
    }

    pub(crate) fn from_vm(raw: RawRef, world: usize) -> Self {
        Self {
            raw,
            world: if raw.is_null() || raw.is_host() {
                0
            } else {
                world
            },
        }
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
    Ref(RefValue, RefType),
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

impl From<Value> for RefValue {
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
            ValueType::Ref(ref_type) => Value::Ref(RefValue::null(), ref_type),
            ValueType::Unknown => Value::Unknown,
        }
    }
}

impl Value {
    pub(crate) fn from_vm(value: RawValue, world: usize) -> Self {
        match value {
            RawValue::I32(x) => Self::I32(x),
            RawValue::I64(x) => Self::I64(x),
            RawValue::F32(x) => Self::F32(x),
            RawValue::F64(x) => Self::F64(x),
            #[cfg(sf_has_simd)]
            RawValue::V128(x) => Self::V128(x),
            RawValue::Ref(r, ty) => Self::Ref(RefValue::from_vm(r, world), ty),
            RawValue::Unknown => Self::Unknown,
        }
    }

    /// Representation conversion only. The embedding entry must validate
    /// world ownership and liveness before an engine can consume this value.
    pub(crate) fn vm_value(self) -> RawValue {
        match self {
            Self::I32(x) => RawValue::I32(x),
            Self::I64(x) => RawValue::I64(x),
            Self::F32(x) => RawValue::F32(x),
            Self::F64(x) => RawValue::F64(x),
            #[cfg(sf_has_simd)]
            Self::V128(x) => RawValue::V128(x),
            Self::Ref(r, ty) => RawValue::Ref(r.raw, ty),
            Self::Unknown => RawValue::Unknown,
        }
    }
}
