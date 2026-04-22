//! WebAssembly 3.0 Type System
//!
//! This module keeps nano's `no_std` value-type model aligned with the richer
//! reference and GC type surface used by `silverfir-rs`.

use crate::{
    collections, error::WasmError, module::type_context::TypeContext,
    module::type_defs::CompositeType, utils::leb128,
};
use core::fmt;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AbstractHeapType {
    Func = 0x70,
    NoFunc = 0x73,
    Extern = 0x6F,
    NoExtern = 0x72,
    Exn = 0x69,
    NoExn = 0x74,
    Any = 0x6E,
    Eq = 0x6D,
    I31 = 0x6C,
    Struct = 0x6B,
    Array = 0x6A,
    None = 0x71,
}

impl TryFrom<u8> for AbstractHeapType {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x70 => Ok(Self::Func),
            0x73 => Ok(Self::NoFunc),
            0x6F => Ok(Self::Extern),
            0x72 => Ok(Self::NoExtern),
            0x69 => Ok(Self::Exn),
            0x74 => Ok(Self::NoExn),
            0x6E => Ok(Self::Any),
            0x6D => Ok(Self::Eq),
            0x6C => Ok(Self::I31),
            0x6B => Ok(Self::Struct),
            0x6A => Ok(Self::Array),
            0x71 => Ok(Self::None),
            _ => Err(()),
        }
    }
}

impl AbstractHeapType {
    pub fn is_top(&self) -> bool {
        matches!(self, Self::Func | Self::Extern | Self::Exn | Self::Any)
    }

    pub fn is_bottom(&self) -> bool {
        matches!(
            self,
            Self::NoFunc | Self::NoExtern | Self::NoExn | Self::None
        )
    }

    pub fn top_type(&self) -> Self {
        match self {
            Self::Func | Self::NoFunc => Self::Func,
            Self::Extern | Self::NoExtern => Self::Extern,
            Self::Exn | Self::NoExn => Self::Exn,
            Self::Any | Self::Eq | Self::I31 | Self::Struct | Self::Array | Self::None => Self::Any,
        }
    }

    pub fn bottom_type(&self) -> Self {
        match self {
            Self::Func | Self::NoFunc => Self::NoFunc,
            Self::Extern | Self::NoExtern => Self::NoExtern,
            Self::Exn | Self::NoExn => Self::NoExn,
            Self::Any | Self::Eq | Self::I31 | Self::Struct | Self::Array | Self::None => {
                Self::None
            }
        }
    }

    pub fn is_eq(&self) -> bool {
        matches!(
            self,
            Self::Eq | Self::I31 | Self::Struct | Self::Array | Self::None
        )
    }

    pub fn is_subtype_of(&self, other: &AbstractHeapType) -> bool {
        if self == other {
            return true;
        }

        match (self, other) {
            (Self::NoFunc, Self::Func) => true,
            (Self::NoExtern, Self::Extern) => true,
            (Self::NoExn, Self::Exn) => true,
            (Self::None, Self::I31 | Self::Struct | Self::Array | Self::Eq | Self::Any) => true,
            (Self::I31, Self::Eq | Self::Any) => true,
            (Self::Struct, Self::Eq | Self::Any) => true,
            (Self::Array, Self::Eq | Self::Any) => true,
            (Self::Eq, Self::Any) => true,
            _ => false,
        }
    }
}

impl fmt::Display for AbstractHeapType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Func => "func",
            Self::NoFunc => "nofunc",
            Self::Extern => "extern",
            Self::NoExtern => "noextern",
            Self::Exn => "exn",
            Self::NoExn => "noexn",
            Self::Any => "any",
            Self::Eq => "eq",
            Self::I31 => "i31",
            Self::Struct => "struct",
            Self::Array => "array",
            Self::None => "none",
        };
        write!(f, "{}", s)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeapType {
    Abstract(AbstractHeapType),
    Concrete(u32),
}

impl HeapType {
    pub fn from_index(idx: u32) -> Self {
        Self::Concrete(idx)
    }

    pub fn is_abstract(&self) -> bool {
        matches!(self, Self::Abstract(_))
    }

    pub fn is_concrete(&self) -> bool {
        matches!(self, Self::Concrete(_))
    }

    pub fn as_abstract(&self) -> Option<AbstractHeapType> {
        match self {
            Self::Abstract(aht) => Some(*aht),
            Self::Concrete(_) => None,
        }
    }

    pub fn as_concrete(&self) -> Option<u32> {
        match self {
            Self::Abstract(_) => None,
            Self::Concrete(idx) => Some(*idx),
        }
    }

    pub fn is_subtype_of(&self, other: &HeapType, types: &TypeContext) -> bool {
        match (self, other) {
            (a, b) if a == b => true,
            (Self::Abstract(sub), Self::Abstract(sup)) => sub.is_subtype_of(sup),
            (Self::Concrete(idx1), Self::Concrete(idx2)) => {
                if types.types_equivalent(*idx1, *idx2) {
                    return true;
                }

                let Some(type1) = types.get(*idx1) else {
                    return false;
                };
                if type1.supertypes.contains(idx2) {
                    return true;
                }
                for &supertype_idx in &type1.supertypes {
                    if Self::Concrete(supertype_idx).is_subtype_of(&Self::Concrete(*idx2), types) {
                        return true;
                    }
                }
                false
            }
            (Self::Concrete(idx), Self::Abstract(AbstractHeapType::Func)) => types
                .get(*idx)
                .is_some_and(|def_type| matches!(def_type.composite, CompositeType::Func(_))),
            (Self::Concrete(idx), Self::Abstract(AbstractHeapType::Struct)) => types
                .get(*idx)
                .is_some_and(|def_type| matches!(def_type.composite, CompositeType::Struct(_))),
            (Self::Concrete(idx), Self::Abstract(AbstractHeapType::Array)) => types
                .get(*idx)
                .is_some_and(|def_type| matches!(def_type.composite, CompositeType::Array(_))),
            (Self::Concrete(idx), Self::Abstract(AbstractHeapType::Eq | AbstractHeapType::Any)) => {
                types.get(*idx).is_some_and(|def_type| {
                    matches!(
                        def_type.composite,
                        CompositeType::Struct(_) | CompositeType::Array(_)
                    )
                })
            }
            (Self::Abstract(AbstractHeapType::NoFunc), Self::Concrete(idx)) => types
                .get(*idx)
                .is_some_and(|def_type| matches!(def_type.composite, CompositeType::Func(_))),
            (Self::Abstract(AbstractHeapType::None), Self::Concrete(idx)) => {
                types.get(*idx).is_some_and(|def_type| {
                    matches!(
                        def_type.composite,
                        CompositeType::Struct(_) | CompositeType::Array(_)
                    )
                })
            }
            _ => false,
        }
    }

    pub fn top_type(&self, types: &TypeContext) -> HeapType {
        match self {
            Self::Abstract(aht) => Self::Abstract(aht.top_type()),
            Self::Concrete(idx) => match types.get(*idx) {
                Some(def_type) => match &def_type.composite {
                    CompositeType::Func(_) => Self::Abstract(AbstractHeapType::Func),
                    CompositeType::Struct(_) | CompositeType::Array(_) => {
                        Self::Abstract(AbstractHeapType::Any)
                    }
                },
                None => Self::Abstract(AbstractHeapType::Func),
            },
        }
    }

    pub fn parse(payload: &mut crate::utils::payload::Payload) -> Result<HeapType, WasmError> {
        let first_byte = payload.peek_u8().map_err(WasmError::from)?;

        if let Ok(aht) = AbstractHeapType::try_from(first_byte) {
            payload.read_u8().map_err(WasmError::from)?;
            return Ok(HeapType::Abstract(aht));
        }

        let type_idx = payload.read_leb128_i32().map_err(WasmError::from)?;
        if type_idx < 0 {
            return Err(WasmError::invalid("Type index cannot be negative"));
        }

        Ok(HeapType::Concrete(type_idx as u32))
    }

    pub fn encode(&self) -> collections::Vec<u8> {
        match self {
            HeapType::Abstract(aht) => collections::vec![*aht as u8],
            HeapType::Concrete(idx) => leb128::write_leb128_i32(*idx as i32),
        }
    }
}

impl fmt::Display for HeapType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Abstract(aht) => write!(f, "{}", aht),
            Self::Concrete(idx) => write!(f, "${}", idx),
        }
    }
}

impl From<AbstractHeapType> for HeapType {
    fn from(aht: AbstractHeapType) -> Self {
        Self::Abstract(aht)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RefType {
    pub nullable: bool,
    pub heap_type: HeapType,
}

impl RefType {
    pub fn new(nullable: bool, heap_type: HeapType) -> Self {
        Self {
            nullable,
            heap_type,
        }
    }

    pub fn nullable_abstract(heap_type: AbstractHeapType) -> Self {
        Self {
            nullable: true,
            heap_type: HeapType::Abstract(heap_type),
        }
    }

    pub fn non_nullable_abstract(heap_type: AbstractHeapType) -> Self {
        Self {
            nullable: false,
            heap_type: HeapType::Abstract(heap_type),
        }
    }

    pub fn nullable_concrete(type_idx: u32) -> Self {
        Self {
            nullable: true,
            heap_type: HeapType::Concrete(type_idx),
        }
    }

    pub fn non_nullable_concrete(type_idx: u32) -> Self {
        Self {
            nullable: false,
            heap_type: HeapType::Concrete(type_idx),
        }
    }

    pub fn funcref() -> Self {
        Self::nullable_abstract(AbstractHeapType::Func)
    }

    pub fn externref() -> Self {
        Self::nullable_abstract(AbstractHeapType::Extern)
    }

    pub fn anyref() -> Self {
        Self::nullable_abstract(AbstractHeapType::Any)
    }

    pub fn eqref() -> Self {
        Self::nullable_abstract(AbstractHeapType::Eq)
    }

    pub fn i31ref() -> Self {
        Self::nullable_abstract(AbstractHeapType::I31)
    }

    pub fn structref() -> Self {
        Self::nullable_abstract(AbstractHeapType::Struct)
    }

    pub fn arrayref() -> Self {
        Self::nullable_abstract(AbstractHeapType::Array)
    }

    pub fn exnref() -> Self {
        Self::nullable_abstract(AbstractHeapType::Exn)
    }

    pub fn nullref() -> Self {
        Self::nullable_abstract(AbstractHeapType::None)
    }

    pub fn nullfuncref() -> Self {
        Self::nullable_abstract(AbstractHeapType::NoFunc)
    }

    pub fn nullexternref() -> Self {
        Self::nullable_abstract(AbstractHeapType::NoExtern)
    }

    pub fn nullexnref() -> Self {
        Self::nullable_abstract(AbstractHeapType::NoExn)
    }

    pub fn to_nullable(self) -> Self {
        Self {
            nullable: true,
            ..self
        }
    }

    pub fn encode_to_u64(&self) -> u64 {
        let mut encoded = if self.nullable { 1u64 } else { 0u64 };

        match self.heap_type {
            HeapType::Abstract(abstract_type) => {
                encoded |= (abstract_type as u64) << 2;
            }
            HeapType::Concrete(type_idx) => {
                encoded |= 1u64 << 1;
                encoded |= (type_idx as u64) << 2;
            }
        }

        encoded
    }

    pub fn decode_from_u64(encoded: u64) -> Self {
        let nullable = (encoded & 1) != 0;
        let is_concrete = (encoded & 2) != 0;

        let heap_type = if is_concrete {
            HeapType::Concrete((encoded >> 2) as u32)
        } else {
            let discriminant = (encoded >> 2) as u8;
            let abstract_type = AbstractHeapType::try_from(discriminant)
                .expect("Invalid AbstractHeapType discriminant in encoded RefType");
            HeapType::Abstract(abstract_type)
        };

        Self::new(nullable, heap_type)
    }

    pub fn to_non_nullable(self) -> Self {
        Self {
            nullable: false,
            ..self
        }
    }

    pub fn difference(a: RefType, b: RefType) -> Self {
        Self {
            nullable: if b.nullable { false } else { a.nullable },
            heap_type: a.heap_type,
        }
    }

    pub fn is_funcref(&self) -> bool {
        matches!(
            self.heap_type,
            HeapType::Abstract(AbstractHeapType::Func | AbstractHeapType::NoFunc)
                | HeapType::Concrete(_)
        )
    }

    pub fn is_externref(&self) -> bool {
        matches!(
            self.heap_type,
            HeapType::Abstract(AbstractHeapType::Extern | AbstractHeapType::NoExtern)
        )
    }

    pub fn is_subtype_of(&self, other: &RefType, types: &TypeContext) -> bool {
        let nullability_ok = self.nullable == other.nullable || other.nullable;
        let heap_ok = self.heap_type.is_subtype_of(&other.heap_type, types);
        nullability_ok && heap_ok
    }
}

impl fmt::Display for RefType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.nullable {
            write!(f, "(ref null {})", self.heap_type)
        } else {
            write!(f, "(ref {})", self.heap_type)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueType {
    I32,
    I64,
    F32,
    F64,
    V128,
    Ref(RefType),
    Unknown,
}

impl ValueType {
    pub fn funcref() -> Self {
        Self::Ref(RefType::funcref())
    }

    pub fn externref() -> Self {
        Self::Ref(RefType::externref())
    }

    pub fn anyref() -> Self {
        Self::Ref(RefType::anyref())
    }

    pub fn eqref() -> Self {
        Self::Ref(RefType::eqref())
    }

    pub fn i31ref() -> Self {
        Self::Ref(RefType::i31ref())
    }

    pub fn structref() -> Self {
        Self::Ref(RefType::structref())
    }

    pub fn arrayref() -> Self {
        Self::Ref(RefType::arrayref())
    }

    pub fn exnref() -> Self {
        Self::Ref(RefType::exnref())
    }

    /// Non-null `(ref exn)` — distinct from the nullable `exnref`. The caught
    /// exception on `catch_ref` / `catch_all_ref` label targets is always a
    /// live exception reference, never null.
    pub fn ref_exn() -> Self {
        Self::Ref(RefType::non_nullable_abstract(AbstractHeapType::Exn))
    }

    pub fn is_num(&self) -> bool {
        matches!(
            self,
            ValueType::I32 | ValueType::I64 | ValueType::F32 | ValueType::F64 | ValueType::Unknown
        )
    }

    #[inline]
    pub fn is_int(&self) -> bool {
        matches!(self, ValueType::I32 | ValueType::I64 | ValueType::Unknown)
    }

    #[inline]
    pub fn is_float(&self) -> bool {
        matches!(self, ValueType::F32 | ValueType::F64 | ValueType::Unknown)
    }

    #[inline]
    pub fn is_vec(&self) -> bool {
        matches!(self, ValueType::V128 | ValueType::Unknown)
    }

    #[inline]
    pub fn is_ref(&self) -> bool {
        matches!(self, ValueType::Ref(_) | ValueType::Unknown)
    }

    #[inline]
    pub fn is_i32(&self) -> bool {
        matches!(self, ValueType::I32 | ValueType::Unknown)
    }

    #[inline]
    pub fn is_i64(&self) -> bool {
        matches!(self, ValueType::I64 | ValueType::Unknown)
    }

    #[inline]
    pub fn is_f32(&self) -> bool {
        matches!(self, ValueType::F32 | ValueType::Unknown)
    }

    #[inline]
    pub fn is_f64(&self) -> bool {
        matches!(self, ValueType::F64 | ValueType::Unknown)
    }

    #[inline]
    pub fn is_v128(&self) -> bool {
        matches!(self, ValueType::V128 | ValueType::Unknown)
    }

    #[inline]
    pub fn ensure_enabled_in_build(&self) -> Result<(), WasmError> {
        #[cfg(not(sf_has_simd))]
        if matches!(self, ValueType::V128) {
            return Err(WasmError::invalid("SIMD is not supported on this CPU"));
        }
        Ok(())
    }

    pub fn is_funcref(&self) -> bool {
        match self {
            ValueType::Ref(rt) => rt.is_funcref(),
            ValueType::Unknown => true,
            _ => false,
        }
    }

    pub fn is_externref(&self) -> bool {
        match self {
            ValueType::Ref(rt) => rt.is_externref(),
            ValueType::Unknown => true,
            _ => false,
        }
    }

    pub fn is_nullable(&self) -> bool {
        match self {
            ValueType::Ref(rt) => rt.nullable,
            ValueType::Unknown => true,
            _ => false,
        }
    }

    pub fn is_defaultable(&self) -> bool {
        match self {
            ValueType::I32 | ValueType::I64 | ValueType::F32 | ValueType::F64 | ValueType::V128 => {
                true
            }
            ValueType::Ref(rt) => rt.nullable,
            ValueType::Unknown => true,
        }
    }

    pub fn is_subtype_of(&self, other: &ValueType, types: &TypeContext) -> bool {
        if *self == ValueType::Unknown {
            return true;
        }
        if self == other {
            return true;
        }
        match (self, other) {
            (ValueType::Ref(sub_rt), ValueType::Ref(sup_rt)) => sub_rt.is_subtype_of(sup_rt, types),
            _ => false,
        }
    }

    pub fn is_compatible_with(&self, other: &ValueType) -> bool {
        self == other || *self == ValueType::Unknown || *other == ValueType::Unknown
    }

    #[inline]
    pub fn can_initialize(&self, target_type: &ValueType, types: &TypeContext) -> bool {
        self.is_subtype_of(target_type, types)
    }

    pub fn bit_width(&self) -> Option<u32> {
        match self {
            ValueType::I32 | ValueType::F32 => Some(32),
            ValueType::I64 | ValueType::F64 => Some(64),
            ValueType::V128 => Some(128),
            _ => None,
        }
    }

    pub fn to_nullable(&self) -> ValueType {
        match self {
            ValueType::Ref(rt) => ValueType::Ref(rt.to_nullable()),
            other => *other,
        }
    }

    pub fn to_non_nullable(&self) -> ValueType {
        match self {
            ValueType::Ref(rt) => ValueType::Ref(rt.to_non_nullable()),
            other => *other,
        }
    }

    pub fn is_subtype_of_eqref(&self) -> bool {
        use AbstractHeapType::*;
        match self {
            ValueType::Ref(rt) => match &rt.heap_type {
                HeapType::Abstract(aht) => matches!(aht, Eq | I31 | Struct | Array | None),
                HeapType::Concrete(_) => true,
            },
            _ => false,
        }
    }

    pub fn to_lowercase_name(&self) -> &'static str {
        match self {
            ValueType::I32 => "i32",
            ValueType::I64 => "i64",
            ValueType::F32 => "f32",
            ValueType::F64 => "f64",
            ValueType::V128 => "v128",
            ValueType::Ref(_) => "ref",
            ValueType::Unknown => "unknown",
        }
    }

    pub fn parse(payload: &mut crate::utils::payload::Payload) -> Result<ValueType, WasmError> {
        let first_byte = payload.read_u8().map_err(WasmError::from)?;

        match first_byte {
            0x7F => Ok(ValueType::I32),
            0x7E => Ok(ValueType::I64),
            0x7D => Ok(ValueType::F32),
            0x7C => Ok(ValueType::F64),
            0x7B => Ok(ValueType::V128),
            0x63 => {
                let heap_type = HeapType::parse(payload)?;
                Ok(ValueType::Ref(RefType::new(true, heap_type)))
            }
            0x64 => {
                let heap_type = HeapType::parse(payload)?;
                Ok(ValueType::Ref(RefType::new(false, heap_type)))
            }
            _ => {
                if let Ok(aht) = AbstractHeapType::try_from(first_byte) {
                    Ok(ValueType::Ref(RefType::nullable_abstract(aht)))
                } else {
                    Err(WasmError::invalid("Invalid value type byte"))
                }
            }
        }
    }

    pub fn encode(&self) -> collections::Vec<u8> {
        match self {
            ValueType::I32 => collections::vec![0x7F],
            ValueType::I64 => collections::vec![0x7E],
            ValueType::F32 => collections::vec![0x7D],
            ValueType::F64 => collections::vec![0x7C],
            ValueType::V128 => collections::vec![0x7B],
            ValueType::Ref(rt) => {
                if rt.nullable {
                    if let HeapType::Abstract(aht) = rt.heap_type {
                        return collections::vec![aht as u8];
                    }
                }

                let mut bytes = collections::vec![if rt.nullable { 0x63 } else { 0x64 }];
                bytes.extend(rt.heap_type.encode());
                bytes
            }
            ValueType::Unknown => panic!("Cannot encode Unknown value type"),
        }
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValueType::I32 => write!(f, "i32"),
            ValueType::I64 => write!(f, "i64"),
            ValueType::F32 => write!(f, "f32"),
            ValueType::F64 => write!(f, "f64"),
            ValueType::V128 => write!(f, "v128"),
            ValueType::Ref(rt) => write!(f, "{}", rt),
            ValueType::Unknown => write!(f, "unknown"),
        }
    }
}

impl TryFrom<u8> for ValueType {
    type Error = WasmError;

    fn try_from(byte: u8) -> Result<Self, Self::Error> {
        let data = [byte];
        let mut payload = crate::utils::payload::Payload::from(&data[..]);
        ValueType::parse(&mut payload)
    }
}
