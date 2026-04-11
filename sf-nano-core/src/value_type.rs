//! WebAssembly 2.0 Type System
//!
//! This module implements the WebAssembly 2.0 type system including:
//! - Number types (i32, i64, f32, f64)
//! - Vector types (v128)
//! - Reference types (funcref, externref)

use crate::collections;

use crate::{error::WasmError, utils::leb128};
use core::fmt;

// ============================================================================
// Heap Types
// ============================================================================

/// Abstract heap types for WASM 2.0.
///
/// Only `Func` and `Extern` are used in WASM 2.0.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AbstractHeapType {
    Func = 0x70,
    Extern = 0x6F,
}

impl TryFrom<u8> for AbstractHeapType {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x70 => Ok(AbstractHeapType::Func),
            0x6F => Ok(AbstractHeapType::Extern),
            _ => Err(()),
        }
    }
}

impl fmt::Display for AbstractHeapType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Func => "func",
            Self::Extern => "extern",
        };
        write!(f, "{}", s)
    }
}

// ============================================================================
// Heap Type (abstract or concrete index)
// ============================================================================

/// Heap type — either an abstract heap type or a concrete type index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeapType {
    /// Abstract heap type (func, extern)
    Abstract(AbstractHeapType),
    /// Concrete type index referring to a defined type
    Concrete(u32),
}

impl HeapType {
    /// Create a heap type from a type index
    pub fn from_index(idx: u32) -> Self {
        Self::Concrete(idx)
    }

    /// Check if this is an abstract heap type
    pub fn is_abstract(&self) -> bool {
        matches!(self, Self::Abstract(_))
    }

    /// Check if this is a concrete (indexed) heap type
    pub fn is_concrete(&self) -> bool {
        matches!(self, Self::Concrete(_))
    }

    /// Get the abstract heap type if this is abstract
    pub fn as_abstract(&self) -> Option<AbstractHeapType> {
        match self {
            Self::Abstract(aht) => Some(*aht),
            Self::Concrete(_) => None,
        }
    }

    /// Get the type index if this is concrete
    pub fn as_concrete(&self) -> Option<u32> {
        match self {
            Self::Abstract(_) => None,
            Self::Concrete(idx) => Some(*idx),
        }
    }

    /// Parse a heap type from binary format
    ///
    /// Heap types are encoded as:
    /// - Single byte for abstract heap types (0x70, 0x6F)
    /// - Signed LEB128 for type indices (must be non-negative)
    pub fn parse(payload: &mut crate::utils::payload::Payload) -> Result<HeapType, WasmError> {
        let first_byte = payload.peek_u8().map_err(WasmError::from)?;

        // Try to parse as abstract heap type first
        if let Ok(aht) = AbstractHeapType::try_from(first_byte) {
            payload.read_u8().map_err(WasmError::from)?; // consume the byte
            return Ok(HeapType::Abstract(aht));
        }

        // Otherwise, it's a type index encoded as signed LEB128
        let type_idx = payload.read_leb128_i32().map_err(WasmError::from)?;

        if type_idx < 0 {
            return Err(WasmError::invalid("Type index cannot be negative"));
        }

        Ok(HeapType::Concrete(type_idx as u32))
    }

    /// Encode this heap type to binary format
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

// ============================================================================
// Reference Types
// ============================================================================

/// Reference type (WASM 2.0)
///
/// A reference type consists of:
/// - A heap type (what it points to)
/// - A nullability flag (whether null is allowed)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RefType {
    pub nullable: bool,
    pub heap_type: HeapType,
}

impl RefType {
    /// Create a new reference type
    pub fn new(nullable: bool, heap_type: HeapType) -> Self {
        Self {
            nullable,
            heap_type,
        }
    }

    /// Create a nullable reference to an abstract heap type
    pub fn nullable_abstract(heap_type: AbstractHeapType) -> Self {
        Self {
            nullable: true,
            heap_type: HeapType::Abstract(heap_type),
        }
    }

    /// Create a non-nullable reference to an abstract heap type
    pub fn non_nullable_abstract(heap_type: AbstractHeapType) -> Self {
        Self {
            nullable: false,
            heap_type: HeapType::Abstract(heap_type),
        }
    }

    /// Create a nullable reference to a concrete type
    pub fn nullable_concrete(type_idx: u32) -> Self {
        Self {
            nullable: true,
            heap_type: HeapType::Concrete(type_idx),
        }
    }

    /// Create a non-nullable reference to a concrete type
    pub fn non_nullable_concrete(type_idx: u32) -> Self {
        Self {
            nullable: false,
            heap_type: HeapType::Concrete(type_idx),
        }
    }

    /// funcref = (ref null func)
    pub fn funcref() -> Self {
        Self::nullable_abstract(AbstractHeapType::Func)
    }

    /// externref = (ref null extern)
    pub fn externref() -> Self {
        Self::nullable_abstract(AbstractHeapType::Extern)
    }

    /// Convert to nullable variant
    pub fn to_nullable(self) -> Self {
        Self {
            nullable: true,
            ..self
        }
    }

    /// Encode RefType into a u64 for passing in instruction immediates.
    ///
    /// Encoding:
    /// - Bit 0: nullable (0 or 1)
    /// - Bit 1: is_concrete (0 = Abstract, 1 = Concrete)
    /// - Bits 2+: AbstractHeapType u8 discriminant (if Abstract) or u32 type index (if Concrete)
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

    /// Decode RefType from a u64 immediate value.
    ///
    /// The encoded value must have been created by `encode_to_u64`.
    pub fn decode_from_u64(encoded: u64) -> Self {
        let nullable = (encoded & 1) != 0;
        let is_concrete = (encoded & 2) != 0;

        let heap_type = if is_concrete {
            let type_idx = (encoded >> 2) as u32;
            HeapType::Concrete(type_idx)
        } else {
            let discriminant = (encoded >> 2) as u8;
            let abstract_type = AbstractHeapType::try_from(discriminant)
                .expect("Invalid AbstractHeapType discriminant in encoded RefType");
            HeapType::Abstract(abstract_type)
        };

        Self::new(nullable, heap_type)
    }

    /// Convert to non-nullable variant
    pub fn to_non_nullable(self) -> Self {
        Self {
            nullable: false,
            ..self
        }
    }

    /// Check if this is a funcref-compatible type
    pub fn is_funcref(&self) -> bool {
        matches!(
            self.heap_type,
            HeapType::Abstract(AbstractHeapType::Func) | HeapType::Concrete(_)
        )
    }

    /// Check if this is an externref-compatible type
    pub fn is_externref(&self) -> bool {
        matches!(self.heap_type, HeapType::Abstract(AbstractHeapType::Extern))
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

// ============================================================================
// Value Types
// ============================================================================

/// WebAssembly 2.0 Value Types
///
/// Value types classify the individual values that WebAssembly code can
/// compute with and the values that a variable accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueType {
    // Number types
    I32,
    I64,
    F32,
    F64,

    // Vector types
    V128,

    // Reference types
    Ref(RefType),

    /// Unknown type (called `bot` in the spec) — a bottom type for validation.
    ///
    /// Used only during validation for stack-polymorphic typing (e.g., after
    /// `unreachable` or unconditional branches).
    Unknown,
}

impl ValueType {
    // ========================================================================
    // Constructors for common types
    // ========================================================================

    pub fn funcref() -> Self {
        Self::Ref(RefType::funcref())
    }

    pub fn externref() -> Self {
        Self::Ref(RefType::externref())
    }

    // ========================================================================
    // Type Category Checks
    // ========================================================================

    /// Check if this is a number type (i32, i64, f32, f64)
    #[inline]
    pub fn is_num(&self) -> bool {
        matches!(
            self,
            ValueType::I32 | ValueType::I64 | ValueType::F32 | ValueType::F64 | ValueType::Unknown
        )
    }

    /// Check if this is an integer type (i32, i64)
    #[inline]
    pub fn is_int(&self) -> bool {
        matches!(self, ValueType::I32 | ValueType::I64 | ValueType::Unknown)
    }

    /// Check if this is a float type (f32, f64)
    #[inline]
    pub fn is_float(&self) -> bool {
        matches!(self, ValueType::F32 | ValueType::F64 | ValueType::Unknown)
    }

    /// Check if this is a vector type (v128)
    #[inline]
    pub fn is_vec(&self) -> bool {
        matches!(self, ValueType::V128 | ValueType::Unknown)
    }

    /// Check if this is a reference type
    #[inline]
    pub fn is_ref(&self) -> bool {
        matches!(self, ValueType::Ref(_) | ValueType::Unknown)
    }

    // ========================================================================
    // Specific Type Checks
    // ========================================================================

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

    /// Check if this is a funcref-compatible type
    pub fn is_funcref(&self) -> bool {
        match self {
            ValueType::Ref(rt) => matches!(
                rt.heap_type,
                HeapType::Abstract(AbstractHeapType::Func) | HeapType::Concrete(_)
            ),
            ValueType::Unknown => true,
            _ => false,
        }
    }

    /// Check if this is an externref-compatible type
    pub fn is_externref(&self) -> bool {
        match self {
            ValueType::Ref(rt) => {
                matches!(rt.heap_type, HeapType::Abstract(AbstractHeapType::Extern))
            }
            ValueType::Unknown => true,
            _ => false,
        }
    }

    /// Check if this type is nullable
    pub fn is_nullable(&self) -> bool {
        match self {
            ValueType::Ref(rt) => rt.nullable,
            ValueType::Unknown => true,
            _ => false,
        }
    }

    /// Check if this type is defaultable
    ///
    /// - Number types have default value 0
    /// - Vector types have default value 0
    /// - Nullable references have default value null
    /// - Non-nullable references have NO default value
    pub fn is_defaultable(&self) -> bool {
        match self {
            ValueType::I32 | ValueType::I64 | ValueType::F32 | ValueType::F64 | ValueType::V128 => {
                true
            }
            ValueType::Ref(rt) => rt.nullable,
            ValueType::Unknown => true,
        }
    }

    // ========================================================================
    // Type Matching
    // ========================================================================

    /// Check if this type is compatible with another type (subtype relationship).
    /// In WASM 2.0: non-nullable ≤ nullable, Concrete(idx) ≤ Abstract(Func).
    pub fn is_compatible_with(&self, other: &ValueType) -> bool {
        if self == other || *self == ValueType::Unknown || *other == ValueType::Unknown {
            return true;
        }
        // Reference subtyping
        if let (ValueType::Ref(actual_ref), ValueType::Ref(expected_ref)) = (self, other) {
            // Non-nullable is subtype of nullable (not the reverse)
            if !expected_ref.nullable && actual_ref.nullable {
                return false;
            }
            if actual_ref.heap_type == expected_ref.heap_type {
                return true;
            }
            // Concrete(idx) is a subtype of Abstract(Func)
            if let (HeapType::Concrete(_), HeapType::Abstract(AbstractHeapType::Func)) =
                (actual_ref.heap_type, expected_ref.heap_type)
            {
                return true;
            }
        }
        false
    }

    // ========================================================================
    // Utility Methods
    // ========================================================================

    /// Get the bit width of a number or vector type
    pub fn bit_width(&self) -> Option<u32> {
        match self {
            ValueType::I32 | ValueType::F32 => Some(32),
            ValueType::I64 | ValueType::F64 => Some(64),
            ValueType::V128 => Some(128),
            _ => None,
        }
    }

    /// Convert to nullable variant (only for reference types)
    pub fn to_nullable(&self) -> ValueType {
        match self {
            ValueType::Ref(rt) => ValueType::Ref(rt.to_nullable()),
            other => *other,
        }
    }

    /// Convert to non-nullable variant (only for reference types)
    pub fn to_non_nullable(&self) -> ValueType {
        match self {
            ValueType::Ref(rt) => ValueType::Ref(rt.to_non_nullable()),
            other => *other,
        }
    }

    /// Returns the lowercase name of this type for IR formatting.
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

    // ========================================================================
    // Binary Format Parsing
    // ========================================================================

    /// Parse a value type from a payload
    ///
    /// Handles all encoding forms per WASM 2.0:
    /// - Number types (0x7F, 0x7E, 0x7D, 0x7C)
    /// - Vector types (0x7B)
    /// - Short form ref types: single byte abstract heap type = (ref null <heaptype>)
    /// - General form ref types: 0x63/0x64 prefix + heap type
    pub fn parse(payload: &mut crate::utils::payload::Payload) -> Result<ValueType, WasmError> {
        let first_byte = payload.read_u8().map_err(WasmError::from)?;

        match first_byte {
            // Number types
            0x7F => Ok(ValueType::I32),
            0x7E => Ok(ValueType::I64),
            0x7D => Ok(ValueType::F32),
            0x7C => Ok(ValueType::F64),

            // Vector types
            0x7B => Ok(ValueType::V128),

            // General form: ref null <heaptype>
            0x63 => {
                let heap_type = HeapType::parse(payload)?;
                Ok(ValueType::Ref(RefType::new(true, heap_type)))
            }

            // General form: ref <heaptype>
            0x64 => {
                let heap_type = HeapType::parse(payload)?;
                Ok(ValueType::Ref(RefType::new(false, heap_type)))
            }

            // Short form: abstract heap type alone = (ref null <heaptype>)
            _ => {
                if let Ok(aht) = AbstractHeapType::try_from(first_byte) {
                    Ok(ValueType::Ref(RefType::nullable_abstract(aht)))
                } else {
                    Err(WasmError::invalid("Invalid value type byte: 0x at position . Expected number type (0x7C-0x7F), vector type (0x7B), ref prefix (0x63/0x64), or abstract heap type (0x6F/0x70)"))
                }
            }
        }
    }

    /// Encode this value type to binary format
    pub fn encode(&self) -> collections::Vec<u8> {
        match self {
            ValueType::I32 => collections::vec![0x7F],
            ValueType::I64 => collections::vec![0x7E],
            ValueType::F32 => collections::vec![0x7D],
            ValueType::F64 => collections::vec![0x7C],
            ValueType::V128 => collections::vec![0x7B],

            ValueType::Ref(rt) => {
                // Check if we can use short form (nullable abstract heap type)
                if rt.nullable {
                    if let HeapType::Abstract(aht) = rt.heap_type {
                        return collections::vec![aht as u8];
                    }
                }

                // General form required
                let mut bytes = collections::vec![if rt.nullable { 0x63 } else { 0x64 }];
                bytes.extend(rt.heap_type.encode());
                bytes
            }

            ValueType::Unknown => {
                panic!("Cannot encode Unknown value type")
            }
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
