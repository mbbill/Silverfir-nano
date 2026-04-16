//! WebAssembly 3.0 Type Definitions
//!
//! This module contains all composite type definitions used in the unified
//! type section:
//! - Function types (0x60)
//! - Struct types (0x5F)
//! - Array types (0x5E)

use crate::collections;
use crate::{
    error::WasmError,
    value_type::{HeapType, ValueType},
};
use core::fmt;
use tracked_alloc::rc::Rc;
use tracked_alloc::string::String;

/// A complete type definition from the type section.
#[derive(Debug, Clone, PartialEq)]
pub struct DefType {
    /// The composite type (function, struct, or array).
    pub composite: CompositeType,
    /// Indices of declared supertypes.
    pub supertypes: collections::Vec<u32>,
    /// Whether this type is final.
    pub is_final: bool,
    /// Explicit recursion-group id, if any.
    pub rec_group: Option<u32>,
}

impl DefType {
    pub fn func(params: collections::Vec<ValueType>, results: collections::Vec<ValueType>) -> Self {
        Self {
            composite: CompositeType::Func(Rc::new(FunctionType::new(params, results))),
            supertypes: collections::vec![],
            is_final: true,
            rec_group: None,
        }
    }

    pub fn struct_type(fields: collections::Vec<FieldType>) -> Self {
        Self {
            composite: CompositeType::Struct(StructType { fields }),
            supertypes: collections::vec![],
            is_final: true,
            rec_group: None,
        }
    }

    pub fn array(element: FieldType) -> Self {
        Self {
            composite: CompositeType::Array(ArrayType { element }),
            supertypes: collections::vec![],
            is_final: true,
            rec_group: None,
        }
    }

    pub fn validate_type_references(
        &self,
        current_idx: usize,
        type_count: usize,
        all_types: &[Rc<DefType>],
    ) -> Result<(), WasmError> {
        for &supertype_idx in &self.supertypes {
            if supertype_idx as usize >= type_count {
                return Err(WasmError::invalid("unknown supertype"));
            }
        }

        match &self.composite {
            CompositeType::Func(func_type) => {
                for param in func_type.params() {
                    validate_valtype_references(
                        param,
                        current_idx,
                        self.rec_group,
                        type_count,
                        all_types,
                    )?;
                }
                for result in func_type.results() {
                    validate_valtype_references(
                        result,
                        current_idx,
                        self.rec_group,
                        type_count,
                        all_types,
                    )?;
                }
            }
            CompositeType::Struct(struct_type) => {
                for field in &struct_type.fields {
                    validate_storage_type_references(
                        &field.storage,
                        current_idx,
                        self.rec_group,
                        type_count,
                        all_types,
                    )?;
                }
            }
            CompositeType::Array(array_type) => {
                validate_storage_type_references(
                    &array_type.element.storage,
                    current_idx,
                    self.rec_group,
                    type_count,
                    all_types,
                )?;
            }
        }

        Ok(())
    }
}

pub fn validate_valtype_references(
    vt: &ValueType,
    current_idx: usize,
    rec_group: Option<u32>,
    type_count: usize,
    all_types: &[Rc<DefType>],
) -> Result<(), WasmError> {
    if let ValueType::Ref(ref_type) = vt {
        if let HeapType::Concrete(idx) = ref_type.heap_type {
            let idx_usize = idx as usize;
            if idx_usize >= type_count {
                return Err(WasmError::invalid("unknown type"));
            }

            if idx_usize > current_idx {
                if let Some(my_rec_group) = rec_group {
                    if let Some(referenced_type) = all_types.get(idx_usize) {
                        if referenced_type.rec_group != Some(my_rec_group) {
                            return Err(WasmError::invalid(
                                "forward reference crosses recursion group boundaries",
                            ));
                        }
                    }
                } else {
                    return Err(WasmError::invalid(
                        "forward reference outside recursion group",
                    ));
                }
            }
        }
    }

    Ok(())
}

fn validate_storage_type_references(
    st: &StorageType,
    current_idx: usize,
    rec_group: Option<u32>,
    type_count: usize,
    all_types: &[Rc<DefType>],
) -> Result<(), WasmError> {
    if let StorageType::Val(vt) = st {
        validate_valtype_references(vt, current_idx, rec_group, type_count, all_types)?;
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub enum CompositeType {
    Func(Rc<FunctionType>),
    Struct(StructType),
    Array(ArrayType),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionType {
    params: collections::Vec<ValueType>,
    results: collections::Vec<ValueType>,
}

impl FunctionType {
    pub fn new(params: collections::Vec<ValueType>, results: collections::Vec<ValueType>) -> Self {
        Self { params, results }
    }

    pub fn params(&self) -> &[ValueType] {
        &self.params
    }

    pub fn results(&self) -> &[ValueType] {
        &self.results
    }
}

impl fmt::Display for FunctionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let params: collections::Vec<String> = self
            .params
            .iter()
            .map(|v| {
                let mut buf = String::new();
                core::fmt::Write::write_fmt(&mut buf, format_args!("{}", v)).unwrap();
                buf
            })
            .collect();
        let results: collections::Vec<String> = self
            .results
            .iter()
            .map(|v| {
                let mut buf = String::new();
                core::fmt::Write::write_fmt(&mut buf, format_args!("{}", v)).unwrap();
                buf
            })
            .collect();
        write!(f, "({}) -> ({})", params.join(", "), results.join(", "))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructType {
    pub fields: collections::Vec<FieldType>,
}

impl StructType {
    pub fn new(fields: collections::Vec<FieldType>) -> Self {
        Self { fields }
    }

    pub fn field_offset(&self, field_idx: usize) -> usize {
        self.fields[..field_idx]
            .iter()
            .map(|field| field.storage.size_bytes())
            .sum()
    }

    pub fn size_bytes(&self) -> usize {
        self.fields
            .iter()
            .map(|field| field.storage.size_bytes())
            .sum()
    }

    pub fn field_count(&self) -> usize {
        self.fields.len()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArrayType {
    pub element: FieldType,
}

impl ArrayType {
    pub fn new(element: FieldType) -> Self {
        Self { element }
    }

    pub fn element_valtype(&self) -> ValueType {
        self.element.storage.to_valtype()
    }

    pub fn element_size_bytes(&self) -> usize {
        self.element.storage.size_bytes()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldType {
    pub storage: StorageType,
    pub mutable: bool,
}

impl FieldType {
    pub fn new(storage: StorageType, mutable: bool) -> Self {
        Self { storage, mutable }
    }

    pub fn immutable(storage: StorageType) -> Self {
        Self {
            storage,
            mutable: false,
        }
    }

    pub fn mutable_field(storage: StorageType) -> Self {
        Self {
            storage,
            mutable: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StorageType {
    Val(ValueType),
    Packed(PackedType),
}

impl StorageType {
    pub fn to_valtype(&self) -> ValueType {
        match self {
            StorageType::Val(vt) => *vt,
            StorageType::Packed(PackedType::I8 | PackedType::I16) => ValueType::I32,
        }
    }

    pub fn size_bytes(&self) -> usize {
        match self {
            StorageType::Val(_) => 8,
            StorageType::Packed(PackedType::I8) => 1,
            StorageType::Packed(PackedType::I16) => 2,
        }
    }

    pub fn is_packed(&self) -> bool {
        matches!(self, StorageType::Packed(_))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedType {
    I8,
    I16,
}

impl PackedType {
    pub fn size_bytes(&self) -> usize {
        match self {
            PackedType::I8 => 1,
            PackedType::I16 => 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_type() {
        let ft = FunctionType::new(
            collections::vec![ValueType::I32, ValueType::I64],
            collections::vec![ValueType::F32],
        );

        assert_eq!(ft.params().len(), 2);
        assert_eq!(ft.results().len(), 1);
    }

    #[test]
    fn test_struct_size() {
        let st = StructType {
            fields: collections::vec![
                FieldType::immutable(StorageType::Val(ValueType::I32)),
                FieldType::immutable(StorageType::Val(ValueType::I64)),
                FieldType::immutable(StorageType::Packed(PackedType::I8)),
            ],
        };

        assert_eq!(st.size_bytes(), 17);
        assert_eq!(st.field_offset(0), 0);
        assert_eq!(st.field_offset(1), 8);
        assert_eq!(st.field_offset(2), 16);
    }
}
