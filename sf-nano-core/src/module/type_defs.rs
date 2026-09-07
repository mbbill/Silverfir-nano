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
pub(crate) struct DefType {
    /// The composite type (function, struct, or array).
    pub(crate) composite: CompositeType,
    /// Indices of declared supertypes.
    pub(crate) supertypes: collections::Vec<u32>,
    /// Whether this type is final.
    pub(crate) is_final: bool,
    /// Explicit recursion-group id, if any.
    pub(crate) rec_group: Option<u32>,
}

impl DefType {
    pub(crate) fn validate_type_references(
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

pub(crate) fn validate_valtype_references(
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
pub(crate) enum CompositeType {
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
pub(crate) struct StructType {
    pub(crate) fields: collections::Vec<FieldType>,
}

impl StructType {
    pub(crate) fn new(fields: collections::Vec<FieldType>) -> Self {
        Self { fields }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ArrayType {
    pub(crate) element: FieldType,
}

impl ArrayType {
    pub(crate) fn new(element: FieldType) -> Self {
        Self { element }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FieldType {
    pub(crate) storage: StorageType,
    pub(crate) mutable: bool,
}

impl FieldType {
    pub(crate) fn new(storage: StorageType, mutable: bool) -> Self {
        Self { storage, mutable }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum StorageType {
    Val(ValueType),
    Packed(PackedType),
}

impl StorageType {
    pub(crate) fn to_valtype(&self) -> ValueType {
        match self {
            StorageType::Val(vt) => *vt,
            StorageType::Packed(PackedType::I8 | PackedType::I16) => ValueType::I32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PackedType {
    I8,
    I16,
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
}
