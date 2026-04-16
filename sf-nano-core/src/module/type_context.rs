//! Type context for type resolution and subtyping.
//!
//! The module keeps the unified WebAssembly 3.0 type table but still lets most
//! of nano work with resolved `FunctionType`s on functions.

use crate::collections;
use crate::module::type_defs::{CompositeType, DefType, FunctionType};
use crate::value_type::{HeapType, ValueType};
use tracked_alloc::rc::Rc;

#[derive(Clone)]
pub struct TypeContext {
    types: Rc<[Rc<DefType>]>,
}

impl TypeContext {
    pub fn new(types: collections::Vec<Rc<DefType>>) -> Self {
        Self {
            types: collections::into_alloc_vec(types).into(),
        }
    }

    pub fn empty() -> Self {
        Self {
            types: Rc::from([]),
        }
    }

    pub fn get(&self, idx: u32) -> Option<&Rc<DefType>> {
        self.types.get(idx as usize)
    }

    pub fn as_slice(&self) -> &[Rc<DefType>] {
        &self.types
    }

    pub fn len(&self) -> usize {
        self.types.len()
    }

    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }

    pub fn get_function_type(&self, idx: u32) -> Option<&Rc<FunctionType>> {
        self.get(idx)
            .and_then(|def_type| match &def_type.composite {
                CompositeType::Func(func_type) => Some(func_type),
                _ => None,
            })
    }

    pub fn types_equivalent(&self, idx1: u32, idx2: u32) -> bool {
        let mut visiting = collections::Vec::new();
        self.types_equivalent_inner(idx1, idx2, &mut visiting)
    }

    fn types_equivalent_inner(
        &self,
        idx1: u32,
        idx2: u32,
        visiting: &mut collections::Vec<(u32, u32)>,
    ) -> bool {
        let pair = if idx1 < idx2 {
            (idx1, idx2)
        } else {
            (idx2, idx1)
        };

        if visiting.contains(&pair) {
            return true;
        }
        if idx1 == idx2 {
            return true;
        }

        let Some(type1) = self.get(idx1) else {
            return false;
        };
        let Some(type2) = self.get(idx2) else {
            return false;
        };

        visiting.push(pair);

        let result = match (&type1.composite, &type2.composite) {
            (CompositeType::Func(_), CompositeType::Func(_)) => {
                self.composite_types_equivalent(&type1.composite, &type2.composite, visiting)
            }
            (CompositeType::Struct(_), CompositeType::Struct(_))
            | (CompositeType::Array(_), CompositeType::Array(_)) => {
                match (type1.rec_group, type2.rec_group) {
                    (Some(g1), Some(g2)) if g1 == g2 => self.composite_types_equivalent(
                        &type1.composite,
                        &type2.composite,
                        visiting,
                    ),
                    _ => false,
                }
            }
            _ => false,
        };

        if let Some(pos) = visiting.iter().position(|entry| *entry == pair) {
            visiting.swap_remove(pos);
        }

        result
    }

    fn composite_types_equivalent(
        &self,
        c1: &CompositeType,
        c2: &CompositeType,
        visiting: &mut collections::Vec<(u32, u32)>,
    ) -> bool {
        match (c1, c2) {
            (CompositeType::Func(f1), CompositeType::Func(f2)) => {
                if f1.params().len() != f2.params().len()
                    || f1.results().len() != f2.results().len()
                {
                    return false;
                }

                for (p1, p2) in f1.params().iter().zip(f2.params().iter()) {
                    if !self.value_types_equivalent(p1, p2, visiting) {
                        return false;
                    }
                }
                for (r1, r2) in f1.results().iter().zip(f2.results().iter()) {
                    if !self.value_types_equivalent(r1, r2, visiting) {
                        return false;
                    }
                }
                true
            }
            (CompositeType::Struct(s1), CompositeType::Struct(s2)) => {
                if s1.fields.len() != s2.fields.len() {
                    return false;
                }

                for (field1, field2) in s1.fields.iter().zip(s2.fields.iter()) {
                    if field1.mutable != field2.mutable {
                        return false;
                    }
                    if !self.storage_types_equivalent(&field1.storage, &field2.storage, visiting) {
                        return false;
                    }
                }

                true
            }
            (CompositeType::Array(a1), CompositeType::Array(a2)) => {
                a1.element.mutable == a2.element.mutable
                    && self.storage_types_equivalent(
                        &a1.element.storage,
                        &a2.element.storage,
                        visiting,
                    )
            }
            _ => false,
        }
    }

    fn storage_types_equivalent(
        &self,
        s1: &crate::module::type_defs::StorageType,
        s2: &crate::module::type_defs::StorageType,
        visiting: &mut collections::Vec<(u32, u32)>,
    ) -> bool {
        use crate::module::type_defs::{PackedType, StorageType};

        match (s1, s2) {
            (StorageType::Val(v1), StorageType::Val(v2)) => {
                self.value_types_equivalent(v1, v2, visiting)
            }
            (StorageType::Packed(PackedType::I8), StorageType::Packed(PackedType::I8)) => true,
            (StorageType::Packed(PackedType::I16), StorageType::Packed(PackedType::I16)) => true,
            _ => false,
        }
    }

    fn value_types_equivalent(
        &self,
        v1: &ValueType,
        v2: &ValueType,
        visiting: &mut collections::Vec<(u32, u32)>,
    ) -> bool {
        match (v1, v2) {
            (ValueType::I32, ValueType::I32) => true,
            (ValueType::I64, ValueType::I64) => true,
            (ValueType::F32, ValueType::F32) => true,
            (ValueType::F64, ValueType::F64) => true,
            (ValueType::V128, ValueType::V128) => true,
            (ValueType::Unknown, ValueType::Unknown) => true,
            (ValueType::Ref(rt1), ValueType::Ref(rt2)) => {
                if rt1.nullable != rt2.nullable {
                    return false;
                }

                match (&rt1.heap_type, &rt2.heap_type) {
                    (HeapType::Abstract(a1), HeapType::Abstract(a2)) => a1 == a2,
                    (HeapType::Concrete(idx1), HeapType::Concrete(idx2)) => {
                        self.types_equivalent_inner(*idx1, *idx2, visiting)
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }
}

impl core::ops::Deref for TypeContext {
    type Target = [Rc<DefType>];

    fn deref(&self) -> &Self::Target {
        &self.types
    }
}

impl AsRef<[Rc<DefType>]> for TypeContext {
    fn as_ref(&self) -> &[Rc<DefType>] {
        &self.types
    }
}

impl core::fmt::Debug for TypeContext {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TypeContext")
            .field("num_types", &self.types.len())
            .finish()
    }
}

pub fn check_function_types_equivalent(
    export_type: &FunctionType,
    import_type: &FunctionType,
    export_type_ctx: &TypeContext,
) -> bool {
    if export_type.params().len() != import_type.params().len()
        || export_type.results().len() != import_type.results().len()
    {
        return false;
    }

    for (exp_param, imp_param) in export_type.params().iter().zip(import_type.params().iter()) {
        if !value_types_equivalent_cross_module(exp_param, imp_param, export_type_ctx) {
            return false;
        }
    }

    for (exp_result, imp_result) in export_type
        .results()
        .iter()
        .zip(import_type.results().iter())
    {
        if !value_types_equivalent_cross_module(exp_result, imp_result, export_type_ctx) {
            return false;
        }
    }

    true
}

pub fn value_types_equivalent_cross_module(
    exp_type: &ValueType,
    imp_type: &ValueType,
    export_type_ctx: &TypeContext,
) -> bool {
    match (exp_type, imp_type) {
        (ValueType::I32, ValueType::I32) => true,
        (ValueType::I64, ValueType::I64) => true,
        (ValueType::F32, ValueType::F32) => true,
        (ValueType::F64, ValueType::F64) => true,
        (ValueType::V128, ValueType::V128) => true,
        (ValueType::Ref(exp_ref), ValueType::Ref(imp_ref)) => {
            if exp_ref.nullable != imp_ref.nullable {
                return false;
            }

            match (&exp_ref.heap_type, &imp_ref.heap_type) {
                (HeapType::Abstract(a1), HeapType::Abstract(a2)) => a1 == a2,
                (HeapType::Concrete(exp_idx), HeapType::Concrete(imp_idx)) => {
                    if exp_idx == imp_idx {
                        true
                    } else {
                        export_type_ctx.types_equivalent(*exp_idx, *imp_idx)
                    }
                }
                _ => false,
            }
        }
        _ => false,
    }
}
