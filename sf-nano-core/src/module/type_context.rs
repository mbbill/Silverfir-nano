//! Type context for type resolution and subtyping.
//!
//! The module keeps the unified WebAssembly 3.0 type table but still lets most
//! of nano work with resolved `FunctionType`s on functions.

use crate::collections;
use crate::module::type_defs::{CompositeType, DefType, FunctionType};
use crate::value_type::{HeapType, ValueType};
use tracked_alloc::rc::Rc;

#[derive(Clone, Copy)]
struct TypeGroupInfo {
    start: u32,
    len: u32,
    offset: u32,
}

#[derive(Clone, Copy)]
struct ActiveTypeGroup {
    lhs_start: u32,
    rhs_start: u32,
    len: u32,
}

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
        let mut visiting_groups = collections::Vec::new();
        self.types_equivalent_inner(idx1, idx2, &mut visiting_groups)
    }

    fn types_equivalent_inner(
        &self,
        idx1: u32,
        idx2: u32,
        visiting_groups: &mut collections::Vec<(u32, u32)>,
    ) -> bool {
        if idx1 == idx2 {
            return true;
        }

        let Some(group1) = self.group_info(idx1) else {
            return false;
        };
        let Some(group2) = self.group_info(idx2) else {
            return false;
        };

        if group1.len != group2.len || group1.offset != group2.offset {
            return false;
        }

        self.groups_equivalent(group1, group2, visiting_groups)
    }

    fn groups_equivalent(
        &self,
        lhs: TypeGroupInfo,
        rhs: TypeGroupInfo,
        visiting_groups: &mut collections::Vec<(u32, u32)>,
    ) -> bool {
        let pair = if lhs.start < rhs.start {
            (lhs.start, rhs.start)
        } else {
            (rhs.start, lhs.start)
        };
        if visiting_groups.contains(&pair) {
            return true;
        }

        visiting_groups.push(pair);
        let active_group = ActiveTypeGroup {
            lhs_start: lhs.start,
            rhs_start: rhs.start,
            len: lhs.len,
        };
        let result = (0..lhs.len).all(|offset| {
            self.def_types_equivalent(
                lhs.start + offset,
                rhs.start + offset,
                Some(active_group),
                visiting_groups,
            )
        });
        if let Some(pos) = visiting_groups.iter().position(|entry| *entry == pair) {
            visiting_groups.swap_remove(pos);
        }
        result
    }

    fn def_types_equivalent(
        &self,
        idx1: u32,
        idx2: u32,
        active_group: Option<ActiveTypeGroup>,
        visiting_groups: &mut collections::Vec<(u32, u32)>,
    ) -> bool {
        let Some(type1) = self.get(idx1) else {
            return false;
        };
        let Some(type2) = self.get(idx2) else {
            return false;
        };

        if type1.is_final != type2.is_final || type1.supertypes.len() != type2.supertypes.len() {
            return false;
        }

        for (&super1, &super2) in type1.supertypes.iter().zip(type2.supertypes.iter()) {
            if !self.type_indices_equivalent(super1, super2, active_group, visiting_groups) {
                return false;
            }
        }

        match (&type1.composite, &type2.composite) {
            (CompositeType::Func(f1), CompositeType::Func(f2)) => {
                if f1.params().len() != f2.params().len()
                    || f1.results().len() != f2.results().len()
                {
                    return false;
                }

                for (p1, p2) in f1.params().iter().zip(f2.params().iter()) {
                    if !self.value_types_equivalent(p1, p2, active_group, visiting_groups) {
                        return false;
                    }
                }
                for (r1, r2) in f1.results().iter().zip(f2.results().iter()) {
                    if !self.value_types_equivalent(r1, r2, active_group, visiting_groups) {
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
                    if !self.storage_types_equivalent(
                        &field1.storage,
                        &field2.storage,
                        active_group,
                        visiting_groups,
                    ) {
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
                        active_group,
                        visiting_groups,
                    )
            }
            _ => false,
        }
    }

    fn storage_types_equivalent(
        &self,
        s1: &crate::module::type_defs::StorageType,
        s2: &crate::module::type_defs::StorageType,
        active_group: Option<ActiveTypeGroup>,
        visiting_groups: &mut collections::Vec<(u32, u32)>,
    ) -> bool {
        use crate::module::type_defs::{PackedType, StorageType};

        match (s1, s2) {
            (StorageType::Val(v1), StorageType::Val(v2)) => {
                self.value_types_equivalent(v1, v2, active_group, visiting_groups)
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
        active_group: Option<ActiveTypeGroup>,
        visiting_groups: &mut collections::Vec<(u32, u32)>,
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
                        self.type_indices_equivalent(*idx1, *idx2, active_group, visiting_groups)
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    fn type_indices_equivalent(
        &self,
        idx1: u32,
        idx2: u32,
        active_group: Option<ActiveTypeGroup>,
        visiting_groups: &mut collections::Vec<(u32, u32)>,
    ) -> bool {
        if let Some(group) = active_group {
            let lhs_local = idx1.checked_sub(group.lhs_start);
            let rhs_local = idx2.checked_sub(group.rhs_start);
            match (lhs_local, rhs_local) {
                (Some(lhs_offset), Some(rhs_offset))
                    if lhs_offset < group.len && rhs_offset < group.len =>
                {
                    return lhs_offset == rhs_offset;
                }
                (Some(lhs_offset), None) if lhs_offset < group.len => return false,
                (None, Some(rhs_offset)) if rhs_offset < group.len => return false,
                _ => {}
            }
        }

        self.types_equivalent_inner(idx1, idx2, visiting_groups)
    }

    fn group_info(&self, idx: u32) -> Option<TypeGroupInfo> {
        let def = self.get(idx)?;
        let Some(rec_group) = def.rec_group else {
            return Some(TypeGroupInfo {
                start: idx,
                len: 1,
                offset: 0,
            });
        };

        let mut start = idx;
        while start > 0
            && self
                .get(start - 1)
                .is_some_and(|prev| prev.rec_group == Some(rec_group))
        {
            start -= 1;
        }

        let mut end = idx + 1;
        while self
            .get(end)
            .is_some_and(|next| next.rec_group == Some(rec_group))
        {
            end += 1;
        }

        Some(TypeGroupInfo {
            start,
            len: end - start,
            offset: idx - start,
        })
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

pub fn concrete_type_matches_cross_context(
    actual_type_ctx: &TypeContext,
    actual_type_idx: u32,
    expected_type_ctx: &TypeContext,
    expected_type_idx: u32,
) -> bool {
    let mut stack = collections::vec![actual_type_idx];
    let mut visited = collections::Vec::new();

    while let Some(current_idx) = stack.pop() {
        if visited.contains(&current_idx) {
            continue;
        }
        visited.push(current_idx);

        let mut visiting_groups = collections::Vec::new();
        if types_equivalent_cross_context_inner(
            actual_type_ctx,
            current_idx,
            expected_type_ctx,
            expected_type_idx,
            &mut visiting_groups,
        ) {
            return true;
        }

        let Some(def_type) = actual_type_ctx.get(current_idx) else {
            continue;
        };
        for &super_idx in &def_type.supertypes {
            stack.push(super_idx);
        }
    }

    false
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

fn types_equivalent_cross_context_inner(
    lhs_ctx: &TypeContext,
    lhs_idx: u32,
    rhs_ctx: &TypeContext,
    rhs_idx: u32,
    visiting_groups: &mut collections::Vec<(u32, u32)>,
) -> bool {
    let Some(lhs_group) = lhs_ctx.group_info(lhs_idx) else {
        return false;
    };
    let Some(rhs_group) = rhs_ctx.group_info(rhs_idx) else {
        return false;
    };

    if lhs_group.len != rhs_group.len || lhs_group.offset != rhs_group.offset {
        return false;
    }

    groups_equivalent_cross_context(lhs_ctx, lhs_group, rhs_ctx, rhs_group, visiting_groups)
}

fn groups_equivalent_cross_context(
    lhs_ctx: &TypeContext,
    lhs_group: TypeGroupInfo,
    rhs_ctx: &TypeContext,
    rhs_group: TypeGroupInfo,
    visiting_groups: &mut collections::Vec<(u32, u32)>,
) -> bool {
    let pair = (lhs_group.start, rhs_group.start);
    if visiting_groups.contains(&pair) {
        return true;
    }

    visiting_groups.push(pair);
    let active_group = ActiveTypeGroup {
        lhs_start: lhs_group.start,
        rhs_start: rhs_group.start,
        len: lhs_group.len,
    };
    let result = (0..lhs_group.len).all(|offset| {
        def_types_equivalent_cross_context(
            lhs_ctx,
            lhs_group.start + offset,
            rhs_ctx,
            rhs_group.start + offset,
            Some(active_group),
            visiting_groups,
        )
    });
    if let Some(pos) = visiting_groups.iter().position(|entry| *entry == pair) {
        visiting_groups.swap_remove(pos);
    }
    result
}

fn def_types_equivalent_cross_context(
    lhs_ctx: &TypeContext,
    lhs_idx: u32,
    rhs_ctx: &TypeContext,
    rhs_idx: u32,
    active_group: Option<ActiveTypeGroup>,
    visiting_groups: &mut collections::Vec<(u32, u32)>,
) -> bool {
    let Some(lhs) = lhs_ctx.get(lhs_idx) else {
        return false;
    };
    let Some(rhs) = rhs_ctx.get(rhs_idx) else {
        return false;
    };

    if lhs.is_final != rhs.is_final || lhs.supertypes.len() != rhs.supertypes.len() {
        return false;
    }

    for (&lhs_super, &rhs_super) in lhs.supertypes.iter().zip(rhs.supertypes.iter()) {
        if !type_indices_equivalent_cross_context(
            lhs_ctx,
            lhs_super,
            rhs_ctx,
            rhs_super,
            active_group,
            visiting_groups,
        ) {
            return false;
        }
    }

    match (&lhs.composite, &rhs.composite) {
        (CompositeType::Func(lhs_func), CompositeType::Func(rhs_func)) => {
            if lhs_func.params().len() != rhs_func.params().len()
                || lhs_func.results().len() != rhs_func.results().len()
            {
                return false;
            }

            for (lhs_ty, rhs_ty) in lhs_func.params().iter().zip(rhs_func.params().iter()) {
                if !value_types_equivalent_cross_context(
                    lhs_ctx,
                    lhs_ty,
                    rhs_ctx,
                    rhs_ty,
                    active_group,
                    visiting_groups,
                ) {
                    return false;
                }
            }
            for (lhs_ty, rhs_ty) in lhs_func.results().iter().zip(rhs_func.results().iter()) {
                if !value_types_equivalent_cross_context(
                    lhs_ctx,
                    lhs_ty,
                    rhs_ctx,
                    rhs_ty,
                    active_group,
                    visiting_groups,
                ) {
                    return false;
                }
            }
            true
        }
        (CompositeType::Struct(lhs_struct), CompositeType::Struct(rhs_struct)) => {
            if lhs_struct.fields.len() != rhs_struct.fields.len() {
                return false;
            }

            for (lhs_field, rhs_field) in lhs_struct.fields.iter().zip(rhs_struct.fields.iter()) {
                if lhs_field.mutable != rhs_field.mutable
                    || !storage_types_equivalent_cross_context(
                        lhs_ctx,
                        &lhs_field.storage,
                        rhs_ctx,
                        &rhs_field.storage,
                        active_group,
                        visiting_groups,
                    )
                {
                    return false;
                }
            }

            true
        }
        (CompositeType::Array(lhs_array), CompositeType::Array(rhs_array)) => {
            lhs_array.element.mutable == rhs_array.element.mutable
                && storage_types_equivalent_cross_context(
                    lhs_ctx,
                    &lhs_array.element.storage,
                    rhs_ctx,
                    &rhs_array.element.storage,
                    active_group,
                    visiting_groups,
                )
        }
        _ => false,
    }
}

fn storage_types_equivalent_cross_context(
    lhs_ctx: &TypeContext,
    lhs: &crate::module::type_defs::StorageType,
    rhs_ctx: &TypeContext,
    rhs: &crate::module::type_defs::StorageType,
    active_group: Option<ActiveTypeGroup>,
    visiting_groups: &mut collections::Vec<(u32, u32)>,
) -> bool {
    use crate::module::type_defs::{PackedType, StorageType};

    match (lhs, rhs) {
        (StorageType::Val(lhs_ty), StorageType::Val(rhs_ty)) => {
            value_types_equivalent_cross_context(
                lhs_ctx,
                lhs_ty,
                rhs_ctx,
                rhs_ty,
                active_group,
                visiting_groups,
            )
        }
        (StorageType::Packed(PackedType::I8), StorageType::Packed(PackedType::I8)) => true,
        (StorageType::Packed(PackedType::I16), StorageType::Packed(PackedType::I16)) => true,
        _ => false,
    }
}

fn value_types_equivalent_cross_context(
    lhs_ctx: &TypeContext,
    lhs: &ValueType,
    rhs_ctx: &TypeContext,
    rhs: &ValueType,
    active_group: Option<ActiveTypeGroup>,
    visiting_groups: &mut collections::Vec<(u32, u32)>,
) -> bool {
    match (lhs, rhs) {
        (ValueType::I32, ValueType::I32)
        | (ValueType::I64, ValueType::I64)
        | (ValueType::F32, ValueType::F32)
        | (ValueType::F64, ValueType::F64)
        | (ValueType::V128, ValueType::V128)
        | (ValueType::Unknown, ValueType::Unknown) => true,
        (ValueType::Ref(lhs_ref), ValueType::Ref(rhs_ref)) => {
            lhs_ref.nullable == rhs_ref.nullable
                && match (&lhs_ref.heap_type, &rhs_ref.heap_type) {
                    (HeapType::Abstract(lhs_abs), HeapType::Abstract(rhs_abs)) => {
                        lhs_abs == rhs_abs
                    }
                    (HeapType::Concrete(lhs_idx), HeapType::Concrete(rhs_idx)) => {
                        type_indices_equivalent_cross_context(
                            lhs_ctx,
                            *lhs_idx,
                            rhs_ctx,
                            *rhs_idx,
                            active_group,
                            visiting_groups,
                        )
                    }
                    _ => false,
                }
        }
        _ => false,
    }
}

fn type_indices_equivalent_cross_context(
    lhs_ctx: &TypeContext,
    lhs_idx: u32,
    rhs_ctx: &TypeContext,
    rhs_idx: u32,
    active_group: Option<ActiveTypeGroup>,
    visiting_groups: &mut collections::Vec<(u32, u32)>,
) -> bool {
    if let Some(group) = active_group {
        let lhs_local = lhs_idx.checked_sub(group.lhs_start);
        let rhs_local = rhs_idx.checked_sub(group.rhs_start);
        match (lhs_local, rhs_local) {
            (Some(lhs_offset), Some(rhs_offset))
                if lhs_offset < group.len && rhs_offset < group.len =>
            {
                return lhs_offset == rhs_offset;
            }
            (Some(lhs_offset), None) if lhs_offset < group.len => return false,
            (None, Some(rhs_offset)) if rhs_offset < group.len => return false,
            _ => {}
        }
    }

    types_equivalent_cross_context_inner(lhs_ctx, lhs_idx, rhs_ctx, rhs_idx, visiting_groups)
}

#[cfg(test)]
mod tests {
    use super::TypeContext;
    use crate::{
        collections,
        module::type_defs::{CompositeType, DefType, FunctionType},
        value_type::{HeapType, RefType, ValueType},
    };
    use tracked_alloc::rc::Rc;

    fn func_type(
        params: collections::Vec<ValueType>,
        results: collections::Vec<ValueType>,
    ) -> Rc<DefType> {
        Rc::new(DefType {
            composite: CompositeType::Func(Rc::new(FunctionType::new(params, results))),
            supertypes: collections::vec![],
            is_final: true,
            rec_group: None,
        })
    }

    fn rec_func_type(
        rec_group: u32,
        params: collections::Vec<ValueType>,
        results: collections::Vec<ValueType>,
    ) -> Rc<DefType> {
        Rc::new(DefType {
            composite: CompositeType::Func(Rc::new(FunctionType::new(params, results))),
            supertypes: collections::vec![],
            is_final: true,
            rec_group: Some(rec_group),
        })
    }

    #[test]
    fn singleton_function_types_remain_structurally_equivalent() {
        let ctx = TypeContext::new(collections::vec![
            func_type(
                collections::vec![ValueType::I32],
                collections::vec![ValueType::I64]
            ),
            func_type(
                collections::vec![ValueType::I32],
                collections::vec![ValueType::I64]
            ),
        ]);

        assert!(ctx.types_equivalent(0, 1));
    }

    #[test]
    fn larger_recursive_group_is_not_equivalent_to_standalone_function() {
        let ctx = TypeContext::new(collections::vec![
            rec_func_type(0, collections::vec![], collections::vec![]),
            rec_func_type(0, collections::vec![], collections::vec![]),
            func_type(collections::vec![], collections::vec![]),
        ]);

        assert!(!ctx.types_equivalent(0, 2));
        assert!(!ctx.types_equivalent(1, 2));
    }

    #[test]
    fn isorecursive_groups_match_only_at_corresponding_offsets() {
        let ctx = TypeContext::new(collections::vec![
            rec_func_type(
                0,
                collections::vec![ValueType::Ref(RefType::non_nullable_concrete(1))],
                collections::vec![],
            ),
            rec_func_type(
                0,
                collections::vec![ValueType::Ref(RefType::non_nullable_concrete(0))],
                collections::vec![],
            ),
            rec_func_type(
                1,
                collections::vec![ValueType::Ref(RefType::non_nullable_concrete(3))],
                collections::vec![],
            ),
            rec_func_type(
                1,
                collections::vec![ValueType::Ref(RefType::non_nullable_concrete(2))],
                collections::vec![],
            ),
        ]);

        assert!(ctx.types_equivalent(0, 2));
        assert!(ctx.types_equivalent(1, 3));
        assert!(!ctx.types_equivalent(0, 3));
    }

    #[test]
    fn recursive_group_members_must_keep_internal_reference_shape() {
        let ctx = TypeContext::new(collections::vec![
            rec_func_type(
                0,
                collections::vec![ValueType::Ref(RefType::new(false, HeapType::Concrete(0),))],
                collections::vec![],
            ),
            rec_func_type(
                1,
                collections::vec![ValueType::Ref(RefType::new(false, HeapType::Concrete(1),))],
                collections::vec![],
            ),
            rec_func_type(
                2,
                collections::vec![ValueType::Ref(RefType::new(false, HeapType::Concrete(0),))],
                collections::vec![],
            ),
        ]);

        assert!(ctx.types_equivalent(0, 1));
        assert!(!ctx.types_equivalent(0, 2));
    }

    #[test]
    fn cross_context_recursive_types_require_matching_group_shape() {
        let actual = TypeContext::new(collections::vec![
            rec_func_type(0, collections::vec![], collections::vec![]),
            Rc::new(DefType {
                composite: CompositeType::Struct(crate::module::type_defs::StructType::new(
                    collections::vec![],
                )),
                supertypes: collections::vec![],
                is_final: true,
                rec_group: Some(0),
            }),
        ]);
        let expected = TypeContext::new(collections::vec![
            Rc::new(DefType {
                composite: CompositeType::Struct(crate::module::type_defs::StructType::new(
                    collections::vec![],
                )),
                supertypes: collections::vec![],
                is_final: true,
                rec_group: Some(1),
            }),
            rec_func_type(1, collections::vec![], collections::vec![]),
        ]);

        assert!(!super::concrete_type_matches_cross_context(
            &actual, 0, &expected, 1
        ));
    }
}
