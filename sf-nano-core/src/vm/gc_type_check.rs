//! Runtime GC/reference type checks for `ref.cast` and `ref.test`.

use crate::{
    collections,
    error::WasmError,
    module::{
        type_context::TypeContext,
        type_defs::{CompositeType, DefType, StorageType},
    },
    value_type::{AbstractHeapType, HeapType, ValueType},
    vm::{store::RefRegistryEntry, store::Store, value::RefHandle},
};

const MAX_TYPE_EQ_DEPTH: usize = 64;

pub(crate) fn check_ref_type_match(
    ref_handle: RefHandle,
    heap_type: &HeapType,
    current_store: &Store,
) -> Result<bool, WasmError> {
    if ref_handle.is_null() {
        return Ok(false);
    }

    if ref_handle.is_extern() {
        return Ok(matches!(
            heap_type,
            HeapType::Abstract(AbstractHeapType::Extern)
        ));
    }

    if ref_handle.is_pooled() {
        return match current_store.ref_entry_for_handle(ref_handle) {
            Some(RefRegistryEntry::I31(_)) => Ok(matches!(
                heap_type,
                HeapType::Abstract(AbstractHeapType::I31)
                    | HeapType::Abstract(AbstractHeapType::Eq)
                    | HeapType::Abstract(AbstractHeapType::Any)
            )),
            Some(RefRegistryEntry::Gc { store, gc_ref }) => {
                let origin_store = unsafe { store.as_ref() }
                    .ok_or_else(|| WasmError::internal("GC ref points to missing store".into()))?;
                check_gc_ref_type(origin_store, gc_ref, heap_type, current_store)
            }
            None => Err(WasmError::invalid("invalid pooled reference")),
        };
    }

    if ref_handle.is_host() {
        return Ok(matches!(
            heap_type,
            HeapType::Abstract(AbstractHeapType::Any)
        ));
    }

    check_func_ref_type(ref_handle, heap_type, current_store)
}

fn check_gc_ref_type(
    origin_store: &Store,
    gc_ref: crate::vm::gc_heap::GcRef,
    heap_type: &HeapType,
    current_store: &Store,
) -> Result<bool, WasmError> {
    match heap_type {
        HeapType::Concrete(target_idx) => {
            let src_type_idx = origin_store.gc_heap().borrow().type_idx(gc_ref)?;
            Ok(concrete_type_matches(
                &origin_store.module().types,
                src_type_idx,
                &current_store.module().types,
                *target_idx,
            ))
        }
        HeapType::Abstract(AbstractHeapType::Struct) => {
            Ok(origin_store.gc_heap().borrow().is_struct(gc_ref))
        }
        HeapType::Abstract(AbstractHeapType::Array) => {
            Ok(origin_store.gc_heap().borrow().is_array(gc_ref))
        }
        HeapType::Abstract(AbstractHeapType::Eq | AbstractHeapType::Any) => Ok(true),
        _ => Ok(false),
    }
}

fn check_func_ref_type(
    ref_handle: RefHandle,
    heap_type: &HeapType,
    current_store: &Store,
) -> Result<bool, WasmError> {
    match heap_type {
        HeapType::Abstract(AbstractHeapType::Func) => Ok(true),
        HeapType::Concrete(target_idx) => {
            let entry = current_store
                .function_entry_for_handle(ref_handle)
                .ok_or_else(|| WasmError::invalid("invalid function reference"))?;
            let origin_store = unsafe { entry.store.as_ref() }.ok_or_else(|| {
                WasmError::internal("function ref points to missing store".into())
            })?;
            let source_func = origin_store.function(entry.local_index);
            let source_type_idx = source_func.type_index();
            if source_type_idx == u32::MAX {
                return Ok(false);
            }
            Ok(concrete_type_matches(
                &origin_store.module().types,
                source_type_idx,
                &current_store.module().types,
                *target_idx,
            ))
        }
        _ => Ok(false),
    }
}

fn concrete_type_matches(
    source_types: &TypeContext,
    source_idx: u32,
    target_types: &TypeContext,
    target_idx: u32,
) -> bool {
    let mut stack = collections::vec![source_idx];
    let mut visited = collections::Vec::new();

    while let Some(current_idx) = stack.pop() {
        if visited.contains(&current_idx) {
            continue;
        }
        visited.push(current_idx);

        let mut equiv_visiting = collections::Vec::new();
        if types_equivalent_cross_module(
            source_types,
            current_idx,
            target_types,
            target_idx,
            &mut equiv_visiting,
            0,
        ) {
            return true;
        }

        let Some(current_def) = source_types.get(current_idx) else {
            continue;
        };
        for &super_idx in &current_def.supertypes {
            stack.push(super_idx);
        }
    }

    false
}

fn types_equivalent_cross_module(
    lhs_types: &TypeContext,
    lhs_idx: u32,
    rhs_types: &TypeContext,
    rhs_idx: u32,
    visiting: &mut collections::Vec<(u32, u32)>,
    depth: usize,
) -> bool {
    if depth > MAX_TYPE_EQ_DEPTH {
        return false;
    }

    let pair = (lhs_idx, rhs_idx);
    if visiting.contains(&pair) {
        return true;
    }

    let Some(lhs) = lhs_types.get(lhs_idx) else {
        return false;
    };
    let Some(rhs) = rhs_types.get(rhs_idx) else {
        return false;
    };

    visiting.push(pair);
    let result =
        def_types_equivalent_cross_module(lhs, lhs_types, rhs, rhs_types, visiting, depth + 1);
    if let Some(pos) = visiting.iter().position(|entry| *entry == pair) {
        visiting.swap_remove(pos);
    }
    result
}

fn def_types_equivalent_cross_module(
    lhs: &DefType,
    lhs_types: &TypeContext,
    rhs: &DefType,
    rhs_types: &TypeContext,
    visiting: &mut collections::Vec<(u32, u32)>,
    depth: usize,
) -> bool {
    match (&lhs.composite, &rhs.composite) {
        (CompositeType::Func(lhs_func), CompositeType::Func(rhs_func)) => {
            if lhs_func.params().len() != rhs_func.params().len()
                || lhs_func.results().len() != rhs_func.results().len()
            {
                return false;
            }

            for (lhs_ty, rhs_ty) in lhs_func.params().iter().zip(rhs_func.params().iter()) {
                if !value_types_equivalent_cross_module(
                    lhs_ty, lhs_types, rhs_ty, rhs_types, visiting, depth,
                ) {
                    return false;
                }
            }
            for (lhs_ty, rhs_ty) in lhs_func.results().iter().zip(rhs_func.results().iter()) {
                if !value_types_equivalent_cross_module(
                    lhs_ty, lhs_types, rhs_ty, rhs_types, visiting, depth,
                ) {
                    return false;
                }
            }
            true
        }
        (CompositeType::Struct(lhs_struct), CompositeType::Struct(rhs_struct)) => {
            if lhs_struct.fields.len() != rhs_struct.fields.len()
                || lhs.supertypes.len() != rhs.supertypes.len()
            {
                return false;
            }

            for (lhs_field, rhs_field) in lhs_struct.fields.iter().zip(rhs_struct.fields.iter()) {
                if lhs_field.mutable != rhs_field.mutable
                    || !storage_types_equivalent_cross_module(
                        &lhs_field.storage,
                        lhs_types,
                        &rhs_field.storage,
                        rhs_types,
                        visiting,
                        depth,
                    )
                {
                    return false;
                }
            }

            for (&lhs_super, &rhs_super) in lhs.supertypes.iter().zip(rhs.supertypes.iter()) {
                if !types_equivalent_cross_module(
                    lhs_types, lhs_super, rhs_types, rhs_super, visiting, depth,
                ) {
                    return false;
                }
            }

            true
        }
        (CompositeType::Array(lhs_array), CompositeType::Array(rhs_array)) => {
            lhs_array.element.mutable == rhs_array.element.mutable
                && lhs.supertypes.len() == rhs.supertypes.len()
                && storage_types_equivalent_cross_module(
                    &lhs_array.element.storage,
                    lhs_types,
                    &rhs_array.element.storage,
                    rhs_types,
                    visiting,
                    depth,
                )
                && lhs.supertypes.iter().zip(rhs.supertypes.iter()).all(
                    |(&lhs_super, &rhs_super)| {
                        types_equivalent_cross_module(
                            lhs_types, lhs_super, rhs_types, rhs_super, visiting, depth,
                        )
                    },
                )
        }
        _ => false,
    }
}

fn storage_types_equivalent_cross_module(
    lhs: &StorageType,
    lhs_types: &TypeContext,
    rhs: &StorageType,
    rhs_types: &TypeContext,
    visiting: &mut collections::Vec<(u32, u32)>,
    depth: usize,
) -> bool {
    use crate::module::type_defs::PackedType;

    match (lhs, rhs) {
        (StorageType::Val(lhs_ty), StorageType::Val(rhs_ty)) => {
            value_types_equivalent_cross_module(
                lhs_ty, lhs_types, rhs_ty, rhs_types, visiting, depth,
            )
        }
        (StorageType::Packed(PackedType::I8), StorageType::Packed(PackedType::I8)) => true,
        (StorageType::Packed(PackedType::I16), StorageType::Packed(PackedType::I16)) => true,
        _ => false,
    }
}

fn value_types_equivalent_cross_module(
    lhs: &ValueType,
    lhs_types: &TypeContext,
    rhs: &ValueType,
    rhs_types: &TypeContext,
    visiting: &mut collections::Vec<(u32, u32)>,
    depth: usize,
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
                        types_equivalent_cross_module(
                            lhs_types, *lhs_idx, rhs_types, *rhs_idx, visiting, depth,
                        )
                    }
                    _ => false,
                }
        }
        _ => false,
    }
}
