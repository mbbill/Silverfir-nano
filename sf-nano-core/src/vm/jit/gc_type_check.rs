//! Runtime GC/reference type checks for `ref.cast` and `ref.test`.

use crate::{
    error::WasmError,
    module::type_context::concrete_type_matches_cross_context,
    value_type::{AbstractHeapType, HeapType},
    vm::{
        jit::{store::Store, value_encoding::absolutize},
        link::RefRegistryEntry,
        value::RefHandle,
    },
};

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
            Some(RefRegistryEntry::Gc { owner, gc_ref }) => {
                if owner == current_store.instance_handle().self_id() {
                    return check_gc_ref_type(current_store, gc_ref, heap_type, current_store);
                }
                let owner = current_store
                    .instance_handle()
                    .checkout(owner)
                    .ok_or_else(|| {
                        WasmError::internal("GC ref points to missing instance".into())
                    })?;
                let origin_store = owner.jit().ok_or_else(|| {
                    WasmError::internal("GC ref points to a non-JIT instance".into())
                })?;
                check_gc_ref_type(origin_store, gc_ref, heap_type, current_store)
            }
            Some(RefRegistryEntry::Exn(_)) => Ok(matches!(
                heap_type,
                HeapType::Abstract(AbstractHeapType::Exn)
            )),
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
    gc_ref: crate::vm::jit::gc_heap::GcRef,
    heap_type: &HeapType,
    current_store: &Store,
) -> Result<bool, WasmError> {
    match heap_type {
        HeapType::Concrete(target_idx) => {
            let src_type_idx = origin_store.gc_heap().borrow().type_idx(gc_ref)?;
            Ok(concrete_type_matches_cross_context(
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
            let absolute = absolutize(current_store, ref_handle);
            let entry = current_store
                .function_entry_for_handle(absolute)
                .ok_or_else(|| WasmError::invalid("invalid function reference"))?;
            if entry.owner == current_store.instance_handle().self_id() {
                return check_func_type_index(
                    current_store,
                    entry.local_index,
                    current_store,
                    *target_idx,
                );
            }
            let owner = current_store
                .instance_handle()
                .checkout(entry.owner)
                .ok_or_else(|| {
                    WasmError::internal("function ref points to missing instance".into())
                })?;
            let origin_store = owner.jit().ok_or_else(|| {
                WasmError::internal("function ref points to a non-JIT instance".into())
            })?;
            check_func_type_index(origin_store, entry.local_index, current_store, *target_idx)
        }
        _ => Ok(false),
    }
}

fn check_func_type_index(
    origin_store: &Store,
    local_index: u32,
    current_store: &Store,
    target_idx: u32,
) -> Result<bool, WasmError> {
    let source_func = origin_store
        .module()
        .functions
        .get(local_index as usize)
        .ok_or_else(|| WasmError::internal("function ref index is out of range".into()))?;
    let source_type_idx = source_func.type_index();
    if source_type_idx == u32::MAX {
        return Ok(false);
    }
    Ok(concrete_type_matches_cross_context(
        &origin_store.module().types,
        source_type_idx,
        &current_store.module().types,
        target_idx,
    ))
}
