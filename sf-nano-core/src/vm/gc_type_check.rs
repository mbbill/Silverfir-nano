//! Runtime GC/reference type checks for `ref.cast` and `ref.test`.

use crate::{
    error::WasmError,
    module::type_context::concrete_type_matches_cross_context,
    value_type::{AbstractHeapType, HeapType},
    vm::{store::RefRegistryEntry, store::Store, value::RefHandle},
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
            Some(RefRegistryEntry::Gc { store, gc_ref }) => {
                let origin_store = unsafe { store.as_ref() }
                    .ok_or_else(|| WasmError::internal("GC ref points to missing store".into()))?;
                check_gc_ref_type(origin_store, gc_ref, heap_type, current_store)
            }
            Some(RefRegistryEntry::Exn(_)) => Ok(matches!(
                heap_type,
                HeapType::Abstract(AbstractHeapType::Exn)
            )),
            // Deliberately matches NOTHING, `func` included. Such a handle
            // exists only where an embedder shares a registry without
            // installing a `FuncRefHost`, so nothing can call it or say which
            // function it names -- the spec answer for `ref.test (ref func)`
            // would be `true`, and reporting that would promise a callability
            // this build cannot deliver. Install a resolver and the reference
            // is a published funcref instead, which answers normally.
            #[cfg(sf_interp)]
            Some(RefRegistryEntry::OpaqueInterpFunc) => Ok(false),
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
            Ok(concrete_type_matches_cross_context(
                &origin_store.module().types,
                source_type_idx,
                &current_store.module().types,
                *target_idx,
            ))
        }
        _ => Ok(false),
    }
}
