//! The JIT engine's runtime world.
//!
//! A `Store` owns one instantiated module's entities plus the caches native
//! code addresses directly. It is JIT-owned by construction: the interpreter
//! keeps its own flat runtime state and only ever *reads* a foreign `Store`
//! through a registry entry when both engines are compiled in.
//!
//! The revision counters exist for exactly one consumer: the native
//! context's cached raw views (`vm::jit::runtime::context`), which must be
//! refreshed whenever an entity the generated code points into may have
//! moved.

use crate::collections;
use crate::config::Config;
use tracked_alloc::rc::Rc;

use crate::vm::entities::{FunctionInst, GlobalInst, MemInst, TableInst};
use crate::vm::jit::entities::ModuleInst;
use crate::vm::jit::gc_heap::{GcHeap, GcRef};
#[cfg(sf_has_simd)]
use crate::vm::link::SharedSimdRegistry;
use crate::vm::link::{
    alloc_exn_in, FunctionRegistryEntry, RefRegistryEntry, SharedFunctionRegistry,
    SharedRefRegistry,
};
use crate::vm::tag::TagHandle;
use crate::vm::value::{RefHandle, Value};
use core::cell::RefCell;

pub struct Store {
    module: ModuleInst,
    function_registry: SharedFunctionRegistry,
    ref_registry: SharedRefRegistry,
    #[cfg(sf_has_simd)]
    simd_registry: SharedSimdRegistry,
    gc_heap: Rc<RefCell<GcHeap>>,
    module_revision: u64,
    native_context_cache: Option<crate::vm::jit::runtime::context::NativeContextBox>,
    #[cfg(sf_has_guard_pages)]
    native_stack_cache: Option<crate::vm::jit::runtime::guard_pages::GuardPageStack>,
    #[cfg(not(sf_has_guard_pages))]
    native_stack_cache: Option<collections::Vec<u64>>,
}

impl Store {
    #[inline]
    pub fn new(module: ModuleInst) -> Self {
        Self::new_with_registries(
            module,
            SharedFunctionRegistry::new(),
            Rc::new(RefCell::new(collections::Vec::new())),
            #[cfg(sf_has_simd)]
            SharedSimdRegistry::new(),
        )
    }

    #[inline]
    pub(crate) fn new_with_registries(
        module: ModuleInst,
        function_registry: SharedFunctionRegistry,
        ref_registry: SharedRefRegistry,
        #[cfg(sf_has_simd)] simd_registry: SharedSimdRegistry,
    ) -> Self {
        Self {
            module,
            function_registry,
            ref_registry,
            #[cfg(sf_has_simd)]
            simd_registry,
            gc_heap: Rc::new(RefCell::new(GcHeap::new())),
            module_revision: 0,
            native_context_cache: None,
            native_stack_cache: None,
        }
    }

    /// The engine configuration this store's module was created with.
    #[inline]
    pub(crate) fn config(&self) -> &Config {
        self.module.config()
    }

    #[inline]
    pub fn module(&self) -> &ModuleInst {
        &self.module
    }

    #[inline]
    pub fn module_mut(&mut self) -> &mut ModuleInst {
        self.module_revision = self.module_revision.wrapping_add(1);
        &mut self.module
    }

    #[inline]
    pub fn function(&self, idx: usize) -> &FunctionInst {
        &self.module.functions[idx]
    }

    #[inline]
    pub fn table(&self, idx: usize) -> &TableInst {
        &self.module.tables[idx]
    }

    #[inline]
    pub fn table_mut(&mut self, idx: usize) -> &mut TableInst {
        self.module_revision = self.module_revision.wrapping_add(1);
        &mut self.module.tables[idx]
    }

    #[inline]
    pub fn memory(&self, idx: usize) -> &MemInst {
        &self.module.memories[idx]
    }

    #[inline]
    pub fn memory_mut(&mut self, idx: usize) -> &mut MemInst {
        self.module_revision = self.module_revision.wrapping_add(1);
        &mut self.module.memories[idx]
    }

    #[inline]
    pub fn global(&self, idx: usize) -> &GlobalInst {
        &self.module.globals[idx]
    }

    #[inline]
    pub fn global_mut(&mut self, idx: usize) -> &mut GlobalInst {
        self.module_revision = self.module_revision.wrapping_add(1);
        &mut self.module.globals[idx]
    }

    #[inline]
    pub(crate) fn clone_function_registry(&self) -> SharedFunctionRegistry {
        self.function_registry.clone()
    }

    #[inline]
    pub(crate) fn module_revision(&self) -> u64 {
        self.module_revision
    }

    #[inline]
    pub(crate) fn function_registry_revision(&self) -> u64 {
        self.function_registry.revision()
    }

    #[inline]
    pub(crate) fn take_native_context_cache(
        &mut self,
    ) -> Option<crate::vm::jit::runtime::context::NativeContextBox> {
        self.native_context_cache.take()
    }

    #[inline]
    pub(crate) fn cache_native_context(
        &mut self,
        context: crate::vm::jit::runtime::context::NativeContextBox,
    ) {
        if self.native_context_cache.is_none() {
            self.native_context_cache = Some(context);
        }
    }

    #[cfg(sf_has_guard_pages)]
    #[inline]
    pub(crate) fn take_native_stack_cache(
        &mut self,
    ) -> Option<crate::vm::jit::runtime::guard_pages::GuardPageStack> {
        self.native_stack_cache.take()
    }

    #[cfg(sf_has_guard_pages)]
    #[inline]
    pub(crate) fn cache_native_stack(
        &mut self,
        stack: crate::vm::jit::runtime::guard_pages::GuardPageStack,
    ) {
        if self.native_stack_cache.is_none() {
            self.native_stack_cache = Some(stack);
        }
    }

    #[cfg(not(sf_has_guard_pages))]
    #[inline]
    pub(crate) fn take_native_stack_cache(&mut self) -> Option<collections::Vec<u64>> {
        self.native_stack_cache.take()
    }

    #[cfg(not(sf_has_guard_pages))]
    #[inline]
    pub(crate) fn cache_native_stack(&mut self, stack: collections::Vec<u64>) {
        if self.native_stack_cache.is_none() {
            self.native_stack_cache = Some(stack);
        }
    }

    #[inline]
    pub(crate) fn clone_ref_registry(&self) -> SharedRefRegistry {
        Rc::clone(&self.ref_registry)
    }

    #[inline]
    #[cfg(sf_has_simd)]
    pub(crate) fn clone_simd_registry(&self) -> SharedSimdRegistry {
        self.simd_registry.clone()
    }

    #[inline]
    pub(crate) fn gc_heap(&self) -> &Rc<RefCell<GcHeap>> {
        &self.gc_heap
    }

    /// Allocate a fresh exception object carrying the given tag identity and
    /// payload. The object is registry-owned, so the returned handle stays
    /// valid after this store is dropped.
    pub(crate) fn alloc_exn(
        &mut self,
        tag: TagHandle,
        fields: collections::Vec<Value>,
    ) -> RefHandle {
        alloc_exn_in(&self.ref_registry, tag, fields)
    }

    pub(crate) fn register_local_function(&mut self, local_index: usize) -> RefHandle {
        let self_ptr = self as *mut Store;
        let handle = {
            let mut registry = self.function_registry.borrow_mut();
            let handle = RefHandle::new(registry.len());
            registry.push(FunctionRegistryEntry {
                store: self_ptr,
                local_index,
            });
            handle
        };
        self.module.ensure_function_handle_capacity(local_index + 1);
        self.module.set_function_handle(local_index, handle);
        handle
    }

    pub(crate) fn function_entry_for_handle(
        &self,
        handle: RefHandle,
    ) -> Option<FunctionRegistryEntry> {
        if handle.is_null() || handle.is_special() {
            return None;
        }
        let entry = self
            .function_registry
            .borrow()
            .get(handle.payload())
            .copied()?;
        if entry.store.is_null() {
            return None;
        }
        Some(entry)
    }

    pub(crate) fn register_i31(&mut self, value: i32) -> RefHandle {
        if let Some((idx, _)) = self.ref_registry.borrow().iter().enumerate().find(
            |(_, entry)| matches!(entry, RefRegistryEntry::I31(existing) if *existing == value),
        ) {
            return RefHandle::from_pool_index(idx);
        }
        let idx = {
            let mut registry = self.ref_registry.borrow_mut();
            let idx = registry.len();
            registry.push(RefRegistryEntry::I31(value));
            idx
        };
        RefHandle::from_pool_index(idx)
    }

    pub(crate) fn register_gc_ref(&mut self, gc_ref: GcRef) -> RefHandle {
        let self_ptr = self as *mut Store;
        let idx = {
            let mut registry = self.ref_registry.borrow_mut();
            let idx = registry.len();
            registry.push(RefRegistryEntry::Gc {
                store: self_ptr,
                gc_ref,
            });
            idx
        };
        RefHandle::from_pool_index(idx)
    }

    pub(crate) fn ref_entry_for_handle(&self, handle: RefHandle) -> Option<RefRegistryEntry> {
        let idx = handle.pooled_index()?;
        self.ref_registry.borrow().get(idx).cloned()
    }

    #[cfg(sf_has_simd)]
    // Store a v128 payload out-of-line and return the 64-bit raw handle used by
    // the current frame/global storage ABI.
    pub(crate) fn intern_v128(&self, value: [u8; 16]) -> u64 {
        self.simd_registry.intern(value)
    }

    #[inline]
    #[cfg(sf_has_simd)]
    // Resolve a stored raw handle back into the concrete v128 payload.
    pub(crate) fn get_v128(&self, raw: u64) -> Option<[u8; 16]> {
        self.simd_registry.get(raw)
    }
}

impl Drop for Store {
    fn drop(&mut self) {
        let self_ptr = self as *mut Store;
        for entry in self.function_registry.borrow_mut().iter_mut() {
            if entry.store == self_ptr {
                entry.store = core::ptr::null_mut();
            }
        }
        // GC handles still point into their owner's arena. Poison that pointer
        // before the Store allocation disappears. Exception entries need no
        // cleanup: their `Rc` owns the object independently of this Store.
        for entry in self.ref_registry.borrow_mut().iter_mut() {
            match entry {
                RefRegistryEntry::Gc { store, .. } if *store == self_ptr => {
                    *store = core::ptr::null_mut();
                }
                _ => {}
            }
        }
    }
}

#[cfg(all(test, sf_interp))]
mod tests {
    use super::*;
    use crate::vm::link::LinkRegistry;

    fn store_with(registry: &LinkRegistry) -> alloc::boxed::Box<Store> {
        alloc::boxed::Box::new(Store::new_with_registries(
            ModuleInst::default(),
            registry.function_registry_shared(),
            registry.ref_registry_shared(),
            #[cfg(sf_has_simd)]
            registry.simd_registry_shared(),
        ))
    }

    #[test]
    fn shared_registry_resolves_store_owned_exceptions() {
        let registry = LinkRegistry::new();
        let mut store = store_with(&registry);
        let tag = TagHandle::mint_fresh();
        let fields = collections::vec![Value::I32(23)];
        let handle = store.alloc_exn(tag, fields.clone());

        let resolved = registry.resolve_exn(handle).expect("Store-owned exception");

        assert_eq!(resolved.tag, tag);
        assert_eq!(resolved.fields, fields);
    }

    #[test]
    fn store_owned_exception_survives_its_origin_store() {
        let registry = LinkRegistry::new();
        let (handle, tag, fields) = {
            let mut store = store_with(&registry);
            let tag = TagHandle::mint_fresh();
            let fields = collections::vec![Value::I32(41)];
            (store.alloc_exn(tag, fields.clone()), tag, fields)
        };

        let first = registry
            .resolve_exn(handle)
            .expect("registry must keep the exception alive");
        let second = registry
            .resolve_exn(handle)
            .expect("repeated resolution must retain the same object");
        assert!(Rc::ptr_eq(&first, &second));
        assert_eq!(first.tag, tag);
        assert_eq!(first.fields, fields);
    }

    #[test]
    fn exception_and_other_pooled_handles_do_not_alias() {
        let registry = LinkRegistry::new();
        let mut store = store_with(&registry);
        let i31 = store.register_i31(7);
        let exn = registry.alloc_exn(TagHandle::mint_fresh(), collections::vec![]);

        assert_ne!(i31, exn);
        assert!(registry.resolve_exn(i31).is_none());
        assert!(registry.resolve_exn(exn).is_some());
    }
}
