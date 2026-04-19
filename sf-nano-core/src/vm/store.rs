//! Store/runtime state shared by the VM backends.
//!
//! This layer should stay backend-agnostic.

use crate::collections;
use tracked_alloc::rc::Rc;

use crate::vm::entities::{FunctionInst, GlobalInst, MemInst, ModuleInst, TableInst};
use crate::vm::gc_heap::{GcHeap, GcRef};
use crate::vm::value::RefHandle;
use core::cell::RefCell;

#[derive(Clone, Copy, Debug)]
pub(crate) struct FunctionRegistryEntry {
    pub(crate) store: *mut Store,
    pub(crate) local_index: usize,
}

pub(crate) type SharedFunctionRegistry = Rc<RefCell<collections::Vec<FunctionRegistryEntry>>>;
pub(crate) type SharedRefRegistry = Rc<RefCell<collections::Vec<RefRegistryEntry>>>;
#[cfg(sf_has_simd)]
#[derive(Clone)]
pub(crate) struct SharedSimdRegistry(Rc<RefCell<collections::Vec<[u8; 16]>>>);

#[cfg(sf_has_simd)]
impl SharedSimdRegistry {
    #[inline]
    pub(crate) fn new() -> Self {
        Self(Rc::new(RefCell::new(collections::vec![[0; 16]])))
    }

    #[inline]
    /// Interns a SIMD lane payload into the shared registry.
    ///
    /// This registry is append-only for the lifetime of its owners, and dedup
    /// is currently a linear scan. That keeps bring-up simple, but it means
    /// unique SIMD values grow memory monotonically and repeated inserts become
    /// O(n). TODO: replace this with a reclaimed/deduplicated representation
    /// once the native SIMD backend surface settles.
    pub(crate) fn intern(&self, value: [u8; 16]) -> u64 {
        if let Some((index, _)) = self
            .0
            .borrow()
            .iter()
            .enumerate()
            .find(|(_, existing)| **existing == value)
        {
            return index as u64;
        }
        let mut registry = self.0.borrow_mut();
        let index = registry.len();
        registry.push(value);
        index as u64
    }

    #[inline]
    pub(crate) fn get(&self, raw: u64) -> Option<[u8; 16]> {
        self.0.borrow().get(raw as usize).copied()
    }
}

#[cfg(sf_has_simd)]
impl Default for SharedSimdRegistry {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum RefRegistryEntry {
    I31(i32),
    Gc { store: *mut Store, gc_ref: GcRef },
}

#[derive(Clone)]
pub struct LinkRegistry {
    functions: SharedFunctionRegistry,
    refs: SharedRefRegistry,
    #[cfg(sf_has_simd)]
    simd: SharedSimdRegistry,
}

impl LinkRegistry {
    #[inline]
    pub fn new() -> Self {
        Self {
            functions: Rc::new(RefCell::new(collections::Vec::new())),
            refs: Rc::new(RefCell::new(collections::Vec::new())),
            #[cfg(sf_has_simd)]
            simd: SharedSimdRegistry::new(),
        }
    }

    #[inline]
    pub(crate) fn function_registry_shared(&self) -> SharedFunctionRegistry {
        Rc::clone(&self.functions)
    }

    #[inline]
    pub(crate) fn ref_registry_shared(&self) -> SharedRefRegistry {
        Rc::clone(&self.refs)
    }

    #[inline]
    #[cfg(sf_has_simd)]
    pub(crate) fn simd_registry_shared(&self) -> SharedSimdRegistry {
        self.simd.clone()
    }

    #[inline]
    #[cfg(sf_has_simd)]
    pub(crate) fn from_shared(
        functions: SharedFunctionRegistry,
        refs: SharedRefRegistry,
        simd: SharedSimdRegistry,
    ) -> Self {
        Self {
            functions,
            refs,
            simd,
        }
    }

    #[inline]
    #[cfg(not(sf_has_simd))]
    pub(crate) fn from_shared(functions: SharedFunctionRegistry, refs: SharedRefRegistry) -> Self {
        Self { functions, refs }
    }
}

pub struct Store {
    module: ModuleInst,
    function_registry: SharedFunctionRegistry,
    ref_registry: SharedRefRegistry,
    #[cfg(sf_has_simd)]
    simd_registry: SharedSimdRegistry,
    gc_heap: Rc<RefCell<GcHeap>>,
}

impl Store {
    #[inline]
    pub fn new(module: ModuleInst) -> Self {
        Self::new_with_registries(
            module,
            Rc::new(RefCell::new(collections::Vec::new())),
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
        }
    }

    #[inline]
    pub fn module(&self) -> &ModuleInst {
        &self.module
    }

    #[inline]
    pub fn module_mut(&mut self) -> &mut ModuleInst {
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
        &mut self.module.tables[idx]
    }

    #[inline]
    pub fn memory(&self, idx: usize) -> &MemInst {
        &self.module.memories[idx]
    }

    #[inline]
    pub fn memory_mut(&mut self, idx: usize) -> &mut MemInst {
        &mut self.module.memories[idx]
    }

    #[inline]
    pub fn global(&self, idx: usize) -> &GlobalInst {
        &self.module.globals[idx]
    }

    #[inline]
    pub fn global_mut(&mut self, idx: usize) -> &mut GlobalInst {
        &mut self.module.globals[idx]
    }

    #[inline]
    pub(crate) fn clone_function_registry(&self) -> SharedFunctionRegistry {
        Rc::clone(&self.function_registry)
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
        self.function_registry
            .borrow()
            .get(handle.payload())
            .copied()
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
        self.ref_registry.borrow().get(idx).copied()
    }

    pub(crate) fn intern_v128(&self, value: [u8; 16]) -> u64 {
        self.simd_registry.intern(value)
    }

    #[inline]
    #[cfg(sf_has_simd)]
    pub(crate) fn get_v128(&self, raw: u64) -> Option<[u8; 16]> {
        self.simd_registry.get(raw)
    }
}
