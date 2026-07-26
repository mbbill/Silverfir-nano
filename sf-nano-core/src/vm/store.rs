//! Store/runtime state shared by the VM backends.
//!
//! This layer should stay backend-agnostic.

use crate::collections;
use crate::config::Config;
use tracked_alloc::rc::Rc;

use crate::vm::entities::{FunctionInst, GlobalInst, MemInst, ModuleInst, TableInst};
use crate::vm::exn_heap::{ExnHeap, ExnInstance, ExnRef};
use crate::vm::gc_heap::{GcHeap, GcRef};
#[cfg(sf_interp)]
use crate::vm::tag::TagHandle;
use crate::vm::value::RefHandle;
#[cfg(sf_interp)]
use crate::vm::value::Value;
use core::cell::{Cell, Ref, RefCell, RefMut};

#[derive(Clone, Copy, Debug)]
pub(crate) struct FunctionRegistryEntry {
    pub(crate) store: *mut Store,
    pub(crate) local_index: usize,
}

#[derive(Clone)]
pub(crate) struct SharedFunctionRegistry {
    entries: Rc<RefCell<collections::Vec<FunctionRegistryEntry>>>,
    revision: Rc<Cell<u64>>,
}

impl SharedFunctionRegistry {
    #[inline]
    fn new() -> Self {
        Self {
            entries: Rc::new(RefCell::new(collections::Vec::new())),
            revision: Rc::new(Cell::new(0)),
        }
    }

    #[inline]
    pub(crate) fn borrow(&self) -> Ref<'_, collections::Vec<FunctionRegistryEntry>> {
        self.entries.borrow()
    }

    #[inline]
    fn borrow_mut(&self) -> RefMut<'_, collections::Vec<FunctionRegistryEntry>> {
        self.revision.set(self.revision.get().wrapping_add(1));
        self.entries.borrow_mut()
    }

    #[inline]
    pub(crate) fn revision(&self) -> u64 {
        self.revision.get()
    }
}
pub(crate) type SharedRefRegistry = Rc<RefCell<collections::Vec<RefRegistryEntry>>>;
#[cfg(sf_has_simd)]
// Shared out-of-line storage for v128 payloads.
//
// Much of nano's storage ABI still assumes that one value fits in one 64-bit
// raw slot (locals/frame slots/globals). SIMD values do not fit that model, so
// the current bring-up stores each 16-byte v128 payload here and passes around
// the registry index as the raw `u64` handle.
//
// The registry lives alongside the other cross-store registries so linked
// stores can resolve the same raw handle back to the original bytes.
#[derive(Clone)]
pub(crate) struct SharedSimdRegistry(Rc<RefCell<collections::Vec<[u8; 16]>>>);

#[cfg(sf_has_simd)]
impl SharedSimdRegistry {
    #[inline]
    pub(crate) fn new() -> Self {
        // Reserve index 0 for the all-zero vector so the common default value
        // already has a stable raw handle without a first-use allocation.
        Self(Rc::new(RefCell::new(collections::vec![[0; 16]])))
    }

    #[inline]
    /// Interns a full v128 payload and returns its raw-handle slot index.
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
    /// Resolves a raw-handle slot index back to the original v128 bytes.
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

#[derive(Clone, Debug)]
pub(crate) enum RefRegistryEntry {
    I31(i32),
    Gc {
        store: *mut Store,
        gc_ref: GcRef,
    },
    /// Globally unique identity for an interpreter-local function when no
    /// callable `FuncRefHost` resolver is installed. The originating
    /// interpreter can localize it through its own publication map; another
    /// instance must treat it as opaque rather than aliasing the same numeric
    /// local index.
    ///
    /// Opaque means opaque to type tests too: `check_ref_type_match` reports
    /// no match for it, `func` included, which diverges from the spec answer
    /// for `ref.test (ref func)`. See the note there.
    #[cfg(sf_interp)]
    OpaqueInterpFunc,
    /// Exception objects are registry-owned as well as arena-owned. Keeping
    /// the shared allocation here makes handles independent of the lifetime
    /// of the `Store` that originally allocated them.
    Exn(Rc<ExnInstance>),
}

#[cfg(sf_interp)]
impl RefRegistryEntry {
    fn resolve_exn(self) -> Option<Rc<ExnInstance>> {
        match self {
            Self::Exn(exn) => Some(exn),
            Self::I31(_) | Self::Gc { .. } | Self::OpaqueInterpFunc => None,
        }
    }
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
            functions: SharedFunctionRegistry::new(),
            refs: Rc::new(RefCell::new(collections::Vec::new())),
            #[cfg(sf_has_simd)]
            simd: SharedSimdRegistry::new(),
        }
    }

    #[inline]
    pub(crate) fn function_registry_shared(&self) -> SharedFunctionRegistry {
        self.functions.clone()
    }

    #[inline]
    pub(crate) fn ref_registry_shared(&self) -> SharedRefRegistry {
        Rc::clone(&self.refs)
    }

    /// Allocate a backend-neutral exception object in the shared reference
    /// registry.
    #[cfg(sf_interp)]
    pub(crate) fn alloc_exn(&self, tag: TagHandle, fields: collections::Vec<Value>) -> RefHandle {
        let idx = {
            let mut registry = self.refs.borrow_mut();
            let idx = registry.len();
            registry.push(RefRegistryEntry::Exn(Rc::new(ExnInstance { tag, fields })));
            idx
        };
        RefHandle::from_pool_index(idx)
    }

    /// Mint a non-callable but globally unique identity for an interpreter
    /// function reference. This is the safe fallback for linked runtimes that
    /// share a registry but did not provide the mutable cross-instance call
    /// resolver required to publish a callable host reference.
    #[cfg(sf_interp)]
    pub(crate) fn alloc_opaque_interp_funcref(&self) -> RefHandle {
        let idx = {
            let mut registry = self.refs.borrow_mut();
            let idx = registry.len();
            registry.push(RefRegistryEntry::OpaqueInterpFunc);
            idx
        };
        RefHandle::from_pool_index(idx)
    }

    /// Resolve an exception handle without copying its payload.
    #[cfg(sf_interp)]
    pub(crate) fn resolve_exn(&self, handle: RefHandle) -> Option<Rc<ExnInstance>> {
        let idx = handle.pooled_index()?;
        let entry = self.refs.borrow().get(idx).cloned()?;
        entry.resolve_exn()
    }

    /// Resolve a pooled reference for interpreter-side dynamic type checks.
    ///
    /// Entries are cloned out of the `RefCell` so callers never retain a
    /// registry borrow while consulting an origin store. Exception clones
    /// remain O(1) because their payload is shared through `Rc`.
    #[cfg(sf_interp)]
    pub(crate) fn ref_entry_for_handle(&self, handle: RefHandle) -> Option<RefRegistryEntry> {
        let idx = handle.pooled_index()?;
        self.refs.borrow().get(idx).cloned()
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
    exn_heap: Rc<RefCell<ExnHeap>>,
    module_revision: u64,
    #[cfg(sf_jit)]
    native_context_cache: Option<crate::vm::jit::runtime::context::NativeContextBox>,
    #[cfg(all(sf_jit, sf_has_guard_pages))]
    native_stack_cache: Option<crate::vm::jit::runtime::guard_pages::GuardPageStack>,
    #[cfg(all(sf_jit, not(sf_has_guard_pages)))]
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
            exn_heap: Rc::new(RefCell::new(ExnHeap::new())),
            module_revision: 0,
            #[cfg(sf_jit)]
            native_context_cache: None,
            #[cfg(sf_jit)]
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

    #[cfg(sf_jit)]
    #[inline]
    pub(crate) fn take_native_context_cache(
        &mut self,
    ) -> Option<crate::vm::jit::runtime::context::NativeContextBox> {
        self.native_context_cache.take()
    }

    #[cfg(sf_jit)]
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

    #[cfg(all(sf_jit, not(sf_has_guard_pages)))]
    #[inline]
    pub(crate) fn take_native_stack_cache(&mut self) -> Option<collections::Vec<u64>> {
        self.native_stack_cache.take()
    }

    #[cfg(all(sf_jit, not(sf_has_guard_pages)))]
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

    /// Allocate a fresh `ExnInstance` carrying the given tag identity and
    /// payload; returns an opaque `ExnRef` index into the store's heap.
    pub(crate) fn alloc_exn(
        &mut self,
        tag: crate::vm::tag::TagHandle,
        fields: crate::collections::Vec<crate::vm::value::Value>,
    ) -> ExnRef {
        self.exn_heap.borrow_mut().alloc(tag, fields)
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

    pub(crate) fn register_exn_ref(&mut self, exn_ref: ExnRef) -> RefHandle {
        let exn = self
            .exn_heap
            .borrow()
            .get_shared(exn_ref)
            .expect("register_exn_ref requires an allocated ExnRef");
        let idx = {
            let mut registry = self.ref_registry.borrow_mut();
            let idx = registry.len();
            registry.push(RefRegistryEntry::Exn(exn));
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

    #[test]
    fn shared_registry_owns_exception_payloads() {
        let registry = LinkRegistry::new();
        let tag = TagHandle::mint_fresh();
        let fields = collections::vec![Value::I32(7), Value::I64(11)];

        let handle = registry.alloc_exn(tag, fields.clone());
        let resolved = registry.resolve_exn(handle).expect("shared exception");

        assert_eq!(resolved.tag, tag);
        assert_eq!(resolved.fields, fields);
    }

    #[test]
    fn shared_registry_resolves_store_owned_exceptions() {
        let registry = LinkRegistry::new();
        let mut store = alloc::boxed::Box::new(Store::new_with_registries(
            ModuleInst::default(),
            registry.function_registry_shared(),
            registry.ref_registry_shared(),
            #[cfg(sf_has_simd)]
            registry.simd_registry_shared(),
        ));
        let tag = TagHandle::mint_fresh();
        let fields = collections::vec![Value::I32(23)];
        let exn_ref = store.alloc_exn(tag, fields.clone());
        let arena_exn = store
            .exn_heap
            .borrow()
            .get_shared(exn_ref)
            .expect("arena exception");
        let handle = store.register_exn_ref(exn_ref);

        let resolved = registry.resolve_exn(handle).expect("Store-owned exception");

        assert!(Rc::ptr_eq(&resolved, &arena_exn));
        assert_eq!(resolved.tag, tag);
        assert_eq!(resolved.fields, fields);
    }

    #[test]
    fn store_owned_exception_survives_its_origin_store() {
        let registry = LinkRegistry::new();
        let (handle, tag, fields) = {
            let mut store = alloc::boxed::Box::new(Store::new_with_registries(
                ModuleInst::default(),
                registry.function_registry_shared(),
                registry.ref_registry_shared(),
                #[cfg(sf_has_simd)]
                registry.simd_registry_shared(),
            ));
            let tag = TagHandle::mint_fresh();
            let fields = collections::vec![Value::I32(41)];
            let exn_ref = store.alloc_exn(tag, fields.clone());
            (store.register_exn_ref(exn_ref), tag, fields)
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
        let mut store = alloc::boxed::Box::new(Store::new_with_registries(
            ModuleInst::default(),
            registry.function_registry_shared(),
            registry.ref_registry_shared(),
            #[cfg(sf_has_simd)]
            registry.simd_registry_shared(),
        ));
        let i31 = store.register_i31(7);
        let exn = registry.alloc_exn(TagHandle::mint_fresh(), collections::vec![]);

        assert_ne!(i31, exn);
        assert!(registry.resolve_exn(i31).is_none());
        assert!(registry.resolve_exn(exn).is_some());
    }
}
