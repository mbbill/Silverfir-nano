//! The JIT engine's runtime world.
//!
//! A `JitInstance` owns one instantiated module's entities plus the caches native
//! code addresses directly. It is JIT-owned by construction: the interpreter
//! keeps its own flat runtime state and only ever *reads* a foreign `JitInstance`
//! through a registry entry when both engines are compiled in.
//!
//! The revision counters exist for exactly one consumer: the native
//! context's cached raw views (`vm::jit::runtime::context`), which must be
//! refreshed whenever an entity the generated code points into may have
//! moved.

use crate::collections;
use crate::config::Config;
use tracked_alloc::rc::Rc;
use tracked_alloc::string::String;

use crate::vm::entities::{FunctionInst, GlobalInst, MemInst, TableInst};
use crate::vm::jit::entities::ModuleInst;
use crate::vm::jit::gc_heap::{GcHeap, GcRef};
use crate::vm::jit::instantiate::ExportKind;
#[cfg(sf_has_simd)]
use crate::vm::link::SharedSimdRegistry;
use crate::vm::link::{
    alloc_exn_in, FuncEntry, InstanceBackref, RefRegistryEntry, SharedFunctionRegistry,
    SharedRefRegistry,
};
use crate::vm::tag::TagIdentity;
use crate::vm::value::{RefValue, Value};
use core::cell::RefCell;

const UNREGISTERED_FUNCADDR: usize = usize::MAX;

pub(crate) struct JitInstance {
    module: ModuleInst,
    exports: collections::Vec<(String, ExportKind, usize)>,
    instance_backref: InstanceBackref,
    function_registry: SharedFunctionRegistry,
    local_funcaddrs: collections::Vec<usize>,
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

impl JitInstance {
    #[inline]
    pub(crate) fn new_with_registries(
        module: ModuleInst,
        instance_backref: InstanceBackref,
        function_registry: SharedFunctionRegistry,
        ref_registry: SharedRefRegistry,
        #[cfg(sf_has_simd)] simd_registry: SharedSimdRegistry,
    ) -> Self {
        let function_count = module.functions.len();
        function_registry.observe_local_function_count(function_count);
        Self {
            module,
            exports: collections::Vec::new(),
            instance_backref,
            function_registry,
            local_funcaddrs: collections::vec![UNREGISTERED_FUNCADDR; function_count],
            ref_registry,
            #[cfg(sf_has_simd)]
            simd_registry,
            gc_heap: Rc::new(RefCell::new(GcHeap::new())),
            module_revision: 0,
            native_context_cache: None,
            native_stack_cache: None,
        }
    }

    #[inline]
    pub(crate) fn instance_backref(&self) -> &InstanceBackref {
        &self.instance_backref
    }

    /// The engine configuration this store's module was created with.
    #[inline]
    pub(crate) fn config(&self) -> &Config {
        self.module.config()
    }

    #[inline]
    pub(crate) fn module(&self) -> &ModuleInst {
        &self.module
    }

    #[inline]
    pub(crate) fn module_mut(&mut self) -> &mut ModuleInst {
        self.module_revision = self.module_revision.wrapping_add(1);
        &mut self.module
    }

    #[inline]
    pub(crate) fn exports(&self) -> &[(String, ExportKind, usize)] {
        &self.exports
    }

    #[inline]
    pub(crate) fn set_exports(&mut self, exports: collections::Vec<(String, ExportKind, usize)>) {
        self.exports = exports;
    }

    #[inline]
    pub(crate) fn function(&self, idx: usize) -> &FunctionInst {
        &self.module.functions[idx]
    }

    #[inline]
    pub(crate) fn table(&self, idx: usize) -> &TableInst {
        &self.module.tables[idx]
    }

    #[inline]
    pub(crate) fn table_mut(&mut self, idx: usize) -> &mut TableInst {
        self.module_revision = self.module_revision.wrapping_add(1);
        &mut self.module.tables[idx]
    }

    #[inline]
    pub(crate) fn memory(&self, idx: usize) -> &MemInst {
        &self.module.memories[idx]
    }

    #[inline]
    pub(crate) fn memory_mut(&mut self, idx: usize) -> &mut MemInst {
        self.module_revision = self.module_revision.wrapping_add(1);
        &mut self.module.memories[idx]
    }

    #[inline]
    pub(crate) fn global(&self, idx: usize) -> &GlobalInst {
        &self.module.globals[idx]
    }

    #[inline]
    pub(crate) fn global_mut(&mut self, idx: usize) -> &mut GlobalInst {
        self.module_revision = self.module_revision.wrapping_add(1);
        &mut self.module.globals[idx]
    }

    #[inline]
    pub(crate) fn module_revision(&self) -> u64 {
        self.module_revision
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
    pub(crate) fn gc_heap(&self) -> &Rc<RefCell<GcHeap>> {
        &self.gc_heap
    }

    /// Allocate a fresh exception object carrying the given tag identity and
    /// payload. The object is registry-owned, so the returned handle stays
    /// valid after this store is dropped.
    pub(crate) fn alloc_exn(
        &mut self,
        tag: TagIdentity,
        fields: collections::Vec<Value>,
    ) -> RefValue {
        alloc_exn_in(&self.ref_registry, tag, fields)
    }

    pub(crate) fn register_local_function(&mut self, local_index: usize) -> RefValue {
        self.function_registry
            .observe_local_function_count(self.module.functions.len());
        let funcaddr = self.function_registry.register(FuncEntry {
            owner: self.instance_backref.self_id(),
            local_index: u32::try_from(local_index)
                .expect("local function index exceeds the funcaddr encoding"),
        });
        if self.local_funcaddrs.len() <= local_index {
            self.local_funcaddrs
                .resize(local_index + 1, UNREGISTERED_FUNCADDR);
        }
        self.local_funcaddrs[local_index] = funcaddr;
        let handle = RefValue::new(local_index);
        self.module.ensure_function_handle_capacity(local_index + 1);
        self.module.set_function_handle(local_index, handle);
        handle
    }

    #[inline]
    pub(crate) fn funcaddr_for_local_index(&self, local_index: usize) -> Option<usize> {
        let funcaddr = self.local_funcaddrs.get(local_index).copied()?;
        (funcaddr != UNREGISTERED_FUNCADDR).then_some(funcaddr)
    }

    pub(crate) fn function_entry_for_handle(&self, handle: RefValue) -> Option<FuncEntry> {
        self.function_registry.entry_for_handle(handle)
    }

    #[inline]
    pub(crate) fn function_entry_for_funcaddr(&self, funcaddr: usize) -> Option<FuncEntry> {
        self.function_registry.borrow().get(funcaddr).copied()
    }

    pub(crate) fn register_i31(&mut self, value: i32) -> RefValue {
        if let Some((idx, _)) = self.ref_registry.borrow().iter().enumerate().find(
            |(_, entry)| matches!(entry, RefRegistryEntry::I31(existing) if *existing == value),
        ) {
            return RefValue::from_pool_index(idx);
        }
        let idx = {
            let mut registry = self.ref_registry.borrow_mut();
            let idx = registry.len();
            registry.push(RefRegistryEntry::I31(value));
            idx
        };
        RefValue::from_pool_index(idx)
    }

    pub(crate) fn register_gc_ref(&mut self, gc_ref: GcRef) -> RefValue {
        let idx = {
            let mut registry = self.ref_registry.borrow_mut();
            let idx = registry.len();
            registry.push(RefRegistryEntry::Gc {
                owner: self.instance_backref.self_id(),
                gc_ref,
            });
            idx
        };
        RefValue::from_pool_index(idx)
    }

    pub(crate) fn ref_entry_for_handle(&self, handle: RefValue) -> Option<RefRegistryEntry> {
        let idx = handle.pooled_index()?;
        self.ref_registry.borrow().get(idx).cloned()
    }

    #[cfg(sf_has_simd)]
    // JitInstance a v128 payload out-of-line and return the 64-bit raw handle used by
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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::vm::link::LinkRegistry;
    use tracked_alloc::boxed::Box;

    pub(crate) fn store(module: ModuleInst) -> Box<JitInstance> {
        let registry = LinkRegistry::new();
        let (_, instance_backref) = registry.reserve_instance();
        Box::new(JitInstance::new_with_registries(
            module,
            instance_backref,
            registry.function_registry_shared(),
            registry.ref_registry_shared(),
            #[cfg(sf_has_simd)]
            registry.simd_registry_shared(),
        ))
    }

    #[cfg(sf_interp)]
    mod interp {
        use super::*;

        fn store_with(registry: &LinkRegistry) -> alloc::boxed::Box<JitInstance> {
            let (_, instance_backref) = registry.reserve_instance();
            alloc::boxed::Box::new(JitInstance::new_with_registries(
                ModuleInst::default(),
                instance_backref,
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
            let tag = TagIdentity::mint_fresh();
            let fields = collections::vec![Value::I32(23)];
            let handle = store.alloc_exn(tag, fields.clone());

            let resolved = registry
                .arenas()
                .resolve_exn(handle)
                .expect("JitInstance-owned exception");

            assert_eq!(resolved.tag, tag);
            assert_eq!(resolved.fields, fields);
        }

        #[test]
        fn store_owned_exception_survives_its_origin_store() {
            let registry = LinkRegistry::new();
            let (handle, tag, fields) = {
                let mut store = store_with(&registry);
                let tag = TagIdentity::mint_fresh();
                let fields = collections::vec![Value::I32(41)];
                (store.alloc_exn(tag, fields.clone()), tag, fields)
            };

            let arenas = registry.arenas();
            let first = arenas
                .resolve_exn(handle)
                .expect("registry must keep the exception alive");
            let second = arenas
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
            let arenas = registry.arenas();
            let exn = arenas.alloc_exn(TagIdentity::mint_fresh(), collections::vec![]);

            assert_ne!(i31, exn);
            assert!(arenas.resolve_exn(i31).is_none());
            assert!(arenas.resolve_exn(exn).is_some());
        }
    }
}
