//! Cross-instance linking: the one place the engines actually meet.
//!
//! A [`LinkRegistry`] is the sharing handle an embedder passes to
//! `Instance::from_module_with_registry` so that separately instantiated
//! modules can exchange references. It bundles independently shared arenas;
//! a registry entry is only as engine-specific as the engine that minted it,
//! so entry payloads are gated on the engine that produces them:
//!
//! - Function-registry entries point back into a JIT `Store` and exist only
//!   when the JIT is compiled in. The interpreter deliberately does not
//!   participate; it publishes function references through its embedder's
//!   `FuncRefHost` instead.
//! - `RefRegistryEntry::Gc`/`I31` are minted by the JIT runtime's GC
//!   helpers. `OpaqueInterpFunc` is minted by the interpreter. `Exn` is
//!   engine-neutral: both engines allocate exceptions here, and the `Rc`
//!   keeps the object alive independently of whichever instance threw it.

use crate::collections;
use crate::vm::tag::TagHandle;
use crate::vm::value::{RefHandle, Value};
use core::cell::RefCell;
#[cfg(sf_jit)]
use core::cell::{Cell, Ref, RefMut};
use tracked_alloc::rc::Rc;

#[cfg(sf_jit)]
use crate::vm::jit::gc_heap::GcRef;
#[cfg(sf_jit)]
use crate::vm::jit::store::Store;

/// A backend-neutral exception object. Registry-owned: the shared `Rc` makes
/// an exception's lifetime independent of the instance that allocated it.
#[derive(Debug, Clone)]
pub(crate) struct ExnInstance {
    pub(crate) tag: TagHandle,
    // JIT-only builds retain payloads for exception identity/lifetime even
    // though only the interpreter currently reads them back at a catch.
    #[cfg_attr(
        not(sf_interp),
        allow(
            dead_code,
            reason = "payloads keep exception identity alive; only the \
                      interpreter's catch path reads them back today"
        )
    )]
    pub(crate) fields: collections::Vec<Value>,
}

#[cfg(sf_jit)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct FunctionRegistryEntry {
    pub(crate) store: *mut Store,
    pub(crate) local_index: usize,
}

#[cfg(sf_jit)]
#[derive(Clone)]
pub(crate) struct SharedFunctionRegistry {
    entries: Rc<RefCell<collections::Vec<FunctionRegistryEntry>>>,
    revision: Rc<Cell<u64>>,
}

#[cfg(sf_jit)]
impl SharedFunctionRegistry {
    #[inline]
    pub(crate) fn new() -> Self {
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
    pub(crate) fn borrow_mut(&self) -> RefMut<'_, collections::Vec<FunctionRegistryEntry>> {
        self.revision.set(self.revision.get().wrapping_add(1));
        self.entries.borrow_mut()
    }

    #[inline]
    pub(crate) fn revision(&self) -> u64 {
        self.revision.get()
    }
}

pub(crate) type SharedRefRegistry = Rc<RefCell<collections::Vec<RefRegistryEntry>>>;

#[cfg(all(sf_jit, sf_has_simd))]
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

#[cfg(all(sf_jit, sf_has_simd))]
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

#[cfg(all(sf_jit, sf_has_simd))]
impl Default for SharedSimdRegistry {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub(crate) enum RefRegistryEntry {
    /// An `i31` payload interned by the JIT runtime's `ref.i31` helper.
    #[cfg(sf_jit)]
    I31(i32),
    /// A GC object owned by a JIT `Store`'s heap. The raw pointer is nulled
    /// by `Store::drop` when the owning store goes away.
    #[cfg(sf_jit)]
    Gc { store: *mut Store, gc_ref: GcRef },
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
    /// Exception objects are registry-owned. Keeping the shared allocation
    /// here makes handles independent of the lifetime of the instance that
    /// originally allocated them.
    Exn(Rc<ExnInstance>),
}

#[cfg(sf_interp)]
impl RefRegistryEntry {
    fn resolve_exn(self) -> Option<Rc<ExnInstance>> {
        match self {
            Self::Exn(exn) => Some(exn),
            _ => None,
        }
    }
}

/// Allocate a backend-neutral exception object in a shared reference
/// registry. Both engines' exception allocation funnels through here.
pub(crate) fn alloc_exn_in(
    refs: &SharedRefRegistry,
    tag: TagHandle,
    fields: collections::Vec<Value>,
) -> RefHandle {
    let idx = {
        let mut registry = refs.borrow_mut();
        let idx = registry.len();
        registry.push(RefRegistryEntry::Exn(Rc::new(ExnInstance { tag, fields })));
        idx
    };
    RefHandle::from_pool_index(idx)
}

#[derive(Clone)]
pub struct LinkRegistry {
    #[cfg(sf_jit)]
    functions: SharedFunctionRegistry,
    refs: SharedRefRegistry,
    #[cfg(all(sf_jit, sf_has_simd))]
    simd: SharedSimdRegistry,
}

impl LinkRegistry {
    #[inline]
    pub fn new() -> Self {
        Self {
            #[cfg(sf_jit)]
            functions: SharedFunctionRegistry::new(),
            refs: Rc::new(RefCell::new(collections::Vec::new())),
            #[cfg(all(sf_jit, sf_has_simd))]
            simd: SharedSimdRegistry::new(),
        }
    }

    #[cfg(sf_jit)]
    #[inline]
    pub(crate) fn function_registry_shared(&self) -> SharedFunctionRegistry {
        self.functions.clone()
    }

    // Only the JIT's instantiation path attaches a whole store to a shared
    // registry; the interpreter goes through the typed helpers below.
    #[cfg(sf_jit)]
    #[inline]
    pub(crate) fn ref_registry_shared(&self) -> SharedRefRegistry {
        Rc::clone(&self.refs)
    }

    /// Allocate a backend-neutral exception object in the shared reference
    /// registry.
    #[cfg(sf_interp)]
    pub(crate) fn alloc_exn(&self, tag: TagHandle, fields: collections::Vec<Value>) -> RefHandle {
        alloc_exn_in(&self.refs, tag, fields)
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
    #[cfg(all(sf_jit, sf_has_simd))]
    pub(crate) fn simd_registry_shared(&self) -> SharedSimdRegistry {
        self.simd.clone()
    }

    #[inline]
    #[cfg(all(sf_jit, sf_has_simd))]
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
    #[cfg(all(sf_jit, not(sf_has_simd)))]
    pub(crate) fn from_shared(functions: SharedFunctionRegistry, refs: SharedRefRegistry) -> Self {
        Self { functions, refs }
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
}
