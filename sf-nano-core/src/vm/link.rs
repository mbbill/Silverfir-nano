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
#[cfg(sf_jit)]
use crate::vm::value::FUNCADDR_TOP;
use crate::vm::value::{RefHandle, Value};
use core::cell::RefCell;
#[cfg(sf_jit)]
use core::cell::{Cell, Ref};
use tracked_alloc::{
    boxed::Box,
    rc::{Rc, Weak},
};

#[cfg(sf_interp)]
use crate::vm::interpreter::InterpInstance;
#[cfg(sf_jit)]
use crate::vm::jit::gc_heap::GcRef;
#[cfg(sf_jit)]
use crate::vm::jit::store::Store;

/// Stable identity for one instance in a runtime world.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct InstanceId {
    index: u32,
    generation: u32,
}

impl InstanceId {
    #[inline]
    pub(crate) const fn from_parts(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    #[inline]
    pub(crate) const fn index(self) -> u32 {
        self.index
    }

    #[inline]
    pub(crate) const fn generation(self) -> u32 {
        self.generation
    }
}

/// The instance storage owned by the world table.
///
/// The boxes are load-bearing: a table borrow covers the vector's slot and
/// box-pointer storage, never the separately allocated instance body.
pub(crate) enum WorldSlot {
    #[cfg(sf_jit)]
    Jit(Box<Store>),
    #[cfg(sf_interp)]
    Interp(Box<InterpInstance>),
    Vacant,
}

struct InstanceTableInner {
    slots: RefCell<collections::Vec<WorldSlot>>,
    generations: RefCell<collections::Vec<u32>>,
    in_use: RefCell<collections::Vec<u32>>,
    release_pending: RefCell<collections::Vec<bool>>,
    reusable: RefCell<collections::Vec<u32>>,
}

/// The sole strong owner of the generational instance table.
#[derive(Clone)]
pub(crate) struct InstanceTable(Rc<InstanceTableInner>);

/// A non-owning back-reference carried by each stored engine instance.
#[derive(Clone)]
pub(crate) struct InstanceHandle {
    table: Weak<InstanceTableInner>,
    self_id: InstanceId,
}

enum InstancePointer {
    #[cfg(sf_jit)]
    Jit(*mut Store),
    #[cfg(sf_interp)]
    Interp(*mut InterpInstance),
}

/// A generation-checked, RAII checkout of one instance slot.
pub(crate) struct InstanceToken {
    table: Rc<InstanceTableInner>,
    id: InstanceId,
    pointer: InstancePointer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InstanceFreeError {
    InvalidId,
    InUse,
}

impl InstanceTable {
    #[inline]
    fn new() -> Self {
        Self(Rc::new(InstanceTableInner {
            slots: RefCell::new(collections::Vec::new()),
            generations: RefCell::new(collections::Vec::new()),
            in_use: RefCell::new(collections::Vec::new()),
            release_pending: RefCell::new(collections::Vec::new()),
            reusable: RefCell::new(collections::Vec::new()),
        }))
    }

    pub(crate) fn reserve(&self) -> InstanceId {
        while let Some(index) = self.0.reusable.borrow_mut().pop() {
            let generation = self.0.generations.borrow()[index as usize];
            if generation != u32::MAX {
                return InstanceId::from_parts(index, generation);
            }
        }

        let mut slots = self.0.slots.borrow_mut();
        let mut generations = self.0.generations.borrow_mut();
        let mut in_use = self.0.in_use.borrow_mut();
        let mut release_pending = self.0.release_pending.borrow_mut();
        let index = slots.len();
        assert!(u32::try_from(index).is_ok(), "instance table exhausted");
        slots.push(WorldSlot::Vacant);
        generations.push(0);
        in_use.push(0);
        release_pending.push(false);
        InstanceId::from_parts(index as u32, 0)
    }

    #[cfg(any(sf_interp, test))]
    pub(crate) fn abandon(&self, id: InstanceId) -> Result<(), InstanceFreeError> {
        let generations = self.0.generations.borrow();
        if generations.get(id.index() as usize).copied() != Some(id.generation()) {
            return Err(InstanceFreeError::InvalidId);
        }
        let slots = self.0.slots.borrow();
        if !matches!(slots.get(id.index() as usize), Some(WorldSlot::Vacant)) {
            return Err(InstanceFreeError::InvalidId);
        }
        if self.0.in_use.borrow()[id.index() as usize] != 0 {
            return Err(InstanceFreeError::InUse);
        }
        drop(slots);
        drop(generations);
        if self.advance_generation(id.index()) {
            self.0.reusable.borrow_mut().push(id.index());
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn handle(&self, id: InstanceId) -> InstanceHandle {
        InstanceHandle {
            table: Rc::downgrade(&self.0),
            self_id: id,
        }
    }

    #[cfg(sf_jit)]
    pub(crate) fn occupy_jit(
        &self,
        id: InstanceId,
        store: Box<Store>,
    ) -> Result<(), InstanceFreeError> {
        self.occupy(id, WorldSlot::Jit(store))
    }

    #[cfg(sf_interp)]
    pub(crate) fn occupy_interp(
        &self,
        id: InstanceId,
        instance: Box<InterpInstance>,
    ) -> Result<(), InstanceFreeError> {
        self.occupy(id, WorldSlot::Interp(instance))
    }

    fn occupy(&self, id: InstanceId, value: WorldSlot) -> Result<(), InstanceFreeError> {
        let generations = self.0.generations.borrow();
        let Some(&generation) = generations.get(id.index() as usize) else {
            return Err(InstanceFreeError::InvalidId);
        };
        if generation != id.generation() {
            return Err(InstanceFreeError::InvalidId);
        }
        let mut slots = self.0.slots.borrow_mut();
        let Some(slot) = slots.get_mut(id.index() as usize) else {
            return Err(InstanceFreeError::InvalidId);
        };
        if !matches!(slot, WorldSlot::Vacant) {
            return Err(InstanceFreeError::InvalidId);
        }
        *slot = value;
        Ok(())
    }

    pub(crate) fn checkout(&self, id: InstanceId) -> Option<InstanceToken> {
        let generations = self.0.generations.borrow();
        if generations.get(id.index() as usize).copied()? != id.generation() {
            return None;
        }

        let slots = self.0.slots.borrow();
        let pointer = match slots.get(id.index() as usize)? {
            #[cfg(sf_jit)]
            WorldSlot::Jit(store) => {
                // Do not form a reference to the pointee inside the table
                // borrow. The table covers only the box pointer.
                InstancePointer::Jit((&raw const **store).cast_mut())
            }
            #[cfg(sf_interp)]
            WorldSlot::Interp(instance) => {
                InstancePointer::Interp((&raw const **instance).cast_mut())
            }
            WorldSlot::Vacant => return None,
        };

        let mut in_use = self.0.in_use.borrow_mut();
        let count = in_use.get_mut(id.index() as usize)?;
        *count = count.checked_add(1)?;
        drop(in_use);
        drop(slots);
        drop(generations);

        Some(InstanceToken {
            table: Rc::clone(&self.0),
            id,
            pointer,
        })
    }

    pub(crate) fn free(&self, id: InstanceId) -> Result<(), InstanceFreeError> {
        {
            let generations = self.0.generations.borrow();
            if generations.get(id.index() as usize).copied() != Some(id.generation()) {
                return Err(InstanceFreeError::InvalidId);
            }
        }
        if self
            .0
            .in_use
            .borrow()
            .get(id.index() as usize)
            .copied()
            .ok_or(InstanceFreeError::InvalidId)?
            != 0
        {
            return Err(InstanceFreeError::InUse);
        }

        if matches!(
            self.0.slots.borrow().get(id.index() as usize),
            Some(WorldSlot::Vacant) | None
        ) {
            return Err(InstanceFreeError::InvalidId);
        }
        self.reclaim(id);
        Ok(())
    }

    fn reclaim(&self, id: InstanceId) {
        // `free` validates an occupied, unused slot; deferred release is
        // reached only by the last live token for the same generation.
        let removed = core::mem::replace(
            &mut self.0.slots.borrow_mut()[id.index() as usize],
            WorldSlot::Vacant,
        );
        let reusable = self.advance_generation(id.index());
        self.0.release_pending.borrow_mut()[id.index() as usize] = false;
        if reusable {
            self.0.reusable.borrow_mut().push(id.index());
        }
        drop(removed);
    }

    fn advance_generation(&self, index: u32) -> bool {
        let mut generations = self.0.generations.borrow_mut();
        let generation = &mut generations[index as usize];
        if *generation != u32::MAX {
            *generation += 1;
        }
        *generation != u32::MAX
    }

    #[inline]
    pub(crate) fn in_use(&self, id: InstanceId) -> Option<u32> {
        if self
            .0
            .generations
            .borrow()
            .get(id.index() as usize)
            .copied()?
            != id.generation()
        {
            return None;
        }
        self.0.in_use.borrow().get(id.index() as usize).copied()
    }
}

impl InstanceHandle {
    #[inline]
    pub(crate) const fn self_id(&self) -> InstanceId {
        self.self_id
    }

    pub(crate) fn checkout(&self, id: InstanceId) -> Option<InstanceToken> {
        let table = self.table.upgrade()?;
        InstanceTable(table.into()).checkout(id)
    }

    #[cfg(sf_jit)]
    pub(crate) fn table(&self) -> Option<InstanceTable> {
        let table = self.table.upgrade()?;
        Some(InstanceTable(table.into()))
    }
}

impl InstanceToken {
    #[inline]
    pub(crate) const fn id(&self) -> InstanceId {
        self.id
    }

    #[cfg(sf_jit)]
    #[inline]
    pub(crate) fn jit(&self) -> Option<&Store> {
        match self.pointer {
            InstancePointer::Jit(pointer) => Some(unsafe { &*pointer }),
            #[cfg(sf_interp)]
            InstancePointer::Interp(_) => None,
        }
    }

    /// The generation-checked slot pointer without materializing its store.
    #[cfg(sf_jit)]
    #[inline]
    pub(crate) fn jit_pointer(&self) -> Option<*mut Store> {
        match self.pointer {
            InstancePointer::Jit(pointer) => Some(pointer),
            #[cfg(sf_interp)]
            InstancePointer::Interp(_) => None,
        }
    }

    #[cfg(sf_jit)]
    #[inline]
    pub(crate) fn jit_mut(&mut self) -> Option<&mut Store> {
        match self.pointer {
            InstancePointer::Jit(pointer) => Some(unsafe { &mut *pointer }),
            #[cfg(sf_interp)]
            InstancePointer::Interp(_) => None,
        }
    }

    #[cfg(sf_interp)]
    #[inline]
    pub(crate) fn interp(&self) -> Option<&InterpInstance> {
        match self.pointer {
            #[cfg(sf_jit)]
            InstancePointer::Jit(_) => None,
            InstancePointer::Interp(pointer) => Some(unsafe { &*pointer }),
        }
    }

    #[cfg(sf_interp)]
    #[inline]
    pub(crate) fn interp_mut(&mut self) -> Option<&mut InterpInstance> {
        match self.pointer {
            #[cfg(sf_jit)]
            InstancePointer::Jit(_) => None,
            InstancePointer::Interp(pointer) => Some(unsafe { &mut *pointer }),
        }
    }
}

impl Drop for InstanceToken {
    fn drop(&mut self) {
        let reclaim = {
            let mut in_use = self.table.in_use.borrow_mut();
            let count = &mut in_use[self.id.index() as usize];
            *count = count
                .checked_sub(1)
                .expect("instance checkout count underflow");
            *count == 0 && self.table.release_pending.borrow()[self.id.index() as usize]
        };
        if reclaim {
            InstanceTable(Rc::clone(&self.table)).reclaim(self.id);
        }
    }
}

/// The existing instance APIs own one lease for as long as their facade is
/// alive. Dropping the facade releases its checkout and frees the slot.
pub(crate) struct InstanceLease {
    token: Option<InstanceToken>,
}

impl InstanceLease {
    pub(crate) fn checkout(handle: &InstanceHandle) -> Option<Self> {
        Some(Self {
            token: Some(handle.checkout(handle.self_id())?),
        })
    }

    #[inline]
    pub(crate) fn id(&self) -> InstanceId {
        self.token
            .as_ref()
            .expect("instance lease already released")
            .id()
    }

    #[inline]
    pub(crate) fn token(&self) -> &InstanceToken {
        self.token
            .as_ref()
            .expect("instance lease already released")
    }

    #[inline]
    pub(crate) fn token_mut(&mut self) -> &mut InstanceToken {
        self.token
            .as_mut()
            .expect("instance lease already released")
    }

    pub(crate) fn is_exclusive(&self) -> bool {
        let token = self
            .token
            .as_ref()
            .expect("instance lease already released");
        InstanceTable(Rc::clone(&token.table)).in_use(token.id) == Some(1)
    }
}

impl Drop for InstanceLease {
    fn drop(&mut self) {
        let Some(token) = self.token.take() else {
            return;
        };
        let id = token.id;
        let table = InstanceTable(Rc::clone(&token.table));
        drop(token);
        match table.free(id) {
            Ok(()) => {}
            Err(InstanceFreeError::InUse) => {
                table.0.release_pending.borrow_mut()[id.index() as usize] = true;
            }
            // A prior lease release can have requested reclamation, making
            // this token's drop the operation that already freed the slot.
            Err(InstanceFreeError::InvalidId) => {}
        }
    }
}

/// A backend-neutral exception object. Registry-owned: the shared `Rc` makes
/// an exception's lifetime independent of the instance that allocated it.
#[derive(Debug, Clone)]
pub(crate) struct ExnInstance {
    pub(crate) tag: TagHandle,
    // JIT-only builds retain payloads for exception identity/lifetime even
    // though only the interpreter currently reads them back at a catch.
    #[cfg_attr(not(sf_interp), allow(dead_code))]
    pub(crate) fields: collections::Vec<Value>,
}

#[cfg(sf_jit)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct FuncEntry {
    pub(crate) owner: InstanceId,
    pub(crate) local_index: u32,
}

#[cfg(sf_jit)]
#[derive(Clone)]
pub(crate) struct SharedFunctionRegistry {
    entries: Rc<RefCell<collections::Vec<FuncEntry>>>,
    // This is a lifetime high-water mark, not a cache revision. Absolute
    // handles may outlive their owner, so shrinking it on instance teardown
    // could make a stale absolute value alias a later instance's local range.
    max_local_functions_ever: Rc<Cell<usize>>,
}

#[cfg(sf_jit)]
impl SharedFunctionRegistry {
    #[inline]
    pub(crate) fn new() -> Self {
        Self {
            entries: Rc::new(RefCell::new(collections::Vec::new())),
            max_local_functions_ever: Rc::new(Cell::new(0)),
        }
    }

    #[inline]
    pub(crate) fn borrow(&self) -> Ref<'_, collections::Vec<FuncEntry>> {
        self.entries.borrow()
    }

    #[inline]
    pub(crate) fn observe_local_function_count(&self, count: usize) {
        assert!(
            count <= FUNCADDR_TOP,
            "local function index space exceeds the funcaddr encoding"
        );
        let max_local_functions = self.max_local_functions_ever.get().max(count);
        if let Some(last_funcaddr) = self.entries.borrow().len().checked_sub(1) {
            let lowest_absolute = FUNCADDR_TOP
                .checked_sub(last_funcaddr)
                .expect("world function address space exhausted");
            assert!(
                lowest_absolute >= max_local_functions,
                "local and absolute function address ranges overlap"
            );
        }
        self.max_local_functions_ever.set(max_local_functions);
    }

    #[inline]
    pub(crate) fn register(&self, entry: FuncEntry) -> usize {
        let mut entries = self.entries.borrow_mut();
        let funcaddr = entries.len();
        let encoded = FUNCADDR_TOP
            .checked_sub(funcaddr)
            .expect("world function address space exhausted");
        assert!(
            encoded >= self.max_local_functions_ever.get(),
            "local and absolute function address ranges overlap"
        );
        entries.push(entry);
        funcaddr
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
pub(crate) struct LinkArenas {
    #[cfg(sf_jit)]
    functions: SharedFunctionRegistry,
    refs: SharedRefRegistry,
    #[cfg(all(sf_jit, sf_has_simd))]
    simd: SharedSimdRegistry,
}

impl LinkArenas {
    #[inline]
    fn new() -> Self {
        Self {
            #[cfg(sf_jit)]
            functions: SharedFunctionRegistry::new(),
            refs: Rc::new(RefCell::new(collections::Vec::new())),
            #[cfg(all(sf_jit, sf_has_simd))]
            simd: SharedSimdRegistry::new(),
        }
    }

    /// Allocate a backend-neutral exception object in the shared reference
    /// registry.
    #[cfg(sf_interp)]
    pub(crate) fn alloc_exn(&self, tag: TagHandle, fields: collections::Vec<Value>) -> RefHandle {
        alloc_exn_in(&self.refs, tag, fields)
    }

    /// Mint a non-callable but globally unique identity for an interpreter
    /// function reference.
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

    #[cfg(sf_interp)]
    pub(crate) fn resolve_exn(&self, handle: RefHandle) -> Option<Rc<ExnInstance>> {
        let idx = handle.pooled_index()?;
        let entry = self.refs.borrow().get(idx).cloned()?;
        entry.resolve_exn()
    }

    #[cfg(sf_interp)]
    pub(crate) fn ref_entry_for_handle(&self, handle: RefHandle) -> Option<RefRegistryEntry> {
        let idx = handle.pooled_index()?;
        self.refs.borrow().get(idx).cloned()
    }
}

#[derive(Clone)]
pub struct LinkRegistry {
    arenas: LinkArenas,
    instances: InstanceTable,
}

impl LinkRegistry {
    #[inline]
    pub fn new() -> Self {
        Self {
            arenas: LinkArenas::new(),
            instances: InstanceTable::new(),
        }
    }

    #[inline]
    pub(crate) fn reserve_instance(&self) -> (InstanceId, InstanceHandle) {
        let id = self.instances.reserve();
        (id, self.instances.handle(id))
    }

    #[inline]
    #[cfg(any(sf_interp, test))]
    pub(crate) fn arenas(&self) -> LinkArenas {
        self.arenas.clone()
    }

    #[inline]
    pub(crate) fn instance_table(&self) -> InstanceTable {
        self.instances.clone()
    }

    #[cfg(sf_jit)]
    #[inline]
    pub(crate) fn function_registry_shared(&self) -> SharedFunctionRegistry {
        self.arenas.functions.clone()
    }

    // Only the JIT's instantiation path attaches a whole store to a shared
    // registry; the interpreter goes through the typed helpers below.
    #[cfg(sf_jit)]
    #[inline]
    pub(crate) fn ref_registry_shared(&self) -> SharedRefRegistry {
        Rc::clone(&self.arenas.refs)
    }

    #[inline]
    #[cfg(all(sf_jit, sf_has_simd))]
    pub(crate) fn simd_registry_shared(&self) -> SharedSimdRegistry {
        self.arenas.simd.clone()
    }

    #[inline]
    #[cfg(all(sf_jit, sf_has_simd))]
    pub(crate) fn from_shared(
        functions: SharedFunctionRegistry,
        refs: SharedRefRegistry,
        simd: SharedSimdRegistry,
        instances: InstanceTable,
    ) -> Self {
        Self {
            arenas: LinkArenas {
                functions,
                refs,
                simd,
            },
            instances,
        }
    }

    #[inline]
    #[cfg(all(sf_jit, not(sf_has_simd)))]
    pub(crate) fn from_shared(
        functions: SharedFunctionRegistry,
        refs: SharedRefRegistry,
        instances: InstanceTable,
    ) -> Self {
        Self {
            arenas: LinkArenas { functions, refs },
            instances,
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

        let arenas = registry.arenas();
        let handle = arenas.alloc_exn(tag, fields.clone());
        let resolved = arenas.resolve_exn(handle).expect("shared exception");

        assert_eq!(resolved.tag, tag);
        assert_eq!(resolved.fields, fields);
    }
}

#[cfg(all(test, sf_jit))]
mod instance_table_tests {
    use super::*;
    use crate::config::Config;
    use crate::module::type_context::TypeContext;
    use crate::vm::jit::entities::ModuleInst;
    #[cfg(any(sf_ir_dump, sf_jitdump))]
    use tracked_alloc::string::String;

    fn test_store(registry: &LinkRegistry, handle: InstanceHandle) -> Box<Store> {
        Box::new(Store::new_with_registries(
            ModuleInst::new(
                Config::new(),
                #[cfg(any(sf_ir_dump, sf_jitdump))]
                String::from("instance-table-test"),
                TypeContext::new(collections::Vec::new()),
            ),
            handle,
            registry.function_registry_shared(),
            registry.ref_registry_shared(),
            #[cfg(sf_has_simd)]
            registry.simd_registry_shared(),
        ))
    }

    fn occupied_store(registry: &LinkRegistry) -> (InstanceId, InstanceHandle) {
        let (id, handle) = registry.reserve_instance();
        let store = test_store(registry, handle.clone());
        registry
            .instance_table()
            .occupy_jit(id, store)
            .expect("fresh instance slot");
        (id, handle)
    }

    #[test]
    fn checkout_rejects_generation_mismatch() {
        let registry = LinkRegistry::new();
        let (id, _) = occupied_store(&registry);
        let stale = InstanceId::from_parts(id.index(), id.generation().wrapping_add(1));

        assert!(registry.instance_table().checkout(stale).is_none());
        assert_eq!(registry.instance_table().free(id), Ok(()));
    }

    #[test]
    fn pending_reservations_are_distinct_and_abandoned_slots_reuse() {
        let registry = LinkRegistry::new();
        let table = registry.instance_table();
        let first = table.reserve();
        let second = table.reserve();

        assert_ne!(first.index(), second.index());
        table.abandon(first).expect("abandon first reservation");
        let reused = table.reserve();
        assert_eq!(reused.index(), first.index());
        assert_ne!(reused.generation(), first.generation());
        table
            .occupy_jit(reused, test_store(&registry, table.handle(reused)))
            .expect("occupy reused reservation");
        assert!(table.checkout(first).is_none());
        let replacement = table.checkout(reused).expect("new generation checks out");
        drop(replacement);
        table.free(reused).expect("free reused reservation");
        table.abandon(second).expect("abandon second reservation");
    }

    #[test]
    fn abandoning_maximum_generation_retires_slot() {
        let registry = LinkRegistry::new();
        let table = registry.instance_table();
        let id = table.reserve();
        table.0.generations.borrow_mut()[id.index() as usize] = u32::MAX - 1;
        let last = InstanceId::from_parts(id.index(), u32::MAX - 1);

        table.abandon(last).expect("retire vacant slot");

        let replacement = table.reserve();
        assert_ne!(replacement.index(), last.index());
        assert_eq!(
            table.0.generations.borrow()[last.index() as usize],
            u32::MAX
        );
    }

    #[test]
    fn free_rejects_checked_out_slot() {
        let registry = LinkRegistry::new();
        let (id, _) = occupied_store(&registry);
        let token = registry
            .instance_table()
            .checkout(id)
            .expect("occupied instance");

        assert_eq!(
            registry.instance_table().free(id),
            Err(InstanceFreeError::InUse)
        );
        drop(token);
        assert_eq!(registry.instance_table().free(id), Ok(()));
    }

    #[test]
    fn lease_release_reclaims_after_the_last_checkout() {
        let registry = LinkRegistry::new();
        let table = registry.instance_table();
        let (id, handle) = occupied_store(&registry);
        let lease = InstanceLease::checkout(&handle).expect("instance lease");
        let checkout = table.checkout(id).expect("second checkout");

        drop(lease);
        assert_eq!(table.in_use(id), Some(1));
        assert_eq!(table.free(id), Err(InstanceFreeError::InUse));

        drop(checkout);
        assert!(table.checkout(id).is_none());
        let replacement = table.reserve();
        assert_eq!(replacement.index(), id.index());
        assert_ne!(replacement.generation(), id.generation());
    }

    #[test]
    fn two_checkout_tokens_can_coexist() {
        let registry = LinkRegistry::new();
        let (id, _) = occupied_store(&registry);
        let first = registry
            .instance_table()
            .checkout(id)
            .expect("first checkout");
        let second = registry
            .instance_table()
            .checkout(id)
            .expect("nested checkout");

        assert_eq!(registry.instance_table().in_use(id), Some(2));
        drop(second);
        assert_eq!(registry.instance_table().in_use(id), Some(1));
        drop(first);
        assert_eq!(registry.instance_table().free(id), Ok(()));
    }

    #[test]
    fn miri_instance_table_scopes_materializations() {
        let registry = LinkRegistry::new();
        let table = registry.instance_table();
        let (a_id, _) = occupied_store(&registry);
        let (b_id, _) = occupied_store(&registry);
        assert_ne!(a_id, b_id);

        let mut first_a = table.checkout(a_id).expect("first A checkout");
        let a_handle = {
            let a_store = first_a.jit_mut().expect("first A materialization");
            assert_eq!(a_store.instance_handle().self_id(), a_id);
            a_store.instance_handle().clone()
        };
        assert_eq!(table.in_use(a_id), Some(1));
        assert_eq!(table.in_use(b_id), Some(0));

        let mut b = a_handle.checkout(b_id).expect("B checkout from A");
        let b_handle = {
            let b_store = b.jit_mut().expect("B materialization");
            assert_eq!(b_store.instance_handle().self_id(), b_id);
            b_store.instance_handle().clone()
        };
        assert_eq!(table.in_use(a_id), Some(1));
        assert_eq!(table.in_use(b_id), Some(1));

        let mut reentered_a = b_handle.checkout(a_id).expect("A re-entry from B");
        {
            let a_store = reentered_a.jit_mut().expect("re-entered A materialization");
            assert_eq!(a_store.instance_handle().self_id(), a_id);
        }
        assert_eq!(table.in_use(a_id), Some(2));
        assert_eq!(table.in_use(b_id), Some(1));
        assert_eq!(table.free(a_id), Err(InstanceFreeError::InUse));
        assert_eq!(table.free(b_id), Err(InstanceFreeError::InUse));

        drop(reentered_a);
        assert_eq!(table.in_use(a_id), Some(1));
        {
            let a_store = first_a
                .jit_mut()
                .expect("A rematerialization after re-entry");
            assert_eq!(a_store.instance_handle().self_id(), a_id);
        }
        drop(b);
        assert_eq!(table.in_use(b_id), Some(0));
        drop(first_a);
        assert_eq!(table.in_use(a_id), Some(0));

        table.free(a_id).expect("free Miri A");
        table.free(b_id).expect("free Miri B");
        assert!(table.checkout(a_id).is_none());
        assert!(table.checkout(b_id).is_none());
    }

    #[test]
    fn maximum_generation_retires_slot() {
        let registry = LinkRegistry::new();
        let table = registry.instance_table();
        let id = table.reserve();
        table.0.generations.borrow_mut()[id.index() as usize] = u32::MAX - 1;
        let old = InstanceId::from_parts(id.index(), u32::MAX - 1);
        let handle = table.handle(old);
        let store = Box::new(Store::new_with_registries(
            ModuleInst::new(
                Config::new(),
                #[cfg(any(sf_ir_dump, sf_jitdump))]
                String::from("retired-instance-slot"),
                TypeContext::new(collections::Vec::new()),
            ),
            handle,
            registry.function_registry_shared(),
            registry.ref_registry_shared(),
            #[cfg(sf_has_simd)]
            registry.simd_registry_shared(),
        ));
        table
            .occupy_jit(old, store)
            .expect("penultimate generation");
        table.free(old).expect("retire slot");

        let replacement = table.reserve();
        assert_ne!(replacement.index(), old.index());
        assert_eq!(table.0.generations.borrow()[old.index() as usize], u32::MAX);
    }
}
