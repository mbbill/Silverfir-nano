//! Native runtime context and ABI-visible cached storage views.
//!
//! This is the runtime object that native lowering targets. It stays local to
//! the native runtime boundary rather than becoming part of the generic VM API.
//!
//! NOTE: This Rust struct follows the host compiler ABI. Shared lowering and
//! the emulator must use the explicit target-side layout helpers in
//! [`super::layout`] rather than assuming these host offsets/strides directly.
//!
//! ### Globals raw-ptr tail
//!
//! Global values live in `Rc<GlobalCell>` owned by individual `GlobalInst`s
//! (so imported globals share storage with the exporter). Every `global.get` /
//! `global.set` in JIT code needs a `*mut u64` for the backing cell.
//!
//! To keep the JIT lowering to two machine loads per access (indexed fetch
//! then deref) without pinning a dedicated register, the per-global raw
//! pointers are cached **inline at the tail of `NativeContext`**. The single
//! allocation layout is `header | [*mut u64; globals_len]`, with the tail
//! anchored by the zero-sized [`NativeContext::globals_ptrs_tail`] marker so
//! its offset is a compile-time constant (`offset_of!`). JIT code reads one
//! entry from that tail via runtime_base + const offset + idx * ptr size, then
//! dereferences.
//!
//! [`NativeContextBox`] is the owning handle (custom alloc/dealloc because
//! the allocation size depends on `globals_len` and `Box<NativeContext>` has
//! no way to track the tail).

use crate::collections;

use crate::{
    error::WasmError,
    module::type_defs::CompositeType,
    vm::{
        entities::{FunctionInst, ModuleInst},
        runtime::{code::CompiledNativeModule, dispatch_view::NativeDispatchMetadata},
        store::Store,
        tag::TagHandle,
        value::RefHandle,
    },
};

/// Runtime escape state for a single wasm invocation. Thrown wasm exceptions
/// are carried separately from host-level `WasmError` traps so future
/// catch-dispatcher blocks can discriminate at the call boundary.
#[derive(Debug, Default)]
pub(crate) enum PendingEscape {
    #[default]
    None,
    /// Uncaught wasm exception — carries an `exnref` pool handle that
    /// dereferences through `Store::exn_heap()` to the payload.
    Throw { exn: RefHandle, tag: TagHandle },
}

impl PendingEscape {
    #[inline]
    pub(crate) fn into_error(self) -> Option<WasmError> {
        match self {
            Self::None => None,
            Self::Throw { exn, tag } => Some(WasmError::Exception {
                exn,
                tag,
                module_tag_name: None,
            }),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct NativeMemoryView {
    pub(crate) base: *mut u8,
    pub(crate) len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct NativeTableView {
    pub(crate) elements_base: *mut RefHandle,
    pub(crate) elements_len: usize,
}

pub(crate) use super::dispatch_view::function_kind;
pub(crate) use super::dispatch_view::CallDispatchView as NativeFunctionView;

#[repr(C)]
#[derive(Debug)]
pub(crate) struct NativeContext {
    pub(crate) stack_end: *mut u64,
    pub(crate) mem0_base: *mut u8,
    pub(crate) mem0_size: u64,
    pub(crate) memory_views_base: *const NativeMemoryView,
    pub(crate) memory_views_len: usize,
    pub(crate) table_views_base: *const NativeTableView,
    pub(crate) table_views_len: usize,
    pub(crate) function_views_base: *const NativeFunctionView,
    pub(crate) function_views_len: usize,
    pub(crate) local_call_infos_base: *const u8,
    pub(crate) local_call_infos_len: usize,
    pub(crate) type_canon_base: *const u32,
    pub(crate) type_canon_len: usize,
    /// Number of globals the trailing `[*mut u64]` tail holds. Fixed for the
    /// life of the context (modules cannot change their global count at
    /// runtime). Kept in the ABI-visible area only for the emulator's
    /// bounds-checking; JIT code never reloads it.
    pub(crate) globals_len: usize,
    pub(crate) store: *mut Store,
    pub(crate) current_module: *const ModuleInst,
    pub(crate) error: Option<WasmError>,
    /// Pending wasm exception / trap in transit between runtime-call entry
    /// and the catch-dispatcher block (or the function escape path).
    pub(crate) pending_escape: PendingEscape,
    /// Trap kind set by the guard-page signal handler (no allocation needed).
    /// 0 = no trap, 1 = memory out of bounds.
    #[cfg(sf_has_guard_pages)]
    pub(crate) trap_kind: u32,
    memory_views: collections::Vec<NativeMemoryView>,
    table_views: collections::Vec<NativeTableView>,
    function_views: collections::Vec<NativeFunctionView>,
    type_canon: collections::Vec<u32>,
    #[cfg(sf_call_trace)]
    pub(crate) trace_stack: collections::Vec<u32>,
    /// Zero-sized anchor marking the start of the trailing raw-ptr array.
    ///
    /// Uses `[u64; 0]` rather than `[*mut u64; 0]` so the marker's alignment
    /// (8) matches the struct's natural alignment on every target —
    /// otherwise on 32-bit hosts with 8-aligned `u64` fields, the tail
    /// marker would land at a 4-aligned offset while the struct size rounds
    /// up to 8, leaving stray padding between `offset_of!(tail)` and
    /// `size_of`. JIT code addresses the tail via
    /// `size_of::<NativeContext>()`, so the two must agree; this is checked
    /// by the `_GLOBALS_TAIL_IS_LAST` const assert.
    ///
    /// The actual tail data is `[*mut u64; globals_len]` (one host pointer
    /// per global); only the *starting offset* is what this marker pins.
    pub(crate) globals_ptrs_tail: [u64; 0],
}

impl NativeContext {
    /// Allocate and initialize a runtime context, returning an owning handle.
    ///
    /// The allocation size is `size_of::<NativeContext>() + n_globals *
    /// size_of::<*mut u64>()`; the trailing raw-ptr slots are zeroed. When
    /// `store` is non-null the initial `refresh_cached_views` call pulls the
    /// per-global `raw_ptr` into the tail; otherwise the tail stays nulled.
    ///
    /// Callers pass the wasm module's `globals.len()` as `n_globals`. The
    /// count is fixed for the life of the context: modules cannot change
    /// their global count at runtime, and `refresh_globals_ptrs` only
    /// overwrites existing slots.
    #[inline]
    pub(crate) fn new(
        store: *mut Store,
        stack_end: *mut u64,
        n_globals: usize,
    ) -> NativeContextBox {
        NativeContextBox::new(store, stack_end, n_globals)
    }

    #[inline]
    pub(crate) fn refresh_cached_views(&mut self) {
        if let Some(store) = self.store() {
            self.current_module = store.module() as *const ModuleInst;
        } else {
            self.current_module = core::ptr::null();
        }
        self.refresh_globals_ptrs();
        self.refresh_memory_views();
        self.refresh_table_views();
        self.refresh_type_canon();
        self.refresh_function_views();
    }

    #[inline]
    pub(crate) fn seed_local_call_infos(&mut self, compiled: &CompiledNativeModule) {
        self.seed_dispatch_metadata(compiled.dispatch_metadata());
    }

    #[inline]
    pub(crate) fn seed_dispatch_metadata(&mut self, metadata: &NativeDispatchMetadata) {
        self.local_call_infos_base = metadata.local_call_infos().base();
        self.local_call_infos_len = metadata.local_call_infos().len();
    }

    #[inline]
    pub(crate) fn refresh_memory_views(&mut self) {
        let Some(store) = self.store() else {
            self.mem0_base = core::ptr::null_mut();
            self.mem0_size = 0;
            self.memory_views.clear();
            self.memory_views_base = core::ptr::null();
            self.memory_views_len = 0;
            return;
        };

        let memory_views: collections::Vec<_> = store
            .module()
            .memories
            .iter()
            .map(|memory| {
                let ptr = memory.memory_ptr();
                let len = memory.memory_len();
                NativeMemoryView {
                    base: if len == 0 { core::ptr::null_mut() } else { ptr },
                    len,
                }
            })
            .collect();

        self.memory_views = memory_views;

        self.memory_views_base = if self.memory_views.is_empty() {
            core::ptr::null()
        } else {
            self.memory_views.as_ptr()
        };
        self.memory_views_len = self.memory_views.len();

        if let Some(view) = self.memory_views.first().copied() {
            self.mem0_base = view.base;
            self.mem0_size = view.len as u64;
        } else {
            self.mem0_base = core::ptr::null_mut();
            self.mem0_size = 0;
        }
    }

    #[inline]
    pub(crate) fn refresh_table_views(&mut self) {
        let Some(store) = self.store() else {
            self.table_views.clear();
            self.table_views_base = core::ptr::null();
            self.table_views_len = 0;
            return;
        };

        let table_views: collections::Vec<_> = store
            .module()
            .tables
            .iter()
            .map(|table| {
                let elements = table.elements();
                NativeTableView {
                    elements_base: if elements.is_empty() {
                        core::ptr::null_mut()
                    } else {
                        elements.as_ptr() as *mut RefHandle
                    },
                    elements_len: elements.len(),
                }
            })
            .collect();

        self.table_views = table_views;

        self.table_views_base = if self.table_views.is_empty() {
            core::ptr::null()
        } else {
            self.table_views.as_ptr()
        };
        self.table_views_len = self.table_views.len();
    }

    /// Repopulate the inline raw-ptr tail with the current per-global backing
    /// pointers. Called from [`refresh_cached_views`]; the tail length is
    /// fixed at allocation time and is never resized here.
    ///
    /// When `store` is non-null but reports fewer globals than the tail
    /// length (should not happen for a well-formed module; instance counts
    /// are baked at construction), extra tail slots are nulled defensively.
    /// When `store` is null, every slot is nulled.
    #[inline]
    pub(crate) fn refresh_globals_ptrs(&mut self) {
        // Snapshot the raw Store pointer (Copy) before taking an exclusive
        // borrow of the tail — otherwise `self.store` and `self.globals_ptrs_mut()`
        // would alias self concurrently.
        let store_ptr = self.store;
        let tail = self.globals_ptrs_mut();
        if tail.is_empty() {
            return;
        }
        let mut written = 0usize;
        // SAFETY: the pointer's provenance and lifetime are managed by the
        // owning `Instance` / `Store`; the context only reads through it
        // between refreshes. Accessing it here while `tail` is borrowed is
        // sound because the store is an independent allocation.
        if let Some(store) = unsafe { store_ptr.as_ref() } {
            let globals = &store.module().globals;
            for (slot, global) in tail.iter_mut().zip(globals.iter()) {
                *slot = global.raw_ptr();
                written += 1;
            }
        }
        for slot in tail.iter_mut().skip(written) {
            *slot = core::ptr::null_mut();
        }
    }

    /// Pointer to the first inline global raw-ptr slot, or null when there
    /// are no globals. Intended for the emulator — JIT code does not need
    /// this (it accesses the tail via constant offset from `runtime_base`).
    #[inline]
    pub(crate) fn globals_ptrs_base(&self) -> *mut *mut u64 {
        if self.globals_len == 0 {
            core::ptr::null_mut()
        } else {
            // SAFETY: the tail was allocated with `globals_len` slots and is
            // part of the same allocation as `self`.
            unsafe {
                (self as *const NativeContext as *mut u8)
                    .add(core::mem::size_of::<NativeContext>())
                    .cast::<*mut u64>()
            }
        }
    }

    /// Mutable slice over the inline raw-ptr tail.
    ///
    /// # Safety
    ///
    /// Safe to call in this crate because every `NativeContext` is
    /// constructed through [`NativeContextBox`], which allocates
    /// `globals_len` trailing slots.
    #[inline]
    fn globals_ptrs_mut(&mut self) -> &mut [*mut u64] {
        if self.globals_len == 0 {
            return &mut [];
        }
        let base = self.globals_ptrs_base();
        unsafe { core::slice::from_raw_parts_mut(base, self.globals_len) }
    }

    #[inline]
    pub(crate) fn refresh_function_views(&mut self) {
        let Some(store) = self.store() else {
            self.function_views.clear();
            self.function_views_base = core::ptr::null();
            self.function_views_len = 0;
            return;
        };

        let module = store.module();
        let type_canon = &self.type_canon;
        let current_store = self.store;
        let registry = store.clone_function_registry();
        let invalid_view = NativeFunctionView {
            kind: function_kind::INVALID,
            type_canon: u32::MAX,
            local_target: u32::MAX,
        };
        let function_views: collections::Vec<_> = registry
            .borrow()
            .iter()
            .map(|entry| {
                let Some(owner_store) = (unsafe { entry.store.as_ref() }) else {
                    return invalid_view;
                };
                let Some(func) = owner_store.module().functions.get(entry.local_index) else {
                    return invalid_view;
                };
                let local_index = entry.local_index;
                let is_local =
                    core::ptr::eq(owner_store as *const Store, current_store as *const Store)
                        && matches!(func, FunctionInst::Local { .. });
                let (kind, local_target) = if is_local {
                    (function_kind::LOCAL, local_index as u32)
                } else {
                    (function_kind::EXTERNAL, u32::MAX)
                };
                let func_type = func.func_type();
                let type_canon = match func {
                    FunctionInst::Local { type_index, .. } if is_local => type_canon
                        .get(*type_index as usize)
                        .copied()
                        .unwrap_or(u32::MAX),
                    _ => module
                        .types
                        .as_slice()
                        .iter()
                        .position(|candidate| {
                            matches!(
                                &candidate.composite,
                                CompositeType::Func(candidate_func)
                                    if candidate_func.as_ref() == func_type
                            )
                        })
                        .and_then(|index| type_canon.get(index).copied())
                        .unwrap_or(u32::MAX),
                };
                NativeFunctionView {
                    kind,
                    type_canon,
                    local_target,
                }
            })
            .collect();

        self.function_views = function_views;
        self.function_views_base = if self.function_views.is_empty() {
            core::ptr::null()
        } else {
            self.function_views.as_ptr()
        };
        self.function_views_len = self.function_views.len();
    }

    #[inline]
    pub(crate) fn refresh_type_canon(&mut self) {
        let Some(store) = self.store() else {
            self.type_canon.clear();
            self.type_canon_base = core::ptr::null();
            self.type_canon_len = 0;
            return;
        };

        let type_ctx = &store.module().types;
        let mut type_canon = collections::Vec::with_capacity(type_ctx.len());
        for idx in 0..type_ctx.len() {
            let idx_u32 = idx as u32;
            let mut canonical = idx_u32;
            for prior in 0..idx {
                let prior_u32 = prior as u32;
                if type_ctx.types_equivalent(prior_u32, idx_u32) {
                    canonical = type_canon[prior];
                    break;
                }
            }
            type_canon.push(canonical);
        }

        self.type_canon = type_canon;
        self.type_canon_base = if self.type_canon.is_empty() {
            core::ptr::null()
        } else {
            self.type_canon.as_ptr()
        };
        self.type_canon_len = self.type_canon.len();
    }

    #[inline]
    pub(crate) fn store(&self) -> Option<&Store> {
        unsafe { self.store.as_ref() }
    }

    #[inline]
    pub(crate) fn store_mut(&mut self) -> Option<&mut Store> {
        unsafe { self.store.as_mut() }
    }
}

/// Owning handle for a `NativeContext` with a trailing `[*mut u64; globals_len]`
/// tail. Keeps the header and tail in one allocation so the tail lives at a
/// constant offset from `runtime_base`, which is what keeps the JIT lowering
/// for `global.get`/`global.set` to two loads per access.
///
/// The backing allocation is a `tracked_alloc::vec::Vec<u64>` — this routes
/// through the project's tracked-allocation layer (sf-nano-core must not use
/// the `alloc` crate directly). `u64`'s 8-byte alignment matches
/// `NativeContext`'s alignment, and the Vec's fixed length holds the header
/// plus `n_globals` pointer-sized tail slots (rounded up to u64 units).
/// `Box<NativeContext>` would not work here: Box's drop uses `size_of`,
/// which only sees the header, leaving the tail bytes astray.
pub(crate) struct NativeContextBox {
    /// Sized tracked-alloc backing. Indexed as `u64` for 8-byte alignment.
    /// The first `size_of::<NativeContext>() / 8` slots hold the header;
    /// the rest hold the inline globals-ptr tail. We never push/reserve, so
    /// the data pointer is stable and can be reinterpreted as
    /// `*mut NativeContext`.
    backing: tracked_alloc::vec::Vec<u64>,
}

impl NativeContextBox {
    fn new(store: *mut Store, stack_end: *mut u64, n_globals: usize) -> Self {
        // Backing size in bytes: fixed header + inline raw-ptr tail.
        let header_size = core::mem::size_of::<NativeContext>();
        let ptr_size = core::mem::size_of::<*mut u64>();
        let tail_size = n_globals
            .checked_mul(ptr_size)
            .expect("NativeContext tail size overflow");
        let total_bytes = header_size
            .checked_add(tail_size)
            .expect("NativeContext allocation size overflow");
        // Round up to u64 (8-byte) units. u64's alignment (8) is the
        // alignment we need to reinterpret as `*mut NativeContext` on every
        // supported target (including armv7 where `NativeContext` contains
        // u64 fields).
        let unit = core::mem::size_of::<u64>();
        debug_assert!(core::mem::align_of::<u64>() >= core::mem::align_of::<NativeContext>());
        let backing_units = total_bytes.div_ceil(unit);
        // `vec![0u64; N]` yields a Vec of exact length N with zero-initialised
        // contents, so the tail bytes beyond the header start as null raw_ptrs
        // before the first `refresh_globals_ptrs` overwrites them.
        let mut backing: tracked_alloc::vec::Vec<u64> = collections::vec![0u64; backing_units];
        let raw = backing.as_mut_ptr() as *mut NativeContext;
        assert!(
            !raw.is_null(),
            "tracked_alloc Vec returned a null data pointer",
        );

        // SAFETY: `raw` points at zero-initialised, 8-aligned memory sized
        // to hold the header + tail. We write the full header here; tail
        // bytes beyond are already zero from the Vec's initialisation.
        unsafe {
            raw.write(NativeContext {
                stack_end,
                mem0_base: core::ptr::null_mut(),
                mem0_size: 0,
                memory_views_base: core::ptr::null(),
                memory_views_len: 0,
                table_views_base: core::ptr::null(),
                table_views_len: 0,
                function_views_base: core::ptr::null(),
                function_views_len: 0,
                local_call_infos_base: core::ptr::null(),
                local_call_infos_len: 0,
                type_canon_base: core::ptr::null(),
                type_canon_len: 0,
                globals_len: n_globals,
                store,
                current_module: core::ptr::null(),
                error: None,
                pending_escape: PendingEscape::None,
                #[cfg(sf_has_guard_pages)]
                trap_kind: 0,
                memory_views: collections::Vec::new(),
                table_views: collections::Vec::new(),
                function_views: collections::Vec::new(),
                type_canon: collections::Vec::new(),
                #[cfg(sf_call_trace)]
                trace_stack: collections::Vec::new(),
                globals_ptrs_tail: [0u64; 0],
            });
        }
        let mut this = Self { backing };
        // Seed cached views (globals tail, memory/table/function views) from
        // the freshly bound store.
        this.refresh_cached_views();
        this
    }

    /// Reinterpret the backing's data pointer as `*mut NativeContext`. The
    /// pointer is stable for the life of `self` because `backing` is never
    /// resized after construction.
    #[inline]
    fn as_ctx_ptr(&self) -> *mut NativeContext {
        self.backing.as_ptr() as *mut NativeContext
    }
}

impl core::ops::Deref for NativeContextBox {
    type Target = NativeContext;

    #[inline]
    fn deref(&self) -> &NativeContext {
        // SAFETY: the header was initialised in `new` and the backing Vec
        // outlives any `&Self` borrow; no push/reserve ever moves it.
        unsafe { &*self.as_ctx_ptr() }
    }
}

impl core::ops::DerefMut for NativeContextBox {
    #[inline]
    fn deref_mut(&mut self) -> &mut NativeContext {
        // SAFETY: see Deref. Exclusive borrow on self implies exclusive
        // access to the underlying NativeContext.
        unsafe { &mut *self.as_ctx_ptr() }
    }
}

impl Drop for NativeContextBox {
    fn drop(&mut self) {
        // Run NativeContext's Drop on the header first (cleans up its inner
        // Vecs), then let `backing` deallocate through tracked_alloc.
        // SAFETY: the NativeContext at the backing's start was initialised
        // in `new` and has not been dropped since.
        unsafe {
            core::ptr::drop_in_place(self.as_ctx_ptr());
        }
    }
}

// The inline raw-ptr tail must sit at the very end of the struct so the
// allocated bytes past `size_of::<NativeContext>()` host the tail slots.
// `native_runtime_abi_layout` also pins `globals_ptrs_inline_offset` to
// `size_of::<NativeContext>()`, so this invariant keeps JIT addresses
// agreeing with the actual in-memory tail position on native builds.
const _GLOBALS_TAIL_IS_LAST: () = {
    assert!(
        core::mem::offset_of!(NativeContext, globals_ptrs_tail)
            == core::mem::size_of::<NativeContext>()
    );
};

pub(crate) mod ctx_offset {
    use super::NativeContext;

    #[cfg(test)]
    pub(crate) const STACK_END: u32 = core::mem::offset_of!(NativeContext, stack_end) as u32;
    pub(crate) const MEM0_BASE: u32 = core::mem::offset_of!(NativeContext, mem0_base) as u32;
    pub(crate) const MEM0_SIZE: u32 = core::mem::offset_of!(NativeContext, mem0_size) as u32;
    #[cfg(sf_has_guard_pages)]
    pub(crate) const TRAP_KIND: u32 = core::mem::offset_of!(NativeContext, trap_kind) as u32;

    #[cfg(test)]
    mod test_only {
        use super::super::NativeContext;
        pub(crate) const MEMORY_VIEWS_BASE: u32 =
            core::mem::offset_of!(NativeContext, memory_views_base) as u32;
        pub(crate) const MEMORY_VIEWS_LEN: u32 =
            core::mem::offset_of!(NativeContext, memory_views_len) as u32;
        pub(crate) const TABLE_VIEWS_BASE: u32 =
            core::mem::offset_of!(NativeContext, table_views_base) as u32;
        pub(crate) const TABLE_VIEWS_LEN: u32 =
            core::mem::offset_of!(NativeContext, table_views_len) as u32;
        pub(crate) const FUNCTION_VIEWS_BASE: u32 =
            core::mem::offset_of!(NativeContext, function_views_base) as u32;
        pub(crate) const FUNCTION_VIEWS_LEN: u32 =
            core::mem::offset_of!(NativeContext, function_views_len) as u32;
        pub(crate) const LOCAL_CALL_INFOS_BASE: u32 =
            core::mem::offset_of!(NativeContext, local_call_infos_base) as u32;
        pub(crate) const LOCAL_CALL_INFOS_LEN: u32 =
            core::mem::offset_of!(NativeContext, local_call_infos_len) as u32;
        pub(crate) const TYPE_CANON_BASE: u32 =
            core::mem::offset_of!(NativeContext, type_canon_base) as u32;
        pub(crate) const TYPE_CANON_LEN: u32 =
            core::mem::offset_of!(NativeContext, type_canon_len) as u32;
    }
    #[cfg(test)]
    pub(crate) use test_only::*;
}

#[cfg(test)]
pub(crate) mod memory_view_offset {
    use super::NativeMemoryView;

    pub(crate) const BASE: u32 = core::mem::offset_of!(NativeMemoryView, base) as u32;
    pub(crate) const LEN: u32 = core::mem::offset_of!(NativeMemoryView, len) as u32;
}

#[cfg(test)]
pub(crate) mod table_view_offset {
    use super::NativeTableView;

    pub(crate) const ELEMENTS_BASE: u32 =
        core::mem::offset_of!(NativeTableView, elements_base) as u32;
    pub(crate) const ELEMENTS_LEN: u32 =
        core::mem::offset_of!(NativeTableView, elements_len) as u32;
}

#[cfg(test)]
pub(crate) mod function_view_offset {
    use super::NativeFunctionView;

    pub(crate) const KIND: u32 = core::mem::offset_of!(NativeFunctionView, kind) as u32;
    pub(crate) const TYPE_CANON: u32 = core::mem::offset_of!(NativeFunctionView, type_canon) as u32;
    pub(crate) const LOCAL_TARGET: u32 =
        core::mem::offset_of!(NativeFunctionView, local_target) as u32;
}

#[cfg(test)]
mod tests {
    use tracked_alloc::{boxed::Box, rc::Rc, string::String};

    use super::*;
    use crate::{
        module::{
            entities::FunctionSpec,
            type_context::TypeContext,
            type_defs::{CompositeType, DefType, FunctionType},
        },
        value_type::ValueType,
        vm::{
            entities::{Caller, ModuleInst},
            store::{LinkRegistry, Store},
            value::Value,
        },
    };

    fn host_noop(
        _caller: &mut Caller<'_>,
        _args: &[Value],
        _results: &mut [Value],
    ) -> Result<(), WasmError> {
        Ok(())
    }

    fn func_def(
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

    fn store_with_registry(module: ModuleInst, registry: &LinkRegistry) -> Box<Store> {
        Box::new(Store::new_with_registries(
            module,
            registry.function_registry_shared(),
            registry.ref_registry_shared(),
            #[cfg(sf_has_simd)]
            registry.simd_registry_shared(),
        ))
    }

    fn module_with_local(name: &str) -> ModuleInst {
        let func_type = Rc::new(FunctionType::new(collections::vec![], collections::vec![]));
        let types = TypeContext::new(collections::vec![Rc::new(DefType {
            composite: CompositeType::Func(Rc::clone(&func_type)),
            supertypes: collections::vec![],
            is_final: true,
            rec_group: None,
        })]);
        let mut module = ModuleInst::new(String::from(name), types);
        module.functions.push(FunctionInst::Local {
            spec: FunctionSpec::new(func_type, 1),
            type_index: 0,
        });
        module
    }

    #[test]
    fn refresh_function_views_canonicalizes_equivalent_type_indices() {
        let types = TypeContext::new(collections::vec![
            func_def(
                collections::vec![ValueType::I32],
                collections::vec![ValueType::I64],
            ),
            func_def(
                collections::vec![ValueType::I32],
                collections::vec![ValueType::I64],
            ),
        ]);
        let mut module = ModuleInst::new(String::from("m"), types);
        module.functions.push(FunctionInst::Local {
            spec: FunctionSpec::new(
                Rc::new(FunctionType::new(
                    collections::vec![ValueType::I32],
                    collections::vec![ValueType::I64],
                )),
                1,
            ),
            type_index: 1,
        });
        let mut store = Box::new(Store::new(module));
        for func_idx in 0..store.module().functions.len() {
            let _ = store.register_local_function(func_idx);
        }
        let n_globals = store.module().globals.len();
        let ctx = NativeContext::new(
            (&mut *store) as *mut Store,
            core::ptr::null_mut(),
            n_globals,
        );

        assert_eq!(ctx.type_canon_len, 2);
        let type_canon =
            unsafe { core::slice::from_raw_parts(ctx.type_canon_base, ctx.type_canon_len) };
        assert_eq!(type_canon, &[0, 0]);
        assert_eq!(ctx.function_views_len, 1);
        let view = unsafe { &*ctx.function_views_base };
        assert_eq!(view.type_canon, 0);
    }

    #[test]
    fn refresh_function_views_preserves_dead_registry_slots() {
        let registry = LinkRegistry::new();
        {
            let mut dropped_store = store_with_registry(module_with_local("dropped"), &registry);
            let dropped_handle = dropped_store.register_local_function(0);
            assert_eq!(dropped_handle.payload(), 0);
        }

        let mut live_store = store_with_registry(module_with_local("live"), &registry);
        let live_handle = live_store.register_local_function(0);
        assert_eq!(live_handle.payload(), 1);

        let n_globals = live_store.module().globals.len();
        let ctx = NativeContext::new(
            (&mut *live_store) as *mut Store,
            core::ptr::null_mut(),
            n_globals,
        );

        assert_eq!(ctx.function_views_len, 2);
        let views =
            unsafe { core::slice::from_raw_parts(ctx.function_views_base, ctx.function_views_len) };
        assert_eq!(
            views[0],
            NativeFunctionView {
                kind: function_kind::INVALID,
                type_canon: u32::MAX,
                local_target: u32::MAX,
            }
        );
        assert_eq!(
            views[1],
            NativeFunctionView {
                kind: function_kind::LOCAL,
                type_canon: 0,
                local_target: 0,
            }
        );
    }

    #[test]
    fn refresh_function_views_canonicalizes_equivalent_host_signatures() {
        let types = TypeContext::new(collections::vec![
            func_def(
                collections::vec![ValueType::I32],
                collections::vec![ValueType::I64],
            ),
            func_def(
                collections::vec![ValueType::I32],
                collections::vec![ValueType::I64],
            ),
        ]);
        let mut module = ModuleInst::new(String::from("m"), types);
        module.functions.push(FunctionInst::Host {
            func_type: Rc::new(FunctionType::new(
                collections::vec![ValueType::I32],
                collections::vec![ValueType::I64],
            )),
            callback: host_noop,
        });
        let mut store = Box::new(Store::new(module));
        for func_idx in 0..store.module().functions.len() {
            let _ = store.register_local_function(func_idx);
        }
        let n_globals = store.module().globals.len();
        let ctx = NativeContext::new(
            (&mut *store) as *mut Store,
            core::ptr::null_mut(),
            n_globals,
        );

        assert_eq!(ctx.type_canon_len, 2);
        assert_eq!(ctx.function_views_len, 1);
        let view = unsafe { &*ctx.function_views_base };
        assert_eq!(view.type_canon, 0);
    }
}
