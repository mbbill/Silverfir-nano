//! Instantiation-time entities of the JIT's `Store` world, plus the JIT's
//! own extensions to the shared entity model.
//!
//! The interpreter instantiates from the parsed `Module` directly and keeps
//! flat runtime state, so nothing here exists in an interpreter-only build —
//! which is exactly why it lives inside the `jit` subtree instead of behind
//! per-item cfgs in `vm::entities`. The shared entity types (`MemInst`,
//! `TableInst`, `GlobalInst`, `FunctionInst`) stay in `vm::entities`; the
//! JIT-only operations on them live here as extension traits.

use crate::collections;
use crate::config::Config;
use crate::error::WasmError;
use crate::module::{type_context::TypeContext, type_defs::FunctionType};
use crate::value_type::ValueType;
use crate::vm::entities::{FunctionInst, GlobalInst, MemBacking, MemInst, TableInst};
use crate::vm::jit::runtime::code_buf::CodeBuffer;
use crate::vm::tag::TagHandle;
use crate::vm::value::RefHandle;
use core::cell::{Cell, RefCell};
use tracked_alloc::{rc::Rc, string::String};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TableDispatchMode {
    Generic,
    FixedLocalOnly { len: u32 },
}

/// JIT-only operations on the shared [`TableInst`]: the revision readers
/// that feed the native context's cached-view invalidation. The revision
/// *bump* stays in `TableInst::elements_mut` itself — a shared table
/// mutated by any engine must invalidate JIT views.
pub(crate) trait TableInstJit {
    fn revision(&self) -> u64;
    fn clone_shared_revision(&self) -> Rc<Cell<u64>>;
}

impl TableInstJit for TableInst {
    #[inline]
    fn revision(&self) -> u64 {
        self.revision.get()
    }

    #[inline]
    fn clone_shared_revision(&self) -> Rc<Cell<u64>> {
        Rc::clone(&self.revision)
    }
}

/// JIT-only operations on the shared [`MemInst`]: sharing a backing across
/// linked stores, and the lazy-allocation path used when guard pages will
/// (or may) take over the reservation.
pub(crate) trait MemInstJit: Sized {
    #[cfg(not(sf_has_guard_pages))]
    fn new_unallocated(
        config: &Config,
        limits: crate::utils::limits::Limits,
    ) -> Result<Self, WasmError>;
    fn from_shared(limits: crate::utils::limits::Limits, backing: Rc<RefCell<MemBacking>>) -> Self;
    fn clone_shared_backing(&self) -> Rc<RefCell<MemBacking>>;
    fn ensure_allocated(&self) -> Result<(), WasmError>;
}

impl MemInstJit for MemInst {
    #[cfg(not(sf_has_guard_pages))]
    fn new_unallocated(
        config: &Config,
        limits: crate::utils::limits::Limits,
    ) -> Result<Self, WasmError> {
        crate::vm::entities::check_memory_quota(config, &limits)?;
        Ok(MemInst {
            backing: Rc::new(RefCell::new(MemBacking {
                data: collections::Vec::new(),
                #[cfg(sf_has_guard_pages)]
                guard: None,
            })),
            limits,
        })
    }

    #[inline]
    fn from_shared(limits: crate::utils::limits::Limits, backing: Rc<RefCell<MemBacking>>) -> Self {
        MemInst { backing, limits }
    }

    #[inline]
    fn clone_shared_backing(&self) -> Rc<RefCell<MemBacking>> {
        Rc::clone(&self.backing)
    }

    /// Fill a lazily-created backing to its declared initial size. A backing
    /// is unallocated exactly when its data is shorter than the declared
    /// minimum (guard-page backings reserve through the guard instead).
    fn ensure_allocated(&self) -> Result<(), WasmError> {
        let mut backing = self.backing.borrow_mut();
        #[cfg(sf_has_guard_pages)]
        if backing.guard.is_some() {
            return Ok(());
        }
        let initial_bytes = self.limits.min() * crate::constants::WASM_PAGE_SIZE;
        if backing.data.len() < initial_bytes {
            backing.data = collections::vec![0u8; initial_bytes];
        }
        Ok(())
    }
}

// Instantiation-time entity of the JIT's `Store` world. The interpreter
// tracks tag identity with bare `TagHandle`s in its own state.
#[derive(Debug, Clone, Copy)]
pub struct TagInst {
    pub handle: TagHandle,
    pub type_index: u32,
    /// `true` when the tag is imported — its `handle` may alias a handle
    /// imported into a different tag index. Enables the static-catch
    /// matcher in the semantic decoder to fall back to the runtime
    /// throw path when it can't prove two tag indices refer to distinct
    /// runtime identities.
    pub is_import: bool,
}

#[derive(Debug, Clone)]
pub struct ElementInst {
    pub refs: collections::Vec<RefHandle>,
    pub value_type: ValueType,
    pub dropped: bool,
}

impl ElementInst {
    pub fn new(refs: collections::Vec<RefHandle>, value_type: ValueType) -> Self {
        ElementInst {
            refs,
            value_type,
            dropped: false,
        }
    }

    #[inline]
    pub fn is_dropped(&self) -> bool {
        self.dropped
    }

    pub fn drop_segment(&mut self) {
        self.refs.clear();
        self.refs.shrink_to_fit();
        self.dropped = true;
    }
}

#[derive(Debug, Clone)]
pub struct DataInst {
    pub bytes: collections::Vec<u8>,
    pub dropped: bool,
}

impl DataInst {
    pub fn new(bytes: collections::Vec<u8>) -> Self {
        DataInst {
            bytes,
            dropped: false,
        }
    }

    #[inline]
    pub fn is_dropped(&self) -> bool {
        self.dropped
    }

    pub fn drop_segment(&mut self) {
        self.bytes.clear();
        self.bytes.shrink_to_fit();
        self.dropped = true;
    }
}

#[derive(Debug)]
pub struct ModuleInst {
    /// The engine's configuration, carried here because every stage that
    /// needs a budget already has the module in hand.
    pub(crate) config: Config,
    pub name: String,
    pub types: TypeContext,
    pub functions: collections::Vec<FunctionInst>,
    pub function_handles: collections::Vec<RefHandle>,
    pub tables: collections::Vec<TableInst>,
    pub memories: collections::Vec<MemInst>,
    pub globals: collections::Vec<GlobalInst>,
    pub tags: collections::Vec<TagInst>,
    pub elements: collections::Vec<ElementInst>,
    pub data: collections::Vec<DataInst>,
    pub(crate) table_dispatch_modes: collections::Vec<TableDispatchMode>,
    native_buf: RefCell<Option<CodeBuffer>>,
}

impl ModuleInst {
    pub fn new(config: Config, name: String, types: TypeContext) -> Self {
        ModuleInst {
            config,
            name,
            types,
            functions: collections::Vec::new(),
            function_handles: collections::Vec::new(),
            tables: collections::Vec::new(),
            memories: collections::Vec::new(),
            globals: collections::Vec::new(),
            tags: collections::Vec::new(),
            elements: collections::Vec::new(),
            data: collections::Vec::new(),
            table_dispatch_modes: collections::Vec::new(),
            native_buf: RefCell::new(None),
        }
    }

    #[inline]
    pub fn get_type(&self, index: u32) -> Option<&Rc<FunctionType>> {
        self.types.get_function_type(index)
    }

    #[inline]
    pub fn function_handle(&self, index: usize) -> Option<RefHandle> {
        self.function_handles.get(index).copied()
    }

    #[inline]
    pub(crate) fn ensure_function_handle_capacity(&mut self, len: usize) {
        if self.function_handles.len() < len {
            self.function_handles.resize(len, RefHandle::null());
        }
    }

    #[inline]
    pub(crate) fn set_function_handle(&mut self, index: usize, handle: RefHandle) {
        self.ensure_function_handle_capacity(index + 1);
        self.function_handles[index] = handle;
    }

    #[inline]
    pub(crate) fn table_dispatch_modes(&self) -> &[TableDispatchMode] {
        &self.table_dispatch_modes
    }

    pub(crate) fn clear_native_code(&self) {
        for function in &self.functions {
            if let Some(spec) = function.spec() {
                spec.clear_native_code();
            }
        }
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn native_code_buffer(&self) -> Result<core::cell::RefMut<'_, CodeBuffer>, &'static str> {
        let mut native_buf = self
            .native_buf
            .try_borrow_mut()
            .map_err(|_| "module native code buffer is already borrowed")?;
        if native_buf.is_none() {
            *native_buf = Some(CodeBuffer::new(&self.config)?);
        }
        Ok(core::cell::RefMut::map(
            native_buf,
            |native_buf: &mut Option<CodeBuffer>| {
                native_buf.as_mut().expect("native code buffer initialized")
            },
        ))
    }

    /// Take ownership of an already-built `CodeBuffer` and publish it
    /// as this module's persistent native buffer. Any previous buffer
    /// is dropped (its executable region is released back to the OS
    /// layer). Used by the streaming compile pipeline to install the
    /// just-emitted code without a second allocation + swap.
    ///
    /// Returns `Err` only if a caller is currently holding a borrow of
    /// the buffer via `native_code_buffer()` — a programmer error.
    pub(crate) fn install_native_code_buffer(&self, buf: CodeBuffer) -> Result<(), &'static str> {
        let mut native_buf = self
            .native_buf
            .try_borrow_mut()
            .map_err(|_| "module native code buffer is already borrowed")?;
        *native_buf = Some(buf);
        Ok(())
    }
}

impl Default for ModuleInst {
    fn default() -> Self {
        Self {
            config: Config::new(),
            name: String::new(),
            types: TypeContext::empty(),
            functions: collections::Vec::new(),
            function_handles: collections::Vec::new(),
            tables: collections::Vec::new(),
            memories: collections::Vec::new(),
            globals: collections::Vec::new(),
            tags: collections::Vec::new(),
            elements: collections::Vec::new(),
            data: collections::Vec::new(),
            table_dispatch_modes: collections::Vec::new(),
            native_buf: RefCell::new(None),
        }
    }
}
