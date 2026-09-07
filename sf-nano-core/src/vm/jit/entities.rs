//! Instantiation-time entities of the JIT's `JitInstance` world, plus the JIT's
//! own extensions to the shared entity model.
//!
//! The interpreter instantiates from the parsed `Module` directly and keeps
//! flat runtime state, so nothing here exists in an interpreter-only build —
//! which is exactly why it lives inside the `jit` subtree instead of behind
//! per-item cfgs in `vm::entities`. The shared entity types (`MemInst`,
//! `TableInst`, `GlobalInst`) stay in `vm::entities`; their JIT operations
//! live here as extensions. `FunctionInst` is the JIT-owned runtime function
//! representation and lives here in full.

use crate::collections;
use crate::config::Config;
use crate::error::WasmError;
use crate::module::{entities::FunctionSpec, type_context::TypeContext, type_defs::FunctionType};
use crate::utils::limits::Limits;
use crate::value_type::{AbstractHeapType, HeapType, ValueType};
use crate::vm::entities::{GlobalCell, GlobalInst, HostCallback, MemBacking, MemInst, TableInst};
use crate::vm::jit::runtime::code_buf::CodeBuffer;
use crate::vm::tag::TagIdentity;
use crate::vm::value::RefValue;
use core::cell::{Cell, RefCell};
use tracked_alloc::rc::Rc;
#[cfg(any(sf_ir_dump, sf_jitdump))]
use tracked_alloc::string::String;

#[derive(Debug)]
pub(crate) enum FunctionInst {
    Local {
        spec: FunctionSpec,
        type_index: u32,
    },
    Host {
        type_index: u32,
        func_type: Rc<FunctionType>,
        callback: HostCallback,
    },
    Linked {
        type_index: u32,
        func_type: Rc<FunctionType>,
        handle: RefValue,
    },
}

impl FunctionInst {
    #[inline]
    pub(crate) fn func_type(&self) -> &FunctionType {
        match self {
            FunctionInst::Local { spec, .. } => spec.func_type(),
            FunctionInst::Host { func_type, .. } => func_type,
            FunctionInst::Linked { func_type, .. } => func_type,
        }
    }

    #[inline]
    pub(crate) fn type_index(&self) -> u32 {
        match self {
            FunctionInst::Local { type_index, .. } => *type_index,
            FunctionInst::Host { type_index, .. } | FunctionInst::Linked { type_index, .. } => {
                *type_index
            }
        }
    }

    #[inline]
    pub(crate) fn spec(&self) -> Option<&FunctionSpec> {
        match self {
            FunctionInst::Local { spec, .. } => Some(spec),
            FunctionInst::Host { .. } | FunctionInst::Linked { .. } => None,
        }
    }

    #[inline]
    pub(crate) fn linked_handle(&self) -> Option<RefValue> {
        match self {
            FunctionInst::Linked { handle, .. } => Some(*handle),
            _ => None,
        }
    }
}

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
                host_callback_borrowed: Cell::new(false),
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

// Instantiation-time entity of the JIT's `JitInstance` world. The interpreter
// tracks tag identity with bare `TagIdentity`s in its own state.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TagInst {
    pub handle: TagIdentity,
    pub type_index: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct ElementInst {
    pub refs: collections::Vec<RefValue>,
    pub dropped: bool,
}

impl ElementInst {
    pub(crate) fn new(refs: collections::Vec<RefValue>) -> Self {
        ElementInst {
            refs,
            dropped: false,
        }
    }

    #[inline]
    pub(crate) fn is_dropped(&self) -> bool {
        self.dropped
    }

    pub(crate) fn drop_segment(&mut self) {
        self.refs.clear();
        self.refs.shrink_to_fit();
        self.dropped = true;
    }
}

#[derive(Debug, Clone)]
pub(crate) struct DataInst {
    pub bytes: collections::Vec<u8>,
    pub dropped: bool,
}

impl DataInst {
    pub(crate) fn new(bytes: collections::Vec<u8>) -> Self {
        DataInst {
            bytes,
            dropped: false,
        }
    }

    #[inline]
    pub(crate) fn is_dropped(&self) -> bool {
        self.dropped
    }

    pub(crate) fn drop_segment(&mut self) {
        self.bytes.clear();
        self.bytes.shrink_to_fit();
        self.dropped = true;
    }
}

#[derive(Debug)]
pub(crate) struct ModuleInst {
    /// The engine's configuration, carried here because every stage that
    /// needs a budget already has the module in hand.
    pub(crate) config: Config,
    #[cfg(any(sf_ir_dump, sf_jitdump))]
    pub name: String,
    pub types: TypeContext,
    pub functions: collections::Vec<FunctionInst>,
    pub function_handles: collections::Vec<RefValue>,
    pub tables: collections::Vec<TableInst>,
    pub memories: collections::Vec<MemInst>,
    pub globals: collections::Vec<GlobalInst>,
    pub tags: collections::Vec<TagInst>,
    pub elements: collections::Vec<ElementInst>,
    pub data: collections::Vec<DataInst>,
    pub(crate) table_dispatch_modes: collections::Vec<TableDispatchMode>,
    /// Static container reachability, keyed by module table/global index.
    ///
    /// These facts are independent of dispatch mode: growable or otherwise
    /// generic private tables remain unreachable.
    pub(crate) table_reachable: collections::Vec<bool>,
    pub(crate) global_reachable: collections::Vec<bool>,
    native_buf: RefCell<Option<CodeBuffer>>,
}

impl ModuleInst {
    pub(crate) fn new(
        config: Config,
        #[cfg(any(sf_ir_dump, sf_jitdump))] name: String,
        types: TypeContext,
    ) -> Self {
        ModuleInst {
            config,
            #[cfg(any(sf_ir_dump, sf_jitdump))]
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
            table_reachable: collections::Vec::new(),
            global_reachable: collections::Vec::new(),
            native_buf: RefCell::new(None),
        }
    }

    #[inline]
    pub(crate) fn get_type(&self, index: u32) -> Option<&Rc<FunctionType>> {
        self.types.get_function_type(index)
    }

    #[inline]
    pub(crate) fn function_handle(&self, index: usize) -> Option<RefValue> {
        self.function_handles.get(index).copied()
    }

    #[inline]
    pub(crate) fn ensure_function_handle_capacity(&mut self, len: usize) {
        if self.function_handles.len() < len {
            self.function_handles.resize(len, RefValue::null());
        }
    }

    #[inline]
    pub(crate) fn set_function_handle(&mut self, index: usize, handle: RefValue) {
        self.ensure_function_handle_capacity(index + 1);
        self.function_handles[index] = handle;
    }

    #[inline]
    pub(crate) fn table_dispatch_modes(&self) -> &[TableDispatchMode] {
        &self.table_dispatch_modes
    }

    #[inline]
    pub(crate) fn table_is_reachable(&self, index: usize) -> bool {
        self.table_reachable.get(index).copied().unwrap_or(true)
    }

    #[inline]
    pub(crate) fn global_is_reachable(&self, index: usize) -> bool {
        self.global_reachable.get(index).copied().unwrap_or(true)
    }

    #[inline]
    pub(crate) fn global_needs_funcref_retag(&self, index: usize) -> bool {
        if !self.global_is_reachable(index) {
            return false;
        }
        let Some(ValueType::Ref(ref_type)) = self.globals.get(index).map(|g| g.value_type) else {
            return false;
        };
        matches!(
            ref_type.heap_type.top_type(&self.types),
            HeapType::Abstract(AbstractHeapType::Func)
        )
    }

    pub(crate) fn config(&self) -> &Config {
        &self.config
    }

    pub(crate) fn native_code_buffer(
        &self,
    ) -> Result<core::cell::RefMut<'_, CodeBuffer>, &'static str> {
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
            #[cfg(any(sf_ir_dump, sf_jitdump))]
            name: String::new(),
            types: TypeContext::new(collections::Vec::new()),
            functions: collections::Vec::new(),
            function_handles: collections::Vec::new(),
            tables: collections::Vec::new(),
            memories: collections::Vec::new(),
            globals: collections::Vec::new(),
            tags: collections::Vec::new(),
            elements: collections::Vec::new(),
            data: collections::Vec::new(),
            table_dispatch_modes: collections::Vec::new(),
            table_reachable: collections::Vec::new(),
            global_reachable: collections::Vec::new(),
            native_buf: RefCell::new(None),
        }
    }
}

// Native instantiation rebuilds shared wrappers with the importing module's
// declared types. The interpreter retains the original shared wrappers.
impl TableInst {
    #[inline]
    pub(crate) fn from_shared(
        limits: Limits,
        value_type: ValueType,
        elements: Rc<RefCell<collections::Vec<RefValue>>>,
        revision: Rc<Cell<u64>>,
    ) -> Self {
        Self {
            elements,
            revision,
            limits,
            value_type,
        }
    }
    #[inline]
    pub(crate) fn clone_shared_elements(&self) -> Rc<RefCell<collections::Vec<RefValue>>> {
        Rc::clone(&self.elements)
    }
}

impl GlobalInst {
    pub(crate) fn from_shared(cell: Rc<GlobalCell>, mutable: bool, value_type: ValueType) -> Self {
        let raw_ptr = cell.raw_ptr();
        GlobalInst {
            raw_ptr,
            cell,
            mutable,
            value_type,
        }
    }
    #[inline]
    pub(crate) fn clone_shared_cell(&self) -> Rc<GlobalCell> {
        Rc::clone(&self.cell)
    }
}

#[cfg(sf_has_guard_pages)]
impl MemInst {
    pub(crate) fn has_guard_pages(&self) -> bool {
        self.backing.borrow().guard.is_some()
    }
}
