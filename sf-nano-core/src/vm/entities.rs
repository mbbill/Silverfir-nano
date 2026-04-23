//! WebAssembly 2.0 Runtime Instances (no_std, single-module).

use crate::collections;

use core::cell::{Ref, RefCell, RefMut, UnsafeCell};
use tracked_alloc::{rc::Rc, string::String};

use crate::error::WasmError;
use crate::module::{entities::FunctionSpec, type_context::TypeContext, type_defs::FunctionType};
use crate::utils::limits::Limits;
use crate::value_type::ValueType;

#[cfg(sf_jit)]
use crate::vm::runtime::code_buf::CodeBuffer;
#[cfg(sf_has_guard_pages)]
use crate::vm::runtime::guard_pages::GuardPageMemory;
use crate::vm::store::Store;
use crate::vm::tag::TagHandle;
use crate::vm::value::{RefHandle, Value};

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
use crate::vm::value_encoding::{try_raw_to_value_in_store, value_to_raw_in_store};

pub type HostFn = fn(&mut Caller, &[Value], &mut [Value]) -> Result<(), WasmError>;

pub struct Caller<'a> {
    memory: Option<&'a mut [u8]>,
}

impl<'a> Caller<'a> {
    #[inline]
    pub fn new(memory: Option<&'a mut [u8]>) -> Self {
        Self { memory }
    }

    #[inline]
    pub fn memory(&self) -> Option<&[u8]> {
        self.memory.as_deref()
    }

    #[inline]
    pub fn memory_mut(&mut self) -> Option<&mut [u8]> {
        self.memory.as_deref_mut()
    }

    /// Construct a wasm-catchable throw from host code. Use as:
    ///
    /// ```ignore
    /// return Err(Caller::throw(my_tag, vec![Value::I32(42)]));
    /// ```
    ///
    /// `tag` must be a live `TagHandle` (obtained via
    /// `Instance::tag_handle(...)` or `Import::tag_typed_with_handle(...)`).
    /// Payload arity and value types must match the tag's function-type
    /// params; a mistyped throw surfaces to wasm as
    /// `WasmError::Trap("host threw mistyped exception")`.
    #[inline]
    pub fn throw(tag: TagHandle, args: impl Into<collections::Vec<Value>>) -> WasmError {
        WasmError::HostThrow {
            tag,
            args: args.into(),
        }
    }
}

#[derive(Debug)]
pub enum FunctionInst {
    Local {
        spec: FunctionSpec,
        type_index: u32,
    },
    Host {
        func_type: Rc<FunctionType>,
        callback: HostFn,
    },
    Linked {
        func_type: Rc<FunctionType>,
        handle: RefHandle,
    },
}

impl FunctionInst {
    #[inline]
    pub fn func_type(&self) -> &FunctionType {
        match self {
            FunctionInst::Local { spec, .. } => spec.func_type(),
            FunctionInst::Host { func_type, .. } => func_type,
            FunctionInst::Linked { func_type, .. } => func_type,
        }
    }

    #[inline]
    pub fn type_index(&self) -> u32 {
        match self {
            FunctionInst::Local { type_index, .. } => *type_index,
            FunctionInst::Host { .. } | FunctionInst::Linked { .. } => u32::MAX,
        }
    }

    #[inline]
    pub fn is_host(&self) -> bool {
        matches!(self, FunctionInst::Host { .. })
    }

    #[inline]
    pub fn spec(&self) -> Option<&FunctionSpec> {
        match self {
            FunctionInst::Local { spec, .. } => Some(spec),
            FunctionInst::Host { .. } | FunctionInst::Linked { .. } => None,
        }
    }

    #[inline]
    pub fn linked_handle(&self) -> Option<RefHandle> {
        match self {
            FunctionInst::Linked { handle, .. } => Some(*handle),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TableInst {
    elements: Rc<RefCell<collections::Vec<RefHandle>>>,
    pub limits: Limits,
    pub value_type: ValueType,
}

impl TableInst {
    pub fn new(limits: Limits, value_type: ValueType) -> Self {
        let initial_size = limits.min();
        TableInst {
            elements: Rc::new(RefCell::new(collections::vec![
                RefHandle::null();
                initial_size
            ])),
            limits,
            value_type,
        }
    }

    #[inline]
    pub fn from_shared(
        limits: Limits,
        value_type: ValueType,
        elements: Rc<RefCell<collections::Vec<RefHandle>>>,
    ) -> Self {
        Self {
            elements,
            limits,
            value_type,
        }
    }

    #[inline]
    pub fn elements(&self) -> Ref<'_, collections::Vec<RefHandle>> {
        self.elements.borrow()
    }

    #[inline]
    pub fn elements_mut(&self) -> RefMut<'_, collections::Vec<RefHandle>> {
        self.elements.borrow_mut()
    }

    #[inline]
    pub fn clone_shared_elements(&self) -> Rc<RefCell<collections::Vec<RefHandle>>> {
        Rc::clone(&self.elements)
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.elements.borrow().len()
    }
}

#[derive(Debug)]
pub(crate) struct MemBacking {
    pub(crate) data: collections::Vec<u8>,
    #[cfg(sf_has_guard_pages)]
    pub(crate) guard: Option<GuardPageMemory>,
}

#[derive(Debug, Clone)]
pub struct MemInst {
    backing: Rc<RefCell<MemBacking>>,
    pub limits: Limits,
}

/// Enforce `runtime_config().wasm_memory_max_pages` against the memory's
/// initial page count. Declared maximums are type-level growth ceilings;
/// `memory.grow` applies the runtime cap when growth is requested.
fn check_memory_quota(limits: &Limits) -> Result<(), WasmError> {
    let initial_pages = limits.min();
    let configured = crate::runtime_config().wasm_memory_max_pages as usize;
    if initial_pages > configured {
        return Err(WasmError::memory_exceeds_runtime_limit());
    }
    Ok(())
}

impl MemInst {
    pub fn new(limits: Limits) -> Result<Self, WasmError> {
        check_memory_quota(&limits)?;
        let initial_bytes = limits.min() * crate::constants::WASM_PAGE_SIZE;
        Ok(MemInst {
            backing: Rc::new(RefCell::new(MemBacking {
                data: collections::vec![0u8; initial_bytes],
                #[cfg(sf_has_guard_pages)]
                guard: None,
            })),
            limits,
        })
    }

    /// Allocate with guard-page backing (mmap + PROT_NONE guard region).
    #[cfg(sf_has_guard_pages)]
    pub fn new_guarded(limits: Limits) -> Result<Self, WasmError> {
        check_memory_quota(&limits)?;
        let guard = GuardPageMemory::new(limits.min())?;
        Ok(MemInst {
            backing: Rc::new(RefCell::new(MemBacking {
                data: collections::Vec::new(),
                guard: Some(guard),
            })),
            limits,
        })
    }

    #[inline]
    pub(crate) fn from_shared(limits: Limits, backing: Rc<RefCell<MemBacking>>) -> Self {
        Self { backing, limits }
    }

    #[inline]
    pub(crate) fn backing_mut(&self) -> RefMut<'_, MemBacking> {
        self.backing.borrow_mut()
    }

    #[inline]
    pub(crate) fn clone_shared_backing(&self) -> Rc<RefCell<MemBacking>> {
        Rc::clone(&self.backing)
    }

    #[inline]
    pub fn current_pages(&self) -> usize {
        self.memory_len() / crate::constants::WASM_PAGE_SIZE
    }

    /// Whether this memory uses guard-page backing.
    #[inline]
    pub fn has_guard_pages(&self) -> bool {
        #[cfg(sf_has_guard_pages)]
        {
            self.backing.borrow().guard.is_some()
        }
        #[cfg(not(sf_has_guard_pages))]
        {
            false
        }
    }

    /// Pointer to the memory buffer (works for both Vec and guard-page backing).
    #[inline]
    pub fn memory_ptr(&self) -> *mut u8 {
        let backing = self.backing.borrow();
        #[cfg(sf_has_guard_pages)]
        if let Some(ref g) = backing.guard {
            return g.base();
        }
        backing.data.as_ptr() as *mut u8
    }

    /// Current committed size in bytes.
    #[inline]
    pub fn memory_len(&self) -> usize {
        let backing = self.backing.borrow();
        #[cfg(sf_has_guard_pages)]
        if let Some(ref g) = backing.guard {
            return g.len();
        }
        backing.data.len()
    }
}

#[derive(Debug)]
pub struct GlobalCell {
    raw: UnsafeCell<u64>,
}

impl GlobalCell {
    #[inline]
    fn new(raw: u64) -> Self {
        Self {
            raw: UnsafeCell::new(raw),
        }
    }

    #[inline]
    pub(crate) fn raw_ptr(&self) -> *mut u64 {
        self.raw.get()
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct GlobalInst {
    raw_ptr: *mut u64,
    cell: Rc<GlobalCell>,
    pub mutable: bool,
    pub value_type: ValueType,
}

impl GlobalInst {
    pub fn new_raw(raw: u64, mutable: bool, value_type: ValueType) -> Self {
        let cell = Rc::new(GlobalCell::new(raw));
        let raw_ptr = cell.raw_ptr();
        GlobalInst {
            raw_ptr,
            cell,
            mutable,
            value_type,
        }
    }

    pub fn from_shared(cell: Rc<GlobalCell>, mutable: bool, value_type: ValueType) -> Self {
        let raw_ptr = cell.raw_ptr();
        GlobalInst {
            raw_ptr,
            cell,
            mutable,
            value_type,
        }
    }

    #[inline]
    pub fn value(&self, store: &Store) -> Result<Value, WasmError> {
        try_raw_to_value_in_store(self.raw(), self.value_type, store)
    }

    #[inline]
    pub fn set_value(&mut self, store: &mut Store, value: Value) {
        self.set_raw(value_to_raw_in_store(value, store));
    }

    #[inline]
    pub fn raw(&self) -> u64 {
        // Safety: sf-nano stores are single-threaded; generated code and host
        // API access use the same raw cell identity for imported globals.
        unsafe { *self.raw_ptr }
    }

    #[inline]
    pub fn set_raw(&mut self, raw: u64) {
        // Safety: see `raw`.
        unsafe {
            *self.raw_ptr = raw;
        }
    }

    #[inline]
    pub fn raw_ptr(&self) -> *mut u64 {
        self.raw_ptr
    }

    #[inline]
    pub fn clone_shared_cell(&self) -> Rc<GlobalCell> {
        Rc::clone(&self.cell)
    }
}

pub(crate) mod global_offset {
    use super::GlobalInst;

    pub(crate) const RAW_PTR: u32 = core::mem::offset_of!(GlobalInst, raw_ptr) as u32;
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
    #[cfg(sf_jit)]
    native_buf: RefCell<Option<CodeBuffer>>,
}

impl ModuleInst {
    pub fn new(name: String, types: TypeContext) -> Self {
        ModuleInst {
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
            #[cfg(sf_jit)]
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

    #[cfg(sf_jit)]
    pub fn native_code_buffer(&self) -> Result<core::cell::RefMut<'_, CodeBuffer>, &'static str> {
        let mut native_buf = self
            .native_buf
            .try_borrow_mut()
            .map_err(|_| "module native code buffer is already borrowed")?;
        if native_buf.is_none() {
            *native_buf = Some(CodeBuffer::new()?);
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
    #[cfg(sf_jit)]
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
            #[cfg(sf_jit)]
            native_buf: RefCell::new(None),
        }
    }
}
