//! WebAssembly 2.0 Runtime Instances (no_std, single-module).

use crate::collections;
use crate::config::Config;

use alloc::rc::Rc as AllocRc;
use core::cell::{Cell, Ref, RefCell, RefMut, UnsafeCell};
use tracked_alloc::rc::Rc;

use crate::error::WasmError;
use crate::module::{entities::FunctionSpec, type_defs::FunctionType};
use crate::utils::limits::Limits;
use crate::value_type::ValueType;

#[cfg(sf_has_guard_pages)]
use crate::vm::jit::runtime::guard_pages::GuardPageMemory;
use crate::vm::tag::TagIdentity;
use crate::vm::value::{RefValue, Value};

/// A non-capturing host function pointer.
///
/// Kept as a public alias for source compatibility with embedders that name or
/// cast plain host functions explicitly. [`Import::func`](crate::Import::func)
/// also accepts capturing [`Fn`] callbacks.
pub type HostFn = fn(&mut Caller, &[Value], &mut [Value]) -> Result<(), WasmError>;

type DynHostCallback =
    dyn for<'a> Fn(&mut Caller<'a>, &[Value], &mut [Value]) -> Result<(), WasmError>;

/// A clonable, potentially capturing host callback.
///
/// Instances borrow their import list during construction, so callbacks use
/// shared ownership when they are installed into a module. Silverfir-nano's
/// runtime is single-threaded; captured mutable state can use interior
/// mutability such as [`core::cell::Cell`] or [`core::cell::RefCell`].
#[derive(Clone)]
pub struct HostCallback {
    callback: Rc<DynHostCallback>,
}

impl HostCallback {
    pub fn new<F>(callback: F) -> Self
    where
        F: for<'a, 'b, 'c, 'd> Fn(
                &'a mut Caller<'b>,
                &'c [Value],
                &'d mut [Value],
            ) -> Result<(), WasmError>
            + 'static,
    {
        let callback: AllocRc<DynHostCallback> = AllocRc::new(callback);
        let callback: Rc<DynHostCallback> = callback.into();
        Self { callback }
    }

    #[inline]
    pub fn call(
        &self,
        caller: &mut Caller<'_>,
        params: &[Value],
        results: &mut [Value],
    ) -> Result<(), WasmError> {
        (self.callback)(caller, params, results)
    }
}

impl core::fmt::Debug for HostCallback {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HostCallback").finish_non_exhaustive()
    }
}

enum CallerMemory<'a> {
    Borrowed(&'a mut [u8]),
    Shared(MemInst),
}

pub struct Caller<'a> {
    memory: Option<CallerMemory<'a>>,
    shared_borrow_active: Cell<bool>,
}

impl<'a> Caller<'a> {
    #[inline]
    pub fn new(memory: Option<&'a mut [u8]>) -> Self {
        Self {
            memory: memory.map(CallerMemory::Borrowed),
            shared_borrow_active: Cell::new(false),
        }
    }

    #[inline]
    pub(crate) fn from_shared_memory(memory: Option<MemInst>) -> Caller<'static> {
        Caller {
            memory: memory.map(CallerMemory::Shared),
            shared_borrow_active: Cell::new(false),
        }
    }

    fn begin_shared_borrow(&self) {
        if self.shared_borrow_active.get() {
            return;
        }
        let Some(CallerMemory::Shared(memory)) = self.memory.as_ref() else {
            return;
        };
        memory
            .begin_host_callback_borrow()
            .expect("host memory borrow must pass invocation preflight");
        self.shared_borrow_active.set(true);
    }

    #[inline]
    pub fn memory(&self) -> Option<&[u8]> {
        self.begin_shared_borrow();
        match self.memory.as_ref()? {
            CallerMemory::Borrowed(memory) => Some(memory),
            CallerMemory::Shared(memory) => {
                let ptr = memory.memory_ptr();
                let len = memory.memory_len();
                Some(unsafe { core::slice::from_raw_parts(ptr, len) })
            }
        }
    }

    #[inline]
    pub fn memory_mut(&mut self) -> Option<&mut [u8]> {
        self.begin_shared_borrow();
        match self.memory.as_mut()? {
            CallerMemory::Borrowed(memory) => Some(memory),
            CallerMemory::Shared(memory) => {
                let ptr = memory.memory_ptr();
                let len = memory.memory_len();
                Some(unsafe { core::slice::from_raw_parts_mut(ptr, len) })
            }
        }
    }

    /// Construct a wasm-catchable throw from host code. Use as:
    ///
    /// ```ignore
    /// return Err(Caller::throw(my_tag, vec![Value::I32(42)]));
    /// ```
    ///
    /// `tag` must be a live `TagIdentity` (obtained via
    /// `Instance::tag_identity(...)` or `Import::tag_typed_with_handle(...)`).
    /// Payload arity and value types must match the tag's function-type
    /// params; a mistyped throw surfaces to wasm as
    /// `WasmError::Trap("host threw mistyped exception")`.
    #[inline]
    pub fn throw(tag: TagIdentity, args: impl Into<collections::Vec<Value>>) -> WasmError {
        WasmError::HostThrow {
            tag,
            args: args.into(),
        }
    }
}

impl Drop for Caller<'_> {
    fn drop(&mut self) {
        if !self.shared_borrow_active.get() {
            return;
        }
        let Some(CallerMemory::Shared(memory)) = self.memory.as_ref() else {
            return;
        };
        memory.end_host_callback_borrow();
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
        callback: HostCallback,
    },
    Linked {
        func_type: Rc<FunctionType>,
        handle: RefValue,
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
    pub fn linked_handle(&self) -> Option<RefValue> {
        match self {
            FunctionInst::Linked { handle, .. } => Some(*handle),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TableInst {
    elements: Rc<RefCell<collections::Vec<RefValue>>>,
    // pub(crate) so `vm::jit::entities::TableInstJit` can read it; the bump
    // in `elements_mut` is unconditional because a shared table mutated by
    // any engine must invalidate JIT cached views.
    pub(crate) revision: Rc<Cell<u64>>,
    pub limits: Limits,
    pub value_type: ValueType,
}

impl TableInst {
    pub fn new(limits: Limits, value_type: ValueType) -> Self {
        let initial_size = limits.min();
        TableInst {
            elements: Rc::new(RefCell::new(collections::vec![
                RefValue::null();
                initial_size
            ])),
            revision: Rc::new(Cell::new(0)),
            limits,
            value_type,
        }
    }

    #[inline]
    pub fn from_shared(
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
    pub fn elements(&self) -> Ref<'_, collections::Vec<RefValue>> {
        self.elements.borrow()
    }

    #[inline]
    pub fn elements_mut(&self) -> RefMut<'_, collections::Vec<RefValue>> {
        self.revision.set(self.revision.get().wrapping_add(1));
        self.elements.borrow_mut()
    }

    #[inline]
    pub fn clone_shared_elements(&self) -> Rc<RefCell<collections::Vec<RefValue>>> {
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
    pub(crate) host_callback_borrowed: Cell<bool>,
    #[cfg(sf_has_guard_pages)]
    pub(crate) guard: Option<GuardPageMemory>,
}

#[derive(Debug, Clone)]
pub struct MemInst {
    pub(crate) backing: Rc<RefCell<MemBacking>>,
    pub limits: Limits,
}

/// Enforce the engine's `wasm_memory_max_pages` against the memory's
/// initial page count. Declared maximums are type-level growth ceilings;
/// `memory.grow` applies the runtime cap when growth is requested.
pub(crate) fn check_memory_quota(config: &Config, limits: &Limits) -> Result<(), WasmError> {
    let initial_pages = limits.min();
    let configured = config.get_wasm_memory_max_pages() as usize;
    if initial_pages > configured {
        return Err(WasmError::memory_exceeds_runtime_limit());
    }
    Ok(())
}

impl MemInst {
    pub fn new(config: &Config, limits: Limits) -> Result<Self, WasmError> {
        check_memory_quota(config, &limits)?;
        let initial_bytes = limits.min() * crate::constants::WASM_PAGE_SIZE;
        Ok(MemInst {
            backing: Rc::new(RefCell::new(MemBacking {
                data: collections::vec![0u8; initial_bytes],
                host_callback_borrowed: Cell::new(false),
                #[cfg(sf_has_guard_pages)]
                guard: None,
            })),
            limits,
        })
    }

    /// Allocate with guard-page backing (mmap + PROT_NONE guard region).
    #[cfg(sf_has_guard_pages)]
    pub fn new_guarded(config: &Config, limits: Limits) -> Result<Self, WasmError> {
        check_memory_quota(config, &limits)?;
        let guard = GuardPageMemory::new(limits.min())?;
        Ok(MemInst {
            backing: Rc::new(RefCell::new(MemBacking {
                data: collections::Vec::new(),
                host_callback_borrowed: Cell::new(false),
                guard: Some(guard),
            })),
            limits,
        })
    }

    #[inline]
    pub(crate) fn backing_mut(&self) -> RefMut<'_, MemBacking> {
        self.backing.borrow_mut()
    }

    #[inline]
    pub(crate) fn host_callback_borrowed(&self) -> bool {
        self.backing.borrow().host_callback_borrowed.get()
    }

    pub(crate) fn begin_host_callback_borrow(&self) -> Result<(), WasmError> {
        let backing = self.backing.borrow();
        if backing.host_callback_borrowed.replace(true) {
            return Err(WasmError::trap(
                "linear memory is already borrowed by a host callback",
            ));
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn end_host_callback_borrow(&self) {
        self.backing.borrow().host_callback_borrowed.set(false);
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
