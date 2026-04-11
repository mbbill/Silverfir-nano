//! WebAssembly 2.0 Runtime Instances (no_std, single-module).

use crate::collections;

#[cfg(sf_jit)]
use core::cell::RefCell;
use tracked_alloc::{rc::Rc, string::String};

use crate::module::{entities::FunctionSpec, type_context::TypeContext, type_defs::FunctionType};
use crate::utils::limits::Limits;
use crate::value_type::ValueType;

#[cfg(sf_jit)]
use crate::vm::runtime::code_buf::CodeBuffer;
use crate::vm::value::{RefHandle, Value};

pub type ExternalFn =
    fn(&mut Caller, &[Value], &mut [Value]) -> Result<(), crate::error::WasmError>;

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
}

#[derive(Debug)]
pub enum FunctionInst {
    Local {
        spec: FunctionSpec,
        type_index: u32,
    },
    External {
        func_type: Rc<FunctionType>,
        callback: ExternalFn,
    },
}

impl FunctionInst {
    #[inline]
    pub fn func_type(&self) -> &FunctionType {
        match self {
            FunctionInst::Local { spec, .. } => spec.func_type(),
            FunctionInst::External { func_type, .. } => func_type,
        }
    }

    #[inline]
    pub fn type_index(&self) -> u32 {
        match self {
            FunctionInst::Local { type_index, .. } => *type_index,
            FunctionInst::External { .. } => u32::MAX,
        }
    }

    #[inline]
    pub fn is_external(&self) -> bool {
        matches!(self, FunctionInst::External { .. })
    }

    #[inline]
    pub fn spec(&self) -> Option<&FunctionSpec> {
        match self {
            FunctionInst::Local { spec, .. } => Some(spec),
            FunctionInst::External { .. } => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TableInst {
    pub elements: collections::Vec<RefHandle>,
    pub limits: Limits,
    pub value_type: ValueType,
}

impl TableInst {
    pub fn new(limits: Limits, value_type: ValueType) -> Self {
        let initial_size = limits.min();
        TableInst {
            elements: collections::vec![RefHandle::null(); initial_size],
            limits,
            value_type,
        }
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.elements.len()
    }
}

#[derive(Debug)]
pub struct MemInst {
    pub data: collections::Vec<u8>,
    pub limits: Limits,
    #[cfg(sf_has_guard_pages)]
    guard: Option<crate::vm::runtime::guard_pages::GuardPageMemory>,
}

impl Clone for MemInst {
    fn clone(&self) -> Self {
        MemInst {
            data: self.data.clone(),
            limits: self.limits.clone(),
            #[cfg(sf_has_guard_pages)]
            guard: None, // guard-page backing is not cloneable; clone falls back to Vec
        }
    }
}

impl MemInst {
    pub fn new(limits: Limits) -> Self {
        let initial_bytes = limits.min() * crate::constants::WASM_PAGE_SIZE;
        MemInst {
            data: collections::vec![0u8; initial_bytes],
            limits,
            #[cfg(sf_has_guard_pages)]
            guard: None,
        }
    }

    /// Allocate with guard-page backing (mmap + PROT_NONE guard region).
    #[cfg(sf_has_guard_pages)]
    pub fn new_guarded(limits: Limits) -> Result<Self, crate::error::WasmError> {
        let guard = crate::vm::runtime::guard_pages::GuardPageMemory::new(limits.min())?;
        Ok(MemInst {
            data: collections::Vec::new(),
            limits,
            guard: Some(guard),
        })
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
            self.guard.is_some()
        }
        #[cfg(not(sf_has_guard_pages))]
        {
            false
        }
    }

    /// Pointer to the memory buffer (works for both Vec and guard-page backing).
    #[inline]
    pub fn memory_ptr(&self) -> *mut u8 {
        #[cfg(sf_has_guard_pages)]
        if let Some(ref g) = self.guard {
            return g.base();
        }
        self.data.as_ptr() as *mut u8
    }

    /// Current committed size in bytes.
    #[inline]
    pub fn memory_len(&self) -> usize {
        #[cfg(sf_has_guard_pages)]
        if let Some(ref g) = self.guard {
            return g.len();
        }
        self.data.len()
    }

    /// Mutable access to the guard-page backing (for grow).
    #[cfg(sf_has_guard_pages)]
    #[inline]
    pub fn guard_mut(&mut self) -> Option<&mut crate::vm::runtime::guard_pages::GuardPageMemory> {
        self.guard.as_mut()
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct GlobalInst {
    raw: u64,
    pub mutable: bool,
    pub value_type: ValueType,
}

impl GlobalInst {
    pub fn new(value: Value, mutable: bool, value_type: ValueType) -> Self {
        GlobalInst {
            raw: value.to_raw(),
            mutable,
            value_type,
        }
    }

    #[inline]
    pub fn value(&self) -> Value {
        Value::from_raw(self.raw, self.value_type)
    }

    #[inline]
    pub fn set_value(&mut self, value: Value) {
        self.raw = value.to_raw();
    }

    #[inline]
    pub fn raw(&self) -> u64 {
        self.raw
    }

    #[inline]
    pub fn set_raw(&mut self, raw: u64) {
        self.raw = raw;
    }
}

pub(crate) mod global_offset {
    use super::GlobalInst;

    pub(crate) const RAW: u32 = core::mem::offset_of!(GlobalInst, raw) as u32;
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
    pub tables: collections::Vec<TableInst>,
    pub memories: collections::Vec<MemInst>,
    pub globals: collections::Vec<GlobalInst>,
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
            tables: collections::Vec::new(),
            memories: collections::Vec::new(),
            globals: collections::Vec::new(),
            elements: collections::Vec::new(),
            data: collections::Vec::new(),
            #[cfg(sf_jit)]
            native_buf: RefCell::new(None),
        }
    }

    #[inline]
    pub fn get_type(&self, index: u32) -> Option<&Rc<FunctionType>> {
        self.types.get(index)
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
}

impl Default for ModuleInst {
    fn default() -> Self {
        Self {
            name: String::new(),
            types: TypeContext::empty(),
            functions: collections::Vec::new(),
            tables: collections::Vec::new(),
            memories: collections::Vec::new(),
            globals: collections::Vec::new(),
            elements: collections::Vec::new(),
            data: collections::Vec::new(),
            #[cfg(sf_jit)]
            native_buf: RefCell::new(None),
        }
    }
}
