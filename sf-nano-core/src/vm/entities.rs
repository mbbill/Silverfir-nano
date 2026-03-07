//! WebAssembly 2.0 Runtime Instances (no_std, single-module)
//!
//! Simplified runtime instance definitions for a single-module WASM 2.0 interpreter:
//! - No multi-module linking (no LinkableInstance traits)
//! - No Rc<RefCell<>> — all instances owned directly by ModuleInst
//! - No GC heap references
//! - External functions use plain fn pointers

use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
#[cfg(feature = "micro-jit")]
use core::cell::RefCell;

use crate::error::WasmError;
use crate::module::entities::FunctionSpec;
use crate::module::type_context::TypeContext;
use crate::module::type_defs::FunctionType;
use crate::utils::limits::Limits;
use crate::value_type::ValueType;
use crate::vm::value::{RefHandle, Value};
#[cfg(feature = "micro-jit")]
use crate::vm::native::CodeBuffer;

// ---------------------------------------------------------------------------
// ExternalFn / Caller
// ---------------------------------------------------------------------------

/// Provides host functions with access to the caller's linear memory.
pub struct Caller<'a> {
    memory: Option<&'a mut [u8]>,
}

impl<'a> Caller<'a> {
    /// Create a Caller with access to the module's first memory (if any).
    pub(crate) fn new(memory: Option<&'a mut [u8]>) -> Self {
        Self { memory }
    }

    /// Returns the caller's linear memory, if one exists.
    #[inline]
    pub fn memory(&self) -> Option<&[u8]> {
        self.memory.as_deref()
    }

    /// Returns the caller's linear memory mutably, if one exists.
    #[inline]
    pub fn memory_mut(&mut self) -> Option<&mut [u8]> {
        self.memory.as_deref_mut()
    }
}

/// Function pointer type for host-provided external functions.
///
/// The `Caller` provides access to the caller's linear memory,
/// enabling WASI and other host functions that need to read/write
/// the WASM module's memory.
pub type ExternalFn = fn(&mut Caller, &[Value], &mut [Value]) -> Result<(), WasmError>;

// ---------------------------------------------------------------------------
// FunctionInst
// ---------------------------------------------------------------------------

/// A runtime function instance — either a local WASM function or an external host function.
pub enum FunctionInst {
    /// A function defined in the WASM module.
    Local {
        spec: FunctionSpec,
        type_index: u32,
        // fast_code will be added later in Phase 5
    },
    /// A host-provided external function.
    External {
        func_type: Rc<FunctionType>,
        callback: ExternalFn,
    },
}

impl FunctionInst {
    /// Returns the function type for this instance.
    #[inline(always)]
    pub fn func_type(&self) -> &FunctionType {
        match self {
            FunctionInst::Local { spec, .. } => spec.func_type(),
            FunctionInst::External { func_type, .. } => func_type,
        }
    }

    /// Returns the type index (only meaningful for local functions).
    pub fn type_index(&self) -> u32 {
        match self {
            FunctionInst::Local { type_index, .. } => *type_index,
            FunctionInst::External { .. } => u32::MAX,
        }
    }

    /// Returns `true` if this is an external (host) function.
    #[inline]
    pub fn is_external(&self) -> bool {
        matches!(self, FunctionInst::External { .. })
    }

    /// Returns the FunctionSpec for local functions, `None` for externals.
    pub fn spec(&self) -> Option<&FunctionSpec> {
        match self {
            FunctionInst::Local { spec, .. } => Some(spec),
            FunctionInst::External { .. } => None,
        }
    }
}

// ---------------------------------------------------------------------------
// TableInst
// ---------------------------------------------------------------------------

/// A runtime table instance holding reference values.
#[derive(Debug, Clone)]
pub struct TableInst {
    pub elements: Vec<RefHandle>,
    pub limits: Limits,
    pub value_type: ValueType,
}

impl TableInst {
    /// Creates a new table instance filled with null references.
    pub fn new(limits: Limits, value_type: ValueType) -> Self {
        let initial_size = limits.min();
        TableInst {
            elements: alloc::vec![RefHandle::null(); initial_size],
            limits,
            value_type,
        }
    }

    /// Returns the current number of elements.
    #[inline]
    pub fn size(&self) -> usize {
        self.elements.len()
    }
}

// ---------------------------------------------------------------------------
// MemInst
// ---------------------------------------------------------------------------

/// A runtime linear memory instance.
#[derive(Debug, Clone)]
pub struct MemInst {
    pub data: Vec<u8>,
    pub limits: Limits,
}

impl MemInst {
    /// Creates a new memory instance with the initial size from limits (in pages).
    pub fn new(limits: Limits) -> Self {
        let initial_bytes = limits.min() * crate::constants::WASM_PAGE_SIZE;
        MemInst {
            data: alloc::vec![0u8; initial_bytes],
            limits,
        }
    }

    /// Returns the current memory size in WASM pages (64 KiB each).
    #[inline]
    pub fn current_pages(&self) -> usize {
        self.data.len() / crate::constants::WASM_PAGE_SIZE
    }
}

// ---------------------------------------------------------------------------
// GlobalInst
// ---------------------------------------------------------------------------

/// A runtime global variable instance.
#[derive(Debug, Clone)]
pub struct GlobalInst {
    pub value: Value,
    pub mutable: bool,
    pub value_type: ValueType,
}

impl GlobalInst {
    pub fn new(value: Value, mutable: bool, value_type: ValueType) -> Self {
        GlobalInst {
            value,
            mutable,
            value_type,
        }
    }
}

// ---------------------------------------------------------------------------
// ElementInst
// ---------------------------------------------------------------------------

/// A runtime element segment instance for bulk table operations.
#[derive(Debug, Clone)]
pub struct ElementInst {
    pub refs: Vec<RefHandle>,
    pub value_type: ValueType,
    dropped: bool,
}

impl ElementInst {
    pub fn new(refs: Vec<RefHandle>, value_type: ValueType) -> Self {
        ElementInst {
            refs,
            value_type,
            dropped: false,
        }
    }

    /// Returns `true` if this segment has been dropped via `elem.drop`.
    #[inline]
    pub fn is_dropped(&self) -> bool {
        self.dropped
    }

    /// Drop this element segment, releasing its references.
    pub fn drop_segment(&mut self) {
        self.refs.clear();
        self.refs.shrink_to_fit();
        self.dropped = true;
    }
}

// ---------------------------------------------------------------------------
// DataInst
// ---------------------------------------------------------------------------

/// A runtime data segment instance for bulk memory operations.
#[derive(Debug, Clone)]
pub struct DataInst {
    pub bytes: Vec<u8>,
    dropped: bool,
}

impl DataInst {
    pub fn new(bytes: Vec<u8>) -> Self {
        DataInst {
            bytes,
            dropped: false,
        }
    }

    /// Returns `true` if this segment has been dropped via `data.drop`.
    #[inline]
    pub fn is_dropped(&self) -> bool {
        self.dropped
    }

    /// Drop this data segment, releasing its bytes.
    pub fn drop_segment(&mut self) {
        self.bytes.clear();
        self.bytes.shrink_to_fit();
        self.dropped = true;
    }
}

// ---------------------------------------------------------------------------
// ModuleInst
// ---------------------------------------------------------------------------

/// A simplified single-module runtime instance.
///
/// All sub-instances are owned directly — no `Rc<RefCell<>>` indirection.
pub struct ModuleInst {
    pub name: String,
    pub types: TypeContext,
    pub functions: Vec<FunctionInst>,
    pub tables: Vec<TableInst>,
    pub memories: Vec<MemInst>,
    pub globals: Vec<GlobalInst>,
    pub elements: Vec<ElementInst>,
    pub data: Vec<DataInst>,
    #[cfg(feature = "micro-jit")]
    native_buf: RefCell<Option<CodeBuffer>>,
}

impl ModuleInst {
    pub fn new(name: String, types: TypeContext) -> Self {
        ModuleInst {
            name,
            types,
            functions: Vec::new(),
            tables: Vec::new(),
            memories: Vec::new(),
            globals: Vec::new(),
            elements: Vec::new(),
            data: Vec::new(),
            #[cfg(feature = "micro-jit")]
            native_buf: RefCell::new(None),
        }
    }

    /// Look up a function type by type index.
    #[inline]
    pub fn get_type(&self, index: u32) -> Option<&Rc<FunctionType>> {
        self.types.get(index)
    }

    #[cfg(feature = "micro-jit")]
    pub fn native_code_buffer(&self) -> Result<core::cell::RefMut<'_, CodeBuffer>, &'static str> {
        let mut native_buf = self
            .native_buf
            .try_borrow_mut()
            .map_err(|_| "module native code buffer is already borrowed")?;
        if native_buf.is_none() {
            *native_buf = Some(CodeBuffer::new()?);
        }
        Ok(core::cell::RefMut::map(native_buf, |native_buf| {
            native_buf.as_mut().expect("native code buffer initialized")
        }))
    }
}
