//! WebAssembly 2.0 Module Entities (no_std, single-module)
//!
//! Simplified entity definitions for a single-module WASM 2.0 interpreter:
//! - No multi-module linking (no LinkableData/LinkableInstance)
//! - No GC types
//! - Import vs local distinguished via enums

use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::ops::Deref;

use crate::constants;
use crate::error::WasmError;
use crate::utils::limits::{Limitable, Limits};
use crate::value_type::ValueType;
#[cfg(feature = "interp")]
use crate::vm::interp::fast_code::{FastCode, FastCodeCache};
#[cfg(feature = "micro-jit")]
use crate::vm::runtime::code::{NativeCode, NativeCodeCache};
use core::cell::UnsafeCell;

pub use super::type_defs::FunctionType;

// ---------------------------------------------------------------------------
// Bytecode / ConstExpr
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Bytecode {
    data: Rc<[u8]>,
}

impl From<&[u8]> for Bytecode {
    fn from(data: &[u8]) -> Self {
        Bytecode {
            data: Rc::from(data),
        }
    }
}

impl Deref for Bytecode {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl Default for Bytecode {
    fn default() -> Self {
        Bytecode { data: Rc::from([]) }
    }
}

#[derive(Clone, Debug)]
pub struct ConstExpr {
    data: Rc<[u8]>,
}

impl From<&[u8]> for ConstExpr {
    fn from(data: &[u8]) -> Self {
        ConstExpr {
            data: Rc::from(data),
        }
    }
}

impl Deref for ConstExpr {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl Default for ConstExpr {
    fn default() -> Self {
        ConstExpr { data: Rc::from([]) }
    }
}

// ---------------------------------------------------------------------------
// FunctionSpec
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct FunctionSpec {
    func_type: Rc<FunctionType>,
    type_index: u32,
    locals: Vec<ValueType>,
    code: Bytecode,
    code_offset: usize,
    #[cfg(feature = "interp")]
    fast_code: UnsafeCell<Option<FastCode>>,
    #[cfg(feature = "interp")]
    fast_cache: UnsafeCell<FastCodeCache>,
    #[cfg(feature = "micro-jit")]
    native_code: UnsafeCell<Option<NativeCode>>,
    #[cfg(feature = "micro-jit")]
    native_cache: UnsafeCell<NativeCodeCache>,
}

// SAFETY: FunctionSpec is only mutated during compilation (single-threaded).
unsafe impl Send for FunctionSpec {}
unsafe impl Sync for FunctionSpec {}

impl FunctionSpec {
    pub fn new(func_type: Rc<FunctionType>, type_index: u32) -> Self {
        FunctionSpec {
            func_type,
            type_index,
            locals: Vec::new(),
            code: Bytecode::default(),
            code_offset: 0,
            #[cfg(feature = "interp")]
            fast_code: UnsafeCell::new(None),
            #[cfg(feature = "interp")]
            fast_cache: UnsafeCell::new(FastCodeCache::default()),
            #[cfg(feature = "micro-jit")]
            native_code: UnsafeCell::new(None),
            #[cfg(feature = "micro-jit")]
            native_cache: UnsafeCell::new(NativeCodeCache::default()),
        }
    }

    pub fn locals(&self) -> &[ValueType] {
        &self.locals
    }

    pub fn set_locals(&mut self, locals: Vec<ValueType>) {
        self.locals = locals;
    }

    pub fn code(&self) -> &Bytecode {
        &self.code
    }

    pub fn set_code(&mut self, code: Bytecode) {
        self.code = code;
    }

    pub fn code_offset(&self) -> usize {
        self.code_offset
    }

    pub fn set_code_offset(&mut self, offset: usize) {
        self.code_offset = offset;
    }

    #[inline(always)]
    pub fn func_type(&self) -> &FunctionType {
        &self.func_type
    }

    #[inline]
    pub fn func_type_rc(&self) -> Rc<FunctionType> {
        self.func_type.clone()
    }

    pub fn type_index(&self) -> u32 {
        self.type_index
    }

    /// Check if fast code has been compiled for this function.
    #[cfg(feature = "interp")]
    #[inline(always)]
    pub fn has_fast_code(&self) -> bool {
        unsafe { (*self.fast_cache.get()).is_compiled() }
    }

    /// Get the fast code cache (hot-path metadata).
    #[cfg(feature = "interp")]
    #[inline(always)]
    pub fn fast_cache(&self) -> FastCodeCache {
        unsafe { *self.fast_cache.get() }
    }

    /// Get a reference to the FastCode (for iteration/patching).
    #[cfg(feature = "interp")]
    pub fn get_fast_code(&self) -> Option<&FastCode> {
        unsafe { (*self.fast_code.get()).as_ref() }
    }

    /// Set the compiled fast code and cache.
    ///
    /// # Safety
    /// Must only be called during compilation (single-threaded).
    #[cfg(feature = "interp")]
    pub fn set_fast_code(&self, code: FastCode, cache: FastCodeCache) {
        unsafe {
            *self.fast_code.get() = Some(code);
            *self.fast_cache.get() = cache;
        }
    }

    #[cfg(feature = "micro-jit")]
    #[inline(always)]
    pub fn has_native_code(&self) -> bool {
        unsafe { (*self.native_cache.get()).is_compiled() }
    }

    #[cfg(feature = "micro-jit")]
    #[inline(always)]
    pub(crate) fn native_cache(&self) -> NativeCodeCache {
        unsafe { *self.native_cache.get() }
    }

    #[cfg(feature = "micro-jit")]
    pub(crate) fn get_native_code(&self) -> Option<&NativeCode> {
        unsafe { (*self.native_code.get()).as_ref() }
    }

    #[cfg(feature = "micro-jit")]
    pub(crate) fn with_native_code_mut<R>(&self, f: impl FnOnce(&mut NativeCode) -> R) -> Option<R> {
        unsafe { (*self.native_code.get()).as_mut().map(f) }
    }

    #[cfg(feature = "micro-jit")]
    pub(crate) fn set_native_code(&self, code: NativeCode, cache: NativeCodeCache) {
        unsafe {
            *self.native_code.get() = Some(code);
            *self.native_cache.get() = cache;
        }
    }
}

// ---------------------------------------------------------------------------
// Function (import vs local enum)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum FunctionDef {
    Local(FunctionSpec),
    Import {
        module: String,
        name: String,
        type_index: u32,
        func_type: Rc<FunctionType>,
    },
}

#[derive(Debug)]
pub struct Function {
    export_names: Vec<String>,
    def: FunctionDef,
}

impl Function {
    pub fn new_local(func_type: Rc<FunctionType>, type_index: u32) -> Self {
        Function {
            export_names: Vec::new(),
            def: FunctionDef::Local(FunctionSpec::new(func_type, type_index)),
        }
    }

    pub fn new_import(
        module: String,
        name: String,
        func_type: Rc<FunctionType>,
        type_index: u32,
    ) -> Self {
        Function {
            export_names: Vec::new(),
            def: FunctionDef::Import {
                module,
                name,
                type_index,
                func_type,
            },
        }
    }

    pub fn def(&self) -> &FunctionDef {
        &self.def
    }

    pub fn def_mut(&mut self) -> &mut FunctionDef {
        &mut self.def
    }

    pub fn is_import(&self) -> bool {
        matches!(self.def, FunctionDef::Import { .. })
    }

    /// Returns a reference to the function type.
    #[inline(always)]
    pub fn func_type(&self) -> &FunctionType {
        match &self.def {
            FunctionDef::Local(spec) => spec.func_type(),
            FunctionDef::Import { func_type, .. } => func_type,
        }
    }

    #[inline]
    pub fn func_type_rc(&self) -> Rc<FunctionType> {
        match &self.def {
            FunctionDef::Local(spec) => spec.func_type_rc(),
            FunctionDef::Import { func_type, .. } => func_type.clone(),
        }
    }

    pub fn type_index(&self) -> u32 {
        match &self.def {
            FunctionDef::Local(spec) => spec.type_index(),
            FunctionDef::Import { type_index, .. } => *type_index,
        }
    }

    pub fn spec(&self) -> Option<&FunctionSpec> {
        match &self.def {
            FunctionDef::Local(spec) => Some(spec),
            FunctionDef::Import { .. } => None,
        }
    }

    pub fn spec_mut(&mut self) -> Option<&mut FunctionSpec> {
        match &mut self.def {
            FunctionDef::Local(spec) => Some(spec),
            FunctionDef::Import { .. } => None,
        }
    }

    pub fn export_names(&self) -> &[String] {
        &self.export_names
    }

    pub fn add_export_name(&mut self, name: String) {
        self.export_names.push(name);
    }

    /// Consume the Function, returning its export names and definition.
    pub fn into_parts(self) -> (Vec<String>, FunctionDef) {
        (self.export_names, self.def)
    }
}

// ---------------------------------------------------------------------------
// TableSpec / Table
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TableSpec {
    value_type: ValueType,
    limits: Limits,
}

impl Limitable for TableSpec {
    fn limits(&self) -> &Limits {
        &self.limits
    }
}

impl TableSpec {
    pub fn new(value_type: ValueType, limits: Limits) -> Result<Self, WasmError> {
        let default_max = if limits.is64 {
            constants::MAX_TABLE_SIZE_64
        } else {
            constants::MAX_TABLE_SIZE
        };
        Ok(TableSpec {
            value_type,
            limits: limits
                .with_default_max(default_max)
                .map_err(|e| crate::wasm_error!(invalid, "table limits: {}", e))?,
        })
    }

    pub fn value_type(&self) -> ValueType {
        self.value_type
    }
}

#[derive(Debug, Clone)]
pub enum TableDef {
    Local(TableSpec),
    Import {
        module: String,
        name: String,
        spec: TableSpec,
    },
}

#[derive(Debug, Clone)]
pub struct Table {
    export_names: Vec<String>,
    def: TableDef,
}

impl Table {
    pub fn new_local(value_type: ValueType, limits: Limits) -> Result<Self, WasmError> {
        Ok(Table {
            export_names: Vec::new(),
            def: TableDef::Local(TableSpec::new(value_type, limits)?),
        })
    }

    pub fn new_import(
        module: String,
        name: String,
        value_type: ValueType,
        limits: Limits,
    ) -> Result<Self, WasmError> {
        Ok(Table {
            export_names: Vec::new(),
            def: TableDef::Import {
                module,
                name,
                spec: TableSpec::new(value_type, limits)?,
            },
        })
    }

    pub fn def(&self) -> &TableDef {
        &self.def
    }

    pub fn is_import(&self) -> bool {
        matches!(self.def, TableDef::Import { .. })
    }

    pub fn spec(&self) -> &TableSpec {
        match &self.def {
            TableDef::Local(spec) => spec,
            TableDef::Import { spec, .. } => spec,
        }
    }

    pub fn value_type(&self) -> ValueType {
        self.spec().value_type()
    }

    pub fn export_names(&self) -> &[String] {
        &self.export_names
    }

    pub fn add_export_name(&mut self, name: String) {
        self.export_names.push(name);
    }
}

impl Limitable for Table {
    fn limits(&self) -> &Limits {
        self.spec().limits()
    }
}

// ---------------------------------------------------------------------------
// MemorySpec / Memory
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MemorySpec {
    limits: Limits,
}

impl Limitable for MemorySpec {
    fn limits(&self) -> &Limits {
        &self.limits
    }
}

impl MemorySpec {
    pub fn new(limits: Limits) -> Result<Self, WasmError> {
        let default_max = if limits.is64 {
            constants::MAX_MEM_PAGES_64
        } else {
            constants::MAX_MEM_PAGES
        };
        Ok(MemorySpec {
            limits: limits
                .with_default_max(default_max)
                .map_err(|e| crate::wasm_error!(invalid, "memory limits: {}", e))?,
        })
    }
}

#[derive(Debug, Clone)]
pub enum MemoryDef {
    Local(MemorySpec),
    Import {
        module: String,
        name: String,
        spec: MemorySpec,
    },
}

#[derive(Debug, Clone)]
pub struct Memory {
    export_names: Vec<String>,
    def: MemoryDef,
}

impl Memory {
    pub fn new_local(limits: Limits) -> Result<Self, WasmError> {
        Ok(Memory {
            export_names: Vec::new(),
            def: MemoryDef::Local(MemorySpec::new(limits)?),
        })
    }

    pub fn new_import(module: String, name: String, limits: Limits) -> Result<Self, WasmError> {
        Ok(Memory {
            export_names: Vec::new(),
            def: MemoryDef::Import {
                module,
                name,
                spec: MemorySpec::new(limits)?,
            },
        })
    }

    pub fn def(&self) -> &MemoryDef {
        &self.def
    }

    pub fn is_import(&self) -> bool {
        matches!(self.def, MemoryDef::Import { .. })
    }

    pub fn spec(&self) -> &MemorySpec {
        match &self.def {
            MemoryDef::Local(spec) => spec,
            MemoryDef::Import { spec, .. } => spec,
        }
    }

    pub fn export_names(&self) -> &[String] {
        &self.export_names
    }

    pub fn add_export_name(&mut self, name: String) {
        self.export_names.push(name);
    }
}

impl Limitable for Memory {
    fn limits(&self) -> &Limits {
        self.spec().limits()
    }
}

// ---------------------------------------------------------------------------
// GlobalSpec / Global
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GlobalSpec {
    value_type: ValueType,
    mutable: bool,
    init_expr: ConstExpr,
}

impl GlobalSpec {
    pub fn new(value_type: ValueType, mutable: bool, init_expr: ConstExpr) -> Self {
        GlobalSpec {
            value_type,
            mutable,
            init_expr,
        }
    }

    pub fn value_type(&self) -> ValueType {
        self.value_type
    }

    pub fn mutable(&self) -> bool {
        self.mutable
    }

    pub fn init_expr(&self) -> &ConstExpr {
        &self.init_expr
    }
}

#[derive(Debug, Clone)]
pub enum GlobalDef {
    Local(GlobalSpec),
    Import {
        module: String,
        name: String,
        value_type: ValueType,
        mutable: bool,
    },
}

#[derive(Debug, Clone)]
pub struct Global {
    export_names: Vec<String>,
    def: GlobalDef,
}

impl Global {
    pub fn new_local(value_type: ValueType, mutable: bool, init_expr: ConstExpr) -> Self {
        Global {
            export_names: Vec::new(),
            def: GlobalDef::Local(GlobalSpec::new(value_type, mutable, init_expr)),
        }
    }

    pub fn new_import(module: String, name: String, value_type: ValueType, mutable: bool) -> Self {
        Global {
            export_names: Vec::new(),
            def: GlobalDef::Import {
                module,
                name,
                value_type,
                mutable,
            },
        }
    }

    pub fn def(&self) -> &GlobalDef {
        &self.def
    }

    pub fn is_import(&self) -> bool {
        matches!(self.def, GlobalDef::Import { .. })
    }

    pub fn value_type(&self) -> ValueType {
        match &self.def {
            GlobalDef::Local(spec) => spec.value_type(),
            GlobalDef::Import { value_type, .. } => *value_type,
        }
    }

    pub fn mutable(&self) -> bool {
        match &self.def {
            GlobalDef::Local(spec) => spec.mutable(),
            GlobalDef::Import { mutable, .. } => *mutable,
        }
    }

    pub fn spec(&self) -> Option<&GlobalSpec> {
        match &self.def {
            GlobalDef::Local(spec) => Some(spec),
            GlobalDef::Import { .. } => None,
        }
    }

    pub fn export_names(&self) -> &[String] {
        &self.export_names
    }

    pub fn add_export_name(&mut self, name: String) {
        self.export_names.push(name);
    }
}

// ---------------------------------------------------------------------------
// ElementInit / Element
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum ElementInit {
    FunctionIndexes(Vec<usize>),
    InitExprs {
        value_type: ValueType,
        exprs: Vec<ConstExpr>,
    },
}

impl ElementInit {
    pub fn len(&self) -> usize {
        match self {
            ElementInit::FunctionIndexes(vec) => vec.len(),
            ElementInit::InitExprs { exprs, .. } => exprs.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn value_type(&self) -> ValueType {
        match self {
            ElementInit::FunctionIndexes(_) => ValueType::funcref(),
            ElementInit::InitExprs { value_type, .. } => *value_type,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Element {
    Active {
        table_index: usize,
        offset_expr: ConstExpr,
        init: ElementInit,
    },
    Passive {
        init: ElementInit,
    },
    Declarative {
        init: ElementInit,
    },
}

impl Element {
    pub fn new_active(table_index: usize, offset_expr: ConstExpr, init: ElementInit) -> Self {
        Element::Active {
            table_index,
            offset_expr,
            init,
        }
    }

    pub fn new_passive(init: ElementInit) -> Self {
        Element::Passive { init }
    }

    pub fn new_declarative(init: ElementInit) -> Self {
        Element::Declarative { init }
    }

    pub fn get_init(&self) -> &ElementInit {
        match self {
            Element::Active { init, .. }
            | Element::Passive { init }
            | Element::Declarative { init } => init,
        }
    }

    pub fn value_type(&self) -> ValueType {
        self.get_init().value_type()
    }
}

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Data {
    Active {
        memory_index: usize,
        offset_expr: ConstExpr,
        init: Rc<[u8]>,
    },
    Passive {
        memory_index: usize,
        init: Rc<[u8]>,
    },
}

impl Data {
    pub fn new_active(memory_index: usize, offset_expr: ConstExpr, init: &[u8]) -> Self {
        Data::Active {
            memory_index,
            offset_expr,
            init: init.into(),
        }
    }

    pub fn new_passive(memory_index: usize, init: &[u8]) -> Self {
        Data::Passive {
            memory_index,
            init: init.into(),
        }
    }

    pub fn get_init(&self) -> &[u8] {
        match self {
            Data::Active { init, .. } | Data::Passive { init, .. } => init,
        }
    }
}
