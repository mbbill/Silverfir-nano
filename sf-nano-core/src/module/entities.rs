//! WebAssembly 2.0 Module Entities (no_std, single-module)
//!
//! Simplified entity definitions for a single-module WASM 2.0 interpreter:
//! - No multi-module linking (no LinkableData/LinkableInstance)
//! - No GC types
//! - Import vs local distinguished via enums

use crate::collections;

use core::ops::Deref;
use tracked_alloc::rc::Rc;
use tracked_alloc::string::String;

use crate::constants;
use crate::error::WasmError;
use crate::utils::limits::{Limitable, Limits};
use crate::value_type::ValueType;
#[cfg(sf_jit)]
use crate::vm::jit::runtime::code::{NativeCode, NativeCodeCache};
// Only the JIT's per-function native-code cache needs interior mutability.
#[cfg(sf_jit)]
use core::cell::UnsafeCell;

pub(crate) use super::type_defs::FunctionType;

// ---------------------------------------------------------------------------
// Bytecode / ConstExpr
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub(crate) struct Bytecode {
    data: Rc<[u8]>,
    start: usize,
    end: usize,
}

impl From<&[u8]> for Bytecode {
    fn from(data: &[u8]) -> Self {
        let end = data.len();
        Bytecode {
            data: Rc::from(data),
            start: 0,
            end,
        }
    }
}

impl Bytecode {
    /// A function-body view into one module-wide code-section allocation.
    pub(crate) fn from_shared(data: Rc<[u8]>, start: usize, end: usize) -> Result<Self, WasmError> {
        if start > end || end > data.len() {
            return Err(WasmError::malformed(
                "Function body range exceeds code section",
            ));
        }
        Ok(Self { data, start, end })
    }

    #[cfg(test)]
    pub(crate) fn shares_storage_with(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.data, &other.data)
    }
}

impl Deref for Bytecode {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.data[self.start..self.end]
    }
}

impl Default for Bytecode {
    fn default() -> Self {
        Bytecode {
            data: Rc::from([]),
            start: 0,
            end: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ConstExpr {
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
pub(crate) struct FunctionSpec {
    func_type: Rc<FunctionType>,
    type_index: u32,
    locals: collections::Vec<ValueType>,
    code: Bytecode,
    #[cfg(sf_jit)]
    native_code: UnsafeCell<Option<NativeCode>>,
    #[cfg(sf_jit)]
    native_cache: UnsafeCell<NativeCodeCache>,
}

// SAFETY: FunctionSpec is only mutated during compilation (single-threaded).
unsafe impl Send for FunctionSpec {}
unsafe impl Sync for FunctionSpec {}

impl FunctionSpec {
    pub(crate) fn new(func_type: Rc<FunctionType>, type_index: u32) -> Self {
        FunctionSpec {
            func_type,
            type_index,
            locals: collections::Vec::new(),
            code: Bytecode::default(),
            #[cfg(sf_jit)]
            native_code: UnsafeCell::new(None),
            #[cfg(sf_jit)]
            native_cache: UnsafeCell::new(NativeCodeCache::default()),
        }
    }

    pub(crate) fn locals(&self) -> &[ValueType] {
        &self.locals
    }

    pub(crate) fn set_locals(&mut self, locals: collections::Vec<ValueType>) {
        self.locals = locals;
    }

    pub(crate) fn code(&self) -> &Bytecode {
        &self.code
    }

    pub(crate) fn set_code(&mut self, code: Bytecode) {
        self.code = code;
    }

    #[inline(always)]
    pub(crate) fn func_type(&self) -> &FunctionType {
        &self.func_type
    }

    #[inline]
    pub(crate) fn func_type_rc(&self) -> Rc<FunctionType> {
        self.func_type.clone()
    }

    pub(crate) fn type_index(&self) -> u32 {
        self.type_index
    }

    #[cfg(sf_jit)]
    #[inline(always)]
    pub(crate) fn has_native_code(&self) -> bool {
        unsafe { (*self.native_cache.get()).is_compiled() }
    }

    #[cfg(sf_jit)]
    pub(crate) fn get_native_code(&self) -> Option<&NativeCode> {
        unsafe { (*self.native_code.get()).as_ref() }
    }

    #[cfg(sf_jit)]
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
pub(crate) enum FunctionDef {
    Local(FunctionSpec),
    Import {
        module: String,
        name: String,
        type_index: u32,
        func_type: Rc<FunctionType>,
    },
}

#[derive(Debug)]
pub(crate) struct Function {
    export_names: collections::Vec<String>,
    pub(crate) def: FunctionDef,
}

impl Function {
    pub(crate) fn new_local(func_type: Rc<FunctionType>, type_index: u32) -> Self {
        Function {
            export_names: collections::Vec::new(),
            def: FunctionDef::Local(FunctionSpec::new(func_type, type_index)),
        }
    }

    pub(crate) fn new_import(
        module: String,
        name: String,
        func_type: Rc<FunctionType>,
        type_index: u32,
    ) -> Self {
        Function {
            export_names: collections::Vec::new(),
            def: FunctionDef::Import {
                module,
                name,
                type_index,
                func_type,
            },
        }
    }

    pub(crate) fn is_import(&self) -> bool {
        matches!(self.def, FunctionDef::Import { .. })
    }

    /// Returns a reference to the function type.
    #[inline(always)]
    pub(crate) fn func_type(&self) -> &FunctionType {
        match &self.def {
            FunctionDef::Local(spec) => spec.func_type(),
            FunctionDef::Import { func_type, .. } => func_type,
        }
    }

    pub(crate) fn type_index(&self) -> u32 {
        match &self.def {
            FunctionDef::Local(spec) => spec.type_index(),
            FunctionDef::Import { type_index, .. } => *type_index,
        }
    }

    pub(crate) fn spec(&self) -> Option<&FunctionSpec> {
        match &self.def {
            FunctionDef::Local(spec) => Some(spec),
            FunctionDef::Import { .. } => None,
        }
    }

    pub(crate) fn spec_mut(&mut self) -> Option<&mut FunctionSpec> {
        match &mut self.def {
            FunctionDef::Local(spec) => Some(spec),
            FunctionDef::Import { .. } => None,
        }
    }

    pub(crate) fn export_names(&self) -> &[String] {
        &self.export_names
    }

    pub(crate) fn add_export_name(&mut self, name: String) {
        self.export_names.push(name);
    }
}

// ---------------------------------------------------------------------------
// TableSpec / Table
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub(crate) struct TableSpec {
    value_type: ValueType,
    limits: Limits,
    init_expr: Option<ConstExpr>,
}

impl Limitable for TableSpec {
    fn limits(&self) -> &Limits {
        &self.limits
    }
}

impl TableSpec {
    pub(crate) fn new(value_type: ValueType, limits: Limits) -> Result<Self, WasmError> {
        let default_max = if limits.is64 {
            constants::MAX_TABLE_SIZE_64
        } else {
            constants::MAX_TABLE_SIZE
        };
        Ok(TableSpec {
            value_type,
            limits: limits
                .with_default_max(default_max)
                .map_err(|_| WasmError::invalid("table limits"))?,
            init_expr: None,
        })
    }

    pub(crate) fn new_with_init(
        value_type: ValueType,
        limits: Limits,
        init_expr: ConstExpr,
    ) -> Result<Self, WasmError> {
        let default_max = if limits.is64 {
            constants::MAX_TABLE_SIZE_64
        } else {
            constants::MAX_TABLE_SIZE
        };
        Ok(TableSpec {
            value_type,
            limits: limits
                .with_default_max(default_max)
                .map_err(|_| WasmError::invalid("table limits"))?,
            init_expr: Some(init_expr),
        })
    }

    pub(crate) fn value_type(&self) -> ValueType {
        self.value_type
    }

    pub(crate) fn init_expr(&self) -> Option<&ConstExpr> {
        self.init_expr.as_ref()
    }
}

#[derive(Debug, Clone)]
pub(crate) enum TableDef {
    Local(TableSpec),
    Import {
        module: String,
        name: String,
        spec: TableSpec,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct Table {
    export_names: collections::Vec<String>,
    def: TableDef,
}

impl Table {
    pub(crate) fn new_local(value_type: ValueType, limits: Limits) -> Result<Self, WasmError> {
        Ok(Table {
            export_names: collections::Vec::new(),
            def: TableDef::Local(TableSpec::new(value_type, limits)?),
        })
    }

    pub(crate) fn new_local_with_init(
        value_type: ValueType,
        limits: Limits,
        init_expr: ConstExpr,
    ) -> Result<Self, WasmError> {
        Ok(Table {
            export_names: collections::Vec::new(),
            def: TableDef::Local(TableSpec::new_with_init(value_type, limits, init_expr)?),
        })
    }

    pub(crate) fn new_import(
        module: String,
        name: String,
        value_type: ValueType,
        limits: Limits,
    ) -> Result<Self, WasmError> {
        Ok(Table {
            export_names: collections::Vec::new(),
            def: TableDef::Import {
                module,
                name,
                spec: TableSpec::new(value_type, limits)?,
            },
        })
    }

    pub(crate) fn def(&self) -> &TableDef {
        &self.def
    }

    pub(crate) fn is_import(&self) -> bool {
        matches!(self.def, TableDef::Import { .. })
    }

    pub(crate) fn spec(&self) -> &TableSpec {
        match &self.def {
            TableDef::Local(spec) => spec,
            TableDef::Import { spec, .. } => spec,
        }
    }

    pub(crate) fn value_type(&self) -> ValueType {
        self.spec().value_type()
    }

    pub(crate) fn export_names(&self) -> &[String] {
        &self.export_names
    }

    pub(crate) fn add_export_name(&mut self, name: String) {
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
pub(crate) struct MemorySpec {
    limits: Limits,
}

impl Limitable for MemorySpec {
    fn limits(&self) -> &Limits {
        &self.limits
    }
}

impl MemorySpec {
    pub(crate) fn new(limits: Limits) -> Result<Self, WasmError> {
        let default_max = if limits.is64 {
            constants::MAX_MEM_PAGES_64
        } else {
            constants::MAX_MEM_PAGES
        };
        Ok(MemorySpec {
            limits: limits
                .with_default_max(default_max)
                .map_err(|_| WasmError::invalid("memory limits"))?,
        })
    }
}

#[derive(Debug, Clone)]
pub(crate) enum MemoryDef {
    Local(MemorySpec),
    Import {
        module: String,
        name: String,
        spec: MemorySpec,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct Memory {
    export_names: collections::Vec<String>,
    def: MemoryDef,
}

impl Memory {
    pub(crate) fn new_local(limits: Limits) -> Result<Self, WasmError> {
        Ok(Memory {
            export_names: collections::Vec::new(),
            def: MemoryDef::Local(MemorySpec::new(limits)?),
        })
    }

    pub(crate) fn new_import(
        module: String,
        name: String,
        limits: Limits,
    ) -> Result<Self, WasmError> {
        Ok(Memory {
            export_names: collections::Vec::new(),
            def: MemoryDef::Import {
                module,
                name,
                spec: MemorySpec::new(limits)?,
            },
        })
    }

    pub(crate) fn def(&self) -> &MemoryDef {
        &self.def
    }

    pub(crate) fn spec(&self) -> &MemorySpec {
        match &self.def {
            MemoryDef::Local(spec) => spec,
            MemoryDef::Import { spec, .. } => spec,
        }
    }

    pub(crate) fn export_names(&self) -> &[String] {
        &self.export_names
    }

    pub(crate) fn add_export_name(&mut self, name: String) {
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
pub(crate) struct GlobalSpec {
    value_type: ValueType,
    mutable: bool,
    init_expr: ConstExpr,
}

impl GlobalSpec {
    pub(crate) fn new(value_type: ValueType, mutable: bool, init_expr: ConstExpr) -> Self {
        GlobalSpec {
            value_type,
            mutable,
            init_expr,
        }
    }

    pub(crate) fn value_type(&self) -> ValueType {
        self.value_type
    }

    pub(crate) fn mutable(&self) -> bool {
        self.mutable
    }

    pub(crate) fn init_expr(&self) -> &ConstExpr {
        &self.init_expr
    }
}

#[derive(Debug, Clone)]
pub(crate) enum GlobalDef {
    Local(GlobalSpec),
    Import {
        module: String,
        name: String,
        value_type: ValueType,
        mutable: bool,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct Global {
    export_names: collections::Vec<String>,
    def: GlobalDef,
}

impl Global {
    pub(crate) fn new_local(value_type: ValueType, mutable: bool, init_expr: ConstExpr) -> Self {
        Global {
            export_names: collections::Vec::new(),
            def: GlobalDef::Local(GlobalSpec::new(value_type, mutable, init_expr)),
        }
    }

    pub(crate) fn new_import(
        module: String,
        name: String,
        value_type: ValueType,
        mutable: bool,
    ) -> Self {
        Global {
            export_names: collections::Vec::new(),
            def: GlobalDef::Import {
                module,
                name,
                value_type,
                mutable,
            },
        }
    }

    pub(crate) fn def(&self) -> &GlobalDef {
        &self.def
    }

    pub(crate) fn is_import(&self) -> bool {
        matches!(self.def, GlobalDef::Import { .. })
    }

    pub(crate) fn value_type(&self) -> ValueType {
        match &self.def {
            GlobalDef::Local(spec) => spec.value_type(),
            GlobalDef::Import { value_type, .. } => *value_type,
        }
    }

    pub(crate) fn mutable(&self) -> bool {
        match &self.def {
            GlobalDef::Local(spec) => spec.mutable(),
            GlobalDef::Import { mutable, .. } => *mutable,
        }
    }

    pub(crate) fn spec(&self) -> Option<&GlobalSpec> {
        match &self.def {
            GlobalDef::Local(spec) => Some(spec),
            GlobalDef::Import { .. } => None,
        }
    }

    pub(crate) fn export_names(&self) -> &[String] {
        &self.export_names
    }

    pub(crate) fn add_export_name(&mut self, name: String) {
        self.export_names.push(name);
    }
}

// ---------------------------------------------------------------------------
// TagSpec / Tag
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub(crate) struct TagSpec {
    func_type: Rc<FunctionType>,
    type_index: u32,
}

impl TagSpec {
    pub(crate) fn new(func_type: Rc<FunctionType>, type_index: u32) -> Self {
        TagSpec {
            func_type,
            type_index,
        }
    }

    pub(crate) fn func_type(&self) -> &FunctionType {
        &self.func_type
    }

    pub(crate) fn type_index(&self) -> u32 {
        self.type_index
    }
}

#[derive(Debug, Clone)]
pub(crate) enum TagDef {
    Local(TagSpec),
    Import {
        module: String,
        name: String,
        func_type: Rc<FunctionType>,
        type_index: u32,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct Tag {
    export_names: collections::Vec<String>,
    def: TagDef,
}

impl Tag {
    pub(crate) fn new_local(func_type: Rc<FunctionType>, type_index: u32) -> Self {
        Tag {
            export_names: collections::Vec::new(),
            def: TagDef::Local(TagSpec::new(func_type, type_index)),
        }
    }

    pub(crate) fn new_import(
        module: String,
        name: String,
        func_type: Rc<FunctionType>,
        type_index: u32,
    ) -> Self {
        Tag {
            export_names: collections::Vec::new(),
            def: TagDef::Import {
                module,
                name,
                func_type,
                type_index,
            },
        }
    }

    pub(crate) fn def(&self) -> &TagDef {
        &self.def
    }

    pub(crate) fn func_type(&self) -> &FunctionType {
        match &self.def {
            TagDef::Local(spec) => spec.func_type(),
            TagDef::Import { func_type, .. } => func_type,
        }
    }

    pub(crate) fn export_names(&self) -> &[String] {
        &self.export_names
    }

    pub(crate) fn add_export_name(&mut self, name: String) {
        self.export_names.push(name);
    }
}

// ---------------------------------------------------------------------------
// ElementInit / Element
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub(crate) enum ElementInit {
    FunctionIndexes(collections::Vec<usize>),
    InitExprs {
        value_type: ValueType,
        exprs: collections::Vec<ConstExpr>,
    },
}

impl ElementInit {
    pub(crate) fn value_type(&self) -> ValueType {
        match self {
            ElementInit::FunctionIndexes(_) => ValueType::funcref(),
            ElementInit::InitExprs { value_type, .. } => *value_type,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum Element {
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
    pub(crate) fn new_active(
        table_index: usize,
        offset_expr: ConstExpr,
        init: ElementInit,
    ) -> Self {
        Element::Active {
            table_index,
            offset_expr,
            init,
        }
    }

    pub(crate) fn new_passive(init: ElementInit) -> Self {
        Element::Passive { init }
    }

    pub(crate) fn new_declarative(init: ElementInit) -> Self {
        Element::Declarative { init }
    }

    pub(crate) fn get_init(&self) -> &ElementInit {
        match self {
            Element::Active { init, .. }
            | Element::Passive { init }
            | Element::Declarative { init } => init,
        }
    }

    pub(crate) fn value_type(&self) -> ValueType {
        self.get_init().value_type()
    }
}

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub(crate) enum Data {
    Active {
        memory_index: usize,
        offset_expr: ConstExpr,
        init: Rc<[u8]>,
    },
    Passive {
        init: Rc<[u8]>,
    },
}

impl Data {
    pub(crate) fn new_active(memory_index: usize, offset_expr: ConstExpr, init: &[u8]) -> Self {
        Data::Active {
            memory_index,
            offset_expr,
            init: init.into(),
        }
    }

    pub(crate) fn new_passive(init: &[u8]) -> Self {
        Data::Passive { init: init.into() }
    }

    pub(crate) fn get_init(&self) -> &[u8] {
        match self {
            Data::Active { init, .. } | Data::Passive { init, .. } => init,
        }
    }
}
