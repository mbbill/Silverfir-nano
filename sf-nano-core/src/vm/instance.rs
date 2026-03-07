//! Public API for sf-nano: parse, instantiate, and invoke WebAssembly modules.
//!
//! # Example
//! ```ignore
//! use sf_nano::{Instance, Value, WasmError};
//!
//! let wasm_bytes = include_bytes!("module.wasm");
//! let instance = Instance::new(wasm_bytes, &[])?;
//! let results = instance.invoke("add", &[Value::I32(1), Value::I32(2)])?;
//! ```

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use crate::constants;
use crate::error::WasmError;
use crate::module::entities::{
    Data, Element, ElementInit, FunctionDef, GlobalDef, MemoryDef, TableDef,
};
use crate::module::type_defs::FunctionType;
use crate::module::Module;
use crate::utils::limits::{Limitable, Limits};
use crate::vm::entities::{
    DataInst, ElementInst, ExternalFn, FunctionInst, GlobalInst, MemInst, ModuleInst, TableInst,
};
use crate::vm::expr_eval::eval_const_expr;
use crate::vm::interp::fast::precompile;
use crate::vm::runtime;
use crate::vm::store::Store;
use crate::vm::value::{RefHandle, Value};

/// An import to be provided when instantiating a module.
pub struct Import {
    pub module: String,
    pub name: String,
    pub value: ImportValue,
}

/// The value of an import.
pub enum ImportValue {
    /// A host function, with optional type info for signature checking.
    Func(ExternalFn, Option<FunctionType>),
    /// A global value (initial value, mutability).
    Global(Value, bool),
    /// A memory (initial pages, optional max pages).
    Memory(usize, Option<usize>),
    /// A table (initial size, optional max size).
    Table(usize, Option<usize>),
}

impl Import {
    /// Create a function import.
    pub fn func(module: &str, name: &str, f: ExternalFn) -> Self {
        Import {
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Func(f, None),
        }
    }

    /// Create a function import with type info for signature checking.
    pub fn func_typed(module: &str, name: &str, f: ExternalFn, func_type: FunctionType) -> Self {
        Import {
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Func(f, Some(func_type)),
        }
    }

    /// Create a global import.
    pub fn global(module: &str, name: &str, value: Value, mutable: bool) -> Self {
        Import {
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Global(value, mutable),
        }
    }

    /// Create a memory import.
    pub fn memory(module: &str, name: &str, initial_pages: usize, max_pages: Option<usize>) -> Self {
        Import {
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Memory(initial_pages, max_pages),
        }
    }

    /// Create a table import.
    pub fn table(module: &str, name: &str, initial_size: usize, max_size: Option<usize>) -> Self {
        Import {
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Table(initial_size, max_size),
        }
    }
}

/// A fully instantiated WebAssembly module, ready for execution.
pub struct Instance {
    store: Store,
    /// Export name → (kind, index) mapping for fast lookup.
    exports: Vec<(String, ExportKind, usize)>,
}

#[derive(Clone, Copy)]
enum ExportKind {
    Func,
    Table,
    Memory,
    Global,
}

impl Instance {
    /// Parse a WASM binary and instantiate it with the given imports.
    pub fn new(wasm_bytes: &[u8], imports: &[Import]) -> Result<Self, WasmError> {
        let module = Module::new("main", wasm_bytes)?;
        Self::from_module(module, imports)
    }

    /// Instantiate a pre-parsed module with the given imports.
    pub fn from_module(module: Module, imports: &[Import]) -> Result<Self, WasmError> {
        // Validate the module if the validate feature is enabled
        #[cfg(feature = "validate")]
        {
            use crate::module::validator::Validator;
            let mut validator = Validator::new(&module);
            validator.validate()?;
        }

        // Build export map before consuming
        let mut exports = Vec::new();
        for (i, f) in module.functions().iter().enumerate() {
            for name in f.export_names() {
                exports.push((name.clone(), ExportKind::Func, i));
            }
        }
        for (i, t) in module.tables().iter().enumerate() {
            for name in t.export_names() {
                exports.push((name.clone(), ExportKind::Table, i));
            }
        }
        for (i, m) in module.memories().iter().enumerate() {
            for name in m.export_names() {
                exports.push((name.clone(), ExportKind::Memory, i));
            }
        }
        for (i, g) in module.globals().iter().enumerate() {
            for name in g.export_names() {
                exports.push((name.clone(), ExportKind::Global, i));
            }
        }

        let start_func_index = module.start_function_index();
        let (types, mod_functions, mod_tables, mod_memories, mod_globals, mod_elements, mod_data, _start) =
            module.into_parts();

        // --- Build function instances (consuming mod_functions) ---
        let mut functions: Vec<FunctionInst> = Vec::with_capacity(mod_functions.len());
        for func in mod_functions {
            let type_index = func.type_index();
            let (_export_names, def) = func.into_parts();
            match def {
                FunctionDef::Local(spec) => {
                    functions.push(FunctionInst::Local { spec, type_index });
                }
                FunctionDef::Import {
                    module: mod_name,
                    name,
                    func_type,
                    ..
                } => {
                    let import = imports.iter().find(|i| {
                        i.module == mod_name && i.name == name
                    });
                    match import {
                        Some(Import { value: ImportValue::Func(f, ref import_type), .. }) => {
                            // Check function signature compatibility if type info provided
                            if let Some(actual_type) = import_type {
                                if actual_type.params() != func_type.params()
                                    || actual_type.results() != func_type.results()
                                {
                                    return Err(WasmError::unlinkable(format!(
                                        "incompatible import type: {}.{}",
                                        mod_name, name
                                    )));
                                }
                            }
                            functions.push(FunctionInst::External {
                                func_type,
                                callback: *f,
                            });
                        }
                        _ => {
                            return Err(WasmError::unlinkable(format!(
                                "missing function import: {}.{}",
                                mod_name, name
                            )));
                        }
                    }
                }
            }
        }

        // --- Build table instances ---
        let mut tables: Vec<TableInst> = Vec::with_capacity(mod_tables.len());
        for table in &mod_tables {
            match table.def() {
                TableDef::Local(_spec) => {
                    tables.push(TableInst::new(table.limits().clone(), table.value_type()));
                }
                TableDef::Import { module: mod_name, name, .. } => {
                    let import = imports.iter().find(|i| {
                        i.module == *mod_name && i.name == *name
                    });
                    match import {
                        Some(Import { value: ImportValue::Table(initial_size, max_size), .. }) => {
                            let declared_min = table.limits().min();
                            let declared_max = table.limits().max();
                            // Import compatibility: actual >= declared min
                            if *initial_size < declared_min {
                                return Err(WasmError::unlinkable(format!(
                                    "incompatible import type: {}.{}",
                                    mod_name, name
                                )));
                            }
                            // If declared max exists, actual max must exist and be <= declared max
                            if let Some(d_max) = declared_max {
                                match max_size {
                                    Some(a_max) if *a_max <= d_max => {}
                                    _ => {
                                        return Err(WasmError::unlinkable(format!(
                                            "incompatible import type: {}.{}",
                                            mod_name, name
                                        )));
                                    }
                                }
                            }
                            let import_limits = Limits::new(*initial_size, *max_size)?;
                            tables.push(TableInst::new(import_limits, table.value_type()));
                        }
                        _ => {
                            return Err(WasmError::unlinkable(format!(
                                "missing table import: {}.{}",
                                mod_name, name
                            )));
                        }
                    }
                }
            }
        }

        // --- Build memory instances ---
        let mut memories: Vec<MemInst> = Vec::with_capacity(mod_memories.len());
        for mem in &mod_memories {
            match mem.def() {
                MemoryDef::Local(_spec) => {
                    memories.push(MemInst::new(mem.limits().clone()));
                }
                MemoryDef::Import { module: mod_name, name, .. } => {
                    let import = imports.iter().find(|i| {
                        i.module == *mod_name && i.name == *name
                    });
                    match import {
                        Some(Import { value: ImportValue::Memory(initial_pages, max_pages), .. }) => {
                            let declared_min = mem.limits().min();
                            let declared_max = mem.limits().max();
                            // Import compatibility: actual >= declared min
                            if *initial_pages < declared_min {
                                return Err(WasmError::unlinkable(format!(
                                    "incompatible import type: {}.{}",
                                    mod_name, name
                                )));
                            }
                            // If declared max exists, actual max must exist and be <= declared max
                            if let Some(d_max) = declared_max {
                                match max_pages {
                                    Some(a_max) if *a_max <= d_max => {}
                                    _ => {
                                        return Err(WasmError::unlinkable(format!(
                                            "incompatible import type: {}.{}",
                                            mod_name, name
                                        )));
                                    }
                                }
                            }
                            let import_limits = Limits::new(*initial_pages, *max_pages)?;
                            memories.push(MemInst::new(import_limits));
                        }
                        _ => {
                            return Err(WasmError::unlinkable(format!(
                                "missing memory import: {}.{}",
                                mod_name, name
                            )));
                        }
                    }
                }
            }
        }

        // --- Build global instances (uninitialized, will init below) ---
        let mut globals: Vec<GlobalInst> = Vec::with_capacity(mod_globals.len());
        for global in &mod_globals {
            match global.def() {
                GlobalDef::Local(_spec) => {
                    globals.push(GlobalInst::new(
                        Value::I32(0),
                        global.mutable(),
                        global.value_type(),
                    ));
                }
                GlobalDef::Import { module: mod_name, name, value_type, mutable } => {
                    let import = imports.iter().find(|i| {
                        i.module == *mod_name && i.name == *name
                    });
                    match import {
                        Some(Import { value: ImportValue::Global(val, imp_mutable), .. }) => {
                            // Check type compatibility
                            let val_type = val.value_type();
                            if val_type != *value_type {
                                return Err(WasmError::unlinkable(format!(
                                    "incompatible import type: {}.{}",
                                    mod_name, name
                                )));
                            }
                            if *imp_mutable != *mutable {
                                return Err(WasmError::unlinkable(format!(
                                    "incompatible import type: {}.{}",
                                    mod_name, name
                                )));
                            }
                            globals.push(GlobalInst::new(*val, *mutable, *value_type));
                        }
                        _ => {
                            return Err(WasmError::unlinkable(format!(
                                "missing global import: {}.{}",
                                mod_name, name
                            )));
                        }
                    }
                }
            }
        }

        // --- Build element instances ---
        let elements: Vec<ElementInst> = mod_elements
            .iter()
            .map(|e| {
                let vt = e.value_type();
                ElementInst::new(Vec::new(), vt)
            })
            .collect();

        // --- Build data instances ---
        let data: Vec<DataInst> = mod_data
            .iter()
            .map(|d| DataInst::new(d.get_init().to_vec()))
            .collect();

        // --- Create ModuleInst and Store ---
        let mut module_inst = ModuleInst::new("main".to_string(), types);
        module_inst.functions = functions;
        module_inst.tables = tables;
        module_inst.memories = memories;
        module_inst.globals = globals;
        module_inst.elements = elements;
        module_inst.data = data;
        let mut store = Store::new(module_inst);

        // --- Initialize globals (const expr evaluation) ---
        for (i, global) in mod_globals.iter().enumerate() {
            if let GlobalDef::Local(spec) = global.def() {
                let value = eval_const_expr(spec.init_expr(), store.module())?;
                store.global_mut(i).value = value;
            }
        }

        // --- Initialize element segments ---
        for (i, element) in mod_elements.iter().enumerate() {
            match element {
                Element::Active { table_index, offset_expr, init } => {
                    if *table_index >= store.module().tables.len() {
                        return Err(WasmError::unlinkable(
                            "unknown table".to_string(),
                        ));
                    }
                    let offset = eval_offset(offset_expr, store.module())?;
                    let refs = materialize_element_init(init, store.module())?;

                    let table = store.table_mut(*table_index);
                    if offset + refs.len() > table.elements.len() {
                        return Err(WasmError::unlinkable(
                            "out of bounds table access".to_string(),
                        ));
                    }
                    table.elements[offset..offset + refs.len()].copy_from_slice(&refs);
                    store.module_mut().elements[i].drop_segment();
                }
                Element::Passive { init } => {
                    let refs = materialize_element_init(init, store.module())?;
                    store.module_mut().elements[i] = ElementInst::new(refs, element.value_type());
                }
                Element::Declarative { .. } => {
                    store.module_mut().elements[i].drop_segment();
                }
            }
        }

        // --- Initialize data segments ---
        for (_i, data_seg) in mod_data.iter().enumerate() {
            match data_seg {
                Data::Active { memory_index, offset_expr, init } => {
                    let offset = eval_offset(offset_expr, store.module())?;
                    let mem = store.memory_mut(*memory_index);
                    if offset + init.len() > mem.data.len() {
                        return Err(WasmError::unlinkable(
                            "out of bounds memory access".to_string(),
                        ));
                    }
                    mem.data[offset..offset + init.len()].copy_from_slice(init);
                }
                Data::Passive { .. } => {
                    // Already materialized above
                }
            }
        }

        // --- Precompile handler-based fast IR when the active runtime family needs it ---
        if !matches!(crate::vm::backend::backend_mode(), crate::vm::backend::BackendMode::Native) {
            precompile::precompile_module_two_pass(&store)?;
        }

        // --- Run start function ---
        if let Some(start_idx) = start_func_index {
            let func_ptr = &store.module().functions[start_idx] as *const FunctionInst;
            let func_ref = unsafe { &*func_ptr };
            runtime::eval(func_ref, &mut store, &[])?;
        }

        Ok(Instance { store, exports })
    }

    /// Invoke an exported function by name.
    pub fn invoke(&mut self, name: &str, args: &[Value]) -> Result<Vec<Value>, WasmError> {
        let (_, kind, idx) = self
            .exports
            .iter()
            .find(|(n, k, _)| matches!(k, ExportKind::Func) && n == name)
            .ok_or_else(|| {
                WasmError::invalid(format!("exported function not found: {}", name))
            })?;
        let idx = *idx;

        let func_ptr = &self.store.module().functions[idx] as *const FunctionInst;
        let func_ref = unsafe { &*func_ptr };

        let result_stack = runtime::eval(func_ref, &mut self.store, args)?;
        let ft = func_ref.func_type();
        let result_types = ft.results();

        let mut results = Vec::with_capacity(result_types.len());
        for (i, ty) in result_types.iter().enumerate() {
            let raw = result_stack.peek_at_index(i);
            results.push(Value::from_raw(raw, *ty));
        }
        Ok(results)
    }

    /// Get the store (for advanced use).
    pub fn store(&self) -> &Store {
        &self.store
    }

    /// Get the store mutably (for advanced use).
    pub fn store_mut(&mut self) -> &mut Store {
        &mut self.store
    }

    /// Get an exported global's value by name.
    pub fn get_global(&self, name: &str) -> Option<Value> {
        self.exports
            .iter()
            .find(|(n, k, _)| matches!(k, ExportKind::Global) && n == name)
            .map(|(_, _, idx)| self.store.global(*idx).value)
    }

    /// Set an exported mutable global's value by name.
    pub fn set_global(&mut self, name: &str, value: Value) -> Result<(), WasmError> {
        let idx = self
            .exports
            .iter()
            .find(|(n, k, _)| matches!(k, ExportKind::Global) && n == name)
            .map(|(_, _, idx)| *idx)
            .ok_or_else(|| WasmError::invalid(format!("global not found: {}", name)))?;
        let global = self.store.global_mut(idx);
        if !global.mutable {
            return Err(WasmError::invalid("cannot set immutable global".into()));
        }
        global.value = value;
        Ok(())
    }

    /// Get the exported memory data (memory index 0) for reading/writing.
    pub fn memory(&self) -> Option<&[u8]> {
        if self.store.module().memories.is_empty() {
            None
        } else {
            Some(&self.store.memory(0).data)
        }
    }

    /// Get mutable access to the exported memory data.
    pub fn memory_mut(&mut self) -> Option<&mut Vec<u8>> {
        if self.store.module().memories.is_empty() {
            None
        } else {
            Some(&mut self.store.memory_mut(0).data)
        }
    }

    /// Get the current size (in pages) of an exported memory.
    pub fn memory_pages(&self, name: &str) -> Option<usize> {
        for (n, kind, idx) in &self.exports {
            if n == name && matches!(kind, ExportKind::Memory) {
                return Some(self.store.memory(*idx).current_pages());
            }
        }
        None
    }

    /// Get the current size (number of entries) of an exported table.
    pub fn table_size(&self, name: &str) -> Option<usize> {
        for (n, kind, idx) in &self.exports {
            if n == name && matches!(kind, ExportKind::Table) {
                return Some(self.store.table(*idx).size());
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn eval_offset(expr: &crate::module::entities::ConstExpr, module: &ModuleInst) -> Result<usize, WasmError> {
    let value = eval_const_expr(expr, module)?;
    match value {
        Value::I32(v) => {
            if v < 0 {
                Err(WasmError::unlinkable("offset is negative".into()))
            } else {
                Ok(v as usize)
            }
        }
        Value::I64(v) => {
            if v < 0 {
                Err(WasmError::unlinkable("offset is negative".into()))
            } else {
                Ok(v as usize)
            }
        }
        _ => Err(WasmError::invalid("offset must be i32 or i64".into())),
    }
}

fn materialize_element_init(
    init: &ElementInit,
    module: &ModuleInst,
) -> Result<Vec<RefHandle>, WasmError> {
    match init {
        ElementInit::FunctionIndexes(indices) => {
            indices
                .iter()
                .map(|&idx| {
                    if idx < module.functions.len() {
                        Ok(RefHandle::new(idx))
                    } else {
                        Err(WasmError::invalid("element function index out of range".into()))
                    }
                })
                .collect()
        }
        ElementInit::InitExprs { exprs, .. } => {
            exprs
                .iter()
                .map(|expr| {
                    let value = eval_const_expr(expr, module)?;
                    match value {
                        Value::Ref(handle, _) => Ok(handle),
                        _ => Err(WasmError::invalid("element init must be a reference".into())),
                    }
                })
                .collect()
        }
    }
}
