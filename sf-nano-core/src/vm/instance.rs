//! Public API for sf-nano: parse, instantiate, and invoke WebAssembly modules.

use crate::collections;

use tracked_alloc::string::{String, ToString};

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
use crate::vm::runtime;
use crate::vm::store::Store;
use crate::vm::value::{RefHandle, Value};

pub struct Import {
    pub module: String,
    pub name: String,
    pub value: ImportValue,
}

pub enum ImportValue {
    Func(ExternalFn, Option<FunctionType>),
    Global(Value, bool),
    Memory(usize, Option<usize>),
    Table(usize, Option<usize>),
}

impl Import {
    pub fn func(module: &str, name: &str, f: ExternalFn) -> Self {
        Import {
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Func(f, None),
        }
    }

    pub fn func_typed(module: &str, name: &str, f: ExternalFn, func_type: FunctionType) -> Self {
        Import {
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Func(f, Some(func_type)),
        }
    }

    pub fn global(module: &str, name: &str, value: Value, mutable: bool) -> Self {
        Import {
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Global(value, mutable),
        }
    }

    pub fn memory(
        module: &str,
        name: &str,
        initial_pages: usize,
        max_pages: Option<usize>,
    ) -> Self {
        Import {
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Memory(initial_pages, max_pages),
        }
    }

    pub fn table(module: &str, name: &str, initial_size: usize, max_size: Option<usize>) -> Self {
        Import {
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Table(initial_size, max_size),
        }
    }
}

pub struct Instance {
    store: Store,
    exports: collections::Vec<(String, ExportKind, usize)>,
}

#[derive(Clone, Copy)]
enum ExportKind {
    Func,
    Table,
    Memory,
    Global,
}

impl Instance {
    pub fn new(wasm_bytes: &[u8], imports: &[Import]) -> Result<Self, WasmError> {
        let module = Module::new("main", wasm_bytes)?;
        Self::from_module(module, imports)
    }

    pub fn from_module(module: Module, imports: &[Import]) -> Result<Self, WasmError> {
        #[cfg(sf_module_validator)]
        {
            use crate::module::validator::Validator;
            let mut validator = Validator::new(&module);
            validator.validate()?;
        }

        let mut exports = collections::Vec::new();
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
        let (
            types,
            mod_functions,
            mod_tables,
            mod_memories,
            mod_globals,
            mod_elements,
            mod_data,
            _start,
        ) = module.into_parts();

        let mut functions: collections::Vec<FunctionInst> =
            collections::Vec::with_capacity(mod_functions.len());
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
                    let import = imports
                        .iter()
                        .find(|i| i.module == mod_name && i.name == name);
                    match import {
                        Some(Import {
                            value: ImportValue::Func(f, ref import_type),
                            ..
                        }) => {
                            if let Some(actual_type) = import_type {
                                if actual_type.params() != func_type.params()
                                    || actual_type.results() != func_type.results()
                                {
                                    return Err(WasmError::unlinkable(
                                        "incompatible import type: .",
                                    ));
                                }
                            }
                            functions.push(FunctionInst::External {
                                func_type,
                                callback: *f,
                            });
                        }
                        _ => {
                            return Err(WasmError::unlinkable("missing function import: ."));
                        }
                    }
                }
            }
        }

        let mut tables: collections::Vec<TableInst> =
            collections::Vec::with_capacity(mod_tables.len());
        for table in &mod_tables {
            match table.def() {
                TableDef::Local(_spec) => {
                    tables.push(TableInst::new(table.limits().clone(), table.value_type()));
                }
                TableDef::Import {
                    module: mod_name,
                    name,
                    ..
                } => {
                    let import = imports
                        .iter()
                        .find(|i| i.module == *mod_name && i.name == *name);
                    match import {
                        Some(Import {
                            value: ImportValue::Table(initial_size, max_size),
                            ..
                        }) => {
                            let declared_min = table.limits().min();
                            let declared_max = table.limits().max();
                            if *initial_size < declared_min {
                                return Err(WasmError::unlinkable("incompatible import type: ."));
                            }
                            if let Some(d_max) = declared_max {
                                match max_size {
                                    Some(a_max) if *a_max <= d_max => {}
                                    _ => {
                                        return Err(WasmError::unlinkable(
                                            "incompatible import type: .",
                                        ));
                                    }
                                }
                            }
                            let import_limits = Limits::new(*initial_size, *max_size)?;
                            tables.push(TableInst::new(import_limits, table.value_type()));
                        }
                        _ => {
                            return Err(WasmError::unlinkable("missing table import: ."));
                        }
                    }
                }
            }
        }

        let mut memories: collections::Vec<MemInst> =
            collections::Vec::with_capacity(mod_memories.len());
        for mem in &mod_memories {
            match mem.def() {
                MemoryDef::Local(_spec) => {
                    #[cfg(sf_has_guard_pages)]
                    {
                        memories.push(MemInst::new_guarded(mem.limits().clone())?);
                    }
                    #[cfg(not(sf_has_guard_pages))]
                    {
                        memories.push(MemInst::new(mem.limits().clone()));
                    }
                }
                MemoryDef::Import {
                    module: mod_name,
                    name,
                    ..
                } => {
                    let import = imports
                        .iter()
                        .find(|i| i.module == *mod_name && i.name == *name);
                    match import {
                        Some(Import {
                            value: ImportValue::Memory(initial_pages, max_pages),
                            ..
                        }) => {
                            let declared_min = mem.limits().min();
                            let declared_max = mem.limits().max();
                            if *initial_pages < declared_min {
                                return Err(WasmError::unlinkable("incompatible import type: ."));
                            }
                            if let Some(d_max) = declared_max {
                                match max_pages {
                                    Some(a_max) if *a_max <= d_max => {}
                                    _ => {
                                        return Err(WasmError::unlinkable(
                                            "incompatible import type: .",
                                        ));
                                    }
                                }
                            }
                            let import_limits = Limits::new(*initial_pages, *max_pages)?;
                            #[cfg(sf_has_guard_pages)]
                            {
                                memories.push(MemInst::new_guarded(import_limits)?);
                            }
                            #[cfg(not(sf_has_guard_pages))]
                            {
                                memories.push(MemInst::new(import_limits));
                            }
                        }
                        _ => {
                            return Err(WasmError::unlinkable("missing memory import: ."));
                        }
                    }
                }
            }
        }

        let mut globals: collections::Vec<GlobalInst> =
            collections::Vec::with_capacity(mod_globals.len());
        for global in &mod_globals {
            match global.def() {
                GlobalDef::Local(_spec) => {
                    globals.push(GlobalInst::new(
                        Value::I32(0),
                        global.mutable(),
                        global.value_type(),
                    ));
                }
                GlobalDef::Import {
                    module: mod_name,
                    name,
                    value_type,
                    mutable,
                } => {
                    let import = imports
                        .iter()
                        .find(|i| i.module == *mod_name && i.name == *name);
                    match import {
                        Some(Import {
                            value: ImportValue::Global(val, imp_mutable),
                            ..
                        }) => {
                            let val_type = val.value_type();
                            if val_type != *value_type {
                                return Err(WasmError::unlinkable("incompatible import type: ."));
                            }
                            if *imp_mutable != *mutable {
                                return Err(WasmError::unlinkable("incompatible import type: ."));
                            }
                            globals.push(GlobalInst::new(*val, *mutable, *value_type));
                        }
                        _ => {
                            return Err(WasmError::unlinkable("missing global import: ."));
                        }
                    }
                }
            }
        }

        let elements: collections::Vec<ElementInst> = mod_elements
            .iter()
            .map(|e| ElementInst::new(collections::Vec::new(), e.value_type()))
            .collect();

        let data: collections::Vec<DataInst> = mod_data
            .iter()
            .map(|d| DataInst::new(d.get_init().to_vec().into()))
            .collect();

        let mut module_inst = ModuleInst::new("main".to_string(), types);
        module_inst.functions = functions;
        module_inst.tables = tables;
        module_inst.memories = memories;
        module_inst.globals = globals;
        module_inst.elements = elements;
        module_inst.data = data;
        let mut store = Store::new(module_inst);

        for (i, global) in mod_globals.iter().enumerate() {
            if let GlobalDef::Local(spec) = global.def() {
                let value = eval_const_expr(spec.init_expr(), store.module())?;
                store.global_mut(i).set_value(value);
            }
        }

        for (i, element) in mod_elements.iter().enumerate() {
            match element {
                Element::Active {
                    table_index,
                    offset_expr,
                    init,
                } => {
                    if *table_index >= store.module().tables.len() {
                        return Err(WasmError::unlinkable("unknown table"));
                    }
                    let offset = eval_offset(offset_expr, store.module())?;
                    let refs = materialize_element_init(init, store.module())?;

                    let table = store.table_mut(*table_index);
                    if offset + refs.len() > table.elements.len() {
                        return Err(WasmError::unlinkable("out of bounds table access"));
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

        for data_seg in &mod_data {
            match data_seg {
                Data::Active {
                    memory_index,
                    offset_expr,
                    init,
                } => {
                    let offset = eval_offset(offset_expr, store.module())?;
                    let mem = store.memory_mut(*memory_index);
                    let mem_len = mem.memory_len();
                    if offset + init.len() > mem_len {
                        return Err(WasmError::unlinkable("out of bounds memory access"));
                    }
                    let dst = unsafe {
                        core::slice::from_raw_parts_mut(mem.memory_ptr().add(offset), init.len())
                    };
                    dst.copy_from_slice(init);
                }
                Data::Passive { .. } => {}
            }
        }

        if let Some(start_idx) = start_func_index {
            let func_ptr = &store.module().functions[start_idx] as *const FunctionInst;
            let func_ref = unsafe { &*func_ptr };
            runtime::eval(func_ref, &mut store, &[])?;
        }

        Ok(Instance { store, exports })
    }

    pub fn invoke(
        &mut self,
        name: &str,
        args: &[Value],
    ) -> Result<collections::Vec<Value>, WasmError> {
        let (_, _, idx) = self
            .exports
            .iter()
            .find(|(n, k, _)| matches!(k, ExportKind::Func) && n == name)
            .ok_or_else(|| WasmError::invalid("exported function not found"))?;
        let idx = *idx;

        let func_ptr = &self.store.module().functions[idx] as *const FunctionInst;
        let func_ref = unsafe { &*func_ptr };

        let result_stack = runtime::eval(func_ref, &mut self.store, args)?;
        let ft = func_ref.func_type();
        let result_types = ft.results();

        let mut results = collections::Vec::with_capacity(result_types.len());
        for (i, ty) in result_types.iter().enumerate() {
            let raw = result_stack.peek_at_index(i);
            results.push(Value::from_raw(raw, *ty));
        }
        Ok(results)
    }

    pub fn has_function_export(&self, name: &str) -> bool {
        self.exports
            .iter()
            .any(|(n, k, _)| matches!(k, ExportKind::Func) && n == name)
    }

    pub fn store(&self) -> &Store {
        &self.store
    }

    pub fn store_mut(&mut self) -> &mut Store {
        &mut self.store
    }

    pub fn get_global(&self, name: &str) -> Option<Value> {
        self.exports
            .iter()
            .find(|(n, k, _)| matches!(k, ExportKind::Global) && n == name)
            .map(|(_, _, idx)| self.store.global(*idx).value())
    }

    pub fn set_global(&mut self, name: &str, value: Value) -> Result<(), WasmError> {
        let idx = self
            .exports
            .iter()
            .find(|(n, k, _)| matches!(k, ExportKind::Global) && n == name)
            .map(|(_, _, idx)| *idx)
            .ok_or_else(|| WasmError::invalid("global not found"))?;
        let global = self.store.global_mut(idx);
        if !global.mutable {
            return Err(WasmError::invalid("cannot set immutable global"));
        }
        global.set_value(value);
        Ok(())
    }

    pub fn memory(&self) -> Option<&[u8]> {
        if self.store.module().memories.is_empty() {
            None
        } else {
            Some(&self.store.memory(0).data)
        }
    }

    pub fn memory_mut(&mut self) -> Option<&mut collections::Vec<u8>> {
        if self.store.module().memories.is_empty() {
            None
        } else {
            Some(&mut self.store.memory_mut(0).data)
        }
    }

    pub fn memory_pages(&self, name: &str) -> Option<usize> {
        for (n, kind, idx) in &self.exports {
            if n == name && matches!(kind, ExportKind::Memory) {
                return Some(self.store.memory(*idx).current_pages());
            }
        }
        None
    }

    pub fn table_size(&self, name: &str) -> Option<usize> {
        for (n, kind, idx) in &self.exports {
            if n == name && matches!(kind, ExportKind::Table) {
                return Some(self.store.table(*idx).size());
            }
        }
        None
    }
}

fn eval_offset(
    expr: &crate::module::entities::ConstExpr,
    module: &ModuleInst,
) -> Result<usize, WasmError> {
    let value = eval_const_expr(expr, module)?;
    match value {
        Value::I32(v) => {
            if v < 0 {
                Err(WasmError::unlinkable("offset is negative"))
            } else {
                Ok(v as usize)
            }
        }
        Value::I64(v) => {
            if v < 0 {
                Err(WasmError::unlinkable("offset is negative"))
            } else {
                Ok(v as usize)
            }
        }
        _ => Err(WasmError::invalid("offset must be i32 or i64")),
    }
}

fn materialize_element_init(
    init: &ElementInit,
    module: &ModuleInst,
) -> Result<collections::Vec<RefHandle>, WasmError> {
    match init {
        ElementInit::FunctionIndexes(indices) => indices
            .iter()
            .map(|&idx| {
                if idx < module.functions.len() {
                    Ok(RefHandle::new(idx))
                } else {
                    Err(WasmError::invalid(
                        "element function index out of range".into(),
                    ))
                }
            })
            .collect(),
        ElementInit::InitExprs { exprs, .. } => exprs
            .iter()
            .map(|expr| {
                let value = eval_const_expr(expr, module)?;
                match value {
                    Value::Ref(handle, _) => Ok(handle),
                    _ => Err(WasmError::invalid(
                        "element init must be a reference".into(),
                    )),
                }
            })
            .collect(),
    }
}

// NOTE: Old debug_coremark_direct_coverage test removed — it referenced the
// legacy native::ir types (NativeProgram, NativeInstKind, etc.) that no longer
// exist after the native/ → machine/ + arch/ + runtime/ restructuring.
