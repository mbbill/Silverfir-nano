//! WebAssembly 2.0 Module Builder (no_std)
//!
//! Provides a builder pattern for constructing Module instances.
//! Used during module instantiation to incrementally set up module data.

use crate::collections;

use alloc::rc::Rc;
use alloc::string::{String, ToString};

use crate::error::WasmError;
use crate::module::entities::{Data, Element, Function, FunctionType, Global, Memory, Table};
use crate::module::type_context::TypeContext;
use crate::module::Module;

pub struct ModuleBuilder {
    name: String,
    binary_version: u32,
    types: collections::Vec<Rc<FunctionType>>,
    functions: collections::Vec<Function>,
    memories: collections::Vec<Memory>,
    tables: collections::Vec<Table>,
    globals: collections::Vec<Global>,
    elements: collections::Vec<Element>,
    data: collections::Vec<Data>,
    start_func_index: Option<usize>,
    data_count: Option<usize>,
    export_names: collections::Vec<String>,
}

impl ModuleBuilder {
    pub fn new() -> Self {
        ModuleBuilder {
            name: String::new(),
            binary_version: 0,
            types: collections::Vec::new(),
            functions: collections::Vec::new(),
            memories: collections::Vec::new(),
            tables: collections::Vec::new(),
            globals: collections::Vec::new(),
            elements: collections::Vec::new(),
            data: collections::Vec::new(),
            start_func_index: None,
            data_count: None,
            export_names: collections::Vec::new(),
        }
    }

    /// Registers an export name, ensuring uniqueness across all entity types.
    pub fn register_export_name(&mut self, name: &str) -> Result<(), WasmError> {
        if self.export_names.iter().any(|n| n == name) {
            return Err(WasmError::invalid("duplicate export name".into()));
        }
        self.export_names.push(name.to_string());
        Ok(())
    }

    pub fn with_name(&mut self, name: &str) {
        self.name = name.to_string();
    }

    pub fn with_binary_version(&mut self, version: u32) {
        self.binary_version = version;
    }

    pub fn with_types(&mut self, types: collections::Vec<Rc<FunctionType>>) {
        self.types = types;
    }

    /// Get a function type by index.
    pub fn get_function_type(&self, index: usize) -> Option<Rc<FunctionType>> {
        self.types.get(index).cloned()
    }

    pub fn append_function(&mut self, func: Function) {
        self.functions.push(func);
    }

    pub fn get_function_mut(&mut self, index: usize) -> Result<&mut Function, WasmError> {
        self.functions
            .get_mut(index)
            .ok_or_else(|| WasmError::invalid("index out of range".into()))
    }

    pub fn get_imported_function_count(&self) -> usize {
        self.functions.iter().filter(|f| f.is_import()).count()
    }

    pub fn append_memory(&mut self, memory: Memory) {
        self.memories.push(memory);
    }

    pub fn get_memory_mut(&mut self, index: usize) -> Result<&mut Memory, WasmError> {
        self.memories
            .get_mut(index)
            .ok_or_else(|| WasmError::invalid("index out of range".into()))
    }

    pub fn append_table(&mut self, table: Table) {
        self.tables.push(table);
    }

    pub fn get_table_mut(&mut self, index: usize) -> Result<&mut Table, WasmError> {
        self.tables
            .get_mut(index)
            .ok_or_else(|| WasmError::invalid("index out of range".into()))
    }

    pub fn append_global(&mut self, global: Global) {
        self.globals.push(global);
    }

    pub fn get_global_mut(&mut self, index: usize) -> Result<&mut Global, WasmError> {
        self.globals
            .get_mut(index)
            .ok_or_else(|| WasmError::invalid("index out of range".into()))
    }

    pub fn set_start_function(&mut self, index: usize) -> Result<(), WasmError> {
        let func_type = self.get_function_mut(index)?.func_type().clone();
        if !func_type.params().is_empty() || !func_type.results().is_empty() {
            return Err(WasmError::invalid(
                "Start function must not have params or results".into(),
            ));
        }
        self.start_func_index = Some(index);
        Ok(())
    }

    pub fn with_elements(&mut self, elements: collections::Vec<Element>) {
        self.elements = elements;
    }

    pub fn with_data_count(&mut self, count: usize) {
        self.data_count = Some(count);
    }

    pub fn data_count(&self) -> Option<usize> {
        self.data_count
    }

    pub fn data_is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn with_data(&mut self, data: collections::Vec<Data>) {
        self.data = data;
    }

    pub fn build(mut self) -> Module {
        self.functions.shrink_to_fit();
        self.memories.shrink_to_fit();
        self.tables.shrink_to_fit();
        self.globals.shrink_to_fit();
        self.elements.shrink_to_fit();

        Module {
            name: self.name,
            binary_version: self.binary_version,
            types: TypeContext::new(self.types),
            functions: self.functions,
            tables: self.tables,
            memories: self.memories,
            globals: self.globals,
            elements: self.elements,
            data: self.data,
            start_func_index: self.start_func_index,
            data_count: self.data_count,
        }
    }

    /// Helper for tests: set function types directly.
    #[cfg(test)]
    pub fn with_function_types(&mut self, function_types: collections::Vec<Rc<FunctionType>>) {
        self.types = function_types;
    }
}
