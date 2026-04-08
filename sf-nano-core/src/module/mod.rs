use alloc::string::String;
use alloc::vec::Vec;

use entities::{Data, Element, Function, Global, Memory, Table};

pub mod builder;
pub mod entities;
pub(crate) mod parser;
pub mod type_context;
pub mod type_defs;
#[cfg(sf_module_validator)]
pub mod validator;

use crate::error::WasmError;

#[derive(Debug)]
pub struct Module {
    name: String,
    binary_version: u32,
    types: type_context::TypeContext,
    functions: Vec<Function>,
    tables: Vec<Table>,
    memories: Vec<Memory>,
    globals: Vec<Global>,
    elements: Vec<Element>,
    data: Vec<Data>,
    start_func_index: Option<usize>,
    data_count: Option<usize>,
}

impl Module {
    pub fn new(name: &str, bin: &[u8]) -> Result<Self, WasmError> {
        parser::parse_module(name, bin)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn version(&self) -> u32 {
        self.binary_version
    }

    pub fn types(&self) -> &type_context::TypeContext {
        &self.types
    }

    pub fn functions(&self) -> &[Function] {
        &self.functions
    }

    pub fn memories(&self) -> &[Memory] {
        &self.memories
    }

    pub fn tables(&self) -> &[Table] {
        &self.tables
    }

    pub fn globals(&self) -> &[Global] {
        &self.globals
    }

    pub fn elements(&self) -> &[Element] {
        &self.elements
    }

    pub fn data(&self) -> &[Data] {
        &self.data
    }

    pub fn start_function_index(&self) -> Option<usize> {
        self.start_func_index
    }

    pub fn data_count(&self) -> Option<usize> {
        self.data_count
    }

    /// Consume the module, returning all internal fields.
    pub fn into_parts(
        self,
    ) -> (
        type_context::TypeContext,
        Vec<Function>,
        Vec<Table>,
        Vec<Memory>,
        Vec<Global>,
        Vec<Element>,
        Vec<Data>,
        Option<usize>,
    ) {
        (
            self.types,
            self.functions,
            self.tables,
            self.memories,
            self.globals,
            self.elements,
            self.data,
            self.start_func_index,
        )
    }
}
