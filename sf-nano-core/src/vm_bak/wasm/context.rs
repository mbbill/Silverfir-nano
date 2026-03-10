//! Compile context: immutable function metadata and type resolution.

use alloc::rc::Rc;
use crate::{
    module::type_defs::FunctionType,
    op_decoder::{BlockType, Immediate},
    vm::{
        entities::{FunctionInst, ModuleInst},
        store::Store,
    },
};

/// Immutable context for compilation.
/// Provides type resolution without mutation.
pub struct CompileContext<'a> {
    types: Option<&'a [Rc<FunctionType>]>,
    store: &'a Store,
    module: &'a ModuleInst,
    results_count: usize,
}

impl<'a> CompileContext<'a> {
    pub fn new(
        types: Option<&'a [Rc<FunctionType>]>,
        store: &'a Store,
        module: &'a ModuleInst,
        results_count: usize,
    ) -> Self {
        Self { types, store, module, results_count }
    }

    /// Resolve block type to (param_count, result_count).
    pub fn resolve_block_type(&self, bt: &BlockType) -> (usize, usize) {
        match bt {
            BlockType::Empty => (0, 0),
            BlockType::ValueType(_) => (0, 1),
            BlockType::TypeIndex(idx) => self.resolve_type_index(*idx as usize),
        }
    }

    /// Resolve block type from immediate.
    pub fn resolve_block_type_from_imm(&self, imm: &Immediate) -> (usize, usize) {
        match imm {
            Immediate::Block(bt) => self.resolve_block_type(bt),
            _ => (0, 0),
        }
    }

    /// Resolve type index to (param_count, result_count).
    pub fn resolve_type_index(&self, type_idx: usize) -> (usize, usize) {
        if let Some(types) = self.types {
            if let Some(func_type) = types.get(type_idx) {
                return (func_type.params().len(), func_type.results().len());
            }
        }
        (0, 0)
    }

    /// Resolve function index to (param_count, result_count).
    pub fn resolve_func_type(&self, func_idx: usize) -> (usize, usize) {
        let func_inst = self.store.function(func_idx);
        let ft = func_inst.func_type();
        (ft.params().len(), ft.results().len())
    }

    /// Function results count (for return).
    pub fn results_count(&self) -> usize {
        self.results_count
    }

    /// Check if function at given index is an internal (non-external) function.
    /// Returns true if the function is defined in the module, false if it's imported/external.
    pub fn is_func_internal(&self, func_idx: usize) -> bool {
        !self.store.function(func_idx).is_external()
    }

    /// Get a function instance by index in the current module.
    #[inline]
    pub fn get_func_inst(&self, func_idx: usize) -> Option<&FunctionInst> {
        Some(self.store.function(func_idx))
    }
}
