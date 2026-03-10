//! Simplified Store for single-module WebAssembly execution.
//!
//! In sf-core, the Store holds all instances across multiple modules.
//! In sf-nano, since we target single-module execution, the Store is
//! essentially a wrapper around a single `ModuleInst`.

use crate::vm::entities::{FunctionInst, GlobalInst, MemInst, ModuleInst, TableInst};

/// A simplified single-module store.
pub struct Store {
    module: ModuleInst,
}

impl Store {
    pub fn new(module: ModuleInst) -> Self {
        Store { module }
    }

    #[inline]
    pub fn module(&self) -> &ModuleInst {
        &self.module
    }

    #[inline]
    pub fn module_mut(&mut self) -> &mut ModuleInst {
        &mut self.module
    }

    // -- Convenience accessors ------------------------------------------------

    #[inline]
    pub fn function(&self, idx: usize) -> &FunctionInst {
        &self.module.functions[idx]
    }

    #[inline]
    pub fn table(&self, idx: usize) -> &TableInst {
        &self.module.tables[idx]
    }

    #[inline]
    pub fn table_mut(&mut self, idx: usize) -> &mut TableInst {
        &mut self.module.tables[idx]
    }

    #[inline]
    pub fn memory(&self, idx: usize) -> &MemInst {
        &self.module.memories[idx]
    }

    #[inline]
    pub fn memory_mut(&mut self, idx: usize) -> &mut MemInst {
        &mut self.module.memories[idx]
    }

    #[inline]
    pub fn global(&self, idx: usize) -> &GlobalInst {
        &self.module.globals[idx]
    }

    #[inline]
    pub fn global_mut(&mut self, idx: usize) -> &mut GlobalInst {
        &mut self.module.globals[idx]
    }
}
