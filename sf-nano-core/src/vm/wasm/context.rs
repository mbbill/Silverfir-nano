//! Immutable semantic decode context.
//!
//! This is semantic/frontend-only metadata used while decoding a function body.

use crate::{
    module::type_context::TypeContext,
    op_decoder::{BlockType, Immediate},
    vm::{
        entities::{FunctionInst, ModuleInst},
        store::Store,
    },
};

/// Immutable decode context for one function body.
#[derive(Clone, Copy)]
pub struct CompileContext<'a> {
    pub types: &'a TypeContext,
    pub store: &'a Store,
    pub module: &'a ModuleInst,
    pub params: u16,
    pub local_count: u16,
    pub results: u16,
}

impl<'a> CompileContext<'a> {
    #[inline]
    pub const fn new(
        types: &'a TypeContext,
        store: &'a Store,
        module: &'a ModuleInst,
        params: u16,
        local_count: u16,
        results: u16,
    ) -> Self {
        Self {
            types,
            store,
            module,
            params,
            local_count,
            results,
        }
    }

    #[inline]
    pub fn resolve_block_type(&self, block_type: &BlockType) -> (u16, u16) {
        match block_type {
            BlockType::Empty => (0, 0),
            BlockType::ValueType(_) => (0, 1),
            BlockType::TypeIndex(idx) => self
                .types
                .get(*idx as u32)
                .map(|ty| (ty.params().len() as u16, ty.results().len() as u16))
                .unwrap_or((0, 0)),
        }
    }

    #[inline]
    pub fn resolve_block_type_from_imm(&self, imm: &Immediate) -> (u16, u16) {
        match imm {
            Immediate::Block(block_type) => self.resolve_block_type(block_type),
            _ => (0, 0),
        }
    }

    #[inline]
    pub fn resolve_type_index(&self, type_idx: u32) -> (u16, u16) {
        self.types
            .get(type_idx)
            .map(|ty| (ty.params().len() as u16, ty.results().len() as u16))
            .unwrap_or((0, 0))
    }

    #[inline]
    pub fn resolve_func_type(&self, func_idx: u32) -> (u16, u16) {
        let func = self.store.function(func_idx as usize);
        let ty = func.func_type();
        (ty.params().len() as u16, ty.results().len() as u16)
    }

    #[inline]
    pub fn is_func_internal(&self, func_idx: u32) -> bool {
        !self.store.function(func_idx as usize).is_external()
    }

    #[inline]
    pub fn get_func_inst(&self, func_idx: u32) -> Option<&FunctionInst> {
        Some(self.store.function(func_idx as usize))
    }
}
