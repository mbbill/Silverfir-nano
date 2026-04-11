//! Immutable semantic decode context.
//!
//! This is semantic/frontend-only metadata used while decoding a function body.

use crate::collections;

use crate::{
    module::type_context::TypeContext,
    op_decoder::{BlockType, Immediate},
    value_type::ValueType,
    vm::store::Store,
};

/// Immutable decode context for one function body.
#[derive(Clone, Copy)]
pub(crate) struct CompileContext<'a> {
    pub types: &'a TypeContext,
    pub store: &'a Store,
    pub params: u16,
    pub local_count: u16,
    pub results: u16,
    /// Per-local value types (params ++ non-param locals).
    /// When empty, the decode stage will not propagate type info.
    pub local_types: &'a [ValueType],
    /// Function result types in signature order.
    pub result_types: &'a [ValueType],
}

impl<'a> CompileContext<'a> {
    #[inline]
    pub(crate) const fn with_value_types(
        types: &'a TypeContext,
        store: &'a Store,
        params: u16,
        local_count: u16,
        results: u16,
        local_types: &'a [ValueType],
        result_types: &'a [ValueType],
    ) -> Self {
        Self {
            types,
            store,
            params,
            local_count,
            results,
            local_types,
            result_types,
        }
    }

    #[inline]
    pub(crate) fn resolve_block_type(&self, block_type: &BlockType) -> (u16, u16) {
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
    pub(crate) fn resolve_block_type_from_imm(&self, imm: &Immediate) -> (u16, u16) {
        match imm {
            Immediate::Block(block_type) => self.resolve_block_type(block_type),
            _ => (0, 0),
        }
    }

    #[inline]
    pub(crate) fn resolve_block_result_types(
        &self,
        block_type: &BlockType,
    ) -> collections::Vec<ValueType> {
        match block_type {
            BlockType::Empty => collections::Vec::new(),
            BlockType::ValueType(value_type) => collections::vec![*value_type],
            BlockType::TypeIndex(idx) => self
                .types
                .get(*idx as u32)
                .map(|ty| ty.results().to_vec().into())
                .unwrap_or_else(collections::Vec::new),
        }
    }

    #[inline]
    pub(crate) fn resolve_block_result_types_from_imm(
        &self,
        imm: &Immediate,
    ) -> collections::Vec<ValueType> {
        match imm {
            Immediate::Block(block_type) => self.resolve_block_result_types(block_type),
            _ => collections::Vec::new(),
        }
    }

    #[inline]
    pub(crate) fn resolve_type_index(&self, type_idx: u32) -> (u16, u16) {
        self.types
            .get(type_idx)
            .map(|ty| (ty.params().len() as u16, ty.results().len() as u16))
            .unwrap_or((0, 0))
    }

    #[inline]
    pub(crate) fn resolve_func_type(&self, func_idx: u32) -> (u16, u16) {
        let func = self.store.function(func_idx as usize);
        let ty = func.func_type();
        (ty.params().len() as u16, ty.results().len() as u16)
    }
}
