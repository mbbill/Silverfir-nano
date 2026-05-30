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
    pub(in crate::vm::wasm) types: &'a TypeContext,
    pub(in crate::vm::wasm) store: &'a Store,
    pub(in crate::vm::wasm) params: u16,
    pub(in crate::vm::wasm) local_count: u16,
    pub(in crate::vm::wasm) results: u16,
    /// Per-local value types (params ++ non-param locals).
    /// When empty, the decode stage will not propagate type info.
    pub(in crate::vm::wasm) local_types: &'a [ValueType],
    /// Function result types in signature order.
    pub(in crate::vm::wasm) result_types: &'a [ValueType],
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
    fn resolve_block_type(&self, block_type: &BlockType) -> (u16, u16) {
        match block_type {
            BlockType::Empty => (0, 0),
            BlockType::ValueType(_) => (0, 1),
            BlockType::TypeIndex(idx) => self
                .types
                .get_function_type(*idx as u32)
                .map(|ty| (ty.params().len() as u16, ty.results().len() as u16))
                .unwrap_or((0, 0)),
        }
    }

    #[inline]
    pub(in crate::vm::wasm) fn resolve_block_type_from_imm(&self, imm: &Immediate) -> (u16, u16) {
        match imm {
            Immediate::Block(block_type) => self.resolve_block_type(block_type),
            _ => (0, 0),
        }
    }

    #[inline]
    fn resolve_block_result_types(&self, block_type: &BlockType) -> collections::Vec<ValueType> {
        match block_type {
            BlockType::Empty => collections::Vec::new(),
            BlockType::ValueType(value_type) => collections::vec![*value_type],
            BlockType::TypeIndex(idx) => self
                .types
                .get_function_type(*idx as u32)
                .map(|ty| ty.results().to_vec().into())
                .unwrap_or_else(collections::Vec::new),
        }
    }

    #[inline]
    pub(in crate::vm::wasm) fn resolve_block_result_types_from_imm(
        &self,
        imm: &Immediate,
    ) -> collections::Vec<ValueType> {
        match imm {
            Immediate::Block(block_type) => self.resolve_block_result_types(block_type),
            _ => collections::Vec::new(),
        }
    }

    #[inline]
    pub(in crate::vm::wasm) fn resolve_type_index(&self, type_idx: u32) -> (u16, u16) {
        self.types
            .get_function_type(type_idx)
            .map(|ty| (ty.params().len() as u16, ty.results().len() as u16))
            .unwrap_or((0, 0))
    }

    #[inline]
    pub(in crate::vm::wasm) fn resolve_func_type(&self, func_idx: u32) -> (u16, u16) {
        let func = self.store.function(func_idx as usize);
        let ty = func.func_type();
        (ty.params().len() as u16, ty.results().len() as u16)
    }

    /// Resolve an EH tag index to `(params, results)`. Results are always 0
    /// for valid wasm tags (validator rejects non-zero), but we read it out so
    /// the SIR layer stays honest.
    #[inline]
    pub(in crate::vm::wasm) fn resolve_tag_type(&self, tag_idx: u32) -> (u16, u16) {
        let Some(tag_inst) = self.store.module().tags.get(tag_idx as usize) else {
            return (0, 0);
        };
        self.types
            .get_function_type(tag_inst.type_index)
            .map(|ty| (ty.params().len() as u16, ty.results().len() as u16))
            .unwrap_or((0, 0))
    }

    /// Resolve the block type embedded inside a `TryTable` immediate.
    #[inline]
    pub(in crate::vm::wasm) fn resolve_try_table_block_type(&self, imm: &Immediate) -> (u16, u16) {
        match imm {
            Immediate::TryTable { block_type, .. } => self.resolve_block_type(block_type),
            _ => (0, 0),
        }
    }

    #[inline]
    pub(in crate::vm::wasm) fn resolve_try_table_result_types(
        &self,
        imm: &Immediate,
    ) -> collections::Vec<ValueType> {
        match imm {
            Immediate::TryTable { block_type, .. } => self.resolve_block_result_types(block_type),
            _ => collections::Vec::new(),
        }
    }
}
