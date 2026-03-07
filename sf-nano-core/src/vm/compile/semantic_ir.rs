//! Semantic Wasm decode IR before backend-local placement.
//!
//! This layer is intentionally incremental. It keeps control-flow metadata,
//! variants, and abstract TOS-cache management markers from the current
//! lowering pipeline. Backend-local placement, `InitLocals`, and concrete
//! lowered spill/fill opcodes are applied by `backend_lower`.

use super::common::{BrTableEntry, OpIndex, SlotRef};
use super::core_op::{self, CoreOpKind};

#[derive(Debug, Clone)]
pub struct SemanticOp {
    pub kind: SemanticOpKind,
    pub variant: u8,
    pub pre_height: u16,
    pub fallthrough: Option<OpIndex>,
    pub alt_target: Option<OpIndex>,
    pub has_target: bool,
}

#[derive(Debug, Clone)]
pub enum SemanticOpKind {
    Core(CoreOpKind),
    LocalGet { idx: u16 },
    LocalSet { idx: u16 },
    LocalTee { idx: u16 },
    Block { params: u16, results: u16 },
    Loop { params: u16, results: u16 },
    If { params: u16, results: u16 },
    Else,
    End,
    CacheSpill { slot: u16, count: u8 },
    CacheFill { slot: u16, count: u8 },
    Br { stack_drop: u32, arity: u16 },
    BrIf { stack_drop: u32, arity: u16 },
    BrTable { entries: alloc::vec::Vec<BrTableEntry> },
    CallExternal { func_idx: u32, delta: SlotRef, params: u16, results: u16 },
    CallInternal { callee: u64, delta: SlotRef, params: u16, results: u16 },
    CallIndirect { type_idx: u32, table_idx: u32, delta: SlotRef, params: u16, results: u16 },
    ReturnVoid,
    ReturnOne,
    Return { arity: u16 },
}

impl From<CoreOpKind> for SemanticOpKind {
    #[inline]
    fn from(kind: CoreOpKind) -> Self {
        Self::Core(kind)
    }
}

pub fn stack_effect(kind: &SemanticOpKind) -> (u8, u8) {
    match kind {
        SemanticOpKind::Core(kind) => core_op::stack_effect(kind),
        SemanticOpKind::LocalGet { .. } => (0, 1),
        SemanticOpKind::LocalSet { .. } => (1, 0),
        SemanticOpKind::LocalTee { .. } => (0, 0),
        SemanticOpKind::Block { .. } | SemanticOpKind::Loop { .. } |
        SemanticOpKind::Else | SemanticOpKind::End => (0, 0),
        SemanticOpKind::If { .. } => (1, 0),
        SemanticOpKind::CacheSpill { .. } | SemanticOpKind::CacheFill { .. } => (0, 0),
        SemanticOpKind::Br { .. } => (0, 0),
        SemanticOpKind::BrIf { .. } | SemanticOpKind::BrTable { .. } => (1, 0),
        SemanticOpKind::CallExternal { .. } | SemanticOpKind::CallInternal { .. } => (0, 0),
        // Decode pops the callee-table index before emitting CallIndirect, so
        // pre_height reconstruction must add that implicit pop back.
        SemanticOpKind::CallIndirect { .. } => (1, 0),
        SemanticOpKind::ReturnVoid | SemanticOpKind::ReturnOne | SemanticOpKind::Return { .. } => (0, 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::compile::common::SlotRef;

    #[test]
    fn test_call_indirect_stack_effect_accounts_for_popped_table_index() {
        let kind = SemanticOpKind::CallIndirect {
            type_idx: 0,
            table_idx: 0,
            delta: SlotRef::operand_relative(0),
            params: 2,
            results: 1,
        };

        assert_eq!(stack_effect(&kind), (1, 0));
    }
}
