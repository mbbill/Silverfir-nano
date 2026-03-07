//! Semantic Wasm decode IR before backend-local placement.
//!
//! This layer is intentionally incremental. It keeps control-flow metadata,
//! variants, and abstract TOS-cache management markers from the current
//! lowering pipeline. Backend-local placement, `InitLocals`, and concrete
//! lowered spill/fill opcodes are applied by `backend_lower`.

use super::common::{BrTableEntry, OpIndex};
use super::lowered_ir::{self, IrOpKind as LoweredIrOpKind};

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
    Core(LoweredIrOpKind),
    LocalGet { idx: u16 },
    LocalSet { idx: u16 },
    LocalTee { idx: u16 },
    CacheSpill { slot: u16, count: u8 },
    CacheFill { slot: u16, count: u8 },
    Br { stack_drop: u32, arity: u16 },
    BrIf { stack_drop: u32, arity: u16 },
    BrTable { entries: alloc::vec::Vec<BrTableEntry> },
    CallExternal { func_idx: u32, delta: lowered_ir::SlotRef },
    CallInternal { callee: u64, delta: lowered_ir::SlotRef },
    CallIndirect { type_idx: u32, table_idx: u32, delta: lowered_ir::SlotRef },
    ReturnVoid,
    ReturnOne,
    Return { arity: u16 },
}

impl From<LoweredIrOpKind> for SemanticOpKind {
    #[inline]
    fn from(kind: LoweredIrOpKind) -> Self {
        Self::Core(kind)
    }
}

pub fn stack_effect(kind: &SemanticOpKind) -> (u8, u8) {
    match kind {
        SemanticOpKind::Core(kind) => lowered_ir::stack_effect(kind),
        SemanticOpKind::LocalGet { .. } => (0, 1),
        SemanticOpKind::LocalSet { .. } => (1, 0),
        SemanticOpKind::LocalTee { .. } => (0, 0),
        SemanticOpKind::CacheSpill { .. } | SemanticOpKind::CacheFill { .. } => (0, 0),
        SemanticOpKind::Br { .. } => (0, 0),
        SemanticOpKind::BrIf { .. } | SemanticOpKind::BrTable { .. } => (1, 0),
        SemanticOpKind::CallExternal { .. } | SemanticOpKind::CallInternal { .. } => (0, 0),
        SemanticOpKind::CallIndirect { .. } => (1, 0),
        SemanticOpKind::ReturnVoid | SemanticOpKind::ReturnOne | SemanticOpKind::Return { .. } => (0, 0),
    }
}
