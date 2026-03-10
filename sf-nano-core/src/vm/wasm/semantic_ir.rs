//! Semantic function-body IR before planning.
//!
//! `CoreOpKind` carries the shared leaf-op vocabulary. `SemanticOpKind` is the
//! larger per-function IR that embeds those leaf ops alongside locals, calls,
//! returns, structured control markers, and branch targets.
//!
//! Important:
//! - no backend-facing `variant`
//! - no `pre_height`
//! - no spill/fill planning artifacts
//! - no backend helper-entry specialization

use alloc::vec::Vec;

use super::common::{BrTableEntry, SemanticIndex, SemanticTarget};
use super::core_op::CoreOpKind;

/// One semantic Wasm operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticOp {
    pub kind: SemanticOpKind,
    pub next: Option<SemanticIndex>,
    pub alt: Option<SemanticTarget>,
}

/// Semantic function-body op kind.
///
/// This owns the parts of Wasm that are not just reusable leaf ops: locals,
/// calls, returns, structured control markers, and branch metadata. Ordinary
/// non-structural ops are represented as `Core(CoreOpKind)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SemanticOpKind {
    Core(CoreOpKind),
    LocalGet {
        idx: u16,
    },
    LocalSet {
        idx: u16,
    },
    LocalTee {
        idx: u16,
    },
    Block {
        params: u16,
        results: u16,
    },
    Loop {
        params: u16,
        results: u16,
    },
    If {
        params: u16,
        results: u16,
    },
    Else,
    End,
    Br {
        stack_drop: u32,
        arity: u16,
    },
    BrIf {
        stack_drop: u32,
        arity: u16,
    },
    BrTable {
        entries: Vec<BrTableEntry>,
    },
    CallExternal {
        func_idx: u32,
        params: u16,
        results: u16,
    },
    CallInternal {
        callee: u32,
        params: u16,
        results: u16,
    },
    CallIndirect {
        type_idx: u32,
        table_idx: u32,
        params: u16,
        results: u16,
    },
    ReturnVoid,
    ReturnOne,
    Return {
        arity: u16,
    },
}

/// Semantic program for one function body.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SemanticProgram {
    pub params: u16,
    pub results: u16,
    pub local_count: u16,
    pub max_stack_height: u16,
    pub ops: alloc::vec::Vec<SemanticOp>,
}

impl From<CoreOpKind> for SemanticOpKind {
    #[inline]
    fn from(kind: CoreOpKind) -> Self {
        Self::Core(kind)
    }
}

/// Semantic stack effect.
#[inline]
pub fn stack_effect(kind: &SemanticOpKind) -> (u8, u8) {
    match kind {
        SemanticOpKind::Core(kind) => super::core_op::stack_effect(kind),
        SemanticOpKind::LocalGet { .. } => (0, 1),
        SemanticOpKind::LocalSet { .. } => (1, 0),
        SemanticOpKind::LocalTee { .. } => (0, 0),
        SemanticOpKind::Block { .. }
        | SemanticOpKind::Loop { .. }
        | SemanticOpKind::Else
        | SemanticOpKind::End => (0, 0),
        SemanticOpKind::If { .. } => (1, 0),
        SemanticOpKind::Br { .. } => (0, 0),
        SemanticOpKind::BrIf { .. } | SemanticOpKind::BrTable { .. } => (1, 0),
        SemanticOpKind::CallExternal { .. } | SemanticOpKind::CallInternal { .. } => (0, 0),
        SemanticOpKind::CallIndirect { .. } => (1, 0),
        SemanticOpKind::ReturnVoid | SemanticOpKind::ReturnOne | SemanticOpKind::Return { .. } => {
            (0, 0)
        }
    }
}
