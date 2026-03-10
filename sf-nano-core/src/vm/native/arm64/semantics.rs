//! Structured ARM64 codegen semantics helpers.

use crate::vm::lir::legacy::ir::LirOpKind;

/// Structured semantic helper for ARM64 codegen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticAction {
    pub kind: &'static str,
}

pub fn describe(kind: &LirOpKind) -> alloc::vec::Vec<SemanticAction> {
    match kind {
        LirOpKind::Core(_) => alloc::vec![
            SemanticAction {
                kind: "read-window"
            },
            SemanticAction {
                kind: "apply-core-op"
            },
            SemanticAction {
                kind: "write-window"
            },
        ],
        LirOpKind::CacheTransfer(_) => alloc::vec![SemanticAction {
            kind: "frame-cache-transfer",
        }],
        LirOpKind::LocalGetHot { .. }
        | LirOpKind::LocalSetHot { .. }
        | LirOpKind::LocalTeeHot { .. } => {
            alloc::vec![SemanticAction {
                kind: "move-hot-local",
            }]
        }
        LirOpKind::LocalGetFrame { .. }
        | LirOpKind::LocalSetFrame { .. }
        | LirOpKind::LocalTeeFrame { .. } => alloc::vec![SemanticAction {
            kind: "move-frame-local",
        }],
        LirOpKind::CallExternal { .. } | LirOpKind::CallIndirect { .. } => {
            alloc::vec![SemanticAction {
                kind: "cold-helper-wrapper",
            }]
        }
        LirOpKind::CallInternal { .. } => alloc::vec![SemanticAction {
            kind: "direct-internal-call",
        }],
        LirOpKind::Branch { .. } | LirOpKind::BrTable { .. } => alloc::vec![SemanticAction {
            kind: "direct-control-transfer",
        }],
        LirOpKind::Return { .. } => alloc::vec![SemanticAction {
            kind: "return-transfer",
        }],
        LirOpKind::Marker(_) | LirOpKind::InitLocals { .. } => alloc::vec![SemanticAction {
            kind: "marker-or-prologue",
        }],
    }
}
