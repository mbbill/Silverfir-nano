//! Shared resolved instruction form for fast-interpreter backends.
//!
//! Base, fusion, and native all target this representation before finalization
//! into the fast interpreter's final `Instruction` stream.

use alloc::vec::Vec;

use crate::vm::compaction::CompactionDisposition;
use crate::vm::lowered::{IrOp, IrOpKind, OpIndex};

use super::compiler::ir_resolve::resolve_handler;
use super::handlers::OpHandler as Handler;
use super::handlers::full_set::op_nop;

/// A fully resolved instruction — handler and encoding decided.
#[derive(Clone)]
pub struct ResolvedInst {
    pub handler: Handler,
    pub kind: IrOpKind,
    pub alt_target: Option<OpIndex>,
    pub has_target: bool,
    pub compaction: CompactionDisposition,
}

impl ResolvedInst {
    /// Create a 1:1 resolved instruction from a lowered IR op.
    #[inline]
    pub fn from_ir(op: &IrOp) -> Self {
        Self {
            handler: resolve_handler(&op.kind, op.variant),
            kind: op.kind.clone(),
            alt_target: op.alt_target,
            has_target: op.has_target,
            compaction: match op.kind {
                IrOpKind::Nop | IrOpKind::Block | IrOpKind::Loop | IrOpKind::End => {
                    CompactionDisposition::RedirectBranchTarget
                }
                _ => CompactionDisposition::Keep,
            },
        }
    }

    /// Create an internal-only marker removed during compaction.
    #[inline]
    pub fn skip() -> Self {
        Self {
            handler: op_nop as Handler,
            kind: IrOpKind::Nop,
            alt_target: None,
            has_target: false,
            compaction: CompactionDisposition::InternalOnly,
        }
    }

    #[inline]
    pub fn is_removed(&self) -> bool {
        !self.compaction.is_kept()
    }

    #[inline]
    pub fn redirects_branch_target(&self) -> bool {
        self.compaction.may_redirect_branch_target()
    }

    #[inline]
    pub fn is_internal_only(&self) -> bool {
        matches!(self.compaction, CompactionDisposition::InternalOnly)
    }
}

/// Resolve all lowered IR ops to base (1:1) handlers.
pub fn resolve_base(ir: &[IrOp]) -> Vec<ResolvedInst> {
    ir.iter().map(ResolvedInst::from_ir).collect()
}
