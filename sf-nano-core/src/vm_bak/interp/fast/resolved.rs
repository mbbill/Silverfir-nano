//! Shared resolved instruction form for fast-interpreter backends.
//!
//! Base, fusion, and native all target this representation before finalization
//! into the fast interpreter's final `Instruction` stream.

use alloc::vec::Vec;

use crate::vm::abi::compaction::CompactionDisposition;
use crate::vm::lir::{
    IrOp, IrOpKind, OpIndex, current_variant_from_window, push_variant_from_window, stack_effect,
};

use super::resolve::resolve_handler;
use super::handlers::OpHandler as Handler;
use super::handlers::full_set::op_nop;

/// A fully resolved instruction — handler and encoding decided.
#[derive(Clone)]
pub struct ResolvedInst {
    pub handler: Handler,
    pub kind: IrOpKind,
    pub frame_slot: Option<crate::vm::wasm::SlotRef>,
    pub alt_target: Option<OpIndex>,
    pub has_target: bool,
    pub compaction: CompactionDisposition,
}

impl ResolvedInst {
    /// Create a 1:1 resolved instruction from a lowered IR op.
    #[inline]
    pub fn from_ir(op: &IrOp) -> Self {
        Self {
            handler: resolve_handler(&op.kind, handler_variant(op)),
            kind: op.kind.clone(),
            frame_slot: op.frame_slot,
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
            frame_slot: None,
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

fn handler_variant(op: &IrOp) -> u8 {
    match op.kind {
        IrOpKind::Block
        | IrOpKind::Loop
        | IrOpKind::End
        | IrOpKind::Nop
        | IrOpKind::InitLocals { .. }
        | IrOpKind::Term
        | IrOpKind::Data { .. }
        | IrOpKind::Fused { .. } => 0,
        _ => {
            let (pop, push) = stack_effect(&op.kind);
            if pop == 0 && push > 0 {
                push_variant_from_window(op.window)
            } else {
                current_variant_from_window(op.window)
            }
        }
    }
}

/// Resolve all lowered IR ops to base (1:1) handlers.
pub fn resolve_base(ir: &[IrOp]) -> Vec<ResolvedInst> {
    ir.iter().map(ResolvedInst::from_ir).collect()
}
