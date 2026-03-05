//! Backend pass: resolve IR ops to handler + encoding.
//!
//! Provides `ResolvedInst` — the output type shared by JIT, fusion, and base backends.
//! All backends fall back to 1:1 base handler resolution for ops they can't optimize.

use alloc::vec::Vec;

use super::ir::{IrOp, IrOpKind};
use super::ir_resolve::resolve_handler;
use super::super::handlers::full_set::op_nop;
use super::super::handlers::OpHandler as Handler;

/// A fully resolved instruction — handler and encoding decided.
///
/// Produced by the backend pass (JIT, fusion, or base 1:1).
/// Consumed by the finalizer to build the final `Box<[Instruction]>`.
#[derive(Clone)]
pub struct ResolvedInst {
    /// Handler function pointer.
    pub handler: Handler,
    /// IR kind — used for encoding and structural checks in the finalizer.
    /// For backend-resolved ops, this carries a compatible kind for encoding
    /// (e.g. `BrIfSimple` for JIT br_if groups, `Data { 0,0,0 }` for JIT linear groups).
    pub kind: IrOpKind,
    /// Branch target (IR index, pre-compaction).
    pub alt_target: Option<usize>,
    /// Whether this instruction encodes a target field that needs pointer patching.
    pub has_target: bool,
    /// Whether this is a structural no-op (removed during compaction).
    pub structural: bool,
}

impl ResolvedInst {
    /// Create a 1:1 resolved instruction from an IR op.
    #[inline]
    pub fn from_ir(op: &IrOp) -> Self {
        Self {
            handler: resolve_handler(&op.kind, op.variant),
            kind: op.kind.clone(),
            alt_target: op.alt_target,
            has_target: op.has_target,
            structural: matches!(
                op.kind,
                IrOpKind::Nop | IrOpKind::Block | IrOpKind::Loop | IrOpKind::End
            ),
        }
    }

    /// Create a structural no-op (removed during compaction).
    #[inline]
    pub fn skip() -> Self {
        Self {
            handler: op_nop as Handler,
            kind: IrOpKind::Nop,
            alt_target: None,
            has_target: false,
            structural: true,
        }
    }
}

/// Resolve all IR ops to base (1:1) handlers.
///
/// Used when neither JIT nor fusion is enabled, or when the backend
/// can't allocate resources (e.g. JIT buffer allocation fails).
pub fn resolve_base(ir: &[IrOp]) -> Vec<ResolvedInst> {
    ir.iter().map(ResolvedInst::from_ir).collect()
}
