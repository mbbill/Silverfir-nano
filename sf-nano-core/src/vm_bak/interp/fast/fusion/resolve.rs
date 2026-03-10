//! Fusion backend: resolve IR ops by matching fused patterns.
//!
//! Scans `&[IrOp]` for consecutive ops that match fused handler patterns,
//! and falls back to 1:1 base handler resolution for non-matching ops.

use alloc::vec::Vec;

use crate::vm::lir::{IrOp, IrOpKind, OpIndex};
use super::super::resolved::{CompactionDisposition, ResolvedInst};
use super::super::handlers::OpHandler as Handler;
use super::super::handlers::full_set;
#[allow(unused_imports)]
use super::super::handler_lookup;
use super::super::encoding;
use super::super::resolve::resolve_handler;

// Include the generated IR-level fusion pattern matching code.
// The generated file defines `try_fuse()` and per-pattern `try_fuse_*()` functions.
// It does NOT contain any `use` statements — all imports are above.
#[allow(non_snake_case)]
mod _fusion_ir_match {
    use super::*;
    include!(concat!(env!("OUT_DIR"), "/fast_interp/fast_fusion_ir_match.rs"));
}
use _fusion_ir_match::try_fuse;

/// Resolve IR ops via fusion pattern matching.
///
/// Matches consecutive IrOps against fused handler patterns.
/// Non-matching ops fall back to 1:1 base handler resolution.
pub fn resolve_fusion(ir: &[IrOp]) -> Vec<ResolvedInst> {
    let mut out = Vec::with_capacity(ir.len());
    let mut i = 0;

    while i < ir.len() {
        if let Some((len, handler, kind, alt_target, has_target)) = try_fuse(ir, i) {
            // Fused match: emit the fused entry
            out.push(ResolvedInst {
                handler,
                kind,
                alt_target,
                has_target,
                compaction: CompactionDisposition::Keep,
            });
            // Mark remaining ops in the group as internal-only removed slots.
            for _ in 1..len {
                out.push(ResolvedInst::skip());
            }
            i += len;
            continue;
        }

        // 1:1 fallback
        out.push(ResolvedInst::from_ir(&ir[i]));
        i += 1;
    }

    out
}
