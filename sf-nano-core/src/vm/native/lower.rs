//! Native-specific lowering on top of shared LIR, if needed.
//!
//! This file exists only if the native backend needs an extra internal
//! representation after shared LIR. It must not reintroduce stack semantics.

use alloc::vec::Vec;

use crate::vm::lir::ir::LirOp;

/// Native-internal op, only if ARM64/native needs a post-LIR normalization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeOp {
    pub lir_index: usize,
}

pub fn lower_native(_lir: &[LirOp]) -> Vec<NativeOp> {
    // TODO: Keep this empty if native can consume shared LIR directly.
    Vec::new()
}
