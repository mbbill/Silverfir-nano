//! Post-rewrite cleanup passes.
//!
//! This file intentionally starts as a no-op. Later passes like block merging,
//! jump threading, and redundant ensure/drop cleanup belong here.

use super::ssa_ir::ir::SsaProgram;

pub(crate) fn cleanup_program(_program: &mut SsaProgram) {}
