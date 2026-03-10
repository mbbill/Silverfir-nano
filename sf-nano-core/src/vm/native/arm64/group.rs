//! ARM64 group code-shape helpers.
//!
//! Important:
//! this file should *not* rediscover groups. It should only consume planned
//! group contents and emit code shape for them.

use crate::vm::{
    lir::{ir::LirOp, lower::LirGroup},
};

use super::codegen::Arm64Codegen;

/// One ARM64-native compiled group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompiledGroup {
    pub lir_range: core::ops::Range<usize>,
}

pub fn compile_group(
    _lir: &[LirOp],
    group: &LirGroup,
    _codegen: &mut Arm64Codegen,
) -> CompiledGroup {
    // TODO: Compile exactly the provided planned group. Do not rediscover or
    // reshape grouping policy here.
    CompiledGroup {
        lir_range: group.lir_range.clone(),
    }
}
