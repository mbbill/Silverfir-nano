//! ARM64 codegen for native groups and singleton entries.

use crate::vm::{
    lir::legacy::{ir::LirOp, lower::LirGroup},
    native::bridge::ColdWrapperSite,
};

/// ARM64 code emission state.
#[derive(Debug, Default)]
pub struct Arm64Codegen;

impl Arm64Codegen {
    pub fn new() -> Self {
        Self
    }

    pub fn emit_group(&mut self, _ops: &[LirOp], _group: &LirGroup) {
        // TODO: Emit exactly the planned group.
    }

    pub fn emit_singleton(&mut self, _op: &LirOp, _lir_index: usize) {
        // TODO: Emit one direct native entry for one LIR op.
    }

    pub fn emit_cold_wrapper(&mut self, _site: &ColdWrapperSite) {
        // TODO: Emit one uniform helper wrapper site.
    }
}
