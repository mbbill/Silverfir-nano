//! Naive joint planning implementation.
//!
//! This module owns the currently simple algorithm. The goal is that future
//! planning improvements replace logic here without requiring lowering to be
//! rewritten.

use crate::{
    error::WasmError,
    vm::{
        backend::BackendConfig, middle::ssa_ir::ir::SsaProgram, wasm::semantic_ir::SemanticProgram,
    },
};

use super::{boundaries, exact_stack, LinearPlan};
use crate::vm::middle::frame::FrameLayoutPlan;

pub(crate) fn plan_linear_semantics<'a>(
    semantic: &'a SemanticProgram,
    frame: FrameLayoutPlan,
    config: BackendConfig,
    gp_dynamic_budget: u8,
    fp_dynamic_budget: u8,
) -> Result<LinearPlan<'a>, WasmError> {
    exact_stack::prepare_semantic_ops(
        semantic,
        frame,
        config,
        gp_dynamic_budget,
        fp_dynamic_budget,
    )
}

pub(crate) fn plan_boundaries(
    program: &mut SsaProgram,
    gp_unit_bytes: u8,
    gp_dynamic_budget: u8,
    fp_dynamic_budget: u8,
) {
    boundaries::plan_resources(program, gp_unit_bytes, gp_dynamic_budget, fp_dynamic_budget);
}
