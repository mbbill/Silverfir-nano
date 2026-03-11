//! Planning pipeline orchestration.
//!
//! This is the handoff between semantic Wasm and backend-facing IR lowering.

use crate::error::WasmError;
use crate::vm::wasm::semantic_ir::SemanticProgram;

use super::{
    frame::plan_frame_layout,
    group::plan_groups,
    hot_local::{analyze_hot_locals, plan_hot_local_inits},
    place::place_semantic_ops,
    tos::plan_tos_ops,
};

pub use super::types::*;

/// Plan semantic IR into the stack-aware planned layer.
pub fn build_planned_program(
    input: PlanningInput,
    semantic: &SemanticProgram,
) -> Result<PlannedProgram, WasmError> {
    semantic.validate()?;
    let PlanningInput { config, policy } = input;
    let hot_locals = analyze_hot_locals(semantic, config);
    let frame = plan_frame_layout(
        semantic.local_count,
        semantic.max_stack_height,
        config.hot_local_count,
    );
    let placement = place_semantic_ops(semantic, frame, hot_locals.as_ref(), &policy);
    let ops = plan_tos_ops(
        placement.ops,
        semantic.results,
        frame,
        config.tos_register_count,
        plan_hot_local_inits(hot_locals.as_ref(), frame),
    );
    let groups = plan_groups(&placement.group_inputs);
    let program = PlannedProgram {
        frame,
        hot_locals,
        ops,
        groups,
    };
    program.validate(semantic, config)?;
    Ok(program)
}
