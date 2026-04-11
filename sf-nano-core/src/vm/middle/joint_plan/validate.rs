//! Joint-plan validation.

use crate::{
    error::WasmError,
    vm::middle::{cfg::SemanticCfg, slot_ssa::SlotSsaProgram},
};

use super::facts::FunctionPlan;

pub(crate) fn validate_plan(
    cfg: &SemanticCfg,
    slot_program: &SlotSsaProgram,
    plan: &FunctionPlan,
) -> Result<(), WasmError> {
    if plan.blocks.len() != cfg.blocks.len() {
        return Err(WasmError::internal(
            "joint plan has blocks, but cfg has blocks",
        ));
    }
    if slot_program.blocks.len() != cfg.blocks.len() {
        return Err(WasmError::internal(
            "slot-only SSA has blocks, but cfg has blocks",
        ));
    }
    if plan.op_plans.len() != cfg.semantic_to_block.len() {
        return Err(WasmError::internal(
            "joint plan has per-op plans, but semantic length is",
        ));
    }
    if plan.entry_states.len() != cfg.semantic_to_block.len() {
        return Err(WasmError::internal(
            "joint plan has entry states, but semantic length is",
        ));
    }
    Ok(())
}
