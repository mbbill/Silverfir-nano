//! Joint-plan validation.

use crate::{
    error::WasmError,
    vm::middle::{cfg::SemanticCfg, slot_ssa::SlotSsaProgram},
};

use super::types::FunctionPlan;

pub(crate) fn validate_plan(
    cfg: &SemanticCfg,
    slot_program: &SlotSsaProgram,
    plan: &FunctionPlan,
) -> Result<(), WasmError> {
    if plan.blocks.len() != cfg.blocks.len() {
        return Err(WasmError::internal(alloc::format!(
            "joint plan has {} blocks, but cfg has {} blocks",
            plan.blocks.len(),
            cfg.blocks.len(),
        )));
    }
    if slot_program.blocks.len() != cfg.blocks.len() {
        return Err(WasmError::internal(alloc::format!(
            "slot-only SSA has {} blocks, but cfg has {} blocks",
            slot_program.blocks.len(),
            cfg.blocks.len(),
        )));
    }
    if plan.op_plans.len() != cfg.semantic_to_block.len() {
        return Err(WasmError::internal(alloc::format!(
            "joint plan has {} per-op plans, but semantic length is {}",
            plan.op_plans.len(),
            cfg.semantic_to_block.len(),
        )));
    }
    if plan.entry_states.len() != cfg.semantic_to_block.len() {
        return Err(WasmError::internal(alloc::format!(
            "joint plan has {} entry states, but semantic length is {}",
            plan.entry_states.len(),
            cfg.semantic_to_block.len(),
        )));
    }
    Ok(())
}
