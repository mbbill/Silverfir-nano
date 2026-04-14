//! Joint-plan validation.

use crate::{error::WasmError, vm::middle::cfg::SemanticCfg};

use super::facts::FunctionPlan;

pub(crate) fn validate_plan(
    cfg: &SemanticCfg,
    slot_block_count: usize,
    plan: &FunctionPlan,
) -> Result<(), WasmError> {
    if plan.blocks.len() != cfg.blocks.len() {
        return Err(WasmError::internal(
            "joint plan has blocks, but cfg has blocks",
        ));
    }
    if slot_block_count != cfg.blocks.len() {
        return Err(WasmError::internal(
            "slot-only SSA has blocks, but cfg has blocks",
        ));
    }
    if plan.compact_entries.len() != cfg.semantic_to_block.len() {
        return Err(WasmError::internal(
            "joint plan has compact entries, but semantic length is",
        ));
    }
    Ok(())
}
