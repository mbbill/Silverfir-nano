//! Joint-plan validation.

use crate::{error::WasmError, vm::middle::cfg::SemanticCfg};

use super::facts::FunctionPlan;

pub(crate) fn validate_plan(cfg: &SemanticCfg, plan: &FunctionPlan) -> Result<(), WasmError> {
    if plan.blocks.len() != cfg.blocks.len() {
        return Err(WasmError::internal(
            "joint plan has blocks, but cfg has blocks",
        ));
    }
    Ok(())
}
