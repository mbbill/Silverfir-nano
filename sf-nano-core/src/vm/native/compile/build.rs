//! Native backend build entry.
//!
//! This file runs the shared frontend pipeline for the native backend:
//! Wasm decode -> planning -> CFG + SSA LIR.
//! Native-owned lowering starts after this bundle in `lower.rs`.
//!
//! The native backend owns the planning inputs it passes into the shared
//! frontend, and it must preserve the full backend register budget for later
//! native IR lowering.

use crate::error::WasmError;
use crate::vm::{
    backend::{BackendConfig, BackendKind, BackendMode},
    entities::ModuleInst,
    lir::{ir::LirProgram, lower},
    plan::{
        build_planned_program,
        config::PlanConfig,
        policy::{GroupingMode, PlanPolicy},
        PlannedProgram, PlanningInput,
    },
    store::Store,
    wasm::{context::CompileContext, decode, semantic_ir::SemanticProgram},
};

/// Native build bundle for one function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeBuildBundle {
    pub backend: BackendKind,
    pub backend_config: BackendConfig,
    pub plan_config: PlanConfig,
    pub semantic: SemanticProgram,
    pub planned: PlannedProgram,
    pub lir: LirProgram,
}

#[inline]
pub const fn native_backend_config() -> BackendConfig {
    BackendConfig {
        ctx_register_count: 1,
        fp_register_count: 1,
        tmp_register_count: 4,
        hot_local_count: 3,
        tos_register_count: 4,
    }
}

#[inline]
pub const fn native_plan_policy() -> PlanPolicy {
    PlanPolicy::new(BackendKind::Native, GroupingMode::Maximal)
}

pub fn build_native_function(
    code: &[u8],
    compile: CompileContext<'_>,
) -> Result<NativeBuildBundle, WasmError> {
    let backend = BackendKind::Native;
    let backend_config = native_backend_config();
    let plan_config = PlanConfig::for_backend(backend, backend_config);
    let semantic = decode::decode_to_semantic_ir(code, compile)?;
    let planned = build_planned_program(
        PlanningInput {
            config: plan_config,
            policy: native_plan_policy(),
        },
        &semantic,
    )?;
    let lir = lower::lower_to_lir(&semantic, &planned, plan_config)?;
    Ok(NativeBuildBundle {
        backend,
        backend_config,
        plan_config,
        semantic,
        planned,
        lir,
    })
}

pub fn native_backend_mode() -> BackendMode {
    BackendMode::Native
}

pub fn build_native_function_for_spec(
    code: &[u8],
    store: &Store,
    module: &ModuleInst,
    params: u16,
    local_count: u16,
    results: u16,
) -> Result<NativeBuildBundle, WasmError> {
    build_native_function(
        code,
        CompileContext::new(&module.types, store, module, params, local_count, results),
    )
}

#[cfg(test)]
mod tests {
    use super::native_plan_policy;
    use crate::vm::plan::policy::GroupingMode;

    #[test]
    fn native_build_uses_maximal_grouping() {
        assert_eq!(native_plan_policy().grouping, GroupingMode::Maximal);
    }
}
