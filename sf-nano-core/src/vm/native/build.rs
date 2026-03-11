//! Native backend build entry.
//!
//! Intended shape:
//! 1. decode Wasm to semantic IR
//! 2. run stack-aware planning
//! 3. form shared groups
//! 4. lower to LIR
//! 5. resolve native entries from planned groups + LIR
//! 6. finalize code / metadata
//!
//! This file must not bypass planning by rediscovering groups in the backend.
//! It also owns the planning inputs it passes into the shared frontend.

use crate::error::WasmError;
use crate::vm::{
    backend::{BackendConfig, BackendKind, BackendMode},
    entities::ModuleInst,
    lir::legacy::lower::{self, LirProgram},
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
    pub config: PlanConfig,
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
    let config = PlanConfig::for_backend(backend, native_backend_config());
    let semantic = decode::decode_to_semantic_ir(code, compile)?;
    let planned = build_planned_program(
        PlanningInput {
            config,
            policy: native_plan_policy(),
        },
        &semantic,
    );
    let lir = lower::lower_to_lir(planned.clone())?;
    Ok(NativeBuildBundle {
        backend,
        config,
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
