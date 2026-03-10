//! Fast backend build entry.
//!
//! This mirrors the native path:
//! 1. decode Wasm to semantic IR
//! 2. run shared stack-aware planning
//! 3. form shared groups
//! 4. lower to LIR
//! 5. resolve/finalize into fast interpreter artifacts

use crate::error::WasmError;
use crate::vm::{
    backend::{BackendKind, BackendMode},
    lir::lower::{self, LirProgram},
    entities::ModuleInst,
    plan::{
        config::PlanConfig,
        plan::{self, PlannedProgram, PlanningInput},
        policy::PlanPolicy,
    },
    store::Store,
    wasm::{context::CompileContext, decode, semantic_ir::SemanticProgram},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FastBuildBundle {
    pub backend: BackendKind,
    pub config: PlanConfig,
    pub semantic: SemanticProgram,
    pub planned: PlannedProgram,
    pub lir: LirProgram,
}

pub fn build_fast_function(
    code: &[u8],
    backend: BackendKind,
    compile: CompileContext<'_>,
) -> Result<FastBuildBundle, WasmError> {
    let config = PlanConfig::for_backend(backend, backend.default_config());
    let semantic = decode::decode_to_semantic_ir(code, compile)?;
    let planned = plan::build_planned_program(
        PlanningInput {
            config,
            policy: PlanPolicy::for_backend(backend),
            hot_locals: None,
        },
        &semantic,
    );
    let lir = lower::lower_to_lir(planned.clone())?;

    Ok(FastBuildBundle {
        backend,
        config,
        semantic,
        planned,
        lir,
    })
}

pub fn fast_build_mode() -> BackendMode {
    BackendMode::Fusion
}

pub fn build_fast_function_for_spec(
    code: &[u8],
    backend: BackendKind,
    store: &Store,
    module: &ModuleInst,
    params: u16,
    local_count: u16,
    results: u16,
) -> Result<FastBuildBundle, WasmError> {
    build_fast_function(
        code,
        backend,
        CompileContext::new(&module.types, store, module, params, local_count, results),
    )
}
