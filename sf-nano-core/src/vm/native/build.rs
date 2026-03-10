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

use crate::error::WasmError;
use crate::vm::{
    backend::{BackendKind, BackendMode},
    entities::ModuleInst,
    lir::legacy::lower::{self, LirProgram},
    plan::{
        build_planned_program, config::PlanConfig, policy::PlanPolicy, PlannedProgram,
        PlanningInput,
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

pub fn build_native_function(
    code: &[u8],
    compile: CompileContext<'_>,
) -> Result<NativeBuildBundle, WasmError> {
    let backend = BackendKind::Native;
    let config = PlanConfig::for_backend(backend, backend.default_config());
    let semantic = decode::decode_to_semantic_ir(code, compile)?;
    let planned = build_planned_program(
        PlanningInput {
            config,
            policy: PlanPolicy::for_backend(backend),
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
