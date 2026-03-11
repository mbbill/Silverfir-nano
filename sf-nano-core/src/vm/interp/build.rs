//! Interpreter compilation entry.
//!
//! This file runs the shared frontend pipeline for the interpreter backend:
//! Wasm decode -> planning -> LIR lowering.
//! Final interpreter instruction emission happens in `finalizer.rs`.
//!
//! The interpreter owns the planning inputs it passes into the shared frontend.
//! Shared VM code should not choose those values on its behalf.

use crate::error::WasmError;
use crate::vm::{
    backend::{BackendConfig, BackendKind},
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InterpreterBuildBundle {
    pub backend: BackendKind,
    pub config: PlanConfig,
    pub semantic: SemanticProgram,
    pub planned: PlannedProgram,
    pub lir: LirProgram,
}

#[inline]
pub const fn normalize_interpreter_backend(_requested: BackendKind) -> BackendKind {
    // Fusion is not re-enabled yet on top of the new frontend/backend boundary.
    BackendKind::Base
}

#[inline]
pub const fn interpreter_backend_config() -> BackendConfig {
    BackendConfig {
        ctx_register_count: 1,
        fp_register_count: 1,
        tmp_register_count: 0, // not needed for the interpreter.
        hot_local_count: 3,
        tos_register_count: 4,
    }
}

#[inline]
pub const fn interpreter_plan_policy(backend: BackendKind) -> PlanPolicy {
    let backend = normalize_interpreter_backend(backend);
    PlanPolicy::new(backend, GroupingMode::Ignore)
}

pub fn build_interpreter_function(
    code: &[u8],
    backend: BackendKind,
    compile: CompileContext<'_>,
) -> Result<InterpreterBuildBundle, WasmError> {
    let backend = normalize_interpreter_backend(backend);
    let config = PlanConfig::for_backend(backend, interpreter_backend_config());
    let semantic = decode::decode_to_semantic_ir(code, compile)?;
    let planned = build_planned_program(
        PlanningInput {
            config,
            policy: interpreter_plan_policy(backend),
        },
        &semantic,
    );
    let lir = lower::lower_to_lir(&semantic, &planned, config)?;

    Ok(InterpreterBuildBundle {
        backend,
        config,
        semantic,
        planned,
        lir,
    })
}

pub fn build_interpreter_function_for_spec(
    code: &[u8],
    backend: BackendKind,
    store: &Store,
    module: &ModuleInst,
    params: u16,
    local_count: u16,
    results: u16,
) -> Result<InterpreterBuildBundle, WasmError> {
    build_interpreter_function(
        code,
        backend,
        CompileContext::new(&module.types, store, module, params, local_count, results),
    )
}
