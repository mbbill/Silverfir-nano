//! Interpreter compilation entry.
//!
//! This file runs the shared frontend pipeline for the interpreter backend:
//! Wasm decode -> preparation -> prepared SSA-IR.
//! Final interpreter instruction emission happens in `finalizer.rs`.
//!
//! The interpreter owns the planning inputs it passes into the shared frontend.
//! Shared VM code should not choose those values on its behalf.

use crate::error::WasmError;
use crate::vm::{
    backend::{BackendConfig, BackendKind},
    entities::ModuleInst,
    ssa_ir::ir::SsaProgram,
    plan::{
        config::PlanConfig,
        frame::FrameLayoutPlan,
        group::GroupPlan,
        policy::{GroupingMode, PlanPolicy},
        prepare_function, PrepareInput,
    },
    store::Store,
    wasm::{context::CompileContext, decode, semantic_ir::SemanticProgram},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InterpreterBuildBundle {
    pub backend: BackendKind,
    pub config: PlanConfig,
    pub semantic: SemanticProgram,
    pub frame: FrameLayoutPlan,
    pub groups: GroupPlan,
    pub ssa: SsaProgram,
}

#[inline]
pub const fn normalize_interpreter_backend(_requested: BackendKind) -> BackendKind {
    // Fusion is not re-enabled yet on top of the new frontend/backend boundary.
    BackendKind::Base
}

#[inline]
pub const fn interpreter_backend_config() -> BackendConfig {
    BackendConfig::new(3, 4)
}

#[inline]
pub const fn interpreter_plan_policy(backend: BackendKind) -> PlanPolicy {
    let _backend = normalize_interpreter_backend(backend);
    PlanPolicy::new(GroupingMode::Ignore)
}

pub fn build_interpreter_function(
    code: &[u8],
    backend: BackendKind,
    compile: CompileContext<'_>,
) -> Result<InterpreterBuildBundle, WasmError> {
    let backend = normalize_interpreter_backend(backend);
    let config = PlanConfig::from_backend_config(interpreter_backend_config(), 0);
    let semantic = decode::decode_to_semantic_ir(code, compile)?;
    let prepared = prepare_function(
        PrepareInput {
            config,
            policy: interpreter_plan_policy(backend),
        },
        &semantic,
    )?;

    Ok(InterpreterBuildBundle {
        backend,
        config,
        semantic,
        frame: prepared.frame,
        groups: prepared.groups,
        lir: prepared.lir,
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
        CompileContext {
            types: &module.types,
            store,
            params,
            local_count,
            results,
            local_types: &[],
            result_types: &[],
        },
    )
}
