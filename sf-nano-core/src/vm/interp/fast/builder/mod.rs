//! Modular IR builder for the fast interpreter.
//!
//! Pipeline:
//!   Wasm bytecode → ir_lower → Vec<IrOp> → selected backend → Vec<ResolvedInst> → finalizer → Box<[Instruction]>
//!
//! Modules:
//! - `context`: Function metadata and type resolution
//! - `stack`: Compile-time stack tracking
//! - `ir`: Neutral IR types (purely semantic)
//! - `ir_lower`: Wasm→IR lowering
//! - `ir_resolve`: Handler/operand resolution for IR ops
//! - `backend`: ResolvedInst type, base 1:1 resolution
//! - `finalizer_ir`: Compact, patch, and build final instructions

pub mod backend;
mod context;
mod finalizer_ir;
pub mod hot_local;
pub mod ir;
#[cfg(feature = "ir-dump")]
mod ir_dump;
pub mod ir_lower;
pub mod ir_resolve;
mod stack;

pub use context::CompileContext;
pub use stack::{BlockKind, ControlFrame, StackTracker, HOT_LOCAL_COUNT};

use crate::{
    error::WasmError,
    module::entities::FunctionSpec,
    vm::{
        entities::ModuleInst,
        interp::fast::instruction::Instruction,
        store::Store,
    },
};

use alloc::{rc::Rc, vec::Vec};
use crate::module::type_defs::FunctionType;
use super::{BackendKind, BackendMode};

#[cfg(feature = "micro-jit")]
fn try_resolve_jit_backend(
    ir_ops: &[ir::IrOp],
    module: &ModuleInst,
    hot_mask: [bool; 3],
) -> Result<Vec<backend::ResolvedInst>, &'static str> {
    use super::jit::group;

    let mut buf = module.jit_code_buffer()?;
    Ok(group::resolve_jit(ir_ops, &mut buf, hot_mask))
}

#[cfg(not(feature = "micro-jit"))]
fn try_resolve_jit_backend(
    _ir_ops: &[ir::IrOp],
    _module: &ModuleInst,
    _hot_mask: [bool; 3],
) -> Result<Vec<backend::ResolvedInst>, &'static str> {
    Err("micro-jit backend not compiled in")
}

#[cfg(feature = "fusion")]
fn resolve_fusion_backend(ir_ops: &[ir::IrOp]) -> Result<Vec<backend::ResolvedInst>, &'static str> {
    Ok(super::fusion::resolve::resolve_fusion(ir_ops))
}

#[cfg(not(feature = "fusion"))]
fn resolve_fusion_backend(_ir_ops: &[ir::IrOp]) -> Result<Vec<backend::ResolvedInst>, &'static str> {
    Err("fusion backend not compiled in")
}

/// Build fast IR via the unified IR pipeline.
///
/// Lowers Wasm bytecode to neutral IR, resolves via backend (JIT or base 1:1),
/// then finalizes into the interpreter's `Box<[Instruction]>`.
pub fn build_for_function(
    function: &FunctionSpec,
    types: Option<&[Rc<FunctionType>]>,
    store: &Store,
    module: &ModuleInst,
    func_idx: u32,
) -> Result<*mut Instruction, WasmError> {
    let code = function.code();
    let func_type = function.func_type();
    let params_count = func_type.params().len();
    let results_count = func_type.results().len();
    let locals_count = function.locals().len();

    let ctx = CompileContext::new(types, store, module, results_count);
    let frame_size = params_count + locals_count;
    let raw_hot_locals = hot_local::find_hot_locals(code, frame_size);
    let hot_locals = hot_local::compute_effective_indices(&raw_hot_locals, frame_size);

    let mut stack = StackTracker::new(params_count, locals_count, results_count, hot_locals);

    // IR pipeline: lower to neutral IR
    let ir_ops = ir_lower::lower_to_ir(code, &ctx, &mut stack, hot_locals)?;

    #[cfg(feature = "ir-dump")]
    ir_dump::dump_ir(func_idx, &ir_ops, hot_locals);

    let hot_mask = [
        hot_locals[0].is_some(),
        hot_locals[1].is_some(),
        hot_locals[2].is_some(),
    ];

    // Backend: resolve IR to handlers.
    let resolved = match super::backend_mode() {
        BackendMode::Base => backend::resolve_base(&ir_ops),
        BackendMode::Fusion => resolve_fusion_backend(&ir_ops)
            .map_err(|err| WasmError::invalid(alloc::format!("requested backend fusion is unavailable: {}", err)))?,
        BackendMode::Jit => try_resolve_jit_backend(&ir_ops, module, hot_mask)
            .map_err(|err| WasmError::invalid(alloc::format!("requested backend jit is unavailable: {}", err)))?,
        BackendMode::Auto => match super::active_backend().unwrap_or(BackendKind::Base) {
            BackendKind::Jit => match try_resolve_jit_backend(&ir_ops, module, hot_mask) {
                Ok(resolved) => resolved,
                Err(_) => match resolve_fusion_backend(&ir_ops) {
                    Ok(resolved) => resolved,
                    Err(_) => backend::resolve_base(&ir_ops),
                },
            },
            BackendKind::Fusion => match resolve_fusion_backend(&ir_ops) {
                Ok(resolved) => resolved,
                Err(_) => backend::resolve_base(&ir_ops),
            },
            BackendKind::Base => backend::resolve_base(&ir_ops),
        },
    };

    #[cfg(feature = "ir-dump")]
    ir_dump::dump_resolved(func_idx, &resolved, &ir_ops);

    // Finalize: Vec<ResolvedInst> → Box<[Instruction]>
    let code_box = finalizer_ir::finalize(resolved, &mut stack);

    // Store in function spec
    use crate::vm::interp::fast::fast_code::create_fast_code;
    let (fast_code, fast_cache) = create_fast_code(code_box, params_count, locals_count, results_count);
    let entry = fast_cache.entry();
    function.set_fast_code(fast_code, fast_cache);
    Ok(entry)
}
