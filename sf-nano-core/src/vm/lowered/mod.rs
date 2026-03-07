//! Backend-lowered IR shared by the fast/fusion/native execution backends.
//!
//! `vm::compile` owns semantic decode and planning.
//! `vm::lowered` owns the backend-specific lowering step that turns semantic IR
//! into the current hot-local/frame-layout/spill-aware IR consumed by the fast
//! interpreter family and native backend.

pub mod backend_lower;
pub mod ir;

use alloc::vec::Vec;

use crate::{
    error::WasmError,
    vm::{compile::{CompileContext, StackTracker}, planner::HotLocalPlan},
};

pub use ir::{BrTableEntry, IrOp, IrOpKind, OpIndex, SlotRef, stack_effect};

/// Lower Wasm function body all the way to backend-lowered IR.
pub fn lower_to_ir<'a>(
    code: &'a [u8],
    ctx: &'a CompileContext<'a>,
    stack: &'a mut StackTracker,
    hot_locals: HotLocalPlan,
) -> Result<Vec<IrOp>, WasmError> {
    let semantic_ops = crate::vm::compile::ir_lower::lower_to_semantic_ir(code, ctx, stack)?;
    Ok(backend_lower::lower_to_lowered_ir(
        semantic_ops,
        hot_locals,
        stack.config(),
        stack.frame_size(),
    ))
}
