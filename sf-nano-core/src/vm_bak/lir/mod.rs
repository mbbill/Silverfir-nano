//! Backend-lowered IR shared by the fast/fusion/native execution backends.
//!
//! `vm::wasm` owns semantic decode and planning.
//! `vm::lir` owns the backend-specific lowering step that turns semantic IR
//! into the current hot-local/frame-layout/spill-aware IR consumed by the fast
//! interpreter family and native backend.

pub mod lower;
pub mod ir;
#[cfg(feature = "ir-dump")]
pub mod dump;

use alloc::vec::Vec;

use crate::{
    error::WasmError,
    vm::{plan::HotLocalPlan, wasm::{CompileContext, StackTracker, semantic_ir::SemanticOp}},
};

pub use ir::{
    BrTableEntry, IrOp, IrOpKind, OpIndex, SlotRef, TOS_REGISTER_COUNT,
    current_variant_from_window, push_variant_from_window, stack_effect,
    window_from_height,
};
pub use lower::LoweredProgram;

/// Lower Wasm function body all the way to backend-lowered IR.
pub fn lower_to_ir<'a>(
    code: &'a [u8],
    ctx: &'a CompileContext<'a>,
    stack: &'a mut StackTracker,
    hot_locals: HotLocalPlan,
) -> Result<Vec<IrOp>, WasmError> {
    let semantic_ops = crate::vm::wasm::decode::lower_to_semantic_ir(code, ctx, stack)?;
    Ok(lower::lower_to_lowered_ir(
        semantic_ops,
        hot_locals,
        stack.config(),
        stack.frame_size(),
    ))
}

pub fn lower_semantic_to_program(
    semantic_ops: Vec<SemanticOp>,
    hot_locals: HotLocalPlan,
    config: crate::vm::plan::CompileConfig,
    frame_size: usize,
) -> LoweredProgram {
    lower::lower_to_lowered_program(semantic_ops, hot_locals, config, frame_size)
}
