//! Modular IR builder for the fast interpreter.
//!
//! Pipeline:
//!   Wasm bytecode → ir_lower → Vec<IrOp> → backend (JIT or fusion) → Vec<ResolvedInst> → finalizer → Box<[Instruction]>
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
#[cfg(feature = "fusion")]
mod fusion;
pub mod hot_local;
pub mod ir;
pub mod ir_lower;
pub mod ir_resolve;
mod stack;

pub use context::CompileContext;
pub use stack::{BlockKind, ControlFrame, StackTracker, HOT_LOCAL_COUNT};

use crate::{
    module::entities::FunctionSpec,
    vm::{
        entities::ModuleInst,
        interp::fast::instruction::Instruction,
        store::Store,
    },
    error::WasmError,
};

use alloc::rc::Rc;
use crate::module::type_defs::FunctionType;

/// Build fast IR via the unified IR pipeline.
///
/// Lowers Wasm bytecode to neutral IR, resolves via backend (JIT or base 1:1),
/// then finalizes into the interpreter's `Box<[Instruction]>`.
pub fn build_for_function(
    function: &FunctionSpec,
    types: Option<&[Rc<FunctionType>]>,
    store: &Store,
    module: &ModuleInst,
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

    // Backend: resolve IR to handlers
    #[cfg(feature = "micro-jit")]
    let (resolved, jit_buf) = {
        use super::jit::code_buf::CodeBuffer;
        use super::jit::group;

        let hot_mask = [
            hot_locals[0].is_some(),
            hot_locals[1].is_some(),
            hot_locals[2].is_some(),
        ];

        match CodeBuffer::new() {
            Ok(mut buf) => {
                let resolved = group::resolve_jit(&ir_ops, &mut buf, hot_mask);
                (resolved, Some(buf))
            }
            Err(_) => (backend::resolve_base(&ir_ops), None),
        }
    };

    #[cfg(all(not(feature = "micro-jit"), feature = "fusion"))]
    let resolved = fusion::resolve_fusion(&ir_ops);

    #[cfg(all(not(feature = "micro-jit"), not(feature = "fusion")))]
    let resolved = backend::resolve_base(&ir_ops);

    // Finalize: Vec<ResolvedInst> → Box<[Instruction]>
    let code_box = finalizer_ir::finalize(resolved, &mut stack);

    // Store in function spec
    #[cfg(not(feature = "micro-jit"))]
    {
        use crate::vm::interp::fast::fast_code::create_fast_code;
        let (fast_code, fast_cache) = create_fast_code(code_box, params_count, locals_count, results_count);
        let entry = fast_cache.entry();
        function.set_fast_code(fast_code, fast_cache);
        Ok(entry)
    }
    #[cfg(feature = "micro-jit")]
    {
        use crate::vm::interp::fast::fast_code::create_fast_code_with_jit;
        let (fast_code, fast_cache) = create_fast_code_with_jit(code_box, jit_buf, params_count, locals_count, results_count);
        let entry = fast_cache.entry();
        function.set_fast_code(fast_code, fast_cache);
        Ok(entry)
    }
}
