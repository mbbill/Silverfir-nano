//! Modular IR builder for the fast interpreter.
//!
//! Architecture:
//! - `context`: Function metadata and type resolution
//! - `stack`: Compile-time stack tracking
//! - `emitter`: Instruction emission
//! - `finalizer`: Compact, patch, and build final instructions
//! - `fusion`: Instruction fusion pattern matching
//! - `dispatch`: Opcode-to-handler dispatch

mod context;
mod dispatch;
mod emitter;
mod finalizer;
pub mod hot_local;
pub mod ir;
// Auto-generated from handlers.toml [[fused]] entries.
// See `build/fast_interp/gen_fusion.rs` for the generator.
#[cfg(feature = "fusion")]
include!(concat!(env!("OUT_DIR"), "/fast_interp/fast_fusion.rs"));
#[cfg(feature = "fusion")]
include!(concat!(env!("OUT_DIR"), "/fast_interp/fast_fusion_emit.rs"));
mod stack;
mod temp_inst;

pub use context::CompileContext;
pub use emitter::CodeEmitter;
pub use stack::{BlockKind, ControlFrame, StackTracker, HOT_LOCAL_COUNT};
pub use temp_inst::{Handler, TempInst};

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

/// Baseline IR builder: one fast instruction per Wasm opcode.
///
/// This struct provides the same interface as the old monolithic FastOpBuilder
/// but uses the modular builder internally.
#[derive(Default)]
pub struct FastOpBuilder;

impl FastOpBuilder {
    pub fn new() -> Self {
        Self
    }

    /// Build the fast IR for a function, threading control-flow and packing branch fixups.
    ///
    /// On success, attaches IR to `function` via `set_fast_ir` and returns the entry pointer.
    ///
    /// `types` is the module's type section, needed for resolving TypeIndex block types
    /// (multi-value blocks). If None, TypeIndex blocks will be treated as (0, 0) arity.
    ///
    /// `store` and `module` are used to look up function signatures for CALL stack tracking.
    pub fn build_for_function(
        &mut self,
        function: &FunctionSpec,
        types: Option<&[Rc<FunctionType>]>,
        store: &Store,
        module: &ModuleInst,
    ) -> Result<*mut Instruction, WasmError> {
        build_for_function(function, types, store, module)
    }
}

/// Build fast IR for a function.
///
/// This is the main entry point. It orchestrates:
/// 1. Decode Wasm opcodes
/// 2. Dispatch each opcode to update slots and emit instructions
/// 3. Finalize: compact, patch, fixup, build
///
/// All internal calls emit `call_internal`. During module precompilation,
/// same-module calls are optimized to `call_local` via in-place patching.
pub fn build_for_function(
    function: &FunctionSpec,
    types: Option<&[Rc<FunctionType>]>,
    store: &Store,
    module: &ModuleInst,
) -> Result<*mut Instruction, WasmError> {
    // Setup
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
    let mut emitter = CodeEmitter::new();

    // Emit combined init_locals to swap+fill all 3 hot locals in one dispatch.
    // Call/return handlers unconditionally spill/fill via fp[0], fp[1], fp[2].
    emitter.emit_init_locals(
        hot_locals[0].unwrap_or(0),
        hot_locals[1].unwrap_or(1),
        hot_locals[2].unwrap_or(2),
    );

    // Decode and dispatch (includes fusion at decode time)
    dispatch::decode_and_dispatch(code, &ctx, &mut stack, &mut emitter)?;

    // Take temps for optional JIT pass + finalize
    let mut temps = emitter.take_temps();

    // JIT group compilation (ARM64 only, micro-jit feature)
    #[cfg(feature = "micro-jit")]
    let jit_buf = {
        use super::jit::code_buf::CodeBuffer;
        use super::jit::group;

        let hot_mask = [
            hot_locals[0].is_some(),
            hot_locals[1].is_some(),
            hot_locals[2].is_some(),
        ];

        match CodeBuffer::new() {
            Ok(mut buf) => {
                group::compile_jit_groups(&mut temps, &mut buf, hot_mask);
                Some(buf)
            }
            Err(_) => None,
        }
    };

    // Finalize (br_table data is now stored inline in the instruction stream)
    let code_box = finalizer::finalize(temps, &mut stack);

    // Store in function spec
    #[cfg(not(feature = "micro-jit"))]
    {
        use crate::vm::interp::fast::fast_code::{FastCode, create_fast_code};
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
