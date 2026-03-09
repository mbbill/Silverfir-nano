//! Native backend: runtime fusion via micro-assembly.
//!
//! ARM64-only for now. This module is the current home of the former
//! micro-JIT backend after being split out from `interp/fast`.
//! See `docs/NATIVE_BACKEND_ROADMAP.md` for the architectural direction.

use crate::vm::entities::ModuleInst;
use crate::vm::lowered::IrOp;
use alloc::string::{String, ToString};

mod code_buf;
mod bridge;
pub mod code;
pub mod compiler;
pub mod context;
mod group_meta;
mod debug_map;
mod finalizer;
pub mod instruction;
pub mod precompile;
pub mod resolved;
pub mod runtime;
#[cfg(feature = "lockstep-debug")]
pub mod lockstep;
mod samply_jitdump;
mod arm64;

pub(crate) use code_buf::CodeBuffer;
pub(crate) use arm64::{resolve_native, resolve_native_with_context};
#[cfg(feature = "lockstep-debug")]
pub(crate) use arm64::resolve_native_with_context_and_checkpoints;
pub use arm64::{
    JitStatsSnapshot,
    NativeStatsSnapshot,
    jit_capacity_skips,
    jit_stats,
    jit_stats_snapshot,
    native_capacity_skips,
    native_stats,
    native_stats_snapshot,
};

pub fn resolve_backend(
    ir_ops: &[IrOp],
    _operand_base: usize,
    _tos_register_count: usize,
    module: &ModuleInst,
    hot_local_mask: [bool; 3],
    func_idx: u32,
) -> Result<alloc::vec::Vec<resolved::ResolvedNativeInst>, String> {
    let mut buf = module
        .native_code_buffer()
        .map_err(|err| err.to_string())?;
    arm64::resolve_native_with_context(
        ir_ops,
        &mut buf,
        hot_local_mask,
        &module.name,
        func_idx,
    )
}
