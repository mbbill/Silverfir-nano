//! Native backend: runtime fusion via micro-assembly.
//!
//! ARM64-only for now. This module is the current home of the former
//! micro-JIT backend after being split out from `interp/fast`.
//! See `docs/NATIVE_BACKEND_ROADMAP.md` for the architectural direction.

use crate::vm::entities::ModuleInst;
use crate::vm::compile::lowered_ir::IrOp;
use crate::vm::interp::fast::builder::backend::ResolvedInst;

mod code_buf;
mod group_meta;
mod debug_map;
mod samply_jitdump;
mod arm64;

pub(crate) use code_buf::CodeBuffer;
pub(crate) use arm64::{resolve_native, resolve_native_with_context};
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
    module: &ModuleInst,
    hot_local_mask: [bool; 3],
    func_idx: u32,
) -> Result<alloc::vec::Vec<ResolvedInst>, &'static str> {
    let mut buf = module.native_code_buffer()?;
    Ok(arm64::resolve_native_with_context(
        ir_ops,
        &mut buf,
        hot_local_mask,
        &module.name,
        func_idx,
    ))
}
