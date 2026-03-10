//! Native backend: runtime fusion via micro-assembly.
//!
//! ARM64-only for now. This module is the current home of the former
//! micro-JIT backend after being split out from `interp/fast`.

use crate::vm::{entities::ModuleInst, plan::GroupPlan};
use crate::vm::lir::IrOp;
use alloc::string::{String, ToString};

mod code_buf;
mod bridge;
pub(crate) mod group_policy;
mod canonicalize;
pub mod code;
pub mod build;
pub mod context;
mod group_meta;
mod map;
#[cfg(feature = "native-dump")]
mod dump;
mod finalizer;
pub mod entry;
pub mod precompile;
pub mod resolved;
pub mod runtime;
mod jitdump;
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
    group_plan: &GroupPlan,
    semantic_to_lir: &[usize],
    _operand_base: usize,
    _tos_register_count: usize,
    module: &ModuleInst,
    hot_local_mask: [bool; 3],
    func_idx: u32,
) -> Result<alloc::vec::Vec<resolved::ResolvedNativeInst>, String> {
    let planned_ranges: alloc::vec::Vec<core::ops::Range<usize>> = group_plan
        .ranges
        .iter()
        .map(|range| {
            let start = semantic_to_lir[range.semantic_start];
            let end = semantic_to_lir[range.semantic_end - 1] + 1;
            start..end
        })
        .collect();
    let mut buf = module
        .native_code_buffer()
        .map_err(|err| err.to_string())?;
    arm64::resolve_native_with_plan_context(
        ir_ops,
        &planned_ranges,
        &mut buf,
        hot_local_mask,
        &module.name,
        func_idx,
        _operand_base,
        _tos_register_count,
    )
}
