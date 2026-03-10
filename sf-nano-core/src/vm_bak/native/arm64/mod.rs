pub(crate) mod reg;
pub(crate) mod enc;
pub(crate) mod emit;
mod codegen;
pub(crate) mod op_meta;
mod semantics;
mod group;

pub(crate) use group::{resolve_native, resolve_native_with_context, resolve_native_with_plan_context};
pub(crate) use codegen::{EntryPatchSites, current_variant_from_window, tos_reg};
pub use group::{
    JitStatsSnapshot,
    NativeStatsSnapshot,
    jit_capacity_skips,
    jit_stats,
    jit_stats_snapshot,
    native_capacity_skips,
    native_stats,
    native_stats_snapshot,
};
