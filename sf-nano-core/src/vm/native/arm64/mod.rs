pub(crate) mod reg;
pub(crate) mod arm64_enc;
pub(crate) mod emit;
mod codegen;
mod op_meta;
mod semantics;
mod group;

pub(crate) use group::{resolve_native, resolve_native_with_context};
pub(crate) use codegen::{EntryPatchSites, depth_variant, tos_reg};
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
