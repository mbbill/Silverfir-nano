//! Native backend: runtime fusion via micro-assembly.
//!
//! ARM64-only for now. This module is the current home of the former
//! micro-JIT backend after being split out from `interp/fast`.
//! See `docs/NATIVE_BACKEND_ROADMAP.md` for the architectural direction.

pub mod reg;
pub mod arm64_enc;
pub mod code_buf;
pub mod emit;
pub mod codegen;
pub mod op_meta;
pub mod semantics;
pub mod group;
mod group_meta;
mod debug_map;
mod samply_jitdump;
