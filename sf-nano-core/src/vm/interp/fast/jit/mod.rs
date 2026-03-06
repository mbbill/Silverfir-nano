//! Micro-JIT: runtime fusion via micro-assembly.
//!
//! ARM64-only. Replaces static fusion on JIT-capable platforms.
//! See docs/MICRO_JIT.md for architecture details.

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
