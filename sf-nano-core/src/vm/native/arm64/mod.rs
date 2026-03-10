//! ARM64 implementation of the standalone native backend.
//!
//! This subtree should contain ISA-specific codegen only. It should not own
//! Wasm semantics, grouping policy, or descriptor-era contracts.

pub mod codegen;
pub mod emit;
pub mod enc;
pub mod group;
pub mod op_meta;
pub mod reg;
pub mod semantics;
