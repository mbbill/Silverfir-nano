//! Semantic Wasm frontend.
//!
//! This layer decodes Wasm bytecode into the Semantic IR (SIR), which is
//! the contract consumed by `middle/`.
//!
//! - `sir/` contains the SIR definitions (the contract)
//! - The remaining files contain the processing logic (decode, inline, etc.)

pub(crate) mod sir;

// Re-export SIR submodules at the wasm level for compatibility.
pub(crate) use sir::common;
pub(crate) use sir::primitive_op;
pub(crate) use sir::semantic_ir;

pub(crate) mod context;
pub(crate) mod control;
pub(crate) mod decode;
pub(crate) mod inline;
