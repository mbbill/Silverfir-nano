//! Semantic Wasm frontend.
//!
//! This layer decodes Wasm bytecode into the Semantic IR (SIR), which is
//! the contract consumed by `middle/`.
//!
//! - `sir/` contains the SIR definitions (the contract)
//! - The remaining files contain the processing logic (decode, inline, etc.)

pub mod sir;

// Re-export SIR submodules at the wasm level for compatibility.
pub use sir::common;
pub use sir::primitive_op;
pub use sir::semantic_ir;

pub mod context;
pub mod control;
pub mod decode;
pub mod inline;
