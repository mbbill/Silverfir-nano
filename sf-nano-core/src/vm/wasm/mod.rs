//! Semantic Wasm frontend.
//!
//! This layer may think in terms of Wasm semantics and structured control flow.
//! It must not commit to backend-local placement such as:
//! - hot locals
//! - spill/fill insertion
//! - grouping
//! - frame-slot layout
//! - rotating-window codegen contracts

pub mod common;
pub mod context;
pub mod control;
pub mod primitive_op;
pub mod decode;
pub mod semantic_ir;
