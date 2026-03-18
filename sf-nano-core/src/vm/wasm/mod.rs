//! Semantic Wasm frontend.
//!
//! This layer may think in terms of Wasm semantics and structured control flow.
//! It produces the decoded semantic function model used by `plan/`.
//! It must not commit to backend-local preparation such as:
//! - hot locals
//! - spill/fill insertion
//! - prepared block shaping
//! - frame-slot layout
//! - prepared transient-window contracts

pub mod common;
pub mod context;
pub mod control;
pub mod decode;
pub mod inline;
pub mod primitive_op;
pub mod semantic_ir;
