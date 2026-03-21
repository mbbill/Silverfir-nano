//! Semantic IR definitions.
//!
//! This is the contract between the Wasm frontend (`wasm/`) and the middle
//! layer (`middle/`). It defines the decoded semantic function model:
//! primitive ops, structured control flow, locals, calls, and branch targets.
//!
//! This module contains ONLY definitions — no decoding, no inlining, no
//! transformation logic.

pub(crate) mod common;
pub(crate) mod primitive_op;
pub(crate) mod semantic_ir;
