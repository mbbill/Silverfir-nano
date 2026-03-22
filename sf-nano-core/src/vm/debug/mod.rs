//! Shared debug infrastructure.
//!
//! This layer should stay optional and off the release path by default.

pub mod dump_layout;
pub mod function_trace;
#[cfg(feature = "micro-jit")]
pub mod ir_dump;
pub mod trace_compare;
