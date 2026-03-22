//! Shared debug infrastructure.
//!
//! This layer should stay optional and off the release path by default.

pub(crate) mod dump_layout;
pub(crate) mod function_trace;
#[cfg(feature = "micro-jit")]
pub(crate) mod ir_dump;
pub(crate) mod trace_compare;
