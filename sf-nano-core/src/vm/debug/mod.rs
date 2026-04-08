//! Shared debug infrastructure.
//!
//! This layer should stay optional and off the release path by default.

pub(crate) mod function_trace;
#[cfg(sf_jit)]
pub(crate) mod ir_dump;
