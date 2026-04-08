//! Shared debug infrastructure.
//!
//! This layer should stay optional and off the release path by default.

#[cfg(sf_call_trace)]
pub(crate) mod function_trace;
#[cfg(sf_ir_dump)]
pub(crate) mod ir_dump;
