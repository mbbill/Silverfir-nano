//! Native compilation pipeline above architecture-specific code generation.
//!
//! This subtree owns:
//! - frontend bundle construction
//! - LIR-to-native precompile orchestration
//! - native entry resolution metadata

pub mod build;
pub mod precompile;
pub mod resolve;
