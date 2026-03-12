//! Native runtime boundary.
//!
//! This subtree owns:
//! - runtime context ABI
//! - native entry ABI
//! - frame/layout sizing for finalized native programs
//! - direct runtime execution flow

pub mod context;
pub mod entry;
pub mod layout;
mod run;

pub use run::{eval, run_function, NativeRuntime};
