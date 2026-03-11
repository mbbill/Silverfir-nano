//! Native helper ABI and cold-path support.
//!
//! This subtree owns the uniform helper ABI, typed metadata, and the Rust
//! implementations for cold/complex helper behavior.

pub mod bridge;
mod impls;
pub mod meta;

pub use impls::{DEFAULT_HELPER, helper_for_kind, unimplemented_helper};
