//! ARM64 lowering and emission for the native backend.
//!
//! This subtree owns only ARM64-specific register naming, encoding, and final
//! emission glue on top of target-independent native IR.

pub mod emit;
pub mod enc;
pub mod entry;
pub mod lower;
pub mod reg;
