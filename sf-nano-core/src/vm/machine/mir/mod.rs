//! Machine-facing IR definitions.
//!
//! This is the contract between the machine lowering layer (`machine/`) and
//! the architecture backends (`arch/`). It defines the target shape:
//! registers, values, addresses, instructions, blocks, and module containers.
//!
//! This module contains ONLY definitions — no transforms, no optimization,
//! no validation logic.

mod cfg;
mod contract;
mod inst;
mod module;
mod regs;
mod types;

#[cfg(test)]
mod tests;

pub use cfg::*;
pub use contract::*;
pub use inst::*;
pub use module::*;
pub use regs::*;
pub use types::*;
