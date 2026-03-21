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

pub(crate) use cfg::*;
pub(crate) use contract::*;
pub(crate) use inst::*;
pub(crate) use module::*;
pub(crate) use regs::*;
pub(crate) use types::*;
