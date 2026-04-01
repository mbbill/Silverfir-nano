//! Machine-facing IR definitions.
//!
//! This is the contract between the machine lowering layer (`machine/`) and
//! the architecture backends (`arch/`). It defines the target shape:
//! registers, values, addresses, instructions, blocks, module containers, and
//! the static ABI/layout metadata backends need in order to lower them.
//!
//! This module contains ONLY definitions — no transforms, no optimization,
//! no validation logic.

mod abi;
mod cfg;
mod inst;
mod module;
mod regs;
mod types;

pub(crate) use abi::*;
pub(crate) use cfg::*;
pub(crate) use inst::*;
pub(crate) use module::*;
pub(crate) use regs::*;
pub(crate) use types::*;
