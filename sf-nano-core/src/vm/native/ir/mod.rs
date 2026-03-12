//! Machine-facing IR and its shared runtime contract.
//!
//! This folder is split into two parts:
//! - `machine/` contains code: registers, addresses, instructions, and CFG.
//! - `runtime/` contains non-code contract metadata shared by real ISA
//!   backends and the debug emulator.

pub mod machine;
pub mod runtime;
