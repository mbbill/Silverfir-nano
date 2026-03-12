//! Shared runtime contract for machine IR execution.
//!
//! Machine IR itself describes only code. This module describes the shared
//! runtime-side contract that both real ISA backends and the debug emulator
//! execute against.

mod call;
mod contract;
mod externs;

pub use call::*;
pub use contract::*;
pub use externs::*;
