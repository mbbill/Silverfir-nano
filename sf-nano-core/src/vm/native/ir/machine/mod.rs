//! New machine-facing IR.
//!
//! This module defines the target shape that `NativeIR` should converge to.
//!
//! Layout:
//! - `types.rs` holds generic registers, values, addresses, widths, and trap
//!   kinds
//! - `inst.rs` holds straight-line machine instructions
//! - `cfg.rs` holds block structure, edges, and terminators
//! - `module.rs` holds per-function and per-module containers plus sidecar
//!   constant records
//! - `validate.rs` holds structural validation only
//! - `tests/` holds validator and shape tests only
//!
//! It intentionally avoids VM-flavored storage concepts such as hot locals,
//! TOS lanes, frame slots, or spills.

mod cfg;
mod inst;
mod legalize;
mod module;
pub mod peephole;
mod regs;
mod types;
mod validate;

#[cfg(test)]
mod tests;

pub use cfg::*;
pub use inst::*;
pub use module::*;
pub use regs::*;
pub use types::*;
