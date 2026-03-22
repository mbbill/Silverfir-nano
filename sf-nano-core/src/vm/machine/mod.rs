//! Machine layer: LIR → MIR lowering and transforms.
//!
//! This layer sits between the middle layer (`middle/`) and the architecture
//! backends (`arch/`). It owns:
//! - `mir/` — the MachineIR contract definitions consumed by `arch/`
//! - `lower_*` modules — the lowering passes that transform LIR into MachineIR
//! - MachineIR transforms (peephole optimization, validation)

pub(crate) mod ir_dump;
mod lower_boundary;
mod lower_context;
mod lower_module;
mod lower_ops;
mod lower_regfile;
mod lower_sidecar;
mod lower_util;
pub(crate) mod machine_ir;
pub(crate) mod peephole;
pub(crate) mod validate;

#[cfg(test)]
mod lower_tests;
#[cfg(test)]
mod peephole_tests;
#[cfg(test)]
mod validate_tests;

pub(crate) use lower_module::{lower_module, LowerFunctionInput, LowerModuleInput};
