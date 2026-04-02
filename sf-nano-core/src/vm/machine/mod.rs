//! Machine layer: SSA-IR → MachineIR lowering and transforms.
//!
//! This layer sits between the middle layer (`middle/`) and the architecture
//! backends (`arch/`). It owns:
//! - `machine_ir/` — the MachineIR contract definitions consumed by `arch/`
//! - `lower_*` modules — the lowering passes that transform SSA-IR into MachineIR
//! - MachineIR transforms (peephole optimization, validation)

mod gp32;
mod lower_cached;
mod lower_call;
mod lower_const_pool;
mod lower_context;
mod lower_i64;
mod lower_i64_gp64;
mod lower_inst;
mod lower_leaf_arith;
mod lower_leaf_special;
mod lower_module;
mod lower_regalloc;
mod lower_util;
pub(crate) mod machine_ir;
mod optimize;
pub(crate) mod peephole;
pub(crate) mod validate;

#[cfg(test)]
mod lower_tests;
#[cfg(test)]
mod peephole_tests;
#[cfg(test)]
mod validate_tests;

pub(crate) use lower_module::{lower_module, LowerFunctionInput, LowerModuleInput};
pub(crate) use optimize::optimize_module;
