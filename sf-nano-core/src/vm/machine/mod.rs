//! Machine layer: SSA-IR → MachineIR lowering and transforms.
//!
//! This layer sits between the middle layer (`middle/`) and the architecture
//! backends (`arch/`). It owns:
//! - `machine_ir/` — the MachineIR contract definitions consumed by `arch/`
//! - `lower_*` modules — the lowering passes that transform SSA-IR into MachineIR
//! - MachineIR transforms (peephole optimization, validation)

mod call_abi;
mod gp32;
#[cfg(any(sf_backend_armv7a, sf_backend_thumbm, sf_backend_riscv32))]
pub(crate) mod low32_liveness;
mod lower_cache_layout;
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
#[cfg(sf_has_simd)]
mod lower_simd;
mod lower_util;
pub(crate) mod machine_ir;
mod optimize;
mod ownership;
pub(crate) mod peephole;
pub(crate) mod validate;

#[cfg(test)]
mod lower_tests;
#[cfg(test)]
mod peephole_tests;
#[cfg(test)]
mod validate_tests;

pub(crate) use lower_const_pool::ConstPoolBuilder;
pub(crate) use lower_module::LoweredMachineModule;
pub(crate) use lower_module::{
    derive_param_locs_from_types, derive_return_abi, lower_module, lower_single_function,
    LowerFunctionInput, LowerModuleInput,
};
pub(crate) use optimize::{optimize_function, optimize_module};
