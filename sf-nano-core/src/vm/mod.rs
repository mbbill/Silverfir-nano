//! VM architecture.
//!
//! Pipeline layers:
//! - `wasm/` decodes Wasm bytecode into Semantic IR (SIR)
//! - `middle/` prepares SIR into SSA-IR with explicit spill/fill
//! - `machine/` lowers SSA-IR into MachineIR (MIR)
//! - `arch/` compiles MIR into native code
//! - `runtime/` provides execution infrastructure

#[cfg(sf_jit)]
pub(crate) mod arch;
pub(crate) mod backend;
#[cfg(sf_jit)]
pub(crate) mod build;
pub(crate) mod debug;
pub(crate) mod entities;
pub(crate) mod exn_heap;
pub(crate) mod expr_eval;
pub(crate) mod gc_heap;
pub(crate) mod gc_type_check;
pub(crate) mod instance;
#[cfg(sf_interp)]
pub(crate) mod interpreter;
#[cfg(sf_jit)]
pub(crate) mod machine;
#[cfg(sf_jit)]
pub(crate) mod middle;
pub(crate) mod result_buffer;
pub(crate) mod runtime;
pub(crate) mod store;
pub(crate) mod tag;
#[cfg(sf_jit)]
pub(crate) mod template;
pub(crate) mod value;
pub(crate) mod value_encoding;
#[cfg(sf_jit)]
pub(crate) mod wasm;
