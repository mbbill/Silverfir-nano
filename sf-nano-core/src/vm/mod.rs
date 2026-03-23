//! VM architecture.
//!
//! Pipeline layers:
//! - `wasm/` decodes Wasm bytecode into Semantic IR (SIR)
//! - `middle/` prepares SIR into LIR with explicit spill/fill
//! - `machine/` lowers LIR into MachineIR (MIR)
//! - `arch/` compiles MIR into native code
//! - `runtime/` provides execution infrastructure

pub(crate) mod backend;
#[cfg(feature = "micro-jit")]
pub(crate) mod build;
pub(crate) mod debug;
pub(crate) mod entities;
pub(crate) mod expr_eval;
pub(crate) mod instance;
#[cfg(feature = "interp")]
pub(crate) mod interp;
#[cfg(feature = "micro-jit")]
pub(crate) mod machine;
pub(crate) mod middle;
#[cfg(feature = "micro-jit")]
pub(crate) mod arch;
pub(crate) mod raw_value;
pub(crate) mod runtime;
pub(crate) mod result_buffer;
pub(crate) mod store;
pub(crate) mod value;
pub(crate) mod wasm;
