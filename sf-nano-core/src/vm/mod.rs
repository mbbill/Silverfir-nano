//! VM architecture.
//!
//! Pipeline layers:
//! - `wasm/` decodes Wasm bytecode into Semantic IR (SIR)
//! - `middle/` prepares SIR into LIR with explicit spill/fill
//! - `machine/` lowers LIR into MachineIR (MIR)
//! - `arch/` compiles MIR into native code
//! - `runtime/` provides execution infrastructure

pub mod backend;
#[cfg(feature = "micro-jit")]
pub mod build;
pub mod debug;
pub mod entities;
pub mod expr_eval;
pub mod instance;
#[cfg(feature = "interp")]
pub mod interp;
#[cfg(feature = "micro-jit")]
pub mod machine;
pub mod middle;
#[cfg(feature = "micro-jit")]
pub mod arch;
pub mod raw_value;
pub mod runtime;
pub mod stack;
pub mod store;
pub mod value;
pub mod wasm;
