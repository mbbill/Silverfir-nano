//! Interpreter v2: the folded stack machine.
//!
//! Design of record: `docs/INTERPRETER_V2.md`. Quantitative basis:
//! `tools/foldsim` v4 over the `benchmarks/wasi` corpus.
//!
//! Architecture split: the runtime substrate is SHARED — module parsing and
//! decode (`module`, `op_decoder`), `Store`, instances, entities, and the
//! `runtime/` layer serve both engines. What is deliberately independent
//! during bring-up is the compile-and-execute tier: this module never
//! touches `middle/`, `machine/`, or `arch/`, so interpreter iteration can
//! never break JIT builds. Deeper code reuse is a planned refactor AFTER
//! performance validation, not before.
//!
//! Staging (see the design doc §12):
//! - A1 (this): predecoder — wasm bytecode → fixed-width folded instruction
//!   stream. Routing opcodes (`local.get/set/tee`, consts) fold into the
//!   operand and destination fields of semantic instructions.
//! - A2: safe-Rust dispatch-loop executor, spectest subset.
//! - B: build-time generated handlers under a self-defined register
//!   contract; trace-driven dispatch microbenchmark decides the substrate.

#[cfg(all(sf_jit, sf_backend_arm64))]
mod dispatch_arm64;
mod exec;
mod instr;
mod predecode;

pub use exec::{HostDispatch, InterpInstance};
pub use instr::{Instr, Op, FLAG_A_CONST, FLAG_B_CONST};
pub use predecode::{predecode_function, PredecodedFunction};
