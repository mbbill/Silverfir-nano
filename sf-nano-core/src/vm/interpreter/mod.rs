//! Interpreter v2: the folded stack machine.
//!
//! Design of record: `mcts_mem/silverfir/interpreter/`. Quantitative
//! basis: `tools/foldsim` v4 over the `benchmarks/wasi` corpus, plus the
//! measured record in the same subtree.
//!
//! Architecture split: what is SHARED is the module layer — parsing and
//! decode (`module`, `op_decoder`), the value-type model — and the WASI
//! host. Runtime state is not: the `Store`, the entity model and the
//! `jit/runtime/` layer belong to the JIT, and this module imports
//! nothing from `vm/` outside itself. The compile-and-execute tier
//! is deliberately independent: this module never touches `middle/`,
//! `machine/`, or `arch/`, so interpreter work can never break JIT builds.
//!
//! Pipeline:
//! - `predecode` folds wasm bytecode into fixed 32-byte instruction cells.
//!   Routing opcodes (`local.get/set/tee`, consts) fold into the operand
//!   and destination fields of semantic instructions and never dispatch.
//! - `layout` defines the handler variant space — which operand residency
//!   classes exist and where each combination's handler lives.
//! - `engine` links a predecoded function into dispatch cells pointing at
//!   handlers that were generated at BUILD time (`interp_gen/`, driven by
//!   `build.rs`) and live in this binary's `.text`. No executable memory
//!   is allocated or mapped at run time.
//! - `exec` drives the chain and provides its slow path: one shared
//!   single-instruction executor covering every op without a native
//!   handler, plus host calls, traps with messages, and the activation
//!   boundary.

#[cfg(sf_interp_engine)]
mod engine;
mod exec;
// Only the slow path calls these, and the slow path exists only where a
// dispatch engine does.
#[cfg(sf_interp_engine)]
mod fmath;
mod instr;
// The variant layout describes the generated handler set, so it exists
// only where one was generated. The build script compiles the same file
// independently, via `#[path]`.
#[cfg(sf_interp_engine)]
mod layout;
mod predecode;

// `InterpInstance` is the engine's public face. The predecoded
// representation behind it -- instructions, the opcode enum, operand
// flags -- stays inside the engine: it is how a function is stored, not
// an interface anything outside builds against.
pub use exec::InterpInstance;
