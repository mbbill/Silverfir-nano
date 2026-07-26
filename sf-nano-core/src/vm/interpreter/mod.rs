//! Interpreter v2: the folded stack machine.
//!
//! Design of record: `mcts_mem/silverfir/interpreter/`. Quantitative
//! basis: `tools/foldsim` v4 over the `benchmarks/wasi` corpus, plus the
//! measured record in the same subtree.
//!
//! Architecture split: the two engines differ only in how code is RUN.
//! Shared are the module layer (parsing, decode, the value-type model),
//! validation, the entity model (`vm::entities` — a memory here is the
//! same `MemInst` the JIT exports, which is what lets one instance import
//! another's), imports (`vm::imports`), values, config, and the WASI host.
//!
//! Not shared is code generation: this module never touches `middle/`,
//! `machine/`, or `arch/`, and the JIT's `Store` and `jit/runtime/` layer
//! stay the JIT's, so interpreter work can never break JIT builds.
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

mod engine;
mod exec;
mod fmath;
mod instr;
// The variant layout describes the generated handler set. The build script
// compiles this same file independently, via `#[path]`, so the generator and
// the linker agree on the space by construction.
mod layout;
mod predecode;

// `InterpInstance` is the engine's public face. The predecoded
// representation behind it -- instructions, the opcode enum, operand
// flags -- stays inside the engine: it is how a function is stored, not
// an interface anything outside builds against.
pub use exec::InterpInstance;
