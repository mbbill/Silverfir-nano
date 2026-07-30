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

// Interpreter slots are always backed by `u64`, but references inside them
// use the target GP wire width consumed by the generated dispatch chain.
const SLOT_GP_UNIT_BYTES: u8 = core::mem::size_of::<usize>() as u8;

const EXTERNAL_FUNCREF_HOST_REQUIRED: &str =
    "interp: external function reference calls require a FuncRefHost hook";

// `InterpInstance` is the engine's public face. The predecoded
// representation behind it -- instructions, the opcode enum, operand
// flags -- stays inside the engine: it is how a function is stored, not
// an interface anything outside builds against.
pub use exec::{FuncRefHost, InterpInstance};
// The boundary converts host values through the same shared slot encoding
// that the executor imports from `vm::value`.
pub(crate) use exec::{
    raw_to_value_for_interp, value_to_raw_for_interp, InterpInstanceAccess, InterpInstanceLease,
};
