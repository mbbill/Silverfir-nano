//! The optimizing JIT: everything between Wasm bytecode and native code, and
//! the native runtime that executes the result.
//!
//! The whole subtree is gated on `sf_jit` at the [`super`] level, which is
//! the point of it having a subtree at all. "Belongs to the JIT" is now a
//! directory rather than a per-module judgement call, so an interpreter-only
//! build drops the pipeline by construction instead of by remembering to add
//! an attribute.
//!
//! Stages, in order:
//! - `wasm/` decodes Wasm bytecode into Semantic IR (SIR)
//! - `middle/` prepares SIR into SSA-IR with explicit spill/fill
//! - `machine/` lowers SSA-IR into MachineIR (MIR)
//! - `arch/` compiles MIR into native code for the selected backend
//! - `build` drives those stages; `template` is the fast path that skips them
//!
//! `backend` is the register-ABI contract those stages plan against, and
//! `runtime/` is what the emitted code runs inside: executable memory, the
//! native context and frame layout, traps, and the runtime-call boundary.
//! Every module here was already individually `sf_jit`-gated; the directory
//! states it once instead.
//!
//! The engine's runtime *storage* also lives here: `store` (the runtime
//! world native code addresses), `gc_heap`, `expr_eval` (instantiation-time
//! constant evaluation against a store), `gc_type_check`, and
//! `value_encoding` (the raw-slot and 32-bit wire encodings the native
//! boundary marshals through). The interpreter shares none of these; what
//! the engines exchange goes through `crate::vm::link` instead.
//!
//! `debug/` is the IR/jitdump tooling for the same pipeline.
//!
//! Note the word "backend" carries two meanings in this tree, and they are
//! not the same axis: here it is the *ISA* the JIT emits for, while
//! which engine runs a module at all is [`crate::vm::engine::Engine`].

pub(crate) mod arch;
pub(crate) mod backend;
pub(crate) mod build;
pub(crate) mod debug;
pub(crate) mod entities;
pub(crate) mod expr_eval;
pub(crate) mod gc_heap;
pub(crate) mod gc_type_check;
pub(crate) mod instance;
pub(crate) mod machine;
pub(crate) mod middle;
pub(crate) mod runtime;
pub(crate) mod store;
pub(crate) mod template;
pub(crate) mod value_encoding;
pub(crate) mod wasm;
