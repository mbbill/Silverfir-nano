//! The optimizing JIT: everything between Wasm bytecode and native code.
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
//! `debug/` is the IR/jitdump tooling for the same pipeline. What is NOT here
//! is the substrate both engines share -- module parsing, `Store`, instances,
//! entities, values -- which stays in [`super`].

pub(crate) mod arch;
pub(crate) mod build;
pub(crate) mod debug;
pub(crate) mod machine;
pub(crate) mod middle;
pub(crate) mod template;
pub(crate) mod wasm;
