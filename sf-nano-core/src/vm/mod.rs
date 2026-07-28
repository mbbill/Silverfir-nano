//! The VM: a shared substrate with one sibling engine per execution strategy.
//!
//! The substrate is what both engines genuinely share: the entity model
//! ([`entities`]), the public value model ([`value`]), tags, import
//! declarations, instance dispatch — and [`link`], the registry through
//! which separately instantiated modules exchange references. Runtime
//! *storage* is not part of it: each engine owns its own (the JIT's `Store`
//! lives in [`jit`], the interpreter's flat state in [`interpreter`]).
//!
//! The engines sit beside each other on top of it:
//! - [`jit`] compiles each function to native code before it runs, through
//!   its own four-stage pipeline.
//! - [`interpreter`] predecodes each function into a threaded instruction
//!   stream and runs it through handlers generated at build time.
//!
//! Each is gated on its own feature and at least one must be present (the
//! crate root enforces it). Neither is privileged in the layout: the
//! substrate does not reach into either engine except in [`link`], whose
//! registry entries carry engine-minted payloads under that engine's cfg —
//! that is the declared meeting point, and the only one.
//!
//! [`engine`] is how an embedder says which one to use. Its variants are
//! gated on the same features, so in a single-engine build the choice is a
//! zero-sized type and the switch disappears.

pub(crate) mod engine;
pub(crate) mod entities;
pub(crate) mod imports;
pub(crate) mod instance;
pub(crate) mod link;
pub(crate) mod tag;
pub(crate) mod value;

// --- Execution engines ---

#[cfg(sf_interp)]
pub(crate) mod interpreter;
#[cfg(sf_jit)]
pub(crate) mod jit;
