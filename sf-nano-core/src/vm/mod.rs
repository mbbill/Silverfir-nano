//! The VM: a shared substrate with one sibling engine per execution strategy.
//!
//! The substrate is everything both engines need and neither owns — the
//! `Store`, instances, entities, the value model, the reference and GC heaps,
//! constant-expression evaluation. It makes no assumption about how a
//! function body actually runs.
//!
//! The engines sit beside each other on top of it:
//! - [`jit`] compiles each function to native code before it runs, through
//!   its own four-stage pipeline.
//! - [`interpreter`] predecodes each function into a threaded instruction
//!   stream and runs it through handlers generated at build time.
//!
//! Each is gated on its own feature and at least one must be present (the
//! crate root enforces it). Neither is privileged in the layout: the
//! substrate does not reach into either engine, and the engines do not reach
//! into each other.
//!
//! [`engine`] is how an embedder says which one to use. Its variants are
//! gated on the same features, so in a single-engine build the choice is a
//! zero-sized type and the switch disappears.

pub(crate) mod engine;
pub(crate) mod entities;
pub(crate) mod exn_heap;
pub(crate) mod expr_eval;
pub(crate) mod gc_heap;
pub(crate) mod gc_type_check;
pub(crate) mod imports;
pub(crate) mod instance;
pub(crate) mod result_buffer;
pub(crate) mod store;
pub(crate) mod tag;
pub(crate) mod value;
pub(crate) mod value_encoding;

// --- Execution engines ---

#[cfg(sf_interp)]
pub(crate) mod interpreter;
#[cfg(sf_jit)]
pub(crate) mod jit;
