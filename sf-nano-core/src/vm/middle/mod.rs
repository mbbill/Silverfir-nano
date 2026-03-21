//! Middle layer: SIR → LIR preparation.
//!
//! This layer sits between the Wasm semantic frontend (`wasm/`) and the
//! machine lowering layer (`machine/`). It owns:
//! - `lir/` — the LIR contract definitions consumed by `machine/`
//! - preparation passes that transform semantic IR into prepared LIR
//!   with explicit spill/fill and canonical frame layout

pub mod config;
pub mod frame;
pub mod lir;
pub mod prepare;

mod local_cache;

pub use local_cache::analyze_local_cache_prefs;
pub use prepare::{prepare_function, PrepareInput, PreparedFunction};
