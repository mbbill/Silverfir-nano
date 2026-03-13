//! Frontend preparation between decoded Wasm semantics and prepared LIR.
//!
//! Responsibilities:
//! - canonical frame layout
//! - local-cache preference analysis
//! - semantic-to-LIR preparation with explicit spill/fill

pub mod config;
pub mod frame;
pub mod prepare;

mod local_cache;

pub use local_cache::analyze_local_cache_prefs;
pub use prepare::{prepare_function, PrepareInput, PreparedFunction};
