//! ISA-specific native backend implementations.
//!
//! This layer lowers target-independent native IR into encoded machine code.
//! It must not own Wasm/LIR semantics or backend-wide optimization policy.

pub mod arm64;
pub mod reference;
