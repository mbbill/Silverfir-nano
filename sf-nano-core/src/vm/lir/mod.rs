//! Prepared backend-facing LIR.
//!
//! The live `lir/` boundary preserves the engine's prepared stack-window
//! contract:
//! - canonical locals and deep stack values live in frame slots
//! - only the bounded TOS window stays live as SSA values
//! - explicit spill/fill ops make operand-slot publication visible to the
//!   backend

pub mod dump;
pub mod ir;
pub mod leaf;
pub mod lower;
pub mod runtime;
pub mod slot;
pub mod target;
