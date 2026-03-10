//! Backend-facing LIR.
//!
//! The live `lir/` boundary is CFG + block params + SSA values.

pub mod dump;
pub mod ir;
pub mod leaf;
pub mod lower;
pub mod reg;
pub mod slot;
pub mod target;
