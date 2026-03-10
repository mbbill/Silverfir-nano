//! Backend-facing lower IR.
//!
//! This is the boundary where stack-machine semantics end.
//! LIR may carry:
//! - rotating-window top/offset
//! - explicit `fp[...]` slots
//! - immediates / targets
//!
//! LIR must not carry:
//! - `pre_height`
//! - generic `variant`
//! - spill-depth reasoning
//! - explicit `T0..T3` operands

pub mod dump;
pub mod ir;
pub mod lower;
pub mod reg;
pub mod slot;
pub mod target;
