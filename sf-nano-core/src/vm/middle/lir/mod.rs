//! LIR contract definitions.
//!
//! The LIR boundary preserves the engine's prepared stack-window contract:
//! - canonical locals and deep stack values live in frame slots
//! - only a bounded transient live set stays live as SSA values
//! - explicit slot traffic makes operand-slot publication visible to the
//!   backend

pub mod ir;
pub mod leaf;
pub mod target;
pub mod validate;
