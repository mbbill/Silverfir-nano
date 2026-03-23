//! SSA-IR contract definitions.
//!
//! The SSA-IR boundary preserves the engine's prepared stack-window contract:
//! - canonical locals and deep stack values live in frame slots
//! - only a bounded transient live set stays live as SSA values
//! - explicit slot traffic makes operand-slot publication visible to the
//!   backend

pub(crate) mod ir;
pub(crate) mod leaf;
pub(crate) mod target;
pub(crate) mod validate;
