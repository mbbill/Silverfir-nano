//! Joint transient/cache planning.
//!
//! This module owns policy, not mutable rewrite state.

pub(super) mod canonical;
pub(super) mod naive;
pub(super) mod scan;
pub(super) mod types;
pub(super) mod validate;
