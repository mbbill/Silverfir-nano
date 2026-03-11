//! Placement onto the native VM ABI.
//!
//! This phase will own:
//! - mapping transient values onto `Tos*`, `Hot*`, `Tmp*`, and `fp[...]`
//! - edge parallel-copy construction
//! - hot-local and frame-local alias tracking
//! - call/helper clobber handling
//!
//! ISA backends should consume the placed result mechanically.

use crate::vm::{backend::BackendConfig, lir::ir::LirProgram};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct PlacementPlan {
    pub tos_width: u8,
    pub hot_local_width: u8,
}

#[inline]
pub(super) fn placement_plan(_lir: &LirProgram, backend_config: BackendConfig) -> PlacementPlan {
    PlacementPlan {
        tos_width: backend_config.tos_register_count,
        hot_local_width: backend_config.hot_local_count,
    }
}
