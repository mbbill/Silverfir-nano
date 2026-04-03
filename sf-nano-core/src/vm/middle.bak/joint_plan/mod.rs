//! Joint transient + local-cache planning interface.
//!
//! The planning algorithm should be replaceable. Lowering consumes these
//! explicit per-op plans rather than reaching back into planner internals.

use alloc::vec::Vec;

use crate::{
    value_type::ValueType,
    vm::{
        middle::frame::{FrameSlot, FrameSpan},
        wasm::semantic_ir::SemanticOp,
    },
};

use super::state::EntryState;

pub(super) mod boundaries;
pub(super) mod exact_stack;
pub(super) mod naive;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum PrepAction {
    Spill(FrameSpan),
    /// Fill from frame slots, with the value type for each reloaded entry.
    Fill(FrameSpan, Vec<ValueType>),
    DropCache(FrameSlot),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum PreparedLocalAccess {
    #[default]
    Slot,
    Cache,
}

#[derive(Clone, Debug)]
pub(super) struct OpPlan<'a> {
    pub(super) semantic: &'a SemanticOp,
    pub(super) prefix: Vec<PrepAction>,
    pub(super) local_access: PreparedLocalAccess,
}

#[derive(Clone, Debug)]
pub(super) struct LinearPlan<'a> {
    pub(super) ops: Vec<OpPlan<'a>>,
    pub(super) entry_states: Vec<EntryState>,
}
