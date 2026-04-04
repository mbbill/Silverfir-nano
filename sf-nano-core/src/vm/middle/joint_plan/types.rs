//! Core joint-plan data types.

use alloc::vec::Vec;

use crate::value_type::ValueType;
use crate::vm::middle::frame::{FrameSlot, FrameSpan};

/// Straight-line local access choice for one semantic local op.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum PreparedLocalAccess {
    #[default]
    Slot,
    Cache,
}

/// Prefix actions chosen before one semantic op executes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PrepAction {
    Spill(FrameSpan),
    Fill(FrameSpan, Vec<ValueType>),
    DropCache(FrameSlot),
}

/// Exact transient entry state at one semantic program point.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct EntryState {
    pub stack_height: u16,
    pub spill_depth: u16,
    pub live_types: Vec<ValueType>,
}

impl EntryState {}

/// Planned straight-line behavior for one semantic op.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct OpPlan {
    pub before: EntryState,
    pub drop_cached_locals: Vec<FrameSlot>,
    pub local_access: PreparedLocalAccess,
}

/// Planned block boundary state used by the planner facade.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct BlockPlan {
    pub entry: EntryState,
    pub entry_cached_locals: Vec<FrameSlot>,
    pub exit_cached_locals: Vec<FrameSlot>,
}

/// Whole-function joint plan.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct FunctionPlan {
    pub gp_unit_bytes: u8,
    pub gp_dynamic_budget: u8,
    pub fp_dynamic_budget: u8,
    pub op_plans: Vec<OpPlan>,
    pub entry_states: Vec<EntryState>,
    pub blocks: Vec<BlockPlan>,
}
