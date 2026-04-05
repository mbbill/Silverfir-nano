//! Rewrite-facing planner interface.
//!
//! The planner owns immutable facts and policy. The rewriter owns mutable
//! lowering state. Every planner consultation should flow through these query
//! and decision types so the boundary stays explicit.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::{
    value_type::ValueType,
    vm::middle::{cfg::CfgBlockId, frame::FrameSlot},
};

/// Function-wide setup facts needed by the rewriter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FunctionSetupDecision {
    pub gp_unit_bytes: u8,
    pub gp_dynamic_budget: u8,
    pub fp_dynamic_budget: u8,
}

/// Exact transient stack contract at one rewrite boundary.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TransientContract<'a> {
    pub stack_height: u16,
    pub spill_depth: u16,
    pub live_types: &'a [ValueType],
}

impl TransientContract<'_> {
    #[inline]
    pub(crate) const fn live_value_count(&self) -> u16 {
        self.stack_height.saturating_sub(self.spill_depth)
    }
}

/// Chosen block-open boundary.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BlockOpenDecision<'a> {
    pub transient: TransientContract<'a>,
    pub cached_locals: &'a [FrameSlot],
}

/// Target boundary facts for control-flow canonicalization.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TargetEntryDecision<'a> {
    pub transient: TransientContract<'a>,
}

/// Authoritative planner decision before one semantic op executes.
#[derive(Clone, Debug)]
pub(crate) struct BeforeOpDecision<'a> {
    pub transient: TransientContract<'a>,
    pub drop_cached_locals: Vec<FrameSlot>,
}

/// Query for the planner-owned pre-op boundary transition.
#[derive(Clone, Debug)]
pub(crate) struct BeforeOpQuery<'a> {
    pub semantic_index: usize,
    pub resident_cache: &'a BTreeSet<FrameSlot>,
}

/// Query for slot-vs-cache lowering of a local op.
#[derive(Clone, Debug)]
pub(crate) struct LocalAccessQuery<'a> {
    pub semantic_index: usize,
    pub slot: FrameSlot,
    pub resident_cache: &'a BTreeSet<FrameSlot>,
}

/// Planner decision for a local op lowering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LocalAccessDecision {
    Slot,
    Cache,
}

/// Query for repairing a block edge.
#[derive(Clone, Debug)]
pub(crate) struct EdgeRepairQuery<'a> {
    pub succ_block: Option<CfgBlockId>,
    pub pred_exit: &'a [FrameSlot],
    pub succ_entry: &'a [FrameSlot],
}

/// Required repair actions for a mismatching edge.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct EdgeRepairDecision {
    pub ensure_cached_locals: Vec<FrameSlot>,
    pub reserve_cached_locals: Vec<FrameSlot>,
    pub drop_cached_locals: Vec<FrameSlot>,
}
