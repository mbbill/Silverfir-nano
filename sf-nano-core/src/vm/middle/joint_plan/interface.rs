//! Rewrite-facing planner interface.
//!
//! The planner owns immutable facts and policy. The rewriter owns mutable
//! lowering state. Every planner consultation should flow through these query
//! and decision types so the boundary stays explicit.

use tracked_alloc::collections::BTreeSet;

use crate::{value_type::ValueType, vm::middle::frame::FrameSlot};

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

/// Chosen block-open boundary.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BlockOpenDecision<'a> {
    pub transient: TransientContract<'a>,
    pub cached_locals: &'a [FrameSlot],
    /// Full semantic type stack at block entry.
    /// Used by the rewriter to know types when filling spilled values inline.
    pub stack_types: &'a [ValueType],
}

/// Target boundary facts for control-flow canonicalization.
///
/// Only `stack_height` and `spill_depth` are needed by the rewriter for
/// branch target contracts. This avoids requiring a full `live_types` slice.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TargetEntryDecision {
    pub stack_height: u16,
    pub spill_depth: u16,
}

impl TargetEntryDecision {
    #[inline]
    pub(crate) const fn live_value_count(&self) -> u16 {
        self.stack_height.saturating_sub(self.spill_depth)
    }
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
