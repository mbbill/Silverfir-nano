//! Planning-stage IR and public planning inputs/outputs.

use alloc::vec::Vec;

use crate::vm::wasm::{common::SemanticTarget, primitive_op::PrimitiveOpKind};

use super::{
    config::PlanConfig,
    frame::{FrameLayoutPlan, FrameSlot, FrameSpan},
    group::GroupPlan,
    hot_local::HotLocalPlan,
    policy::PlanPolicy,
    tos::{SpillArtifact, TosRotation},
};

/// Planned local placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlannedLocal {
    Hot(u8),
    Frame(FrameSlot),
}

/// Planning-stage op vocabulary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlannedOpKind {
    Primitive(PrimitiveOpKind),
    InitHotLocals {
        inits: Vec<PlannedHotLocalInit>,
    },
    Spill(SpillArtifact),
    LocalGet {
        local: PlannedLocal,
    },
    LocalSet {
        local: PlannedLocal,
    },
    LocalTee {
        local: PlannedLocal,
    },
    Marker(PlannedMarkerKind),
    Branch {
        kind: PlannedBranchKind,
        condition_slot: Option<FrameSlot>,
        payload: Option<FrameSpan>,
        target: Option<SemanticTarget>,
    },
    BrTable {
        index_slot: FrameSlot,
        entries: Vec<PlannedBrTableEntry>,
    },
    CallExternal {
        func_idx: u32,
        args: FrameSpan,
        results: FrameSpan,
    },
    CallInternal {
        callee: u32,
        args: FrameSpan,
        results: FrameSpan,
    },
    CallIndirect {
        type_idx: u32,
        table_idx: u32,
        index_slot: FrameSlot,
        args: FrameSpan,
        results: FrameSpan,
    },
    Return {
        results: Option<FrameSpan>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlannedHotLocalInit {
    pub reg: u8,
    pub frame_slot: FrameSlot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlannedMarkerKind {
    Block { params: u16, results: u16 },
    Loop { params: u16, results: u16 },
    If { params: u16, results: u16 },
    Else,
    End,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlannedBranchKind {
    Br,
    BrIf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlannedBrTableEntry {
    pub target: Option<SemanticTarget>,
    pub payload: Option<FrameSpan>,
}

/// One planned op. This is still stack-aware, but the stack-aware concepts live
/// here instead of leaking into LIR/backends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedOp {
    pub kind: PlannedOpKind,
    pub rotation: TosRotation,
    pub height: u16,
    pub alt: Option<SemanticTarget>,
}

/// Planned program bundle.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PlannedProgram {
    pub frame: FrameLayoutPlan,
    pub hot_locals: Option<HotLocalPlan>,
    pub ops: Vec<PlannedOp>,
    pub groups: GroupPlan,
}

/// Planning input bundle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanningInput {
    pub config: PlanConfig,
    pub policy: PlanPolicy,
}
