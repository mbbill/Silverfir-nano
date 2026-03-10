//! Backend-facing IR.
//!
//! Important:
//! - carries rotating-window top only
//! - does not carry explicit `T0..T3`
//! - does not carry `pre_height`
//! - does not force backend to reconstruct operand locations from stack height

use alloc::vec::Vec;

use super::slot::{FrameSlot, FrameSpan};
use super::target::LirTarget;
use crate::vm::{
    plan::{plan::PlannedBranchKind, spill::SpillArtifact},
    wasm::core_op::CoreOpKind,
};

pub use crate::vm::plan::tos::TosRotation as WindowRotation;

/// Lowered IR op.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LirOp {
    pub kind: LirOpKind,
    pub window: WindowRotation,
    pub alt: Option<LirTarget>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LirFrameCopy {
    pub src: FrameSpan,
    pub dst: FrameSpan,
}

/// Backend-facing op vocabulary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LirOpKind {
    Core(CoreOpKind),
    InitLocals {
        l0: FrameSlot,
        l1: FrameSlot,
        l2: FrameSlot,
    },
    CacheTransfer(SpillArtifact),
    LocalGetHot {
        reg: u8,
    },
    LocalSetHot {
        reg: u8,
    },
    LocalTeeHot {
        reg: u8,
    },
    LocalGetFrame {
        frame_slot: FrameSlot,
    },
    LocalSetFrame {
        frame_slot: FrameSlot,
    },
    LocalTeeFrame {
        frame_slot: FrameSlot,
    },
    Marker(LirMarkerKind),
    Branch {
        kind: PlannedBranchKind,
        fixup: Option<LirFrameCopy>,
        target: Option<LirTarget>,
    },
    BrTable {
        entries: Vec<LirBrTableEntry>,
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
pub enum LirMarkerKind {
    Block { params: u16, results: u16 },
    Loop { params: u16, results: u16 },
    If { params: u16, results: u16 },
    Else,
    End,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LirBrTableEntry {
    pub target: Option<LirTarget>,
    pub fixup: Option<LirFrameCopy>,
}

impl LirOpKind {
    pub fn reads_frame(&self) -> Vec<FrameSpan> {
        match self {
            LirOpKind::Core(_) => Vec::new(),
            LirOpKind::InitLocals { .. } => Vec::new(),
            LirOpKind::CacheTransfer(artifact) => match artifact {
                crate::vm::plan::spill::SpillArtifact::CacheTransfer(transfer) => {
                    alloc::vec![transfer.frame]
                }
            },
            LirOpKind::LocalGetHot { .. }
            | LirOpKind::LocalSetHot { .. }
            | LirOpKind::LocalTeeHot { .. } => Vec::new(),
            LirOpKind::LocalGetFrame { frame_slot } => alloc::vec![FrameSpan::single(*frame_slot)],
            LirOpKind::LocalSetFrame { frame_slot } => alloc::vec![FrameSpan::single(*frame_slot)],
            LirOpKind::LocalTeeFrame { frame_slot } => alloc::vec![FrameSpan::single(*frame_slot)],
            LirOpKind::Marker(_) => Vec::new(),
            LirOpKind::Branch { fixup, .. } => {
                let mut spans = Vec::new();
                if let Some(fixup) = fixup {
                    spans.push(fixup.src);
                }
                spans
            }
            LirOpKind::BrTable { entries } => entries
                .iter()
                .filter_map(|entry| entry.fixup.map(|fixup| fixup.src))
                .collect(),
            LirOpKind::CallExternal { args, .. }
            | LirOpKind::CallInternal { args, .. } => alloc::vec![*args],
            LirOpKind::CallIndirect {
                index_slot, args, ..
            } => alloc::vec![FrameSpan::single(*index_slot), *args],
            LirOpKind::Return { results } => results.iter().copied().collect(),
        }
    }

    pub fn writes_frame(&self) -> Vec<FrameSpan> {
        match self {
            LirOpKind::Core(_) => Vec::new(),
            LirOpKind::InitLocals { l0, l1, l2 } => alloc::vec![
                FrameSpan::single(*l0),
                FrameSpan::single(*l1),
                FrameSpan::single(*l2),
            ],
            LirOpKind::CacheTransfer(artifact) => match artifact {
                crate::vm::plan::spill::SpillArtifact::CacheTransfer(transfer) => {
                    alloc::vec![transfer.frame]
                }
            },
            LirOpKind::LocalGetHot { .. }
            | LirOpKind::LocalSetHot { .. }
            | LirOpKind::LocalTeeHot { .. } => Vec::new(),
            LirOpKind::LocalGetFrame { .. } => Vec::new(),
            LirOpKind::LocalSetFrame { frame_slot } => alloc::vec![FrameSpan::single(*frame_slot)],
            LirOpKind::LocalTeeFrame { frame_slot } => alloc::vec![FrameSpan::single(*frame_slot)],
            LirOpKind::Marker(_) => Vec::new(),
            LirOpKind::Branch { fixup, .. } => fixup.iter().map(|fixup| fixup.dst).collect(),
            LirOpKind::BrTable { entries } => entries
                .iter()
                .filter_map(|entry| entry.fixup.map(|fixup| fixup.dst))
                .collect(),
            LirOpKind::CallExternal { results, .. }
            | LirOpKind::CallInternal { results, .. }
            | LirOpKind::CallIndirect { results, .. } => alloc::vec![*results],
            LirOpKind::Return { .. } => Vec::new(),
        }
    }

    pub fn stack_effect(&self) -> (u8, u8) {
        match self {
            LirOpKind::Core(kind) => crate::vm::wasm::core_op::stack_effect(kind),
            LirOpKind::InitLocals { .. } | LirOpKind::CacheTransfer(_) | LirOpKind::Marker(_) => {
                (0, 0)
            }
            LirOpKind::LocalGetHot { .. } | LirOpKind::LocalGetFrame { .. } => (0, 1),
            LirOpKind::LocalSetHot { .. } | LirOpKind::LocalSetFrame { .. } => (1, 0),
            LirOpKind::LocalTeeHot { .. } | LirOpKind::LocalTeeFrame { .. } => (0, 0),
            LirOpKind::Branch { kind, .. } => match kind {
                PlannedBranchKind::Br => (0, 0),
                PlannedBranchKind::BrIf => (1, 0),
            },
            LirOpKind::BrTable { .. } => (1, 0),
            LirOpKind::CallExternal { .. }
            | LirOpKind::CallInternal { .. }
            | LirOpKind::CallIndirect { .. }
            | LirOpKind::Return { .. } => (0, 0),
        }
    }
}
