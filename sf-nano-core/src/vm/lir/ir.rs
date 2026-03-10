//! CFG + SSA backend-facing LIR.
//!
//! This is the shared semantic handoff between planning/grouping and the
//! backend family. It must not carry rotating-window metadata, stack height,
//! or backend-side stack reconstruction hints.

use alloc::vec::Vec;

use crate::vm::plan::hot_local::HotLocalPlan;

use super::{
    leaf::LirLeafOp,
    slot::{FrameSlot, FrameSpan},
    target::LirTarget,
};

/// One SSA value inside LIR.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LirValue(pub u32);

/// LIR-visible subset of the backend VM register budget.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LirAbi {
    pub tos_register_count: u8,
    pub hot_local_count: u8,
}

/// Full LIR program for one function.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LirProgram {
    pub entry: LirTarget,
    pub blocks: Vec<LirBlock>,
    pub abi: LirAbi,
    /// Function-static mapping for named hot-local slots.
    pub hot_locals: Option<HotLocalPlan>,
}

/// One LIR basic block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LirBlock {
    pub id: LirTarget,
    pub params: LirBlockParams,
    pub ops: Vec<LirInst>,
    pub terminator: LirTerminator,
}

/// Explicit incoming block state.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LirBlockParams {
    /// Incoming TOS-lane values. Position is the lane index.
    ///
    /// Hot locals have function-static identity and therefore do not appear
    /// here as edge-threaded state.
    pub tos: Vec<LirValue>,
}

/// One SSA operation inside a block body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LirInst {
    pub kind: LirInstKind,
}

/// One control-flow edge with explicit outgoing state arguments.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LirEdge {
    pub target: LirTarget,
    /// Outgoing TOS-lane values for the successor. Position is the lane index.
    ///
    /// Hot locals have function-static identity and therefore do not appear
    /// here as edge-threaded state.
    pub tos: Vec<LirValue>,
}

/// Target-facing operation vocabulary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LirInstKind {
    Leaf {
        op: LirLeafOp,
        args: Vec<LirValue>,
        results: Vec<LirValue>,
    },
    WriteOperandSlot {
        slot: FrameSlot,
        src: LirValue,
    },
    ReadOperandSlot {
        slot: FrameSlot,
        dst: LirValue,
    },
    ReadHotLocal {
        /// Hot-local identity comes from `LirProgram::hot_locals`.
        reg: u8,
        dst: LirValue,
    },
    WriteHotLocal {
        /// Hot-local identity comes from `LirProgram::hot_locals`.
        reg: u8,
        src: LirValue,
    },
    ReadFrameLocal {
        frame_slot: FrameSlot,
        dst: LirValue,
    },
    WriteFrameLocal {
        frame_slot: FrameSlot,
        src: LirValue,
    },
    CallExternal {
        func_idx: u32,
        args: Vec<LirValue>,
        results: Vec<LirValue>,
    },
    CallInternal {
        callee: u32,
        args: Vec<LirValue>,
        results: Vec<LirValue>,
    },
    CallIndirect {
        type_idx: u32,
        table_idx: u32,
        index: LirValue,
        args: Vec<LirValue>,
        results: Vec<LirValue>,
    },
}

/// Explicit CFG terminator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LirTerminator {
    Goto(LirEdge),
    Branch {
        cond: LirValue,
        then_edge: LirEdge,
        else_edge: LirEdge,
    },
    BrTable {
        index: LirValue,
        entries: Vec<LirEdge>,
    },
    Return {
        values: Vec<LirValue>,
    },
    TrapUnreachable,
}

impl LirInstKind {
    pub fn reads_frame(&self) -> Vec<FrameSpan> {
        match self {
            LirInstKind::Leaf { .. } => Vec::new(),
            LirInstKind::WriteOperandSlot { .. } => Vec::new(),
            LirInstKind::ReadOperandSlot { slot, .. } => alloc::vec![FrameSpan::single(*slot)],
            LirInstKind::ReadHotLocal { .. } | LirInstKind::WriteHotLocal { .. } => Vec::new(),
            LirInstKind::ReadFrameLocal { frame_slot, .. } => {
                alloc::vec![FrameSpan::single(*frame_slot)]
            }
            LirInstKind::WriteFrameLocal { .. } => Vec::new(),
            LirInstKind::CallExternal { .. }
            | LirInstKind::CallInternal { .. }
            | LirInstKind::CallIndirect { .. } => Vec::new(),
        }
    }

    pub fn writes_frame(&self) -> Vec<FrameSpan> {
        match self {
            LirInstKind::Leaf { .. } => Vec::new(),
            LirInstKind::WriteOperandSlot { slot, .. } => alloc::vec![FrameSpan::single(*slot)],
            LirInstKind::ReadOperandSlot { .. } => Vec::new(),
            LirInstKind::ReadHotLocal { .. } | LirInstKind::WriteHotLocal { .. } => Vec::new(),
            LirInstKind::ReadFrameLocal { .. } => Vec::new(),
            LirInstKind::WriteFrameLocal { frame_slot, .. } => {
                alloc::vec![FrameSpan::single(*frame_slot)]
            }
            LirInstKind::CallExternal { .. }
            | LirInstKind::CallInternal { .. }
            | LirInstKind::CallIndirect { .. } => Vec::new(),
        }
    }
}
