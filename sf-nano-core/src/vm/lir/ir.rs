//! Prepared backend-facing LIR.
//!
//! This is the frontend/native handoff for the engine's prepared single-pass
//! pipeline:
//! - canonical locals and deep stack values live in frame slots
//! - only a bounded set of transient values stays live as SSA values
//! - explicit slot traffic publishes and reloads transient values through
//!   operand slots so the backend never needs general register allocation

use alloc::vec::Vec;

use super::{
    leaf::LirLeafOp,
    slot::{FrameSlot, FrameSpan},
    target::LirTarget,
};

/// One SSA value in prepared LIR.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LirValue(pub u32);

/// Preferred canonical local-slot ranking selected by planning.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LirLocalCachePrefs {
    pub preferred_slots: Vec<FrameSlot>,
}

/// Full prepared LIR program for one function.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LirProgram {
    pub entry: LirTarget,
    pub local_cache: LirLocalCachePrefs,
    pub blocks: Vec<LirBlock>,
}

/// One LIR basic block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LirBlock {
    pub id: LirTarget,
    /// Live SSA parameters required on block entry.
    pub params: Vec<LirValue>,
    pub ops: Vec<LirInst>,
    pub terminator: LirTerminator,
}

/// One explicit mapping from a predecessor live-out value to a successor block
/// parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LirBinding {
    pub param: LirValue,
    pub value: LirValue,
}

/// One control-flow edge with explicit live-in bindings for the successor.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LirEdge {
    pub target: LirTarget,
    pub bindings: Vec<LirBinding>,
}

/// One SSA operation inside a block body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LirInst {
    pub kind: LirInstKind,
}

/// Prepared frontend operation vocabulary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LirInstKind {
    Value {
        op: LirLeafOp,
        args: Vec<LirValue>,
        results: Vec<LirValue>,
    },
    /// Read a canonical frame slot, usually a local slot.
    LoadSlot { slot: FrameSlot, dst: LirValue },
    /// Write a canonical frame slot, usually a local slot.
    StoreSlot { slot: FrameSlot, src: LirValue },
    /// Slot-based call or runtime boundary.
    Boundary(LirBoundaryOp),
}

/// Prepared slot-based boundary operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LirBoundaryOp {
    MemoryGrow {
        mem_idx: u32,
        io: FrameSpan,
    },
    MemoryFill {
        mem_idx: u32,
        args: FrameSpan,
    },
    MemoryCopy {
        dst_mem_idx: u32,
        src_mem_idx: u32,
        args: FrameSpan,
    },
    TableGrow {
        table_idx: u32,
        args: FrameSpan,
        results: FrameSpan,
    },
    TableFill {
        table_idx: u32,
        args: FrameSpan,
    },
    TableCopy {
        dst_table_idx: u32,
        src_table_idx: u32,
        args: FrameSpan,
    },
    MemoryInit {
        data_idx: u32,
        mem_idx: u32,
        args: FrameSpan,
    },
    DataDrop {
        data_idx: u32,
    },
    TableInit {
        elem_idx: u32,
        table_idx: u32,
        args: FrameSpan,
    },
    ElemDrop {
        elem_idx: u32,
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
    /// Return using canonical frame result slots prepared before the terminator.
    Return {
        results: Option<FrameSpan>,
    },
    TrapUnreachable,
}
