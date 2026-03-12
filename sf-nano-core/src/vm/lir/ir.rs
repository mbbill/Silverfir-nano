//! Prepared backend-facing LIR.
//!
//! This is the frontend/native handoff for the engine's prepared single-pass
//! pipeline:
//! - canonical locals and deep stack values live in frame slots
//! - only a bounded set of transient values stays live as SSA values
//! - explicit spill/fill ops publish and reload transient values from operand
//!   slots so the backend never needs general register allocation

use alloc::vec::Vec;

use super::{
    leaf::LirLeafOp,
    runtime::LirRuntimeOp,
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
    Leaf {
        op: LirLeafOp,
        args: Vec<LirValue>,
        results: Vec<LirValue>,
    },
    Runtime {
        op: LirRuntimeOp,
        args: Vec<LirValue>,
        results: Vec<LirValue>,
    },
    /// Read a canonical frame slot, usually a local slot.
    ReadSlot {
        slot: FrameSlot,
        dst: LirValue,
    },
    /// Write a canonical frame slot, usually a local slot.
    WriteSlot {
        slot: FrameSlot,
        src: LirValue,
    },
    /// Publish one transient live value into its canonical operand slot.
    Spill {
        slot: FrameSlot,
        src: LirValue,
    },
    /// Reload one transient live value from its canonical operand slot.
    Fill {
        slot: FrameSlot,
        dst: LirValue,
    },
    /// Call using canonical frame spans for arguments and results.
    CallExternal {
        func_idx: u32,
        args: FrameSpan,
        results: FrameSpan,
    },
    /// Call using canonical frame spans for arguments and results.
    CallInternal {
        callee: u32,
        args: FrameSpan,
        results: FrameSpan,
    },
    /// Indirect call using canonical frame spans plus an explicit spilled index.
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
