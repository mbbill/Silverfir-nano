//! Prepared backend-facing SSA-IR.

use alloc::vec::Vec;

use crate::value_type::ValueType;

use super::{leaf::SsaLeafOp, target::SsaTarget};
use crate::vm::middle::frame::{FrameSlot, FrameSpan};

/// One SSA value in prepared SSA-IR.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct SsaValue(pub u32);

/// An operand to a leaf SSA operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SsaOperand {
    Value(SsaValue),
    Const(u64),
}

impl SsaOperand {
    #[inline]
    pub(crate) fn unwrap_value(self) -> SsaValue {
        match self {
            Self::Value(v) => v,
            Self::Const(_) => panic!("expected SsaOperand::Value, got Const"),
        }
    }
}

/// Stable facts about a local slot, carried from preparation to the backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LocalSlotInfo {
    pub is_param: bool,
    pub reads_before_write: bool,
}

/// Full prepared SSA-IR program for one function.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SsaProgram {
    pub entry: SsaTarget,
    pub blocks: Vec<SsaBlock>,
    pub local_slot_types: Vec<ValueType>,
    pub local_slot_info: Vec<LocalSlotInfo>,
    pub block_entry_cached_slots: Vec<Vec<FrameSlot>>,
    pub value_types: Vec<ValueType>,
    pub value_sink_local: Vec<Option<FrameSlot>>,
}

impl SsaProgram {
    #[inline]
    pub(crate) fn value_sink(&self, value: SsaValue) -> Option<FrameSlot> {
        self.value_sink_local
            .get(value.0 as usize)
            .copied()
            .flatten()
    }
}

/// One SSA-IR basic block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SsaBlock {
    pub id: SsaTarget,
    pub params: Vec<SsaValue>,
    pub ops: Vec<SsaInst>,
    pub terminator: SsaTerminator,
}

/// One explicit mapping from a predecessor live-out value to a successor block parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SsaBinding {
    pub param: SsaValue,
    pub value: SsaValue,
}

/// One control-flow edge with explicit live-in bindings for the successor.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SsaEdge {
    pub target: SsaTarget,
    pub bindings: Vec<SsaBinding>,
}

/// One SSA operation inside a block body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SsaInst {
    pub kind: SsaInstKind,
}

/// Prepared frontend operation vocabulary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SsaInstKind {
    Value {
        op: SsaLeafOp,
        args: Vec<SsaOperand>,
        results: Vec<SsaValue>,
    },
    Fill {
        slot: FrameSlot,
        dst: SsaValue,
    },
    Spill {
        slot: FrameSlot,
        src: SsaValue,
    },
    LocalGetSlot {
        slot: FrameSlot,
        dst: SsaValue,
    },
    LocalGetCache {
        slot: FrameSlot,
        dst: SsaValue,
    },
    LocalSetSlot {
        slot: FrameSlot,
        src: SsaValue,
    },
    LocalSetCache {
        slot: FrameSlot,
        src: SsaValue,
    },
    LocalEnsureCache {
        slot: FrameSlot,
    },
    LocalReserveCache {
        slot: FrameSlot,
    },
    LocalDropCache {
        slot: FrameSlot,
    },
    Call(SsaCallOp),
}

/// Prepared slot-based call operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SsaCallOp {
    CallDirect {
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
pub(crate) enum SsaTerminator {
    Goto(SsaEdge),
    Branch {
        cond: SsaValue,
        then_edge: SsaEdge,
        else_edge: SsaEdge,
    },
    BrTable {
        index: SsaValue,
        entries: Vec<SsaEdge>,
    },
    Return {
        results: Option<FrameSpan>,
    },
    TrapUnreachable,
}
