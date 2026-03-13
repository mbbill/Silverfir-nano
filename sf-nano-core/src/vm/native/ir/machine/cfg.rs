use alloc::vec::Vec;

use super::inst::MachineInst;
use super::types::{
    MachineBlockId, MachineCompareKind, MachineFloatWidth, MachineFuncId, MachineIntWidth,
    MachineReg, MachineSign, MachineTrapKind, MachineValue,
};

/// One explicit edge into another block.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MachineEdge {
    pub target: MachineBlockId,
    pub args: Vec<MachineValue>,
}

/// One explicit branch condition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MachineBranchCond {
    Value(MachineValue),
    IntCompare {
        width: MachineIntWidth,
        kind: MachineCompareKind,
        sign: MachineSign,
        lhs: MachineValue,
        rhs: MachineValue,
    },
    FloatCompare {
        width: MachineFloatWidth,
        kind: MachineCompareKind,
        lhs: MachineValue,
        rhs: MachineValue,
    },
}

/// One machine terminator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MachineTerminator {
    Jump(MachineEdge),
    Branch {
        cond: MachineBranchCond,
        then_edge: MachineEdge,
        else_edge: MachineEdge,
    },
    JumpTable {
        index: MachineValue,
        entries: Vec<MachineEdge>,
    },
    /// Direct local call. Call setup stores and call-link writes are explicit
    /// instructions before this terminator; the terminator performs the actual
    /// transfer to the callee internal entry.
    CallDirect {
        callee: MachineFuncId,
        callee_frame_base: MachineReg,
        continuation: MachineBlockId,
    },
    /// Indirect local call after the target entry has already been resolved by
    /// earlier machine-level code.
    CallIndirect {
        callee_entry: MachineValue,
        callee_frame_base: MachineReg,
        continuation: MachineBlockId,
    },
    /// Return using canonical frame result slots already prepared before the
    /// terminator. The return itself performs only the call-link/frame
    /// restoration transfer.
    Return,
    Trap {
        kind: MachineTrapKind,
    },
}

/// One machine IR block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MachineBlock {
    pub id: MachineBlockId,
    /// Block parameters are generic registers. Incoming values are supplied by
    /// the predecessor edge, the root public shim, or a local-call boundary.
    pub params: Vec<MachineReg>,
    pub ops: Vec<MachineInst>,
    pub terminator: MachineTerminator,
}
