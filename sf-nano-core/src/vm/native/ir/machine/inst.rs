use crate::vm::native::ir::machine::{MachineBranchCond, MachineTrapKind};

use super::types::{
    MachineAddr, MachineCompareKind, MachineConstId, MachineConvertOp, MachineExternId,
    MachineFloatBinaryOp, MachineFloatUnaryOp, MachineFloatWidth, MachineIntBinaryOp,
    MachineIntUnaryOp, MachineIntWidth, MachineLoadExtension, MachineMemWidth, MachineReg,
    MachineSign, MachineValue,
};

/// Helper call that falls through in the same function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MachineHelperCall {
    /// Opaque external target id. Sidecar binding data resolves this to the
    /// real Rust helper wrapper address during backend finalization.
    pub target: MachineExternId,
    /// Read-only sidecar metadata for this call site.
    ///
    /// The backend treats this as an opaque constant reference. Helper-specific
    /// interpretation stays out of the ISA layer. Helpers operate on canonical
    /// frame regions named by this metadata, so unrelated live machine values
    /// remain live across the call in machine semantics.
    pub metadata: MachineConstId,
}

/// One machine instruction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MachineInst {
    pub kind: MachineInstKind,
}

/// Straight-line machine instruction vocabulary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MachineInstKind {
    Move {
        dst: MachineReg,
        src: MachineValue,
    },
    Lea {
        dst: MachineReg,
        addr: MachineAddr,
    },
    Load {
        dst: MachineReg,
        addr: MachineAddr,
        width: MachineMemWidth,
        extension: MachineLoadExtension,
    },
    Store {
        addr: MachineAddr,
        width: MachineMemWidth,
        src: MachineValue,
    },
    IntUnary {
        width: MachineIntWidth,
        op: MachineIntUnaryOp,
        dst: MachineReg,
        src: MachineValue,
    },
    IntBinary {
        width: MachineIntWidth,
        op: MachineIntBinaryOp,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    },
    IntCompare {
        width: MachineIntWidth,
        kind: MachineCompareKind,
        sign: MachineSign,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    },
    FloatUnary {
        width: MachineFloatWidth,
        op: MachineFloatUnaryOp,
        dst: MachineReg,
        src: MachineValue,
    },
    FloatBinary {
        width: MachineFloatWidth,
        op: MachineFloatBinaryOp,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    },
    FloatCompare {
        width: MachineFloatWidth,
        kind: MachineCompareKind,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    },
    Convert {
        op: MachineConvertOp,
        dst: MachineReg,
        src: MachineValue,
    },
    Select {
        dst: MachineReg,
        on_true: MachineValue,
        on_false: MachineValue,
        cond: MachineValue,
    },
    TrapIf {
        kind: MachineTrapKind,
        cond: MachineBranchCond,
    },
    CallHelper(MachineHelperCall),
}
