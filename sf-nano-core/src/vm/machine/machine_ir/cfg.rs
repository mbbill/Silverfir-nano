use alloc::vec::Vec;

use super::inst::MachineInst;
use super::types::{
    MachineBlockId, MachineCompareKind, MachineFloatWidth, MachineFuncId, MachineIntWidth,
    MachineReg, MachineSign, MachineStorageType, MachineTrapKind, MachineValue,
};

/// One explicit machine block parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MachineBlockParam {
    pub reg: MachineReg,
    pub ty: MachineStorageType,
}

impl MachineBlockParam {
    #[inline]
    pub(crate) const fn gp_word(reg: MachineReg) -> Self {
        Self {
            reg,
            ty: MachineStorageType::GpWord,
        }
    }

    #[inline]
    pub(crate) const fn gp_i64(reg: MachineReg) -> Self {
        Self {
            reg,
            ty: MachineStorageType::GpI64,
        }
    }

    #[inline]
    pub(crate) const fn fp(reg: MachineReg, width: MachineFloatWidth) -> Self {
        Self {
            reg,
            ty: match width {
                MachineFloatWidth::F32 => MachineStorageType::Fp32,
                MachineFloatWidth::F64 => MachineStorageType::Fp64,
            },
        }
    }
}

/// One explicit edge into another block.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct MachineEdge {
    pub target: MachineBlockId,
    pub args: Vec<MachineValue>,
}

/// One explicit branch condition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum MachineBranchCond {
    Value(MachineValue),
    IntCompare {
        width: MachineIntWidth,
        kind: MachineCompareKind,
        sign: MachineSign,
        lhs: MachineValue,
        rhs: MachineValue,
    },
}

/// One machine terminator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum MachineTerminator {
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
    /// Indirect local call after earlier machine-level code has already
    /// resolved and validated the local callee target.
    ///
    /// Unlike `CallDirect`, the remaining per-callee setup is dynamic:
    /// execution below MachineIR may need runtime metadata for the resolved
    /// target in order to finish any call-link/local-prefix handling before the
    /// actual transfer. Lowering above MachineIR therefore resolves the local
    /// callee target, computes the callee frame base, and publishes the
    /// canonical argument/result slot facts needed by that later step.
    CallIndirect {
        callee_target: MachineValue,
        callee_frame_base: MachineReg,
        arg_slots: u16,
        caller_result_base: u16,
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
pub(crate) struct MachineBlock {
    pub id: MachineBlockId,
    /// Block parameters are generic registers. Incoming values are supplied by
    /// the predecessor edge, the root public shim, or a local-call boundary.
    pub params: Vec<MachineBlockParam>,
    pub ops: Vec<MachineInst>,
    pub terminator: MachineTerminator,
}
