use super::inst::MachineInst;
use crate::collections;

use super::abi::{MachineCallArgs, MachineCallResults, MachineReturnValue};
use super::types::{
    MachineBlockId, MachineCompareKind, MachineFloatWidth, MachineFuncId, MachineIntWidth,
    MachineReg, MachineRegOwner, MachineSign, MachineStorageType, MachineTrapKind, MachineValue,
};

/// One explicit machine block parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MachineBlockParam {
    pub reg: MachineReg,
    pub ty: MachineStorageType,
    pub owner: MachineRegOwner,
}

impl MachineBlockParam {
    #[inline]
    pub(crate) const fn gp_word(reg: MachineReg) -> Self {
        Self {
            reg,
            ty: MachineStorageType::GpWord,
            owner: MachineRegOwner::LinearValue,
        }
    }

    #[inline]
    pub(crate) const fn gp_i64(reg: MachineReg) -> Self {
        Self {
            reg,
            ty: MachineStorageType::GpI64,
            owner: MachineRegOwner::LinearValue,
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
            owner: MachineRegOwner::LinearValue,
        }
    }

    #[inline]
    pub(crate) const fn v128(reg: MachineReg) -> Self {
        Self {
            reg,
            ty: MachineStorageType::V128,
            owner: MachineRegOwner::LinearValue,
        }
    }

    #[inline]
    pub(crate) const fn with_owner(mut self, owner: MachineRegOwner) -> Self {
        self.owner = owner;
        self
    }
}

/// One explicit edge into another block.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct MachineEdge {
    pub target: MachineBlockId,
    pub args: collections::Vec<MachineValue>,
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
    /// Test bits: branch on `(src & mask) == 0` or `!= 0`.
    TestBits {
        width: MachineIntWidth,
        kind: MachineCompareKind,
        src: MachineValue,
        mask: MachineValue,
    },
}

/// One compiled-call target form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum MachineCallTarget {
    /// Compile-time-known compiled callee in the current machine module.
    ///
    /// Backends use this only to resolve the callee's internal entry address
    /// via module-link patching. Any metadata needed for frame setup has
    /// already been consumed by MachineIR before the call terminator is
    /// reached.
    Direct(MachineFuncId),
    /// Runtime-resolved compiled callee.
    ///
    /// The resolved target may live in the current module or another linked
    /// compiled module once MachineIR can represent that path directly.
    /// `callee_target` preserves the logical callee identity for consumers
    /// that care about it (the emulator, debug dumps), while `callee_entry`
    /// is the resolved native entry address used by native backends.
    Indirect {
        callee_target: MachineReg,
        callee_entry: MachineReg,
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
        entries: collections::Vec<MachineEdge>,
    },
    /// Compiled call transfer into another MachineIR function.
    ///
    /// MachineIR has already published frame-passed arguments and describes
    /// every register-passed argument through `args`. The backend prepares
    /// argument lanes from pre-switch sources, advances the machine frame
    /// pointer by `frame_delta`, performs a native call, restores the caller
    /// frame pointer, checks status, and then transfers to `success`.
    Call {
        /// Compiled callee description.
        target: MachineCallTarget,
        /// Byte delta from the current frame pointer to the callee frame.
        frame_delta: u32,
        args: MachineCallArgs,
        results: MachineCallResults,
        /// Caller CFG edge to resume after a successful return. Edge args are
        /// post-call values: result regs are produced by the call, and any
        /// non-result args must be preserved survivors.
        success: MachineEdge,
    },
    /// Tail call transfer into another compiled MachineIR function.
    ///
    /// Unlike [`MachineTerminator::Call`], this does not create a new
    /// backend-private call record and it does not return to the current
    /// function. The backend:
    /// - reuses the current caller's call record
    /// - publishes/repackages arguments into the current frame prefix and
    ///   argument lanes
    /// - undoes the current function's body-entry shim/link save so the host
    ///   stack matches the callee body's expected entry shape
    /// - jumps to the callee internal entry described by `target`
    ///
    /// The callee then returns or traps directly to this function's caller.
    TailCall {
        target: MachineCallTarget,
        args: MachineCallArgs,
    },
    /// Return from the function. The backend's `Return` lowering pops the
    /// backend-private call record left behind by the caller's `Call`,
    /// copies the function's `return_results` region
    /// into `*caller_result_base`, restores `MACHINE_FP_REG` to the
    /// caller's frame pointer, sets `C_RET0 = 0` (the success status),
    /// and executes the platform's native return.
    Return,
    /// Return one scalar value through the backend's wasm-to-wasm return lane.
    /// Backends still use the normal status register for success/trap
    /// propagation; the scalar lane is separate from that status channel.
    ReturnScalar {
        value: MachineReturnValue,
    },
    Trap {
        kind: MachineTrapKind,
    },
}

/// One machine IR block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MachineBlock {
    pub id: MachineBlockId,
    /// Block parameters are generic registers plus explicit semantic owners.
    /// Incoming values are supplied by the predecessor edge, the root public
    /// shim, or a local-call boundary.
    pub params: collections::Vec<MachineBlockParam>,
    pub ops: collections::Vec<MachineInst>,
    pub terminator: MachineTerminator,
}
