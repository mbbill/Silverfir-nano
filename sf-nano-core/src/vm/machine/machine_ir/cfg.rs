use super::inst::MachineInst;
use crate::collections;

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
    /// MachineIR has already emitted everything except the final control
    /// transfer:
    /// - dirty cached-local flushes
    /// - `callee_frame_base` computation
    /// - `caller_result_base` computation (the absolute pointer at which the
    ///   callee's `Return` will copy its results)
    /// - zero-fill of the callee frame prefix beyond the argument span
    ///
    /// The arch/backend contract for this terminator is:
    /// - perform the stack overflow precheck for `callee_frame_base` using the
    ///   callee's runtime frame-size metadata
    /// - save the caller frame pointer and `caller_result_base` in a
    ///   backend-private call record (typically pushed onto the host stack
    ///   so the unified `Return` can recover them — the *exact* layout is
    ///   backend-private, not part of the abstract MIR contract)
    /// - move the machine frame pointer register to `callee_frame_base`
    /// - issue a native call (`bl`/`call`) to the compiled callee entry
    ///   described by `target`
    /// - after the call returns, check `C_RET0` for the trap-propagation
    ///   status: zero means success and control falls through to
    ///   `continuation`; non-zero means a descendant trapped and the
    ///   backend must branch to the function's `body_local_error_label`
    ///
    /// `continuation` stays in the MIR as the abstract CFG successor edge.
    /// The emulator and any future non-native backend uses it directly. On
    /// native backends it is preferred-fall-through; if the block layout
    /// pass cannot achieve adjacency, the backend emits one
    /// `b/jmp continuation_label` after the call.
    ///
    /// The backend must not redo any other frame setup or arrange any call
    /// record at MIR-visible offsets.
    Call {
        /// Compiled callee description.
        target: MachineCallTarget,
        /// GP register containing the absolute address of the callee frame
        /// base.
        ///
        /// MachineIR has already computed and validated this address. The
        /// backend should treat it as the new frame pointer value for the
        /// transfer.
        callee_frame_base: MachineReg,
        /// GP register containing the absolute address of the caller's
        /// result-receive region.
        caller_result_base: MachineReg,
        /// Caller CFG block to resume after a successful return.
        continuation: MachineBlockId,
    },
    /// Tail call transfer into another compiled MachineIR function.
    ///
    /// Unlike [`MachineTerminator::Call`], this does not create a new
    /// backend-private call record and it does not return to the current
    /// function. The backend:
    /// - reuses the current caller's call record
    /// - switches `MACHINE_FP_REG` to `callee_frame_base`
    /// - undoes the current function's body-entry shim/link save so the host
    ///   stack matches the callee body's expected entry shape
    /// - jumps to the callee internal entry described by `target`
    ///
    /// The callee then returns or traps directly to this function's caller.
    TailCall {
        target: MachineCallTarget,
        callee_frame_base: MachineReg,
    },
    /// Return from the function. The backend's `Return` lowering pops the
    /// backend-private call record left behind by the caller's `Call`,
    /// copies the function's `return_results` region
    /// into `*caller_result_base`, restores `MACHINE_FP_REG` to the
    /// caller's frame pointer, sets `C_RET0 = 0` (the success status),
    /// and executes the platform's native return.
    Return,
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
