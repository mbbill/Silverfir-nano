use alloc::vec::Vec;

use super::inst::MachineInst;
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
    /// Test bits: branch on `(src & mask) == 0` or `!= 0`.
    TestBits {
        width: MachineIntWidth,
        kind: MachineCompareKind,
        src: MachineValue,
        mask: MachineValue,
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
    /// Direct local call to a compile-time-known local callee.
    ///
    /// MachineIR has already emitted everything except the final control
    /// transfer:
    /// - dirty cached-local flushes
    /// - `callee_frame_base` computation
    /// - `caller_result_base` computation (the absolute pointer at which the
    ///   callee's `Return` will copy its results)
    /// - stack overflow precheck
    /// - zero-fill of the callee frame prefix beyond the argument span
    ///
    /// The arch/backend contract for this terminator is:
    /// - save the caller frame pointer and `caller_result_base` in a
    ///   backend-private call record (typically pushed onto the host stack
    ///   so the unified `Return` can recover them — the *exact* layout is
    ///   backend-private, not part of the abstract MIR contract)
    /// - move the machine frame pointer register to `callee_frame_base`
    /// - issue a native call (`bl`/`call`) to the callee's internal entry,
    ///   resolved at link time from `callee` via `direct_call_patches`
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
    /// The backend must not redo frame setup, stack checks, or arrange any
    /// call-link memory record at MIR-visible offsets.
    CallDirect {
        /// Compile-time-known local callee id.
        ///
        /// Backends use this only to resolve the callee's internal entry
        /// address (via `direct_call_patches`). Any metadata needed for
        /// frame setup has already been consumed by MachineIR before this
        /// terminator is reached.
        callee: MachineFuncId,
        /// GP register containing the absolute address of the callee frame
        /// base.
        ///
        /// MachineIR has already computed and validated this address. The
        /// backend should treat it as the new frame pointer value for the
        /// transfer.
        callee_frame_base: MachineReg,
        /// GP register containing the absolute address of the first byte
        /// of the caller's result-receive region (i.e.
        /// `caller_fp + results_offset`).
        ///
        /// MachineIR has already computed this address. The backend hands
        /// it off to the callee via the backend-private call record so
        /// that the callee's unified `Return` can copy results directly
        /// into the caller's frame without consulting any MIR-visible
        /// call-link layout.
        caller_result_base: MachineReg,
        /// Caller CFG block that should execute after the callee returns
        /// successfully.
        ///
        /// On native backends this is the preferred fall-through target.
        /// On the emulator it is consulted directly to drive control flow.
        continuation: MachineBlockId,
    },
    /// Indirect local call after earlier machine-level code has already
    /// resolved and validated the local callee target.
    ///
    /// Unlike `CallDirect`, the callee entry is a runtime value loaded
    /// through the indirect dispatch path rather than a compile-time-known
    /// function id. MachineIR has already emitted:
    /// - bounds / type / kind checks on the indirect target
    /// - `callee_frame_base` computation
    /// - `caller_result_base` computation
    /// - stack overflow precheck
    /// - zero-fill of the callee frame prefix beyond the argument span
    ///
    /// The arch/backend contract is the same as `CallDirect` except that
    /// the call target is the runtime register `callee_entry` rather than
    /// a compile-time-known function id.
    CallIndirect {
        /// GP register containing the resolved local target id from the
        /// indirect dispatch path.
        ///
        /// Current native backends may not need this once `callee_entry`
        /// has been loaded, but it remains part of the contract for
        /// consumers that care about the logical callee identity (the
        /// emulator, debug dumps).
        callee_target: MachineReg,
        /// GP register containing the resolved native entry address of
        /// the local callee.
        ///
        /// The backend should jump/branch to this address after
        /// installing the call record and updating the frame pointer.
        callee_entry: MachineReg,
        /// GP register containing the absolute address of the callee
        /// frame base.
        callee_frame_base: MachineReg,
        /// GP register containing the absolute address of the caller's
        /// result-receive region. See `CallDirect::caller_result_base`.
        caller_result_base: MachineReg,
        /// Caller CFG block to resume after a successful return.
        continuation: MachineBlockId,
    },
    /// Return from the function. The backend's `Return` lowering pops the
    /// backend-private call record left behind by the caller's `CallDirect`
    /// / `CallIndirect`, copies the function's `return_results` region
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
    pub params: Vec<MachineBlockParam>,
    pub ops: Vec<MachineInst>,
    pub terminator: MachineTerminator,
}
