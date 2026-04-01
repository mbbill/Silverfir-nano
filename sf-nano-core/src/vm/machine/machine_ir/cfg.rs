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
    /// - stack overflow precheck
    /// - zero-fill of the callee frame prefix beyond the argument span
    /// - call-link writes for caller frame pointer and caller result-base
    ///
    /// The arch/backend contract for this terminator is:
    /// - resolve `callee` to the callee's native entry point
    /// - materialize the native address of `continuation`
    /// - store that continuation address at
    ///   `call_link_base + call_link.continuation_offset`
    /// - move the machine frame pointer register to `callee_frame_base`
    /// - branch/jump to the callee entry
    ///
    /// The backend must not redo frame setup, stack checks, or rebuild the
    /// call-link record. MachineIR has already computed `call_link_base` and
    /// written every call-link field except the continuation address.
    CallDirect {
        /// Compile-time-known local callee id.
        ///
        /// Backends use this only to resolve the callee entry address. Any
        /// metadata needed for frame setup or call-link placement has already
        /// been consumed by MachineIR before this terminator is reached.
        callee: MachineFuncId,
        /// GP register containing the absolute address of the callee frame
        /// base.
        ///
        /// MachineIR has already computed and validated this address. The
        /// backend should treat it as the new frame pointer value for the
        /// transfer.
        callee_frame_base: MachineReg,
        /// GP register containing the absolute address of the first byte of
        /// the callee call-link record.
        ///
        /// MachineIR has already chosen the record location and written every
        /// field except the continuation address. The backend must store the
        /// native continuation address at
        /// `call_link_base + call_link.continuation_offset`.
        call_link_base: MachineReg,
        /// Caller CFG block that should execute after the callee returns.
        ///
        /// The backend must lower this block id to a native code address and
        /// store that address into the call-link continuation slot before
        /// branching to the callee.
        continuation: MachineBlockId,
    },
    /// Indirect local call after earlier machine-level code has already
    /// resolved and validated the local callee target.
    ///
    /// Unlike `CallDirect`, the callee entry is a runtime value loaded through
    /// the indirect dispatch path rather than a compile-time-known function
    /// id. MachineIR has already emitted:
    /// - bounds / type / kind checks on the indirect target
    /// - `callee_frame_base` computation
    /// - stack overflow precheck
    /// - zero-fill of the callee frame prefix beyond the argument span
    /// - call-link writes for caller frame pointer and caller result-base
    /// - computation of `call_link_base`
    ///
    /// The arch/backend contract for this terminator is:
    /// - materialize the native address of `continuation`
    /// - store that continuation address at
    ///   `call_link_base + call_link.continuation_offset`
    /// - move the machine frame pointer register to `callee_frame_base`
    /// - branch/jump to `callee_entry`
    ///
    /// The backend must not reinterpret tables, redo dispatch checks, or
    /// rebuild the call-link record. That work already happened in MachineIR.
    CallIndirect {
        /// GP register containing the resolved local target id from the
        /// indirect dispatch path.
        ///
        /// Current native backends may not need this once `callee_entry` has
        /// been loaded, but it remains part of the contract for consumers that
        /// care about the logical callee identity, such as the emulator or a
        /// backend that chooses to re-consult runtime metadata.
        callee_target: MachineReg,
        /// GP register containing the resolved native entry address of the
        /// local callee.
        ///
        /// The backend should jump/branch to this address directly after
        /// installing the continuation pointer and updating the frame pointer.
        callee_entry: MachineReg,
        /// GP register containing the absolute address of the callee frame
        /// base.
        ///
        /// MachineIR has already computed and validated this address. The
        /// backend should treat it as the new frame pointer value for the
        /// transfer.
        callee_frame_base: MachineReg,
        /// GP register containing the absolute address of the first byte of
        /// the callee call-link record.
        ///
        /// MachineIR has already chosen the record location and written every
        /// field except the continuation address. The backend must store the
        /// native continuation address at
        /// `call_link_base + call_link.continuation_offset`.
        call_link_base: MachineReg,
        /// Caller CFG block that should execute after the callee returns.
        ///
        /// The backend must lower this block id to a native code address and
        /// store that address into the call-link continuation slot before
        /// branching to `callee_entry`.
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
