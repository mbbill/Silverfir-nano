//! Target-independent native IR.
//!
//! This is the single IR layer after shared CFG + SSA LIR.
//! It owns:
//! - native-facing virtual registers
//! - explicit edge copies
//! - native-owned call/helper lowering shape
//!
//! It must not reintroduce stack height, TOS rotation, or legacy LIR metadata.

use alloc::vec::Vec;

use crate::vm::{
    backend::BackendConfig,
    lir::{
        leaf::LirLeafOp,
        slot::FrameSlot,
        target::LirTarget,
    },
    plan::{frame::FrameLayoutPlan, group::GroupPlan, hot_local::HotLocalPlan},
};

/// One native virtual register.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NativeReg(pub u32);

/// One native CFG block id.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NativeBlockId(pub u32);

impl NativeBlockId {
    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl From<LirTarget> for NativeBlockId {
    #[inline]
    fn from(value: LirTarget) -> Self {
        Self(value.0)
    }
}

/// Native-visible backend budget.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeAbi {
    pub ctx_register_count: u8,
    pub fp_register_count: u8,
    pub tmp_register_count: u8,
    pub hot_local_count: u8,
    pub tos_register_count: u8,
}

impl From<BackendConfig> for NativeAbi {
    #[inline]
    fn from(value: BackendConfig) -> Self {
        Self {
            ctx_register_count: value.ctx_register_count,
            fp_register_count: value.fp_register_count,
            tmp_register_count: value.tmp_register_count,
            hot_local_count: value.hot_local_count,
            tos_register_count: value.tos_register_count,
        }
    }
}

/// Full native IR program for one function.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NativeProgram {
    pub entry: NativeBlockId,
    pub blocks: Vec<NativeBlock>,
    pub abi: NativeAbi,
    pub frame: FrameLayoutPlan,
    pub groups: GroupPlan,
    pub hot_locals: Option<HotLocalPlan>,
}

/// One native IR basic block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeBlock {
    pub id: NativeBlockId,
    pub params: Vec<NativeReg>,
    pub ops: Vec<NativeInst>,
    pub terminator: NativeTerminator,
}

/// One native IR instruction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeInst {
    pub kind: NativeInstKind,
}

/// One edge-local copy needed to satisfy successor live-ins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeCopy {
    pub src: NativeReg,
    pub dst: NativeReg,
}

/// One explicit native CFG edge.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NativeEdge {
    pub target: NativeBlockId,
    pub copies: Vec<NativeCopy>,
}

/// Native instruction vocabulary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeInstKind {
    Leaf {
        op: LirLeafOp,
        args: Vec<NativeReg>,
        results: Vec<NativeReg>,
    },
    ReadOperandSlot {
        slot: FrameSlot,
        dst: NativeReg,
    },
    WriteOperandSlot {
        slot: FrameSlot,
        src: NativeReg,
    },
    ReadHotLocal {
        reg: u8,
        dst: NativeReg,
    },
    WriteHotLocal {
        reg: u8,
        src: NativeReg,
    },
    ReadFrameLocal {
        frame_slot: FrameSlot,
        dst: NativeReg,
    },
    WriteFrameLocal {
        frame_slot: FrameSlot,
        src: NativeReg,
    },
    CallExternal {
        func_idx: u32,
        args: Vec<NativeReg>,
        results: Vec<NativeReg>,
    },
    CallInternal {
        callee: u32,
        args: Vec<NativeReg>,
        results: Vec<NativeReg>,
    },
    CallIndirect {
        type_idx: u32,
        table_idx: u32,
        index: NativeReg,
        args: Vec<NativeReg>,
        results: Vec<NativeReg>,
    },
}

/// Native IR terminator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeTerminator {
    Goto(NativeEdge),
    Branch {
        cond: NativeReg,
        then_edge: NativeEdge,
        else_edge: NativeEdge,
    },
    BrTable {
        index: NativeReg,
        entries: Vec<NativeEdge>,
    },
    Return {
        values: Vec<NativeReg>,
    },
    TrapUnreachable,
}
