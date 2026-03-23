//! Prepared backend-facing SSA-IR.
//!
//! This is the frontend/native handoff for the engine's prepared single-pass
//! pipeline:
//! - canonical locals and deep stack values live in frame slots
//! - only a bounded set of transient values stays live as SSA values
//! - explicit slot traffic publishes and reloads transient values through
//!   operand slots so the backend never needs general register allocation

use alloc::vec::Vec;

use crate::value_type::ValueType;

use super::{leaf::SsaLeafOp, target::SsaTarget};
use crate::vm::middle::frame::{FrameSlot, FrameSpan};

/// One SSA value in prepared SSA-IR.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct SsaValue(pub u32);

/// An operand to a leaf SSA operation: either a transient value reference or
/// an inline constant absorbed from a preceding const definition.
///
/// The constant-folding pass in `optimize.rs` rewrites `Value(v)` operands
/// to `Const(bits)` when `v` was produced by a const instruction and has no
/// other uses.  The architecture backend lowers `Const` operands via
/// `MachineValue::Imm64`; it may encode the constant as a native immediate
/// or materialize it into a scratch register as a fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SsaOperand {
    Value(SsaValue),
    Const(u64),
}

impl SsaOperand {
    /// Extract the SsaValue, panicking if this is an inline constant.
    ///
    /// Use this only in lowering paths that do not yet support inline
    /// constants (e.g. i64 pair ops, memory loads/stores).  Paths that
    /// handle `Const` should match on the operand directly.
    #[inline]
    pub(crate) fn unwrap_value(self) -> SsaValue {
        match self {
            Self::Value(v) => v,
            Self::Const(_) => panic!("expected SsaOperand::Value, got Const"),
        }
    }
}

/// Analysis facts about a cached local, carried from planning to the backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CachedLocalInfo {
    /// True if this local is a function parameter (local index < param count).
    pub is_param: bool,
    /// True if this non-param local may be read before being written at
    /// function-entry scope (control depth 0). Only meaningful when `!is_param`.
    pub reads_before_write: bool,
}

/// Preferred canonical local-slot ranking selected by planning, per bank.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SsaLocalCachePrefs {
    /// GP-bank cached local slots (i32, i64, ref).
    pub gp_preferred_slots: Vec<FrameSlot>,
    /// Semantic types for `gp_preferred_slots`, kept in the same order.
    pub gp_preferred_types: Vec<ValueType>,
    /// FP-bank cached local slots (f32, f64).
    pub fp_preferred_slots: Vec<FrameSlot>,
    /// Semantic types for `fp_preferred_slots`, kept in the same order.
    pub fp_preferred_types: Vec<ValueType>,
    /// Per-local analysis facts, parallel to `gp_preferred_slots`.
    pub gp_local_info: Vec<CachedLocalInfo>,
    /// Per-local analysis facts, parallel to `fp_preferred_slots`.
    pub fp_local_info: Vec<CachedLocalInfo>,
}

/// Full prepared SSA-IR program for one function.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SsaProgram {
    pub entry: SsaTarget,
    pub local_cache: SsaLocalCachePrefs,
    pub blocks: Vec<SsaBlock>,
    /// Per-value type information indexed by `SsaValue.0`.
    ///
    /// When non-empty, every allocated SsaValue has a corresponding entry.
    /// Float values (F32, F64) should be placed in FP transients; all others
    /// in GP transients.
    pub value_types: Vec<ValueType>,
}

/// One SSA-IR basic block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SsaBlock {
    pub id: SsaTarget,
    /// Live SSA parameters required on block entry.
    pub params: Vec<SsaValue>,
    pub ops: Vec<SsaInst>,
    pub terminator: SsaTerminator,
}

/// One explicit mapping from a predecessor live-out value to a successor block
/// parameter.
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
    /// Read a canonical frame slot, usually a local slot.
    LoadSlot { slot: FrameSlot, dst: SsaValue },
    /// Write a canonical frame slot, usually a local slot.
    StoreSlot { slot: FrameSlot, src: SsaValue },
    /// Slot-based call or runtime boundary.
    Boundary(SsaBoundaryOp),
}

/// Prepared slot-based boundary operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SsaBoundaryOp {
    MemoryGrow {
        mem_idx: u32,
        io: FrameSpan,
    },
    MemoryFill {
        mem_idx: u32,
        args: FrameSpan,
    },
    MemoryCopy {
        dst_mem_idx: u32,
        src_mem_idx: u32,
        args: FrameSpan,
    },
    TableGrow {
        table_idx: u32,
        args: FrameSpan,
        results: FrameSpan,
    },
    TableFill {
        table_idx: u32,
        args: FrameSpan,
    },
    TableCopy {
        dst_table_idx: u32,
        src_table_idx: u32,
        args: FrameSpan,
    },
    MemoryInit {
        data_idx: u32,
        mem_idx: u32,
        args: FrameSpan,
    },
    DataDrop {
        data_idx: u32,
    },
    TableInit {
        elem_idx: u32,
        table_idx: u32,
        args: FrameSpan,
    },
    ElemDrop {
        elem_idx: u32,
    },
    CallExternal {
        func_idx: u32,
        args: FrameSpan,
        results: FrameSpan,
        /// Per cached-local flag: `true` = skip reload at continuation.
        /// Parallel to `gp_preferred_slots ++ fp_preferred_slots` in cache prefs.
        /// Empty if no analysis was performed.
        skip_reload: Vec<bool>,
    },
    CallInternal {
        callee: u32,
        args: FrameSpan,
        results: FrameSpan,
        /// Per cached-local flag: `true` = skip reload at continuation.
        skip_reload: Vec<bool>,
    },
    CallIndirect {
        type_idx: u32,
        table_idx: u32,
        index_slot: FrameSlot,
        args: FrameSpan,
        results: FrameSpan,
        /// Per cached-local flag: `true` = skip reload at continuation.
        skip_reload: Vec<bool>,
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
    /// Return using canonical frame result slots prepared before the terminator.
    Return {
        results: Option<FrameSpan>,
    },
    TrapUnreachable,
}
