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

/// Stable facts about a local slot, carried from preparation to the backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LocalSlotInfo {
    /// True if this slot belongs to a function parameter (local index < param count).
    pub is_param: bool,
    /// True if this non-param local slot may be read before being written at
    /// function-entry scope (control depth 0). Only meaningful when `!is_param`.
    pub reads_before_write: bool,
}

/// Full prepared SSA-IR program for one function.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SsaProgram {
    pub entry: SsaTarget,
    pub blocks: Vec<SsaBlock>,
    /// Stable semantic type for each canonical local slot.
    ///
    /// The index is the `FrameSlot.0` for local slots only. Non-local frame
    /// slots do not appear here.
    pub local_slot_types: Vec<ValueType>,
    /// Stable semantic facts for each canonical local slot.
    ///
    /// This is parallel to `local_slot_types`.
    pub local_slot_info: Vec<LocalSlotInfo>,
    /// Per-block canonical cached-local entry state, indexed by block id.
    ///
    /// Each entry lists the local frame slots that must already be resident
    /// when control enters that SSA block. The order is canonical and is used
    /// by machine lowering to thread cache-carry state through edge args.
    pub block_entry_cached_slots: Vec<Vec<FrameSlot>>,
    /// Per-value type information indexed by `SsaValue.0`.
    ///
    /// When non-empty, every allocated SsaValue has a corresponding entry.
    /// Float values (F32, F64) should be placed in FP transients; all others
    /// in GP transients.
    pub value_types: Vec<ValueType>,
    /// Per-value optional sink-local annotation indexed by `SsaValue.0`.
    /// When `Some(slot)`, the producer of this value may write directly to
    /// that local's home, and the subsequent `LocalSet` is elidable.
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
    /// Reload a spilled stack/TOS value from an operand slot.
    Fill { slot: FrameSlot, dst: SsaValue },
    /// Publish a live stack/TOS value into an operand slot.
    Spill { slot: FrameSlot, src: SsaValue },
    /// Read a canonical local slot in frame memory.
    LocalGetSlot { slot: FrameSlot, dst: SsaValue },
    /// Read a cached local, loading it into the cache first if needed.
    LocalGetCache { slot: FrameSlot, dst: SsaValue },
    /// Write a canonical local slot in frame memory.
    LocalSetSlot { slot: FrameSlot, src: SsaValue },
    /// Write a cached local, allocating cache residency first if needed.
    LocalSetCache { slot: FrameSlot, src: SsaValue },
    /// Ensure a local is resident in the cache, loading from the canonical
    /// frame slot if needed. Does not produce an SSA value.
    LocalEnsureCache { slot: FrameSlot },
    /// Explicitly evict a cached local. If dirty, this writes back first.
    LocalDropCache { slot: FrameSlot },
    /// Slot-based function call (args/results passed via frame slots, not SSA values).
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
    /// Return using canonical frame result slots prepared before the terminator.
    Return {
        results: Option<FrameSpan>,
    },
    TrapUnreachable,
}
