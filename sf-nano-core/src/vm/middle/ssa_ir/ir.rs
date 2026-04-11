//! Prepared backend-facing SSA-IR.
//!
//! This is the frontend/native handoff for the engine's prepared pipeline:
//! - canonical locals and deep stack values live in frame slots
//! - only a bounded set of transient values stays live as SSA values
//! - explicit slot traffic publishes and reloads transient values through
//!   operand slots so the backend never needs general register allocation

use crate::collections;

use crate::value_type::ValueType;

use super::{leaf::SsaLeafOp, target::SsaTarget};
use crate::vm::middle::frame::{FrameSlot, FrameSpan};

/// One SSA value in prepared SSA-IR.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct SsaValue(pub u32);

/// An operand to a leaf SSA operation: either a transient value reference or
/// an inline constant absorbed from a preceding const definition.
///
/// The constant-folding pass in [`crate::vm::middle::optimize`] rewrites
/// `Value(v)` operands to `Const(bits)` when `v` was produced by a const
/// instruction and has no other uses. The machine layer lowers `Const`
/// operands via `MachineValue::Imm64`; it may encode the constant as a native
/// immediate or materialize it into a scratch register as a fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SsaOperand {
    Value(SsaValue),
    Const(u64),
}

impl SsaOperand {
    /// Extract the SSA value, panicking if this is an inline constant.
    ///
    /// Use this only in lowering paths that do not yet support inline
    /// constants. Paths that handle `Const` should match on the operand
    /// directly.
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
    pub is_param: bool,
    pub reads_before_write: bool,
}

/// Full prepared SSA-IR program for one function.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SsaProgram {
    pub entry: SsaTarget,
    pub blocks: collections::Vec<SsaBlock>,
    pub local_slot_types: collections::Vec<ValueType>,
    pub local_slot_info: collections::Vec<LocalSlotInfo>,
    pub block_entry_cached_slots: collections::Vec<collections::Vec<FrameSlot>>,
    pub block_cfg_origins: collections::Vec<collections::Vec<u32>>,
    pub value_types: collections::Vec<ValueType>,
    pub value_sink_local: collections::Vec<Option<FrameSlot>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EntryCacheRequirement {
    Ensure,
    Reserve,
}

impl SsaProgram {
    #[inline]
    pub(crate) fn value_sink(&self, value: SsaValue) -> Option<FrameSlot> {
        self.value_sink_local
            .get(value.0 as usize)
            .copied()
            .flatten()
    }

    #[cfg(test)]
    pub(crate) fn final_block_for_cfg_block(&self, cfg_block: u32) -> Option<SsaTarget> {
        self.block_cfg_origins
            .iter()
            .position(|origins| origins.contains(&cfg_block))
            .map(|index| SsaTarget(index as u32))
    }
}

pub(crate) fn entry_cache_requirement_from_ops(
    ops: &[SsaInst],
    slot: FrameSlot,
) -> Option<EntryCacheRequirement> {
    for inst in ops {
        match inst.kind {
            SsaInstKind::LocalGetCache {
                slot: accessed_slot,
                ..
            }
            | SsaInstKind::LocalEnsureCache {
                slot: accessed_slot,
            } => {
                if accessed_slot == slot {
                    return Some(EntryCacheRequirement::Ensure);
                }
            }
            SsaInstKind::LocalSetCache {
                slot: accessed_slot,
                ..
            }
            | SsaInstKind::LocalReserveCache {
                slot: accessed_slot,
            } => {
                if accessed_slot == slot {
                    return Some(EntryCacheRequirement::Reserve);
                }
            }
            SsaInstKind::LocalGetSlot {
                slot: accessed_slot,
                ..
            }
            | SsaInstKind::LocalSetSlot {
                slot: accessed_slot,
                ..
            }
            | SsaInstKind::LocalDropCache {
                slot: accessed_slot,
            } => {
                if accessed_slot == slot {
                    return None;
                }
            }
            SsaInstKind::Call(_) => return None,
            SsaInstKind::Value { .. } | SsaInstKind::Fill { .. } | SsaInstKind::Spill { .. } => {}
        }
    }
    None
}

#[inline]
pub(crate) fn entry_cache_requirement(
    ops: &[SsaInst],
    slot: FrameSlot,
    carried_through: bool,
) -> Option<EntryCacheRequirement> {
    entry_cache_requirement_from_ops(ops, slot)
        .or_else(|| carried_through.then_some(EntryCacheRequirement::Ensure))
}

#[inline]
pub(crate) fn block_entry_cache_requirement(
    entry_slots: &[FrameSlot],
    block: &SsaBlock,
    slot: FrameSlot,
) -> Option<EntryCacheRequirement> {
    entry_cache_requirement(&block.ops, slot, entry_slots.contains(&slot))
}

/// One SSA-IR basic block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SsaBlock {
    pub id: SsaTarget,
    pub params: collections::Vec<SsaValue>,
    pub ops: collections::Vec<SsaInst>,
    pub terminator: SsaTerminator,
}

/// One explicit mapping from a predecessor live-out value to a successor block parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SsaBinding {
    pub param: SsaValue,
    pub value: SsaValue,
}

/// One control-flow edge with explicit live-in bindings for the successor.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SsaEdge {
    pub target: SsaTarget,
    pub bindings: collections::Vec<SsaBinding>,
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
        args: collections::Vec<SsaOperand>,
        results: collections::Vec<SsaValue>,
    },
    Fill {
        slot: FrameSlot,
        dst: SsaValue,
    },
    Spill {
        slot: FrameSlot,
        src: SsaValue,
    },
    LocalGetSlot {
        slot: FrameSlot,
        dst: SsaValue,
    },
    LocalGetCache {
        slot: FrameSlot,
        dst: SsaValue,
    },
    LocalSetSlot {
        slot: FrameSlot,
        src: SsaValue,
    },
    LocalSetCache {
        slot: FrameSlot,
        src: SsaValue,
    },
    LocalEnsureCache {
        slot: FrameSlot,
    },
    LocalReserveCache {
        slot: FrameSlot,
    },
    LocalDropCache {
        slot: FrameSlot,
    },
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
        entries: collections::Vec<SsaEdge>,
    },
    Return {
        results: Option<FrameSpan>,
    },
    TrapUnreachable,
}
