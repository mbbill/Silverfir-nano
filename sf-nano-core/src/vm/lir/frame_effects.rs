//! Frame-access helpers for prepared LIR instructions.

use alloc::vec::Vec;

use super::{
    ir::LirInstKind,
    slot::{FrameSpan},
};

pub fn reads_frame(kind: &LirInstKind) -> Vec<FrameSpan> {
    match kind {
        LirInstKind::Leaf { .. }
        | LirInstKind::Runtime { .. }
        | LirInstKind::WriteSlot { .. }
        | LirInstKind::Spill { .. } => Vec::new(),
        LirInstKind::ReadSlot { slot, .. } | LirInstKind::Fill { slot, .. } => {
            alloc::vec![FrameSpan::single(*slot)]
        }
        LirInstKind::CallExternal { args, .. } | LirInstKind::CallInternal { args, .. } => {
            alloc::vec![*args]
        }
        LirInstKind::CallIndirect {
            index_slot, args, ..
        } => alloc::vec![FrameSpan::single(*index_slot), *args],
    }
}

pub fn writes_frame(kind: &LirInstKind) -> Vec<FrameSpan> {
    match kind {
        LirInstKind::Leaf { .. }
        | LirInstKind::Runtime { .. }
        | LirInstKind::ReadSlot { .. }
        | LirInstKind::Fill { .. } => Vec::new(),
        LirInstKind::WriteSlot { slot, .. } | LirInstKind::Spill { slot, .. } => {
            alloc::vec![FrameSpan::single(*slot)]
        }
        LirInstKind::CallExternal { results, .. }
        | LirInstKind::CallInternal { results, .. }
        | LirInstKind::CallIndirect { results, .. } => alloc::vec![*results],
    }
}
