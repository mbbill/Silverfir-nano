//! Frame-access helpers for prepared LIR instructions.

use alloc::vec::Vec;

use super::{
    ir::{LirBoundaryOp, LirInstKind},
    slot::FrameSpan,
};

pub fn reads_frame(kind: &LirInstKind) -> Vec<FrameSpan> {
    match kind {
        LirInstKind::Value { .. } | LirInstKind::Legalized { .. } | LirInstKind::StoreSlot { .. } => Vec::new(),
        LirInstKind::LoadSlot { slot, .. } => {
            alloc::vec![FrameSpan::single(*slot)]
        }
        LirInstKind::Boundary(boundary) => reads_boundary(boundary),
    }
}

pub fn writes_frame(kind: &LirInstKind) -> Vec<FrameSpan> {
    match kind {
        LirInstKind::Value { .. } | LirInstKind::Legalized { .. } | LirInstKind::LoadSlot { .. } => Vec::new(),
        LirInstKind::StoreSlot { slot, .. } => {
            alloc::vec![FrameSpan::single(*slot)]
        }
        LirInstKind::Boundary(boundary) => writes_boundary(boundary),
    }
}

fn reads_boundary(kind: &LirBoundaryOp) -> Vec<FrameSpan> {
    match kind {
        LirBoundaryOp::MemoryGrow { io, .. } => alloc::vec![*io],
        LirBoundaryOp::MemoryFill { args, .. }
        | LirBoundaryOp::MemoryCopy { args, .. }
        | LirBoundaryOp::TableGrow { args, .. }
        | LirBoundaryOp::TableFill { args, .. }
        | LirBoundaryOp::TableCopy { args, .. }
        | LirBoundaryOp::MemoryInit { args, .. }
        | LirBoundaryOp::TableInit { args, .. }
        | LirBoundaryOp::CallExternal { args, .. }
        | LirBoundaryOp::CallInternal { args, .. } => alloc::vec![*args],
        LirBoundaryOp::DataDrop { .. } | LirBoundaryOp::ElemDrop { .. } => Vec::new(),
        LirBoundaryOp::CallIndirect {
            index_slot, args, ..
        } => alloc::vec![FrameSpan::single(*index_slot), *args],
    }
}

fn writes_boundary(kind: &LirBoundaryOp) -> Vec<FrameSpan> {
    match kind {
        LirBoundaryOp::MemoryGrow { io, .. } => alloc::vec![*io],
        LirBoundaryOp::TableGrow { results, .. }
        | LirBoundaryOp::CallExternal { results, .. }
        | LirBoundaryOp::CallInternal { results, .. }
        | LirBoundaryOp::CallIndirect { results, .. } => alloc::vec![*results],
        LirBoundaryOp::MemoryFill { .. }
        | LirBoundaryOp::MemoryCopy { .. }
        | LirBoundaryOp::TableFill { .. }
        | LirBoundaryOp::TableCopy { .. }
        | LirBoundaryOp::MemoryInit { .. }
        | LirBoundaryOp::DataDrop { .. }
        | LirBoundaryOp::TableInit { .. }
        | LirBoundaryOp::ElemDrop { .. } => Vec::new(),
    }
}
