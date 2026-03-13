use alloc::collections::BTreeMap;

use crate::{
    error::WasmError,
    vm::lir::ir::{LirBlock, LirBoundaryOp, LirEdge, LirInstKind, LirTerminator, LirValue},
};

pub(super) fn compute_remaining_uses(block: &LirBlock) -> BTreeMap<LirValue, u32> {
    let mut uses = BTreeMap::new();
    for inst in &block.ops {
        match &inst.kind {
            LirInstKind::Value { args, .. } => {
                for value in args {
                    *uses.entry(*value).or_insert(0) += 1;
                }
            }
            LirInstKind::StoreSlot { src, .. } => {
                *uses.entry(*src).or_insert(0) += 1;
            }
            LirInstKind::LoadSlot { .. } => {}
            LirInstKind::Boundary(LirBoundaryOp::MemoryGrow { .. })
            | LirInstKind::Boundary(LirBoundaryOp::MemoryFill { .. })
            | LirInstKind::Boundary(LirBoundaryOp::MemoryCopy { .. })
            | LirInstKind::Boundary(LirBoundaryOp::TableGrow { .. })
            | LirInstKind::Boundary(LirBoundaryOp::TableFill { .. })
            | LirInstKind::Boundary(LirBoundaryOp::TableCopy { .. })
            | LirInstKind::Boundary(LirBoundaryOp::MemoryInit { .. })
            | LirInstKind::Boundary(LirBoundaryOp::DataDrop { .. })
            | LirInstKind::Boundary(LirBoundaryOp::TableInit { .. })
            | LirInstKind::Boundary(LirBoundaryOp::ElemDrop { .. })
            | LirInstKind::Boundary(LirBoundaryOp::CallExternal { .. })
            | LirInstKind::Boundary(LirBoundaryOp::CallInternal { .. })
            | LirInstKind::Boundary(LirBoundaryOp::CallIndirect { .. }) => {}
        }
    }

    match &block.terminator {
        LirTerminator::Goto(edge) => count_edge_uses(edge, &mut uses),
        LirTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => {
            *uses.entry(*cond).or_insert(0) += 1;
            count_edge_uses(then_edge, &mut uses);
            count_edge_uses(else_edge, &mut uses);
        }
        LirTerminator::BrTable { index, entries } => {
            *uses.entry(*index).or_insert(0) += 1;
            for edge in entries {
                count_edge_uses(edge, &mut uses);
            }
        }
        LirTerminator::Return { .. } | LirTerminator::TrapUnreachable => {}
    }

    uses
}

fn count_edge_uses(edge: &LirEdge, uses: &mut BTreeMap<LirValue, u32>) {
    for binding in &edge.bindings {
        *uses.entry(binding.value).or_insert(0) += 1;
    }
}

pub(super) fn single_result(results: &[LirValue]) -> Result<LirValue, WasmError> {
    match results {
        [value] => Ok(*value),
        _ => Err(WasmError::internal(
            "machine lowering expected exactly one result".into(),
        )),
    }
}

pub(super) fn single_arg(args: &[LirValue]) -> Result<LirValue, WasmError> {
    match args {
        [value] => Ok(*value),
        _ => Err(WasmError::internal(
            "machine lowering expected exactly one argument".into(),
        )),
    }
}

pub(super) fn two_args(args: &[LirValue]) -> Result<(LirValue, LirValue), WasmError> {
    match args {
        [lhs, rhs] => Ok((*lhs, *rhs)),
        _ => Err(WasmError::internal(
            "machine lowering expected exactly two arguments".into(),
        )),
    }
}
