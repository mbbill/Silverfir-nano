use alloc::collections::BTreeMap;

use crate::{
    error::WasmError,
    vm::middle::ssa_ir::ir::{SsaBlock, SsaBoundaryOp, SsaEdge, SsaInstKind, SsaTerminator, SsaValue},
};

pub(super) fn compute_remaining_uses(block: &SsaBlock) -> BTreeMap<SsaValue, u32> {
    let mut uses = BTreeMap::new();
    for inst in &block.ops {
        match &inst.kind {
            SsaInstKind::Value { args, .. } => {
                for value in args {
                    *uses.entry(*value).or_insert(0) += 1;
                }
            }
            SsaInstKind::StoreSlot { src, .. } => {
                *uses.entry(*src).or_insert(0) += 1;
            }
            SsaInstKind::LoadSlot { .. } => {}
            SsaInstKind::Boundary(SsaBoundaryOp::MemoryGrow { .. })
            | SsaInstKind::Boundary(SsaBoundaryOp::MemoryFill { .. })
            | SsaInstKind::Boundary(SsaBoundaryOp::MemoryCopy { .. })
            | SsaInstKind::Boundary(SsaBoundaryOp::TableGrow { .. })
            | SsaInstKind::Boundary(SsaBoundaryOp::TableFill { .. })
            | SsaInstKind::Boundary(SsaBoundaryOp::TableCopy { .. })
            | SsaInstKind::Boundary(SsaBoundaryOp::MemoryInit { .. })
            | SsaInstKind::Boundary(SsaBoundaryOp::DataDrop { .. })
            | SsaInstKind::Boundary(SsaBoundaryOp::TableInit { .. })
            | SsaInstKind::Boundary(SsaBoundaryOp::ElemDrop { .. })
            | SsaInstKind::Boundary(SsaBoundaryOp::CallExternal { .. })
            | SsaInstKind::Boundary(SsaBoundaryOp::CallInternal { .. })
            | SsaInstKind::Boundary(SsaBoundaryOp::CallIndirect { .. }) => {}
        }
    }

    match &block.terminator {
        SsaTerminator::Goto(edge) => count_edge_uses(edge, &mut uses),
        SsaTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => {
            *uses.entry(*cond).or_insert(0) += 1;
            count_edge_uses(then_edge, &mut uses);
            count_edge_uses(else_edge, &mut uses);
        }
        SsaTerminator::BrTable { index, entries } => {
            *uses.entry(*index).or_insert(0) += 1;
            for edge in entries {
                count_edge_uses(edge, &mut uses);
            }
        }
        SsaTerminator::Return { .. } | SsaTerminator::TrapUnreachable => {}
    }

    // Linear-SSA invariant: within the op stream, every value is used exactly
    // once. Edge bindings (terminators) may add additional uses for values
    // that are live across block boundaries.
    #[cfg(debug_assertions)]
    {
        let mut op_uses: BTreeMap<SsaValue, u32> = BTreeMap::new();
        for inst in &block.ops {
            match &inst.kind {
                SsaInstKind::Value { args, .. } => {
                    for value in args {
                        *op_uses.entry(*value).or_insert(0) += 1;
                    }
                }
                SsaInstKind::StoreSlot { src, .. } => {
                    *op_uses.entry(*src).or_insert(0) += 1;
                }
                _ => {}
            }
        }
        for (&value, &count) in &op_uses {
            debug_assert_eq!(
                count, 1,
                "LIR value {:?} has {} uses within ops (linear SSA requires exactly 1)",
                value, count,
            );
        }
    }

    uses
}

fn count_edge_uses(edge: &SsaEdge, uses: &mut BTreeMap<SsaValue, u32>) {
    for binding in &edge.bindings {
        *uses.entry(binding.value).or_insert(0) += 1;
    }
}

pub(super) fn single_result(results: &[SsaValue]) -> Result<SsaValue, WasmError> {
    match results {
        [value] => Ok(*value),
        _ => Err(WasmError::internal(
            "machine lowering expected exactly one result".into(),
        )),
    }
}

pub(super) fn single_arg(args: &[SsaValue]) -> Result<SsaValue, WasmError> {
    match args {
        [value] => Ok(*value),
        _ => Err(WasmError::internal(
            "machine lowering expected exactly one argument".into(),
        )),
    }
}

pub(super) fn two_args(args: &[SsaValue]) -> Result<(SsaValue, SsaValue), WasmError> {
    match args {
        [lhs, rhs] => Ok((*lhs, *rhs)),
        _ => Err(WasmError::internal(
            "machine lowering expected exactly two arguments".into(),
        )),
    }
}
