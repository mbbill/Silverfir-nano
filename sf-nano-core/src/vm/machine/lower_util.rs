use tracked_alloc::collections::BTreeMap;

use crate::{
    error::WasmError,
    vm::middle::ssa_ir::ir::{
        DecodedOperand, SsaBlock, SsaCallArgs, SsaCallOp, SsaCallOperandLoc, SsaEdge, SsaInstView,
        SsaOperand, SsaProgram, SsaScalarResultLoc, SsaTerminator, SsaValue,
    },
};

fn count_operand_use(operand: SsaOperand, uses: &mut BTreeMap<SsaValue, u32>) {
    if let DecodedOperand::Value(v) = operand.decode() {
        *uses.entry(v).or_insert(0) += 1;
    }
    // Const / None operands have no SsaValue reference to track.
}

pub(super) fn compute_remaining_uses(
    block: &SsaBlock,
    program: &SsaProgram,
) -> BTreeMap<SsaValue, u32> {
    let mut uses = BTreeMap::new();
    for inst_idx in 0..block.ops.len() {
        match block.view(inst_idx, program) {
            SsaInstView::Spill { src, .. }
            | SsaInstView::LocalSetSlot { src, .. }
            | SsaInstView::LocalSetCache { src, .. } => {
                *uses.entry(src).or_insert(0) += 1;
            }
            SsaInstView::Value { args, .. } => {
                for operand in args.iter() {
                    count_operand_use(operand, &mut uses);
                }
            }
            SsaInstView::Fill { .. }
            | SsaInstView::LocalGetSlot { .. }
            | SsaInstView::LocalGetCache { .. }
            | SsaInstView::LocalEnsureCache { .. }
            | SsaInstView::LocalReserveCache { .. }
            | SsaInstView::LocalDropCache { .. } => {}
            SsaInstView::Call(call) => count_call_uses(call, &mut uses),
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
        SsaTerminator::Return { .. }
        | SsaTerminator::TrapUnreachable
        | SsaTerminator::EhThrow { .. }
        | SsaTerminator::EhThrowRef { .. } => {}
        SsaTerminator::ReturnScalar { result, .. } => {
            count_scalar_result_loc_use(result, &mut uses);
        }
        SsaTerminator::TailCallDirect { args, .. } => count_call_args_uses(args, &mut uses),
        SsaTerminator::TailCallIndirect { args, index, .. } => {
            count_call_args_uses(args, &mut uses);
            count_call_operand_loc_use(index, &mut uses);
        }
        SsaTerminator::TailCallRef {
            args, callee_ref, ..
        } => {
            count_call_args_uses(args, &mut uses);
            count_call_operand_loc_use(callee_ref, &mut uses);
        }
    }

    // Linear-SSA invariant: within the op stream, every SsaValue operand is
    // used exactly once.  Const operands are not counted.  Edge bindings
    // (terminators) may add additional uses for values that are live across
    // block boundaries.
    #[cfg(debug_assertions)]
    {
        let mut op_uses: BTreeMap<SsaValue, u32> = BTreeMap::new();
        for inst_idx in 0..block.ops.len() {
            match block.view(inst_idx, program) {
                SsaInstView::Spill { src, .. }
                | SsaInstView::LocalSetSlot { src, .. }
                | SsaInstView::LocalSetCache { src, .. } => {
                    *op_uses.entry(src).or_insert(0) += 1;
                }
                SsaInstView::Value { args, .. } => {
                    for operand in args.iter() {
                        if let DecodedOperand::Value(v) = operand.decode() {
                            *op_uses.entry(v).or_insert(0) += 1;
                        }
                    }
                }
                SsaInstView::Call(call) => count_call_uses(call, &mut op_uses),
                _ => {}
            }
        }
        for (&value, &count) in &op_uses {
            debug_assert_eq!(
                count, 1,
                "SSA-IR value {:?} has {} uses within ops (linear SSA requires exactly 1)",
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

fn count_call_uses(call: &SsaCallOp, uses: &mut BTreeMap<SsaValue, u32>) {
    match call {
        SsaCallOp::CallDirect { args, .. } => count_call_args_uses(args, uses),
        SsaCallOp::CallIndirect { args, index, .. } => {
            count_call_args_uses(args, uses);
            count_call_operand_loc_use(index, uses);
        }
        SsaCallOp::CallRef {
            args, callee_ref, ..
        } => {
            count_call_args_uses(args, uses);
            count_call_operand_loc_use(callee_ref, uses);
        }
    }
}

fn count_call_args_uses(args: &SsaCallArgs, uses: &mut BTreeMap<SsaValue, u32>) {
    for arg in &args.live_suffix {
        *uses.entry(arg.value).or_insert(0) += 1;
    }
}

fn count_call_operand_loc_use(loc: &SsaCallOperandLoc, uses: &mut BTreeMap<SsaValue, u32>) {
    if let SsaCallOperandLoc::Live { value, .. } = loc {
        *uses.entry(*value).or_insert(0) += 1;
    }
}

fn count_scalar_result_loc_use(loc: &SsaScalarResultLoc, uses: &mut BTreeMap<SsaValue, u32>) {
    if let SsaScalarResultLoc::Live { value, .. } = loc {
        *uses.entry(*value).or_insert(0) += 1;
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

pub(super) fn single_arg(args: &[SsaOperand]) -> Result<SsaOperand, WasmError> {
    match args {
        [value] => Ok(*value),
        _ => Err(WasmError::internal(
            "machine lowering expected exactly one argument".into(),
        )),
    }
}

pub(super) fn two_args(args: &[SsaOperand]) -> Result<(SsaOperand, SsaOperand), WasmError> {
    match args {
        [lhs, rhs] => Ok((*lhs, *rhs)),
        _ => Err(WasmError::internal(
            "machine lowering expected exactly two arguments".into(),
        )),
    }
}

pub(super) fn three_args(
    args: &[SsaOperand],
) -> Result<(SsaOperand, SsaOperand, SsaOperand), WasmError> {
    match args {
        [a, b, c] => Ok((*a, *b, *c)),
        _ => Err(WasmError::internal(
            "machine lowering expected exactly three arguments".into(),
        )),
    }
}

pub(super) fn four_args(
    args: &[SsaOperand],
) -> Result<(SsaOperand, SsaOperand, SsaOperand, SsaOperand), WasmError> {
    match args {
        [a, b, c, d] => Ok((*a, *b, *c, *d)),
        _ => Err(WasmError::internal(
            "machine lowering expected exactly four arguments".into(),
        )),
    }
}

pub(super) fn five_args(
    args: &[SsaOperand],
) -> Result<(SsaOperand, SsaOperand, SsaOperand, SsaOperand, SsaOperand), WasmError> {
    match args {
        [a, b, c, d, e] => Ok((*a, *b, *c, *d, *e)),
        _ => Err(WasmError::internal(
            "machine lowering expected exactly five arguments".into(),
        )),
    }
}
