use crate::collections;

use crate::{
    error::WasmError,
    vm::middle::ssa_ir::ir::{
        DecodedOperand, SsaBlock, SsaCallArgs, SsaCallOp, SsaCallOperandLoc, SsaEdge, SsaInstView,
        SsaOperand, SsaProgram, SsaScalarResultLoc, SsaTerminator, SsaValue,
    },
};

#[derive(Clone, Copy)]
struct UseCount {
    value: SsaValue,
    count: u32,
}

/// Compact, read-mostly use counts for one SSA block.
///
/// SSA value ids are integers and the lowering pass only needs point lookups
/// and decrements. A sorted flat vector avoids a separately allocated B-tree
/// node for nearly every value while retaining logarithmic lookup.
pub(super) struct RemainingUses {
    entries: collections::Vec<UseCount>,
}

impl RemainingUses {
    fn from_unsorted(mut entries: collections::Vec<UseCount>) -> Self {
        entries.sort_unstable_by_key(|entry| entry.value);

        let mut write = 0usize;
        for read in 0..entries.len() {
            let entry = entries[read];
            if write != 0 && entries[write - 1].value == entry.value {
                entries[write - 1].count += entry.count;
            } else {
                entries[write] = entry;
                write += 1;
            }
        }
        entries.truncate(write);
        Self { entries }
    }

    #[inline]
    fn index_of(&self, value: SsaValue) -> Option<usize> {
        self.entries
            .binary_search_by_key(&value, |entry| entry.value)
            .ok()
    }

    #[inline]
    pub(super) fn count(&self, value: SsaValue) -> u32 {
        self.index_of(value)
            .map(|index| self.entries[index].count)
            .unwrap_or(0)
    }

    #[inline]
    pub(super) fn consume(&mut self, value: SsaValue) -> u32 {
        let Some(index) = self.index_of(value) else {
            return 0;
        };
        let entry = &mut self.entries[index];
        entry.count = entry.count.saturating_sub(1);
        entry.count
    }
}

#[inline]
fn record_use(value: SsaValue, uses: &mut collections::Vec<UseCount>) {
    uses.push(UseCount { value, count: 1 });
}

fn count_operand_use(operand: SsaOperand, uses: &mut collections::Vec<UseCount>) {
    if let DecodedOperand::Value(v) = operand.decode() {
        record_use(v, uses);
    }
    // Const / None operands have no SsaValue reference to track.
}

pub(super) fn compute_remaining_uses(block: &SsaBlock, program: &SsaProgram) -> RemainingUses {
    let mut uses = collections::Vec::new();
    for inst_idx in 0..block.ops.len() {
        match block.view(inst_idx, program) {
            SsaInstView::Spill { src, .. }
            | SsaInstView::CellSetSlot { src, .. }
            | SsaInstView::CellSetCache { src, .. } => {
                record_use(src, &mut uses);
            }
            SsaInstView::Value { args, .. } => {
                for operand in args.iter() {
                    count_operand_use(operand, &mut uses);
                }
            }
            SsaInstView::Fill { .. }
            | SsaInstView::CellGetSlot { .. }
            | SsaInstView::CellGetCache { .. }
            | SsaInstView::CellEnsureCache { .. }
            | SsaInstView::CellReserveCache { .. }
            | SsaInstView::CellDropCache { .. } => {}
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
            record_use(*cond, &mut uses);
            count_edge_uses(then_edge, &mut uses);
            count_edge_uses(else_edge, &mut uses);
        }
        SsaTerminator::BrTable { index, entries } => {
            record_use(*index, &mut uses);
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
        let mut op_uses = collections::Vec::new();
        for inst_idx in 0..block.ops.len() {
            match block.view(inst_idx, program) {
                SsaInstView::Spill { src, .. }
                | SsaInstView::CellSetSlot { src, .. }
                | SsaInstView::CellSetCache { src, .. } => {
                    record_use(src, &mut op_uses);
                }
                SsaInstView::Value { args, .. } => {
                    for operand in args.iter() {
                        if let DecodedOperand::Value(v) = operand.decode() {
                            record_use(v, &mut op_uses);
                        }
                    }
                }
                SsaInstView::Call(call) => count_call_uses(call, &mut op_uses),
                _ => {}
            }
        }
        let op_uses = RemainingUses::from_unsorted(op_uses);
        for entry in &op_uses.entries {
            debug_assert_eq!(
                entry.count, 1,
                "SSA-IR value {:?} has {} uses within ops (linear SSA requires exactly 1)",
                entry.value, entry.count,
            );
        }
    }

    RemainingUses::from_unsorted(uses)
}

fn count_edge_uses(edge: &SsaEdge, uses: &mut collections::Vec<UseCount>) {
    for binding in &edge.bindings {
        record_use(binding.value, uses);
    }
}

fn count_call_uses(call: &SsaCallOp, uses: &mut collections::Vec<UseCount>) {
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

fn count_call_args_uses(args: &SsaCallArgs, uses: &mut collections::Vec<UseCount>) {
    for arg in &args.live_suffix {
        record_use(arg.value, uses);
    }
}

fn count_call_operand_loc_use(loc: &SsaCallOperandLoc, uses: &mut collections::Vec<UseCount>) {
    if let SsaCallOperandLoc::Live { value, .. } = loc {
        record_use(*value, uses);
    }
}

fn count_scalar_result_loc_use(loc: &SsaScalarResultLoc, uses: &mut collections::Vec<UseCount>) {
    if let SsaScalarResultLoc::Live { value, .. } = loc {
        record_use(*value, uses);
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
