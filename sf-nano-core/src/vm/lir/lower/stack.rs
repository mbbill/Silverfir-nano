//! TOS-lane materialization and transient SSA value construction.

use alloc::vec::Vec;

use crate::vm::{
    lir::ir::{LirInst, LirInstKind, LirValue},
    plan::frame::FrameLayoutPlan,
};

use super::state::{BlockState, StackValue, ValueAlloc};

pub(super) fn pop_one(
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    values: &mut ValueAlloc,
) -> LirValue {
    let value = materialize_top_values(state, 1, frame, values)
        .into_iter()
        .next()
        .expect("one value present");
    consume_top(state, 1);
    value
}

pub(super) fn push_results(state: &mut BlockState, results: Vec<LirValue>) {
    state.stack.extend(results.into_iter().map(StackValue::Ssa));
}

pub(super) fn materialize_top_values(
    state: &mut BlockState,
    count: usize,
    frame: FrameLayoutPlan,
    values: &mut ValueAlloc,
) -> Vec<LirValue> {
    if count == 0 {
        return Vec::new();
    }
    let height = state.stack.len();
    let count = core::cmp::min(count, height);
    let start = height - count;
    (start..height)
        .map(|index| materialize_stack_index(state, frame, values, index as u16))
        .collect()
}

pub(super) fn materialize_stack_index(
    state: &mut BlockState,
    _frame: FrameLayoutPlan,
    values: &mut ValueAlloc,
    index: u16,
) -> LirValue {
    match state.stack[index as usize] {
        StackValue::Ssa(value) => value,
        StackValue::Frame(slot) => {
            let dst = values.fresh();
            state.ops.push(LirInst {
                kind: LirInstKind::ReadOperandSlot { slot, dst },
            });
            dst
        }
    }
}

fn consume_top(state: &mut BlockState, count: usize) {
    let new_len = state.stack.len().saturating_sub(count);
    state.stack.truncate(new_len);
}
