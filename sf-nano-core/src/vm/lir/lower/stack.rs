//! TOS-lane materialization and transient SSA value construction.

use alloc::vec::Vec;

use crate::vm::{
    lir::ir::{LirInst, LirInstKind, LirValue},
    plan::{config::PlanConfig, frame::FrameLayoutPlan},
};

use super::state::{BlockState, ValueAlloc};

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

pub(super) fn push_results(
    state: &mut BlockState,
    results: Vec<LirValue>,
    frame: FrameLayoutPlan,
    config: PlanConfig,
    values: &mut ValueAlloc,
) {
    state.height = state.height.saturating_add(results.len() as u16);
    state.tos.extend(results);
    normalize_tos(state, frame, config, values);
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
    let count = core::cmp::min(count, state.height as usize);
    let start = state.height as usize - count;
    (start..state.height as usize)
        .map(|index| materialize_stack_index(state, frame, values, index as u16))
        .collect()
}

pub(super) fn materialize_stack_index(
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    values: &mut ValueAlloc,
    index: u16,
) -> LirValue {
    let cached_start = state.height.saturating_sub(state.tos.len() as u16);
    if index >= cached_start {
        let offset = (index - cached_start) as usize;
        state.tos[offset]
    } else {
        let dst = values.fresh();
        state.ops.push(LirInst {
            kind: LirInstKind::ReadOperandSlot {
                slot: frame.operand_slot(index),
                dst,
            },
        });
        dst
    }
}

fn consume_top(state: &mut BlockState, count: usize) {
    let available = state.tos.len().min(count);
    let new_len = state.tos.len().saturating_sub(available);
    state.tos.truncate(new_len);
    state.height = state.height.saturating_sub(count as u16);
}

fn normalize_tos(
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    config: PlanConfig,
    values: &mut ValueAlloc,
) {
    let target_cached = core::cmp::min(state.height as usize, config.tos_register_count as usize);

    if state.tos.len() > target_cached {
        let spill_count = state.tos.len() - target_cached;
        let spill_base = state.height as usize - state.tos.len();
        for (idx, value) in state.tos.iter().copied().take(spill_count).enumerate() {
            state.ops.push(LirInst {
                kind: LirInstKind::WriteOperandSlot {
                    slot: frame.operand_slot((spill_base + idx) as u16),
                    src: value,
                },
            });
        }
        state.tos.drain(0..spill_count);
    }

    if state.tos.len() < target_cached {
        let fill_count = target_cached - state.tos.len();
        let fill_start = state.height as usize - target_cached;
        let mut filled = Vec::with_capacity(fill_count);
        for idx in 0..fill_count {
            let dst = values.fresh();
            state.ops.push(LirInst {
                kind: LirInstKind::ReadOperandSlot {
                    slot: frame.operand_slot((fill_start + idx) as u16),
                    dst,
                },
            });
            filled.push(dst);
        }
        filled.extend(state.tos.iter().copied());
        state.tos = filled;
    }
}
