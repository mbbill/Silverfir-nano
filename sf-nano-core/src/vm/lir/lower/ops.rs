//! Block-body op lowering into LIR instructions.

use crate::error::WasmError;
use crate::vm::{
    lir::{
        ir::{LirInst, LirInstKind},
        leaf::LirLeafOp,
    },
    plan::{
        config::PlanConfig,
        frame::FrameLayoutPlan,
        plan::{PlannedLocal, PlannedOpKind},
    },
    wasm::core_op::stack_effect,
};

use super::{
    stack::{materialize_top_values, pop_one, push_results},
    state::{BlockState, ValueAlloc},
};

pub(super) fn lower_core(
    kind: &crate::vm::wasm::core_op::CoreOpKind,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    config: PlanConfig,
    values: &mut ValueAlloc,
) {
    let (pop, push) = stack_effect(kind);
    let args = materialize_top_values(state, pop as usize, frame, values);
    consume_top(state, pop as usize);
    let results = values.many(push as usize);
    state.ops.push(LirInst {
        kind: LirInstKind::Leaf {
            op: LirLeafOp::from(kind.clone()),
            args,
            results: results.clone(),
        },
    });
    push_results(state, results, frame, config, values);
}

pub(super) fn lower_local_get(
    planned_kind: &PlannedOpKind,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    config: PlanConfig,
    values: &mut ValueAlloc,
) -> Result<(), WasmError> {
    let dst = values.fresh();
    match planned_kind {
        PlannedOpKind::LocalGet {
            local: PlannedLocal::Hot(reg),
        } => {
            state.ops.push(LirInst {
                kind: LirInstKind::ReadHotLocal { reg: *reg, dst },
            });
        }
        PlannedOpKind::LocalGet {
            local: PlannedLocal::Frame(frame_slot),
        } => {
            state.ops.push(LirInst {
                kind: LirInstKind::ReadFrameLocal {
                    frame_slot: *frame_slot,
                    dst,
                },
            });
        }
        _ => {
            return Err(WasmError::internal(
                "local.get lowering expected planned local.get".into(),
            ));
        }
    }
    push_results(state, alloc::vec![dst], frame, config, values);
    Ok(())
}

pub(super) fn lower_local_set(
    planned_kind: &PlannedOpKind,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    values: &mut ValueAlloc,
) -> Result<(), WasmError> {
    let src = pop_one(state, frame, values);
    match planned_kind {
        PlannedOpKind::LocalSet {
            local: PlannedLocal::Hot(reg),
        } => {
            state.ops.push(LirInst {
                kind: LirInstKind::WriteHotLocal { reg: *reg, src },
            });
        }
        PlannedOpKind::LocalSet {
            local: PlannedLocal::Frame(frame_slot),
        } => {
            state.ops.push(LirInst {
                kind: LirInstKind::WriteFrameLocal {
                    frame_slot: *frame_slot,
                    src,
                },
            });
        }
        _ => {
            return Err(WasmError::internal(
                "local.set lowering expected planned local.set".into(),
            ));
        }
    }
    Ok(())
}

pub(super) fn lower_local_tee(
    planned_kind: &PlannedOpKind,
    state: &mut BlockState,
) -> Result<(), WasmError> {
    let src = *state
        .tos
        .last()
        .ok_or_else(|| WasmError::internal("local.tee requires top cached value".into()))?;
    match planned_kind {
        PlannedOpKind::LocalTee {
            local: PlannedLocal::Hot(reg),
        } => {
            state.ops.push(LirInst {
                kind: LirInstKind::WriteHotLocal { reg: *reg, src },
            });
        }
        PlannedOpKind::LocalTee {
            local: PlannedLocal::Frame(frame_slot),
        } => {
            state.ops.push(LirInst {
                kind: LirInstKind::WriteFrameLocal {
                    frame_slot: *frame_slot,
                    src,
                },
            });
        }
        _ => {
            return Err(WasmError::internal(
                "local.tee lowering expected planned local.tee".into(),
            ));
        }
    }
    Ok(())
}

pub(super) fn lower_call_external(
    func_idx: u32,
    params: u16,
    results: u16,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    config: PlanConfig,
    values: &mut ValueAlloc,
) {
    let args = materialize_top_values(state, params as usize, frame, values);
    consume_top(state, params as usize);
    let result_values = values.many(results as usize);
    state.ops.push(LirInst {
        kind: LirInstKind::CallExternal {
            func_idx,
            args,
            results: result_values.clone(),
        },
    });
    push_results(state, result_values, frame, config, values);
}

pub(super) fn lower_call_internal(
    callee: u32,
    params: u16,
    results: u16,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    config: PlanConfig,
    values: &mut ValueAlloc,
) {
    let args = materialize_top_values(state, params as usize, frame, values);
    consume_top(state, params as usize);
    let result_values = values.many(results as usize);
    state.ops.push(LirInst {
        kind: LirInstKind::CallInternal {
            callee,
            args,
            results: result_values.clone(),
        },
    });
    push_results(state, result_values, frame, config, values);
}

pub(super) fn lower_call_indirect(
    type_idx: u32,
    table_idx: u32,
    params: u16,
    results: u16,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    config: PlanConfig,
    values: &mut ValueAlloc,
) {
    let full = materialize_top_values(state, params as usize + 1, frame, values);
    let index = *full.last().expect("call_indirect index present");
    let args = full[..full.len().saturating_sub(1)].to_vec();
    consume_top(state, params as usize + 1);
    let result_values = values.many(results as usize);
    state.ops.push(LirInst {
        kind: LirInstKind::CallIndirect {
            type_idx,
            table_idx,
            index,
            args,
            results: result_values.clone(),
        },
    });
    push_results(state, result_values, frame, config, values);
}
