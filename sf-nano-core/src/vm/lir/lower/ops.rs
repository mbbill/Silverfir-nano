//! Block-body op lowering into LIR instructions.

use crate::error::WasmError;
use crate::vm::{
    lir::{
        ir::{LirInst, LirInstKind},
        leaf::LirLeafOp,
    },
    plan::{
        frame::FrameLayoutPlan,
        tos::{SpillArtifact, TosTransferDirection},
        PlannedHotLocalInit, PlannedLocal, PlannedOp, PlannedOpKind,
    },
    wasm::primitive_op::stack_effect,
};

use super::{
    stack::{consume_top, materialize_stack_index, materialize_top_values, pop_one, push_results},
    state::{BlockState, StackValue, ValueAlloc},
};

pub(super) fn lower_primitive(
    kind: &crate::vm::wasm::primitive_op::PrimitiveOpKind,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
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
    push_results(state, results);
}

pub(super) fn lower_local_get(
    planned_kind: &PlannedOpKind,
    state: &mut BlockState,
    _frame: FrameLayoutPlan,
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
    push_results(state, alloc::vec![dst]);
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
    frame: FrameLayoutPlan,
    values: &mut ValueAlloc,
) -> Result<(), WasmError> {
    let top_index = state
        .stack
        .len()
        .checked_sub(1)
        .ok_or_else(|| WasmError::internal("local.tee requires a stack value".into()))?;
    let src = materialize_stack_index(state, frame, values, top_index as u16);
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
    push_results(state, result_values);
}

pub(super) fn lower_call_internal(
    callee: u32,
    params: u16,
    results: u16,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
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
    push_results(state, result_values);
}

pub(super) fn lower_call_indirect(
    type_idx: u32,
    table_idx: u32,
    params: u16,
    results: u16,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
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
    push_results(state, result_values);
}

pub(super) fn lower_synthetic_prefix_op(
    planned: &PlannedOp,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    values: &mut ValueAlloc,
) -> Result<(), WasmError> {
    match &planned.kind {
        PlannedOpKind::InitHotLocals { inits } => {
            for init in inits {
                lower_hot_local_init(*init, state, values);
            }
            Ok(())
        }
        PlannedOpKind::Spill(artifact) => lower_spill_artifact(*artifact, state, frame, values),
        _ => Err(WasmError::internal(
            "lower_synthetic_prefix_op expected a synthetic planned op".into(),
        )),
    }
}

fn lower_hot_local_init(
    init: PlannedHotLocalInit,
    state: &mut BlockState,
    values: &mut ValueAlloc,
) {
    let value = values.fresh();
    state.ops.push(LirInst {
        kind: LirInstKind::ReadFrameLocal {
            frame_slot: init.frame_slot,
            dst: value,
        },
    });
    state.ops.push(LirInst {
        kind: LirInstKind::WriteHotLocal {
            reg: init.reg,
            src: value,
        },
    });
}

fn lower_spill_artifact(
    artifact: SpillArtifact,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    values: &mut ValueAlloc,
) -> Result<(), WasmError> {
    match artifact {
        SpillArtifact::TosTransfer(transfer) => {
            for offset in 0..transfer.frame.count {
                let slot = transfer.frame.start.advance(offset);
                let stack_index = slot.0.checked_sub(frame.operand_base).ok_or_else(|| {
                    WasmError::internal("spill/fill slot must be an operand slot".into())
                })? as usize;
                let current = *state.stack.get(stack_index).ok_or_else(|| {
                    WasmError::internal(
                        "spill/fill stack index out of range during LIR lowering".into(),
                    )
                })?;

                match (transfer.direction, current) {
                    (TosTransferDirection::Spill, StackValue::Ssa(value)) => {
                        state.ops.push(LirInst {
                            kind: LirInstKind::WriteOperandSlot { slot, src: value },
                        });
                        state.stack[stack_index] = StackValue::Frame(slot);
                    }
                    (TosTransferDirection::Spill, StackValue::Frame(existing)) => {
                        if existing != slot {
                            return Err(WasmError::internal(
                                "spill artifact disagrees with current operand-slot state".into(),
                            ));
                        }
                    }
                    (TosTransferDirection::Fill, StackValue::Frame(existing)) => {
                        if existing != slot {
                            return Err(WasmError::internal(
                                "fill artifact disagrees with current operand-slot state".into(),
                            ));
                        }
                        let value = values.fresh();
                        state.ops.push(LirInst {
                            kind: LirInstKind::ReadOperandSlot { slot, dst: value },
                        });
                        state.stack[stack_index] = StackValue::Ssa(value);
                    }
                    (TosTransferDirection::Fill, StackValue::Ssa(_)) => {}
                }
            }
            Ok(())
        }
    }
}
