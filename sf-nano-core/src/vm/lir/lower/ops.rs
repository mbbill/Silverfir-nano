//! Prepared block-body op lowering into LIR instructions.

use crate::error::WasmError;
use crate::vm::{
    lir::{
        ir::{LirInst, LirInstKind},
        leaf::LirLeafOp,
        runtime::LirRuntimeOp,
    },
    plan::{
        frame::{FrameLayoutPlan, FrameSpan},
        tos::{SpillArtifact, TosTransferDirection},
        PlannedOp, PlannedOpKind,
    },
    wasm::primitive_op::stack_effect,
};

use super::{
    input::AlignedOp,
    state::{BlockState, ValueAlloc},
};

pub(super) fn lower_prefix_ops(
    op: &AlignedOp<'_>,
    state: &mut BlockState,
    values: &mut ValueAlloc,
) -> Result<(), WasmError> {
    for planned in &op.prefix {
        lower_synthetic(planned, state, values)?;
    }
    Ok(())
}

fn lower_synthetic(
    planned: &PlannedOp,
    state: &mut BlockState,
    values: &mut ValueAlloc,
) -> Result<(), WasmError> {
    match planned.kind {
        PlannedOpKind::InitHotLocals { .. } => Ok(()),
        PlannedOpKind::Spill(artifact) => lower_tos_transfer(artifact, state, values),
        _ => Err(WasmError::internal(
            "prepared LIR synthetic lowering expected spill/fill artifact".into(),
        )),
    }
}

fn lower_tos_transfer(
    artifact: SpillArtifact,
    state: &mut BlockState,
    values: &mut ValueAlloc,
) -> Result<(), WasmError> {
    match artifact {
        SpillArtifact::TosTransfer(transfer) => match transfer.direction {
            TosTransferDirection::Spill => {
                let spilled = state.spill_prefix(transfer.frame.count)?;
                for (offset, src) in spilled.into_iter().enumerate() {
                    state.ops.push(LirInst {
                        kind: LirInstKind::Spill {
                            slot: transfer.frame.start.advance(offset as u16),
                            src,
                        },
                    });
                }
                Ok(())
            }
            TosTransferDirection::Fill => {
                let mut reloaded = alloc::vec::Vec::with_capacity(transfer.frame.count as usize);
                for offset in 0..transfer.frame.count as usize {
                    let dst = values.fresh();
                    state.ops.push(LirInst {
                        kind: LirInstKind::Fill {
                            slot: transfer.frame.start.advance(offset as u16),
                            dst,
                        },
                    });
                    reloaded.push(dst);
                }
                state.fill_prefix(reloaded)
            }
        },
    }
}

pub(super) fn lower_primitive(
    kind: &crate::vm::wasm::primitive_op::PrimitiveOpKind,
    state: &mut BlockState,
    values: &mut ValueAlloc,
) -> Result<(), WasmError> {
    let (pop, push) = stack_effect(kind);
    let args = state.top_values(pop as usize)?;
    state.consume_top(pop as usize)?;
    let results = values.many(push as usize);
    let kind = if let Some(op) = LirRuntimeOp::from_primitive(kind.clone()) {
        LirInstKind::Runtime {
            op,
            args,
            results: results.clone(),
        }
    } else {
        LirInstKind::Leaf {
            op: LirLeafOp::from_primitive(kind.clone())
                .expect("non-runtime primitive must lower as a leaf op"),
            args,
            results: results.clone(),
        }
    };
    state.ops.push(LirInst { kind });
    state.push_results(results)
}

pub(super) fn lower_local_get(
    local_idx: u32,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    values: &mut ValueAlloc,
) -> Result<(), WasmError> {
    let dst = values.fresh();
    state.ops.push(LirInst {
        kind: LirInstKind::ReadSlot {
            slot: frame.local_slot(local_idx as u16),
            dst,
        },
    });
    state.push_results(alloc::vec![dst])
}

pub(super) fn lower_local_set(
    local_idx: u32,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
) -> Result<(), WasmError> {
    let src = state.pop_one()?;
    state.ops.push(LirInst {
        kind: LirInstKind::WriteSlot {
            slot: frame.local_slot(local_idx as u16),
            src,
        },
    });
    Ok(())
}

pub(super) fn lower_local_tee(
    local_idx: u32,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
) -> Result<(), WasmError> {
    let src = *state
        .tos()
        .last()
        .ok_or_else(|| WasmError::internal("local.tee requires a live TOS value".into()))?;
    state.ops.push(LirInst {
        kind: LirInstKind::WriteSlot {
            slot: frame.local_slot(local_idx as u16),
            src,
        },
    });
    Ok(())
}

pub(super) fn lower_call_external(
    func_idx: u32,
    args: FrameSpan,
    results: FrameSpan,
    state: &mut BlockState,
) {
    state.ops.push(LirInst {
        kind: LirInstKind::CallExternal {
            func_idx,
            args,
            results,
        },
    });
    state.finish_call(args.count, results.count);
}

pub(super) fn lower_call_internal(
    callee: u32,
    args: FrameSpan,
    results: FrameSpan,
    state: &mut BlockState,
) {
    state.ops.push(LirInst {
        kind: LirInstKind::CallInternal {
            callee,
            args,
            results,
        },
    });
    state.finish_call(args.count, results.count);
}

pub(super) fn lower_call_indirect(
    type_idx: u32,
    table_idx: u32,
    index_slot: crate::vm::plan::frame::FrameSlot,
    args: FrameSpan,
    results: FrameSpan,
    state: &mut BlockState,
) {
    state.ops.push(LirInst {
        kind: LirInstKind::CallIndirect {
            type_idx,
            table_idx,
            index_slot,
            args,
            results,
        },
    });
    state.finish_call(args.count.saturating_add(1), results.count);
}
