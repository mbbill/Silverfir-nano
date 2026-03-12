//! Block-body op lowering into LIR instructions.

use crate::error::WasmError;
use crate::vm::{
    lir::{
        ir::{LirInst, LirInstKind},
        leaf::LirLeafOp,
        runtime::LirRuntimeOp,
    },
    plan::frame::FrameLayoutPlan,
    wasm::primitive_op::stack_effect,
};

use super::state::{BlockState, ValueAlloc};

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
    state.push_results(results);
    Ok(())
}

pub(super) fn lower_local_get(
    local_idx: u32,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    values: &mut ValueAlloc,
) -> Result<(), WasmError> {
    let dst = values.fresh();
    let slot = frame.local_slot(local_idx as u16);
    state.ops.push(LirInst {
        kind: LirInstKind::ReadSlot { slot, dst },
    });
    state.push_results(alloc::vec![dst]);
    Ok(())
}

pub(super) fn lower_local_set(
    local_idx: u32,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
) -> Result<(), WasmError> {
    let src = state.pop_one()?;
    let slot = frame.local_slot(local_idx as u16);
    state.ops.push(LirInst {
        kind: LirInstKind::WriteSlot { slot, src },
    });
    Ok(())
}

pub(super) fn lower_local_tee(
    local_idx: u32,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
) -> Result<(), WasmError> {
    let top_index = state
        .values()
        .len()
        .checked_sub(1)
        .ok_or_else(|| WasmError::internal("local.tee requires a stack value".into()))?;
    let src = state.value_at(top_index)?;
    let slot = frame.local_slot(local_idx as u16);
    state.ops.push(LirInst {
        kind: LirInstKind::WriteSlot { slot, src },
    });
    Ok(())
}

pub(super) fn lower_call_external(
    func_idx: u32,
    params: u16,
    results: u16,
    state: &mut BlockState,
    values: &mut ValueAlloc,
) -> Result<(), WasmError> {
    let args = state.top_values(params as usize)?;
    state.consume_top(params as usize)?;
    let result_values = values.many(results as usize);
    state.ops.push(LirInst {
        kind: LirInstKind::CallExternal {
            func_idx,
            args,
            results: result_values.clone(),
        },
    });
    state.push_results(result_values);
    Ok(())
}

pub(super) fn lower_call_internal(
    callee: u32,
    params: u16,
    results: u16,
    state: &mut BlockState,
    values: &mut ValueAlloc,
) -> Result<(), WasmError> {
    let args = state.top_values(params as usize)?;
    state.consume_top(params as usize)?;
    let result_values = values.many(results as usize);
    state.ops.push(LirInst {
        kind: LirInstKind::CallInternal {
            callee,
            args,
            results: result_values.clone(),
        },
    });
    state.push_results(result_values);
    Ok(())
}

pub(super) fn lower_call_indirect(
    type_idx: u32,
    table_idx: u32,
    params: u16,
    results: u16,
    state: &mut BlockState,
    values: &mut ValueAlloc,
) -> Result<(), WasmError> {
    let full = state.top_values(params as usize + 1)?;
    let index = *full.last().expect("call_indirect index present");
    let args = full[..full.len().saturating_sub(1)].to_vec();
    state.consume_top(params as usize + 1)?;
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
    state.push_results(result_values);
    Ok(())
}
