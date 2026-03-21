//! Straight-line prepared LIR op lowering.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::{
    error::WasmError,
    value_type::ValueType,
    vm::{
        lir::{
            ir::{LirBoundaryOp, LirInst, LirInstKind},
            leaf::LirLeafOp,
        },
        plan::frame::{FrameLayoutPlan, FrameSpan},
        wasm::{primitive_op, primitive_op::PrimitiveOpKind, semantic_ir::SemanticOpKind},
    },
};

use super::{
    state::{BlockState, ValueAlloc},
    steps::{PrepAction, PreparedOp},
};

pub(super) fn lower_prefix_actions(
    op: &PreparedOp<'_>,
    state: &mut BlockState,
    values: &mut ValueAlloc,
) -> Result<(), WasmError> {
    for action in &op.prefix {
        match action {
            PrepAction::Spill(frame) => {
                let spilled = state.spill_prefix(frame.count)?;
                for (offset, src) in spilled.into_iter().enumerate() {
                    state.ops.push(LirInst {
                        kind: LirInstKind::StoreSlot {
                            slot: frame.start.advance(offset as u16),
                            src,
                        },
                    });
                }
            }
            PrepAction::Fill(frame, fill_types) => {
                let mut reloaded = alloc::vec::Vec::with_capacity(frame.count as usize);
                for offset in 0..frame.count as usize {
                    let ty = fill_types.get(offset).copied();
                    debug_assert!(
                        ty.is_some(),
                        "fill offset {} has no type in fill_types (len={})",
                        offset,
                        fill_types.len(),
                    );
                    let ty = ty.unwrap_or(ValueType::I64);
                    let dst = values.fresh_typed(ty);
                    state.ops.push(LirInst {
                        kind: LirInstKind::LoadSlot {
                            slot: frame.start.advance(offset as u16),
                            dst,
                        },
                    });
                    reloaded.push(dst);
                }
                state.fill_prefix(reloaded, fill_types.clone())?;
            }
        }
    }
    Ok(())
}

pub(super) fn lower_primitive(
    kind: &crate::vm::wasm::primitive_op::PrimitiveOpKind,
    semantic_index: usize,
    state: &mut BlockState,
    values: &mut ValueAlloc,
    op_result_types: &BTreeMap<usize, Vec<ValueType>>,
) -> Result<(), WasmError> {
    let (pop, push) = primitive_op::stack_effect(kind);
    let args = state.top_values(pop as usize).map_err(|err| {
        WasmError::internal(alloc::format!(
            "prepared primitive {:?} could not read {} live operands: {}",
            kind,
            pop,
            err
        ))
    })?;
    let result_ty = if push == 0 {
        ValueType::I64 // unused — no LirValue created
    } else if matches!(kind, PrimitiveOpKind::Select) {
        args.first().map(|v| values.value_type(*v)).unwrap_or_else(|| {
            debug_assert!(false, "select has no on_true operand to derive type from");
            ValueType::I32
        })
    } else if let Some(ty) = primitive_op::result_type(kind) {
        ty
    } else {
        match op_result_types.get(&semantic_index).and_then(|v| v.first().copied()) {
            Some(ty) => ty,
            None => {
                return Err(WasmError::internal(alloc::format!(
                    "context-dependent primitive {:?} at op {} has no op_result_types entry",
                    kind, semantic_index,
                )));
            }
        }
    };
    state.consume_top(pop as usize)?;
    let results: alloc::vec::Vec<_> = (0..push).map(|_| values.fresh_typed(result_ty)).collect();
    state.ops.push(LirInst {
        kind: LirInstKind::Value {
            op: LirLeafOp::from_primitive(kind.clone())
                .expect("non-boundary primitive must lower as a leaf op"),
            args,
            results: results.clone(),
        },
    });
    state.push_results(results, alloc::vec![result_ty; push as usize])
}

pub(super) fn lower_boundary_primitive(
    kind: &PrimitiveOpKind,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
) -> Result<(), WasmError> {
    if !state.live().is_empty() {
        return Err(WasmError::internal(
            "boundary primitive reached LIR lowering with live transient SSA values; preparation must spill all before the boundary".into(),
        ));
    }
    let (pop, push) = primitive_op::stack_effect(kind);
    let boundary_base = call_base_slot(frame, state.height(), pop as u16);
    let boundary = match kind {
        PrimitiveOpKind::MemoryGrow { mem_idx } => LirBoundaryOp::MemoryGrow {
            mem_idx: *mem_idx,
            io: FrameSpan::new(boundary_base, 1),
        },
        PrimitiveOpKind::MemoryFill { imm0, .. } => LirBoundaryOp::MemoryFill {
            mem_idx: *imm0,
            args: FrameSpan::new(boundary_base, 3),
        },
        PrimitiveOpKind::MemoryCopy { imm0, imm1 } => LirBoundaryOp::MemoryCopy {
            dst_mem_idx: *imm0,
            src_mem_idx: *imm1,
            args: FrameSpan::new(boundary_base, 3),
        },
        PrimitiveOpKind::TableGrow { table_idx } => LirBoundaryOp::TableGrow {
            table_idx: *table_idx,
            args: FrameSpan::new(boundary_base, 2),
            results: FrameSpan::new(boundary_base, 1),
        },
        PrimitiveOpKind::TableFill { imm0, .. } => LirBoundaryOp::TableFill {
            table_idx: *imm0,
            args: FrameSpan::new(boundary_base, 3),
        },
        PrimitiveOpKind::TableCopy { imm0, imm1 } => LirBoundaryOp::TableCopy {
            dst_table_idx: *imm0,
            src_table_idx: *imm1,
            args: FrameSpan::new(boundary_base, 3),
        },
        PrimitiveOpKind::MemoryInit { imm0, imm1 } => LirBoundaryOp::MemoryInit {
            data_idx: *imm1,
            mem_idx: *imm0,
            args: FrameSpan::new(boundary_base, 3),
        },
        PrimitiveOpKind::DataDrop { data_idx } => LirBoundaryOp::DataDrop {
            data_idx: *data_idx,
        },
        PrimitiveOpKind::TableInit { imm0, imm1 } => LirBoundaryOp::TableInit {
            elem_idx: *imm1,
            table_idx: *imm0,
            args: FrameSpan::new(boundary_base, 3),
        },
        PrimitiveOpKind::ElemDrop { elem_idx } => LirBoundaryOp::ElemDrop {
            elem_idx: *elem_idx,
        },
        _ => {
            return Err(WasmError::internal(
                "non-boundary primitive reached boundary lowering".into(),
            ))
        }
    };
    state.ops.push(LirInst {
        kind: LirInstKind::Boundary(boundary),
    });
    state.finish_boundary(pop as u16, push as u16);
    Ok(())
}

pub(super) fn lower_local_get(
    local_idx: u16,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    values: &mut ValueAlloc,
    local_types: &[ValueType],
) -> Result<(), WasmError> {
    let ty = if local_types.is_empty() {
        ValueType::I64 // untyped path (no legalization)
    } else {
        *local_types.get(local_idx as usize).ok_or_else(|| {
            WasmError::internal(alloc::format!(
                "local {} has no entry in local_types (len={})",
                local_idx, local_types.len(),
            ))
        })?
    };
    let dst = values.fresh_typed(ty);
    state.ops.push(LirInst {
        kind: LirInstKind::LoadSlot {
            slot: frame.local_slot(local_idx),
            dst,
        },
    });
    state.push_results(alloc::vec![dst], alloc::vec![ty])
}

pub(super) fn lower_local_set(
    local_idx: u16,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
) -> Result<(), WasmError> {
    let src = state.pop_one()?;
    state.ops.push(LirInst {
        kind: LirInstKind::StoreSlot {
            slot: frame.local_slot(local_idx),
            src,
        },
    });
    Ok(())
}

pub(super) fn lower_local_tee(
    local_idx: u16,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    values: &mut ValueAlloc,
    local_types: &[ValueType],
) -> Result<(), WasmError> {
    // Pop the value, store it, then reload from the slot to produce a fresh
    // single-use value. This maintains the linear-SSA invariant: every LIR
    // value is used exactly once.
    let src = state.pop_one()?;
    let slot = frame.local_slot(local_idx);
    state.ops.push(LirInst {
        kind: LirInstKind::StoreSlot { slot, src },
    });
    let ty = local_types
        .get(local_idx as usize)
        .copied()
        .unwrap_or(ValueType::I64);
    let dst = values.fresh_typed(ty);
    state.ops.push(LirInst {
        kind: LirInstKind::LoadSlot { slot, dst },
    });
    state.push_results(alloc::vec![dst], alloc::vec![ty])
}

pub(super) fn lower_call_external(
    func_idx: u32,
    params: u16,
    results: u16,
    frame: FrameLayoutPlan,
    state: &mut BlockState,
    skip_reload: Vec<bool>,
) {
    let call_base = call_base_slot(frame, state.height(), params);
    state.ops.push(LirInst {
        kind: LirInstKind::Boundary(LirBoundaryOp::CallExternal {
            func_idx,
            args: FrameSpan::new(call_base, params),
            results: FrameSpan::new(call_base, results),
            skip_reload,
        }),
    });
    state.finish_boundary(params, results);
}

pub(super) fn lower_call_internal(
    callee: u32,
    params: u16,
    results: u16,
    frame: FrameLayoutPlan,
    state: &mut BlockState,
    skip_reload: Vec<bool>,
) {
    let call_base = call_base_slot(frame, state.height(), params);
    state.ops.push(LirInst {
        kind: LirInstKind::Boundary(LirBoundaryOp::CallInternal {
            callee,
            args: FrameSpan::new(call_base, params),
            results: FrameSpan::new(call_base, results),
            skip_reload,
        }),
    });
    state.finish_boundary(params, results);
}

pub(super) fn lower_call_indirect(
    type_idx: u32,
    table_idx: u32,
    params: u16,
    results: u16,
    frame: FrameLayoutPlan,
    state: &mut BlockState,
    skip_reload: Vec<bool>,
) {
    let consumed = params.saturating_add(1);
    let call_base = call_base_slot(frame, state.height(), consumed);
    state.ops.push(LirInst {
        kind: LirInstKind::Boundary(LirBoundaryOp::CallIndirect {
            type_idx,
            table_idx,
            index_slot: call_base.advance(params),
            args: FrameSpan::new(call_base, params),
            results: FrameSpan::new(call_base, results),
            skip_reload,
        }),
    });
    state.finish_boundary(consumed, results);
}

#[inline]
pub(super) fn branch_payload(
    frame: FrameLayoutPlan,
    current_height: u16,
    stack_drop: u32,
    arity: u16,
) -> Option<FrameSpan> {
    if arity == 0 {
        None
    } else {
        let base = current_height
            .saturating_sub(stack_drop as u16)
            .saturating_sub(arity);
        Some(FrameSpan::new(frame.operand_slot(base), arity))
    }
}

#[inline]
pub(super) fn call_results(frame: FrameLayoutPlan, results: u16) -> FrameSpan {
    FrameSpan::new(frame.operand_slot(0), results)
}

#[inline]
fn call_base_slot(
    frame: FrameLayoutPlan,
    stack_height: u16,
    consumed: u16,
) -> crate::vm::plan::frame::FrameSlot {
    // Calls consume the current canonical top-of-stack window. The argument and
    // result spans therefore start at the stack position that will remain after
    // the consumed inputs are removed.
    frame.operand_slot(stack_height.saturating_sub(consumed))
}

#[inline]
pub(super) fn return_results(frame: FrameLayoutPlan, results: u16) -> Option<FrameSpan> {
    (results != 0).then(|| call_results(frame, results))
}

pub(super) fn lower_block_body_op(
    op: &PreparedOp<'_>,
    semantic_index: usize,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    values: &mut ValueAlloc,
    local_types: &[ValueType],
    op_result_types: &BTreeMap<usize, Vec<ValueType>>,
    skip_reload_iter: &mut dyn Iterator<Item = Vec<bool>>,
) -> Result<(), WasmError> {
    lower_prefix_actions(op, state, values)?;

    match &op.semantic.kind {
        SemanticOpKind::Primitive(kind)
            if matches!(
                kind,
                crate::vm::wasm::primitive_op::PrimitiveOpKind::Unreachable
            ) =>
        {
            Err(WasmError::internal(
                "unreachable must end a prepared LIR block, not appear in the body".into(),
            ))
        }
        SemanticOpKind::Primitive(kind) => {
            if crate::vm::lir::leaf::is_boundary_primitive(kind) {
                lower_boundary_primitive(kind, state, frame)
            } else {
                lower_primitive(kind, semantic_index, state, values, op_result_types)
            }
        }
        SemanticOpKind::LocalGet { idx } => {
            lower_local_get(*idx, state, frame, values, local_types)
        }
        SemanticOpKind::LocalSet { idx } => lower_local_set(*idx, state, frame),
        SemanticOpKind::LocalTee { idx } => {
            lower_local_tee(*idx, state, frame, values, local_types)
        }
        SemanticOpKind::CallExternal {
            func_idx,
            params,
            results,
        } => {
            let skip = skip_reload_iter.next().unwrap_or_default();
            lower_call_external(*func_idx, *params, *results, frame, state, skip);
            Ok(())
        }
        SemanticOpKind::CallInternal {
            callee,
            params,
            results,
        } => {
            let skip = skip_reload_iter.next().unwrap_or_default();
            lower_call_internal(*callee, *params, *results, frame, state, skip);
            Ok(())
        }
        SemanticOpKind::CallIndirect {
            type_idx,
            table_idx,
            params,
            results,
        } => {
            let skip = skip_reload_iter.next().unwrap_or_default();
            lower_call_indirect(*type_idx, *table_idx, *params, *results, frame, state, skip);
            Ok(())
        }
        SemanticOpKind::Block { .. } | SemanticOpKind::Loop { .. } | SemanticOpKind::End => Ok(()),
        SemanticOpKind::Else { .. } => Err(WasmError::internal(
            "else must end a prepared LIR block, not appear in the body".into(),
        )),
        SemanticOpKind::If { .. }
        | SemanticOpKind::Br { .. }
        | SemanticOpKind::BrIf { .. }
        | SemanticOpKind::BrTable { .. }
        | SemanticOpKind::ReturnVoid
        | SemanticOpKind::ReturnOne
        | SemanticOpKind::Return { .. } => Err(WasmError::internal(
            "control-flow terminators must end a prepared LIR block".into(),
        )),
        SemanticOpKind::Legalized(ref leg) => {
            let lir_op = crate::vm::lir::legalized::from_semantic(leg);
            let (pops, pushes) = lir_op.stack_effect();
            let args = state.top_values(pops as usize).map_err(|err| {
                WasmError::internal(alloc::format!(
                    "prepared legalized op {:?} could not read {} live operands: {}",
                    lir_op,
                    pops,
                    err
                ))
            })?;
            state.consume_top(pops as usize)?;
            let result_types: alloc::vec::Vec<_> = (0..pushes)
                .map(|i| lir_op.result_type(i))
                .collect();
            let results: alloc::vec::Vec<_> = result_types
                .iter()
                .map(|ty| values.fresh_typed(*ty))
                .collect();
            state.ops.push(LirInst {
                kind: LirInstKind::Legalized {
                    op: lir_op,
                    args,
                    results: results.clone(),
                },
            });
            state.push_results(results, result_types)
        }
    }
}
