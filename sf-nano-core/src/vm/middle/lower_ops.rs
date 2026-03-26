//! Straight-line prepared SSA-IR op lowering.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::{
    error::WasmError,
    value_type::ValueType,
    vm::{
        middle::{
            frame::{FrameLayoutPlan, FrameSpan},
            ssa_ir::{
                ir::{SsaBoundaryOp, SsaInst, SsaInstKind, SsaOperand},
                leaf::SsaLeafOp,
            },
        },
        wasm::{primitive_op::{self, PrimitiveOpKind}, semantic_ir::SemanticOpKind},
    },
};

use super::{
    state::{BlockState, ValueAlloc},
    spill_plan::{PrepAction, PreparedOp},
};

use crate::vm::middle::ssa_ir::ir::ValueHome;

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
                    state.ops.push(SsaInst {
                        kind: SsaInstKind::Spill {
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
                    state.ops.push(SsaInst {
                        kind: SsaInstKind::Fill {
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
        ValueType::I64 // unused — no SsaValue created
    } else if matches!(kind, PrimitiveOpKind::Select) {
        let ty = args.first().map(|v| values.value_type(*v));
        debug_assert!(
            ty.is_some(),
            "select has no on_true operand to derive type from"
        );
        ty.unwrap_or(ValueType::I64)
    } else if let Some(ty) = primitive_op::result_type(kind) {
        ty
    } else {
        let ty = op_result_types
            .get(&semantic_index)
            .and_then(|v| v.first().copied());
        debug_assert!(
            ty.is_some(),
            "context-dependent primitive {:?} at op {} has no op_result_types entry",
            kind,
            semantic_index,
        );
        ty.unwrap_or(ValueType::I64)
    };
    state.consume_top(pop as usize)?;
    let results: alloc::vec::Vec<_> = (0..push).map(|_| values.fresh_typed(result_ty)).collect();
    state.ops.push(SsaInst {
        kind: SsaInstKind::Value {
            op: SsaLeafOp::from_primitive(kind.clone())
                .expect("non-boundary primitive must lower as a leaf op"),
            args: args.into_iter().map(SsaOperand::Value).collect(),
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
            "boundary primitive reached SSA-IR lowering with live transient SSA values; preparation must spill all before the boundary".into(),
        ));
    }
    let (pop, push) = primitive_op::stack_effect(kind);
    let boundary_base = call_base_slot(frame, state.height(), pop as u16);
    let boundary = match kind {
        PrimitiveOpKind::MemoryGrow { mem_idx } => SsaBoundaryOp::MemoryGrow {
            mem_idx: *mem_idx,
            io: FrameSpan::new(boundary_base, 1),
        },
        PrimitiveOpKind::MemoryFill { imm0, .. } => SsaBoundaryOp::MemoryFill {
            mem_idx: *imm0,
            args: FrameSpan::new(boundary_base, 3),
        },
        PrimitiveOpKind::MemoryCopy { imm0, imm1 } => SsaBoundaryOp::MemoryCopy {
            dst_mem_idx: *imm0,
            src_mem_idx: *imm1,
            args: FrameSpan::new(boundary_base, 3),
        },
        PrimitiveOpKind::TableGrow { table_idx } => SsaBoundaryOp::TableGrow {
            table_idx: *table_idx,
            args: FrameSpan::new(boundary_base, 2),
            results: FrameSpan::new(boundary_base, 1),
        },
        PrimitiveOpKind::TableFill { imm0, .. } => SsaBoundaryOp::TableFill {
            table_idx: *imm0,
            args: FrameSpan::new(boundary_base, 3),
        },
        PrimitiveOpKind::TableCopy { imm0, imm1 } => SsaBoundaryOp::TableCopy {
            dst_table_idx: *imm0,
            src_table_idx: *imm1,
            args: FrameSpan::new(boundary_base, 3),
        },
        PrimitiveOpKind::MemoryInit { imm0, imm1 } => SsaBoundaryOp::MemoryInit {
            data_idx: *imm1,
            mem_idx: *imm0,
            args: FrameSpan::new(boundary_base, 3),
        },
        PrimitiveOpKind::DataDrop { data_idx } => SsaBoundaryOp::DataDrop {
            data_idx: *data_idx,
        },
        PrimitiveOpKind::TableInit { imm0, imm1 } => SsaBoundaryOp::TableInit {
            elem_idx: *imm1,
            table_idx: *imm0,
            args: FrameSpan::new(boundary_base, 3),
        },
        PrimitiveOpKind::ElemDrop { elem_idx } => SsaBoundaryOp::ElemDrop {
            elem_idx: *elem_idx,
        },
        _ => {
            return Err(WasmError::internal(
                "non-boundary primitive reached boundary lowering".into(),
            ))
        }
    };
    state.ops.push(SsaInst {
        kind: SsaInstKind::Boundary(boundary),
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
    local_versions: &[u32],
) -> Result<(), WasmError> {
    let ty = local_types.get(local_idx as usize).copied();
    debug_assert!(
        ty.is_some() || local_types.is_empty(),
        "local {} has no entry in local_types (len={})",
        local_idx,
        local_types.len(),
    );
    let ty = ty.unwrap_or(ValueType::I64);
    let dst = values.fresh_typed(ty);
    let slot = frame.local_slot(local_idx);
    let version = local_versions.get(local_idx as usize).copied().unwrap_or(0);
    values.set_value_home(dst, ValueHome::LocalVersion { slot, version });
    state.ops.push(SsaInst {
        kind: SsaInstKind::LocalGet { slot, dst },
    });
    state.push_results(alloc::vec![dst], alloc::vec![ty])
}

pub(super) fn lower_local_set(
    local_idx: u16,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    local_versions: &mut [u32],
) -> Result<(), WasmError> {
    let src = state.pop_one()?;
    let slot = frame.local_slot(local_idx);
    if let Some(v) = local_versions.get_mut(local_idx as usize) {
        *v += 1;
    }
    let version = local_versions.get(local_idx as usize).copied().unwrap_or(0);
    state.ops.push(SsaInst {
        kind: SsaInstKind::LocalSet { slot, src, version },
    });
    Ok(())
}

pub(super) fn lower_local_tee(
    local_idx: u16,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    values: &mut ValueAlloc,
    local_types: &[ValueType],
    local_versions: &mut [u32],
) -> Result<(), WasmError> {
    // Pop the value, store it, then reload from the slot to produce a fresh
    // single-use value. This maintains the linear-SSA invariant: every SSA-IR
    // value is used exactly once.
    let src = state.pop_one()?;
    let slot = frame.local_slot(local_idx);
    if let Some(v) = local_versions.get_mut(local_idx as usize) {
        *v += 1;
    }
    let version = local_versions.get(local_idx as usize).copied().unwrap_or(0);
    state.ops.push(SsaInst {
        kind: SsaInstKind::LocalSet { slot, src, version },
    });
    let ty = local_types
        .get(local_idx as usize)
        .copied()
        .unwrap_or(ValueType::I64);
    let dst = values.fresh_typed(ty);
    values.set_value_home(dst, ValueHome::LocalVersion { slot, version });
    state.ops.push(SsaInst {
        kind: SsaInstKind::LocalGet { slot, dst },
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
    state.ops.push(SsaInst {
        kind: SsaInstKind::Boundary(SsaBoundaryOp::CallExternal {
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
    state.ops.push(SsaInst {
        kind: SsaInstKind::Boundary(SsaBoundaryOp::CallInternal {
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
    state.ops.push(SsaInst {
        kind: SsaInstKind::Boundary(SsaBoundaryOp::CallIndirect {
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
) -> crate::vm::middle::frame::FrameSlot {
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
    local_versions: &mut [u32],
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
                "unreachable must end a prepared SSA-IR block, not appear in the body".into(),
            ))
        }
        SemanticOpKind::Primitive(kind) => {
            if crate::vm::middle::ssa_ir::leaf::is_boundary_primitive(kind) {
                lower_boundary_primitive(kind, state, frame)
            } else {
                lower_primitive(kind, semantic_index, state, values, op_result_types)
            }
        }
        SemanticOpKind::LocalGet { idx } => {
            lower_local_get(*idx, state, frame, values, local_types, local_versions)
        }
        SemanticOpKind::LocalSet { idx } => lower_local_set(*idx, state, frame, local_versions),
        SemanticOpKind::LocalTee { idx } => {
            lower_local_tee(*idx, state, frame, values, local_types, local_versions)
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
            "else must end a prepared SSA-IR block, not appear in the body".into(),
        )),
        SemanticOpKind::If { .. }
        | SemanticOpKind::Br { .. }
        | SemanticOpKind::BrIf { .. }
        | SemanticOpKind::BrTable { .. }
        | SemanticOpKind::ReturnVoid
        | SemanticOpKind::ReturnOne
        | SemanticOpKind::Return { .. } => Err(WasmError::internal(
            "control-flow terminators must end a prepared SSA-IR block".into(),
        )),
    }
}
