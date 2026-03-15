//! Internal semantic preparation steps.
//!
//! This is private preparation state, not a public IR layer.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::{
    error::WasmError,
    value_type::ValueType,
    vm::{
        plan::{
            config::PlanConfig,
            frame::{FrameLayoutPlan, FrameSpan},
        },
        wasm::{
            primitive_op,
            primitive_op::PrimitiveOpKind,
            semantic_ir::{SemanticOp, SemanticOpKind, SemanticProgram},
        },
    },
};

use super::state::EntryState;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum PrepAction {
    Spill(FrameSpan),
    /// Fill from frame slots, with the value type for each reloaded entry.
    Fill(FrameSpan, Vec<ValueType>),
}

#[derive(Clone, Debug)]
pub(super) struct PreparedOp<'a> {
    pub(super) semantic: &'a SemanticOp,
    pub(super) prefix: Vec<PrepAction>,
}

#[derive(Clone, Debug)]
pub(super) struct PreparedStream<'a> {
    pub(super) ops: Vec<PreparedOp<'a>>,
    pub(super) entry_states: Vec<EntryState>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ControlFrameKind {
    Function,
    Structured,
}

#[derive(Clone, Debug)]
struct ControlFrame {
    kind: ControlFrameKind,
    start_height: u16,
    params: u16,
    results: u16,
    entered_unreachable: bool,
    /// Saved types at `[start_height .. start_height + params]` so that
    /// Else can restore the correct param types after the then-arm may
    /// have overwritten them.
    param_types: Vec<ValueType>,
}

#[derive(Clone, Debug)]
struct PrepareState {
    height: u16,
    spill_depth: u16,
    unreachable: bool,
    control: Vec<ControlFrame>,
    /// Per-stack-position value type, parallel to the conceptual Wasm value stack.
    type_stack: Vec<ValueType>,
    /// Local types (params ++ locals), borrowed from SemanticProgram.
    local_types: Vec<ValueType>,
}

impl PrepareState {
    fn new(results: u16, local_types: Vec<ValueType>) -> Self {
        Self {
            height: 0,
            spill_depth: 0,
            unreachable: false,
            control: alloc::vec![ControlFrame {
                kind: ControlFrameKind::Function,
                start_height: 0,
                params: 0,
                results,
                entered_unreachable: false,
                param_types: Vec::new(),
            }],
            type_stack: Vec::new(),
            local_types,
        }
    }

    fn mark_unreachable(&mut self) {
        if let Some(frame) = self.control.last() {
            let target = frame.start_height.saturating_add(frame.results) as usize;
            self.height = target as u16;
            self.spill_depth = self.height;
            self.type_stack.truncate(target);
        }
        self.unreachable = true;
    }

    fn local_type(&self, idx: u16) -> ValueType {
        let ty = self.local_types.get(idx as usize).copied();
        debug_assert!(
            ty.is_some() || self.local_types.is_empty(),
            "local {} has no entry in local_types (len={})",
            idx,
            self.local_types.len(),
        );
        ty.unwrap_or(ValueType::I64)
    }

    /// Extract types for a range of the type stack.
    ///
    /// When the type stack is shorter than expected (e.g. after unreachable
    /// code truncation), missing entries are padded with I64. This is safe
    /// because values in phantom/unreachable regions never reach codegen.
    fn types_at(&self, start: u16, count: u16) -> Vec<ValueType> {
        if count == 0 {
            return Vec::new();
        }
        let start = start as usize;
        let end = start + count as usize;
        if end <= self.type_stack.len() {
            self.type_stack[start..end].to_vec()
        } else {
            (0..count as usize)
                .map(|i| {
                    self.type_stack
                        .get(start + i)
                        .copied()
                        .unwrap_or(ValueType::I64)
                })
                .collect()
        }
    }
}

pub(super) fn prepare_semantic_ops<'a>(
    semantic: &'a SemanticProgram,
    frame: FrameLayoutPlan,
    config: PlanConfig,
) -> Result<PreparedStream<'a>, WasmError> {
    let mut state = PrepareState::new(semantic.results, semantic.local_types.clone());
    let mut entry_states = Vec::with_capacity(semantic.ops.len());
    let mut ops = Vec::with_capacity(semantic.ops.len());

    for (op_index, semantic_op) in semantic.ops.iter().enumerate() {
        let live_count = state.height.saturating_sub(state.spill_depth);
        let live_types = if state.unreachable {
            Vec::new()
        } else {
            state.types_at(state.spill_depth, live_count)
        };
        entry_states.push(EntryState {
            stack_height: state.height,
            spill_depth: state.spill_depth,
            live_types,
        });
        let prefix = plan_prefix(
            semantic_op,
            op_index,
            &mut state,
            frame,
            config.gp_lir_lanes,
            config.fp_lir_lanes,
            &semantic.op_result_types,
        );
        ops.push(PreparedOp {
            semantic: semantic_op,
            prefix,
        });
        apply_semantic_effect(semantic_op, op_index, &semantic.op_result_types, &mut state);
    }

    Ok(PreparedStream { ops, entry_states })
}

fn plan_prefix(
    op: &SemanticOp,
    op_index: usize,
    state: &mut PrepareState,
    frame: FrameLayoutPlan,
    gp_lanes: u8,
    fp_lanes: u8,
    op_result_types: &BTreeMap<usize, Vec<ValueType>>,
) -> Vec<PrepAction> {
    let mut prefix = Vec::new();

    match &op.kind {
        SemanticOpKind::Primitive(kind) => {
            if matches!(kind, PrimitiveOpKind::Unreachable) {
                return prefix;
            }
            if crate::vm::lir::leaf::is_boundary_primitive(kind) {
                spill_all(&mut prefix, state, frame);
            } else {
                let (pop, push) = primitive_op::stack_effect(kind);
                if push > 0 {
                    let push_ty = if matches!(kind, PrimitiveOpKind::Select) {
                        state
                            .type_stack
                            .len()
                            .checked_sub(3)
                            .and_then(|idx| state.type_stack.get(idx).copied())
                            .or_else(|| primitive_result_type(kind, op_index, op_result_types))
                    } else {
                        primitive_result_type(kind, op_index, op_result_types)
                    };
                    spill_before_result_push(
                        &mut prefix,
                        state,
                        frame,
                        gp_lanes,
                        fp_lanes,
                        pop as u16,
                        push as u16,
                        push_ty.unwrap_or(ValueType::I64),
                    );
                }
                fill_for_operands(&mut prefix, state, frame, pop as u16);
            }
        }
        SemanticOpKind::LocalGet { idx } => {
            let push_ty = state.local_type(*idx);
            spill_before_result_push(&mut prefix, state, frame, gp_lanes, fp_lanes, 0, 1, push_ty);
        }
        SemanticOpKind::LocalSet { .. } | SemanticOpKind::LocalTee { .. } => {
            fill_for_operands(&mut prefix, state, frame, 1);
        }
        SemanticOpKind::Block { .. } => {}
        SemanticOpKind::Loop { .. } => spill_all(&mut prefix, state, frame),
        SemanticOpKind::If { params, .. } => {
            let keep_live = params.saturating_add(1);
            fill_for_operands(&mut prefix, state, frame, keep_live);
            spill_all_except_top(&mut prefix, state, frame, keep_live);
        }
        SemanticOpKind::Else { .. } | SemanticOpKind::End => {
            if let Some(frame_state) = state.control.last() {
                if frame_state.entered_unreachable {
                    return prefix;
                }
                if matches!(frame_state.kind, ControlFrameKind::Function) {
                    return prefix;
                }
                fill_for_operands_inner(&mut prefix, state, frame, frame_state.results, false);
            }
        }
        SemanticOpKind::Br { arity, .. } => {
            fill_for_operands(&mut prefix, state, frame, *arity);
            spill_all_except_top(&mut prefix, state, frame, *arity);
        }
        SemanticOpKind::BrIf { arity, .. } => {
            let keep_live = arity.saturating_add(1);
            fill_for_operands(&mut prefix, state, frame, keep_live);
            spill_all_except_top(&mut prefix, state, frame, keep_live);
        }
        SemanticOpKind::BrTable { entries } => {
            let arity = entries.first().map(|entry| entry.arity).unwrap_or(0);
            let keep_live = arity.saturating_add(1);
            fill_for_operands(&mut prefix, state, frame, keep_live);
            spill_all_except_top(&mut prefix, state, frame, keep_live);
        }
        SemanticOpKind::CallExternal { .. }
        | SemanticOpKind::CallInternal { .. }
        | SemanticOpKind::CallIndirect { .. }
        | SemanticOpKind::ReturnVoid
        | SemanticOpKind::ReturnOne
        | SemanticOpKind::Return { .. } => spill_all(&mut prefix, state, frame),
    }

    prefix
}

fn apply_semantic_effect(
    op: &SemanticOp,
    op_index: usize,
    op_result_types: &BTreeMap<usize, Vec<ValueType>>,
    state: &mut PrepareState,
) {
    match &op.kind {
        SemanticOpKind::Primitive(kind) => {
            if matches!(kind, primitive_op::PrimitiveOpKind::Unreachable) {
                state.mark_unreachable();
                return;
            }
            let (pop, push) = primitive_op::stack_effect(kind);
            let result_ty = if push == 0 {
                ValueType::I64 // unused — no value pushed
            } else if matches!(kind, PrimitiveOpKind::Select) {
                state
                    .type_stack
                    .len()
                    .checked_sub(3)
                    .and_then(|idx| state.type_stack.get(idx).copied())
                    .or_else(|| {
                        op_result_types
                            .get(&op_index)
                            .and_then(|v| v.first().copied())
                    })
                    .unwrap_or(ValueType::I64)
            } else if let Some(ty) = primitive_op::result_type(kind) {
                ty
            } else {
                let ty = op_result_types
                    .get(&op_index)
                    .and_then(|v| v.first().copied());
                debug_assert!(
                    ty.is_some(),
                    "context-dependent primitive at op {} has no op_result_types entry",
                    op_index,
                );
                ty.unwrap_or(ValueType::I64)
            };
            apply_stack_effect_typed(state, pop as u16, push as u16, result_ty);
        }
        SemanticOpKind::LocalGet { idx } => {
            if !state.unreachable {
                let ty = state.local_type(*idx);
                state.height += 1;
                state.type_stack.push(ty);
            }
        }
        SemanticOpKind::LocalSet { .. } => {
            if !state.unreachable {
                state.height = state.height.saturating_sub(1);
                state.spill_depth = state.spill_depth.min(state.height);
                state.type_stack.truncate(state.height as usize);
            }
        }
        SemanticOpKind::LocalTee { .. } => {}
        SemanticOpKind::Block { params, results } | SemanticOpKind::Loop { params, results } => {
            let sh = if state.unreachable {
                state.height
            } else {
                state.height.saturating_sub(*params)
            };
            let param_types = if state.unreachable {
                Vec::new()
            } else {
                state.types_at(sh, *params)
            };
            state.control.push(ControlFrame {
                kind: ControlFrameKind::Structured,
                start_height: sh,
                params: *params,
                results: *results,
                entered_unreachable: state.unreachable,
                param_types,
            });
        }
        SemanticOpKind::If {
            params, results, ..
        } => {
            let entered_unreachable = state.unreachable;
            if !state.unreachable {
                state.height = state.height.saturating_sub(1);
                state.type_stack.truncate(state.height as usize);
                state.spill_depth = state.height.saturating_sub(*params);
            }
            let sh = if entered_unreachable {
                state.height
            } else {
                state.height.saturating_sub(*params)
            };
            let param_types = if entered_unreachable {
                Vec::new()
            } else {
                state.types_at(sh, *params)
            };
            state.control.push(ControlFrame {
                kind: ControlFrameKind::Structured,
                start_height: sh,
                params: *params,
                results: *results,
                entered_unreachable,
                param_types,
            });
        }
        SemanticOpKind::Else { .. } => {
            if let Some(frame) = state.control.last().cloned() {
                if frame.entered_unreachable {
                    state.height = frame.start_height;
                    state.spill_depth = state.height;
                    state.type_stack.truncate(state.height as usize);
                    state.unreachable = true;
                } else {
                    state.height = frame.start_height + frame.params;
                    state.spill_depth = frame.start_height;
                    // Restore type stack to start_height, then append
                    // saved param types from block entry.
                    state.type_stack.truncate(frame.start_height as usize);
                    state.type_stack.extend_from_slice(&frame.param_types);
                    state.unreachable = false;
                }
            }
        }
        SemanticOpKind::End => {
            if let Some(frame) = state.control.pop() {
                let end_height = frame.start_height + frame.results;
                if frame.entered_unreachable {
                    match frame.kind {
                        ControlFrameKind::Function => {
                            state.height = end_height;
                            state.spill_depth = end_height;
                            state.type_stack.truncate(end_height as usize);
                            state.unreachable = false;
                        }
                        ControlFrameKind::Structured => {
                            state.height = frame.start_height;
                            state.spill_depth = state.height;
                            state.type_stack.truncate(state.height as usize);
                            state.unreachable = true;
                        }
                    }
                } else {
                    state.height = end_height;
                    state.spill_depth = match frame.kind {
                        ControlFrameKind::Function => end_height,
                        ControlFrameKind::Structured => frame.start_height,
                    };
                    state.type_stack.truncate(end_height as usize);
                    state.unreachable = false;
                }
            }
        }
        SemanticOpKind::Br { .. } => {
            state.mark_unreachable();
        }
        SemanticOpKind::BrIf { .. } => {
            if !state.unreachable {
                state.height = state.height.saturating_sub(1);
                state.spill_depth = state.height;
                state.type_stack.truncate(state.height as usize);
            }
        }
        SemanticOpKind::BrTable { .. } => {
            if !state.unreachable {
                state.height = state.height.saturating_sub(1);
                state.spill_depth = state.height;
                state.type_stack.truncate(state.height as usize);
            }
            state.mark_unreachable();
        }
        SemanticOpKind::CallExternal {
            params, results, ..
        }
        | SemanticOpKind::CallInternal {
            params, results, ..
        } => {
            if !state.unreachable {
                state.height = state
                    .height
                    .saturating_sub(*params)
                    .saturating_add(*results);
                state.spill_depth = state.height;
                let base = state.height.saturating_sub(*results) as usize;
                state.type_stack.truncate(base);
                push_result_types(&mut state.type_stack, *results, op_index, op_result_types);
            }
        }
        SemanticOpKind::CallIndirect {
            params, results, ..
        } => {
            if !state.unreachable {
                state.height = state
                    .height
                    .saturating_sub(params.saturating_add(1))
                    .saturating_add(*results);
                state.spill_depth = state.height;
                let base = state.height.saturating_sub(*results) as usize;
                state.type_stack.truncate(base);
                push_result_types(&mut state.type_stack, *results, op_index, op_result_types);
            }
        }
        SemanticOpKind::ReturnVoid | SemanticOpKind::ReturnOne | SemanticOpKind::Return { .. } => {
            state.mark_unreachable();
        }
    }
}

/// Push result types from the op_result_types sidecar.
///
/// Asserts in debug mode that the sidecar has an entry for every call with
/// results. Falls back to I64 in release for robustness.
fn push_result_types(
    type_stack: &mut Vec<ValueType>,
    results: u16,
    op_index: usize,
    op_result_types: &BTreeMap<usize, Vec<ValueType>>,
) {
    if let Some(types) = op_result_types.get(&op_index) {
        debug_assert_eq!(
            types.len(),
            results as usize,
            "op_result_types at op {} has {} types but call expects {} results",
            op_index,
            types.len(),
            results,
        );
        type_stack.extend_from_slice(types);
    } else {
        debug_assert!(
            results == 0,
            "call at op {} produces {} results but has no op_result_types entry",
            op_index,
            results,
        );
        for _ in 0..results {
            type_stack.push(ValueType::I64);
        }
    }
}

fn apply_stack_effect_typed(state: &mut PrepareState, pop: u16, push: u16, result_ty: ValueType) {
    if state.unreachable {
        return;
    }
    state.height = state.height.saturating_sub(pop).saturating_add(push);
    state.spill_depth = state.spill_depth.min(state.height);
    // Maintain type_stack: remove popped entries, add pushed entries.
    let new_base = state.height.saturating_sub(push) as usize;
    state.type_stack.truncate(new_base);
    for _ in 0..push {
        state.type_stack.push(result_ty);
    }
}

fn spill_before_result_push(
    prefix: &mut Vec<PrepAction>,
    state: &mut PrepareState,
    frame: FrameLayoutPlan,
    gp_lanes: u8,
    fp_lanes: u8,
    pop: u16,
    push: u16,
    push_ty: ValueType,
) {
    if state.unreachable || push == 0 || (gp_lanes == 0 && fp_lanes == 0) {
        return;
    }
    // Spill from the bottom of the live window until the post-op live suffix
    // fits both bank budgets. This matters even for pop+push ops like
    // i32->f64 converts, which can shift pressure from the GP bank to the FP bank
    // without increasing total stack height.
    loop {
        let post_height = state.height.saturating_sub(pop);
        let live_start = state.spill_depth as usize;
        let live_end = post_height as usize;
        let (mut gp_live, mut fp_live) =
            count_live_banks(state.type_stack.get(live_start..live_end).unwrap_or(&[]), 0);
        if push_ty.is_float() {
            fp_live = fp_live.saturating_add(push);
        } else {
            gp_live = gp_live.saturating_add(push);
        }
        let post_live_count = post_height
            .saturating_sub(state.spill_depth)
            .saturating_add(push);
        let combined_limit = (gp_lanes as u16).saturating_add(fp_lanes as u16);
        if gp_live <= gp_lanes as u16
            && fp_live <= fp_lanes as u16
            && post_live_count <= combined_limit
        {
            return;
        }
        if state.spill_depth >= post_height {
            return;
        }
        prefix.push(PrepAction::Spill(FrameSpan::new(
            frame.operand_slot(state.spill_depth),
            1,
        )));
        state.spill_depth += 1;
    }
}

fn count_live_banks(type_stack: &[ValueType], live_start: usize) -> (u16, u16) {
    let mut gp: u16 = 0;
    let mut fp: u16 = 0;
    for ty in type_stack.get(live_start..).unwrap_or(&[]) {
        if ty.is_float() {
            fp += 1;
        } else {
            gp += 1;
        }
    }
    (gp, fp)
}

fn primitive_result_type(
    kind: &PrimitiveOpKind,
    op_index: usize,
    op_result_types: &BTreeMap<usize, Vec<ValueType>>,
) -> Option<ValueType> {
    primitive_op::result_type(kind).or_else(|| {
        op_result_types
            .get(&op_index)
            .and_then(|v| v.first().copied())
    })
}

fn fill_for_operands(
    prefix: &mut Vec<PrepAction>,
    state: &mut PrepareState,
    frame: FrameLayoutPlan,
    operand_count: u16,
) {
    fill_for_operands_inner(prefix, state, frame, operand_count, true);
}

fn fill_for_operands_inner(
    prefix: &mut Vec<PrepAction>,
    state: &mut PrepareState,
    frame: FrameLayoutPlan,
    operand_count: u16,
    skip_if_unreachable: bool,
) {
    if skip_if_unreachable && state.unreachable {
        return;
    }
    let min_spill_depth = state.height.saturating_sub(operand_count);
    if state.spill_depth <= min_spill_depth {
        return;
    }
    let fill_count = state.spill_depth - min_spill_depth;
    let fill_start = state.spill_depth - fill_count;
    let fill_types = state.types_at(fill_start, fill_count);
    prefix.push(PrepAction::Fill(
        FrameSpan::new(frame.operand_slot(fill_start), fill_count),
        fill_types,
    ));
    state.spill_depth -= fill_count;
}

fn spill_all(prefix: &mut Vec<PrepAction>, state: &mut PrepareState, frame: FrameLayoutPlan) {
    if state.unreachable || state.spill_depth >= state.height {
        return;
    }
    let count = state.height - state.spill_depth;
    prefix.push(PrepAction::Spill(FrameSpan::new(
        frame.operand_slot(state.spill_depth),
        count,
    )));
    state.spill_depth = state.height;
}

fn spill_all_except_top(
    prefix: &mut Vec<PrepAction>,
    state: &mut PrepareState,
    frame: FrameLayoutPlan,
    keep_top: u16,
) {
    if state.unreachable {
        return;
    }
    let live_count = state.height.saturating_sub(state.spill_depth);
    let count = live_count.saturating_sub(keep_top);
    if count == 0 {
        return;
    }
    prefix.push(PrepAction::Spill(FrameSpan::new(
        frame.operand_slot(state.spill_depth),
        count,
    )));
    state.spill_depth += count;
}
