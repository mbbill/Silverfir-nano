//! Internal semantic preparation steps.
//!
//! This is private preparation state, not a public IR layer.

use alloc::vec::Vec;

use crate::{
    error::WasmError,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PrepAction {
    Spill(FrameSpan),
    Fill(FrameSpan),
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

#[derive(Clone, Copy, Debug)]
struct ControlFrame {
    kind: ControlFrameKind,
    start_height: u16,
    params: u16,
    results: u16,
    entered_unreachable: bool,
}

#[derive(Clone, Debug)]
struct PrepareState {
    height: u16,
    spill_depth: u16,
    unreachable: bool,
    control: Vec<ControlFrame>,
}

impl PrepareState {
    fn new(results: u16) -> Self {
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
            }],
        }
    }

    fn mark_unreachable(&mut self) {
        if let Some(frame) = self.control.last().copied() {
            self.height = frame.start_height.saturating_add(frame.results);
            self.spill_depth = self.height;
        }
        self.unreachable = true;
    }
}

pub(super) fn prepare_semantic_ops<'a>(
    semantic: &'a SemanticProgram,
    frame: FrameLayoutPlan,
    config: PlanConfig,
) -> Result<PreparedStream<'a>, WasmError> {
    let mut state = PrepareState::new(semantic.results);
    let mut entry_states = Vec::with_capacity(semantic.ops.len());
    let mut ops = Vec::with_capacity(semantic.ops.len());

    for semantic_op in &semantic.ops {
        entry_states.push(EntryState {
            stack_height: state.height,
            spill_depth: state.spill_depth,
        });
        let prefix = plan_prefix(semantic_op, &mut state, frame, config.lir_lanes);
        ops.push(PreparedOp {
            semantic: semantic_op,
            prefix,
        });
        apply_semantic_effect(semantic_op, &mut state);
    }

    Ok(PreparedStream { ops, entry_states })
}

fn plan_prefix(
    op: &SemanticOp,
    state: &mut PrepareState,
    frame: FrameLayoutPlan,
    tos_lanes: u8,
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
                if pop == 0 && push > 0 {
                    spill_before_push(&mut prefix, state, frame, tos_lanes);
                }
                fill_for_operands(&mut prefix, state, frame, pop as u16);
            }
        }
        SemanticOpKind::LocalGet { .. } => spill_before_push(&mut prefix, state, frame, tos_lanes),
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
            if let Some(frame_state) = state.control.last().copied() {
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

fn apply_semantic_effect(op: &SemanticOp, state: &mut PrepareState) {
    match &op.kind {
        SemanticOpKind::Primitive(kind) => {
            if matches!(kind, primitive_op::PrimitiveOpKind::Unreachable) {
                state.mark_unreachable();
                return;
            }
            let (pop, push) = primitive_op::stack_effect(kind);
            apply_stack_effect(state, pop as u16, push as u16);
        }
        SemanticOpKind::LocalGet { .. } => {
            if !state.unreachable {
                state.height += 1;
            }
        }
        SemanticOpKind::LocalSet { .. } => {
            if !state.unreachable {
                state.height = state.height.saturating_sub(1);
                state.spill_depth = state.spill_depth.min(state.height);
            }
        }
        SemanticOpKind::LocalTee { .. } => {}
        SemanticOpKind::Block { params, results } | SemanticOpKind::Loop { params, results } => {
            state.control.push(ControlFrame {
                kind: ControlFrameKind::Structured,
                start_height: if state.unreachable {
                    state.height
                } else {
                    state.height.saturating_sub(*params)
                },
                params: *params,
                results: *results,
                entered_unreachable: state.unreachable,
            });
        }
        SemanticOpKind::If {
            params, results, ..
        } => {
            let entered_unreachable = state.unreachable;
            if !state.unreachable {
                state.height = state.height.saturating_sub(1);
                state.spill_depth = state.height.saturating_sub(*params);
            }
            state.control.push(ControlFrame {
                kind: ControlFrameKind::Structured,
                start_height: if entered_unreachable {
                    state.height
                } else {
                    state.height.saturating_sub(*params)
                },
                params: *params,
                results: *results,
                entered_unreachable,
            });
        }
        SemanticOpKind::Else { .. } => {
            if let Some(frame) = state.control.last().copied() {
                if frame.entered_unreachable {
                    state.height = frame.start_height;
                    state.spill_depth = state.height;
                    state.unreachable = true;
                } else {
                    state.height = frame.start_height + frame.params;
                    state.spill_depth = frame.start_height;
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
                            state.unreachable = false;
                        }
                        ControlFrameKind::Structured => {
                            state.height = frame.start_height;
                            state.spill_depth = state.height;
                            state.unreachable = true;
                        }
                    }
                } else {
                    state.height = end_height;
                    state.spill_depth = match frame.kind {
                        ControlFrameKind::Function => end_height,
                        ControlFrameKind::Structured => frame.start_height,
                    };
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
            }
        }
        SemanticOpKind::BrTable { .. } => {
            if !state.unreachable {
                state.height = state.height.saturating_sub(1);
                state.spill_depth = state.height;
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
            }
        }
        SemanticOpKind::ReturnVoid | SemanticOpKind::ReturnOne | SemanticOpKind::Return { .. } => {
            state.mark_unreachable();
        }
    }
}

fn apply_stack_effect(state: &mut PrepareState, pop: u16, push: u16) {
    if state.unreachable {
        return;
    }
    state.height = state.height.saturating_sub(pop).saturating_add(push);
    state.spill_depth = state.spill_depth.min(state.height);
}

fn spill_before_push(
    prefix: &mut Vec<PrepAction>,
    state: &mut PrepareState,
    frame: FrameLayoutPlan,
    tos_lanes: u8,
) {
    if state.unreachable || tos_lanes == 0 {
        return;
    }
    let live_count = state.height.saturating_sub(state.spill_depth);
    if live_count < tos_lanes as u16 {
        return;
    }
    prefix.push(PrepAction::Spill(FrameSpan::new(
        frame.operand_slot(state.spill_depth),
        1,
    )));
    state.spill_depth += 1;
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
    prefix.push(PrepAction::Fill(FrameSpan::new(
        frame.operand_slot(state.spill_depth - fill_count),
        fill_count,
    )));
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
