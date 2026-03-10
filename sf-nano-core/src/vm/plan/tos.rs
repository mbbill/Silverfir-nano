//! TOS-window planning and spill/fill artifacts.
//!
//! Important:
//! - the top `N` stack values live in the TOS window, not in mirrored memory
//! - planning decides where the logical top sits in the rotating window
//! - spill/fill artifacts make TOS requirements explicit before LIR
//! - this module must not leak stack-height deduction into backend-facing IR

use alloc::vec::Vec;

use crate::vm::{wasm::common::SemanticTarget, wasm::primitive_op};

use super::{
    frame::{FrameLayoutPlan, FrameSpan},
    place::SeedOp,
    types::{PlannedBranchKind, PlannedHotLocalInit, PlannedMarkerKind, PlannedOp, PlannedOpKind},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TosTransferDirection {
    Spill,
    Fill,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TosTransfer {
    pub direction: TosTransferDirection,
    pub frame: FrameSpan,
}

/// Planning-stage spill/fill artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpillArtifact {
    TosTransfer(TosTransfer),
}

impl SpillArtifact {
    #[inline]
    pub const fn spill(frame: FrameSpan) -> Self {
        Self::TosTransfer(TosTransfer {
            direction: TosTransferDirection::Spill,
            frame,
        })
    }

    #[inline]
    pub const fn fill(frame: FrameSpan) -> Self {
        Self::TosTransfer(TosTransfer {
            direction: TosTransferDirection::Fill,
            frame,
        })
    }
}

/// Rotation of the logical top within the backend-selected TOS ring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TosRotation(pub u8);

impl TosRotation {
    #[inline]
    pub const fn new(raw: u8) -> Self {
        Self(raw)
    }

    #[inline]
    pub const fn for_position(position: u16, register_count: u8) -> Self {
        if register_count == 0 {
            Self::new(0)
        } else {
            Self::new((position % register_count as u16) as u8)
        }
    }

    #[inline]
    pub const fn for_height(height: u16, register_count: u8) -> Self {
        Self::for_position(height, register_count)
    }

    /// Physical register index of the current logical top inside the rotating
    /// TOS window.
    #[inline]
    pub const fn top_index(self, register_count: u8) -> u8 {
        if register_count == 0 {
            0
        } else {
            self.0 % register_count
        }
    }

    /// Physical register index for the `depth_from_top`-th value in the
    /// rotating TOS window.
    #[inline]
    pub const fn nth_from_top(self, depth_from_top: u8, register_count: u8) -> u8 {
        if register_count == 0 {
            0
        } else {
            self.0
                .wrapping_add(register_count)
                .wrapping_sub(depth_from_top)
                % register_count
        }
    }

    /// Rotation after popping `count` values.
    #[inline]
    pub const fn after_pop(self, count: u8, register_count: u8) -> Self {
        if register_count == 0 {
            Self::new(0)
        } else {
            Self::new(self.0.wrapping_add(register_count).wrapping_sub(count) % register_count)
        }
    }

    /// Rotation after pushing `count` values.
    #[inline]
    pub const fn after_push(self, count: u8, register_count: u8) -> Self {
        if register_count == 0 {
            Self::new(0)
        } else {
            Self::new(self.0.wrapping_add(count) % register_count)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ControlFrame {
    start_height: u16,
    params: u16,
    results: u16,
}

#[derive(Clone, Debug)]
struct TosPlanState {
    height: u16,
    spill_depth: u16,
    unreachable: bool,
    control: Vec<ControlFrame>,
}

impl TosPlanState {
    fn new(results: u16) -> Self {
        Self {
            height: 0,
            spill_depth: 0,
            unreachable: false,
            control: alloc::vec![ControlFrame {
                start_height: 0,
                params: 0,
                results,
            }],
        }
    }
}

pub(crate) fn plan_tos_ops(
    ops: Vec<SeedOp>,
    function_results: u16,
    frame: FrameLayoutPlan,
    tos_register_count: u8,
    hot_local_inits: Vec<PlannedHotLocalInit>,
) -> Vec<PlannedOp> {
    let mut state = TosPlanState::new(function_results);
    let mut out = Vec::with_capacity(ops.len() + usize::from(!hot_local_inits.is_empty()));

    if !hot_local_inits.is_empty() {
        emit_planned_op(
            &mut out,
            &state,
            PlannedOpKind::InitHotLocals {
                inits: hot_local_inits,
            },
            TosRotation::new(0),
            None,
        );
    }

    for op in ops {
        plan_tos_op(&mut out, &mut state, frame, tos_register_count, op);
    }

    out
}

fn plan_tos_op(
    out: &mut Vec<PlannedOp>,
    state: &mut TosPlanState,
    frame: FrameLayoutPlan,
    tos_register_count: u8,
    op: SeedOp,
) {
    let alt = op.alt;
    match op.kind {
        PlannedOpKind::Primitive(kind) => {
            if matches!(
                kind,
                crate::vm::wasm::primitive_op::PrimitiveOpKind::Unreachable
            ) {
                emit_planned_op(
                    out,
                    state,
                    PlannedOpKind::Primitive(kind),
                    TosRotation::new(0),
                    alt,
                );
                state.unreachable = true;
                return;
            }

            let (pop, push) = primitive_op::stack_effect(&kind);
            if pop == 0 && push > 0 {
                spill_before_push(out, state, frame, tos_register_count);
            }
            fill_for_operands(out, state, frame, pop as u16, tos_register_count);
            emit_planned_op(
                out,
                state,
                PlannedOpKind::Primitive(kind),
                TosRotation::for_height(state.height, tos_register_count),
                alt,
            );
            apply_stack_effect(state, pop as u16, push as u16);
        }
        PlannedOpKind::LocalGet { local } => {
            spill_before_push(out, state, frame, tos_register_count);
            emit_planned_op(
                out,
                state,
                PlannedOpKind::LocalGet { local },
                TosRotation::for_height(state.height, tos_register_count),
                alt,
            );
            if !state.unreachable {
                state.height += 1;
            }
        }
        PlannedOpKind::LocalSet { local } => {
            fill_for_operands(out, state, frame, 1, tos_register_count);
            emit_planned_op(
                out,
                state,
                PlannedOpKind::LocalSet { local },
                TosRotation::for_height(state.height, tos_register_count),
                alt,
            );
            if !state.unreachable {
                state.height = state.height.saturating_sub(1);
                state.spill_depth = state.spill_depth.min(state.height);
            }
        }
        PlannedOpKind::LocalTee { local } => {
            fill_for_operands(out, state, frame, 1, tos_register_count);
            emit_planned_op(
                out,
                state,
                PlannedOpKind::LocalTee { local },
                TosRotation::for_height(state.height, tos_register_count),
                alt,
            );
        }
        PlannedOpKind::Marker(PlannedMarkerKind::Block { params, results }) => {
            emit_planned_op(
                out,
                state,
                PlannedOpKind::Marker(PlannedMarkerKind::Block { params, results }),
                TosRotation::new(0),
                alt,
            );
            state.control.push(ControlFrame {
                start_height: state.height.saturating_sub(params),
                params,
                results,
            });
        }
        PlannedOpKind::Marker(PlannedMarkerKind::Loop { params, results }) => {
            spill_all(out, state, frame, tos_register_count);
            emit_planned_op(
                out,
                state,
                PlannedOpKind::Marker(PlannedMarkerKind::Loop { params, results }),
                TosRotation::new(0),
                alt,
            );
            state.control.push(ControlFrame {
                start_height: state.height.saturating_sub(params),
                params,
                results,
            });
        }
        PlannedOpKind::Marker(PlannedMarkerKind::If { params, results }) => {
            fill_for_operands(out, state, frame, 1, tos_register_count);
            spill_all_except_top(out, state, frame, 1, tos_register_count);
            emit_planned_op(
                out,
                state,
                PlannedOpKind::Marker(PlannedMarkerKind::If { params, results }),
                TosRotation::for_height(state.height, tos_register_count),
                alt,
            );
            if !state.unreachable {
                state.height = state.height.saturating_sub(1);
                state.spill_depth = state.height;
            }
            state.control.push(ControlFrame {
                start_height: state.height.saturating_sub(params),
                params,
                results,
            });
        }
        PlannedOpKind::Marker(PlannedMarkerKind::Else) => {
            spill_all(out, state, frame, tos_register_count);
            emit_planned_op(
                out,
                state,
                PlannedOpKind::Marker(PlannedMarkerKind::Else),
                TosRotation::new(0),
                alt,
            );
            if let Some(frame) = state.control.last().copied() {
                state.height = frame.start_height + frame.params;
                state.spill_depth = state.height;
                state.unreachable = false;
            }
        }
        PlannedOpKind::Marker(PlannedMarkerKind::End) => {
            spill_all(out, state, frame, tos_register_count);
            emit_planned_op(
                out,
                state,
                PlannedOpKind::Marker(PlannedMarkerKind::End),
                TosRotation::new(0),
                alt,
            );
            if let Some(frame) = state.control.pop() {
                state.height = frame.start_height + frame.results;
                state.spill_depth = state.height;
                state.unreachable = false;
            }
        }
        PlannedOpKind::Branch {
            kind: PlannedBranchKind::Br,
            condition_slot,
            payload,
            target,
        } => {
            spill_all(out, state, frame, tos_register_count);
            emit_planned_op(
                out,
                state,
                PlannedOpKind::Branch {
                    kind: PlannedBranchKind::Br,
                    condition_slot,
                    payload,
                    target,
                },
                TosRotation::new(0),
                alt,
            );
            state.unreachable = true;
        }
        PlannedOpKind::Branch {
            kind: PlannedBranchKind::BrIf,
            condition_slot,
            payload,
            target,
        } => {
            fill_for_operands(out, state, frame, 1, tos_register_count);
            spill_all_except_top(out, state, frame, 1, tos_register_count);
            emit_planned_op(
                out,
                state,
                PlannedOpKind::Branch {
                    kind: PlannedBranchKind::BrIf,
                    condition_slot,
                    payload,
                    target,
                },
                TosRotation::for_height(state.height, tos_register_count),
                alt,
            );
            if !state.unreachable {
                state.height = state.height.saturating_sub(1);
                state.spill_depth = state.height;
            }
        }
        PlannedOpKind::BrTable {
            index_slot,
            entries,
        } => {
            fill_for_operands(out, state, frame, 1, tos_register_count);
            spill_all_except_top(out, state, frame, 1, tos_register_count);
            emit_planned_op(
                out,
                state,
                PlannedOpKind::BrTable {
                    index_slot,
                    entries,
                },
                TosRotation::for_height(state.height, tos_register_count),
                alt,
            );
            if !state.unreachable {
                state.height = state.height.saturating_sub(1);
                state.spill_depth = state.height;
            }
            state.unreachable = true;
        }
        PlannedOpKind::CallExternal {
            func_idx,
            args,
            results,
        } => {
            spill_all(out, state, frame, tos_register_count);
            emit_planned_op(
                out,
                state,
                PlannedOpKind::CallExternal {
                    func_idx,
                    args,
                    results,
                },
                TosRotation::new(0),
                alt,
            );
            if !state.unreachable {
                state.height = state
                    .height
                    .saturating_sub(args.count)
                    .saturating_add(results.count);
                state.spill_depth = state.height;
            }
        }
        PlannedOpKind::CallInternal {
            callee,
            args,
            results,
        } => {
            spill_all(out, state, frame, tos_register_count);
            emit_planned_op(
                out,
                state,
                PlannedOpKind::CallInternal {
                    callee,
                    args,
                    results,
                },
                TosRotation::new(0),
                alt,
            );
            if !state.unreachable {
                state.height = state
                    .height
                    .saturating_sub(args.count)
                    .saturating_add(results.count);
                state.spill_depth = state.height;
            }
        }
        PlannedOpKind::CallIndirect {
            type_idx,
            table_idx,
            index_slot,
            args,
            results,
        } => {
            spill_all(out, state, frame, tos_register_count);
            emit_planned_op(
                out,
                state,
                PlannedOpKind::CallIndirect {
                    type_idx,
                    table_idx,
                    index_slot,
                    args,
                    results,
                },
                TosRotation::new(0),
                alt,
            );
            if !state.unreachable {
                state.height = state
                    .height
                    .saturating_sub(args.count.saturating_add(1))
                    .saturating_add(results.count);
                state.spill_depth = state.height;
            }
        }
        PlannedOpKind::Return { results } => {
            spill_all(out, state, frame, tos_register_count);
            emit_planned_op(
                out,
                state,
                PlannedOpKind::Return { results },
                TosRotation::new(0),
                alt,
            );
            state.unreachable = true;
        }
        PlannedOpKind::Spill(artifact) => {
            emit_planned_op(
                out,
                state,
                PlannedOpKind::Spill(artifact),
                TosRotation::new(0),
                alt,
            );
        }
        PlannedOpKind::InitHotLocals { inits } => {
            emit_planned_op(
                out,
                state,
                PlannedOpKind::InitHotLocals { inits },
                TosRotation::new(0),
                alt,
            );
        }
    }
}

fn emit_planned_op(
    out: &mut Vec<PlannedOp>,
    state: &TosPlanState,
    kind: PlannedOpKind,
    rotation: TosRotation,
    alt: Option<SemanticTarget>,
) {
    out.push(PlannedOp {
        kind,
        rotation,
        height: state.height,
        alt,
    });
}

fn apply_stack_effect(state: &mut TosPlanState, pop: u16, push: u16) {
    if state.unreachable {
        return;
    }
    state.height = state.height.saturating_sub(pop).saturating_add(push);
    state.spill_depth = state.spill_depth.min(state.height);
}

fn spill_before_push(
    out: &mut Vec<PlannedOp>,
    state: &mut TosPlanState,
    frame: FrameLayoutPlan,
    tos_register_count: u8,
) {
    if state.unreachable || tos_register_count == 0 {
        return;
    }
    let tos_count = state.height.saturating_sub(state.spill_depth);
    if tos_count < tos_register_count as u16 {
        return;
    }
    emit_tos_transfer(
        out,
        state,
        SpillArtifact::spill(FrameSpan::new(frame.operand_slot(state.spill_depth), 1)),
        TosRotation::for_position(state.spill_depth + 1, tos_register_count),
    );
    state.spill_depth += 1;
}

fn fill_for_operands(
    out: &mut Vec<PlannedOp>,
    state: &mut TosPlanState,
    frame: FrameLayoutPlan,
    operand_count: u16,
    tos_register_count: u8,
) {
    if state.unreachable {
        return;
    }
    let min_spill_depth = state.height.saturating_sub(operand_count);
    if state.spill_depth <= min_spill_depth {
        return;
    }
    let fill_count = state.spill_depth - min_spill_depth;
    emit_tos_transfer(
        out,
        state,
        SpillArtifact::fill(FrameSpan::new(
            frame.operand_slot(state.spill_depth - fill_count),
            fill_count,
        )),
        TosRotation::for_position(state.spill_depth, tos_register_count),
    );
    state.spill_depth -= fill_count;
}

fn spill_all(
    out: &mut Vec<PlannedOp>,
    state: &mut TosPlanState,
    frame: FrameLayoutPlan,
    tos_register_count: u8,
) {
    if state.unreachable || state.spill_depth >= state.height {
        return;
    }
    let count = state.height - state.spill_depth;
    emit_tos_transfer(
        out,
        state,
        SpillArtifact::spill(FrameSpan::new(frame.operand_slot(state.spill_depth), count)),
        TosRotation::for_position(state.spill_depth + count, tos_register_count),
    );
    state.spill_depth = state.height;
}

fn spill_all_except_top(
    out: &mut Vec<PlannedOp>,
    state: &mut TosPlanState,
    frame: FrameLayoutPlan,
    keep_top: u16,
    tos_register_count: u8,
) {
    if state.unreachable {
        return;
    }
    let tos_count = state.height.saturating_sub(state.spill_depth);
    let count = tos_count.saturating_sub(keep_top);
    if count == 0 {
        return;
    }
    emit_tos_transfer(
        out,
        state,
        SpillArtifact::spill(FrameSpan::new(frame.operand_slot(state.spill_depth), count)),
        TosRotation::for_position(state.spill_depth + count, tos_register_count),
    );
    state.spill_depth += count;
}

fn emit_tos_transfer(
    out: &mut Vec<PlannedOp>,
    state: &TosPlanState,
    artifact: SpillArtifact,
    rotation: TosRotation,
) {
    emit_planned_op(out, state, PlannedOpKind::Spill(artifact), rotation, None);
}
