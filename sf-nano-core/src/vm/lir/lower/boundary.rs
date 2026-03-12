//! Block-entry boundary shaping for prepared LIR lowering.

use crate::vm::{
    plan::{
        tos::{SpillArtifact, TosTransferDirection},
        PlannedBranchKind, PlannedMarkerKind, PlannedOpKind,
    },
    wasm::{primitive_op::stack_effect, semantic_ir::SemanticProgram},
};

use super::{
    input::AlignedOp,
    state::{EntryState, ValueAlloc},
};

#[derive(Clone, Copy, Debug)]
struct ControlFrame {
    start_height: u16,
    params: u16,
    results: u16,
}

#[derive(Clone, Debug)]
struct PreparedState {
    height: u16,
    spill_depth: u16,
    unreachable: bool,
    control: alloc::vec::Vec<ControlFrame>,
}

impl PreparedState {
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

pub(super) fn make_block_params(entry: EntryState, values: &mut ValueAlloc) -> alloc::vec::Vec<crate::vm::lir::ir::LirValue> {
    values.many(entry.live_tos_count() as usize)
}

pub(super) fn compute_entry_states(
    semantic: &SemanticProgram,
    aligned: &[AlignedOp<'_>],
) -> alloc::vec::Vec<EntryState> {
    let mut states = alloc::vec![EntryState::default(); aligned.len()];
    let mut state = PreparedState::new(semantic.results);

    for (index, op) in aligned.iter().enumerate() {
        states[index] = EntryState {
            stack_height: state.height,
            spill_depth: state.spill_depth,
        };
        apply_prefix(&mut state, op);
        apply_op(&mut state, op);
    }

    states
}

fn apply_prefix(state: &mut PreparedState, op: &AlignedOp<'_>) {
    for planned in &op.prefix {
        let PlannedOpKind::Spill(artifact) = planned.kind else {
            continue;
        };
        match artifact {
            SpillArtifact::TosTransfer(transfer) => match transfer.direction {
                TosTransferDirection::Spill => {
                    state.spill_depth = state.spill_depth.saturating_add(transfer.frame.count)
                }
                TosTransferDirection::Fill => {
                    state.spill_depth = state.spill_depth.saturating_sub(transfer.frame.count)
                }
            },
        }
    }
}

fn apply_op(state: &mut PreparedState, op: &AlignedOp<'_>) {
    match &op.planned.kind {
        PlannedOpKind::Primitive(kind) => {
            if matches!(kind, crate::vm::wasm::primitive_op::PrimitiveOpKind::Unreachable) {
                state.unreachable = true;
                return;
            }
            let (pop, push) = stack_effect(kind);
            if !state.unreachable {
                state.height = state.height.saturating_sub(pop as u16).saturating_add(push as u16);
                state.spill_depth = state.spill_depth.min(state.height);
            }
        }
        PlannedOpKind::LocalGet { .. } => {
            if !state.unreachable {
                state.height += 1;
            }
        }
        PlannedOpKind::LocalSet { .. } => {
            if !state.unreachable {
                state.height = state.height.saturating_sub(1);
                state.spill_depth = state.spill_depth.min(state.height);
            }
        }
        PlannedOpKind::LocalTee { .. } => {}
        PlannedOpKind::Marker(PlannedMarkerKind::Block { params, results })
        | PlannedOpKind::Marker(PlannedMarkerKind::Loop { params, results }) => {
            state.control.push(ControlFrame {
                start_height: state.height.saturating_sub(*params),
                params: *params,
                results: *results,
            });
        }
        PlannedOpKind::Marker(PlannedMarkerKind::If { params, results }) => {
            if !state.unreachable {
                state.height = state.height.saturating_sub(1);
                state.spill_depth = state.height;
            }
            state.control.push(ControlFrame {
                start_height: state.height.saturating_sub(*params),
                params: *params,
                results: *results,
            });
        }
        PlannedOpKind::Marker(PlannedMarkerKind::Else) => {
            if let Some(frame) = state.control.last().copied() {
                state.height = frame.start_height + frame.params;
                state.spill_depth = state.height;
                state.unreachable = false;
            }
        }
        PlannedOpKind::Marker(PlannedMarkerKind::End) => {
            if let Some(frame) = state.control.pop() {
                state.height = frame.start_height + frame.results;
                state.spill_depth = state.height;
                state.unreachable = false;
            }
        }
        PlannedOpKind::Branch {
            kind: PlannedBranchKind::Br,
            ..
        } => {
            state.unreachable = true;
        }
        PlannedOpKind::Branch {
            kind: PlannedBranchKind::BrIf,
            ..
        } => {
            if !state.unreachable {
                state.height = state.height.saturating_sub(1);
                state.spill_depth = state.height;
            }
        }
        PlannedOpKind::BrTable { .. } => {
            if !state.unreachable {
                state.height = state.height.saturating_sub(1);
                state.spill_depth = state.height;
            }
            state.unreachable = true;
        }
        PlannedOpKind::CallExternal { args, results, .. }
        | PlannedOpKind::CallInternal { args, results, .. } => {
            if !state.unreachable {
                state.height = state
                    .height
                    .saturating_sub(args.count)
                    .saturating_add(results.count);
                state.spill_depth = state.height;
            }
        }
        PlannedOpKind::CallIndirect { args, results, .. } => {
            if !state.unreachable {
                state.height = state
                    .height
                    .saturating_sub(args.count.saturating_add(1))
                    .saturating_add(results.count);
                state.spill_depth = state.height;
            }
        }
        PlannedOpKind::Return { .. } => {
            state.unreachable = true;
        }
        PlannedOpKind::InitHotLocals { .. } | PlannedOpKind::Spill(_) => {}
    }
}
