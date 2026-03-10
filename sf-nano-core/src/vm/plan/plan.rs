//! Planned program after stack-aware planning and grouping.
//!
//! This is the handoff between semantic Wasm and backend-facing IR lowering.

use alloc::vec::Vec;

use crate::vm::wasm::{
    common::{BrTableEntry, SemanticTarget},
    core_op::CoreOpKind,
    semantic_ir::{SemanticOp, SemanticOpKind, SemanticProgram},
};

use super::{
    config::PlanConfig,
    frame::{FrameLayoutPlan, FramePlanner, FrameSlot, FrameSpan},
    group::{GroupInputOp, GroupPlan, GroupTerminator},
    hot_local::HotLocalPlan,
    policy::{GroupPolicy, PlanPolicy},
    spill::SpillArtifact,
    tos::TosRotation,
};

/// Planned local placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlannedLocal {
    Hot(u8),
    Frame(FrameSlot),
}

/// Planning-stage op vocabulary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlannedOpKind {
    Core(CoreOpKind),
    InitLocals {
        l0: FrameSlot,
        l1: FrameSlot,
        l2: FrameSlot,
    },
    Spill(SpillArtifact),
    LocalGet {
        local: PlannedLocal,
    },
    LocalSet {
        local: PlannedLocal,
    },
    LocalTee {
        local: PlannedLocal,
    },
    Marker(PlannedMarkerKind),
    Branch {
        kind: PlannedBranchKind,
        condition_slot: Option<FrameSlot>,
        payload: Option<FrameSpan>,
        target: Option<SemanticTarget>,
    },
    BrTable {
        index_slot: FrameSlot,
        payload: Option<FrameSpan>,
        entries: Vec<PlannedBrTableEntry>,
    },
    CallExternal {
        func_idx: u32,
        args: FrameSpan,
        results: FrameSpan,
    },
    CallInternal {
        callee: u32,
        args: FrameSpan,
        results: FrameSpan,
    },
    CallIndirect {
        type_idx: u32,
        table_idx: u32,
        index_slot: FrameSlot,
        args: FrameSpan,
        results: FrameSpan,
    },
    Return {
        results: Option<FrameSpan>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlannedMarkerKind {
    Block { params: u16, results: u16 },
    Loop { params: u16, results: u16 },
    If { params: u16, results: u16 },
    Else,
    End,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlannedBranchKind {
    Br,
    BrIf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlannedBrTableEntry {
    pub target: Option<SemanticTarget>,
    pub payload: Option<FrameSpan>,
}

/// One planned op. This is still stack-aware, but the stack-aware concepts live
/// here instead of leaking into LIR/backends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedOp {
    pub kind: PlannedOpKind,
    pub rotation: TosRotation,
    pub height: u16,
    pub alt: Option<SemanticTarget>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PlacedOp {
    kind: PlannedOpKind,
    window: u8,
    height: u16,
    alt: Option<SemanticTarget>,
}

/// Planned program bundle.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PlannedProgram {
    pub frame: FrameLayoutPlan,
    pub hot_locals: Option<HotLocalPlan>,
    pub ops: Vec<PlannedOp>,
    pub groups: GroupPlan,
}

/// Planning input bundle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanningInput {
    pub config: PlanConfig,
    pub policy: PlanPolicy,
    pub hot_locals: Option<HotLocalPlan>,
}

/// Plan semantic IR into the stack-aware planned layer.
pub fn build_planned_program(
    input: PlanningInput,
    semantic: &SemanticProgram,
) -> PlannedProgram {
    let hot_locals = input
        .hot_locals
        .or_else(|| select_hot_locals(semantic, input.config));
    let frame = plan_frame_layout(semantic, input.config);
    let placement = PlacementContext::new(
        PlanningInput {
            hot_locals,
            ..input
        },
        frame,
    );
    let placed_ops = place_semantic_ops(semantic, &placement);
    let cache_safe_ops = insert_cache_transfers(placed_ops, semantic.results, &placement);
    let group_inputs = build_group_inputs(semantic, &cache_safe_ops, &placement.input.policy);
    let ops = assign_rotations(
        prepend_init_locals(cache_safe_ops, placement.input.hot_locals, placement.frame),
        &placement,
    );
    let groups = super::group::plan_groups(&group_inputs);

    PlannedProgram {
        frame: placement.frame,
        hot_locals: placement.input.hot_locals,
        ops,
        groups,
    }
}

fn prepend_init_locals(
    mut ops: Vec<PlacedOp>,
    hot_locals: Option<HotLocalPlan>,
    frame: FrameLayoutPlan,
) -> Vec<PlacedOp> {
    let effective = hot_locals.map(|plan| plan.effective()).unwrap_or([None; 3]);
    let init = PlacedOp {
        kind: PlannedOpKind::InitLocals {
            l0: frame.local_slot(effective[0].unwrap_or(0) as u16),
            l1: frame.local_slot(effective[1].unwrap_or(1) as u16),
            l2: frame.local_slot(effective[2].unwrap_or(2) as u16),
        },
        window: 0,
        height: 0,
        alt: None,
    };

    let mut with_init = Vec::with_capacity(ops.len() + 1);
    with_init.push(init);
    with_init.append(&mut ops);
    with_init
}

#[derive(Clone, Debug)]
struct PlacementContext {
    input: PlanningInput,
    frame: FrameLayoutPlan,
}

impl PlacementContext {
    #[inline]
    fn new(input: PlanningInput, frame: FrameLayoutPlan) -> Self {
        Self { input, frame }
    }
}

fn plan_frame_layout(semantic: &SemanticProgram, config: PlanConfig) -> FrameLayoutPlan {
    let backend_reserved = (config.hot_local_count as u16).saturating_sub(semantic.local_count);
    let planner = FramePlanner::new(semantic.local_count).reserve_backend_reserved(backend_reserved);
    let (planner, _) = planner.reserve_operands(semantic.max_stack_height);
    planner.finish()
}

fn select_hot_locals(
    semantic: &SemanticProgram,
    config: PlanConfig,
) -> Option<HotLocalPlan> {
    if config.hot_local_count == 0 {
        None
    } else {
        let mut selected = [None; super::hot_local::HOT_LOCAL_COUNT];
        let count = core::cmp::min(
            semantic.local_count as usize,
            core::cmp::min(
                config.hot_local_count as usize,
                super::hot_local::HOT_LOCAL_COUNT,
            ),
        );
        let mut i = 0;
        while i < count {
            selected[i] = Some(i as u32);
            i += 1;
        }
        Some(HotLocalPlan::new(
            selected,
            selected,
        ))
    }
}

fn build_group_inputs(
    semantic: &SemanticProgram,
    ops: &[PlacedOp],
    policy: &PlanPolicy,
) -> Vec<GroupInputOp> {
    semantic
        .ops
        .iter()
        .zip(ops.iter())
        .map(|(semantic_op, placed_op)| GroupInputOp {
            can_start: policy.can_start(semantic_op) && groupable_kind(&placed_op.kind),
            can_extend: policy.can_extend(0, semantic_op) && groupable_kind(&placed_op.kind),
            terminator: terminator_for(policy, semantic_op, &placed_op.kind),
            alt: semantic_op.alt,
            br_table_targets: collect_br_table_targets(semantic_op),
        })
        .collect()
}

fn place_semantic_ops(semantic: &SemanticProgram, placement: &PlacementContext) -> Vec<PlacedOp> {
    semantic
        .ops
        .iter()
        .map(|op| place_semantic_op(op, placement))
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ControlFrameKind {
    Function,
    Block,
    Loop,
    If,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ControlFrame {
    kind: ControlFrameKind,
    start_height: u16,
    params: u16,
    results: u16,
}

#[derive(Clone, Debug)]
struct CachePlanState {
    height: u16,
    spill_depth: u16,
    unreachable: bool,
    control: Vec<ControlFrame>,
}

impl CachePlanState {
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
            }],
        }
    }
}

fn insert_cache_transfers(
    ops: Vec<PlacedOp>,
    function_results: u16,
    placement: &PlacementContext,
) -> Vec<PlacedOp> {
    let mut state = CachePlanState::new(function_results);
    let mut out = Vec::with_capacity(ops.len());

    for op in ops {
        insert_cache_safe_op(&mut out, &mut state, placement.frame, op);
    }

    out
}

fn assign_rotations(ops: Vec<PlacedOp>, _placement: &PlacementContext) -> Vec<PlannedOp> {
    ops.into_iter()
        .map(|op| PlannedOp {
            kind: op.kind,
            rotation: TosRotation::new(op.window),
            height: op.height,
            alt: op.alt,
        })
        .collect()
}

fn place_semantic_op(op: &SemanticOp, placement: &PlacementContext) -> PlacedOp {
    let frame = placement.frame;
    let kind = match &op.kind {
        SemanticOpKind::Core(kind) => PlannedOpKind::Core(kind.clone()),
        SemanticOpKind::LocalGet { idx } => PlannedOpKind::LocalGet {
            local: place_local(*idx, placement),
        },
        SemanticOpKind::LocalSet { idx } => PlannedOpKind::LocalSet {
            local: place_local(*idx, placement),
        },
        SemanticOpKind::LocalTee { idx } => PlannedOpKind::LocalTee {
            local: place_local(*idx, placement),
        },
        SemanticOpKind::Block { params, results } => {
            PlannedOpKind::Marker(PlannedMarkerKind::Block {
                params: *params,
                results: *results,
            })
        }
        SemanticOpKind::Loop { params, results } => {
            PlannedOpKind::Marker(PlannedMarkerKind::Loop {
                params: *params,
                results: *results,
            })
        }
        SemanticOpKind::If { params, results } => {
            PlannedOpKind::Marker(PlannedMarkerKind::If {
                params: *params,
                results: *results,
            })
        }
        SemanticOpKind::Else => PlannedOpKind::Marker(PlannedMarkerKind::Else),
        SemanticOpKind::End => PlannedOpKind::Marker(PlannedMarkerKind::End),
        SemanticOpKind::Br { stack_drop, arity } => PlannedOpKind::Branch {
            kind: PlannedBranchKind::Br,
            condition_slot: None,
            payload: branch_payload(frame, *stack_drop, *arity),
            target: op.alt,
        },
        SemanticOpKind::BrIf { stack_drop, arity } => PlannedOpKind::Branch {
            kind: PlannedBranchKind::BrIf,
            condition_slot: Some(frame.operand_slot(0)),
            payload: branch_payload(frame, *stack_drop, *arity),
            target: op.alt,
        },
        SemanticOpKind::BrTable { entries } => PlannedOpKind::BrTable {
            index_slot: frame.operand_slot(0),
            payload: Some(FrameSpan::single(frame.operand_slot(1))),
            entries: entries
                .iter()
                .map(|entry| PlannedBrTableEntry {
                    target: entry.target,
                    payload: branch_payload(frame, entry.stack_drop, entry.arity),
                })
                .collect(),
        },
        SemanticOpKind::CallExternal {
            func_idx,
            params,
            results,
        } => PlannedOpKind::CallExternal {
            func_idx: *func_idx,
            args: call_args(frame, *params),
            results: call_results(frame, *results),
        },
        SemanticOpKind::CallInternal {
            callee,
            params,
            results,
        } => PlannedOpKind::CallInternal {
            callee: *callee,
            args: call_args(frame, *params),
            results: call_results(frame, *results),
        },
        SemanticOpKind::CallIndirect {
            type_idx,
            table_idx,
            params,
            results,
        } => PlannedOpKind::CallIndirect {
            type_idx: *type_idx,
            table_idx: *table_idx,
            index_slot: frame.operand_slot(0),
            args: call_args(frame, *params),
            results: call_results(frame, *results),
        },
        SemanticOpKind::ReturnVoid => PlannedOpKind::Return { results: None },
        SemanticOpKind::ReturnOne => PlannedOpKind::Return {
            results: Some(FrameSpan::single(frame.operand_slot(0))),
        },
        SemanticOpKind::Return { arity } => PlannedOpKind::Return {
            results: Some(call_results(frame, *arity)),
        },
    };

    PlacedOp {
        kind,
        window: 0,
        height: 0,
        alt: op.alt,
    }
}

fn place_local(idx: u16, placement: &PlacementContext) -> PlannedLocal {
    if let Some(hot) = placement
        .input
        .hot_locals
        .and_then(|plan| plan.resolve(idx as u32))
    {
        PlannedLocal::Hot(hot)
    } else {
        PlannedLocal::Frame(placement.frame.local_slot(idx))
    }
}

fn groupable_kind(kind: &PlannedOpKind) -> bool {
    !matches!(
        kind,
        PlannedOpKind::Marker(_)
            | PlannedOpKind::Spill(_)
    )
}

fn terminator_for(
    policy: &PlanPolicy,
    semantic_op: &SemanticOp,
    placed_kind: &PlannedOpKind,
) -> Option<GroupTerminator> {
    if !policy.is_terminator(semantic_op) {
        return None;
    }

    Some(match placed_kind {
        PlannedOpKind::Branch {
            kind: PlannedBranchKind::Br,
            ..
        } => GroupTerminator::Br,
        PlannedOpKind::Branch {
            kind: PlannedBranchKind::BrIf,
            ..
        } => GroupTerminator::BrIf,
        PlannedOpKind::BrTable { .. } => GroupTerminator::BrTable,
        PlannedOpKind::Marker(PlannedMarkerKind::If { .. }) => GroupTerminator::If,
        PlannedOpKind::Return { results: None } => GroupTerminator::ReturnVoid,
        PlannedOpKind::Return {
            results: Some(span),
        } if span.count == 1 => GroupTerminator::ReturnOne,
        PlannedOpKind::Return { .. } => GroupTerminator::Return,
        _ => GroupTerminator::Unreachable,
    })
}

fn collect_br_table_targets(op: &SemanticOp) -> Vec<SemanticTarget> {
    match &op.kind {
        SemanticOpKind::BrTable { entries } => entries
            .iter()
            .filter_map(|entry| entry.target)
            .collect(),
        _ => Vec::new(),
    }
}

fn branch_payload(frame: FrameLayoutPlan, stack_drop: u32, arity: u16) -> Option<FrameSpan> {
    if arity == 0 {
        None
    } else {
        Some(FrameSpan::new(
            frame.operand_slot(stack_drop as u16),
            arity,
        ))
    }
}

fn call_args(frame: FrameLayoutPlan, params: u16) -> FrameSpan {
    FrameSpan::new(frame.operand_slot(0), params)
}

fn call_results(frame: FrameLayoutPlan, results: u16) -> FrameSpan {
    FrameSpan::new(frame.operand_slot(0), results)
}

fn insert_cache_safe_op(
    out: &mut Vec<PlacedOp>,
    state: &mut CachePlanState,
    frame: FrameLayoutPlan,
    op: PlacedOp,
) {
    let alt = op.alt;
    match op.kind {
        PlannedOpKind::Core(kind) => {
            if matches!(kind, CoreOpKind::Unreachable) {
                emit_with_state(out, state, PlannedOpKind::Core(kind), 0, alt);
                state.unreachable = true;
                return;
            }

            let (pop, push) = crate::vm::wasm::core_op::stack_effect(&kind);
            if pop == 0 && push > 0 {
                spill_before_push(out, state, frame);
            }
            fill_for_operands(out, state, frame, pop as u16);
            emit_with_state(
                out,
                state,
                PlannedOpKind::Core(kind.clone()),
                raw_window_for_height(state.height),
                alt,
            );
            apply_stack_effect(state, pop as u16, push as u16);
        }
        PlannedOpKind::LocalGet { local } => {
            spill_before_push(out, state, frame);
            emit_with_state(
                out,
                state,
                PlannedOpKind::LocalGet { local },
                raw_window_for_height(state.height),
                alt,
            );
            if !state.unreachable {
                state.height += 1;
            }
        }
        PlannedOpKind::LocalSet { local } => {
            fill_for_operands(out, state, frame, 1);
            emit_with_state(
                out,
                state,
                PlannedOpKind::LocalSet { local },
                raw_window_for_height(state.height),
                alt,
            );
            if !state.unreachable {
                state.height = state.height.saturating_sub(1);
                state.spill_depth = state.spill_depth.min(state.height);
            }
        }
        PlannedOpKind::LocalTee { local } => {
            fill_for_operands(out, state, frame, 1);
            emit_with_state(
                out,
                state,
                PlannedOpKind::LocalTee { local },
                raw_window_for_height(state.height),
                alt,
            );
        }
        PlannedOpKind::Marker(PlannedMarkerKind::Block { params, results }) => {
            emit_with_state(
                out,
                state,
                PlannedOpKind::Marker(PlannedMarkerKind::Block { params, results }),
                0,
                alt,
            );
            state.control.push(ControlFrame {
                kind: ControlFrameKind::Block,
                start_height: state.height.saturating_sub(params),
                params,
                results,
            });
        }
        PlannedOpKind::Marker(PlannedMarkerKind::Loop { params, results }) => {
            spill_all(out, state, frame);
            emit_with_state(
                out,
                state,
                PlannedOpKind::Marker(PlannedMarkerKind::Loop { params, results }),
                0,
                alt,
            );
            state.control.push(ControlFrame {
                kind: ControlFrameKind::Loop,
                start_height: state.height.saturating_sub(params),
                params,
                results,
            });
        }
        PlannedOpKind::Marker(PlannedMarkerKind::If { params, results }) => {
            fill_for_operands(out, state, frame, 1);
            spill_all_except_top(out, state, frame, 1);
            emit_with_state(
                out,
                state,
                PlannedOpKind::Marker(PlannedMarkerKind::If { params, results }),
                raw_window_for_height(state.height),
                alt,
            );
            if !state.unreachable {
                state.height = state.height.saturating_sub(1);
                state.spill_depth = state.height;
            }
            state.control.push(ControlFrame {
                kind: ControlFrameKind::If,
                start_height: state.height.saturating_sub(params),
                params,
                results,
            });
        }
        PlannedOpKind::Marker(PlannedMarkerKind::Else) => {
            spill_all(out, state, frame);
            emit_with_state(
                out,
                state,
                PlannedOpKind::Marker(PlannedMarkerKind::Else),
                0,
                alt,
            );
            if let Some(frame) = state.control.last().copied() {
                state.height = frame.start_height + frame.params;
                state.spill_depth = state.height;
                state.unreachable = false;
            }
        }
        PlannedOpKind::Marker(PlannedMarkerKind::End) => {
            spill_all(out, state, frame);
            emit_with_state(
                out,
                state,
                PlannedOpKind::Marker(PlannedMarkerKind::End),
                0,
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
            spill_all(out, state, frame);
            emit_with_state(
                out,
                state,
                PlannedOpKind::Branch {
                    kind: PlannedBranchKind::Br,
                    condition_slot,
                    payload,
                    target,
                },
                0,
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
            fill_for_operands(out, state, frame, 1);
            spill_all_except_top(out, state, frame, 1);
            emit_with_state(
                out,
                state,
                PlannedOpKind::Branch {
                    kind: PlannedBranchKind::BrIf,
                    condition_slot,
                    payload,
                    target,
                },
                raw_window_for_height(state.height),
                alt,
            );
            if !state.unreachable {
                state.height = state.height.saturating_sub(1);
                state.spill_depth = state.height;
            }
        }
        PlannedOpKind::BrTable {
            index_slot,
            payload,
            entries,
        } => {
            fill_for_operands(out, state, frame, 1);
            spill_all_except_top(out, state, frame, 1);
            emit_with_state(
                out,
                state,
                PlannedOpKind::BrTable {
                    index_slot,
                    payload,
                    entries,
                },
                raw_window_for_height(state.height),
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
            spill_all(out, state, frame);
            emit_with_state(
                out,
                state,
                PlannedOpKind::CallExternal {
                    func_idx,
                    args,
                    results,
                },
                0,
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
            spill_all(out, state, frame);
            emit_with_state(
                out,
                state,
                PlannedOpKind::CallInternal {
                    callee,
                    args,
                    results,
                },
                0,
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
            spill_all(out, state, frame);
            emit_with_state(
                out,
                state,
                PlannedOpKind::CallIndirect {
                    type_idx,
                    table_idx,
                    index_slot,
                    args,
                    results,
                },
                0,
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
            spill_all(out, state, frame);
            emit_with_state(out, state, PlannedOpKind::Return { results }, 0, alt);
            state.unreachable = true;
        }
        PlannedOpKind::Spill(artifact) => {
            emit_with_state(out, state, PlannedOpKind::Spill(artifact), 0, alt);
        }
        PlannedOpKind::InitLocals { l0, l1, l2 } => {
            emit_with_state(
                out,
                state,
                PlannedOpKind::InitLocals { l0, l1, l2 },
                0,
                alt,
            );
        }
    }
}

fn emit_with_state(
    out: &mut Vec<PlacedOp>,
    state: &CachePlanState,
    kind: PlannedOpKind,
    window: u8,
    alt: Option<SemanticTarget>,
) {
    out.push(PlacedOp {
        kind,
        window,
        height: state.height,
        alt,
    });
}

fn apply_stack_effect(state: &mut CachePlanState, pop: u16, push: u16) {
    if state.unreachable {
        return;
    }
    state.height = state.height.saturating_sub(pop).saturating_add(push);
    state.spill_depth = state.spill_depth.min(state.height);
}

fn spill_before_push(out: &mut Vec<PlacedOp>, state: &mut CachePlanState, frame: FrameLayoutPlan) {
    if state.unreachable {
        return;
    }
    let tos_count = state.height.saturating_sub(state.spill_depth);
    if tos_count < 4 {
        return;
    }
    emit_cache_transfer(
        out,
        state,
        SpillArtifact::spill(FrameSpan::new(frame.operand_slot(state.spill_depth), 1)),
        ((state.spill_depth + 1) & 0b11) as u8,
    );
    state.spill_depth += 1;
}

fn fill_for_operands(
    out: &mut Vec<PlacedOp>,
    state: &mut CachePlanState,
    frame: FrameLayoutPlan,
    operand_count: u16,
) {
    if state.unreachable {
        return;
    }
    let min_spill_depth = state.height.saturating_sub(operand_count);
    if state.spill_depth <= min_spill_depth {
        return;
    }
    let fill_count = state.spill_depth - min_spill_depth;
    emit_cache_transfer(
        out,
        state,
        SpillArtifact::fill(FrameSpan::new(
            frame.operand_slot(state.spill_depth - fill_count),
            fill_count,
        )),
        (state.spill_depth & 0b11) as u8,
    );
    state.spill_depth -= fill_count;
}

fn spill_all(out: &mut Vec<PlacedOp>, state: &mut CachePlanState, frame: FrameLayoutPlan) {
    if state.unreachable || state.spill_depth >= state.height {
        return;
    }
    let count = state.height - state.spill_depth;
    emit_cache_transfer(
        out,
        state,
        SpillArtifact::spill(FrameSpan::new(frame.operand_slot(state.spill_depth), count)),
        ((state.spill_depth + count) & 0b11) as u8,
    );
    state.spill_depth = state.height;
}

fn spill_all_except_top(
    out: &mut Vec<PlacedOp>,
    state: &mut CachePlanState,
    frame: FrameLayoutPlan,
    keep_top: u16,
) {
    if state.unreachable {
        return;
    }
    let tos_count = state.height.saturating_sub(state.spill_depth);
    let count = tos_count.saturating_sub(keep_top);
    if count == 0 {
        return;
    }
    emit_cache_transfer(
        out,
        state,
        SpillArtifact::spill(FrameSpan::new(frame.operand_slot(state.spill_depth), count)),
        ((state.spill_depth + count) & 0b11) as u8,
    );
    state.spill_depth += count;
}

fn emit_cache_transfer(
    out: &mut Vec<PlacedOp>,
    state: &CachePlanState,
    artifact: SpillArtifact,
    window: u8,
) {
    out.push(PlacedOp {
        kind: PlannedOpKind::Spill(artifact),
        window,
        height: state.height,
        alt: None,
    });
}

#[inline]
fn raw_window_for_height(height: u16) -> u8 {
    (height & 0b11) as u8
}
