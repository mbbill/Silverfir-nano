//! Joint-plan builder from first-pass SSA.
//!
//! Under `ALGORITHM4`, the builder has a simple split of responsibilities:
//! - transient legality is still solved per semantic op
//! - public cached-local residency is solved separately on a region tree
//! - block boundaries see one fixed public set, not a tentative per-block seed

use crate::collections;

use tracked_alloc::collections::BTreeMap;

use crate::{
    error::WasmError,
    value_type::ValueType,
    vm::{
        backend::BackendConfig,
        middle::{
            cfg::SemanticCfg,
            discipline::{self, StructuralAction},
            frame::{FrameLayoutPlan, FrameSlot},
        },
        wasm::{
            primitive_op,
            primitive_op::PrimitiveOpKind,
            semantic_ir::{SemanticOp, SemanticOpKind, SemanticProgram},
        },
    },
};

use super::{
    entry_region::analyze_block_local_summaries,
    exact,
    facts::{BlockPlan, EntryState, FunctionPlan, RowSpan},
    region_solver::{solve_public_cache_sets, ResidencyPolicy},
};

pub(crate) fn build_plan(
    semantic: &SemanticProgram,
    cfg: &SemanticCfg,
    frame: FrameLayoutPlan,
    config: BackendConfig,
) -> Result<FunctionPlan, WasmError> {
    let gp_dynamic_budget = config.allocatable_gp_dynamic_budget();
    let fp_dynamic_budget = config.fp_dynamic_budget;

    // The lightweight pass is the only pass that needs to simulate every
    // semantic op boundary. It keeps full EntryState only at CFG block entries,
    // not at every semantic op.
    let mut lightweight = compute_lightweight_plan(semantic, cfg, config.gp_unit_bytes);
    // Lift entry-block peak pressure to cover incoming-param register
    // occupancy. The stack-pressure simulation above tracks the operand
    // stack only; function params arrive in caller-set GP/FP arg lanes and
    // are live in those regs until the body consumes them. Without this
    // lift, cap(R_root) overshoots on arches with tight GP budgets — on
    // x86_64 (7 allocatable, 4 GP arg lanes) ALGORITHM4 was selecting
    // residents the machine layer could not bind, surfacing as the
    // `no free cache register` failure on coremark-class functions.
    let (entry_gp_param_units, entry_fp_param_units) =
        entry_block_param_register_footprint(semantic, &config);
    let entry_block = cfg.entry.as_usize();
    if let Some(slot) = lightweight.peak_gp.get_mut(entry_block) {
        *slot = (*slot).max(entry_gp_param_units);
    }
    if let Some(slot) = lightweight.peak_fp.get_mut(entry_block) {
        *slot = (*slot).max(entry_fp_param_units);
    }
    let block_local_summaries = analyze_block_local_summaries(semantic, cfg, frame);

    let policy = ResidencyPolicy::from_env()?;
    let solution = solve_public_cache_sets(
        semantic,
        cfg,
        config.gp_unit_bytes,
        gp_dynamic_budget,
        fp_dynamic_budget,
        &lightweight.peak_gp,
        &lightweight.peak_fp,
        &block_local_summaries,
        policy,
    );
    let block_entry_cached_locals = &solution;
    let LightweightPlanOutput {
        peak_gp: _,
        peak_fp: _,
        block_entries,
    } = lightweight;
    // Flatten each block's planned-resident set into one arena; the block keeps
    // only an (offset, len) span (the a3a7a102 lesson — per-block `Vec` headers
    // dominate planner memory on multi-thousand-block functions).
    let mut resident_arena: collections::Vec<FrameSlot> = collections::Vec::new();
    let blocks = block_entries
        .into_iter()
        .enumerate()
        .map(|(block_index, entry)| {
            let residents = block_entry_cached_locals
                .get(block_index)
                .map(|r| r.as_slice())
                .unwrap_or(&[]);
            let offset = resident_arena.len() as u32;
            resident_arena.extend_from_slice(residents);
            BlockPlan {
                entry,
                planned_residents: RowSpan {
                    offset,
                    len: residents.len() as u32,
                },
                ..Default::default()
            }
        })
        .collect();
    let mut plan = FunctionPlan {
        gp_unit_bytes: config.gp_unit_bytes,
        gp_dynamic_budget,
        fp_dynamic_budget,
        blocks,
        resident_arena,
        repair_pool: collections::Vec::new(),
        repair_slot_arena: collections::Vec::new(),
        row_arena: collections::Vec::new(),
        repair_index_arena: collections::Vec::new(),
    };

    // Pass D: exact per-block cache boundaries + per-edge repair actions. The
    // machine-facing requirement + preferred-preserved rows are NOT derived here;
    // they are computed over the FINAL SSA in `middle::final_signals` (a
    // pre-cleanup classification cannot see block merges).
    exact::compute_exact_plan(semantic, cfg, frame, config, &mut plan)?;

    Ok(plan)
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
    param_types: collections::Vec<ValueType>,
    result_types: collections::Vec<ValueType>,
}

#[derive(Clone, Debug)]
struct PrepareState {
    height: u16,
    spill_depth: u16,
    unreachable: bool,
    control: collections::Vec<ControlFrame>,
    type_stack: collections::Vec<ValueType>,
    local_types: collections::Vec<ValueType>,
}

impl PrepareState {
    fn new(
        results: u16,
        local_types: collections::Vec<ValueType>,
        result_types: collections::Vec<ValueType>,
        _cache_slots: usize,
    ) -> Self {
        Self {
            height: 0,
            spill_depth: 0,
            unreachable: false,
            control: collections::vec![ControlFrame {
                kind: ControlFrameKind::Function,
                start_height: 0,
                params: 0,
                results,
                entered_unreachable: false,
                param_types: collections::Vec::new(),
                result_types: normalized_result_types(results, Some(result_types.as_slice())),
            }],
            type_stack: collections::Vec::new(),
            local_types,
        }
    }

    fn mark_unreachable(&mut self) {
        if let Some(frame) = self.control.last().cloned() {
            self.height = frame.start_height.saturating_add(frame.results);
            self.spill_depth = self.height;
            self.type_stack.truncate(frame.start_height as usize);
            self.type_stack.extend_from_slice(&frame.result_types);
        }
        self.unreachable = true;
    }

    fn local_type(&self, idx: u16) -> ValueType {
        self.local_types
            .get(idx as usize)
            .copied()
            .unwrap_or(ValueType::I64)
    }

    fn types_at(&self, start: u16, count: u16) -> collections::Vec<ValueType> {
        if count == 0 {
            return collections::Vec::new();
        }
        let start = start as usize;
        let end = start + count as usize;
        if end <= self.type_stack.len() {
            self.type_stack[start..end].to_vec().into()
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

fn apply_semantic_effect(
    op: &SemanticOp,
    op_index: usize,
    op_result_types: &BTreeMap<usize, collections::Vec<ValueType>>,
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
                ValueType::I64
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
                op_result_types
                    .get(&op_index)
                    .and_then(|v| v.first().copied())
                    .unwrap_or(ValueType::I64)
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
        SemanticOpKind::Block { params, results }
        | SemanticOpKind::Loop { params, results }
        | SemanticOpKind::TryTable {
            params, results, ..
        } => {
            let sh = if state.unreachable {
                state.height
            } else {
                state.height.saturating_sub(*params)
            };
            let param_types = if state.unreachable {
                collections::Vec::new()
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
                result_types: control_result_types(*results, op_index, op_result_types),
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
                collections::Vec::new()
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
                result_types: control_result_types(*results, op_index, op_result_types),
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
                        ControlFrameKind::Structured => {
                            state.spill_depth.min(end_height).max(frame.start_height)
                        }
                    };
                    state.type_stack.truncate(frame.start_height as usize);
                    state.type_stack.extend_from_slice(&frame.result_types);
                    state.unreachable = false;
                }
            }
        }
        SemanticOpKind::Br { .. } => state.mark_unreachable(),
        SemanticOpKind::BrIf { .. } => {
            if !state.unreachable {
                state.height = state.height.saturating_sub(1);
                // `br_if` only consumes the condition on the fallthrough path.
                // Any values that were already live below the condition remain
                // live for the following instructions.
                state.spill_depth = state.spill_depth.min(state.height);
                state.type_stack.truncate(state.height as usize);
            }
        }
        SemanticOpKind::BrOnNull { ref_type, .. } => {
            if !state.unreachable {
                let base = state.height.saturating_sub(1) as usize;
                state.type_stack.truncate(base);
                state.type_stack.push(*ref_type);
            }
        }
        SemanticOpKind::BrOnNonNull { .. } => {
            if !state.unreachable {
                state.height = state.height.saturating_sub(1);
                state.spill_depth = state.spill_depth.min(state.height);
                state.type_stack.truncate(state.height as usize);
            }
        }
        SemanticOpKind::BrOnCast { fail_type, .. } => {
            if !state.unreachable {
                let base = state.height.saturating_sub(1) as usize;
                state.type_stack.truncate(base);
                state.type_stack.push(*fail_type);
            }
        }
        SemanticOpKind::BrOnCastFail { cast_type, .. } => {
            if !state.unreachable {
                let base = state.height.saturating_sub(1) as usize;
                state.type_stack.truncate(base);
                state.type_stack.push(*cast_type);
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
        SemanticOpKind::CallDirect {
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
        SemanticOpKind::CallRef {
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
        SemanticOpKind::AllocExnRef { .. } => {
            if !state.unreachable {
                state.height = state.height.saturating_add(1);
                state.spill_depth = state.height;
                push_result_types(&mut state.type_stack, 1, op_index, op_result_types);
            }
        }
        SemanticOpKind::ReturnVoid
        | SemanticOpKind::ReturnOne
        | SemanticOpKind::Return { .. }
        | SemanticOpKind::ReturnCallDirect { .. }
        | SemanticOpKind::ReturnCallIndirect { .. }
        | SemanticOpKind::ReturnCallRef { .. } => {
            state.mark_unreachable();
        }
        SemanticOpKind::Throw { arity, .. } => {
            if !state.unreachable {
                state.height = state.height.saturating_sub(*arity);
                state.spill_depth = state.height;
                state.type_stack.truncate(state.height as usize);
            }
            state.mark_unreachable();
        }
        SemanticOpKind::ThrowRef => {
            if !state.unreachable {
                state.height = state.height.saturating_sub(1);
                state.spill_depth = state.height;
                state.type_stack.truncate(state.height as usize);
            }
            state.mark_unreachable();
        }
    }
}

fn push_result_types(
    type_stack: &mut collections::Vec<ValueType>,
    results: u16,
    op_index: usize,
    op_result_types: &BTreeMap<usize, collections::Vec<ValueType>>,
) {
    if let Some(types) = op_result_types.get(&op_index) {
        type_stack.extend_from_slice(types);
    } else {
        for _ in 0..results {
            type_stack.push(ValueType::I64);
        }
    }
}

fn normalized_result_types(
    results: u16,
    result_types: Option<&[ValueType]>,
) -> collections::Vec<ValueType> {
    if results == 0 {
        return collections::Vec::new();
    }
    if let Some(types) = result_types {
        if types.len() == results as usize {
            return types.to_vec().into();
        }
    }
    collections::vec![ValueType::I64; results as usize]
}

fn control_result_types(
    results: u16,
    op_index: usize,
    op_result_types: &BTreeMap<usize, collections::Vec<ValueType>>,
) -> collections::Vec<ValueType> {
    normalized_result_types(
        results,
        op_result_types
            .get(&op_index)
            .map(collections::Vec::as_slice),
    )
}

fn apply_stack_effect_typed(state: &mut PrepareState, pop: u16, push: u16, result_ty: ValueType) {
    if state.unreachable {
        return;
    }
    state.height = state.height.saturating_sub(pop).saturating_add(push);
    state.spill_depth = state.spill_depth.min(state.height);
    let new_base = state.height.saturating_sub(push) as usize;
    state.type_stack.truncate(new_base);
    for _ in 0..push {
        state.type_stack.push(result_ty);
    }
}

/// Lightweight stack simulation result.
///
/// Computes peak transient pressure per block and compact per-op entry points
/// in a single pass, without storing full `EntryState` snapshots per op.
pub(crate) struct LightweightPlanOutput {
    /// Per-block peak GP transient pressure in budget units.
    pub peak_gp: collections::Vec<usize>,
    /// Per-block peak FP transient pressure in budget units.
    pub peak_fp: collections::Vec<usize>,
    /// Per-block entry `EntryState` with full `stack_types` for spilled value fills.
    pub block_entries: collections::Vec<EntryState>,
}

/// Simulate the backend's GP/FP arg-lane assignment to compute how many
/// register units are consumed by incoming params at function entry.
///
/// The machine ABI keeps only a contiguous suffix of params in registers so
/// the frame prefix remains canonical. Walk from the last param backward and
/// stop when a param cannot fit in the configured arg lanes, matching
/// `compute_param_locs`.
fn entry_block_param_register_footprint(
    semantic: &SemanticProgram,
    config: &BackendConfig,
) -> (usize, usize) {
    let param_count = usize::from(semantic.params).min(semantic.local_types.len());
    let mut gp_used = 0usize;
    let mut fp_used = 0usize;
    let gp_arg_lanes = usize::from(config.gp_arg_lanes);
    let fp_arg_lanes = usize::from(config.fp_arg_lanes);
    let gp_unit_bytes = config.gp_unit_bytes;
    for param_index in (0..param_count).rev() {
        let ty = semantic.local_types[param_index];
        match ty {
            ValueType::V128 => break,
            ValueType::F32 | ValueType::F64 => {
                if fp_used >= fp_arg_lanes {
                    break;
                }
                fp_used += 1;
            }
            ValueType::I64 if gp_unit_bytes < 8 => {
                // i64 on 32-bit GP consumes a pair of arg lanes; if either
                // half spills, the param goes to the frame entirely.
                if gp_used + 2 > gp_arg_lanes {
                    break;
                }
                gp_used += 2;
            }
            _ => {
                if gp_used >= gp_arg_lanes {
                    break;
                }
                gp_used += 1;
            }
        }
    }
    (gp_used, fp_used)
}

/// Compute per-op compact entry points and per-block peak transient pressure
/// using a lightweight stack simulation.
///
/// This replaces the heavy `prepare_semantic_ops` + `compute_block_peak_live_units`
/// pipeline. It reuses the same `PrepareState` machinery for stack simulation
/// but skips `ensure_capacity` (cache-dependent spill decisions). The resulting
/// peak pressure is a conservative upper bound since `ensure_capacity` would only
/// further spill (reducing the live window).
pub(crate) fn compute_lightweight_plan(
    semantic: &SemanticProgram,
    cfg: &SemanticCfg,
    gp_unit_bytes: u8,
) -> LightweightPlanOutput {
    use crate::vm::middle::budget::count_live_bank_budget_units;

    let mut state = PrepareState::new(
        semantic.results,
        semantic.local_types.clone(),
        semantic.result_types.clone(),
        0, // no cache slots
    );

    let mut peak_gp = collections::vec![0usize; cfg.blocks.len()];
    let mut peak_fp = collections::vec![0usize; cfg.blocks.len()];
    let mut block_entries = collections::vec![EntryState::default(); cfg.blocks.len()];

    for (op_index, semantic_op) in semantic.ops.iter().enumerate() {
        let block_index = cfg
            .block_for_semantic_index(op_index)
            .map(|id| id.as_usize())
            .unwrap_or(0);
        let is_block_start = cfg
            .blocks
            .get(block_index)
            .map(|block| block.range.start == op_index)
            .unwrap_or(false);
        if is_block_start {
            // Capture block entry state with full stack_types for spilled value fills.
            let spill = state.spill_depth.min(state.height);
            // Invariant that keeps `EntryState::live_types()` (the slice view
            // `stack_types[spill_depth..]`) exact: the type stack is exactly as
            // tall as the height.
            debug_assert_eq!(
                state.type_stack.len(),
                state.height as usize,
                "block-entry type stack length must equal stack height"
            );
            block_entries[block_index] = EntryState {
                stack_height: state.height,
                spill_depth: spill,
                stack_types: state.type_stack.clone(),
            };
        }

        // Apply structural prefix: fill operands, spill at control flow boundaries.
        // Skip ensure_capacity — it depends on cache state and only further reduces
        // the live window, so omitting it gives a conservative upper bound on pressure.
        apply_structural_prefix(semantic_op, &mut state);

        // Measure live-window pressure after prefix (= "before" the op executes).
        let before_live = state.types_at(
            state.spill_depth,
            state.height.saturating_sub(state.spill_depth),
        );
        let (bg, bf) = count_live_bank_budget_units(&before_live, gp_unit_bytes);
        peak_gp[block_index] = peak_gp[block_index].max(bg);
        peak_fp[block_index] = peak_fp[block_index].max(bf);

        // Apply semantic effect (stack height + type changes).
        apply_semantic_effect(semantic_op, op_index, &semantic.op_result_types, &mut state);

        // Measure "after" pressure.
        let after_live = state.types_at(
            state.spill_depth,
            state.height.saturating_sub(state.spill_depth),
        );
        let (ag, af) = count_live_bank_budget_units(&after_live, gp_unit_bytes);
        peak_gp[block_index] = peak_gp[block_index].max(ag);
        peak_fp[block_index] = peak_fp[block_index].max(af);
    }

    LightweightPlanOutput {
        peak_gp,
        peak_fp,
        block_entries,
    }
}

/// Apply the structural (cache-independent) prefix effects for one op.
///
/// This mirrors the control-flow portions of `plan_prefix` but skips
/// `ensure_capacity` and cache management. It only applies fills and spills
/// that are determined by the op kind and Wasm structured control flow.
fn apply_structural_prefix(op: &SemanticOp, state: &mut PrepareState) {
    if matches!(
        &op.kind,
        SemanticOpKind::Primitive(PrimitiveOpKind::Unreachable)
    ) {
        return;
    }
    match discipline::structural_action(&op.kind) {
        StructuralAction::None => {}
        StructuralAction::PrimitiveFill { pop, .. } => measure_fill(state, pop),
        StructuralAction::FillKeepSpillRest(keep) => {
            measure_fill(state, keep);
            measure_spill_except_top(state, keep);
        }
        // The planner conservatively spills the whole live window for calls and
        // returns alike. The returns' spill is dead (mark_unreachable overwrites
        // spill_depth in apply_semantic_effect right after every return) but is
        // kept explicit; see the ReturnScalar note in the discipline table.
        StructuralAction::SpillAll
        | StructuralAction::PrepareCall { .. }
        | StructuralAction::ReturnScalar => measure_spill_all(state),
        StructuralAction::ElsePlannerFill => measure_else_fill(state),
    }
}

/// Fill operands by lowering spill_depth (state-only). Skips while unreachable.
fn measure_fill(state: &mut PrepareState, operand_count: u16) {
    if state.unreachable {
        return;
    }
    state.spill_depth = discipline::fill_target(state.height, state.spill_depth, operand_count);
}

/// Spill all live transients by raising spill_depth to height.
fn measure_spill_all(state: &mut PrepareState) {
    if state.unreachable {
        return;
    }
    state.spill_depth = discipline::spill_all_target(state.height);
}

/// Spill all except the top `keep_top` live values.
fn measure_spill_except_top(state: &mut PrepareState, keep_top: u16) {
    if state.unreachable {
        return;
    }
    state.spill_depth =
        discipline::spill_except_top_target(state.height, state.spill_depth, keep_top);
}

/// `else`: fill the enclosing frame's result arity, control-stack aware. Unlike
/// the other fills, this ignores `state.unreachable` — the frame checks guard it.
fn measure_else_fill(state: &mut PrepareState) {
    if let Some(frame_state) = state.control.last().cloned() {
        if frame_state.entered_unreachable || matches!(frame_state.kind, ControlFrameKind::Function)
        {
            return;
        }
        state.spill_depth =
            discipline::fill_target(state.height, state.spill_depth, frame_state.results);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn semantic_with_params(types: &[ValueType]) -> SemanticProgram {
        SemanticProgram {
            params: types.len() as u16,
            local_count: types.len() as u16,
            local_types: types.iter().copied().collect(),
            ..SemanticProgram::default()
        }
    }

    fn config_with_arg_lanes(
        gp_unit_bytes: u8,
        gp_arg_lanes: u8,
        fp_arg_lanes: u8,
    ) -> BackendConfig {
        BackendConfig::with_volatility(
            gp_unit_bytes,
            gp_arg_lanes,
            0,
            0,
            fp_arg_lanes,
            0,
            gp_arg_lanes,
            fp_arg_lanes,
            false,
            0,
        )
    }

    #[test]
    fn entry_param_footprint_uses_backend_suffix_selection_for_gp32_pairs() {
        let semantic = semantic_with_params(&[ValueType::I32, ValueType::I64, ValueType::I64]);
        let config = config_with_arg_lanes(4, 4, 0);

        assert_eq!(
            entry_block_param_register_footprint(&semantic, &config),
            (4, 0),
            "the two trailing i64 params occupy all four GP arg lanes"
        );
    }

    #[test]
    fn entry_param_footprint_stops_at_frame_only_param_in_suffix() {
        let semantic = semantic_with_params(&[
            ValueType::I32,
            ValueType::V128,
            ValueType::I32,
            ValueType::F64,
        ]);
        let config = config_with_arg_lanes(8, 4, 4);

        assert_eq!(
            entry_block_param_register_footprint(&semantic, &config),
            (1, 1),
            "only the contiguous register-param suffix after v128 stays in arg lanes"
        );
    }
}
