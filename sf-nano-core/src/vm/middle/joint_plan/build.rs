//! Joint-plan builder from first-pass SSA.
//!
//! Under `ALGORITHM4`, the builder has a simple split of responsibilities:
//! - transient legality is still solved per semantic op
//! - public cached-local residency is solved separately on a region tree
//! - block boundaries see one fixed public set, not a tentative per-block seed

use crate::collections;

use tracked_alloc::collections::BTreeMap;
#[cfg(test)]
use tracked_alloc::collections::BTreeSet;

use crate::{
    error::WasmError,
    value_type::ValueType,
    vm::{
        backend::BackendConfig,
        middle::{cfg::SemanticCfg, frame::FrameLayoutPlan},
        wasm::{
            primitive_op,
            primitive_op::PrimitiveOpKind,
            semantic_ir::{SemanticOp, SemanticOpKind, SemanticProgram},
        },
    },
};

#[cfg(test)]
use super::facts::{BlockTransientRegion, FirstAccessKind};
use super::{
    entry_region::{analyze_block_entry_regions, analyze_block_transient_regions},
    facts::{BlockPlan, CompactEntryPoint, EntryState, FunctionPlan, LocalOpKind, OpInfo},
    region_solver::solve_public_cache_sets,
};
#[cfg(test)]
use crate::vm::middle::budget::gp_value_budget_units;
#[cfg(test)]
use crate::vm::middle::{budget::count_live_bank_budget_units, frame::FrameSlot};

pub(crate) fn build_plan(
    semantic: &SemanticProgram,
    cfg: &SemanticCfg,
    frame: FrameLayoutPlan,
    config: BackendConfig,
) -> Result<FunctionPlan, WasmError> {
    let gp_dynamic_budget = config.allocatable_gp_dynamic_budget();
    let fp_dynamic_budget = config.fp_dynamic_budget;

    let op_info = build_op_info(semantic, cfg, frame);
    let semantic_entry_shapes = analyze_semantic_entry_shapes(semantic);
    let (block_local_summaries, block_stack_regions) =
        analyze_block_entry_regions(semantic, cfg, frame, &semantic_entry_shapes);
    let block_transient_regions =
        analyze_block_transient_regions(semantic, cfg, &semantic_entry_shapes);
    // Lightweight plan provides compact_entries and block entry states + peak
    // pressure for the region solver.
    let lightweight = compute_lightweight_plan(semantic, cfg, &op_info, config.gp_unit_bytes);

    let block_entry_cached_locals = solve_public_cache_sets(
        semantic,
        cfg,
        config.gp_unit_bytes,
        gp_dynamic_budget,
        fp_dynamic_budget,
        &lightweight.peak_gp,
        &lightweight.peak_fp,
        &block_local_summaries,
    );
    let blocks = cfg
        .blocks
        .iter()
        .enumerate()
        .map(|(block_index, _block)| BlockPlan {
            entry: lightweight
                .block_entries
                .get(block_index)
                .cloned()
                .unwrap_or_default(),
            tentative_entry_cached_locals: block_entry_cached_locals
                .get(block_index)
                .cloned()
                .unwrap_or_default(),
        })
        .collect();
    let plan = FunctionPlan {
        gp_unit_bytes: config.gp_unit_bytes,
        gp_dynamic_budget,
        fp_dynamic_budget,
        local_slot_types: semantic.local_types.clone(),
        compact_entries: lightweight.compact_entries,
        op_info,
        block_local_summaries,
        block_stack_regions,
        block_transient_regions,
        blocks,
    };

    Ok(plan)
}

fn build_op_info(
    semantic: &SemanticProgram,
    cfg: &SemanticCfg,
    frame: FrameLayoutPlan,
) -> collections::Vec<OpInfo> {
    let mut out = collections::vec![OpInfo::default(); semantic.ops.len()];
    for (block_index, block) in cfg.blocks.iter().enumerate() {
        for (block_offset, semantic_index) in block.range.clone().enumerate() {
            let local_op = match semantic.ops[semantic_index].kind {
                SemanticOpKind::LocalGet { idx } => Some((frame.local_slot(idx), LocalOpKind::Get)),
                SemanticOpKind::LocalSet { idx } => Some((frame.local_slot(idx), LocalOpKind::Set)),
                SemanticOpKind::LocalTee { idx } => Some((frame.local_slot(idx), LocalOpKind::Tee)),
                _ => None,
            };
            out[semantic_index] = OpInfo {
                block_index: block_index as u32,
                block_offset: block_offset as u16,
                is_block_start: block_offset == 0,
                local_op,
            };
        }
    }
    out
}

/// Compute the exact semantic stack shape at every op boundary without making
/// any planner choices.
///
/// This pass is intentionally policy-free. It exists so the later block-local
/// ranking passes can reason about entry-stack values and transient symbols
/// using the full semantic stack shape, before any spill/cache decisions are
/// introduced.
fn analyze_semantic_entry_shapes(semantic: &SemanticProgram) -> collections::Vec<EntryState> {
    let mut state = PrepareState::new(
        semantic.results,
        semantic.local_types.clone(),
        semantic.result_types.clone(),
        0,
    );
    let mut entries = collections::Vec::with_capacity(semantic.ops.len());

    for (op_index, op) in semantic.ops.iter().enumerate() {
        entries.push(snapshot_entry_state(
            &state,
            if state.unreachable { state.height } else { 0 },
        ));
        apply_semantic_effect(op, op_index, &semantic.op_result_types, &mut state);
    }

    entries
}

#[cfg(test)]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct TentativeBlockEntry {
    transient: EntryState,
    cached_locals: collections::Vec<FrameSlot>,
}

/// Snapshot the planner-visible stack boundary from the current prepare state.
///
/// Invariant:
/// - `stack_types.len() == stack_height`
/// - `live_types.len() == stack_height - spill_depth`
///
/// This must hold even in unreachable regions. Unreachable code can still carry
/// a semantic stack shape through structured control; it is only the resident
/// live suffix that collapses to empty when `spill_depth == stack_height`.
fn snapshot_entry_state(state: &PrepareState, spill_depth: u16) -> EntryState {
    let spill_depth = spill_depth.min(state.height);
    let live_count = state.height.saturating_sub(spill_depth);
    EntryState {
        stack_height: state.height,
        spill_depth,
        stack_types: state.type_stack.clone(),
        live_types: state.types_at(spill_depth, live_count),
    }
}

#[cfg(test)]
fn choose_tentative_block_entry(
    cfg: &SemanticCfg,
    block_index: usize,
    carried: &[FrameSlot],
    plan: &FunctionPlan,
) -> TentativeBlockEntry {
    let base_entry = &plan.blocks[block_index].entry;
    let summary = &plan.block_local_summaries[block_index];
    let successor_bonus = direct_successor_carry_bonus(cfg, block_index, plan);
    let mut scored_locals = collections::Vec::<(FrameSlot, i32, bool, bool)>::new();
    let mut seen = BTreeSet::new();

    for &slot in &summary.ranked_slots {
        if seen.insert(slot) {
            let score = summary.slot_score(slot);
            scored_locals.push((
                slot,
                score.map(|s| s.entry_hot_score).unwrap_or_default()
                    + successor_bonus
                        .get(slot.0 as usize)
                        .copied()
                        .unwrap_or_default(),
                false,
                matches!(
                    score.and_then(|s| s.entry_first_access_kind),
                    Some(FirstAccessKind::ReadFirst)
                ),
            ));
        }
    }
    for (slot_index, bonus) in successor_bonus.iter().copied().enumerate() {
        if bonus <= 0 {
            continue;
        }
        let slot = FrameSlot(slot_index as u16);
        if seen.insert(slot) {
            let score = summary.slot_score(slot);
            scored_locals.push((
                slot,
                score.map(|s| s.entry_hot_score).unwrap_or_default() + bonus,
                false,
                matches!(
                    score.and_then(|s| s.entry_first_access_kind),
                    Some(FirstAccessKind::ReadFirst)
                ),
            ));
        }
    }
    for &slot in carried {
        if seen.insert(slot) {
            let score = summary.slot_score(slot);
            scored_locals.push((
                slot,
                score.map(|s| s.entry_hot_score).unwrap_or_default()
                    + successor_bonus
                        .get(slot.0 as usize)
                        .copied()
                        .unwrap_or_default()
                    + 1024,
                true,
                matches!(
                    score.and_then(|s| s.entry_first_access_kind),
                    Some(FirstAccessKind::ReadFirst)
                ),
            ));
        }
    }
    scored_locals.sort_by_key(|(slot, score, carried, _)| {
        (
            core::cmp::Reverse(*score),
            core::cmp::Reverse(*carried as u8),
            slot.0,
        )
    });

    let mut gp_used;
    let mut fp_used;
    (gp_used, fp_used) = count_live_bank_budget_units(&base_entry.live_types, plan.gp_unit_bytes);

    let mut chosen = collections::Vec::new();
    for (slot, score, carried_slot, _) in &scored_locals {
        if *score <= 0 && !*carried_slot {
            continue;
        }
        let ty = plan
            .local_slot_types
            .get(slot.0 as usize)
            .copied()
            .unwrap_or(ValueType::I64);
        let cost = if ty.is_float() {
            1
        } else {
            gp_value_budget_units(ty, plan.gp_unit_bytes)
        };
        let fits = if ty.is_float() {
            fp_used + cost <= plan.fp_dynamic_budget as usize
        } else {
            gp_used + cost <= plan.gp_dynamic_budget as usize
        };
        if !fits {
            continue;
        }
        if ty.is_float() {
            fp_used += cost;
        } else {
            gp_used += cost;
        }
        chosen.push(*slot);
    }

    let transient = base_entry.clone();
    let cached_locals = chosen;
    TentativeBlockEntry {
        transient,
        cached_locals,
    }
}

#[cfg(test)]
fn direct_successor_carry_bonus(
    cfg: &SemanticCfg,
    block_index: usize,
    plan: &FunctionPlan,
) -> collections::Vec<i32> {
    let mut bonus = collections::vec![0; plan.local_slot_types.len()];
    let Some(block) = cfg.blocks.get(block_index) else {
        return bonus;
    };

    for succ in &block.succs {
        let succ_summary = &plan.block_local_summaries[succ.target.as_usize()];
        for score in &succ_summary.slot_scores {
            let entry_bonus = score.entry_hot_score;
            if entry_bonus > 0 {
                bonus[score.slot.0 as usize] += entry_bonus;
            } else if score.used_anywhere {
                bonus[score.slot.0 as usize] += 1;
            }
        }
    }

    bonus
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
    /// Block-local transient symbol stack, from bottom to top.
    ///
    /// This resets at every CFG block entry. Symbol ids are block-local facts
    /// only; they exist so the builder can compare "spill the bottom live
    /// transient" against "drop the weakest cached local" using one ranking
    /// model.
    block_symbols: collections::Vec<u16>,
    next_block_symbol: u16,
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
            block_symbols: collections::Vec::new(),
            next_block_symbol: 0,
            local_types,
        }
    }

    fn mark_unreachable(&mut self) {
        if let Some(frame) = self.control.last().cloned() {
            self.height = frame.start_height.saturating_add(frame.results);
            self.spill_depth = self.height;
            self.type_stack.truncate(frame.start_height as usize);
            self.type_stack.extend_from_slice(&frame.result_types);
            self.block_symbols.clear();
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

    fn enter_cfg_block(&mut self) {
        self.block_symbols = (0..self.height).collect();
        self.next_block_symbol = self.height;
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
    /// Compact per-op entry point: `(stack_height, spill_depth)`.
    pub compact_entries: collections::Vec<CompactEntryPoint>,
    /// Per-block peak GP transient pressure in budget units.
    pub peak_gp: collections::Vec<usize>,
    /// Per-block peak FP transient pressure in budget units.
    pub peak_fp: collections::Vec<usize>,
    /// Per-block entry `EntryState` with full `stack_types` for spilled value fills.
    pub block_entries: collections::Vec<EntryState>,
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
    op_info: &[OpInfo],
    gp_unit_bytes: u8,
) -> LightweightPlanOutput {
    use crate::vm::middle::budget::count_live_bank_budget_units;

    let mut state = PrepareState::new(
        semantic.results,
        semantic.local_types.clone(),
        semantic.result_types.clone(),
        0, // no cache slots
    );

    let mut compact_entries = collections::Vec::with_capacity(semantic.ops.len());
    let mut peak_gp = collections::vec![0usize; cfg.blocks.len()];
    let mut peak_fp = collections::vec![0usize; cfg.blocks.len()];
    let mut block_entries = collections::vec![EntryState::default(); cfg.blocks.len()];

    for (op_index, semantic_op) in semantic.ops.iter().enumerate() {
        let info = &op_info[op_index];
        if info.is_block_start {
            state.enter_cfg_block();
            // Capture block entry state with full stack_types for spilled value fills.
            let spill = state.spill_depth.min(state.height);
            let live_count = state.height.saturating_sub(spill);
            block_entries[info.block_index as usize] = EntryState {
                stack_height: state.height,
                spill_depth: spill,
                stack_types: state.type_stack.clone(),
                live_types: state.types_at(spill, live_count),
            };
        }

        // Record compact entry state (before prefix).
        compact_entries.push(CompactEntryPoint {
            stack_height: state.height,
            spill_depth: state.spill_depth,
        });

        // Apply structural prefix: fill operands, spill at control flow boundaries.
        // Skip ensure_capacity — it depends on cache state and only further reduces
        // the live window, so omitting it gives a conservative upper bound on pressure.
        apply_structural_prefix(semantic_op, &mut state);

        // Measure live-window pressure after prefix (= "before" the op executes).
        let block_index = info.block_index as usize;
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
        compact_entries,
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
    match &op.kind {
        SemanticOpKind::Primitive(kind) => {
            if matches!(kind, PrimitiveOpKind::Unreachable) {
                return;
            }
            let (pop, _push) = primitive_op::stack_effect(kind);
            lightweight_fill_for_operands(state, pop as u16, true);
        }
        SemanticOpKind::LocalGet { .. } => {
            // LocalGet pushes without popping; no fill needed.
        }
        SemanticOpKind::LocalSet { .. } | SemanticOpKind::LocalTee { .. } => {
            lightweight_fill_for_operands(state, 1, true);
        }
        SemanticOpKind::Block { .. } => {}
        SemanticOpKind::Loop { .. } => {
            // Loop headers spill everything.
            lightweight_spill_all(state);
        }
        SemanticOpKind::If { params, .. } => {
            let keep_live = params.saturating_add(1);
            lightweight_fill_for_operands(state, keep_live, true);
            lightweight_spill_all_except_top(state, keep_live);
        }
        SemanticOpKind::Else { .. } => {
            if let Some(frame_state) = state.control.last().cloned() {
                if frame_state.entered_unreachable
                    || matches!(frame_state.kind, ControlFrameKind::Function)
                {
                    return;
                }
                lightweight_fill_for_operands(state, frame_state.results, false);
            }
        }
        SemanticOpKind::End => {}
        SemanticOpKind::Br { arity, .. } => {
            lightweight_fill_for_operands(state, *arity, true);
            lightweight_spill_all_except_top(state, *arity);
        }
        SemanticOpKind::BrIf { arity, .. } => {
            let keep_live = arity.saturating_add(1);
            lightweight_fill_for_operands(state, keep_live, true);
            lightweight_spill_all_except_top(state, keep_live);
        }
        SemanticOpKind::BrOnNull { arity, .. } => {
            let keep_live = arity.saturating_add(1);
            lightweight_fill_for_operands(state, keep_live, true);
            lightweight_spill_all_except_top(state, keep_live);
        }
        SemanticOpKind::BrOnNonNull { arity, .. } => {
            lightweight_fill_for_operands(state, *arity, true);
            lightweight_spill_all_except_top(state, *arity);
        }
        SemanticOpKind::BrOnCast { arity, .. } | SemanticOpKind::BrOnCastFail { arity, .. } => {
            lightweight_fill_for_operands(state, *arity, true);
            lightweight_spill_all_except_top(state, *arity);
        }
        SemanticOpKind::BrTable { entries } => {
            let arity = entries.first().map(|entry| entry.arity).unwrap_or(0);
            let keep_live = arity.saturating_add(1);
            lightweight_fill_for_operands(state, keep_live, true);
            lightweight_spill_all_except_top(state, keep_live);
        }
        SemanticOpKind::CallDirect { .. }
        | SemanticOpKind::CallIndirect { .. }
        | SemanticOpKind::CallRef { .. }
        | SemanticOpKind::ReturnCallDirect { .. }
        | SemanticOpKind::ReturnCallIndirect { .. }
        | SemanticOpKind::ReturnCallRef { .. }
        | SemanticOpKind::ReturnVoid
        | SemanticOpKind::ReturnOne
        | SemanticOpKind::Return { .. } => {
            lightweight_spill_all(state);
        }
        SemanticOpKind::AllocExnRef { .. } => {
            lightweight_spill_all(state);
        }
        SemanticOpKind::TryTable { .. } => {
            // Same as Block — no pre-op fill/spill needed.
        }
        SemanticOpKind::Throw { arity, .. } => {
            lightweight_fill_for_operands(state, *arity, true);
            lightweight_spill_all_except_top(state, *arity);
        }
        SemanticOpKind::ThrowRef => {
            lightweight_fill_for_operands(state, 1, true);
            lightweight_spill_all_except_top(state, 1);
        }
    }
}

/// Fill operands by lowering spill_depth (state-only, no side effects).
fn lightweight_fill_for_operands(
    state: &mut PrepareState,
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
    state.spill_depth = min_spill_depth;
}

/// Spill all live transients by raising spill_depth to height.
fn lightweight_spill_all(state: &mut PrepareState) {
    if state.unreachable {
        return;
    }
    state.spill_depth = state.height;
}

/// Spill all except the top `keep_top` live values.
fn lightweight_spill_all_except_top(state: &mut PrepareState, keep_top: u16) {
    if state.unreachable {
        return;
    }
    let live_count = state.height.saturating_sub(state.spill_depth);
    let count = live_count.saturating_sub(keep_top);
    if count > 0 {
        state.spill_depth += count;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::middle::cfg::{CfgBlock, CfgBlockFlags, CfgBlockId, CfgTerminator, SemanticCfg};
    use crate::vm::middle::joint_plan::facts::{
        BlockEntryStackRegion, BlockLocalSummary, BlockStackValueInfo, LocalSlotScore,
    };

    fn test_cfg() -> SemanticCfg {
        SemanticCfg {
            entry: CfgBlockId(0),
            blocks: collections::vec![CfgBlock {
                id: CfgBlockId(0),
                range: 0..1,
                preds: collections::Vec::new(),
                succs: collections::Vec::new(),
                terminator: CfgTerminator::Return { op_index: 0 },
                flags: CfgBlockFlags {
                    is_entry: true,
                    ..CfgBlockFlags::default()
                },
            }],
            semantic_to_block: collections::vec![CfgBlockId(0)],
        }
    }

    fn test_plan() -> FunctionPlan {
        FunctionPlan {
            gp_unit_bytes: 8,
            gp_dynamic_budget: 2,
            fp_dynamic_budget: 2,
            local_slot_types: collections::vec![ValueType::I32],
            compact_entries: collections::vec![CompactEntryPoint {
                stack_height: 2,
                spill_depth: 0,
            }],
            op_info: collections::vec![OpInfo {
                block_index: 0,
                block_offset: 0,
                is_block_start: true,
                local_op: None,
            }],
            block_local_summaries: collections::vec![BlockLocalSummary {
                ranked_slots: collections::vec![FrameSlot(0)],
                slot_scores: collections::vec![LocalSlotScore {
                    slot: FrameSlot(0),
                    entry_hot_score: 300,
                    entry_first_access_kind: Some(FirstAccessKind::ReadFirst),
                    used_anywhere: true,
                    read_count: 2,
                    write_count: 0,
                }],
            }],
            block_stack_regions: collections::vec![BlockEntryStackRegion {
                entry_stack_height: 2,
                values: collections::vec![
                    BlockStackValueInfo {
                        stack_index: 0,
                        ty: ValueType::I32,
                        touched_before_barrier: false,
                        first_touch_distance: None,
                        touch_count: 0,
                        hot_score: 0,
                    },
                    BlockStackValueInfo {
                        stack_index: 1,
                        ty: ValueType::I32,
                        touched_before_barrier: true,
                        first_touch_distance: Some(0),
                        touch_count: 2,
                        hot_score: 500,
                    },
                ],
            }],
            block_transient_regions: collections::vec![BlockTransientRegion::default()],
            blocks: collections::vec![BlockPlan {
                entry: EntryState {
                    stack_height: 2,
                    spill_depth: 0,
                    stack_types: collections::Vec::new(),
                    live_types: collections::vec![ValueType::I32, ValueType::I32],
                },
                ..Default::default()
            }],
        }
    }

    #[test]
    fn tentative_entry_keeps_structural_stack_and_admits_hot_carried_local_when_budget_allows() {
        let mut plan = test_plan();
        plan.gp_dynamic_budget = 3;
        let cfg = test_cfg();
        let tentative = choose_tentative_block_entry(&cfg, 0, &[FrameSlot(0)], &plan);
        assert_eq!(tentative.transient.spill_depth, 0);
        assert_eq!(
            tentative.transient.live_types,
            collections::vec![ValueType::I32, ValueType::I32]
        );
        assert_eq!(tentative.cached_locals, collections::vec![FrameSlot(0)]);
    }

    #[test]
    fn tentative_entry_breaks_full_tie_by_preferring_lower_cache_cost() {
        // Candidate A:
        // - keep one hot stack value (score 300)
        // - cache one i32 local (score 500, cost 1)
        // => total 800, ensures 1, cache_cost 1
        //
        // Candidate B:
        // - spill the stack
        // - cache one i64 local (score 800, cost 2)
        // => total 800, ensures 1, cache_cost 2
        //
        // The planner should pick candidate A via the final cache-cost
        // tie-break.
        let plan = FunctionPlan {
            gp_unit_bytes: 4,
            gp_dynamic_budget: 2,
            fp_dynamic_budget: 0,
            local_slot_types: collections::vec![ValueType::I64, ValueType::I32],
            compact_entries: collections::vec![CompactEntryPoint {
                stack_height: 1,
                spill_depth: 0,
            }],
            op_info: collections::vec![OpInfo {
                block_index: 0,
                block_offset: 0,
                is_block_start: true,
                local_op: None,
            }],
            block_local_summaries: collections::vec![BlockLocalSummary {
                ranked_slots: collections::vec![FrameSlot(0), FrameSlot(1)],
                slot_scores: collections::vec![
                    LocalSlotScore {
                        slot: FrameSlot(0),
                        entry_hot_score: 800,
                        entry_first_access_kind: Some(FirstAccessKind::ReadFirst),
                        used_anywhere: true,
                        read_count: 1,
                        write_count: 0,
                    },
                    LocalSlotScore {
                        slot: FrameSlot(1),
                        entry_hot_score: 500,
                        entry_first_access_kind: Some(FirstAccessKind::ReadFirst),
                        used_anywhere: true,
                        read_count: 1,
                        write_count: 0,
                    },
                ],
            }],
            block_stack_regions: collections::vec![BlockEntryStackRegion {
                entry_stack_height: 1,
                values: collections::vec![BlockStackValueInfo {
                    stack_index: 0,
                    ty: ValueType::I32,
                    touched_before_barrier: true,
                    first_touch_distance: Some(0),
                    touch_count: 1,
                    hot_score: 300,
                }],
            }],
            block_transient_regions: collections::vec![BlockTransientRegion::default()],
            blocks: collections::vec![BlockPlan {
                entry: EntryState {
                    stack_height: 1,
                    spill_depth: 0,
                    stack_types: collections::Vec::new(),
                    live_types: collections::vec![ValueType::I32],
                },
                ..Default::default()
            }],
        };

        let cfg = test_cfg();
        let tentative = choose_tentative_block_entry(&cfg, 0, &[], &plan);
        assert_eq!(
            tentative.transient.spill_depth, 0,
            "the equal-score/equal-ensure tie should keep the hot stack value and prefer the cheaper i32 cache"
        );
        assert_eq!(tentative.cached_locals, collections::vec![FrameSlot(1)]);
    }

    #[test]
    fn tentative_entry_breaks_equal_score_tie_by_avoiding_extra_entry_ensure() {
        // Candidate A:
        // - keep one hot entry-stack value (score 320)
        // - no cached locals
        // => total 320, ensures 0
        //
        // Candidate B:
        // - spill the stack
        // - cache one read-first local (score 320)
        // => total 320, ensures 1
        //
        // The planner should keep the stack value and avoid the cold-edge
        // entry ensure.
        let plan = FunctionPlan {
            gp_unit_bytes: 8,
            gp_dynamic_budget: 1,
            fp_dynamic_budget: 0,
            local_slot_types: collections::vec![ValueType::I32],
            compact_entries: collections::vec![CompactEntryPoint {
                stack_height: 1,
                spill_depth: 0,
            }],
            op_info: collections::vec![OpInfo {
                block_index: 0,
                block_offset: 0,
                is_block_start: true,
                local_op: None,
            }],
            block_local_summaries: collections::vec![BlockLocalSummary {
                ranked_slots: collections::vec![FrameSlot(0)],
                slot_scores: collections::vec![LocalSlotScore {
                    slot: FrameSlot(0),
                    entry_hot_score: 320,
                    entry_first_access_kind: Some(FirstAccessKind::ReadFirst),
                    used_anywhere: true,
                    read_count: 1,
                    write_count: 0,
                }],
            }],
            block_stack_regions: collections::vec![BlockEntryStackRegion {
                entry_stack_height: 1,
                values: collections::vec![BlockStackValueInfo {
                    stack_index: 0,
                    ty: ValueType::I32,
                    touched_before_barrier: true,
                    first_touch_distance: Some(0),
                    touch_count: 1,
                    hot_score: 320,
                }],
            }],
            block_transient_regions: collections::vec![BlockTransientRegion::default()],
            blocks: collections::vec![BlockPlan {
                entry: EntryState {
                    stack_height: 1,
                    spill_depth: 0,
                    stack_types: collections::Vec::new(),
                    live_types: collections::vec![ValueType::I32],
                },
                ..Default::default()
            }],
        };

        let cfg = test_cfg();
        let tentative = choose_tentative_block_entry(&cfg, 0, &[], &plan);
        assert_eq!(tentative.transient.spill_depth, 0);
        assert!(tentative.cached_locals.is_empty());
    }

    #[test]
    fn tentative_entry_keeps_structural_stack_when_no_locals_are_admitted() {
        let plan = FunctionPlan {
            gp_unit_bytes: 8,
            gp_dynamic_budget: 2,
            fp_dynamic_budget: 0,
            local_slot_types: collections::vec![],
            compact_entries: collections::vec![CompactEntryPoint {
                stack_height: 2,
                spill_depth: 0,
            }],
            op_info: collections::vec![OpInfo {
                block_index: 0,
                block_offset: 0,
                is_block_start: true,
                local_op: None,
            }],
            block_local_summaries: collections::vec![BlockLocalSummary::default()],
            block_stack_regions: collections::vec![BlockEntryStackRegion {
                entry_stack_height: 2,
                values: collections::vec![
                    BlockStackValueInfo {
                        stack_index: 0,
                        ty: ValueType::I32,
                        touched_before_barrier: false,
                        first_touch_distance: None,
                        touch_count: 0,
                        hot_score: 0,
                    },
                    BlockStackValueInfo {
                        stack_index: 1,
                        ty: ValueType::I32,
                        touched_before_barrier: false,
                        first_touch_distance: None,
                        touch_count: 0,
                        hot_score: 0,
                    },
                ],
            }],
            block_transient_regions: collections::vec![BlockTransientRegion::default()],
            blocks: collections::vec![BlockPlan {
                entry: EntryState {
                    stack_height: 2,
                    spill_depth: 0,
                    stack_types: collections::Vec::new(),
                    live_types: collections::vec![ValueType::I32, ValueType::I32],
                },
                ..Default::default()
            }],
        };

        let cfg = test_cfg();
        let tentative = choose_tentative_block_entry(&cfg, 0, &[], &plan);
        assert_eq!(
            tentative.transient.spill_depth, 0,
            "without local admission pressure, tentative entry must keep the structural transient contract unchanged"
        );
        assert!(tentative.cached_locals.is_empty());
    }
}
