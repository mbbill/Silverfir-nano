//! Joint-plan builder from first-pass SSA.
//!
//! This builder is strict about transient legality and now computes the
//! planner-owned boundary seed from block-local value ranking. It produces:
//! - exact transient spill/fill behavior per semantic op
//! - a canonical predecessor per block
//! - a tentative entry boundary per block
//! - a single-pass finalized cached-local entry set per block

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use crate::{
    error::WasmError,
    value_type::ValueType,
    vm::{
        backend::BackendConfig,
        middle::{
            budget::{count_live_bank_budget_units, gp_value_budget_units},
            cfg::SemanticCfg,
            frame::{FrameLayoutPlan, FrameSlot, FrameSpan},
            slot_ssa::SlotSsaProgram,
        },
        wasm::{
            primitive_op,
            primitive_op::PrimitiveOpKind,
            semantic_ir::{SemanticOp, SemanticOpKind, SemanticProgram},
        },
    },
};

use super::{
    block_open::before_op_decision,
    canonical::choose_canonical_predecessors,
    entry_region::{analyze_block_entry_regions, analyze_block_transient_regions},
    facts::{
        BlockPlan, EntryState, FirstAccessKind, FunctionPlan, LocalOpKind, OpInfo, OpPlan,
        PrepAction,
    },
    interface::BeforeOpQuery,
    local_access::decide_local_access,
    policy::op_must_drop_all_caches,
    pressure::fits_with_cached_locals,
};

pub(crate) fn build_plan(
    semantic: &SemanticProgram,
    cfg: &SemanticCfg,
    _slot_program: &SlotSsaProgram,
    frame: FrameLayoutPlan,
    config: BackendConfig,
) -> Result<FunctionPlan, WasmError> {
    let gp_dynamic_budget = config
        .gp_transient_budget
        .saturating_add(config.gp_local_cache_budget);
    let fp_dynamic_budget = config
        .fp_transient_budget
        .saturating_add(config.fp_local_cache_budget);

    let op_info = build_op_info(semantic, cfg, frame);
    let semantic_entry_shapes = analyze_semantic_entry_shapes(semantic);
    let (block_regions, block_stack_regions) =
        analyze_block_entry_regions(semantic, cfg, frame, &semantic_entry_shapes);
    let block_transient_regions =
        analyze_block_transient_regions(semantic, cfg, &semantic_entry_shapes);
    let (op_plans, entry_states) = prepare_semantic_ops(
        semantic,
        frame,
        config.gp_unit_bytes,
        gp_dynamic_budget,
        fp_dynamic_budget,
        &op_info,
        &block_regions,
        &block_transient_regions,
    )?;
    let canonical = choose_canonical_predecessors(cfg);
    let op_force_drop_all_caches = semantic
        .ops
        .iter()
        .map(|op| op_must_drop_all_caches(&op.kind))
        .collect::<Vec<_>>();

    let mut plan = FunctionPlan {
        gp_unit_bytes: config.gp_unit_bytes,
        gp_dynamic_budget,
        fp_dynamic_budget,
        local_slot_types: semantic.local_types.clone(),
        op_plans,
        entry_states,
        op_info,
        block_regions,
        block_stack_regions,
        block_transient_regions,
        blocks: alloc::vec![BlockPlan::default(); cfg.blocks.len()],
    };
    let (block_entries, block_exits) =
        compute_block_boundaries(cfg, &plan, &op_force_drop_all_caches, &canonical.by_block);

    plan.blocks =
        canonical
            .by_block
            .into_iter()
            .enumerate()
            .map(|(block_index, _canonical_predecessor)| {
                let block = &cfg.blocks[block_index];
                let tentative = block_entries.get(block_index).cloned().unwrap_or_else(|| {
                    TentativeBlockEntry {
                        transient: plan.entry_states[block.range.start].clone(),
                        cached_locals: Vec::new(),
                    }
                });
                BlockPlan {
                    entry: tentative.transient,
                    entry_cached_locals: finalize_block_entry(
                        block_index,
                        &plan,
                        &tentative.cached_locals,
                        block_exits
                            .get(block_index)
                            .map(Vec::as_slice)
                            .unwrap_or(&[]),
                    ),
                    exit_cached_locals: block_exits.get(block_index).cloned().unwrap_or_default(),
                }
            })
            .collect();

    Ok(plan)
}

fn build_op_info(
    semantic: &SemanticProgram,
    cfg: &SemanticCfg,
    frame: FrameLayoutPlan,
) -> Vec<OpInfo> {
    let mut out = alloc::vec![OpInfo::default(); semantic.ops.len()];
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
fn analyze_semantic_entry_shapes(semantic: &SemanticProgram) -> Vec<EntryState> {
    let mut state = PrepareState::new(
        semantic.results,
        semantic.local_types.clone(),
        semantic.result_types.clone(),
        0,
    );
    let mut entries = Vec::with_capacity(semantic.ops.len());

    for (op_index, op) in semantic.ops.iter().enumerate() {
        entries.push(EntryState {
            stack_height: state.height,
            spill_depth: if state.unreachable { state.height } else { 0 },
            stack_types: if state.unreachable {
                Vec::new()
            } else {
                state.type_stack.clone()
            },
            live_types: if state.unreachable {
                Vec::new()
            } else {
                state.type_stack.clone()
            },
        });
        apply_semantic_effect(op, op_index, &semantic.op_result_types, &mut state);
    }

    entries
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct TentativeBlockEntry {
    transient: EntryState,
    cached_locals: Vec<FrameSlot>,
}

fn compute_block_boundaries(
    cfg: &SemanticCfg,
    plan: &FunctionPlan,
    op_force_drop_all_caches: &[bool],
    canonical_predecessors: &[Option<crate::vm::middle::cfg::CfgBlockId>],
) -> (Vec<TentativeBlockEntry>, Vec<Vec<FrameSlot>>) {
    let mut entries = alloc::vec![TentativeBlockEntry::default(); cfg.blocks.len()];
    let mut exits = alloc::vec![Vec::new(); cfg.blocks.len()];

    for (block_index, block) in cfg.blocks.iter().enumerate() {
        let carried = canonical_predecessors
            .get(block_index)
            .and_then(|pred| pred.map(|pred| exits[pred.as_usize()].clone()))
            .unwrap_or_default();
        let tentative = choose_tentative_block_entry(block_index, &carried, plan);
        let actual_exit = simulate_block_exit_cached_locals(
            block,
            plan,
            op_force_drop_all_caches,
            &tentative.cached_locals,
        );
        entries[block_index] = tentative;
        exits[block_index] = actual_exit;
    }

    (entries, exits)
}

fn choose_tentative_block_entry(
    block_index: usize,
    carried: &[FrameSlot],
    plan: &FunctionPlan,
) -> TentativeBlockEntry {
    let block_start_index = plan
        .op_info
        .iter()
        .position(|info| info.block_index as usize == block_index && info.is_block_start)
        .unwrap_or(0);
    let base_entry = &plan.entry_states[block_start_index];
    let region = &plan.block_regions[block_index];
    let stack_region = &plan.block_stack_regions[block_index];

    let mut scored_locals = Vec::<(FrameSlot, i32, bool, bool)>::new();
    let mut seen = BTreeSet::new();

    for &slot in &region.ranked_slots {
        if seen.insert(slot) {
            let info = region.info(slot);
            scored_locals.push((
                slot,
                info.map(|info| info.hot_score).unwrap_or_default(),
                false,
                matches!(
                    info.and_then(|info| info.first_access_kind),
                    Some(FirstAccessKind::ReadFirst)
                ),
            ));
        }
    }
    for &slot in carried {
        if seen.insert(slot) {
            let info = region.info(slot);
            scored_locals.push((
                slot,
                info.map(|info| info.hot_score + 1024).unwrap_or(1024),
                true,
                matches!(
                    info.and_then(|info| info.first_access_kind),
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

    let mut best: Option<(EntryState, Vec<FrameSlot>, i32, usize, usize)> = None;
    for spill_depth in 0..=base_entry.stack_height {
        let live_types = base_entry.stack_types[spill_depth as usize..].to_vec();
        let (mut gp_used, mut fp_used) =
            count_live_bank_budget_units(&live_types, plan.gp_unit_bytes);
        if gp_used > plan.gp_dynamic_budget as usize || fp_used > plan.fp_dynamic_budget as usize {
            continue;
        }

        let mut chosen = Vec::new();
        let mut local_score = 0i32;
        let mut ensure_count = 0usize;
        let mut cache_cost = 0usize;
        for (slot, score, carried_slot, read_first) in &scored_locals {
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
            local_score += *score;
            cache_cost += cost;
            if !carried_slot && *read_first {
                ensure_count += 1;
            }
        }

        let entry = EntryState {
            stack_height: base_entry.stack_height,
            spill_depth,
            stack_types: base_entry.stack_types.clone(),
            live_types,
        };
        let total_score = stack_region.score_suffix(spill_depth) + local_score;
        let candidate = (entry, chosen, total_score, ensure_count, cache_cost);
        match &best {
            None => best = Some(candidate),
            Some((_, _, best_score, best_ensures, best_cost)) => {
                if total_score > *best_score
                    || (total_score == *best_score
                        && (ensure_count < *best_ensures
                            || (ensure_count == *best_ensures && cache_cost < *best_cost)))
                {
                    best = Some(candidate);
                }
            }
        }
    }

    let (transient, cached_locals, _, _, _) = best.unwrap_or_else(|| {
        (
            EntryState {
                stack_height: base_entry.stack_height,
                spill_depth: base_entry.stack_height,
                stack_types: base_entry.stack_types.clone(),
                live_types: Vec::new(),
            },
            Vec::new(),
            0,
            0,
            0,
        )
    });
    TentativeBlockEntry {
        transient,
        cached_locals,
    }
}

fn simulate_block_exit_cached_locals(
    block: &crate::vm::middle::cfg::CfgBlock,
    plan: &FunctionPlan,
    op_force_drop_all_caches: &[bool],
    entry_cached_locals: &[FrameSlot],
) -> Vec<FrameSlot> {
    let mut resident = entry_cached_locals.iter().copied().collect::<BTreeSet<_>>();
    for semantic_index in block.range.clone() {
        let before = before_op_decision(
            plan,
            op_force_drop_all_caches,
            BeforeOpQuery {
                semantic_index,
                resident_cache: &resident,
            },
        );
        for slot in before.drop_cached_locals {
            resident.remove(&slot);
        }

        if !fits_with_cached_locals(plan, semantic_index, &resident) {
            let mut trimmed = resident.clone();
            while !fits_with_cached_locals(plan, semantic_index, &trimmed) {
                let Some(slot) = trimmed.iter().next_back().copied() else {
                    break;
                };
                trimmed.remove(&slot);
            }
            resident = trimmed;
        }

        let Some((slot, kind)) = plan.op_info[semantic_index].local_op else {
            continue;
        };
        let access = decide_local_access(
            plan,
            super::interface::LocalAccessQuery {
                semantic_index,
                slot,
                resident_cache: &resident,
            },
        );
        match kind {
            LocalOpKind::Get | LocalOpKind::Tee => {
                if matches!(access, super::interface::LocalAccessDecision::Cache) {
                    resident.insert(slot);
                }
            }
            LocalOpKind::Set => {
                resident.insert(slot);
            }
        }
    }
    resident.into_iter().collect()
}

fn finalize_block_entry(
    block_index: usize,
    plan: &FunctionPlan,
    tentative_entry: &[FrameSlot],
    actual_exit: &[FrameSlot],
) -> Vec<FrameSlot> {
    let region = &plan.block_regions[block_index];
    tentative_entry
        .iter()
        .copied()
        .filter(|slot| {
            region
                .info(*slot)
                .map(|info| info.used_anywhere())
                .unwrap_or(false)
                || actual_exit.contains(slot)
        })
        .collect()
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
    param_types: Vec<ValueType>,
    result_types: Vec<ValueType>,
}

#[derive(Clone, Debug)]
struct PrepareState {
    height: u16,
    spill_depth: u16,
    unreachable: bool,
    control: Vec<ControlFrame>,
    type_stack: Vec<ValueType>,
    /// Block-local transient symbol stack, from bottom to top.
    ///
    /// This resets at every CFG block entry. Symbol ids are block-local facts
    /// only; they exist so the builder can compare "spill the bottom live
    /// transient" against "drop the weakest cached local" using one ranking
    /// model.
    block_symbols: Vec<u16>,
    next_block_symbol: u16,
    local_types: Vec<ValueType>,
    resident_cache: Vec<bool>,
    resident_gp_units: usize,
    resident_fp_units: usize,
}

impl PrepareState {
    fn new(
        results: u16,
        local_types: Vec<ValueType>,
        result_types: Vec<ValueType>,
        cache_slots: usize,
    ) -> Self {
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
                result_types: normalized_result_types(results, Some(result_types.as_slice())),
            }],
            type_stack: Vec::new(),
            block_symbols: Vec::new(),
            next_block_symbol: 0,
            local_types,
            resident_cache: alloc::vec![false; cache_slots],
            resident_gp_units: 0,
            resident_fp_units: 0,
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

    fn enter_cfg_block(&mut self) {
        self.block_symbols = (0..self.height).collect();
        self.next_block_symbol = self.height;
    }

    fn push_fresh_block_symbol(&mut self) {
        self.block_symbols.push(self.next_block_symbol);
        self.next_block_symbol = self.next_block_symbol.saturating_add(1);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CacheBank {
    Gp,
    Fp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CacheCandidate {
    slot: FrameSlot,
    bank: CacheBank,
    cost: usize,
    rank: usize,
}

#[derive(Clone, Debug)]
struct CachePlan {
    candidates: Vec<CacheCandidate>,
    local_to_candidate: Vec<Option<usize>>,
}

/// Local builder-side ordering for admission and eviction.
///
/// This mirrors the planner pseudocode:
/// - values unused in the block are weakest
/// - values dead after the current point keep only whole-block hotness
/// - values still live later get the strongest dynamic score
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct LocalKeepKey {
    class: u8,
    score: i32,
}

impl LocalKeepKey {
    #[inline]
    const fn weakest() -> Self {
        Self { class: 0, score: 0 }
    }
}

fn prepare_semantic_ops(
    semantic: &SemanticProgram,
    frame: FrameLayoutPlan,
    gp_unit_bytes: u8,
    gp_dynamic_budget: u8,
    fp_dynamic_budget: u8,
    op_info: &[OpInfo],
    block_regions: &[super::facts::BlockLocalRegion],
    block_transient_regions: &[super::facts::BlockTransientRegion],
) -> Result<(Vec<OpPlan>, Vec<EntryState>), WasmError> {
    let cache_plan = build_cache_plan(semantic, frame, gp_unit_bytes);
    let mut state = PrepareState::new(
        semantic.results,
        semantic.local_types.clone(),
        semantic.result_types.clone(),
        cache_plan.candidates.len(),
    );
    let mut entry_states = Vec::with_capacity(semantic.ops.len());
    let mut ops = Vec::with_capacity(semantic.ops.len());

    for (op_index, semantic_op) in semantic.ops.iter().enumerate() {
        if op_info
            .get(op_index)
            .map(|info| info.is_block_start)
            .unwrap_or(false)
        {
            state.enter_cfg_block();
        }
        let live_count = state.height.saturating_sub(state.spill_depth);
        let live_types = if state.unreachable {
            Vec::new()
        } else {
            state.types_at(state.spill_depth, live_count)
        };
        entry_states.push(EntryState {
            stack_height: state.height,
            spill_depth: state.spill_depth,
            stack_types: if state.unreachable {
                Vec::new()
            } else {
                state.type_stack.clone()
            },
            live_types,
        });
        plan_prefix(
            semantic_op,
            op_index,
            &mut state,
            &cache_plan,
            frame,
            gp_unit_bytes,
            gp_dynamic_budget,
            fp_dynamic_budget,
            op_info,
            block_regions,
            block_transient_regions,
            &semantic.op_result_types,
        );
        ops.push(OpPlan {
            before: EntryState {
                stack_height: state.height,
                spill_depth: state.spill_depth,
                stack_types: if state.unreachable {
                    Vec::new()
                } else {
                    state.type_stack.clone()
                },
                live_types: if state.unreachable {
                    Vec::new()
                } else {
                    state.types_at(
                        state.spill_depth,
                        state.height.saturating_sub(state.spill_depth),
                    )
                },
            },
        });
        apply_block_symbolic_effect(semantic_op, &mut state);
        apply_semantic_effect(semantic_op, op_index, &semantic.op_result_types, &mut state);
    }

    Ok((ops, entry_states))
}

fn plan_prefix(
    op: &SemanticOp,
    op_index: usize,
    state: &mut PrepareState,
    cache_plan: &CachePlan,
    frame: FrameLayoutPlan,
    gp_unit_bytes: u8,
    gp_dynamic_budget: u8,
    fp_dynamic_budget: u8,
    op_info: &[OpInfo],
    block_regions: &[super::facts::BlockLocalRegion],
    block_transient_regions: &[super::facts::BlockTransientRegion],
    op_result_types: &BTreeMap<usize, Vec<ValueType>>,
) {
    let mut prefix = Vec::new();

    match &op.kind {
        SemanticOpKind::Primitive(kind) => {
            if matches!(kind, PrimitiveOpKind::Unreachable) {
                return;
            }
            let (pop, push) = primitive_op::stack_effect(kind);
            fill_for_operands(&mut prefix, state, frame, pop as u16);
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
                ensure_capacity(
                    &mut prefix,
                    state,
                    cache_plan,
                    frame,
                    gp_unit_bytes,
                    gp_dynamic_budget,
                    fp_dynamic_budget,
                    op_index,
                    op_info,
                    block_regions,
                    block_transient_regions,
                    pop as u16,
                    core::slice::from_ref(&push_ty.unwrap_or(ValueType::I64)),
                    None,
                );
            }
        }
        SemanticOpKind::LocalGet { idx } => {
            let push_ty = state.local_type(*idx);
            let _ = plan_local_access(
                *idx,
                &mut prefix,
                state,
                cache_plan,
                frame,
                gp_unit_bytes,
                gp_dynamic_budget,
                fp_dynamic_budget,
                op_index,
                op_info,
                block_regions,
                block_transient_regions,
                0,
                core::slice::from_ref(&push_ty),
            );
        }
        SemanticOpKind::LocalSet { idx } => {
            fill_for_operands(&mut prefix, state, frame, 1);
            let _ = plan_local_access(
                *idx,
                &mut prefix,
                state,
                cache_plan,
                frame,
                gp_unit_bytes,
                gp_dynamic_budget,
                fp_dynamic_budget,
                op_index,
                op_info,
                block_regions,
                block_transient_regions,
                1,
                &[],
            );
        }
        SemanticOpKind::LocalTee { idx } => {
            fill_for_operands(&mut prefix, state, frame, 1);
            let push_ty = state.local_type(*idx);
            let _ = plan_local_access(
                *idx,
                &mut prefix,
                state,
                cache_plan,
                frame,
                gp_unit_bytes,
                gp_dynamic_budget,
                fp_dynamic_budget,
                op_index,
                op_info,
                block_regions,
                block_transient_regions,
                1,
                core::slice::from_ref(&push_ty),
            );
        }
        SemanticOpKind::Block { .. } => {}
        SemanticOpKind::Loop { .. } => spill_all(&mut prefix, state, frame),
        SemanticOpKind::If { params, .. } => {
            drop_all_caches(&mut prefix, state, cache_plan);
            let keep_live = params.saturating_add(1);
            fill_for_operands(&mut prefix, state, frame, keep_live);
            spill_all_except_top(&mut prefix, state, frame, keep_live);
        }
        SemanticOpKind::Else { .. } => {
            drop_all_caches(&mut prefix, state, cache_plan);
            if let Some(frame_state) = state.control.last().cloned() {
                if frame_state.entered_unreachable
                    || matches!(frame_state.kind, ControlFrameKind::Function)
                {
                    return;
                }
                fill_for_operands_inner(&mut prefix, state, frame, frame_state.results, false);
            }
        }
        SemanticOpKind::End => {
            drop_all_caches(&mut prefix, state, cache_plan);
        }
        SemanticOpKind::Br { arity, .. } => {
            drop_all_caches(&mut prefix, state, cache_plan);
            fill_for_operands(&mut prefix, state, frame, *arity);
            spill_all_except_top(&mut prefix, state, frame, *arity);
        }
        SemanticOpKind::BrIf { arity, .. } => {
            drop_all_caches(&mut prefix, state, cache_plan);
            let keep_live = arity.saturating_add(1);
            fill_for_operands(&mut prefix, state, frame, keep_live);
            spill_all_except_top(&mut prefix, state, frame, keep_live);
        }
        SemanticOpKind::BrTable { entries } => {
            drop_all_caches(&mut prefix, state, cache_plan);
            let arity = entries.first().map(|entry| entry.arity).unwrap_or(0);
            let keep_live = arity.saturating_add(1);
            fill_for_operands(&mut prefix, state, frame, keep_live);
            spill_all_except_top(&mut prefix, state, frame, keep_live);
        }
        SemanticOpKind::CallDirect { .. }
        | SemanticOpKind::CallIndirect { .. }
        | SemanticOpKind::ReturnVoid
        | SemanticOpKind::ReturnOne
        | SemanticOpKind::Return { .. } => {
            drop_all_caches(&mut prefix, state, cache_plan);
            spill_all(&mut prefix, state, frame);
        }
    }
}

fn build_cache_plan(
    semantic: &SemanticProgram,
    frame: FrameLayoutPlan,
    gp_unit_bytes: u8,
) -> CachePlan {
    let mut candidates = Vec::new();
    let mut local_to_candidate = alloc::vec![None; semantic.local_count as usize];
    for local_idx in 0..semantic.local_count as usize {
        let ty = semantic
            .local_types
            .get(local_idx)
            .copied()
            .unwrap_or(ValueType::I64);
        let index = candidates.len();
        candidates.push(CacheCandidate {
            slot: frame.local_slot(local_idx as u16),
            bank: if ty.is_float() {
                CacheBank::Fp
            } else {
                CacheBank::Gp
            },
            cost: if ty.is_float() {
                1
            } else {
                gp_value_budget_units(ty, gp_unit_bytes)
            },
            rank: local_idx,
        });
        local_to_candidate[local_idx] = Some(index);
    }
    CachePlan {
        candidates,
        local_to_candidate,
    }
}

fn plan_local_access(
    local_idx: u16,
    prefix: &mut Vec<PrepAction>,
    state: &mut PrepareState,
    cache_plan: &CachePlan,
    frame: FrameLayoutPlan,
    gp_unit_bytes: u8,
    gp_dynamic_budget: u8,
    fp_dynamic_budget: u8,
    op_index: usize,
    op_info: &[OpInfo],
    block_regions: &[super::facts::BlockLocalRegion],
    block_transient_regions: &[super::facts::BlockTransientRegion],
    pop: u16,
    push_types: &[ValueType],
) -> bool {
    let Some(index) = cache_plan
        .local_to_candidate
        .get(local_idx as usize)
        .copied()
        .flatten()
    else {
        ensure_capacity(
            prefix,
            state,
            cache_plan,
            frame,
            gp_unit_bytes,
            gp_dynamic_budget,
            fp_dynamic_budget,
            op_index,
            op_info,
            block_regions,
            block_transient_regions,
            pop,
            push_types,
            None,
        );
        return false;
    };

    let target_keep = local_target_keep_key(
        op_index,
        op_info,
        block_regions,
        frame.local_slot(local_idx),
    );
    if !state.resident_cache[index] && target_keep == LocalKeepKey::weakest() {
        ensure_capacity(
            prefix,
            state,
            cache_plan,
            frame,
            gp_unit_bytes,
            gp_dynamic_budget,
            fp_dynamic_budget,
            op_index,
            op_info,
            block_regions,
            block_transient_regions,
            pop,
            push_types,
            None,
        );
        return false;
    }

    if ensure_capacity(
        prefix,
        state,
        cache_plan,
        frame,
        gp_unit_bytes,
        gp_dynamic_budget,
        fp_dynamic_budget,
        op_index,
        op_info,
        block_regions,
        block_transient_regions,
        pop,
        push_types,
        Some(index),
    ) {
        if !state.resident_cache[index] {
            state.resident_cache[index] = true;
            match cache_plan.candidates[index].bank {
                CacheBank::Gp => state.resident_gp_units += cache_plan.candidates[index].cost,
                CacheBank::Fp => state.resident_fp_units += cache_plan.candidates[index].cost,
            }
        }
        return true;
    }

    if state.resident_cache[index] {
        emit_drop_cache(prefix, state, cache_plan, index);
    }
    ensure_capacity(
        prefix,
        state,
        cache_plan,
        frame,
        gp_unit_bytes,
        gp_dynamic_budget,
        fp_dynamic_budget,
        op_index,
        op_info,
        block_regions,
        block_transient_regions,
        pop,
        push_types,
        None,
    );
    false
}

fn ensure_capacity(
    prefix: &mut Vec<PrepAction>,
    state: &mut PrepareState,
    cache_plan: &CachePlan,
    frame: FrameLayoutPlan,
    gp_unit_bytes: u8,
    gp_dynamic_budget: u8,
    fp_dynamic_budget: u8,
    op_index: usize,
    op_info: &[OpInfo],
    block_regions: &[super::facts::BlockLocalRegion],
    block_transient_regions: &[super::facts::BlockTransientRegion],
    pop: u16,
    push_types: &[ValueType],
    preserve_cache: Option<usize>,
) -> bool {
    if state.unreachable {
        return true;
    }

    loop {
        let post_height = state.height.saturating_sub(pop);
        let live_start = state.spill_depth as usize;
        let live_end = post_height as usize;
        let (mut gp_live, mut fp_live) = count_live_bank_budget_units(
            state.type_stack.get(live_start..live_end).unwrap_or(&[]),
            gp_unit_bytes,
        );
        let (push_gp, push_fp) = count_live_bank_budget_units(push_types, gp_unit_bytes);
        gp_live = gp_live.saturating_add(push_gp);
        fp_live = fp_live.saturating_add(push_fp);

        let mut future_gp_cache = state.resident_gp_units;
        let mut future_fp_cache = state.resident_fp_units;
        if let Some(index) = preserve_cache {
            if !state.resident_cache.get(index).copied().unwrap_or(false) {
                match cache_plan.candidates[index].bank {
                    CacheBank::Gp => future_gp_cache += cache_plan.candidates[index].cost,
                    CacheBank::Fp => future_fp_cache += cache_plan.candidates[index].cost,
                }
            }
        }

        let need_gp = gp_live.saturating_add(future_gp_cache) > gp_dynamic_budget as usize;
        let need_fp = fp_live.saturating_add(future_fp_cache) > fp_dynamic_budget as usize;
        if !need_gp && !need_fp {
            return true;
        }

        let cache_victim = choose_cache_victim(
            state,
            cache_plan,
            preserve_cache,
            need_gp,
            need_fp,
            op_index,
            op_info,
            block_regions,
        );
        let transient_victim = choose_transient_victim(
            state,
            need_gp,
            need_fp,
            post_height,
            op_index,
            op_info,
            block_transient_regions,
        );

        match choose_pressure_action(cache_victim, transient_victim) {
            Some(PressureAction::DropCache(victim)) => {
                emit_drop_cache(prefix, state, cache_plan, victim);
                continue;
            }
            Some(PressureAction::SpillBottomTransient) => {
                prefix.push(PrepAction::Spill(FrameSpan::new(
                    frame.operand_slot(state.spill_depth),
                    1,
                )));
                state.spill_depth += 1;
                continue;
            }
            None => {}
        }

        return false;
    }
}

fn choose_cache_victim(
    state: &PrepareState,
    cache_plan: &CachePlan,
    preserve_cache: Option<usize>,
    need_gp: bool,
    need_fp: bool,
    op_index: usize,
    op_info: &[OpInfo],
    block_regions: &[super::facts::BlockLocalRegion],
) -> Option<PressureCandidate<usize>> {
    let mut best: Option<(usize, LocalKeepKey)> = None;
    for (index, candidate) in cache_plan.candidates.iter().copied().enumerate() {
        if Some(index) == preserve_cache
            || !state.resident_cache.get(index).copied().unwrap_or(false)
        {
            continue;
        }
        if (need_gp && candidate.bank != CacheBank::Gp)
            || (need_fp && candidate.bank != CacheBank::Fp)
        {
            continue;
        }
        let keep = local_keep_key(
            op_index,
            op_info,
            block_regions,
            candidate.slot,
            preserve_cache.map(|idx| cache_plan.candidates[idx].slot),
            false,
        );
        if best.map(|(_, current)| keep < current).unwrap_or(true) {
            best = Some((index, keep));
        }
    }

    best.map(|(index, keep)| PressureCandidate {
        item: index,
        keep,
        relief_score: 1,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PressureAction {
    DropCache(usize),
    SpillBottomTransient,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PressureCandidate<T> {
    item: T,
    keep: LocalKeepKey,
    relief_score: u8,
}

fn choose_pressure_action(
    cache_victim: Option<PressureCandidate<usize>>,
    transient_victim: Option<PressureCandidate<()>>,
) -> Option<PressureAction> {
    match (cache_victim, transient_victim) {
        (Some(cache), Some(transient)) => {
            if cache.relief_score > transient.relief_score
                || (cache.relief_score == transient.relief_score && cache.keep <= transient.keep)
            {
                Some(PressureAction::DropCache(cache.item))
            } else {
                Some(PressureAction::SpillBottomTransient)
            }
        }
        (Some(cache), None) => Some(PressureAction::DropCache(cache.item)),
        (None, Some(_)) => Some(PressureAction::SpillBottomTransient),
        (None, None) => None,
    }
}

fn choose_transient_victim(
    state: &PrepareState,
    need_gp: bool,
    need_fp: bool,
    post_height: u16,
    op_index: usize,
    op_info: &[OpInfo],
    block_transient_regions: &[super::facts::BlockTransientRegion],
) -> Option<PressureCandidate<()>> {
    if state.spill_depth >= post_height {
        return None;
    }
    let live_index = state.spill_depth as usize;
    let ty = *state.type_stack.get(live_index)?;
    let bank = if ty.is_float() {
        CacheBank::Fp
    } else {
        CacheBank::Gp
    };
    let symbol = *state.block_symbols.get(live_index)?;
    Some(PressureCandidate {
        item: (),
        keep: transient_keep_key(op_index, op_info, block_transient_regions, symbol),
        relief_score: u8::from(need_gp && bank == CacheBank::Gp)
            .saturating_add(u8::from(need_fp && bank == CacheBank::Fp)),
    })
}

fn local_target_keep_key(
    op_index: usize,
    op_info: &[OpInfo],
    block_regions: &[super::facts::BlockLocalRegion],
    slot: FrameSlot,
) -> LocalKeepKey {
    let info = &op_info[op_index];
    let Some(local_info) = block_regions[info.block_index as usize].info(slot) else {
        return LocalKeepKey::weakest();
    };
    if !local_info.used_after(info.block_offset) {
        return LocalKeepKey::weakest();
    }
    let remaining = i32::from(local_info.remaining_use_count(info.block_offset));
    let next_distance = i32::from(local_info.next_use_distance(info.block_offset).unwrap_or(0));
    LocalKeepKey {
        class: 2,
        score: local_info.hot_score + remaining * 64 - next_distance,
    }
}

fn local_keep_key(
    op_index: usize,
    op_info: &[OpInfo],
    block_regions: &[super::facts::BlockLocalRegion],
    slot: FrameSlot,
    used_now: Option<FrameSlot>,
    carried_bonus: bool,
) -> LocalKeepKey {
    if used_now == Some(slot) {
        return LocalKeepKey {
            class: 3,
            score: i32::MAX,
        };
    }
    let info = &op_info[op_index];
    let Some(local_info) = block_regions[info.block_index as usize].info(slot) else {
        return LocalKeepKey::weakest();
    };
    if local_info.used_after(info.block_offset) {
        let remaining = i32::from(local_info.remaining_use_count(info.block_offset));
        let next_distance = i32::from(local_info.next_use_distance(info.block_offset).unwrap_or(0));
        return LocalKeepKey {
            class: 2,
            score: local_info.hot_score + remaining * 64 - next_distance
                + if carried_bonus { 1024 } else { 0 },
        };
    }
    if local_info.used_anywhere() {
        return LocalKeepKey {
            class: 1,
            score: local_info.hot_score + if carried_bonus { 1024 } else { 0 },
        };
    }
    LocalKeepKey::weakest()
}

fn transient_keep_key(
    op_index: usize,
    op_info: &[OpInfo],
    block_transient_regions: &[super::facts::BlockTransientRegion],
    symbol: u16,
) -> LocalKeepKey {
    let info = &op_info[op_index];
    let Some(symbol_info) = block_transient_regions[info.block_index as usize].info(symbol) else {
        return LocalKeepKey::weakest();
    };
    if symbol_info.used_after(info.block_offset) {
        let remaining = i32::from(symbol_info.remaining_use_count(info.block_offset));
        let next_distance = i32::from(
            symbol_info
                .next_use_distance(info.block_offset)
                .unwrap_or(0),
        );
        return LocalKeepKey {
            class: 2,
            score: symbol_info.hot_score + remaining * 64 - next_distance,
        };
    }
    if symbol_info.used_anywhere() {
        return LocalKeepKey {
            class: 1,
            score: symbol_info.hot_score,
        };
    }
    LocalKeepKey::weakest()
}

fn emit_drop_cache(
    prefix: &mut Vec<PrepAction>,
    state: &mut PrepareState,
    cache_plan: &CachePlan,
    index: usize,
) {
    if !state.resident_cache.get(index).copied().unwrap_or(false) {
        return;
    }
    let candidate = cache_plan.candidates[index];
    prefix.push(PrepAction::DropCache(candidate.slot));
    state.resident_cache[index] = false;
    match candidate.bank {
        CacheBank::Gp => {
            state.resident_gp_units = state.resident_gp_units.saturating_sub(candidate.cost)
        }
        CacheBank::Fp => {
            state.resident_fp_units = state.resident_fp_units.saturating_sub(candidate.cost)
        }
    }
}

fn drop_all_caches(prefix: &mut Vec<PrepAction>, state: &mut PrepareState, cache_plan: &CachePlan) {
    for index in 0..cache_plan.candidates.len() {
        emit_drop_cache(prefix, state, cache_plan, index);
    }
}

fn apply_block_symbolic_effect(op: &SemanticOp, state: &mut PrepareState) {
    match &op.kind {
        SemanticOpKind::Primitive(kind) => {
            if matches!(kind, primitive_op::PrimitiveOpKind::Unreachable) {
                state.block_symbols.clear();
                return;
            }
            let (pop, push) = primitive_op::stack_effect(kind);
            for _ in 0..pop {
                state.block_symbols.pop();
            }
            for _ in 0..push {
                state.push_fresh_block_symbol();
            }
        }
        SemanticOpKind::LocalGet { .. } => state.push_fresh_block_symbol(),
        SemanticOpKind::LocalSet { .. } => {
            state.block_symbols.pop();
        }
        SemanticOpKind::LocalTee { .. } => {}
        SemanticOpKind::If { .. } => {
            state.block_symbols.pop();
        }
        SemanticOpKind::Br { .. }
        | SemanticOpKind::BrIf { .. }
        | SemanticOpKind::BrTable { .. }
        | SemanticOpKind::ReturnVoid
        | SemanticOpKind::ReturnOne
        | SemanticOpKind::Return { .. } => {}
        SemanticOpKind::CallDirect {
            params, results, ..
        } => {
            for _ in 0..*params {
                state.block_symbols.pop();
            }
            for _ in 0..*results {
                state.push_fresh_block_symbol();
            }
        }
        SemanticOpKind::CallIndirect {
            params, results, ..
        } => {
            for _ in 0..params.saturating_add(1) {
                state.block_symbols.pop();
            }
            for _ in 0..*results {
                state.push_fresh_block_symbol();
            }
        }
        // Structured markers do not introduce block-local transient value
        // identities by themselves. Real stack joins are already split into CFG
        // block entries where we reset `block_symbols`.
        SemanticOpKind::Block { .. }
        | SemanticOpKind::Loop { .. }
        | SemanticOpKind::Else { .. }
        | SemanticOpKind::End => {}
    }
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
        SemanticOpKind::ReturnVoid | SemanticOpKind::ReturnOne | SemanticOpKind::Return { .. } => {
            state.mark_unreachable();
        }
    }
}

fn push_result_types(
    type_stack: &mut Vec<ValueType>,
    results: u16,
    op_index: usize,
    op_result_types: &BTreeMap<usize, Vec<ValueType>>,
) {
    if let Some(types) = op_result_types.get(&op_index) {
        type_stack.extend_from_slice(types);
    } else {
        for _ in 0..results {
            type_stack.push(ValueType::I64);
        }
    }
}

fn normalized_result_types(results: u16, result_types: Option<&[ValueType]>) -> Vec<ValueType> {
    if results == 0 {
        return Vec::new();
    }
    if let Some(types) = result_types {
        if types.len() == results as usize {
            return types.to_vec();
        }
    }
    alloc::vec![ValueType::I64; results as usize]
}

fn control_result_types(
    results: u16,
    op_index: usize,
    op_result_types: &BTreeMap<usize, Vec<ValueType>>,
) -> Vec<ValueType> {
    normalized_result_types(results, op_result_types.get(&op_index).map(Vec::as_slice))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::middle::joint_plan::facts::{
        BlockEntryStackRegion, BlockLocalInfo, BlockLocalRegion, BlockStackValueInfo,
    };

    fn test_plan() -> FunctionPlan {
        FunctionPlan {
            gp_unit_bytes: 8,
            gp_dynamic_budget: 2,
            fp_dynamic_budget: 2,
            local_slot_types: alloc::vec![ValueType::I32],
            op_plans: alloc::vec![OpPlan::default()],
            entry_states: alloc::vec![EntryState {
                stack_height: 2,
                spill_depth: 0,
                stack_types: alloc::vec![ValueType::I32, ValueType::I32],
                live_types: alloc::vec![ValueType::I32, ValueType::I32],
            }],
            op_info: alloc::vec![OpInfo {
                block_index: 0,
                block_offset: 0,
                is_block_start: true,
                local_op: None,
            }],
            block_regions: alloc::vec![BlockLocalRegion {
                ranked_slots: alloc::vec![FrameSlot(0)],
                locals: alloc::vec![Some(BlockLocalInfo {
                    slot: FrameSlot(0),
                    first_access_kind: Some(FirstAccessKind::ReadFirst),
                    first_read_distance: Some(0),
                    first_write_distance: None,
                    read_count: 2,
                    write_count: 0,
                    access_offsets: alloc::vec![0, 1],
                    hot_score: 400,
                })],
            }],
            block_stack_regions: alloc::vec![BlockEntryStackRegion {
                entry_stack_height: 2,
                values: alloc::vec![
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
            block_transient_regions: alloc::vec![
                crate::vm::middle::joint_plan::facts::BlockTransientRegion::default()
            ],
            blocks: alloc::vec![BlockPlan::default()],
        }
    }

    #[test]
    fn tentative_entry_prefers_hot_stack_suffix_and_hot_carried_local() {
        let plan = test_plan();
        let tentative = choose_tentative_block_entry(0, &[FrameSlot(0)], &plan);
        assert_eq!(tentative.transient.spill_depth, 1);
        assert_eq!(tentative.transient.live_types, alloc::vec![ValueType::I32]);
        assert_eq!(tentative.cached_locals, alloc::vec![FrameSlot(0)]);
    }

    #[test]
    fn finalize_entry_keeps_used_and_surviving_carried_values() {
        let mut plan = test_plan();
        plan.block_regions[0].locals.push(None);
        let finalized =
            finalize_block_entry(0, &plan, &[FrameSlot(0), FrameSlot(1)], &[FrameSlot(1)]);
        assert_eq!(finalized, alloc::vec![FrameSlot(0), FrameSlot(1)]);

        let finalized = finalize_block_entry(0, &plan, &[FrameSlot(1)], &[]);
        assert!(finalized.is_empty());
    }

    #[test]
    fn ensure_capacity_can_spill_through_mismatched_bottom_bank() {
        let mut state = PrepareState::new(0, alloc::vec![], alloc::vec![], 0);
        state.height = 3;
        state.spill_depth = 0;
        state.type_stack = alloc::vec![ValueType::F32, ValueType::I32, ValueType::I32];
        state.block_symbols = alloc::vec![0, 1, 2];
        state.next_block_symbol = 3;

        let op_info = alloc::vec![OpInfo {
            block_index: 0,
            block_offset: 0,
            is_block_start: true,
            local_op: None,
        }];
        let block_regions = alloc::vec![BlockLocalRegion::default()];
        let block_transient_regions =
            alloc::vec![crate::vm::middle::joint_plan::facts::BlockTransientRegion {
                entry_stack_height: 3,
                symbols: alloc::vec![
                    crate::vm::middle::joint_plan::facts::TransientSymbolInfo {
                        first_touch_distance: None,
                        touch_count: 0,
                        access_offsets: alloc::vec![],
                        hot_score: 0,
                    },
                    crate::vm::middle::joint_plan::facts::TransientSymbolInfo {
                        first_touch_distance: Some(2),
                        touch_count: 1,
                        access_offsets: alloc::vec![2],
                        hot_score: 10,
                    },
                    crate::vm::middle::joint_plan::facts::TransientSymbolInfo {
                        first_touch_distance: Some(3),
                        touch_count: 1,
                        access_offsets: alloc::vec![3],
                        hot_score: 10,
                    },
                ],
            }];
        let mut prefix = Vec::new();

        let ok = ensure_capacity(
            &mut prefix,
            &mut state,
            &CachePlan {
                candidates: Vec::new(),
                local_to_candidate: Vec::new(),
            },
            FrameLayoutPlan::default(),
            8,
            1,
            2,
            0,
            &op_info,
            &block_regions,
            &block_transient_regions,
            0,
            &[],
            None,
        );

        assert!(
            ok,
            "pressure should be resolved by spilling through the bottom FP value first"
        );
        assert_eq!(state.spill_depth, 2);
        assert_eq!(prefix.len(), 2);
    }
}
