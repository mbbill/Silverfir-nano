//! First complete joint-plan implementation.
//!
//! This planner is intentionally simple for cached locals, but it is strict
//! about transient legality. It produces:
//! - exact transient spill/fill behavior per semantic op
//! - local slot-vs-cache access choice per semantic op
//! - a canonical predecessor per block
//! - a chosen cached-local entry set per block

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::{
    error::WasmError,
    value_type::ValueType,
    vm::{
        backend::BackendConfig,
        middle::{
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
    canonical::choose_canonical_predecessors,
    scan::scan_local_rankings,
    types::{
        BlockPlan, BoundaryState, EntryState, FunctionPlan, OpPlan, PrepAction, PreparedLocalAccess,
    },
};

pub(crate) fn build_plan(
    semantic: &SemanticProgram,
    cfg: &SemanticCfg,
    slot_program: &SlotSsaProgram,
    frame: FrameLayoutPlan,
    config: BackendConfig,
) -> Result<FunctionPlan, WasmError> {
    let gp_dynamic_budget = config
        .gp_transient_budget
        .saturating_add(config.gp_local_cache_budget);
    let fp_dynamic_budget = config
        .fp_transient_budget
        .saturating_add(config.fp_local_cache_budget);

    let (op_plans, entry_states) = prepare_semantic_ops(
        semantic,
        frame,
        config.gp_unit_bytes,
        gp_dynamic_budget,
        fp_dynamic_budget,
    )?;
    let cacheable_slots = collect_cacheable_slots(semantic, &op_plans, frame);
    let canonical = choose_canonical_predecessors(cfg);
    let rankings = scan_local_rankings(slot_program, frame);
    let block_peak_live = compute_block_peak_live_counts(cfg, &entry_states, config.gp_unit_bytes);
    let block_entries = choose_block_entry_cached_locals(
        semantic,
        cfg,
        frame,
        &canonical.by_block,
        &rankings.blocks,
        &block_peak_live,
        &cacheable_slots,
        config.gp_unit_bytes,
        gp_dynamic_budget,
        fp_dynamic_budget,
    );

    let blocks = canonical
        .by_block
        .into_iter()
        .enumerate()
        .map(|(block_index, canonical_predecessor)| BlockPlan {
            canonical_predecessor,
            entry: BoundaryState {
                cached_locals: block_entries.get(block_index).cloned().unwrap_or_default(),
                stack_values: Vec::new(),
            },
            exit: BoundaryState::default(),
        })
        .collect();

    Ok(FunctionPlan {
        gp_unit_bytes: config.gp_unit_bytes,
        gp_dynamic_budget,
        fp_dynamic_budget,
        op_plans,
        entry_states,
        blocks,
        repairs: Vec::new(),
    })
}

fn choose_block_entry_cached_locals(
    semantic: &SemanticProgram,
    cfg: &SemanticCfg,
    frame: FrameLayoutPlan,
    canonical_predecessors: &[Option<crate::vm::middle::cfg::CfgBlockId>],
    rankings: &[super::scan::BlockLocalRanking],
    block_peak_live: &[LiveCounts],
    cacheable_slots: &[FrameSlot],
    gp_unit_bytes: u8,
    gp_dynamic_budget: u8,
    fp_dynamic_budget: u8,
) -> Vec<Vec<FrameSlot>> {
    let mut entries = rankings
        .iter()
        .enumerate()
        .map(|(block_index, ranking)| {
            let slots = if cfg.blocks[block_index].flags.is_entry {
                Vec::new()
            } else {
                ranking
                    .slots
                    .iter()
                    .copied()
                    .filter(|slot| cacheable_slots.contains(slot))
                    .collect()
            };
            clip_cached_locals_to_entry_budget(
                slots,
                semantic,
                frame,
                block_peak_live
                    .get(block_index)
                    .copied()
                    .unwrap_or_default(),
                gp_unit_bytes,
                gp_dynamic_budget,
                fp_dynamic_budget,
            )
        })
        .collect::<Vec<_>>();

    let iterations = cfg.blocks.len().max(1) * 2;
    for _ in 0..iterations {
        let mut changed = false;
        for (block_index, block) in cfg.blocks.iter().enumerate() {
            if block.flags.is_entry {
                continue;
            }
            let mut desired = rankings
                .get(block_index)
                .map(|ranking| ranking.slots.clone())
                .unwrap_or_default();
            desired.retain(|slot| cacheable_slots.contains(slot));
            if let Some(pred) = canonical_predecessors.get(block_index).copied().flatten() {
                if let Some(pred_entry) = entries.get(pred.as_usize()) {
                    for &slot in pred_entry {
                        if !desired.contains(&slot) {
                            desired.push(slot);
                        }
                    }
                }
            }
            let clipped = clip_cached_locals_to_entry_budget(
                desired,
                semantic,
                frame,
                block_peak_live.get(block_index).copied().unwrap_or_default(),
                gp_unit_bytes,
                gp_dynamic_budget,
                fp_dynamic_budget,
            );
            if entries.get(block_index) != Some(&clipped) {
                entries[block_index] = clipped;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    entries
}

fn clip_cached_locals_to_entry_budget(
    slots: impl IntoIterator<Item = FrameSlot>,
    semantic: &SemanticProgram,
    frame: FrameLayoutPlan,
    peak_live: LiveCounts,
    gp_unit_bytes: u8,
    gp_dynamic_budget: u8,
    fp_dynamic_budget: u8,
) -> Vec<FrameSlot> {
    let mut gp_live = peak_live.gp;
    let mut fp_live = peak_live.fp;
    let mut out = Vec::new();
    for slot in slots {
        let local_index = slot.0 as usize;
        let ty = semantic
            .local_types
            .get(local_index)
            .copied()
            .unwrap_or(ValueType::I64);
        let cost = if ty.is_float() {
            1
        } else {
            gp_value_budget_units(ty, gp_unit_bytes)
        };
        let fits = if ty.is_float() {
            fp_live + cost <= fp_dynamic_budget as usize
        } else {
            gp_live + cost <= gp_dynamic_budget as usize
        };
        if !fits {
            continue;
        }
        if ty.is_float() {
            fp_live += cost;
        } else {
            gp_live += cost;
        }
        if frame.local_slot(local_index as u16) == slot && !out.contains(&slot) {
            out.push(slot);
        }
    }
    out
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct LiveCounts {
    gp: usize,
    fp: usize,
}

fn compute_block_peak_live_counts(
    cfg: &SemanticCfg,
    entry_states: &[EntryState],
    gp_unit_bytes: u8,
) -> Vec<LiveCounts> {
    cfg.blocks
        .iter()
        .map(|block| {
            let mut peak = LiveCounts::default();
            for semantic_index in block.range.clone() {
                if let Some(entry) = entry_states.get(semantic_index) {
                    let (gp, fp) = count_live_bank_budget_units(&entry.live_types, gp_unit_bytes);
                    peak.gp = peak.gp.max(gp);
                    peak.fp = peak.fp.max(fp);
                }
            }
            peak
        })
        .collect()
}

fn collect_cacheable_slots(
    semantic: &SemanticProgram,
    op_plans: &[OpPlan],
    frame: FrameLayoutPlan,
) -> Vec<FrameSlot> {
    let mut out = Vec::new();
    for (op, plan) in semantic.ops.iter().zip(op_plans.iter()) {
        let local = match op.kind {
            SemanticOpKind::LocalGet { idx }
            | SemanticOpKind::LocalSet { idx }
            | SemanticOpKind::LocalTee { idx } => Some(idx),
            _ => None,
        };
        if plan.local_access != PreparedLocalAccess::Cache {
            continue;
        }
        let Some(local) = local else {
            continue;
        };
        let slot = frame.local_slot(local);
        if !out.contains(&slot) {
            out.push(slot);
        }
    }
    out
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

fn prepare_semantic_ops(
    semantic: &SemanticProgram,
    frame: FrameLayoutPlan,
    gp_unit_bytes: u8,
    gp_dynamic_budget: u8,
    fp_dynamic_budget: u8,
) -> Result<(Vec<OpPlan>, Vec<EntryState>), WasmError> {
    let cache_plan = build_cache_plan(semantic, frame, gp_unit_bytes);
    let future_local_accesses = compute_future_local_accesses_in_region(semantic);
    let mut state = PrepareState::new(
        semantic.results,
        semantic.local_types.clone(),
        semantic.result_types.clone(),
        cache_plan.candidates.len(),
    );
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
        let (prefix, local_access) = plan_prefix(
            semantic_op,
            op_index,
            &mut state,
            &cache_plan,
            frame,
            gp_unit_bytes,
            gp_dynamic_budget,
            fp_dynamic_budget,
            &semantic.op_result_types,
            future_local_accesses
                .get(op_index)
                .copied()
                .unwrap_or_default(),
        );
        ops.push(OpPlan {
            prefix,
            local_access,
        });
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
    op_result_types: &BTreeMap<usize, Vec<ValueType>>,
    has_future_local_access_in_region: bool,
) -> (Vec<PrepAction>, PreparedLocalAccess) {
    let mut prefix = Vec::new();
    let mut local_access = PreparedLocalAccess::Slot;

    match &op.kind {
        SemanticOpKind::Primitive(kind) => {
            if matches!(kind, PrimitiveOpKind::Unreachable) {
                return (prefix, local_access);
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
                    pop as u16,
                    core::slice::from_ref(&push_ty.unwrap_or(ValueType::I64)),
                    None,
                );
            }
        }
        SemanticOpKind::LocalGet { idx } => {
            let push_ty = state.local_type(*idx);
            local_access = plan_local_access(
                *idx,
                &mut prefix,
                state,
                cache_plan,
                frame,
                gp_unit_bytes,
                gp_dynamic_budget,
                fp_dynamic_budget,
                0,
                core::slice::from_ref(&push_ty),
                has_future_local_access_in_region,
            );
        }
        SemanticOpKind::LocalSet { idx } => {
            fill_for_operands(&mut prefix, state, frame, 1);
            local_access = plan_local_access(
                *idx,
                &mut prefix,
                state,
                cache_plan,
                frame,
                gp_unit_bytes,
                gp_dynamic_budget,
                fp_dynamic_budget,
                1,
                &[],
                has_future_local_access_in_region,
            );
        }
        SemanticOpKind::LocalTee { idx } => {
            fill_for_operands(&mut prefix, state, frame, 1);
            let push_ty = state.local_type(*idx);
            local_access = plan_local_access(
                *idx,
                &mut prefix,
                state,
                cache_plan,
                frame,
                gp_unit_bytes,
                gp_dynamic_budget,
                fp_dynamic_budget,
                1,
                core::slice::from_ref(&push_ty),
                has_future_local_access_in_region,
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
                    return (prefix, local_access);
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

    (prefix, local_access)
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
    pop: u16,
    push_types: &[ValueType],
    has_future_local_access_in_region: bool,
) -> PreparedLocalAccess {
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
            pop,
            push_types,
            None,
        );
        return PreparedLocalAccess::Slot;
    };

    if !state.resident_cache[index] && !has_future_local_access_in_region {
        ensure_capacity(
            prefix,
            state,
            cache_plan,
            frame,
            gp_unit_bytes,
            gp_dynamic_budget,
            fp_dynamic_budget,
            pop,
            push_types,
            None,
        );
        return PreparedLocalAccess::Slot;
    }

    if ensure_capacity(
        prefix,
        state,
        cache_plan,
        frame,
        gp_unit_bytes,
        gp_dynamic_budget,
        fp_dynamic_budget,
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
        return PreparedLocalAccess::Cache;
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
        pop,
        push_types,
        None,
    );
    PreparedLocalAccess::Slot
}

fn compute_future_local_accesses_in_region(semantic: &SemanticProgram) -> Vec<bool> {
    let mut future_access = alloc::vec![false; semantic.ops.len()];
    if semantic.local_count == 0 {
        return future_access;
    }

    let mut future_counts = alloc::vec![0u32; semantic.local_count as usize];
    for (op_index, op) in semantic.ops.iter().enumerate().rev() {
        if clears_cache_region(&op.kind) {
            future_counts.fill(0);
        }
        match op.kind {
            SemanticOpKind::LocalGet { idx }
            | SemanticOpKind::LocalSet { idx }
            | SemanticOpKind::LocalTee { idx } => {
                let local_idx = idx as usize;
                if let Some(count) = future_counts.get_mut(local_idx) {
                    future_access[op_index] = *count != 0;
                    *count = count.saturating_add(1);
                }
            }
            _ => {}
        }
    }
    future_access
}

fn clears_cache_region(kind: &SemanticOpKind) -> bool {
    matches!(
        kind,
        SemanticOpKind::Loop { .. }
            | SemanticOpKind::If { .. }
            | SemanticOpKind::Else { .. }
            | SemanticOpKind::End
            | SemanticOpKind::Br { .. }
            | SemanticOpKind::BrIf { .. }
            | SemanticOpKind::BrTable { .. }
            | SemanticOpKind::CallDirect { .. }
            | SemanticOpKind::CallIndirect { .. }
            | SemanticOpKind::ReturnVoid
            | SemanticOpKind::ReturnOne
            | SemanticOpKind::Return { .. }
            | SemanticOpKind::Primitive(PrimitiveOpKind::Unreachable)
    )
}

fn ensure_capacity(
    prefix: &mut Vec<PrepAction>,
    state: &mut PrepareState,
    cache_plan: &CachePlan,
    frame: FrameLayoutPlan,
    gp_unit_bytes: u8,
    gp_dynamic_budget: u8,
    fp_dynamic_budget: u8,
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

        if let Some(victim) =
            choose_cache_victim(state, cache_plan, preserve_cache, need_gp, need_fp)
        {
            emit_drop_cache(prefix, state, cache_plan, victim);
            continue;
        }

        if state.spill_depth < post_height {
            prefix.push(PrepAction::Spill(FrameSpan::new(
                frame.operand_slot(state.spill_depth),
                1,
            )));
            state.spill_depth += 1;
            continue;
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
) -> Option<usize> {
    let mut best = None;
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
        if best
            .map(|current: usize| candidate.rank > cache_plan.candidates[current].rank)
            .unwrap_or(true)
        {
            best = Some(index);
        }
    }

    if best.is_some() || (!need_gp && !need_fp) {
        return best;
    }

    for (index, candidate) in cache_plan.candidates.iter().copied().enumerate() {
        if Some(index) == preserve_cache
            || !state.resident_cache.get(index).copied().unwrap_or(false)
        {
            continue;
        }
        if best
            .map(|current: usize| candidate.rank > cache_plan.candidates[current].rank)
            .unwrap_or(true)
        {
            best = Some(index);
        }
    }

    best
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

pub(crate) fn count_live_bank_budget_units(
    types: &[ValueType],
    gp_unit_bytes: u8,
) -> (usize, usize) {
    let mut gp = 0usize;
    let mut fp = 0usize;
    for ty in types {
        match ty {
            ValueType::F32 | ValueType::F64 => fp += 1,
            _ => gp += gp_value_budget_units(*ty, gp_unit_bytes),
        }
    }
    (gp, fp)
}

pub(crate) fn gp_value_budget_units(ty: ValueType, gp_unit_bytes: u8) -> usize {
    match ty {
        ValueType::I64 if gp_unit_bytes < 8 => 2,
        _ => 1,
    }
}
