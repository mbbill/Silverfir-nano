//! Late resource planning for explicit cached locals.
//!
//! Early preparation picks transient spills/fills and initial cache accesses,
//! but later SSA cleanup can change the final per-block live shape. This pass
//! finalizes the explicit cache plan by:
//! - choosing one canonical cached-local entry state per SSA block,
//! - repairing incompatible edges with explicit `LocalEnsureCache` /
//!   `LocalDropCache` bridge blocks,
//! - dropping resident cached locals when final transient pressure needs room,
//! - demoting `LocalGetCache` / `LocalSetCache` back to slot ops if the final
//!   block shape cannot keep that local resident.

use alloc::{
    collections::{BTreeMap, BTreeSet},
    vec,
    vec::Vec,
};

use crate::{
    value_type::ValueType,
    vm::middle::{
        frame::FrameSlot,
        ssa_ir::ir::{SsaInst, SsaInstKind, SsaOperand, SsaProgram, SsaTerminator, SsaValue},
        state::gp_value_budget_units,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CacheBank {
    Gp,
    Fp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CacheCandidate {
    bank: CacheBank,
    cost: usize,
    rank: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct LiveCounts {
    gp: usize,
    fp: usize,
}

pub(super) fn plan_resources(
    program: &mut SsaProgram,
    gp_unit_bytes: u8,
    gp_dynamic_budget: u8,
    fp_dynamic_budget: u8,
) {
    let cache_candidates = build_cache_candidates(program, gp_unit_bytes);
    let value_types = program.value_types.clone();
    let original_blocks = program.blocks.clone();
    let predecessors = compute_predecessors(&program.blocks);
    let mut entry_hints = original_blocks
        .iter()
        .map(|block| {
            desired_entry_slots_for_block(
                block,
                &cache_candidates,
                &value_types,
                gp_unit_bytes,
                gp_dynamic_budget,
                fp_dynamic_budget,
            )
        })
        .collect::<Vec<_>>();
    let mut exit_cached_slots = vec![Vec::new(); original_blocks.len()];
    let max_iterations = original_blocks.len().max(1) * cache_candidates.len().max(1) * 4;
    let mut converged = false;

    for _ in 0..max_iterations {
        let block_entry_cached_slots =
            canonical_block_entry_cached_slots(program, &entry_hints, &predecessors);
        let wanted_exit_slots =
            compute_wanted_exit_slots(&original_blocks, &block_entry_cached_slots, &cache_candidates);
        let (desired_entry_slots, rewritten_exit_slots) = rewrite_blocks(
            &mut program.blocks,
            &original_blocks,
            &cache_candidates,
            &value_types,
            gp_unit_bytes,
            gp_dynamic_budget,
            fp_dynamic_budget,
            &block_entry_cached_slots,
            &wanted_exit_slots,
        );
        exit_cached_slots = rewritten_exit_slots;
        if desired_entry_slots == entry_hints {
            entry_hints = desired_entry_slots;
            converged = true;
            break;
        }
        entry_hints = desired_entry_slots;
    }

    debug_assert!(
        converged,
        "resource plan canonical entry iteration did not converge"
    );

    program.block_entry_cached_slots =
        canonical_block_entry_cached_slots(program, &entry_hints, &predecessors);
    insert_boundary_repair_blocks(program, &cache_candidates, &exit_cached_slots);
}

fn rewrite_blocks(
    blocks: &mut [crate::vm::middle::ssa_ir::ir::SsaBlock],
    original_blocks: &[crate::vm::middle::ssa_ir::ir::SsaBlock],
    cache_candidates: &BTreeMap<FrameSlot, CacheCandidate>,
    value_types: &[ValueType],
    gp_unit_bytes: u8,
    gp_dynamic_budget: u8,
    fp_dynamic_budget: u8,
    desired_entry_hints: &[Vec<FrameSlot>],
    wanted_exit_slots: &[Vec<FrameSlot>],
) -> (Vec<Vec<FrameSlot>>, Vec<Vec<FrameSlot>>) {
    let block_count = blocks.len();
    let mut desired_entry_slots = vec![Vec::new(); block_count];
    let mut exit_cached_slots = vec![Vec::new(); block_count];

    for (block, original_block) in blocks.iter_mut().zip(original_blocks.iter()) {
        let block_id = block.id.as_usize();
        let desired_entry_hint = desired_entry_hints
            .get(block_id)
            .cloned()
            .unwrap_or_default();
        let wanted_exit = wanted_exit_slots
            .get(block_id)
            .cloned()
            .unwrap_or_default();
        let mut remaining_uses = compute_remaining_uses(original_block);
        let mut remaining_cache_accesses = compute_remaining_cache_accesses(original_block);
        let mut live_values = BTreeMap::<SsaValue, ValueType>::new();
        let mut live = LiveCounts::default();
        for &param in &original_block.params {
            let remaining = remaining_uses.get(&param).copied().unwrap_or(0);
            if remaining == 0 {
                continue;
            }
            let ty = value_type_from_slice(value_types, param);
            live_add(&mut live_values, &mut live, param, ty, gp_unit_bytes);
        }

        let mut resident = BTreeSet::<FrameSlot>::new();
        let mut resident_counts = LiveCounts::default();
        let mut rewritten = Vec::with_capacity(original_block.ops.len());

        for inst in original_block.ops.iter().cloned() {
            match inst.kind {
                SsaInstKind::Value { op, args, results } => {
                    for operand in &args {
                        consume_operand(operand, &mut remaining_uses, &mut live_values, &mut live, gp_unit_bytes);
                    }
                    let result_types = result_types(value_types, &results);
                    let next_live = add_result_counts(live, &result_types, gp_unit_bytes);
                    drop_for_pressure(
                        &mut rewritten,
                        cache_candidates,
                        &mut resident,
                        &mut resident_counts,
                        next_live,
                        gp_dynamic_budget,
                        fp_dynamic_budget,
                        None,
                    );
                    rewritten.push(SsaInst {
                        kind: SsaInstKind::Value { op, args, results: results.clone() },
                    });
                    add_results(&results, &remaining_uses, &mut live_values, &mut live, value_types, gp_unit_bytes);
                }
                SsaInstKind::Fill { slot, dst } => {
                    let dst_ty = value_type_from_slice(value_types, dst);
                    let next_live = add_result_counts(live, core::slice::from_ref(&dst_ty), gp_unit_bytes);
                    drop_for_pressure(
                        &mut rewritten,
                        cache_candidates,
                        &mut resident,
                        &mut resident_counts,
                        next_live,
                        gp_dynamic_budget,
                        fp_dynamic_budget,
                        None,
                    );
                    rewritten.push(SsaInst { kind: SsaInstKind::Fill { slot, dst } });
                    add_result(dst, &remaining_uses, &mut live_values, &mut live, dst_ty, gp_unit_bytes);
                }
                SsaInstKind::LocalGetSlot { slot, dst } => {
                    let dst_ty = value_type_from_slice(value_types, dst);
                    let next_live = add_result_counts(live, core::slice::from_ref(&dst_ty), gp_unit_bytes);
                    drop_for_pressure(
                        &mut rewritten,
                        cache_candidates,
                        &mut resident,
                        &mut resident_counts,
                        next_live,
                        gp_dynamic_budget,
                        fp_dynamic_budget,
                        None,
                    );
                    rewritten.push(SsaInst { kind: SsaInstKind::LocalGetSlot { slot, dst } });
                    add_result(dst, &remaining_uses, &mut live_values, &mut live, dst_ty, gp_unit_bytes);
                }
                SsaInstKind::LocalGetCache { slot, dst } => {
                    let future_slot_accesses = decrement_cache_access(&mut remaining_cache_accesses, slot);
                    let dst_ty = value_type_from_slice(value_types, dst);
                    let next_live = add_result_counts(live, core::slice::from_ref(&dst_ty), gp_unit_bytes);
                    let keep_cache =
                        resident.contains(&slot)
                            || future_slot_accesses != 0
                            || desired_entry_hint.contains(&slot)
                            || wanted_exit.contains(&slot);
                    let keep_cache = if keep_cache {
                        ensure_cache_capacity(
                            &mut rewritten,
                            cache_candidates,
                            &mut resident,
                            &mut resident_counts,
                            slot,
                            next_live,
                            gp_dynamic_budget,
                            fp_dynamic_budget,
                        )
                    } else {
                        false
                    };
                    if keep_cache {
                        resident_add(slot, cache_candidates, &mut resident, &mut resident_counts);
                        rewritten.push(SsaInst { kind: SsaInstKind::LocalGetCache { slot, dst } });
                    } else {
                        emit_drop_slot(
                            &mut rewritten,
                            cache_candidates,
                            &mut resident,
                            &mut resident_counts,
                            slot,
                        );
                        rewritten.push(SsaInst { kind: SsaInstKind::LocalGetSlot { slot, dst } });
                    }
                    add_result(dst, &remaining_uses, &mut live_values, &mut live, dst_ty, gp_unit_bytes);
                }
                SsaInstKind::Spill { slot, src } => {
                    consume_value(src, &mut remaining_uses, &mut live_values, &mut live, gp_unit_bytes);
                    rewritten.push(SsaInst { kind: SsaInstKind::Spill { slot, src } });
                }
                SsaInstKind::LocalSetSlot { slot, src } => {
                    consume_value(src, &mut remaining_uses, &mut live_values, &mut live, gp_unit_bytes);
                    rewritten.push(SsaInst { kind: SsaInstKind::LocalSetSlot { slot, src } });
                }
                SsaInstKind::LocalSetCache { slot, src } => {
                    let future_slot_accesses = decrement_cache_access(&mut remaining_cache_accesses, slot);
                    consume_value(src, &mut remaining_uses, &mut live_values, &mut live, gp_unit_bytes);
                    let keep_cache =
                        resident.contains(&slot)
                            || future_slot_accesses != 0
                            || desired_entry_hint.contains(&slot)
                            || wanted_exit.contains(&slot);
                    let keep_cache = if keep_cache {
                        ensure_cache_capacity(
                            &mut rewritten,
                            cache_candidates,
                            &mut resident,
                            &mut resident_counts,
                            slot,
                            live,
                            gp_dynamic_budget,
                            fp_dynamic_budget,
                        )
                    } else {
                        false
                    };
                    if keep_cache {
                        resident_add(slot, cache_candidates, &mut resident, &mut resident_counts);
                        rewritten.push(SsaInst { kind: SsaInstKind::LocalSetCache { slot, src } });
                    } else {
                        emit_drop_slot(
                            &mut rewritten,
                            cache_candidates,
                            &mut resident,
                            &mut resident_counts,
                            slot,
                        );
                        rewritten.push(SsaInst { kind: SsaInstKind::LocalSetSlot { slot, src } });
                    }
                }
                SsaInstKind::LocalEnsureCache { slot } => {
                    if ensure_cache_capacity(
                        &mut rewritten,
                        cache_candidates,
                        &mut resident,
                        &mut resident_counts,
                        slot,
                        live,
                        gp_dynamic_budget,
                        fp_dynamic_budget,
                    ) {
                        resident_add(slot, cache_candidates, &mut resident, &mut resident_counts);
                    }
                    rewritten.push(SsaInst { kind: SsaInstKind::LocalEnsureCache { slot } });
                }
                SsaInstKind::LocalDropCache { slot } => {
                    if resident.remove(&slot) {
                        resident_sub(slot, cache_candidates, &mut resident_counts);
                        rewritten.push(SsaInst { kind: SsaInstKind::LocalDropCache { slot } });
                    }
                }
                SsaInstKind::Call(call) => {
                    emit_boundary_drops(
                        &mut rewritten,
                        cache_candidates,
                        &mut resident,
                        &mut resident_counts,
                    );
                    rewritten.push(SsaInst {
                        kind: SsaInstKind::Call(call),
                    });
                }
            }
        }
        block.ops = rewritten;
        desired_entry_slots[block_id] = desired_entry_slots_for_block(
            block,
            cache_candidates,
            value_types,
            gp_unit_bytes,
            gp_dynamic_budget,
            fp_dynamic_budget,
        );
        exit_cached_slots[block_id] = sorted_slots_by_rank(&resident, cache_candidates);
    }

    (desired_entry_slots, exit_cached_slots)
}

fn canonical_block_entry_cached_slots(
    program: &SsaProgram,
    desired_entry_slots: &[Vec<FrameSlot>],
    predecessors: &[Vec<usize>],
) -> Vec<Vec<FrameSlot>> {
    let mut block_entry_cached_slots = vec![Vec::new(); program.blocks.len()];
    for block in &program.blocks {
        let block_id = block.id.as_usize();
        if block.id == program.entry {
            continue;
        }
        if predecessors.get(block_id).is_some_and(|preds| preds.is_empty()) {
            continue;
        }
        block_entry_cached_slots[block_id] =
            desired_entry_slots.get(block_id).cloned().unwrap_or_default();
    }
    block_entry_cached_slots
}

fn desired_entry_slots_for_block(
    block: &crate::vm::middle::ssa_ir::ir::SsaBlock,
    cache_candidates: &BTreeMap<FrameSlot, CacheCandidate>,
    value_types: &[ValueType],
    gp_unit_bytes: u8,
    gp_dynamic_budget: u8,
    fp_dynamic_budget: u8,
) -> Vec<FrameSlot> {
    let mut remaining_uses = compute_remaining_uses(block);
    let mut live_values = BTreeMap::<SsaValue, ValueType>::new();
    let mut live = LiveCounts::default();
    for &param in &block.params {
        let remaining = remaining_uses.get(&param).copied().unwrap_or(0);
        if remaining == 0 {
            continue;
        }
        let ty = value_type_from_slice(value_types, param);
        live_add(&mut live_values, &mut live, param, ty, gp_unit_bytes);
    }

    let mut peak = live;
    let mut desired = BTreeSet::<FrameSlot>::new();
    let mut desired_counts = LiveCounts::default();
    let mut seen_cache_slots = BTreeSet::<FrameSlot>::new();

    for inst in &block.ops {
        match &inst.kind {
            SsaInstKind::LocalGetCache { slot, dst } => {
                maybe_add_entry_slot(
                    *slot,
                    cache_candidates,
                    &mut seen_cache_slots,
                    &mut desired,
                    &mut desired_counts,
                    peak,
                    gp_dynamic_budget,
                    fp_dynamic_budget,
                );
                add_result(
                    *dst,
                    &remaining_uses,
                    &mut live_values,
                    &mut live,
                    value_type_from_slice(value_types, *dst),
                    gp_unit_bytes,
                );
            }
            SsaInstKind::LocalSetCache { slot, src } => {
                maybe_add_entry_slot(
                    *slot,
                    cache_candidates,
                    &mut seen_cache_slots,
                    &mut desired,
                    &mut desired_counts,
                    peak,
                    gp_dynamic_budget,
                    fp_dynamic_budget,
                );
                consume_value(
                    *src,
                    &mut remaining_uses,
                    &mut live_values,
                    &mut live,
                    gp_unit_bytes,
                );
            }
            SsaInstKind::LocalEnsureCache { slot } => {
                maybe_add_entry_slot(
                    *slot,
                    cache_candidates,
                    &mut seen_cache_slots,
                    &mut desired,
                    &mut desired_counts,
                    peak,
                    gp_dynamic_budget,
                    fp_dynamic_budget,
                );
            }
            SsaInstKind::Value { args, results, .. } => {
                for operand in args {
                    consume_operand(
                        operand,
                        &mut remaining_uses,
                        &mut live_values,
                        &mut live,
                        gp_unit_bytes,
                    );
                }
                add_results(
                    results,
                    &remaining_uses,
                    &mut live_values,
                    &mut live,
                    value_types,
                    gp_unit_bytes,
                );
            }
            SsaInstKind::Fill { dst, .. } | SsaInstKind::LocalGetSlot { dst, .. } => {
                add_result(
                    *dst,
                    &remaining_uses,
                    &mut live_values,
                    &mut live,
                    value_type_from_slice(value_types, *dst),
                    gp_unit_bytes,
                );
            }
            SsaInstKind::Spill { src, .. } | SsaInstKind::LocalSetSlot { src, .. } => {
                consume_value(
                    *src,
                    &mut remaining_uses,
                    &mut live_values,
                    &mut live,
                    gp_unit_bytes,
                );
            }
            SsaInstKind::LocalEnsureCache { .. }
            | SsaInstKind::LocalDropCache { .. }
            | SsaInstKind::Call(_) => {}
        }
        peak.gp = peak.gp.max(live.gp);
        peak.fp = peak.fp.max(live.fp);
    }

    sorted_slots_by_rank(&desired, cache_candidates)
}

fn maybe_add_entry_slot(
    slot: FrameSlot,
    cache_candidates: &BTreeMap<FrameSlot, CacheCandidate>,
    seen_cache_slots: &mut BTreeSet<FrameSlot>,
    desired: &mut BTreeSet<FrameSlot>,
    desired_counts: &mut LiveCounts,
    peak: LiveCounts,
    gp_dynamic_budget: u8,
    fp_dynamic_budget: u8,
) {
    if !seen_cache_slots.insert(slot) {
        return;
    }
    let Some(candidate) = cache_candidates.get(&slot).copied() else {
        return;
    };
    let fits = match candidate.bank {
        CacheBank::Gp => {
            peak.gp + desired_counts.gp + candidate.cost <= gp_dynamic_budget as usize
        }
        CacheBank::Fp => {
            peak.fp + desired_counts.fp + candidate.cost <= fp_dynamic_budget as usize
        }
    };
    if !fits || !desired.insert(slot) {
        return;
    }
    match candidate.bank {
        CacheBank::Gp => desired_counts.gp += candidate.cost,
        CacheBank::Fp => desired_counts.fp += candidate.cost,
    }
}

fn sorted_slots_by_rank(
    resident: &BTreeSet<FrameSlot>,
    cache_candidates: &BTreeMap<FrameSlot, CacheCandidate>,
) -> Vec<FrameSlot> {
    sorted_slots_vec_by_rank(resident.iter().copied().collect(), cache_candidates)
}

fn sorted_slots_vec_by_rank(
    mut slots: Vec<FrameSlot>,
    cache_candidates: &BTreeMap<FrameSlot, CacheCandidate>,
) -> Vec<FrameSlot> {
    slots.sort_by_key(|slot| cache_candidates.get(slot).map(|candidate| candidate.rank).unwrap_or(usize::MAX));
    slots
}

fn compute_predecessors(
    blocks: &[crate::vm::middle::ssa_ir::ir::SsaBlock],
) -> Vec<Vec<usize>> {
    let mut predecessors = vec![Vec::new(); blocks.len()];
    for block in blocks {
        let pred = block.id.as_usize();
        let mut add_pred = |target: crate::vm::middle::ssa_ir::target::SsaTarget| {
            if let Some(entry) = predecessors.get_mut(target.as_usize()) {
                entry.push(pred);
            }
        };
        match &block.terminator {
            SsaTerminator::Goto(edge) => add_pred(edge.target),
            SsaTerminator::Branch {
                then_edge,
                else_edge,
                ..
            } => {
                add_pred(then_edge.target);
                add_pred(else_edge.target);
            }
            SsaTerminator::BrTable { entries, .. } => {
                for edge in entries {
                    add_pred(edge.target);
                }
            }
            SsaTerminator::Return { .. } | SsaTerminator::TrapUnreachable => {}
        }
    }
    predecessors
}

fn compute_wanted_exit_slots(
    blocks: &[crate::vm::middle::ssa_ir::ir::SsaBlock],
    successor_entry_slots: &[Vec<FrameSlot>],
    cache_candidates: &BTreeMap<FrameSlot, CacheCandidate>,
) -> Vec<Vec<FrameSlot>> {
    let mut wanted = vec![Vec::new(); blocks.len()];
    for block in blocks {
        let block_id = block.id.as_usize();
        let mut slots = BTreeSet::new();
        let mut add_target = |target: crate::vm::middle::ssa_ir::target::SsaTarget| {
            if let Some(entry_slots) = successor_entry_slots.get(target.as_usize()) {
                slots.extend(entry_slots.iter().copied());
            }
        };
        match &block.terminator {
            SsaTerminator::Goto(edge) => add_target(edge.target),
            SsaTerminator::Branch {
                then_edge,
                else_edge,
                ..
            } => {
                add_target(then_edge.target);
                add_target(else_edge.target);
            }
            SsaTerminator::BrTable { entries, .. } => {
                for edge in entries {
                    add_target(edge.target);
                }
            }
            SsaTerminator::Return { .. } | SsaTerminator::TrapUnreachable => {}
        }
        wanted[block_id] = sorted_slots_vec_by_rank(slots.into_iter().collect(), cache_candidates);
    }
    wanted
}

fn insert_boundary_repair_blocks(
    program: &mut SsaProgram,
    cache_candidates: &BTreeMap<FrameSlot, CacheCandidate>,
    exit_cached_slots: &[Vec<FrameSlot>],
) {
    let original_len = program.blocks.len();
    let target_params = program
        .blocks
        .iter()
        .map(|block| block.params.clone())
        .collect::<Vec<_>>();
    let target_entries = program.block_entry_cached_slots.clone();
    let mut extra_blocks = Vec::new();

    for block_index in 0..original_len {
        let pred_exit = exit_cached_slots
            .get(block_index)
            .cloned()
            .unwrap_or_default();
        let terminator = &mut program.blocks[block_index].terminator;
        match terminator {
            SsaTerminator::Goto(edge) => maybe_repair_edge(
                edge,
                &pred_exit,
                &target_entries,
                &target_params,
                cache_candidates,
                &mut extra_blocks,
                &mut program.block_entry_cached_slots,
                original_len,
            ),
            SsaTerminator::Branch {
                then_edge,
                else_edge,
                ..
            } => {
                maybe_repair_edge(
                    then_edge,
                    &pred_exit,
                    &target_entries,
                    &target_params,
                    cache_candidates,
                    &mut extra_blocks,
                    &mut program.block_entry_cached_slots,
                    original_len,
                );
                maybe_repair_edge(
                    else_edge,
                    &pred_exit,
                    &target_entries,
                    &target_params,
                    cache_candidates,
                    &mut extra_blocks,
                    &mut program.block_entry_cached_slots,
                    original_len,
                );
            }
            SsaTerminator::BrTable { entries, .. } => {
                for edge in entries {
                    maybe_repair_edge(
                        edge,
                        &pred_exit,
                        &target_entries,
                        &target_params,
                        cache_candidates,
                        &mut extra_blocks,
                        &mut program.block_entry_cached_slots,
                        original_len,
                    );
                }
            }
            SsaTerminator::Return { .. } | SsaTerminator::TrapUnreachable => {}
        }
    }

    program.blocks.extend(extra_blocks);
}

fn maybe_repair_edge(
    edge: &mut crate::vm::middle::ssa_ir::ir::SsaEdge,
    pred_exit: &[FrameSlot],
    target_entries: &[Vec<FrameSlot>],
    target_params: &[Vec<SsaValue>],
    cache_candidates: &BTreeMap<FrameSlot, CacheCandidate>,
    extra_blocks: &mut Vec<crate::vm::middle::ssa_ir::ir::SsaBlock>,
    block_entry_cached_slots: &mut Vec<Vec<FrameSlot>>,
    original_len: usize,
) {
    let target_id = edge.target.as_usize();
    let target_entry = target_entries.get(target_id).cloned().unwrap_or_default();
    if pred_exit == target_entry {
        return;
    }

    let repair_id = crate::vm::middle::ssa_ir::target::SsaTarget((original_len + extra_blocks.len()) as u32);
    let repair_params = target_params.get(target_id).cloned().unwrap_or_default();
    let mut ops = Vec::new();
    for slot in pred_exit {
        if !target_entry.contains(slot) {
            ops.push(SsaInst {
                kind: SsaInstKind::LocalDropCache { slot: *slot },
            });
        }
    }
    for slot in &target_entry {
        if !pred_exit.contains(slot) {
            ops.push(SsaInst {
                kind: SsaInstKind::LocalEnsureCache { slot: *slot },
            });
        }
    }
    let repair_edge = crate::vm::middle::ssa_ir::ir::SsaEdge {
        target: edge.target,
        bindings: repair_params
            .iter()
            .copied()
            .map(|param| crate::vm::middle::ssa_ir::ir::SsaBinding { param, value: param })
            .collect(),
    };
    extra_blocks.push(crate::vm::middle::ssa_ir::ir::SsaBlock {
        id: repair_id,
        params: repair_params,
        ops,
        terminator: SsaTerminator::Goto(repair_edge),
    });
    block_entry_cached_slots.push(sorted_slots_vec_by_rank(pred_exit.to_vec(), cache_candidates));
    edge.target = repair_id;
}

fn build_cache_candidates(program: &SsaProgram, gp_unit_bytes: u8) -> BTreeMap<FrameSlot, CacheCandidate> {
    let mut candidates = BTreeMap::new();
    for (rank, slot) in program.local_cache.gp_preferred_slots.iter().copied().enumerate() {
        let ty = program
            .local_cache
            .gp_preferred_types
            .get(rank)
            .copied()
            .unwrap_or(ValueType::I64);
        candidates.insert(
            slot,
            CacheCandidate {
                bank: CacheBank::Gp,
                cost: gp_value_budget_units(ty, gp_unit_bytes),
                rank,
            },
        );
    }

    let mut next_rank = candidates.len();
    for (rank, slot) in program.local_cache.fp_preferred_slots.iter().copied().enumerate() {
        candidates.insert(
            slot,
            CacheCandidate {
                bank: CacheBank::Fp,
                cost: 1,
                rank: next_rank + rank,
            },
        );
    }
    next_rank = candidates.len();

    for block in &program.blocks {
        for inst in &block.ops {
            let (slot, ty) = match inst.kind {
                SsaInstKind::LocalGetCache { slot, dst } => {
                    (slot, value_type_from_slice(&program.value_types, dst))
                }
                SsaInstKind::LocalSetCache { slot, src } => {
                    (slot, value_type_from_slice(&program.value_types, src))
                }
                SsaInstKind::LocalEnsureCache { slot }
                | SsaInstKind::LocalDropCache { slot } => (slot, ValueType::I32),
                _ => continue,
            };
            candidates.entry(slot).or_insert_with(|| {
                let is_fp = ty.is_float();
                let rank = next_rank;
                next_rank += 1;
                CacheCandidate {
                    bank: if is_fp { CacheBank::Fp } else { CacheBank::Gp },
                    cost: if is_fp { 1 } else { gp_value_budget_units(ty, gp_unit_bytes) },
                    rank,
                }
            });
        }
    }

    candidates
}

fn compute_remaining_uses(block: &crate::vm::middle::ssa_ir::ir::SsaBlock) -> BTreeMap<SsaValue, u32> {
    let mut uses = BTreeMap::new();
    for inst in &block.ops {
        match &inst.kind {
            SsaInstKind::Value { args, .. } => {
                for operand in args {
                    if let SsaOperand::Value(value) = operand {
                        *uses.entry(*value).or_insert(0) += 1;
                    }
                }
            }
            SsaInstKind::Spill { src, .. }
            | SsaInstKind::LocalSetSlot { src, .. }
            | SsaInstKind::LocalSetCache { src, .. } => {
                *uses.entry(*src).or_insert(0) += 1;
            }
            SsaInstKind::Fill { .. }
            | SsaInstKind::LocalGetSlot { .. }
            | SsaInstKind::LocalGetCache { .. }
            | SsaInstKind::LocalEnsureCache { .. }
            | SsaInstKind::LocalDropCache { .. }
            | SsaInstKind::Call(_) => {}
        }
    }

    match &block.terminator {
        SsaTerminator::Goto(edge) => {
            for binding in &edge.bindings {
                *uses.entry(binding.value).or_insert(0) += 1;
            }
        }
        SsaTerminator::Branch { cond, then_edge, else_edge } => {
            *uses.entry(*cond).or_insert(0) += 1;
            for binding in &then_edge.bindings {
                *uses.entry(binding.value).or_insert(0) += 1;
            }
            for binding in &else_edge.bindings {
                *uses.entry(binding.value).or_insert(0) += 1;
            }
        }
        SsaTerminator::BrTable { index, entries } => {
            *uses.entry(*index).or_insert(0) += 1;
            for edge in entries {
                for binding in &edge.bindings {
                    *uses.entry(binding.value).or_insert(0) += 1;
                }
            }
        }
        SsaTerminator::Return { .. } | SsaTerminator::TrapUnreachable => {}
    }

    uses
}

fn compute_remaining_cache_accesses(
    block: &crate::vm::middle::ssa_ir::ir::SsaBlock,
) -> BTreeMap<FrameSlot, usize> {
    let mut counts = BTreeMap::new();
    for inst in &block.ops {
        match inst.kind {
            SsaInstKind::LocalGetCache { slot, .. }
            | SsaInstKind::LocalSetCache { slot, .. }
            | SsaInstKind::LocalEnsureCache { slot } => {
                *counts.entry(slot).or_insert(0) += 1;
            }
            _ => {}
        }
    }
    counts
}

fn decrement_cache_access(counts: &mut BTreeMap<FrameSlot, usize>, slot: FrameSlot) -> usize {
    let Some(remaining) = counts.get_mut(&slot) else {
        return 0;
    };
    *remaining = remaining.saturating_sub(1);
    *remaining
}

fn ensure_cache_capacity(
    rewritten: &mut Vec<SsaInst>,
    cache_candidates: &BTreeMap<FrameSlot, CacheCandidate>,
    resident: &mut BTreeSet<FrameSlot>,
    resident_counts: &mut LiveCounts,
    slot: FrameSlot,
    live: LiveCounts,
    gp_dynamic_budget: u8,
    fp_dynamic_budget: u8,
) -> bool {
    let Some(candidate) = cache_candidates.get(&slot).copied() else {
        return false;
    };
    let was_resident = resident.contains(&slot);
    if !was_resident {
        resident_add(slot, cache_candidates, resident, resident_counts);
    }
    let fits = drop_for_pressure(
        rewritten,
        cache_candidates,
        resident,
        resident_counts,
        live,
        gp_dynamic_budget,
        fp_dynamic_budget,
        Some(slot),
    );
    if !fits || !resident.contains(&slot) {
        if !was_resident {
            resident.remove(&slot);
            resident_sub(slot, cache_candidates, resident_counts);
        }
        return false;
    }
    match candidate.bank {
        CacheBank::Gp => live.gp + resident_counts.gp <= gp_dynamic_budget as usize,
        CacheBank::Fp => live.fp + resident_counts.fp <= fp_dynamic_budget as usize,
    }
}

fn drop_for_pressure(
    rewritten: &mut Vec<SsaInst>,
    cache_candidates: &BTreeMap<FrameSlot, CacheCandidate>,
    resident: &mut BTreeSet<FrameSlot>,
    resident_counts: &mut LiveCounts,
    live: LiveCounts,
    gp_dynamic_budget: u8,
    fp_dynamic_budget: u8,
    preserve: Option<FrameSlot>,
) -> bool {
    loop {
        let need_gp = live.gp + resident_counts.gp > gp_dynamic_budget as usize;
        let need_fp = live.fp + resident_counts.fp > fp_dynamic_budget as usize;
        if !need_gp && !need_fp {
            return true;
        }
        let Some(victim) = choose_victim(cache_candidates, resident, preserve, need_gp, need_fp) else {
            return false;
        };
        emit_drop_slot(rewritten, cache_candidates, resident, resident_counts, victim);
    }
}

fn choose_victim(
    cache_candidates: &BTreeMap<FrameSlot, CacheCandidate>,
    resident: &BTreeSet<FrameSlot>,
    preserve: Option<FrameSlot>,
    need_gp: bool,
    need_fp: bool,
) -> Option<FrameSlot> {
    let mut best = None::<(FrameSlot, CacheCandidate)>;
    for &slot in resident {
        if Some(slot) == preserve {
            continue;
        }
        let Some(candidate) = cache_candidates.get(&slot).copied() else {
            continue;
        };
        if (need_gp && candidate.bank != CacheBank::Gp) || (need_fp && candidate.bank != CacheBank::Fp) {
            continue;
        }
        if best.map(|(_, current)| candidate.rank > current.rank).unwrap_or(true) {
            best = Some((slot, candidate));
        }
    }
    if let Some((slot, _)) = best {
        return Some(slot);
    }

    if !need_gp && !need_fp {
        return None;
    }

    for &slot in resident {
        if Some(slot) == preserve {
            continue;
        }
        let Some(candidate) = cache_candidates.get(&slot).copied() else {
            continue;
        };
        if best.map(|(_, current)| candidate.rank > current.rank).unwrap_or(true) {
            best = Some((slot, candidate));
        }
    }
    best.map(|(slot, _)| slot)
}

fn emit_drop_slot(
    rewritten: &mut Vec<SsaInst>,
    cache_candidates: &BTreeMap<FrameSlot, CacheCandidate>,
    resident: &mut BTreeSet<FrameSlot>,
    resident_counts: &mut LiveCounts,
    slot: FrameSlot,
) {
    if !resident.remove(&slot) {
        return;
    }
    resident_sub(slot, cache_candidates, resident_counts);
    rewritten.push(SsaInst {
        kind: SsaInstKind::LocalDropCache { slot },
    });
}

fn emit_boundary_drops(
    rewritten: &mut Vec<SsaInst>,
    cache_candidates: &BTreeMap<FrameSlot, CacheCandidate>,
    resident: &mut BTreeSet<FrameSlot>,
    resident_counts: &mut LiveCounts,
) {
    let slots = resident.iter().copied().collect::<Vec<_>>();
    for slot in slots {
        emit_drop_slot(rewritten, cache_candidates, resident, resident_counts, slot);
    }
}

fn resident_add(
    slot: FrameSlot,
    cache_candidates: &BTreeMap<FrameSlot, CacheCandidate>,
    resident: &mut BTreeSet<FrameSlot>,
    resident_counts: &mut LiveCounts,
) {
    if !resident.insert(slot) {
        return;
    }
    if let Some(candidate) = cache_candidates.get(&slot).copied() {
        match candidate.bank {
            CacheBank::Gp => resident_counts.gp += candidate.cost,
            CacheBank::Fp => resident_counts.fp += candidate.cost,
        }
    }
}

fn resident_sub(
    slot: FrameSlot,
    cache_candidates: &BTreeMap<FrameSlot, CacheCandidate>,
    resident_counts: &mut LiveCounts,
) {
    if let Some(candidate) = cache_candidates.get(&slot).copied() {
        match candidate.bank {
            CacheBank::Gp => resident_counts.gp = resident_counts.gp.saturating_sub(candidate.cost),
            CacheBank::Fp => resident_counts.fp = resident_counts.fp.saturating_sub(candidate.cost),
        }
    }
}

fn add_result_counts(live: LiveCounts, result_types: &[ValueType], gp_unit_bytes: u8) -> LiveCounts {
    let mut next = live;
    for &ty in result_types {
        if ty.is_float() {
            next.fp += 1;
        } else {
            next.gp += gp_value_budget_units(ty, gp_unit_bytes);
        }
    }
    next
}

fn add_results(
    results: &[SsaValue],
    remaining_uses: &BTreeMap<SsaValue, u32>,
    live_values: &mut BTreeMap<SsaValue, ValueType>,
    live: &mut LiveCounts,
    value_types: &[ValueType],
    gp_unit_bytes: u8,
) {
    for &result in results {
        add_result(
            result,
            remaining_uses,
            live_values,
            live,
            value_type_from_slice(value_types, result),
            gp_unit_bytes,
        );
    }
}

fn add_result(
    result: SsaValue,
    remaining_uses: &BTreeMap<SsaValue, u32>,
    live_values: &mut BTreeMap<SsaValue, ValueType>,
    live: &mut LiveCounts,
    ty: ValueType,
    gp_unit_bytes: u8,
) {
    if remaining_uses.get(&result).copied().unwrap_or(0) == 0 {
        return;
    }
    live_add(live_values, live, result, ty, gp_unit_bytes);
}

fn live_add(
    live_values: &mut BTreeMap<SsaValue, ValueType>,
    live: &mut LiveCounts,
    value: SsaValue,
    ty: ValueType,
    gp_unit_bytes: u8,
) {
    if live_values.insert(value, ty).is_some() {
        return;
    }
    if ty.is_float() {
        live.fp += 1;
    } else {
        live.gp += gp_value_budget_units(ty, gp_unit_bytes);
    }
}

fn consume_operand(
    operand: &SsaOperand,
    remaining_uses: &mut BTreeMap<SsaValue, u32>,
    live_values: &mut BTreeMap<SsaValue, ValueType>,
    live: &mut LiveCounts,
    gp_unit_bytes: u8,
) {
    if let SsaOperand::Value(value) = operand {
        consume_value(*value, remaining_uses, live_values, live, gp_unit_bytes);
    }
}

fn consume_value(
    value: SsaValue,
    remaining_uses: &mut BTreeMap<SsaValue, u32>,
    live_values: &mut BTreeMap<SsaValue, ValueType>,
    live: &mut LiveCounts,
    gp_unit_bytes: u8,
) {
    let Some(remaining) = remaining_uses.get_mut(&value) else {
        return;
    };
    *remaining = remaining.saturating_sub(1);
    if *remaining != 0 {
        return;
    }
    let Some(ty) = live_values.remove(&value) else {
        return;
    };
    if ty.is_float() {
        live.fp = live.fp.saturating_sub(1);
    } else {
        live.gp = live.gp.saturating_sub(gp_value_budget_units(ty, gp_unit_bytes));
    }
}

fn value_type_from_slice(value_types: &[ValueType], value: SsaValue) -> ValueType {
    value_types
        .get(value.0 as usize)
        .copied()
        .unwrap_or(ValueType::I32)
}

fn result_types(value_types: &[ValueType], results: &[SsaValue]) -> Vec<ValueType> {
    results
        .iter()
        .map(|&value| value_type_from_slice(value_types, value))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::middle::{
        frame::plan_frame_layout,
        ssa_ir::{
            ir::{CachedLocalInfo, SsaLocalCachePrefs},
            leaf::SsaLeafOp,
            target::SsaTarget,
        },
    };
    use crate::vm::wasm::primitive_op::PrimitiveOpKind;

    #[test]
    fn repairs_missing_entry_cache_with_explicit_ensure_block() {
        let frame = plan_frame_layout(1, 1, 4);
        let slot = frame.local_slot(0);
        let mut program = SsaProgram {
            entry: SsaTarget(0),
            local_cache: SsaLocalCachePrefs {
                gp_preferred_slots: alloc::vec![slot],
                gp_preferred_types: alloc::vec![ValueType::I32],
                fp_preferred_slots: alloc::vec![],
                fp_preferred_types: alloc::vec![],
                gp_local_info: alloc::vec![CachedLocalInfo::default()],
                fp_local_info: alloc::vec![],
            },
            blocks: alloc::vec![
                crate::vm::middle::ssa_ir::ir::SsaBlock {
                    id: SsaTarget(0),
                    params: alloc::vec![],
                    ops: alloc::vec![],
                    terminator: SsaTerminator::Goto(crate::vm::middle::ssa_ir::ir::SsaEdge {
                        target: SsaTarget(1),
                        bindings: alloc::vec![],
                    }),
                },
                crate::vm::middle::ssa_ir::ir::SsaBlock {
                    id: SsaTarget(1),
                    params: alloc::vec![],
                    ops: alloc::vec![SsaInst {
                        kind: SsaInstKind::LocalGetCache {
                            slot,
                            dst: SsaValue(0),
                        },
                    }],
                    terminator: SsaTerminator::Return { results: None },
                },
            ],
            block_entry_cached_slots: alloc::vec![alloc::vec![], alloc::vec![]],
            value_types: alloc::vec![ValueType::I32],
            value_sink_local: alloc::vec![],
        };

        plan_resources(&mut program, 8, 2, 0);

        assert_eq!(program.block_entry_cached_slots[1], alloc::vec![slot]);
        assert_eq!(program.blocks.len(), 3);

        let repair = program
            .blocks
            .iter()
            .find(|block| {
                block.id != SsaTarget(0)
                    && block.id != SsaTarget(1)
                    && block.ops.iter().any(|inst| {
                        matches!(inst.kind, SsaInstKind::LocalEnsureCache { slot: repair_slot } if repair_slot == slot)
                    })
            })
            .expect("expected repair block with LocalEnsureCache");

        assert!(matches!(
            &program.blocks[0].terminator,
            SsaTerminator::Goto(crate::vm::middle::ssa_ir::ir::SsaEdge { target, .. }) if *target == repair.id
        ));
    }

    #[test]
    fn carries_cached_locals_across_compatible_edges() {
        let frame = plan_frame_layout(1, 1, 4);
        let slot = frame.local_slot(0);
        let mut program = SsaProgram {
            entry: SsaTarget(0),
            local_cache: SsaLocalCachePrefs {
                gp_preferred_slots: alloc::vec![slot],
                gp_preferred_types: alloc::vec![ValueType::I32],
                fp_preferred_slots: alloc::vec![],
                fp_preferred_types: alloc::vec![],
                gp_local_info: alloc::vec![CachedLocalInfo::default()],
                fp_local_info: alloc::vec![],
            },
            blocks: alloc::vec![
                crate::vm::middle::ssa_ir::ir::SsaBlock {
                    id: SsaTarget(0),
                    params: alloc::vec![],
                    ops: alloc::vec![
                        SsaInst {
                            kind: SsaInstKind::Value {
                                op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 9 })
                                    .unwrap(),
                                args: alloc::vec![],
                                results: alloc::vec![SsaValue(0)],
                            },
                        },
                        SsaInst {
                            kind: SsaInstKind::LocalSetCache {
                                slot,
                                src: SsaValue(0),
                            },
                        },
                    ],
                    terminator: SsaTerminator::Goto(crate::vm::middle::ssa_ir::ir::SsaEdge {
                        target: SsaTarget(1),
                        bindings: alloc::vec![],
                    }),
                },
                crate::vm::middle::ssa_ir::ir::SsaBlock {
                    id: SsaTarget(1),
                    params: alloc::vec![],
                    ops: alloc::vec![
                        SsaInst {
                            kind: SsaInstKind::LocalGetCache {
                                slot,
                                dst: SsaValue(1),
                            },
                        },
                        SsaInst {
                            kind: SsaInstKind::LocalSetCache {
                                slot,
                                src: SsaValue(1),
                            },
                        },
                    ],
                    terminator: SsaTerminator::Return { results: None },
                },
            ],
            block_entry_cached_slots: alloc::vec![alloc::vec![], alloc::vec![]],
            value_types: alloc::vec![ValueType::I32, ValueType::I32],
            value_sink_local: alloc::vec![],
        };

        plan_resources(&mut program, 8, 2, 0);

        assert_eq!(program.blocks.len(), 2);
        assert_eq!(program.block_entry_cached_slots, alloc::vec![alloc::vec![], alloc::vec![slot]]);
        assert!(
            !program.blocks[0]
                .ops
                .iter()
                .any(|inst| matches!(inst.kind, SsaInstKind::LocalDropCache { .. })),
            "compatible edge should carry the cache instead of dropping it"
        );
    }
}
