//! Pass D — the exact-simulation walker.
//!
//! Runs at the end of `build_plan`, after the region solver fills each block's
//! tentative cached-local set. For every semantic block, in the order rewrite
//! lowers, it re-walks the op stream through the SAME shared discipline engine
//! the rewriter drives (measure driver, capacity clamp On), and mirrors the
//! rewriter's per-op cache decisions from `rewrite/function.rs`. From that it
//! derives the exact per-block cached-local entry/exit rows and the per-edge
//! boundary repair actions that the rewriter finalizes post-hoc today.
//!
//! Phase 2a stores these on the plan and the old lowering path keeps computing
//! its own; a rewrite-time debug assert proves the two agree across the whole
//! spectest corpus. The walker never emits SSA values — it evolves only the
//! typed [`Window`] and observes cache eviction through the resident set the
//! engine mutates, so the recorded cache-event stream is a faithful projection
//! of the block's lowered cache ops.

use crate::collections;

use tracked_alloc::collections::{BTreeMap, BTreeSet};

use crate::{
    error::WasmError,
    value_type::ValueType,
    vm::{
        backend::BackendConfig,
        middle::{
            cfg::{CfgBlockId, CfgTerminator, SemanticCfg},
            discipline::{
                apply_op_discipline, cached_local_get_can_source_alias, BankBudget, NoEmit, Window,
            },
            frame::{FrameLayoutPlan, FrameSlot},
            ssa_ir::ir::{EntryCacheRequirement, SsaValue},
        },
        wasm::{
            primitive_op::{self, PrimitiveOpKind},
            semantic_ir::{SemanticOpKind, SemanticProgram},
        },
    },
};

use super::{
    facts::{self, FunctionPlan, RepairActions},
    local_access::decide_local_access,
    LocalAccessDecision, LocalAccessQuery,
};

/// One cache-relevant op in a block's lowered stream, in emission order. This
/// is the subsequence `simulate_materialized_cache_exit` and
/// `entry_cache_requirement` scan; the walker records nothing else.
#[derive(Clone, Copy)]
enum CacheEvent {
    /// `LOCAL_GET_CACHE`: exit set insert; entry requirement Ensure.
    Get(FrameSlot),
    /// `LOCAL_SET_CACHE`: exit set insert; entry requirement Reserve.
    Set(FrameSlot),
    /// `LOCAL_DROP_CACHE` (eviction): exit set remove; entry requirement None.
    Drop(FrameSlot),
    /// `CALL`: exit set clear; entry requirement None.
    Call,
}

/// Per-block walk output.
struct BlockWalk {
    exact_entry: collections::Vec<FrameSlot>,
    exact_exit: collections::Vec<FrameSlot>,
    events: collections::Vec<CacheEvent>,
}

/// Fill in `exact_entry` / `exact_exit` on every block and the deduped
/// per-edge `repair_pool` indices. Runs once at the end of `build_plan`.
pub(crate) fn compute_exact_plan(
    semantic: &SemanticProgram,
    cfg: &SemanticCfg,
    frame: FrameLayoutPlan,
    config: BackendConfig,
    plan: &mut FunctionPlan,
) -> Result<(), WasmError> {
    if plan.blocks.is_empty() {
        return Ok(());
    }

    // Phase 1 — per-block walk yields exact entry/exit rows and the cache-event
    // stream (kept for the edge pass's first-use classification of targets).
    let mut walks = collections::Vec::with_capacity(plan.blocks.len());
    for (block_index, cfg_block) in cfg.blocks.iter().enumerate() {
        walks.push(walk_block(
            block_index,
            cfg_block.range.clone(),
            semantic,
            cfg,
            frame,
            config,
            plan,
        )?);
    }

    // Phase 2 — per-edge boundary repair from (pred exit, succ entry), deduped
    // by the (target, pred_exit, actions) key the repair-block dedup uses.
    let mut repair_pool: collections::Vec<RepairActions> = collections::Vec::new();
    let mut dedup: BTreeMap<(usize, collections::Vec<FrameSlot>, RepairActions), u32> =
        BTreeMap::new();
    for (block_index, cfg_block) in cfg.blocks.iter().enumerate() {
        let pred_exit = walks[block_index].exact_exit.clone();
        let mut edge_repair = collections::Vec::new();
        for target in terminator_edge_targets(&cfg_block.terminator) {
            let t = target.as_usize();
            let succ_entry = &walks[t].exact_entry;
            let target_events = &walks[t].events;
            let actions = facts::derive_edge_repair(&pred_exit, succ_entry, |slot| {
                events_entry_requirement(target_events, slot, succ_entry.contains(&slot))
            });
            if actions.is_empty() {
                edge_repair.push(None);
                continue;
            }
            let key = (t, pred_exit.clone(), actions.clone());
            let idx = *dedup.entry(key).or_insert_with(|| {
                let idx = repair_pool.len() as u32;
                repair_pool.push(actions.clone());
                idx
            });
            edge_repair.push(Some(idx));
        }
        plan.blocks[block_index].repair = edge_repair;
    }

    for (block_index, walk) in walks.into_iter().enumerate() {
        plan.blocks[block_index].exact_entry = walk.exact_entry;
        plan.blocks[block_index].exact_exit = walk.exact_exit;
    }
    plan.repair_pool = repair_pool;
    Ok(())
}

fn walk_block(
    block_index: usize,
    range: core::ops::Range<usize>,
    semantic: &SemanticProgram,
    cfg: &SemanticCfg,
    frame: FrameLayoutPlan,
    config: BackendConfig,
    plan: &FunctionPlan,
) -> Result<BlockWalk, WasmError> {
    let block_id = cfg.blocks[block_index].id;
    let entry = &plan.blocks[block_index].entry;
    let tentative = &plan.blocks[block_index].tentative_entry_cached_locals;
    let gp_unit_bytes = plan.gp_unit_bytes;
    let budget = BankBudget {
        gp_unit_bytes: plan.gp_unit_bytes,
        gp_live_budget: plan.gp_dynamic_budget,
        fp_live_budget: plan.fp_dynamic_budget,
    };
    let local_types = semantic.local_types.as_slice();

    // Seed exactly as `BlockState::from_entry`: full stack types, the live
    // suffix above the spilled prefix, and all-`None` alias tags on that suffix.
    let mut window = Window::new(
        entry.stack_height,
        entry.spill_depth,
        entry.stack_types.clone(),
        entry.live_types().to_vec(),
        collections::vec![None; entry.live_types().len()],
    );
    let mut resident: BTreeSet<FrameSlot> = tentative.iter().copied().collect();
    let mut events: collections::Vec<CacheEvent> = collections::Vec::new();
    let mut before: collections::Vec<FrameSlot> = collections::Vec::new();
    let mut noemit = NoEmit;

    let last_index = range.end - 1;
    for semantic_index in range.clone() {
        let op = &semantic.ops[semantic_index].kind;
        let is_terminator = semantic_index == last_index;

        // A body-side `end` canonicalizes the live window to its fallthrough
        // target BEFORE the structural prefix runs; that reshaped window is
        // what the following capacity clamp measures against. The terminator
        // `end` publishes via the edge instead — no window reshape here.
        if !is_terminator && matches!(op, SemanticOpKind::End) {
            canonicalize_window_for_fallthrough(
                semantic_index,
                semantic,
                cfg,
                plan,
                &mut window,
                frame,
                &mut noemit,
            );
        }

        // Structural prefix + capacity clamp. Evictions remove from `resident`
        // and fire `on_drop_cache` (a no-op here) — diff the resident set to
        // recover each drop, in eviction order (highest-numbered first).
        before.clear();
        before.extend(resident.iter().copied());
        apply_op_discipline(
            op,
            &mut window,
            &mut noemit,
            frame,
            &mut resident,
            local_types,
            budget,
        )?;
        for &slot in before.iter().rev() {
            if !resident.contains(&slot) {
                events.push(CacheEvent::Drop(slot));
            }
        }

        // The op's own stack + cache effect, mirroring `rewrite/function.rs`.
        match op {
            SemanticOpKind::Primitive(PrimitiveOpKind::Unreachable) => {}
            SemanticOpKind::Primitive(kind) => {
                primitive_effect(kind, semantic_index, semantic, &mut window)
            }
            SemanticOpKind::AllocExnRef { tag_idx } => primitive_effect(
                &PrimitiveOpKind::EhAllocExnRef { tag_idx: *tag_idx },
                semantic_index,
                semantic,
                &mut window,
            ),
            SemanticOpKind::LocalGet { idx } => {
                let ty = local_types
                    .get(*idx as usize)
                    .copied()
                    .unwrap_or(ValueType::I64);
                let slot = frame.local_slot(*idx);
                let alias = match decide_local_access(
                    plan,
                    LocalAccessQuery {
                        block: block_id,
                        slot,
                        resident_cache: &resident,
                    },
                ) {
                    LocalAccessDecision::Cache => {
                        resident.insert(slot);
                        events.push(CacheEvent::Get(slot));
                        cached_local_get_can_source_alias(ty, gp_unit_bytes).then_some(slot)
                    }
                    LocalAccessDecision::Slot => None,
                };
                window.push_core(&[ty], &[alias]);
            }
            SemanticOpKind::LocalSet { idx } => {
                window.pop_core();
                let slot = frame.local_slot(*idx);
                if let LocalAccessDecision::Cache = decide_local_access(
                    plan,
                    LocalAccessQuery {
                        block: block_id,
                        slot,
                        resident_cache: &resident,
                    },
                ) {
                    resident.insert(slot);
                    events.push(CacheEvent::Set(slot));
                }
            }
            SemanticOpKind::LocalTee { idx } => {
                window.pop_core();
                let ty = local_types
                    .get(*idx as usize)
                    .copied()
                    .unwrap_or(ValueType::I64);
                let slot = frame.local_slot(*idx);
                let alias = match decide_local_access(
                    plan,
                    LocalAccessQuery {
                        block: block_id,
                        slot,
                        resident_cache: &resident,
                    },
                ) {
                    LocalAccessDecision::Cache => {
                        resident.insert(slot);
                        // tee lowers as set-then-get; both re-cache the slot.
                        events.push(CacheEvent::Set(slot));
                        events.push(CacheEvent::Get(slot));
                        cached_local_get_can_source_alias(ty, gp_unit_bytes).then_some(slot)
                    }
                    LocalAccessDecision::Slot => None,
                };
                window.push_core(&[ty], &[alias]);
            }
            SemanticOpKind::CallDirect {
                params, results, ..
            } => call_effect(
                *params,
                *results,
                call_result_types(semantic, semantic_index),
                config,
                &mut window,
                &mut resident,
                &mut events,
            ),
            SemanticOpKind::CallIndirect {
                params, results, ..
            }
            | SemanticOpKind::CallRef {
                params, results, ..
            } => call_effect(
                params.saturating_add(1),
                *results,
                call_result_types(semantic, semantic_index),
                config,
                &mut window,
                &mut resident,
                &mut events,
            ),
            // Control / branch / return / tail-call ops: the structural prefix
            // already ran; they emit no cached-local op and their stack effect
            // is unobservable at the block boundary.
            _ => {}
        }
    }

    // The exact entry keeps only tentative residents the block requires — the
    // same filter the rewriter applies with `entry_cache_requirement`, seeded
    // by the hint-materialized set (tentative + cache decisions, cleared on
    // call, NOT reduced by eviction).
    let hint = replay_hint(tentative, &events);
    let exact_entry: collections::Vec<FrameSlot> = tentative
        .iter()
        .copied()
        .filter(|slot| events_entry_requirement(&events, *slot, hint.contains(slot)).is_some())
        .collect();
    // The exact exit replays the recorded cache ops over the FINAL (filtered)
    // entry seed — the same replay `simulate_materialized_cache_exit` runs.
    let exact_exit = replay_exit(&exact_entry, &events);

    Ok(BlockWalk {
        exact_entry,
        exact_exit,
        events,
    })
}

/// Mirror `canonicalize_live_window_for_target`: align the live window's spill
/// depth to the fallthrough target's entry when heights match.
fn canonicalize_window_for_fallthrough(
    semantic_index: usize,
    semantic: &SemanticProgram,
    cfg: &SemanticCfg,
    plan: &FunctionPlan,
    window: &mut Window,
    frame: FrameLayoutPlan,
    noemit: &mut NoEmit,
) {
    let next = semantic_index + 1;
    if next >= semantic.ops.len() {
        return;
    }
    let Some(target_block) = cfg.block_for_semantic_index(next) else {
        return;
    };
    let target = &plan.blocks[target_block.as_usize()].entry;
    if target.stack_height != window.height() {
        return;
    }
    if target.spill_depth > window.spill_depth() {
        let count = target.spill_depth - window.spill_depth();
        window.spill(noemit, frame, count);
    } else if target.spill_depth < window.spill_depth() {
        let count = window.height() - target.spill_depth;
        window.fill(noemit, frame, count);
    }
}

/// Apply a primitive's stack effect to the window, mirroring `lower_primitive`'s
/// result-type resolution.
fn primitive_effect(
    kind: &PrimitiveOpKind,
    semantic_index: usize,
    semantic: &SemanticProgram,
    window: &mut Window,
) {
    let (pop, push) = primitive_op::stack_effect(kind);
    let result_ty = if push == 0 {
        None
    } else if matches!(kind, PrimitiveOpKind::Select) {
        // `select` inherits the type of its first operand (the deepest popped).
        let live = window.live_types();
        Some(
            live.get(live.len().saturating_sub(pop))
                .copied()
                .unwrap_or(ValueType::I64),
        )
    } else if let Some(ty) = primitive_op::result_type(kind) {
        Some(ty)
    } else {
        Some(
            semantic
                .op_result_types
                .get(&semantic_index)
                .and_then(|types| types.first().copied())
                .unwrap_or(ValueType::I64),
        )
    };
    window.consume_core(pop);
    if let Some(ty) = result_ty {
        window.push_core(&[ty], &[None]);
    }
}

/// Apply a call's boundary effect, mirroring `lower_call_*`: record the CALL
/// event, run the engine's call finish, and clear the resident cache.
fn call_effect(
    consumed: u16,
    results: u16,
    result_types: &[ValueType],
    config: BackendConfig,
    window: &mut Window,
    resident: &mut BTreeSet<FrameSlot>,
    events: &mut collections::Vec<CacheEvent>,
) {
    events.push(CacheEvent::Call);
    let mut noemit = NoEmit;
    match walker_live_scalar(config, results, result_types) {
        Some(ty) => window.finish_call_scalar(&mut noemit, consumed, SsaValue::NONE, ty),
        None => window.finish_call(&mut noemit, consumed, results, result_types),
    }
    resident.clear();
}

/// Result types published by a call at `semantic_index`, or an empty slice.
fn call_result_types(semantic: &SemanticProgram, semantic_index: usize) -> &[ValueType] {
    semantic
        .op_result_types
        .get(&semantic_index)
        .map(|types| types.as_slice())
        .unwrap_or(&[])
}

/// Mirror `live_scalar_call_result` + `scalar_return_supported`.
fn walker_live_scalar(
    config: BackendConfig,
    results: u16,
    result_types: &[ValueType],
) -> Option<ValueType> {
    if !config.scalar_return_lanes || results != 1 || result_types.len() != 1 {
        return None;
    }
    let ty = result_types[0];
    (!matches!(ty, ValueType::V128)).then_some(ty)
}

/// Mirror `entry_cache_requirement_from_ops` over the recorded event stream.
fn events_first_touch(events: &[CacheEvent], slot: FrameSlot) -> Option<EntryCacheRequirement> {
    for event in events {
        match *event {
            CacheEvent::Get(s) if s == slot => return Some(EntryCacheRequirement::Ensure),
            CacheEvent::Set(s) if s == slot => return Some(EntryCacheRequirement::Reserve),
            CacheEvent::Drop(s) if s == slot => return None,
            CacheEvent::Call => return None,
            _ => {}
        }
    }
    None
}

/// Mirror `entry_cache_requirement(ops, slot, carried_through)`.
fn events_entry_requirement(
    events: &[CacheEvent],
    slot: FrameSlot,
    carried_through: bool,
) -> Option<EntryCacheRequirement> {
    events_first_touch(events, slot)
        .or_else(|| carried_through.then_some(EntryCacheRequirement::Ensure))
}

/// Mirror `simulate_materialized_cache_exit`: replay the cache ops over `seed`.
fn replay_exit(seed: &[FrameSlot], events: &[CacheEvent]) -> collections::Vec<FrameSlot> {
    let mut materialized: BTreeSet<FrameSlot> = seed.iter().copied().collect();
    for event in events {
        match *event {
            CacheEvent::Get(slot) | CacheEvent::Set(slot) => {
                materialized.insert(slot);
            }
            CacheEvent::Drop(slot) => {
                materialized.remove(&slot);
            }
            CacheEvent::Call => materialized.clear(),
        }
    }
    materialized.into_iter().collect()
}

/// The rewriter's `materialized_cache` at block end: tentative seed plus every
/// cache decision, cleared on each call, and — unlike the exit set — never
/// reduced by eviction.
fn replay_hint(seed: &[FrameSlot], events: &[CacheEvent]) -> BTreeSet<FrameSlot> {
    let mut hint: BTreeSet<FrameSlot> = seed.iter().copied().collect();
    for event in events {
        match *event {
            CacheEvent::Get(slot) | CacheEvent::Set(slot) => {
                hint.insert(slot);
            }
            CacheEvent::Call => hint.clear(),
            CacheEvent::Drop(_) => {}
        }
    }
    hint
}

/// The out-edge targets of a semantic terminator in fixed edge order
/// (Goto | BranchThen, BranchElse | BrTable(idx)).
fn terminator_edge_targets(terminator: &CfgTerminator) -> collections::Vec<CfgBlockId> {
    match terminator {
        CfgTerminator::Goto { edge, .. } => collections::vec![edge.target],
        CfgTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } => collections::vec![then_edge.target, else_edge.target],
        CfgTerminator::BrTable { edges, .. } => edges.iter().map(|edge| edge.target).collect(),
        CfgTerminator::Return { .. } | CfgTerminator::TrapUnreachable { .. } => {
            collections::Vec::new()
        }
    }
}
