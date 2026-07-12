//! Pass D — the exact-simulation walker.
//!
//! Runs at the end of `build_plan`, after the region solver fills each block's
//! planned-resident set. For every semantic block, in the order rewrite
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
    facts::{self, FunctionPlan, RepairActions, RepairActionsSpans, RowSpan, NO_REPAIR},
    local_access::decide_local_access,
    LocalAccessDecision, LocalAccessQuery,
};

/// One cache-relevant op in a block's lowered stream, in emission order. This
/// is the subsequence the exit-set replay and `entry_cache_requirement` scan;
/// the walker records nothing else.
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

/// Per-block walk output. Holds only the exact rows plus a compact per-entry
/// requirement for the edge pass — NOT the full event stream, so the walker's
/// retained memory stays bounded by the exact rows, not the block's op count.
struct BlockWalk {
    exact_entry: collections::Vec<FrameSlot>,
    exact_exit: collections::Vec<FrameSlot>,
    /// Parallel to `exact_entry`: each entry slot's first-use requirement, the
    /// value the edge pass would classify it as (carried_through is always true
    /// for an entry slot, so a slot never touched defaults to `Ensure`).
    entry_requirement: collections::Vec<EntryCacheRequirement>,
}

/// Build the plan's flat cache-row arenas and per-block spans from a per-block
/// walk. Runs once at the end of `build_plan`.
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
    let n = plan.blocks.len();

    // Phase 1 — walk each block into the flat row arena. Its exact entry/exit
    // slots accumulate in `row_arena`; the block records only an (offset, len)
    // span. The per-entry requirement (the edge pass's first-use classification)
    // lands in a transient parallel arena feeding the repair derivation below;
    // both are dropped after phase 2. Scratch buffers are reused across blocks,
    // not reallocated per block. The machine-facing requirement + preferred rows
    // are derived later over the FINAL SSA (see `middle::final_signals`), not
    // here — a pre-cleanup classification cannot see block merges.
    let mut row_arena: collections::Vec<FrameSlot> = collections::Vec::new();
    let mut requirement_arena: collections::Vec<EntryCacheRequirement> = collections::Vec::new();
    let mut entry_spans: collections::Vec<RowSpan> = collections::Vec::with_capacity(n);
    let mut exit_spans: collections::Vec<RowSpan> = collections::Vec::with_capacity(n);
    let mut req_spans: collections::Vec<RowSpan> = collections::Vec::with_capacity(n);
    let mut scratch = WalkScratch::default();
    for block_index in 0..n {
        let walk = walk_block(
            block_index,
            semantic,
            cfg,
            frame,
            config,
            plan,
            &mut scratch,
        )?;
        entry_spans.push(append_row(&mut row_arena, &walk.exact_entry));
        exit_spans.push(append_row(&mut row_arena, &walk.exact_exit));
        req_spans.push(append_row(&mut requirement_arena, &walk.entry_requirement));
    }

    // Phase 2 — per-edge boundary repair from (pred exit, succ entry). Each
    // non-empty action list is packed into the flat `repair_slot_arena` (its
    // drop/ensure/reserve groups) and the pool holds POD spans; the flat
    // per-edge index arena maps each edge to a pool entry (`NO_REPAIR` = none).
    // The pool is content-deduped: two edges whose derived actions are byte-
    // identical share one entry. Since the only consumer reads `pool[index]`
    // for its actions, sharing is invisible to it — every edge maps to the same
    // action content regardless. Dedup uses a content hash with arena-slice
    // verification (collision-safe), so no per-edge key is allocated.
    let mut repair_pool: collections::Vec<RepairActionsSpans> = collections::Vec::new();
    let mut repair_slot_arena: collections::Vec<FrameSlot> = collections::Vec::new();
    let mut repair_index_arena: collections::Vec<u32> = collections::Vec::new();
    let mut repair_spans: collections::Vec<RowSpan> = collections::Vec::with_capacity(n);
    let mut dedup: BTreeMap<u64, collections::Vec<u32>> = BTreeMap::new();
    for (block_index, cfg_block) in cfg.blocks.iter().enumerate() {
        let pred_exit = &row_arena[exit_spans[block_index].range()];
        let repair_start = repair_index_arena.len() as u32;
        for target in terminator_edge_targets(&cfg_block.terminator) {
            let t = target.as_usize();
            let succ_entry = &row_arena[entry_spans[t].range()];
            let succ_requirement = &requirement_arena[req_spans[t].range()];
            // The classifier is only ever asked about slots in `succ_entry`
            // (derive_edge_repair iterates it), so the position lookup always
            // resolves to the precomputed requirement.
            let actions = facts::derive_edge_repair(pred_exit, succ_entry, |slot| {
                succ_entry
                    .iter()
                    .position(|s| *s == slot)
                    .map(|idx| succ_requirement[idx])
            });
            if actions.is_empty() {
                repair_index_arena.push(NO_REPAIR);
                continue;
            }
            let bucket = dedup
                .entry(repair_content_hash(&actions))
                .or_insert_with(collections::Vec::new);
            let mut resolved = None;
            for &cand in bucket.iter() {
                let (d, e, r) = repair_slices(&repair_slot_arena, repair_pool[cand as usize]);
                if d == actions.drop_cached_locals.as_slice()
                    && e == actions.ensure_cached_locals.as_slice()
                    && r == actions.reserve_cached_locals.as_slice()
                {
                    resolved = Some(cand);
                    break;
                }
            }
            let idx = match resolved {
                Some(idx) => idx,
                None => {
                    let idx = repair_pool.len() as u32;
                    let spans = pack_repair(&mut repair_slot_arena, &actions);
                    repair_pool.push(spans);
                    bucket.push(idx);
                    idx
                }
            };
            repair_index_arena.push(idx);
        }
        repair_spans.push(RowSpan {
            offset: repair_start,
            len: repair_index_arena.len() as u32 - repair_start,
        });
    }

    for block_index in 0..n {
        plan.blocks[block_index].exact_entry = entry_spans[block_index];
        plan.blocks[block_index].exact_exit = exit_spans[block_index];
        plan.blocks[block_index].repair = repair_spans[block_index];
    }
    plan.row_arena = row_arena;
    plan.repair_index_arena = repair_index_arena;
    plan.repair_pool = repair_pool;
    plan.repair_slot_arena = repair_slot_arena;
    // `requirement_arena`/`req_spans` were consumed by the repair derivation
    // above and drop here — the machine-facing requirement rows are derived from
    // the FINAL SSA in `middle::final_signals`, not exported from the plan.
    Ok(())
}

/// Append `row` to a flat arena, returning its span.
fn append_row<T: Copy>(arena: &mut collections::Vec<T>, row: &[T]) -> RowSpan {
    let offset = arena.len() as u32;
    arena.extend_from_slice(row);
    RowSpan {
        offset,
        len: row.len() as u32,
    }
}

/// Pack an owned repair action list into the flat slot arena, returning the
/// per-group spans.
fn pack_repair(
    arena: &mut collections::Vec<FrameSlot>,
    actions: &RepairActions,
) -> RepairActionsSpans {
    RepairActionsSpans {
        drop: append_row(arena, &actions.drop_cached_locals),
        ensure: append_row(arena, &actions.ensure_cached_locals),
        reserve: append_row(arena, &actions.reserve_cached_locals),
    }
}

/// The drop/ensure/reserve slot groups of a packed repair entry, as slices.
fn repair_slices(
    arena: &[FrameSlot],
    spans: RepairActionsSpans,
) -> (&[FrameSlot], &[FrameSlot], &[FrameSlot]) {
    (
        &arena[spans.drop.range()],
        &arena[spans.ensure.range()],
        &arena[spans.reserve.range()],
    )
}

/// FNV-1a content hash of a repair action list (each group length-prefixed).
/// Dedup verifies candidates against the arena, so hash collisions are safe —
/// this only buckets candidates for the equality check.
fn repair_content_hash(actions: &RepairActions) -> u64 {
    #[inline]
    fn mix(h: &mut u64, x: u32) {
        *h ^= x as u64;
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for group in [
        actions.drop_cached_locals.as_slice(),
        actions.ensure_cached_locals.as_slice(),
        actions.reserve_cached_locals.as_slice(),
    ] {
        mix(&mut h, group.len() as u32);
        for slot in group {
            mix(&mut h, slot.0 as u32);
        }
    }
    h
}

/// Reused per-block scratch: the cache-event stream and the eviction-diff
/// snapshot, cleared at the start of each block rather than reallocated.
#[derive(Default)]
struct WalkScratch {
    events: collections::Vec<CacheEvent>,
    before: collections::Vec<FrameSlot>,
}

fn walk_block(
    block_index: usize,
    semantic: &SemanticProgram,
    cfg: &SemanticCfg,
    frame: FrameLayoutPlan,
    config: BackendConfig,
    plan: &FunctionPlan,
    scratch: &mut WalkScratch,
) -> Result<BlockWalk, WasmError> {
    let block_id = cfg.blocks[block_index].id;
    let range = cfg.blocks[block_index].range.clone();
    let entry = &plan.blocks[block_index].entry;
    // The step-1 walk seeds from the block's planned residents (the admission
    // set) — the same seed the rewriter used before pass D became authoritative
    // — so the filter derivation below reproduces the entry row the old path
    // computed. Lowering itself now seeds from `exact_entry`, but the walker
    // must stay planned-seeded here to feed that filter.
    let planned = plan.planned_residents(block_index);
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
        entry.live_types().to_vec().into(),
        collections::vec![None; entry.live_types().len()],
    );
    let mut resident: BTreeSet<FrameSlot> = planned.iter().copied().collect();
    let WalkScratch { events, before } = scratch;
    events.clear();
    before.clear();
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
            } => {
                events.push(CacheEvent::Call);
                call_effect(
                    *params,
                    *results,
                    call_result_types(semantic, semantic_index),
                    config,
                    &mut window,
                    &mut resident,
                );
            }
            SemanticOpKind::CallIndirect {
                params, results, ..
            }
            | SemanticOpKind::CallRef {
                params, results, ..
            } => {
                events.push(CacheEvent::Call);
                call_effect(
                    params.saturating_add(1),
                    *results,
                    call_result_types(semantic, semantic_index),
                    config,
                    &mut window,
                    &mut resident,
                );
            }
            // Control / branch / return / tail-call ops: the structural prefix
            // already ran; they emit no cached-local op and their stack effect
            // is unobservable at the block boundary.
            _ => {}
        }
    }

    // The exact entry keeps only planned residents the block requires — the
    // same filter the old rewriter applied with `entry_cache_requirement`,
    // seeded by the hint-materialized set (planned residents + cache decisions,
    // cleared on call, NOT reduced by eviction).
    let hint = replay_hint(planned, events);
    let exact_entry: collections::Vec<FrameSlot> = planned
        .iter()
        .copied()
        .filter(|slot| events_entry_requirement(events, *slot, hint.contains(slot)).is_some())
        .collect();
    // The exact exit replays the recorded cache ops over the FINAL (filtered)
    // entry seed — the exit set the old post-hoc simulate pass computed, now the
    // authoritative published row checked against lowered reality.
    let exact_exit = replay_exit(&exact_entry, events);

    // Compact the block's classification for the edge pass: each entry slot's
    // first-use requirement (Ensure when never touched by a cache op, since an
    // entry slot is always carried_through). The event stream is dropped here.
    let entry_requirement: collections::Vec<EntryCacheRequirement> = exact_entry
        .iter()
        .map(|slot| events_first_touch(events, *slot).unwrap_or(EntryCacheRequirement::Ensure))
        .collect();

    Ok(BlockWalk {
        exact_entry,
        exact_exit,
        entry_requirement,
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

/// Apply a call's boundary effect, mirroring `lower_call_*`: run the engine's
/// call finish and clear the resident cache. (The CALL event is recorded by the
/// caller so the too-many-arguments budget stays clear.)
fn call_effect(
    consumed: u16,
    results: u16,
    result_types: &[ValueType],
    config: BackendConfig,
    window: &mut Window,
    resident: &mut BTreeSet<FrameSlot>,
) {
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

/// Replay the cache ops over `seed` to the block's true exit cache set (the
/// drop-subtracted materialized state at block end).
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

/// The old rewriter's `materialized_cache` at block end: planned-resident seed
/// plus every cache decision, cleared on each call, and — unlike the exit set —
/// never reduced by eviction.
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
