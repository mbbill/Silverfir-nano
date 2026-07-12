//! Post-cleanup derivation of the two machine-facing cache signals over the
//! FINAL SSA — the program the machine actually lowers, after cleanup merges,
//! optimize, and sink. Deriving these at plan time (over the pre-cleanup
//! semantic blocks) is unfaithful by construction: a `goto`-successor merge
//! folds the successor's first-touch into the predecessor's block, so a
//! per-plan-block classification keeps the pre-merge answer (e.g. a slot the
//! predecessor carried untouched stays `Ensure` even though the merged body now
//! writes it first and should be `Reserve`). Scanning the final block sees the
//! merged whole.
//!
//! Both derivations are the literal algorithms the deleted machine-side
//! dataflows ran, now living in the middle and fed byte-identical inputs (final
//! SSA ops + final CFG), so their output is faithful to what the machine
//! consumed before the middle-end v2 refactor.

use crate::collections;
use tracked_alloc::collections::{BTreeMap, BTreeSet};

use crate::vm::{
    entities::TableDispatchMode,
    middle::{
        frame::FrameSlot,
        ssa_ir::ir::{
            entry_cache_requirement, EntryCacheRequirement, SsaBlock, SsaCallOp, SsaInst, SsaOp,
            SsaProgram, SsaTerminator,
        },
        ssa_ir::target::SsaTarget,
    },
};

/// (A) Per-block, per-entry-slot requirement (`Ensure` | `Reserve`) from the
/// original `entry_cache_requirement` scan over each FINAL block's emitted ops.
/// Parallel 1:1 with `program.block_entry_cached_slots`.
pub(super) fn derive_entry_cache_requirements(
    program: &SsaProgram,
) -> collections::Vec<collections::Vec<EntryCacheRequirement>> {
    program
        .blocks
        .iter()
        .enumerate()
        .map(|(block_index, block)| {
            let entry_slots = program
                .block_entry_cached_slots
                .get(block_index)
                .map(|slots| slots.as_slice())
                .unwrap_or(&[]);
            entry_slots
                .iter()
                .map(|&slot| {
                    // An entry-resident slot is carried-through by definition, so
                    // one the block never touches needs its value materialized on
                    // entry (`Ensure`); a first-touch `Set`/`Reserve` reserves.
                    entry_cache_requirement(&block.ops, slot, true)
                        .unwrap_or(EntryCacheRequirement::Ensure)
                })
                .collect()
        })
        .collect()
}

/// (B) Whole-function preferred-preserved flag per local slot: a cached local is
/// promoted (preferred into a preserved register wherever it lives) when its
/// total local-JIT-call cross count over the final function reaches
/// `min_local_call_crosses`. This is the 2c2e010
/// `compute_local_call_cache_preferences` global-promotion dataflow — the final
/// loop there broadcasts the global promoted vector to every block, so the
/// per-slot projection here is the whole of its output.
pub(super) fn derive_preferred_preserved(
    program: &SsaProgram,
    is_local_func: &[bool],
    table_dispatch_modes: &[TableDispatchMode],
    min_local_call_crosses: u16,
) -> collections::Vec<bool> {
    let local_count = program.local_slot_types.len();

    // Cached-slot vocabulary: every slot touched by an explicit cache op — the
    // machine's `explicit_cached_locals` set, so cross indices line up.
    let mut cached_set: BTreeSet<FrameSlot> = BTreeSet::new();
    for block in &program.blocks {
        for inst in &block.ops {
            if is_cache_op(inst.op) {
                cached_set.insert(FrameSlot(inst.meta));
            }
        }
    }
    if cached_set.is_empty() {
        return collections::vec![false; local_count];
    }
    let cached_slots: collections::Vec<FrameSlot> = cached_set.into_iter().collect();
    let cached_count = cached_slots.len();
    let slot_to_cached_index: BTreeMap<FrameSlot, usize> = cached_slots
        .iter()
        .enumerate()
        .map(|(index, slot)| (*slot, index))
        .collect();
    let is_ref: collections::Vec<bool> = cached_slots
        .iter()
        .map(|slot| {
            program
                .local_slot_types
                .get(slot.0 as usize)
                .map(|ty| ty.is_ref())
                .unwrap_or(false)
        })
        .collect();

    let mut cross_counts = collections::vec![0u16; cached_count];
    let block_cache_live_in = compute_block_cache_live_in(program, &slot_to_cached_index);

    for block in &program.blocks {
        let block_index = block.id.as_usize();
        let call_needs_after = compute_call_cache_needs_after(
            program,
            block,
            &slot_to_cached_index,
            is_local_func,
            table_dispatch_modes,
            &block_cache_live_in,
        );
        let mut resident = collections::vec![false; cached_count];
        let mut has_value = collections::vec![false; cached_count];

        if let Some(entry_slots) = program.block_entry_cached_slots.get(block_index) {
            for &slot in entry_slots {
                let Some(&cached_index) = slot_to_cached_index.get(&slot) else {
                    continue;
                };
                resident[cached_index] = true;
                has_value[cached_index] = entry_cache_requirement(&block.ops, slot, true)
                    == Some(EntryCacheRequirement::Ensure);
            }
        }

        for (inst_index, inst) in block.ops.iter().enumerate() {
            match inst.op {
                SsaOp::LOCAL_GET_CACHE | SsaOp::LOCAL_ENSURE_CACHE | SsaOp::LOCAL_SET_CACHE => {
                    mark_present(&mut resident, &mut has_value, &slot_to_cached_index, inst);
                }
                SsaOp::LOCAL_RESERVE_CACHE => {
                    mark_reserved(&mut resident, &mut has_value, &slot_to_cached_index, inst);
                }
                SsaOp::LOCAL_DROP_CACHE => {
                    mark_absent(&mut resident, &mut has_value, &slot_to_cached_index, inst);
                }
                SsaOp::CALL => {
                    let local_jit_call =
                        is_local_jit_call(program, is_local_func, table_dispatch_modes, inst.meta);
                    let needs_after = call_needs_after
                        .get(inst_index)
                        .and_then(|bits| bits.as_ref());
                    for cached_index in 0..cached_count {
                        let keep = local_jit_call
                            && !is_ref[cached_index]
                            && needs_after
                                .and_then(|bits| bits.get(cached_index))
                                .copied()
                                .unwrap_or(false)
                            && resident[cached_index]
                            && has_value[cached_index];
                        if keep {
                            cross_counts[cached_index] =
                                cross_counts[cached_index].saturating_add(1);
                            // Lowering publishes the dirty carried cache before the
                            // call, so the success edge carries a clean value.
                            resident[cached_index] = true;
                            has_value[cached_index] = true;
                        } else {
                            resident[cached_index] = false;
                            has_value[cached_index] = false;
                        }
                    }
                }
                _ => {}
            }
        }
    }

    let mut preferred = collections::vec![false; local_count];
    for (cached_index, &count) in cross_counts.iter().enumerate() {
        if count >= min_local_call_crosses {
            let slot = cached_slots[cached_index].0 as usize;
            if slot < local_count {
                preferred[slot] = true;
            }
        }
    }
    preferred
}

fn is_cache_op(op: SsaOp) -> bool {
    matches!(
        op,
        SsaOp::LOCAL_GET_CACHE
            | SsaOp::LOCAL_SET_CACHE
            | SsaOp::LOCAL_ENSURE_CACHE
            | SsaOp::LOCAL_RESERVE_CACHE
            | SsaOp::LOCAL_DROP_CACHE
    )
}

/// A cached local holds a value after a get/ensure/set.
fn mark_present(
    resident: &mut [bool],
    has_value: &mut [bool],
    slot_to_cached_index: &BTreeMap<FrameSlot, usize>,
    inst: &SsaInst,
) {
    if let Some(&cached_index) = slot_to_cached_index.get(&FrameSlot(inst.meta)) {
        resident[cached_index] = true;
        has_value[cached_index] = true;
    }
}

/// A reserved cache is resident but holds no value yet.
fn mark_reserved(
    resident: &mut [bool],
    has_value: &mut [bool],
    slot_to_cached_index: &BTreeMap<FrameSlot, usize>,
    inst: &SsaInst,
) {
    if let Some(&cached_index) = slot_to_cached_index.get(&FrameSlot(inst.meta)) {
        resident[cached_index] = true;
        has_value[cached_index] = false;
    }
}

/// A dropped cache is no longer resident.
fn mark_absent(
    resident: &mut [bool],
    has_value: &mut [bool],
    slot_to_cached_index: &BTreeMap<FrameSlot, usize>,
    inst: &SsaInst,
) {
    if let Some(&cached_index) = slot_to_cached_index.get(&FrameSlot(inst.meta)) {
        resident[cached_index] = false;
        has_value[cached_index] = false;
    }
}

/// Per-call snapshot (indexed by op position): the cached locals still needed
/// after each local-JIT call in the block, from a backward sweep seeded by the
/// block's successors' cache-live-in.
fn compute_call_cache_needs_after(
    program: &SsaProgram,
    block: &SsaBlock,
    slot_to_cached_index: &BTreeMap<FrameSlot, usize>,
    is_local_func: &[bool],
    table_dispatch_modes: &[TableDispatchMode],
    block_cache_live_in: &[collections::Vec<bool>],
) -> collections::Vec<Option<collections::Vec<bool>>> {
    let mut call_needs_after = collections::vec![None; block.ops.len()];
    let mut needed_after = successor_cache_needs(
        &block.terminator,
        block_cache_live_in,
        slot_to_cached_index.len(),
    );
    for (inst_index, inst) in block.ops.iter().enumerate().rev() {
        match inst.op {
            SsaOp::LOCAL_GET_CACHE | SsaOp::LOCAL_ENSURE_CACHE => {
                mark_cached_slot(
                    &mut needed_after,
                    slot_to_cached_index,
                    FrameSlot(inst.meta),
                    true,
                );
            }
            SsaOp::LOCAL_SET_CACHE | SsaOp::LOCAL_RESERVE_CACHE | SsaOp::LOCAL_DROP_CACHE => {
                mark_cached_slot(
                    &mut needed_after,
                    slot_to_cached_index,
                    FrameSlot(inst.meta),
                    false,
                );
            }
            SsaOp::CALL => {
                if is_local_jit_call(program, is_local_func, table_dispatch_modes, inst.meta) {
                    call_needs_after[inst_index] = Some(needed_after.clone());
                } else {
                    needed_after.fill(false);
                }
            }
            _ => {}
        }
    }
    call_needs_after
}

/// Inter-block cache-liveness fixpoint: the cached locals live-in to each block,
/// over the final CFG.
fn compute_block_cache_live_in(
    program: &SsaProgram,
    slot_to_cached_index: &BTreeMap<FrameSlot, usize>,
) -> collections::Vec<collections::Vec<bool>> {
    let slot_count = slot_to_cached_index.len();
    let mut live_in = collections::vec![collections::vec![false; slot_count]; program.blocks.len()];
    if slot_count == 0 {
        return live_in;
    }

    loop {
        let mut changed = false;
        for block in program.blocks.iter().rev() {
            let mut needed = successor_cache_needs(&block.terminator, &live_in, slot_count);
            apply_cache_need_transfer(block.ops.iter().rev(), slot_to_cached_index, &mut needed);
            let block_index = block.id.as_usize();
            if live_in
                .get(block_index)
                .map(|old| old.as_slice() != needed.as_slice())
                .unwrap_or(false)
            {
                live_in[block_index] = needed;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    live_in
}

fn successor_cache_needs(
    terminator: &SsaTerminator,
    block_cache_live_in: &[collections::Vec<bool>],
    slot_count: usize,
) -> collections::Vec<bool> {
    let mut needed = collections::vec![false; slot_count];
    for target in block_successors(terminator) {
        let Some(bits) = block_cache_live_in.get(target.as_usize()) else {
            continue;
        };
        for (dst, src) in needed.iter_mut().zip(bits.iter().copied()) {
            *dst |= src;
        }
    }
    needed
}

fn apply_cache_need_transfer<'a>(
    ops: impl Iterator<Item = &'a SsaInst>,
    slot_to_cached_index: &BTreeMap<FrameSlot, usize>,
    needed: &mut [bool],
) {
    for inst in ops {
        match inst.op {
            SsaOp::LOCAL_GET_CACHE | SsaOp::LOCAL_ENSURE_CACHE => {
                mark_cached_slot(needed, slot_to_cached_index, FrameSlot(inst.meta), true);
            }
            SsaOp::LOCAL_SET_CACHE | SsaOp::LOCAL_RESERVE_CACHE | SsaOp::LOCAL_DROP_CACHE => {
                mark_cached_slot(needed, slot_to_cached_index, FrameSlot(inst.meta), false);
            }
            _ => {}
        }
    }
}

fn mark_cached_slot(
    bits: &mut [bool],
    slot_to_cached_index: &BTreeMap<FrameSlot, usize>,
    slot: FrameSlot,
    value: bool,
) {
    if let Some(&cached_index) = slot_to_cached_index.get(&slot) {
        if let Some(bit) = bits.get_mut(cached_index) {
            *bit = value;
        }
    }
}

/// A call that continues in JIT and can thread a preserved cache across it: a
/// direct call to a locally-compiled function, or a fixed-local-only indirect.
fn is_local_jit_call(
    program: &SsaProgram,
    is_local_func: &[bool],
    table_dispatch_modes: &[TableDispatchMode],
    call_index: u16,
) -> bool {
    match program.call_ops.get(call_index as usize) {
        Some(SsaCallOp::CallDirect { callee, .. }) => is_local_func
            .get(*callee as usize)
            .copied()
            .unwrap_or(false),
        Some(SsaCallOp::CallIndirect { table_idx, .. }) => matches!(
            table_dispatch_modes.get(*table_idx as usize).copied(),
            Some(TableDispatchMode::FixedLocalOnly { .. })
        ),
        Some(SsaCallOp::CallRef { .. }) | None => false,
    }
}

fn block_successors(terminator: &SsaTerminator) -> collections::Vec<SsaTarget> {
    match terminator {
        SsaTerminator::Goto(edge) => collections::vec![edge.target],
        SsaTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } => collections::vec![then_edge.target, else_edge.target],
        SsaTerminator::BrTable { entries, .. } => {
            entries.iter().map(|entry| entry.target).collect()
        }
        SsaTerminator::Return { .. }
        | SsaTerminator::ReturnScalar { .. }
        | SsaTerminator::TailCallDirect { .. }
        | SsaTerminator::TailCallIndirect { .. }
        | SsaTerminator::TailCallRef { .. }
        | SsaTerminator::TrapUnreachable
        | SsaTerminator::EhThrow { .. }
        | SsaTerminator::EhThrowRef { .. } => collections::Vec::new(),
    }
}
