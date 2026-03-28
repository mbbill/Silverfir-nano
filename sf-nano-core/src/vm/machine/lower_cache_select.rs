//! Per-block cache selection and remap cost model.
//!
//! Takes register-agnostic per-block weight vectors from `BlockLocalDemand`
//! and produces `SsaLocalCachePrefs` per block, using the architecture's
//! budget to run the knapsack/greedy selection.

use alloc::vec::Vec;

use crate::{
    value_type::ValueType,
    vm::middle::{
        frame::FrameLayoutPlan,
        local_cache::{select_locals_top_n, select_locals_with_budget, BlockLocalDemand},
        ssa_ir::ir::{CachedLocalInfo, SsaLocalCachePrefs},
        state::gp_value_budget_units,
    },
};

/// Compute `SsaLocalCachePrefs` for a single block from its weight vector.
pub(super) fn select_block_cache(
    block_weights: &[u64],
    local_types: &[ValueType],
    param_count: u16,
    frame: FrameLayoutPlan,
    gp_unit_bytes: u8,
    gp_budget: u8,
    fp_budget: u8,
    is_entry: bool,
    reads_before_write: &[bool],
) -> SsaLocalCachePrefs {
    if block_weights.is_empty() {
        return SsaLocalCachePrefs::default();
    }

    let gp = select_locals_with_budget(
        block_weights,
        gp_budget as usize,
        |idx| {
            local_types
                .get(idx)
                .map_or(true, |ty| !ty.is_float())
        },
        |idx| {
            local_types
                .get(idx)
                .copied()
                .map_or(1, |ty| gp_value_budget_units(ty, gp_unit_bytes))
        },
    );
    let fp = select_locals_top_n(block_weights, fp_budget as usize, |idx| {
        local_types
            .get(idx)
            .map_or(false, |ty| ty.is_float())
    });

    let local_info = |idx: u32| {
        if is_entry {
            CachedLocalInfo {
                is_param: (idx as u16) < param_count,
                reads_before_write: reads_before_write
                    .get(idx as usize)
                    .copied()
                    .unwrap_or(true),
            }
        } else {
            // Non-entry blocks: params were loaded at function entry.
            // reads_before_write is irrelevant (only for zero-init at entry).
            CachedLocalInfo {
                is_param: false,
                reads_before_write: false,
            }
        }
    };

    SsaLocalCachePrefs {
        gp_local_info: gp.iter().map(|idx| local_info(*idx)).collect(),
        gp_preferred_slots: gp.iter().map(|idx| frame.local_slot(*idx as u16)).collect(),
        gp_preferred_types: gp
            .iter()
            .map(|idx| {
                local_types
                    .get(*idx as usize)
                    .copied()
                    .unwrap_or(ValueType::I32)
            })
            .collect(),
        fp_local_info: fp.iter().map(|idx| local_info(*idx)).collect(),
        fp_preferred_slots: fp.iter().map(|idx| frame.local_slot(*idx as u16)).collect(),
        fp_preferred_types: fp
            .into_iter()
            .map(|idx| {
                local_types
                    .get(idx as usize)
                    .copied()
                    .unwrap_or(ValueType::F64)
            })
            .collect(),
    }
}

/// Pre-compute per-block cache preferences for all original SSA blocks.
///
/// Returns one `SsaLocalCachePrefs` per original block. Extra/synthetic blocks
/// (e.g. from call splitting) are not covered — they inherit the predecessor's
/// prefs or use `switch_cache_prefs` at continuation points.
pub(super) fn compute_all_block_prefs(
    demand: &BlockLocalDemand,
    local_types: &[ValueType],
    param_count: u16,
    frame: FrameLayoutPlan,
    gp_unit_bytes: u8,
    gp_budget: u8,
    fp_budget: u8,
    entry_block_index: usize,
) -> Vec<SsaLocalCachePrefs> {
    demand
        .block_weights
        .iter()
        .enumerate()
        .map(|(i, weights)| {
            select_block_cache(
                weights,
                local_types,
                param_count,
                frame,
                gp_unit_bytes,
                gp_budget,
                fp_budget,
                i == entry_block_index,
                &demand.reads_before_write,
            )
        })
        .collect()
}

/// Determine whether remapping from `predecessor_prefs` to `candidate_prefs`
/// is worthwhile for a block with the given weights and loop depth.
///
/// Returns `true` if the block should use its own candidate prefs.
/// Returns `false` if the block should inherit the predecessor's prefs.
pub(super) fn remap_worthwhile(
    predecessor_prefs: &SsaLocalCachePrefs,
    candidate_prefs: &SsaLocalCachePrefs,
    block_weights: &[u64],
    _block_loop_depth: u32,
) -> bool {
    // Count how many GP slots differ.
    let mut changed = 0u32;
    let mut benefit = 0u64;

    for slot in &candidate_prefs.gp_preferred_slots {
        if !predecessor_prefs.gp_preferred_slots.contains(slot) {
            changed += 1;
            // Benefit: the weight of accesses to this newly cached local.
            let local_idx = slot.0 as usize;
            benefit += block_weights.get(local_idx).copied().unwrap_or(0);
        }
    }

    for slot in &candidate_prefs.fp_preferred_slots {
        if !predecessor_prefs.fp_preferred_slots.contains(slot) {
            changed += 1;
            let local_idx = slot.0 as usize;
            benefit += block_weights.get(local_idx).copied().unwrap_or(0);
        }
    }

    if changed == 0 {
        // Same cache set (possibly in different order, but same slots) — no remap.
        return false;
    }

    // Remap cost: one store + one load per changed slot.
    let remap_cost = changed as u64 * 2;

    // The benefit is measured in weighted accesses saved. Each access that
    // would have been a frame load/store becomes a register hit.
    // We require the benefit to exceed 2x the remap cost as a conservative
    // threshold. The weights already incorporate loop depth scaling (10x per
    // nesting level), so deeply nested blocks naturally clear this bar.
    benefit > remap_cost * 2
}

/// Map `continuation_skip_locals` (local indices written-before-read) to a
/// per-cache-index skip vector for the given block's prefs.
pub(super) fn map_skip_reload(
    skip_local_indices: &[u32],
    prefs: &SsaLocalCachePrefs,
    frame: FrameLayoutPlan,
    local_count: u16,
) -> Vec<bool> {
    let total = prefs.gp_preferred_slots.len() + prefs.fp_preferred_slots.len();
    let mut skip = alloc::vec![false; total];

    for (i, slot) in prefs.gp_preferred_slots.iter().enumerate() {
        // Map frame slot back to local index.
        let local_idx = frame.slot_to_local(*slot, local_count);
        if let Some(idx) = local_idx {
            if skip_local_indices.contains(&(idx as u32)) {
                skip[i] = true;
            }
        }
    }
    let gp_len = prefs.gp_preferred_slots.len();
    for (i, slot) in prefs.fp_preferred_slots.iter().enumerate() {
        let local_idx = frame.slot_to_local(*slot, local_count);
        if let Some(idx) = local_idx {
            if skip_local_indices.contains(&(idx as u32)) {
                skip[gp_len + i] = true;
            }
        }
    }

    skip
}
