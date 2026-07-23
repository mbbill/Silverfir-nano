//! Post-cleanup derivation of the machine-facing entry-requirement rows over the
//! FINAL SSA — the program the machine actually lowers, after cleanup merges,
//! optimize, and sink. Deriving these at plan time (over the pre-cleanup
//! semantic blocks) is unfaithful by construction: a `goto`-successor merge
//! folds the successor's first-touch into the predecessor's block, so a
//! per-plan-block classification keeps the pre-merge answer (e.g. a slot the
//! predecessor carried untouched stays `Ensure` even though the merged body now
//! writes it first and should be `Reserve`). Scanning the final block sees the
//! merged whole.
//!
//! The derivation is the literal algorithm the deleted machine-side dataflow
//! ran, now living in the middle and fed byte-identical inputs (final SSA ops +
//! final CFG), so its output is faithful to what the machine consumed before
//! the middle-end v2 refactor. (The preserved-class bit that used to be
//! derived here is now the residency solver's own nomination, published from
//! the plan — see `joint_plan::region_solver::nominate_preserved`.)

use crate::collections;

use crate::vm::middle::ssa_ir::ir::{entry_cache_requirements, EntryCacheRequirement, SsaProgram};

/// (A) Per-block, per-entry-slot requirement (`Ensure` | `Reserve`) from the
/// original `entry_cache_requirement` scan over each FINAL block's emitted ops.
/// Parallel 1:1 with `program.block_entry_cached_cells`.
pub(super) fn derive_entry_cache_requirements(
    program: &SsaProgram,
) -> collections::Vec<collections::Vec<EntryCacheRequirement>> {
    program
        .blocks
        .iter()
        .enumerate()
        .map(|(block_index, block)| {
            let entry_slots = program
                .block_entry_cached_cells
                .get(block_index)
                .map(|slots| slots.as_slice())
                .unwrap_or(&[]);
            // An entry-resident slot is carried-through by definition, so one
            // the block never touches needs its value materialized on entry
            // (`Ensure`); a first-touch `Set`/`Reserve` reserves.
            entry_cache_requirements(&block.ops, entry_slots, entry_slots)
                .into_iter()
                .map(|requirement| requirement.unwrap_or(EntryCacheRequirement::Ensure))
                .collect()
        })
        .collect()
}
