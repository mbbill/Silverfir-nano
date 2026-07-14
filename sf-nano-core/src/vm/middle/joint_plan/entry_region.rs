//! Block-local cached-local access summaries.
//!
//! The region solver only needs to know which locals are touched in each CFG
//! block and how often. Older stack/transient region analyses were removed when
//! the production planner stopped reading them.

use crate::collections;

use crate::vm::{
    middle::{
        cell::CellId,
        cfg::SemanticCfg,
        joint_plan::facts::{BlockCellSummary, CellScore},
    },
    wasm::semantic_ir::{SemanticOpKind, SemanticProgram},
};

pub(crate) fn analyze_block_local_summaries(
    semantic: &SemanticProgram,
    cfg: &SemanticCfg,
) -> collections::Vec<BlockCellSummary> {
    let mut local_counts = collections::vec![0u16; semantic.local_count as usize];
    let mut touched_locals = collections::Vec::<u16>::new();
    let mut summaries = collections::Vec::with_capacity(cfg.blocks.len());

    for block in &cfg.blocks {
        touched_locals.clear();

        for semantic_index in block.range.clone() {
            let Some(local_idx) = local_access_index(&semantic.ops[semantic_index].kind) else {
                continue;
            };
            let local_index = local_idx as usize;
            if local_counts[local_index] == 0 {
                touched_locals.push(local_idx);
            }
            local_counts[local_index] = local_counts[local_index].saturating_add(1);
        }

        let mut slot_scores = collections::Vec::with_capacity(touched_locals.len());
        for &local_idx in &touched_locals {
            let local_index = local_idx as usize;
            let access_count = local_counts[local_index];
            local_counts[local_index] = 0;
            slot_scores.push(CellScore {
                slot: CellId(local_idx),
                access_count,
            });
        }

        summaries.push(BlockCellSummary { slot_scores });
    }

    summaries
}

#[inline]
fn local_access_index(kind: &SemanticOpKind) -> Option<u16> {
    match *kind {
        SemanticOpKind::LocalGet { idx }
        | SemanticOpKind::LocalSet { idx }
        | SemanticOpKind::LocalTee { idx } => Some(idx),
        _ => None,
    }
}
