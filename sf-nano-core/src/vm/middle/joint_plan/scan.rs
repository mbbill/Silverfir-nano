//! Block-local local-use ranking.
//!
//! This is intentionally block-local only. Whole-function local hotness is not
//! part of the new design.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::vm::middle::{
    frame::{FrameLayoutPlan, FrameSlot},
    slot_ssa::{SlotInstKind, SlotSsaProgram},
};

/// Ranked cached-local preference for one block.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct BlockLocalRanking {
    pub slots: Vec<FrameSlot>,
}

/// Per-block local rankings for a function.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct LocalRankings {
    pub blocks: Vec<BlockLocalRanking>,
}

pub(crate) fn scan_local_rankings(
    slot_program: &SlotSsaProgram,
    frame: FrameLayoutPlan,
) -> LocalRankings {
    let mut blocks = Vec::with_capacity(slot_program.blocks.len());
    for block in &slot_program.blocks {
        let mut counts = BTreeMap::<u16, usize>::new();
        let mut first_use = BTreeMap::<u16, usize>::new();
        for (op_index, inst) in block.body.iter().enumerate() {
            let local = match inst.kind {
                SlotInstKind::LocalGetSlot { local }
                | SlotInstKind::LocalSetSlot { local }
                | SlotInstKind::LocalTeeSlot { local } => Some(local),
                _ => None,
            };
            let Some(local) = local else {
                continue;
            };
            *counts.entry(local).or_insert(0) += 1;
            first_use.entry(local).or_insert(op_index);
        }

        let mut locals = counts.into_iter().collect::<Vec<_>>();
        locals.sort_by_key(|(local, count)| {
            (
                core::cmp::Reverse(*count),
                first_use.get(local).copied().unwrap_or(usize::MAX),
                *local,
            )
        });
        blocks.push(BlockLocalRanking {
            slots: locals
                .into_iter()
                .map(|(local, _)| frame.local_slot(local))
                .collect(),
        });
    }
    LocalRankings { blocks }
}
