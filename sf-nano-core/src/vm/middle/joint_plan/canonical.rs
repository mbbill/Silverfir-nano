//! Canonical predecessor choice.

use alloc::vec::Vec;

use crate::vm::middle::cfg::{CfgBlockId, SemanticCfg};

/// One chosen canonical predecessor per block.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct CanonicalPredecessors {
    pub by_block: Vec<Option<CfgBlockId>>,
}

pub(crate) fn choose_canonical_predecessors(cfg: &SemanticCfg) -> CanonicalPredecessors {
    let mut by_block = Vec::with_capacity(cfg.blocks.len());
    for block in &cfg.blocks {
        let choice = if block.flags.is_entry || block.preds.is_empty() {
            None
        } else if block.flags.is_loop_header {
            block
                .preds
                .iter()
                .find(|pred| pred.is_backedge)
                .map(|pred| pred.block)
        } else if block.preds.len() == 1 {
            Some(block.preds[0].block)
        } else {
            Some(block.preds[0].block)
        };
        by_block.push(choice);
    }
    CanonicalPredecessors { by_block }
}
