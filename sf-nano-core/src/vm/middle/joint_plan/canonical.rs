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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::middle::cfg::{
        CfgBlock, CfgBlockFlags, CfgEdge, CfgPredecessor, CfgTerminator, SemanticCfg,
    };

    fn empty_cfg(blocks: Vec<CfgBlock>) -> SemanticCfg {
        SemanticCfg {
            entry: CfgBlockId(0),
            semantic_to_block: Vec::new(),
            blocks,
        }
    }

    #[test]
    fn chooses_only_predecessor_for_straight_line_block() {
        let cfg = empty_cfg(alloc::vec![
            CfgBlock {
                id: CfgBlockId(0),
                range: 0..1,
                preds: Vec::new(),
                succs: alloc::vec![CfgEdge {
                    target: CfgBlockId(1),
                    is_backedge: false,
                }],
                terminator: CfgTerminator::Goto {
                    op_index: 0,
                    edge: CfgEdge {
                        target: CfgBlockId(1),
                        is_backedge: false,
                    },
                },
                flags: CfgBlockFlags {
                    is_entry: true,
                    ..CfgBlockFlags::default()
                },
            },
            CfgBlock {
                id: CfgBlockId(1),
                range: 1..2,
                preds: alloc::vec![CfgPredecessor {
                    block: CfgBlockId(0),
                    is_backedge: false,
                }],
                succs: Vec::new(),
                terminator: CfgTerminator::Return { op_index: 1 },
                flags: CfgBlockFlags::default(),
            },
        ]);
        let canonical = choose_canonical_predecessors(&cfg);

        assert_eq!(canonical.by_block[1], Some(CfgBlockId(0)));
    }

    #[test]
    fn chooses_backedge_for_loop_header() {
        let cfg = empty_cfg(alloc::vec![
            CfgBlock {
                id: CfgBlockId(0),
                range: 0..1,
                preds: Vec::new(),
                succs: alloc::vec![CfgEdge {
                    target: CfgBlockId(1),
                    is_backedge: false,
                }],
                terminator: CfgTerminator::Goto {
                    op_index: 0,
                    edge: CfgEdge {
                        target: CfgBlockId(1),
                        is_backedge: false,
                    },
                },
                flags: CfgBlockFlags {
                    is_entry: true,
                    ..CfgBlockFlags::default()
                },
            },
            CfgBlock {
                id: CfgBlockId(1),
                range: 1..2,
                preds: alloc::vec![
                    CfgPredecessor {
                        block: CfgBlockId(0),
                        is_backedge: false,
                    },
                    CfgPredecessor {
                        block: CfgBlockId(2),
                        is_backedge: true,
                    },
                ],
                succs: alloc::vec![
                    CfgEdge {
                        target: CfgBlockId(2),
                        is_backedge: false,
                    },
                    CfgEdge {
                        target: CfgBlockId(3),
                        is_backedge: false,
                    },
                ],
                terminator: CfgTerminator::Branch {
                    op_index: 1,
                    then_edge: CfgEdge {
                        target: CfgBlockId(2),
                        is_backedge: false,
                    },
                    else_edge: CfgEdge {
                        target: CfgBlockId(3),
                        is_backedge: false,
                    },
                },
                flags: CfgBlockFlags {
                    is_loop_header: true,
                    is_merge: true,
                    ..CfgBlockFlags::default()
                },
            },
            CfgBlock {
                id: CfgBlockId(2),
                range: 2..3,
                preds: alloc::vec![CfgPredecessor {
                    block: CfgBlockId(1),
                    is_backedge: false,
                }],
                succs: alloc::vec![CfgEdge {
                    target: CfgBlockId(1),
                    is_backedge: true,
                }],
                terminator: CfgTerminator::Goto {
                    op_index: 2,
                    edge: CfgEdge {
                        target: CfgBlockId(1),
                        is_backedge: true,
                    },
                },
                flags: CfgBlockFlags::default(),
            },
            CfgBlock {
                id: CfgBlockId(3),
                range: 3..4,
                preds: alloc::vec![CfgPredecessor {
                    block: CfgBlockId(1),
                    is_backedge: false,
                }],
                succs: Vec::new(),
                terminator: CfgTerminator::Return { op_index: 3 },
                flags: CfgBlockFlags::default(),
            },
        ]);
        let canonical = choose_canonical_predecessors(&cfg);

        assert_eq!(canonical.by_block[1], Some(CfgBlockId(2)));
    }

    #[test]
    fn chooses_first_predecessor_for_ordinary_merge() {
        let cfg = empty_cfg(alloc::vec![
            CfgBlock {
                id: CfgBlockId(0),
                range: 0..1,
                preds: Vec::new(),
                succs: alloc::vec![
                    CfgEdge {
                        target: CfgBlockId(1),
                        is_backedge: false,
                    },
                    CfgEdge {
                        target: CfgBlockId(2),
                        is_backedge: false,
                    },
                ],
                terminator: CfgTerminator::Branch {
                    op_index: 0,
                    then_edge: CfgEdge {
                        target: CfgBlockId(1),
                        is_backedge: false,
                    },
                    else_edge: CfgEdge {
                        target: CfgBlockId(2),
                        is_backedge: false,
                    },
                },
                flags: CfgBlockFlags {
                    is_entry: true,
                    ..CfgBlockFlags::default()
                },
            },
            CfgBlock {
                id: CfgBlockId(1),
                range: 1..2,
                preds: alloc::vec![CfgPredecessor {
                    block: CfgBlockId(0),
                    is_backedge: false,
                }],
                succs: alloc::vec![CfgEdge {
                    target: CfgBlockId(3),
                    is_backedge: false,
                }],
                terminator: CfgTerminator::Goto {
                    op_index: 1,
                    edge: CfgEdge {
                        target: CfgBlockId(3),
                        is_backedge: false,
                    },
                },
                flags: CfgBlockFlags::default(),
            },
            CfgBlock {
                id: CfgBlockId(2),
                range: 2..3,
                preds: alloc::vec![CfgPredecessor {
                    block: CfgBlockId(0),
                    is_backedge: false,
                }],
                succs: alloc::vec![CfgEdge {
                    target: CfgBlockId(3),
                    is_backedge: false,
                }],
                terminator: CfgTerminator::Goto {
                    op_index: 2,
                    edge: CfgEdge {
                        target: CfgBlockId(3),
                        is_backedge: false,
                    },
                },
                flags: CfgBlockFlags::default(),
            },
            CfgBlock {
                id: CfgBlockId(3),
                range: 3..4,
                preds: alloc::vec![
                    CfgPredecessor {
                        block: CfgBlockId(1),
                        is_backedge: false,
                    },
                    CfgPredecessor {
                        block: CfgBlockId(2),
                        is_backedge: false,
                    },
                ],
                succs: Vec::new(),
                terminator: CfgTerminator::Return { op_index: 3 },
                flags: CfgBlockFlags {
                    is_merge: true,
                    ..CfgBlockFlags::default()
                },
            },
        ]);
        let canonical = choose_canonical_predecessors(&cfg);

        assert_eq!(canonical.by_block[3], Some(CfgBlockId(1)));
    }
}
