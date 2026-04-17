//! Explicit CFG formation from semantic Wasm IR.

use crate::collections;

use crate::vm::wasm::{
    common::SemanticTarget,
    primitive_op::PrimitiveOpKind,
    semantic_ir::{SemanticOp, SemanticOpKind, SemanticProgram},
};

/// Explicit CFG block id used before final SSA block emission.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct CfgBlockId(pub u32);

impl CfgBlockId {
    #[inline]
    pub(crate) const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

/// One CFG edge between explicit blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CfgEdge {
    pub target: CfgBlockId,
    pub is_backedge: bool,
}

/// A predecessor edge recorded on a block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CfgPredecessor {
    pub block: CfgBlockId,
    pub is_backedge: bool,
}

/// A block terminator in semantic CFG space.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum CfgTerminator {
    Goto {
        op_index: usize,
        edge: CfgEdge,
    },
    Branch {
        op_index: usize,
        then_edge: CfgEdge,
        else_edge: CfgEdge,
    },
    BrTable {
        op_index: usize,
        edges: collections::Vec<CfgEdge>,
    },
    Return {
        op_index: usize,
    },
    TrapUnreachable {
        op_index: usize,
    },
}

/// Simple flags derived from CFG structure.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CfgBlockFlags {
    pub is_entry: bool,
    pub is_merge: bool,
    pub is_loop_header: bool,
}

/// One explicit semantic CFG block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CfgBlock {
    pub id: CfgBlockId,
    pub range: core::ops::Range<usize>,
    pub preds: collections::Vec<CfgPredecessor>,
    pub succs: collections::Vec<CfgEdge>,
    pub terminator: CfgTerminator,
    pub flags: CfgBlockFlags,
}

/// Explicit semantic CFG used by the new middle-end pipeline.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SemanticCfg {
    pub entry: CfgBlockId,
    pub blocks: collections::Vec<CfgBlock>,
    pub semantic_to_block: collections::Vec<CfgBlockId>,
}

pub(crate) fn build_semantic_cfg(semantic: &SemanticProgram) -> SemanticCfg {
    if semantic.ops.is_empty() {
        return SemanticCfg::default();
    }

    let ranges = retain_reachable_blocks(semantic, build_block_ranges(semantic));
    let semantic_to_block = build_semantic_to_block_map(semantic.ops.len(), &ranges);
    let mut blocks = collections::Vec::with_capacity(ranges.len());

    for (block_index, range) in ranges.iter().cloned().enumerate() {
        let block_id = CfgBlockId(block_index as u32);
        let last_op_index = range.end - 1;
        let (terminator, succs) = build_terminator(
            block_id,
            last_op_index,
            &semantic.ops[last_op_index],
            semantic,
            &semantic_to_block,
        );
        blocks.push(CfgBlock {
            id: block_id,
            range,
            preds: collections::Vec::new(),
            succs,
            terminator,
            flags: CfgBlockFlags {
                is_entry: block_index == 0,
                ..CfgBlockFlags::default()
            },
        });
    }

    populate_predecessors(&mut blocks);
    populate_flags(&mut blocks);

    SemanticCfg {
        entry: CfgBlockId(0),
        blocks,
        semantic_to_block,
    }
}

fn build_block_ranges(semantic: &SemanticProgram) -> collections::Vec<core::ops::Range<usize>> {
    let len = semantic.ops.len();
    let mut leaders = collections::vec![false; len + 1];
    leaders[0] = true;
    leaders[len] = true;

    for (index, op) in semantic.ops.iter().enumerate() {
        if !is_plain_fallthrough(index, semantic.ops.len(), op) {
            for_each_semantic_successor(index, semantic.ops.len(), op, |target| {
                let target_index = target.index().as_usize();
                if target_index < len {
                    leaders[target_index] = true;
                }
            });
        }

        if splits_after(&op.kind) && index + 1 < len {
            leaders[index + 1] = true;
        }
    }

    let starts = leaders
        .iter()
        .enumerate()
        .filter_map(|(index, is_leader)| is_leader.then_some(index))
        .collect::<collections::Vec<_>>();

    starts
        .windows(2)
        .filter_map(|window| {
            let start = window[0];
            let end = window[1];
            (start < end).then_some(start..end)
        })
        .collect()
}

fn retain_reachable_blocks(
    semantic: &SemanticProgram,
    block_ranges: collections::Vec<core::ops::Range<usize>>,
) -> collections::Vec<core::ops::Range<usize>> {
    if block_ranges.is_empty() {
        return block_ranges;
    }

    let semantic_to_block = build_semantic_to_block_map(semantic.ops.len(), &block_ranges);
    let mut visited = collections::vec![false; block_ranges.len()];
    let mut pending = collections::vec![0usize];

    while let Some(block_index) = pending.pop() {
        if visited.get(block_index).copied().unwrap_or(false) {
            continue;
        }
        visited[block_index] = true;

        let Some(last_semantic) = block_ranges[block_index].end.checked_sub(1) else {
            continue;
        };
        let op = &semantic.ops[last_semantic];

        for_each_semantic_successor(last_semantic, semantic.ops.len(), op, |target| {
            let target_block = semantic_to_block[target.index().as_usize()].as_usize();
            if target_block < visited.len() && !visited[target_block] {
                pending.push(target_block);
            }
        });
    }

    block_ranges
        .into_iter()
        .enumerate()
        .filter_map(|(index, range)| visited[index].then_some(range))
        .collect()
}

fn build_semantic_to_block_map(
    semantic_len: usize,
    block_ranges: &[core::ops::Range<usize>],
) -> collections::Vec<CfgBlockId> {
    let mut map = collections::vec![CfgBlockId::default(); semantic_len];
    for (block_index, range) in block_ranges.iter().enumerate() {
        for semantic_index in range.clone() {
            map[semantic_index] = CfgBlockId(block_index as u32);
        }
    }
    map
}

fn build_terminator(
    source: CfgBlockId,
    op_index: usize,
    op: &SemanticOp,
    semantic: &SemanticProgram,
    semantic_to_block: &[CfgBlockId],
) -> (CfgTerminator, collections::Vec<CfgEdge>) {
    let len = semantic.ops.len();
    let fallthrough = || {
        if op_index + 1 < len {
            Some(map_edge(
                source,
                SemanticTarget::new(op_index + 1),
                semantic_to_block,
            ))
        } else {
            None
        }
    };

    match &op.kind {
        SemanticOpKind::Primitive(PrimitiveOpKind::Unreachable) => (
            CfgTerminator::TrapUnreachable { op_index },
            collections::Vec::new(),
        ),
        SemanticOpKind::ReturnVoid
        | SemanticOpKind::ReturnOne
        | SemanticOpKind::Return { .. }
        | SemanticOpKind::ReturnCallDirect { .. }
        | SemanticOpKind::ReturnCallIndirect { .. }
        | SemanticOpKind::ReturnCallRef { .. } => {
            (CfgTerminator::Return { op_index }, collections::Vec::new())
        }
        SemanticOpKind::Br { target, .. } => {
            let edge = map_edge(source, *target, semantic_to_block);
            (
                CfgTerminator::Goto { op_index, edge },
                collections::vec![edge],
            )
        }
        SemanticOpKind::BrIf { target, .. }
        | SemanticOpKind::BrOnNull { target, .. }
        | SemanticOpKind::BrOnCast { target, .. } => {
            let then_edge = map_edge(source, *target, semantic_to_block);
            let else_edge = fallthrough().unwrap_or(then_edge);
            (
                CfgTerminator::Branch {
                    op_index,
                    then_edge,
                    else_edge,
                },
                dedup_edges(collections::vec![then_edge, else_edge]),
            )
        }
        SemanticOpKind::BrOnNonNull { target, .. }
        | SemanticOpKind::BrOnCastFail { target, .. } => {
            let then_edge =
                fallthrough().unwrap_or_else(|| map_edge(source, *target, semantic_to_block));
            let else_edge = map_edge(source, *target, semantic_to_block);
            (
                CfgTerminator::Branch {
                    op_index,
                    then_edge,
                    else_edge,
                },
                dedup_edges(collections::vec![then_edge, else_edge]),
            )
        }
        SemanticOpKind::BrTable { entries } => {
            let edges = entries
                .iter()
                .map(|entry| map_edge(source, entry.target, semantic_to_block))
                .collect::<collections::Vec<_>>();
            (
                CfgTerminator::BrTable {
                    op_index,
                    edges: edges.clone(),
                },
                dedup_edges(edges),
            )
        }
        SemanticOpKind::If { else_target, .. } => {
            let then_edge =
                fallthrough().unwrap_or_else(|| map_edge(source, *else_target, semantic_to_block));
            let else_edge = map_edge(source, *else_target, semantic_to_block);
            (
                CfgTerminator::Branch {
                    op_index,
                    then_edge,
                    else_edge,
                },
                dedup_edges(collections::vec![then_edge, else_edge]),
            )
        }
        SemanticOpKind::Else { end_target } => {
            let edge = map_edge(source, *end_target, semantic_to_block);
            (
                CfgTerminator::Goto { op_index, edge },
                collections::vec![edge],
            )
        }
        _ => {
            let edge = fallthrough().unwrap_or(CfgEdge {
                target: source,
                is_backedge: false,
            });
            (
                CfgTerminator::Goto { op_index, edge },
                collections::vec![edge],
            )
        }
    }
}

fn map_edge(
    source: CfgBlockId,
    target: SemanticTarget,
    semantic_to_block: &[CfgBlockId],
) -> CfgEdge {
    let target_block = semantic_to_block[target.index().as_usize()];
    CfgEdge {
        target: target_block,
        is_backedge: target_block.as_usize() <= source.as_usize(),
    }
}

fn dedup_edges(edges: collections::Vec<CfgEdge>) -> collections::Vec<CfgEdge> {
    let mut out: collections::Vec<CfgEdge> = collections::Vec::with_capacity(edges.len());
    for edge in edges {
        if !out.iter().any(|existing| existing.target == edge.target) {
            out.push(edge);
        }
    }
    out
}

fn populate_predecessors(blocks: &mut [CfgBlock]) {
    for source in 0..blocks.len() {
        let succs = blocks[source].succs.clone();
        for succ in succs {
            let target = succ.target.as_usize();
            let pred = CfgPredecessor {
                block: CfgBlockId(source as u32),
                is_backedge: succ.is_backedge,
            };
            if !blocks[target].preds.contains(&pred) {
                blocks[target].preds.push(pred);
            }
        }
    }
}

fn populate_flags(blocks: &mut [CfgBlock]) {
    for block in blocks {
        block.flags.is_merge = block.preds.len() > 1;
        block.flags.is_loop_header = block.preds.iter().any(|pred| pred.is_backedge);
    }
}

fn for_each_semantic_successor(
    index: usize,
    len: usize,
    op: &SemanticOp,
    mut f: impl FnMut(SemanticTarget),
) {
    let fallthrough = || {
        if index + 1 < len {
            Some(SemanticTarget::new(index + 1))
        } else {
            None
        }
    };

    match &op.kind {
        SemanticOpKind::Primitive(PrimitiveOpKind::Unreachable)
        | SemanticOpKind::ReturnVoid
        | SemanticOpKind::ReturnOne
        | SemanticOpKind::Return { .. }
        | SemanticOpKind::ReturnCallDirect { .. }
        | SemanticOpKind::ReturnCallIndirect { .. }
        | SemanticOpKind::ReturnCallRef { .. } => {}
        SemanticOpKind::Br { target, .. } => {
            f(*target);
        }
        SemanticOpKind::BrIf { target, .. }
        | SemanticOpKind::BrOnNull { target, .. }
        | SemanticOpKind::BrOnCast { target, .. } => {
            f(*target);
            if let Some(ft) = fallthrough() {
                f(ft);
            }
        }
        SemanticOpKind::BrOnNonNull { target, .. }
        | SemanticOpKind::BrOnCastFail { target, .. } => {
            if let Some(ft) = fallthrough() {
                f(ft);
            }
            f(*target);
        }
        SemanticOpKind::BrTable { entries } => {
            for entry in entries {
                f(entry.target);
            }
        }
        SemanticOpKind::If { else_target, .. } => {
            if let Some(ft) = fallthrough() {
                f(ft);
            }
            f(*else_target);
        }
        SemanticOpKind::Else { end_target } => {
            f(*end_target);
        }
        _ => {
            if let Some(ft) = fallthrough() {
                f(ft);
            }
        }
    }
}

fn is_plain_fallthrough(index: usize, len: usize, op: &SemanticOp) -> bool {
    match &op.kind {
        SemanticOpKind::Primitive(PrimitiveOpKind::Unreachable)
        | SemanticOpKind::ReturnVoid
        | SemanticOpKind::ReturnOne
        | SemanticOpKind::Return { .. }
        | SemanticOpKind::ReturnCallDirect { .. }
        | SemanticOpKind::ReturnCallIndirect { .. }
        | SemanticOpKind::ReturnCallRef { .. }
        | SemanticOpKind::Br { .. }
        | SemanticOpKind::BrIf { .. }
        | SemanticOpKind::BrOnNull { .. }
        | SemanticOpKind::BrOnNonNull { .. }
        | SemanticOpKind::BrOnCast { .. }
        | SemanticOpKind::BrOnCastFail { .. }
        | SemanticOpKind::BrTable { .. }
        | SemanticOpKind::If { .. }
        | SemanticOpKind::Else { .. } => false,
        _ => index + 1 < len,
    }
}

fn splits_after(kind: &SemanticOpKind) -> bool {
    matches!(
        kind,
        SemanticOpKind::If { .. }
            | SemanticOpKind::Else { .. }
            | SemanticOpKind::Br { .. }
            | SemanticOpKind::BrIf { .. }
            | SemanticOpKind::BrOnNull { .. }
            | SemanticOpKind::BrOnNonNull { .. }
            | SemanticOpKind::BrOnCast { .. }
            | SemanticOpKind::BrOnCastFail { .. }
            | SemanticOpKind::BrTable { .. }
            | SemanticOpKind::ReturnVoid
            | SemanticOpKind::ReturnOne
            | SemanticOpKind::Return { .. }
            | SemanticOpKind::ReturnCallDirect { .. }
            | SemanticOpKind::ReturnCallIndirect { .. }
            | SemanticOpKind::ReturnCallRef { .. }
            | SemanticOpKind::Primitive(PrimitiveOpKind::Unreachable)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::wasm::semantic_ir::{SemanticOp, SemanticOpKind, SemanticProgram};

    #[test]
    fn builds_cfg_for_loop_and_marks_backedge_header() {
        let semantic = SemanticProgram {
            local_count: 3,
            ops: collections::vec![
                SemanticOp {
                    kind: SemanticOpKind::Block {
                        params: 0,
                        results: 0
                    }
                },
                SemanticOp {
                    kind: SemanticOpKind::Loop {
                        params: 0,
                        results: 0
                    }
                },
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 1 }
                },
                SemanticOp {
                    kind: SemanticOpKind::BrIf {
                        stack_drop: 0,
                        arity: 0,
                        target: SemanticTarget::new(8)
                    }
                },
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 0 }
                },
                SemanticOp {
                    kind: SemanticOpKind::LocalSet { idx: 0 }
                },
                SemanticOp {
                    kind: SemanticOpKind::Br {
                        stack_drop: 0,
                        arity: 0,
                        target: SemanticTarget::new(1)
                    }
                },
                SemanticOp {
                    kind: SemanticOpKind::End
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnVoid
                },
            ],
            ..SemanticProgram::default()
        };

        let cfg = build_semantic_cfg(&semantic);

        assert_eq!(cfg.blocks.len(), 4);
        assert!(cfg.blocks[1].flags.is_loop_header);
        assert!(cfg.blocks[2].succs.iter().any(|edge| edge.is_backedge));
    }
}
