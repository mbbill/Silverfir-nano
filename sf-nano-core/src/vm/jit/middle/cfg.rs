//! Explicit CFG formation from semantic Wasm IR.

use crate::collections;

use crate::vm::jit::wasm::{
    common::{BrTableEntry, SemanticTarget},
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
}

impl SemanticCfg {
    pub(crate) fn block_for_semantic_index(&self, semantic_index: usize) -> Option<CfgBlockId> {
        let mut lo = 0usize;
        let mut hi = self.blocks.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.blocks[mid].range.end <= semantic_index {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        self.blocks
            .get(lo)
            .and_then(|block| block.range.contains(&semantic_index).then_some(block.id))
    }
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

/// Which edge of a conditional branch falls through to the next op; the
/// opposite edge is the explicit branch target.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FallthroughEdge {
    /// The `then` edge falls through (`if`, `br_on_non_null`, `br_on_cast_fail`).
    Then,
    /// The `else` edge falls through (`br_if`, `br_on_null`, `br_on_cast`).
    Else,
}

/// How a semantic op transfers control out of its block.
///
/// This is the single source of truth for a semantic op's control behavior.
/// Successor enumeration, block-leader detection, and terminator construction
/// are all derived from it so they cannot disagree.
enum ControlKind<'a> {
    /// Terminates the block with no successors (`unreachable`, `throw`).
    Trap,
    /// Returns from the function; no successors.
    Return,
    /// Unconditional branch to a single target.
    Goto(SemanticTarget),
    /// Conditional two-way branch to `target`, with the opposite edge falling
    /// through to the next op.
    Branch {
        target: SemanticTarget,
        fallthrough: FallthroughEdge,
    },
    /// Multi-way table branch.
    Table(&'a [BrTableEntry]),
    /// Falls through to the next op (or nowhere when it is the last op).
    Fallthrough,
}

fn control_kind(kind: &SemanticOpKind) -> ControlKind<'_> {
    match kind {
        SemanticOpKind::Primitive(PrimitiveOpKind::Unreachable)
        | SemanticOpKind::Throw { .. }
        | SemanticOpKind::ThrowRef => ControlKind::Trap,
        SemanticOpKind::ReturnVoid
        | SemanticOpKind::ReturnOne
        | SemanticOpKind::Return { .. }
        | SemanticOpKind::ReturnCallDirect { .. }
        | SemanticOpKind::ReturnCallIndirect { .. }
        | SemanticOpKind::ReturnCallRef { .. } => ControlKind::Return,
        SemanticOpKind::Br { target, .. } => ControlKind::Goto(*target),
        SemanticOpKind::BrIf { target, .. }
        | SemanticOpKind::BrOnNull { target, .. }
        | SemanticOpKind::BrOnCast { target, .. } => ControlKind::Branch {
            target: *target,
            fallthrough: FallthroughEdge::Else,
        },
        SemanticOpKind::BrOnNonNull { target, .. }
        | SemanticOpKind::BrOnCastFail { target, .. } => ControlKind::Branch {
            target: *target,
            fallthrough: FallthroughEdge::Then,
        },
        SemanticOpKind::BrTable { entries } => ControlKind::Table(entries),
        SemanticOpKind::If { else_target, .. } => ControlKind::Branch {
            target: *else_target,
            fallthrough: FallthroughEdge::Then,
        },
        SemanticOpKind::Else { end_target } => ControlKind::Goto(*end_target),
        _ => ControlKind::Fallthrough,
    }
}

/// The next semantic op, when the current op is not the last one.
fn semantic_fallthrough(index: usize, len: usize) -> Option<SemanticTarget> {
    (index + 1 < len).then(|| SemanticTarget::new(index + 1))
}

fn build_terminator(
    source: CfgBlockId,
    op_index: usize,
    op: &SemanticOp,
    semantic: &SemanticProgram,
    semantic_to_block: &[CfgBlockId],
) -> (CfgTerminator, collections::Vec<CfgEdge>) {
    let len = semantic.ops.len();
    let fallthrough_edge =
        || semantic_fallthrough(op_index, len).map(|t| map_edge(source, t, semantic_to_block));

    match control_kind(&op.kind) {
        ControlKind::Trap => (
            CfgTerminator::TrapUnreachable { op_index },
            collections::Vec::new(),
        ),
        ControlKind::Return => (CfgTerminator::Return { op_index }, collections::Vec::new()),
        ControlKind::Goto(target) => {
            let edge = map_edge(source, target, semantic_to_block);
            (
                CfgTerminator::Goto { op_index, edge },
                collections::vec![edge],
            )
        }
        ControlKind::Branch {
            target,
            fallthrough,
        } => {
            let target_edge = map_edge(source, target, semantic_to_block);
            let (then_edge, else_edge) = match fallthrough {
                FallthroughEdge::Then => (fallthrough_edge().unwrap_or(target_edge), target_edge),
                FallthroughEdge::Else => (target_edge, fallthrough_edge().unwrap_or(target_edge)),
            };
            (
                CfgTerminator::Branch {
                    op_index,
                    then_edge,
                    else_edge,
                },
                dedup_edges(collections::vec![then_edge, else_edge]),
            )
        }
        ControlKind::Table(entries) => {
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
        ControlKind::Fallthrough => {
            let edge = fallthrough_edge().unwrap_or(CfgEdge {
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
    match control_kind(&op.kind) {
        ControlKind::Trap | ControlKind::Return => {}
        ControlKind::Goto(target) => f(target),
        ControlKind::Branch {
            target,
            fallthrough,
        } => match fallthrough {
            FallthroughEdge::Then => {
                if let Some(ft) = semantic_fallthrough(index, len) {
                    f(ft);
                }
                f(target);
            }
            FallthroughEdge::Else => {
                f(target);
                if let Some(ft) = semantic_fallthrough(index, len) {
                    f(ft);
                }
            }
        },
        ControlKind::Table(entries) => {
            for entry in entries {
                f(entry.target);
            }
        }
        ControlKind::Fallthrough => {
            if let Some(ft) = semantic_fallthrough(index, len) {
                f(ft);
            }
        }
    }
}

fn is_plain_fallthrough(index: usize, len: usize, op: &SemanticOp) -> bool {
    matches!(control_kind(&op.kind), ControlKind::Fallthrough) && index + 1 < len
}

fn splits_after(kind: &SemanticOpKind) -> bool {
    !matches!(control_kind(kind), ControlKind::Fallthrough)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::jit::wasm::semantic_ir::{SemanticOp, SemanticOpKind, SemanticProgram};

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
