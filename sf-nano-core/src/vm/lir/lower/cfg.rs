//! CFG block shaping for LIR lowering.

use alloc::vec::Vec;

use crate::vm::{
    lir::target::LirTarget,
    plan::PlannedProgram,
    wasm::semantic_ir::{SemanticOpKind, SemanticProgram},
};

pub(super) fn build_block_ranges(
    semantic: &SemanticProgram,
    planned: &PlannedProgram,
) -> Vec<core::ops::Range<usize>> {
    let len = semantic.ops.len();
    let mut leaders = alloc::vec![false; len + 1];
    leaders[0] = true;
    leaders[len] = true;

    for group in &planned.groups.groups {
        let start = group.start as usize;
        let end = group.end as usize;
        if start < len {
            leaders[start] = true;
        }
        if end <= len {
            leaders[end] = true;
        }
    }

    for (index, op) in semantic.ops.iter().enumerate() {
        if let Some(target) = op.alt {
            let target_index = target.index().as_usize();
            if target_index < len {
                leaders[target_index] = true;
            }
        }

        if let SemanticOpKind::BrTable { entries } = &op.kind {
            for entry in entries {
                if let Some(target) = entry.target {
                    let target_index = target.index().as_usize();
                    if target_index < len {
                        leaders[target_index] = true;
                    }
                }
            }
        }

        if splits_after(&op.kind) && index + 1 < len {
            leaders[index + 1] = true;
        }
    }

    let starts = leaders
        .iter()
        .enumerate()
        .filter_map(|(index, is_leader)| is_leader.then_some(index))
        .collect::<Vec<_>>();

    starts
        .windows(2)
        .filter_map(|window| {
            let start = window[0];
            let end = window[1];
            (start < end).then_some(start..end)
        })
        .collect()
}

pub(super) fn retain_reachable_blocks(
    semantic: &SemanticProgram,
    block_ranges: Vec<core::ops::Range<usize>>,
) -> Vec<core::ops::Range<usize>> {
    if block_ranges.is_empty() {
        return block_ranges;
    }

    let semantic_to_block = build_semantic_to_block_map(semantic.ops.len(), &block_ranges);
    let mut visited = alloc::vec![false; block_ranges.len()];
    let mut pending = alloc::vec![0usize];

    while let Some(block_index) = pending.pop() {
        if visited.get(block_index).copied().unwrap_or(false) {
            continue;
        }
        visited[block_index] = true;

        let Some(last_semantic) = block_ranges
            .get(block_index)
            .and_then(|range| range.end.checked_sub(1))
        else {
            continue;
        };
        let Some(op) = semantic.ops.get(last_semantic) else {
            continue;
        };

        for target in semantic_successors(op) {
            let target_index = target.index().as_usize();
            if let Some(target_block) = semantic_to_block.get(target_index) {
                let target_block = target_block.as_usize();
                if target_block < visited.len() && !visited[target_block] {
                    pending.push(target_block);
                }
            }
        }
    }

    block_ranges
        .into_iter()
        .enumerate()
        .filter_map(|(index, range)| visited[index].then_some(range))
        .collect()
}

pub(super) fn build_semantic_to_block_map(
    semantic_len: usize,
    block_ranges: &[core::ops::Range<usize>],
) -> Vec<LirTarget> {
    let mut map = alloc::vec![LirTarget::default(); semantic_len];
    for (block_index, range) in block_ranges.iter().enumerate() {
        for semantic_index in range.clone() {
            map[semantic_index] = LirTarget(block_index as u32);
        }
    }
    map
}

fn semantic_successors(
    op: &crate::vm::wasm::semantic_ir::SemanticOp,
) -> Vec<crate::vm::wasm::common::SemanticTarget> {
    use crate::vm::wasm::{common::SemanticTarget, semantic_ir::SemanticOpKind};

    let mut targets = Vec::new();
    let push_next = |targets: &mut Vec<SemanticTarget>,
                     next: Option<crate::vm::wasm::common::SemanticIndex>| {
        if let Some(next) = next {
            targets.push(SemanticTarget::new(next.as_usize()));
        }
    };

    match &op.kind {
        SemanticOpKind::Primitive(crate::vm::wasm::primitive_op::PrimitiveOpKind::Unreachable)
        | SemanticOpKind::ReturnVoid
        | SemanticOpKind::ReturnOne
        | SemanticOpKind::Return { .. } => {}
        SemanticOpKind::Br { .. } => {
            if let Some(target) = op.alt {
                targets.push(target);
            }
        }
        SemanticOpKind::BrIf { .. } => {
            if let Some(target) = op.alt {
                targets.push(target);
            }
            push_next(&mut targets, op.next);
        }
        SemanticOpKind::BrTable { entries } => {
            for entry in entries {
                if let Some(target) = entry.target {
                    targets.push(target);
                }
            }
        }
        SemanticOpKind::If { .. } => {
            push_next(&mut targets, op.next);
            if let Some(target) = op.alt {
                targets.push(target);
            }
        }
        SemanticOpKind::Else => {
            if let Some(target) = op.alt {
                targets.push(target);
            }
        }
        _ => push_next(&mut targets, op.next),
    }

    targets
}

fn splits_after(kind: &SemanticOpKind) -> bool {
    matches!(
        kind,
        SemanticOpKind::If { .. }
            | SemanticOpKind::Else
            | SemanticOpKind::Br { .. }
            | SemanticOpKind::BrIf { .. }
            | SemanticOpKind::BrTable { .. }
            | SemanticOpKind::ReturnVoid
            | SemanticOpKind::ReturnOne
            | SemanticOpKind::Return { .. }
            | SemanticOpKind::Primitive(
                crate::vm::wasm::primitive_op::PrimitiveOpKind::Unreachable
            )
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::{
        plan::{group::GroupPlan, PlannedProgram},
        wasm::{
            common::{SemanticIndex, SemanticTarget},
            primitive_op::PrimitiveOpKind,
            semantic_ir::{SemanticOp, SemanticOpKind, SemanticProgram},
        },
    };

    #[test]
    fn retains_reachable_end_block_after_value_branch() {
        let semantic = SemanticProgram {
            params: 0,
            results: 1,
            local_count: 0,
            max_stack_height: 1,
            ops: alloc::vec![
                SemanticOp {
                    kind: SemanticOpKind::Block {
                        params: 0,
                        results: 1,
                    },
                    next: Some(SemanticIndex::new(1)),
                    alt: None,
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 1 }),
                    next: Some(SemanticIndex::new(2)),
                    alt: None,
                },
                SemanticOp {
                    kind: SemanticOpKind::Br {
                        stack_drop: 0,
                        arity: 1,
                    },
                    next: Some(SemanticIndex::new(3)),
                    alt: Some(SemanticTarget::new(4)),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Ctz),
                    next: Some(SemanticIndex::new(4)),
                    alt: None,
                },
                SemanticOp {
                    kind: SemanticOpKind::End,
                    next: Some(SemanticIndex::new(5)),
                    alt: None,
                },
                SemanticOp {
                    kind: SemanticOpKind::End,
                    next: Some(SemanticIndex::new(6)),
                    alt: None,
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnOne,
                    next: None,
                    alt: None,
                },
            ],
        };
        let planned = PlannedProgram {
            groups: GroupPlan::default(),
            ..PlannedProgram::default()
        };

        let ranges = build_block_ranges(&semantic, &planned);
        assert_eq!(ranges, alloc::vec![0..3, 3..4, 4..7]);

        let reachable = retain_reachable_blocks(&semantic, ranges);
        assert_eq!(reachable, alloc::vec![0..3, 4..7]);
    }

    #[test]
    fn retains_if_end_reached_via_else_marker() {
        let semantic = SemanticProgram {
            params: 2,
            results: 1,
            local_count: 2,
            max_stack_height: 1,
            ops: alloc::vec![
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 0 },
                    next: Some(SemanticIndex::new(1)),
                    alt: None,
                },
                SemanticOp {
                    kind: SemanticOpKind::If {
                        params: 0,
                        results: 1,
                    },
                    next: Some(SemanticIndex::new(2)),
                    alt: Some(SemanticTarget::new(4)),
                },
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 1 },
                    next: Some(SemanticIndex::new(3)),
                    alt: None,
                },
                SemanticOp {
                    kind: SemanticOpKind::Else,
                    next: Some(SemanticIndex::new(4)),
                    alt: Some(SemanticTarget::new(5)),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 4 }),
                    next: Some(SemanticIndex::new(5)),
                    alt: None,
                },
                SemanticOp {
                    kind: SemanticOpKind::End,
                    next: Some(SemanticIndex::new(6)),
                    alt: None,
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnOne,
                    next: None,
                    alt: None,
                },
            ],
        };
        let planned = PlannedProgram {
            groups: GroupPlan::default(),
            ..PlannedProgram::default()
        };

        let ranges = build_block_ranges(&semantic, &planned);
        assert_eq!(ranges, alloc::vec![0..2, 2..4, 4..5, 5..7]);

        let reachable = retain_reachable_blocks(&semantic, ranges);
        assert_eq!(reachable, alloc::vec![0..2, 2..4, 4..5, 5..7]);
    }
}
