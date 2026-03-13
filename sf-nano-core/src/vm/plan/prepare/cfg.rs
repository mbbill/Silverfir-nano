//! CFG block shaping from decoded semantics.

use alloc::vec::Vec;

use crate::vm::{
    lir::target::LirTarget,
    wasm::{
        common::SemanticTarget,
        semantic_ir::{SemanticOp, SemanticOpKind, SemanticProgram},
    },
};

pub(super) fn build_block_ranges(semantic: &SemanticProgram) -> Vec<core::ops::Range<usize>> {
    let len = semantic.ops.len();
    let mut leaders = alloc::vec![false; len + 1];
    leaders[0] = true;
    leaders[len] = true;

    for (index, op) in semantic.ops.iter().enumerate() {
        for target in semantic_successors(index, semantic.ops.len(), op) {
            let target_index = target.index().as_usize();
            if target_index < len {
                leaders[target_index] = true;
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

        for target in semantic_successors(last_semantic, semantic.ops.len(), op) {
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

pub(super) fn semantic_successors(
    index: usize,
    len: usize,
    op: &SemanticOp,
) -> Vec<SemanticTarget> {
    let mut targets = Vec::new();
    let push_fallthrough = |targets: &mut Vec<SemanticTarget>| {
        if index + 1 < len {
            targets.push(SemanticTarget::new(index + 1));
        }
    };

    match &op.kind {
        SemanticOpKind::Primitive(crate::vm::wasm::primitive_op::PrimitiveOpKind::Unreachable)
        | SemanticOpKind::ReturnVoid
        | SemanticOpKind::ReturnOne
        | SemanticOpKind::Return { .. } => {}
        SemanticOpKind::Br { target, .. } => {
            targets.push(*target);
        }
        SemanticOpKind::BrIf { target, .. } => {
            targets.push(*target);
            push_fallthrough(&mut targets);
        }
        SemanticOpKind::BrTable { entries } => {
            for entry in entries {
                targets.push(entry.target);
            }
        }
        SemanticOpKind::If { else_target, .. } => {
            push_fallthrough(&mut targets);
            targets.push(*else_target);
        }
        SemanticOpKind::Else { end_target } => {
            targets.push(*end_target);
        }
        _ => push_fallthrough(&mut targets),
    }

    targets
}

fn splits_after(kind: &SemanticOpKind) -> bool {
    matches!(
        kind,
        SemanticOpKind::If { .. }
            | SemanticOpKind::Else { .. }
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
