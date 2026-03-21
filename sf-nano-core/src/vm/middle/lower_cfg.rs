//! CFG block shaping from decoded semantics.

use alloc::vec::Vec;

use crate::vm::{
    middle::lir::target::LirTarget,
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

        for_each_semantic_successor(last_semantic, semantic.ops.len(), op, |target| {
            let target_index = target.index().as_usize();
            if let Some(target_block) = semantic_to_block.get(target_index) {
                let target_block = target_block.as_usize();
                if target_block < visited.len() && !visited[target_block] {
                    pending.push(target_block);
                }
            }
        });
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

pub(super) fn for_each_semantic_successor(
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
        SemanticOpKind::Primitive(crate::vm::wasm::primitive_op::PrimitiveOpKind::Unreachable)
        | SemanticOpKind::ReturnVoid
        | SemanticOpKind::ReturnOne
        | SemanticOpKind::Return { .. } => {}
        SemanticOpKind::Br { target, .. } => {
            f(*target);
        }
        SemanticOpKind::BrIf { target, .. } => {
            f(*target);
            if let Some(ft) = fallthrough() {
                f(ft);
            }
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

/// Returns true if the op has exactly one successor and it's the next op.
pub(super) fn is_plain_fallthrough(index: usize, len: usize, op: &SemanticOp) -> bool {
    match &op.kind {
        SemanticOpKind::Primitive(crate::vm::wasm::primitive_op::PrimitiveOpKind::Unreachable)
        | SemanticOpKind::ReturnVoid
        | SemanticOpKind::ReturnOne
        | SemanticOpKind::Return { .. }
        | SemanticOpKind::Br { .. }
        | SemanticOpKind::BrIf { .. }
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
            | SemanticOpKind::BrTable { .. }
            | SemanticOpKind::ReturnVoid
            | SemanticOpKind::ReturnOne
            | SemanticOpKind::Return { .. }
            | SemanticOpKind::Primitive(
                crate::vm::wasm::primitive_op::PrimitiveOpKind::Unreachable
            )
    )
}
