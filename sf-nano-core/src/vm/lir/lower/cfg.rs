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

fn splits_after(kind: &SemanticOpKind) -> bool {
    matches!(
        kind,
        SemanticOpKind::If { .. }
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
