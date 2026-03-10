use alloc::vec;
use alloc::vec::Vec;

use crate::vm::{
    lir,
    plan::{CompileConfig, HotLocalPlan},
    wasm::{
        common::{BrTableEntry, OpIndex},
        core_op,
        semantic_ir::{SemanticOp, SemanticOpKind},
    },
};

use super::bridge;

fn preview_lowered_kind(
    op: &SemanticOp,
    hot_locals: HotLocalPlan,
    config: CompileConfig,
    frame_size: usize,
) -> lir::IrOpKind {
    lir::lower::preview_lowered_kind(op.kind.clone(), hot_locals, config, frame_size, op.pre_height)
}

fn helper_kind(
    op: &SemanticOp,
    hot_locals: HotLocalPlan,
    config: CompileConfig,
    frame_size: usize,
) -> Option<bridge::ColdHelperKind> {
    let lowered = preview_lowered_kind(op, hot_locals, config, frame_size);
    bridge::cold_helper_kind(&lowered)
}

fn remap_target(index_map: &[usize], target: Option<OpIndex>) -> Option<OpIndex> {
    target.map(|idx| OpIndex::new(index_map[idx.as_usize()]))
}

fn remap_kind_targets(kind: &mut SemanticOpKind, index_map: &[usize]) {
    if let SemanticOpKind::BrTable { entries } = kind {
        for entry in entries.iter_mut() {
            if let Some(target_idx) = entry.target_idx {
                entry.target_idx = Some(OpIndex::new(index_map[target_idx.as_usize()]));
            }
        }
    }
}

fn height_after(kind: &SemanticOpKind, pre_height: usize) -> usize {
    let (pop, push) = crate::vm::wasm::semantic_ir::stack_effect(kind);
    pre_height.saturating_sub(pop as usize) + push as usize
}

fn update_spill_depth_nonhelper(kind: &SemanticOpKind, pre_height: usize, spill_depth: usize) -> usize {
    match kind {
        SemanticOpKind::CacheSpill { count, .. } => {
            core::cmp::min(pre_height, spill_depth + *count as usize)
        }
        SemanticOpKind::CacheFill { count, .. } => spill_depth.saturating_sub(*count as usize),
        _ => core::cmp::min(spill_depth, height_after(kind, pre_height)),
    }
}

fn operand_need(kind: &SemanticOpKind) -> usize {
    match kind {
        SemanticOpKind::Core(kind) => core_op::stack_effect(kind).0 as usize,
        SemanticOpKind::LocalSet { .. } | SemanticOpKind::LocalTee { .. } => 1,
        SemanticOpKind::If { .. } | SemanticOpKind::BrIf { .. } | SemanticOpKind::BrTable { .. } => 1,
        _ => 0,
    }
}

fn fill_count_for_op(kind: &SemanticOpKind, pre_height: usize, spill_depth: usize) -> usize {
    let need = operand_need(kind);
    if need == 0 {
        return 0;
    }
    let min_spill_for_operands = pre_height.saturating_sub(need);
    spill_depth.saturating_sub(min_spill_for_operands)
}

fn make_spill(
    op: &SemanticOp,
    operand_base: usize,
    spill_depth: usize,
    count: usize,
    tos_regs: usize,
) -> SemanticOp {
    let slot = operand_base + spill_depth + count - 1;
    assert!(slot <= u16::MAX as usize, "native spill slot index overflow");
    let window = (((spill_depth + count - 1) % tos_regs) + 1) as u8 % tos_regs as u8;
    SemanticOp {
        kind: SemanticOpKind::CacheSpill {
            slot: slot as u16,
            count: count as u8,
        },
        window,
        pre_height: op.pre_height,
        fallthrough: None,
        alt_target: None,
        has_target: false,
    }
}

fn make_fill(
    op: &SemanticOp,
    operand_base: usize,
    spill_depth: usize,
    count: usize,
    tos_regs: usize,
) -> SemanticOp {
    let slot = operand_base + spill_depth - 1;
    assert!(slot <= u16::MAX as usize, "native fill slot index overflow");
    let window = (((spill_depth - 1) % tos_regs) + 1) as u8 % tos_regs as u8;
    SemanticOp {
        kind: SemanticOpKind::CacheFill {
            slot: slot as u16,
            count: count as u8,
        },
        window,
        pre_height: op.pre_height,
        fallthrough: None,
        alt_target: None,
        has_target: false,
    }
}

pub(crate) fn canonicalize_cold_helpers(
    ops: &[SemanticOp],
    hot_locals: HotLocalPlan,
    config: CompileConfig,
    frame_size: usize,
) -> Vec<SemanticOp> {
    if ops.is_empty() {
        return Vec::new();
    }

    let operand_base = config.operand_stack_base(frame_size);
    let tos_regs = config.tos_register_count;
    let mut expansion_at = vec![0usize; ops.len()];
    let mut total_expansion = 0usize;
    let mut spill_depth = 0usize;

    for (i, op) in ops.iter().enumerate() {
        expansion_at[i] = total_expansion;
        let pre_height = op.pre_height as usize;
        let current_spill = core::cmp::min(spill_depth, pre_height);

        if helper_kind(op, hot_locals, config, frame_size).is_some() {
            let live = pre_height.saturating_sub(current_spill);
            if live > 0 {
                total_expansion += 1;
            }
            spill_depth = height_after(&op.kind, pre_height);
            continue;
        }

        let fill_count = fill_count_for_op(&op.kind, pre_height, current_spill);
        if fill_count > 0 {
            total_expansion += 1;
            spill_depth = spill_depth.saturating_sub(fill_count);
        }

        spill_depth = update_spill_depth_nonhelper(&op.kind, pre_height, spill_depth);
    }

    if total_expansion == 0 {
        return ops.to_vec();
    }

    let old_to_new: Vec<usize> = expansion_at
        .iter()
        .enumerate()
        .map(|(i, &exp)| i + exp)
        .collect();

    let mut remapped_ops = ops.to_vec();
    for op in remapped_ops.iter_mut() {
        op.fallthrough = remap_target(&old_to_new, op.fallthrough);
        op.alt_target = remap_target(&old_to_new, op.alt_target);
        remap_kind_targets(&mut op.kind, &old_to_new);
    }

    let mut result = Vec::with_capacity(remapped_ops.len() + total_expansion);
    spill_depth = 0;

    for op in remapped_ops {
        let pre_height = op.pre_height as usize;
        let current_spill = core::cmp::min(spill_depth, pre_height);

        if helper_kind(&op, hot_locals, config, frame_size).is_some() {
            let live = pre_height.saturating_sub(current_spill);
            if live > 0 {
                result.push(make_spill(&op, operand_base, current_spill, live, tos_regs));
                spill_depth = spill_depth.saturating_add(live);
            }
            result.push(op.clone());
            spill_depth = height_after(&op.kind, pre_height);
            continue;
        }

        let fill_count = fill_count_for_op(&op.kind, pre_height, current_spill);
        if fill_count > 0 {
            result.push(make_fill(&op, operand_base, current_spill, fill_count, tos_regs));
            spill_depth = spill_depth.saturating_sub(fill_count);
        }

        result.push(op.clone());
        spill_depth = update_spill_depth_nonhelper(&op.kind, pre_height, spill_depth);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::plan::{FAST_COMPILE_CONFIG, HOT_LOCAL_COUNT};

    fn test_hot_locals(raw: [Option<u32>; HOT_LOCAL_COUNT]) -> HotLocalPlan {
        HotLocalPlan::from_parts(raw, raw)
    }

    fn op(kind: SemanticOpKind, window: u8, pre_height: u16) -> SemanticOp {
        SemanticOp {
            kind,
            window,
            pre_height,
            fallthrough: None,
            alt_target: None,
            has_target: false,
        }
    }

    #[test]
    fn test_canonicalize_spills_and_refills_around_global_get() {
        let ops = vec![
            op(SemanticOpKind::Core(core_op::CoreOpKind::I32Const { value: 1 }), 0, 0),
            op(SemanticOpKind::Core(core_op::CoreOpKind::GlobalGet { idx: 0 }), 1, 1),
            op(SemanticOpKind::Core(core_op::CoreOpKind::I32Add), 2, 2),
        ];

        let out = canonicalize_cold_helpers(&ops, test_hot_locals([None; HOT_LOCAL_COUNT]), FAST_COMPILE_CONFIG, 1);
        assert_eq!(out.len(), 5);
        assert!(matches!(out[1].kind, SemanticOpKind::CacheSpill { count: 1, .. }));
        assert!(matches!(out[2].kind, SemanticOpKind::Core(core_op::CoreOpKind::GlobalGet { .. })));
        assert!(matches!(out[3].kind, SemanticOpKind::CacheFill { count: 2, .. }));
        assert!(matches!(out[4].kind, SemanticOpKind::Core(core_op::CoreOpKind::I32Add)));
    }
}
