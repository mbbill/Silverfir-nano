use alloc::vec;
use alloc::vec::Vec;

use crate::vm::lowered::{BrTableEntry, IrOp, IrOpKind, OpIndex};

use super::bridge::{self, ColdHelperKind};

fn helper_keep_top(kind: &IrOpKind) -> Option<usize> {
    if matches!(kind, IrOpKind::BrTable { .. }) {
        return Some(1);
    }
    let helper = bridge::cold_helper_kind(kind)?;
    Some(match helper {
        ColdHelperKind::Br
        | ColdHelperKind::Else
        | ColdHelperKind::CallExternal
        | ColdHelperKind::CallInternal
        | ColdHelperKind::CallIndirect
        | ColdHelperKind::GlobalGet
        | ColdHelperKind::DataDrop
        | ColdHelperKind::ElemDrop
        | ColdHelperKind::RefNull
        | ColdHelperKind::RefFunc
        | ColdHelperKind::TableSize => 0,
        ColdHelperKind::BrIf
        | ColdHelperKind::If
        | ColdHelperKind::GlobalSet
        | ColdHelperKind::MemoryGrow
        | ColdHelperKind::I32Popcnt
        | ColdHelperKind::I64Popcnt
        | ColdHelperKind::I32TruncF32S
        | ColdHelperKind::I32TruncF32U
        | ColdHelperKind::I32TruncF64S
        | ColdHelperKind::I32TruncF64U
        | ColdHelperKind::I64TruncF32S
        | ColdHelperKind::I64TruncF32U
        | ColdHelperKind::I64TruncF64S
        | ColdHelperKind::I64TruncF64U
        | ColdHelperKind::TableGet
        | ColdHelperKind::RefIsNull => 1,
        ColdHelperKind::F32Copysign
        | ColdHelperKind::F64Copysign
        | ColdHelperKind::TableSet
        | ColdHelperKind::TableGrow => 2,
        ColdHelperKind::MemoryCopy
        | ColdHelperKind::MemoryFill
        | ColdHelperKind::MemoryInit
        | ColdHelperKind::TableFill
        | ColdHelperKind::TableCopy
        | ColdHelperKind::TableInit => 3,
    })
}

fn remap_target(index_map: &[usize], target: Option<OpIndex>) -> Option<OpIndex> {
    target.map(|idx| OpIndex::new(index_map[idx.as_usize()]))
}

fn remap_kind_targets(kind: &mut IrOpKind, index_map: &[usize]) {
    if let IrOpKind::BrTable { entries, .. } = kind {
        for entry in entries.iter_mut() {
            if let Some(target_idx) = entry.target_idx {
                entry.target_idx = Some(OpIndex::new(index_map[target_idx.as_usize()]));
            }
        }
    }
}

fn height_after(kind: &IrOpKind, pre_height: usize) -> usize {
    let (pop, push) = crate::vm::lowered::stack_effect(kind);
    pre_height.saturating_sub(pop as usize) + push as usize
}

fn update_spill_depth(kind: &IrOpKind, pre_height: usize, spill_depth: usize) -> usize {
    match kind {
        IrOpKind::Spill { count, .. } => core::cmp::min(pre_height, spill_depth + *count as usize),
        IrOpKind::Fill { count, .. } => spill_depth.saturating_sub(*count as usize),
        IrOpKind::CallExternal { .. }
        | IrOpKind::CallInternal { .. }
        | IrOpKind::CallIndirect { .. } => height_after(kind, pre_height),
        _ => core::cmp::min(spill_depth, height_after(kind, pre_height)),
    }
}

fn make_spill(op: &IrOp, operand_base: usize, spill_depth: usize, count: usize, tos_regs: usize) -> IrOp {
    let slot = operand_base + spill_depth + count - 1;
    assert!(slot <= u16::MAX as usize, "native spill slot index overflow");
    IrOp {
        kind: IrOpKind::Spill {
            slot: slot as u16,
            count: count as u8,
        },
        variant: (((spill_depth + count - 1) % tos_regs) + 1) as u8,
        pre_height: op.pre_height,
        fallthrough: None,
        alt_target: None,
        has_target: false,
    }
}

pub(crate) fn canonicalize_cold_helpers(
    ops: &[IrOp],
    operand_base: usize,
    tos_regs: usize,
) -> Vec<IrOp> {
    if ops.is_empty() {
        return Vec::new();
    }

    let mut expansion_at = vec![0usize; ops.len()];
    let mut total_expansion = 0usize;
    let mut spill_depth = 0usize;

    for (i, op) in ops.iter().enumerate() {
        expansion_at[i] = total_expansion;

        let pre_height = op.pre_height as usize;
        let spilled = core::cmp::min(spill_depth, pre_height);
        let live = pre_height.saturating_sub(spilled);
        if let Some(keep_top) = helper_keep_top(&op.kind) {
            if live > keep_top {
                total_expansion += 1;
            }
        }

        spill_depth = update_spill_depth(&op.kind, pre_height, spill_depth);
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
        let spilled = core::cmp::min(spill_depth, pre_height);
        let live = pre_height.saturating_sub(spilled);
        let inserted = helper_keep_top(&op.kind)
            .map(|keep_top| live.saturating_sub(keep_top))
            .unwrap_or(0);

        if inserted > 0 {
            let helper_index = result.len() + 1;
            let mut spill = make_spill(&op, operand_base, spill_depth, inserted, tos_regs);
            spill.fallthrough = Some(OpIndex::new(helper_index));
            result.push(spill);
        }

        result.push(op.clone());
        spill_depth = update_spill_depth(&op.kind, pre_height, spill_depth);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_op(kind: IrOpKind, pre_height: u16, variant: u8) -> IrOp {
        IrOp {
            kind,
            variant,
            pre_height,
            fallthrough: None,
            alt_target: None,
            has_target: false,
        }
    }

    #[test]
    fn test_canonicalize_spills_before_global_set() {
        let ops = vec![
            make_op(IrOpKind::I32Const { value: 1 }, 0, 1),
            make_op(IrOpKind::I32Const { value: 2 }, 1, 2),
            make_op(IrOpKind::I32Const { value: 3 }, 2, 3),
            make_op(IrOpKind::GlobalSet { idx: 0 }, 3, 3),
        ];

        let out = canonicalize_cold_helpers(&ops, 8, 4);
        assert_eq!(out.len(), 5);
        match out[3].kind {
            IrOpKind::Spill { slot, count } => {
                assert_eq!(slot, 9);
                assert_eq!(count, 2);
            }
            _ => panic!("expected inserted spill"),
        }
        assert!(matches!(out[4].kind, IrOpKind::GlobalSet { .. }));
    }

    #[test]
    fn test_canonicalize_preserves_branch_targets() {
        let mut ops = vec![
            make_op(IrOpKind::I32Const { value: 1 }, 0, 1),
            make_op(IrOpKind::GlobalGet { idx: 0 }, 1, 2),
            make_op(IrOpKind::Nop, 2, 2),
        ];
        ops[2].alt_target = Some(OpIndex::new(1));

        let out = canonicalize_cold_helpers(&ops, 8, 4);
        assert_eq!(out.len(), 4);
        assert_eq!(out[3].alt_target, Some(OpIndex::new(1)));
        assert!(matches!(out[1].kind, IrOpKind::GlobalGet { .. }));
    }
}
