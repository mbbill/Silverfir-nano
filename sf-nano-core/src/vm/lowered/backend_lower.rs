//! Lower semantic compile IR into the current backend-lowered IR.

use alloc::vec::Vec;

use crate::vm::{
    compile::{
        BrTableEntry, CoreOpKind, OpIndex,
        semantic_ir::{SemanticOp, SemanticOpKind},
    },
    lowered::ir::{self, IrOp as LoweredIrOp, IrOpKind as LoweredIrOpKind},
    planner::{CompileConfig, HOT_LOCAL_COUNT, HotLocalPlan},
};

#[inline]
fn remap_index(index_map: &[usize], end_index: usize, index: Option<OpIndex>) -> Option<OpIndex> {
    index.map(|idx| {
        let raw = idx.as_usize();
        if raw == index_map.len() {
            OpIndex::new(end_index)
        } else {
            OpIndex::new(index_map[raw])
        }
    })
}

#[inline]
fn remap_kind_targets(kind: &mut LoweredIrOpKind, index_map: &[usize]) {
    if let LoweredIrOpKind::BrTable { entries, .. } = kind {
        for entry in entries.iter_mut() {
            if let Some(target_idx) = entry.target_idx {
                entry.target_idx = Some(OpIndex::new(index_map[target_idx.as_usize()]));
            }
        }
    }
}

#[inline]
fn hot_slot_enabled(hot_locals: HotLocalPlan, config: CompileConfig, slot: usize) -> bool {
    slot < config.hot_local_count && hot_locals.effective()[slot].is_some()
}

fn lower_semantic_kind(
    kind: SemanticOpKind,
    hot_locals: HotLocalPlan,
    config: CompileConfig,
    frame_size: usize,
    pre_height: u16,
) -> LoweredIrOpKind {
    match kind {
        SemanticOpKind::Core(core) => core.into(),
        SemanticOpKind::LocalGet { idx } => {
            let remapped = hot_locals.remap_local(idx as u32);
            match remapped {
                0 if hot_slot_enabled(hot_locals, config, 0) => LoweredIrOpKind::LocalGetHot { reg: 0 },
                1 if hot_slot_enabled(hot_locals, config, 1) => LoweredIrOpKind::LocalGetHot { reg: 1 },
                2 if hot_slot_enabled(hot_locals, config, 2) => LoweredIrOpKind::LocalGetHot { reg: 2 },
                _ => LoweredIrOpKind::LocalGetFrame { idx: remapped as u16 },
            }
        }
        SemanticOpKind::LocalSet { idx } => {
            let remapped = hot_locals.remap_local(idx as u32);
            match remapped {
                0 if hot_slot_enabled(hot_locals, config, 0) => LoweredIrOpKind::LocalSetHot { reg: 0 },
                1 if hot_slot_enabled(hot_locals, config, 1) => LoweredIrOpKind::LocalSetHot { reg: 1 },
                2 if hot_slot_enabled(hot_locals, config, 2) => LoweredIrOpKind::LocalSetHot { reg: 2 },
                _ => LoweredIrOpKind::LocalSetFrame { idx: remapped as u16 },
            }
        }
        SemanticOpKind::LocalTee { idx } => {
            let remapped = hot_locals.remap_local(idx as u32);
            match remapped {
                0 if hot_slot_enabled(hot_locals, config, 0) => LoweredIrOpKind::LocalTeeHot { reg: 0 },
                1 if hot_slot_enabled(hot_locals, config, 1) => LoweredIrOpKind::LocalTeeHot { reg: 1 },
                2 if hot_slot_enabled(hot_locals, config, 2) => LoweredIrOpKind::LocalTeeHot { reg: 2 },
                _ => LoweredIrOpKind::LocalTeeFrame { idx: remapped as u16 },
            }
        }
        SemanticOpKind::CacheSpill { slot, count } => LoweredIrOpKind::Spill { slot, count },
        SemanticOpKind::CacheFill { slot, count } => LoweredIrOpKind::Fill { slot, count },
        SemanticOpKind::Br { stack_drop, arity } => LoweredIrOpKind::Br {
            stack_drop,
            arity,
            height: pre_height,
            operand_base_offset: config.operand_base_offset(frame_size) as u32,
        },
        SemanticOpKind::BrIf { stack_drop, arity } => {
            if stack_drop == 0 && arity == 0 {
                LoweredIrOpKind::BrIfSimple
            } else {
                LoweredIrOpKind::BrIf {
                    stack_drop,
                    arity,
                    height: pre_height,
                    operand_base_offset: config.operand_base_offset(frame_size) as u32,
                }
            }
        }
        SemanticOpKind::BrTable { entries } => LoweredIrOpKind::BrTable {
            entries,
            entry_count: 0,
            data_slot_count: 0,
            height: pre_height,
            operand_base_offset: config.operand_base_offset(frame_size) as u32,
        },
        SemanticOpKind::CallExternal { func_idx, delta } => LoweredIrOpKind::CallExternal { func_idx, delta },
        SemanticOpKind::CallInternal { callee, delta } => LoweredIrOpKind::CallInternal { callee, delta },
        SemanticOpKind::CallIndirect { type_idx, table_idx, delta } => LoweredIrOpKind::CallIndirect {
            type_idx,
            table_idx,
            delta,
            operand_base_offset: config.operand_base_offset(frame_size) as u32,
            height: pre_height,
        },
        SemanticOpKind::ReturnVoid => LoweredIrOpKind::ReturnVoid {
            frame_size: frame_size as u16,
        },
        SemanticOpKind::ReturnOne => LoweredIrOpKind::ReturnOne {
            frame_size: frame_size as u16,
            operand_base_offset: config.operand_base_offset(frame_size) as u32,
            height: pre_height,
        },
        SemanticOpKind::Return { arity } => LoweredIrOpKind::Return {
            arity,
            frame_size: frame_size as u16,
            operand_base_offset: config.operand_base_offset(frame_size) as u32,
            height: pre_height,
        },
    }
}

pub fn lower_to_lowered_ir(
    ops: Vec<SemanticOp>,
    hot_locals: HotLocalPlan,
    config: CompileConfig,
    frame_size: usize,
) -> Vec<LoweredIrOp> {
    let init = LoweredIrOp {
        kind: LoweredIrOpKind::InitLocals {
            k0: hot_locals.effective()[0].unwrap_or(0) as u16,
            k1: hot_locals.effective()[1].unwrap_or(1) as u16,
            k2: hot_locals.effective()[2].unwrap_or(2) as u16,
        },
        variant: 0,
        pre_height: 0,
        fallthrough: Some(OpIndex::new(1)),
        alt_target: None,
        has_target: false,
    };

    let mut lowered = Vec::with_capacity(ops.len() + 1);
    lowered.push(init);

    let mut index_map = Vec::with_capacity(ops.len());
    for op in &ops {
        index_map.push(lowered.len());
        lowered.push(LoweredIrOp {
            kind: lower_semantic_kind(op.kind.clone(), hot_locals, config, frame_size, op.pre_height),
            variant: op.variant,
            pre_height: op.pre_height,
            fallthrough: None,
            alt_target: None,
            has_target: op.has_target,
        });
    }

    let end_index = lowered.len();
    for (semantic_op, lowered_op) in ops.into_iter().zip(lowered.iter_mut().skip(1)) {
        lowered_op.fallthrough = remap_index(&index_map, end_index, semantic_op.fallthrough);
        lowered_op.alt_target = remap_index(&index_map, end_index, semantic_op.alt_target);
        remap_kind_targets(&mut lowered_op.kind, &index_map);
    }

    lowered
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use crate::vm::planner::FAST_COMPILE_CONFIG;

    fn test_hot_locals(effective: [Option<u32>; HOT_LOCAL_COUNT]) -> HotLocalPlan {
        HotLocalPlan::from_parts([None; HOT_LOCAL_COUNT], effective)
    }

    #[test]
    fn test_lower_places_local_ops_after_decode() {
        let hot_locals = test_hot_locals([Some(4), Some(7), None]);
        let lowered = lower_to_lowered_ir(
            vec![
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 4 },
                    variant: 1,
                    pre_height: 0,
                    fallthrough: Some(OpIndex::new(1)),
                    alt_target: None,
                    has_target: false,
                },
                SemanticOp {
                    kind: SemanticOpKind::LocalSet { idx: 0 },
                    variant: 2,
                    pre_height: 1,
                    fallthrough: Some(OpIndex::new(2)),
                    alt_target: None,
                    has_target: false,
                },
                SemanticOp {
                    kind: SemanticOpKind::LocalTee { idx: 7 },
                    variant: 2,
                    pre_height: 1,
                    fallthrough: None,
                    alt_target: None,
                    has_target: false,
                },
            ],
            hot_locals,
            FAST_COMPILE_CONFIG,
            0,
        );

        assert!(matches!(
            lowered[0].kind,
            LoweredIrOpKind::InitLocals { k0: 4, k1: 7, k2: 2 }
        ));
        assert!(matches!(lowered[1].kind, LoweredIrOpKind::LocalGetHot { reg: 0 }));
        assert!(matches!(lowered[2].kind, LoweredIrOpKind::LocalSetFrame { idx: 4 }));
        assert!(matches!(lowered[3].kind, LoweredIrOpKind::LocalTeeHot { reg: 1 }));
    }

    #[test]
    fn test_lower_shifts_control_flow_indices_after_prologue() {
        let lowered = lower_to_lowered_ir(
            vec![
                SemanticOp {
                    kind: SemanticOpKind::Core(CoreOpKind::If),
                    variant: 1,
                    pre_height: 1,
                    fallthrough: Some(OpIndex::new(1)),
                    alt_target: Some(OpIndex::new(2)),
                    has_target: true,
                },
                SemanticOp {
                    kind: SemanticOpKind::Core(CoreOpKind::Else),
                    variant: 0,
                    pre_height: 0,
                    fallthrough: Some(OpIndex::new(2)),
                    alt_target: Some(OpIndex::new(3)),
                    has_target: true,
                },
                SemanticOp {
                    kind: SemanticOpKind::BrTable {
                        entries: vec![
                            BrTableEntry {
                                target_idx: Some(OpIndex::new(0)),
                                stack_offset: 0,
                                arity: 0,
                            },
                            BrTableEntry {
                                target_idx: Some(OpIndex::new(2)),
                                stack_offset: 0,
                                arity: 0,
                            },
                        ],
                    },
                    variant: 1,
                    pre_height: 1,
                    fallthrough: None,
                    alt_target: None,
                    has_target: true,
                },
            ],
            test_hot_locals([None; HOT_LOCAL_COUNT]),
            FAST_COMPILE_CONFIG,
            0,
        );

        assert_eq!(lowered[1].fallthrough, Some(OpIndex::new(2)));
        assert_eq!(lowered[1].alt_target, Some(OpIndex::new(3)));
        assert_eq!(lowered[2].fallthrough, Some(OpIndex::new(3)));
        assert_eq!(lowered[2].alt_target, Some(OpIndex::new(4)));
        match &lowered[3].kind {
            LoweredIrOpKind::BrTable { entries, .. } => {
                assert_eq!(entries[0].target_idx, Some(OpIndex::new(1)));
                assert_eq!(entries[1].target_idx, Some(OpIndex::new(3)));
            }
            _ => panic!("expected br_table"),
        }
    }

    #[test]
    fn test_lower_converts_cache_markers_to_lowered_ops() {
        let lowered = lower_to_lowered_ir(
            vec![
                SemanticOp {
                    kind: SemanticOpKind::CacheSpill { slot: 8, count: 1 },
                    variant: 1,
                    pre_height: 4,
                    fallthrough: Some(OpIndex::new(1)),
                    alt_target: None,
                    has_target: false,
                },
                SemanticOp {
                    kind: SemanticOpKind::CacheFill { slot: 8, count: 1 },
                    variant: 4,
                    pre_height: 4,
                    fallthrough: None,
                    alt_target: None,
                    has_target: false,
                },
            ],
            test_hot_locals([None; HOT_LOCAL_COUNT]),
            FAST_COMPILE_CONFIG,
            0,
        );

        assert!(matches!(lowered[1].kind, LoweredIrOpKind::Spill { slot: 8, count: 1 }));
        assert!(matches!(lowered[2].kind, LoweredIrOpKind::Fill { slot: 8, count: 1 }));
        assert_eq!(lowered[1].fallthrough, Some(OpIndex::new(2)));
    }

    #[test]
    fn test_lower_materializes_branch_and_return_frame_metadata() {
        let lowered = lower_to_lowered_ir(
            vec![
                SemanticOp {
                    kind: SemanticOpKind::BrIf {
                        stack_drop: 2,
                        arity: 1,
                    },
                    variant: 3,
                    pre_height: 7,
                    fallthrough: Some(OpIndex::new(1)),
                    alt_target: Some(OpIndex::new(1)),
                    has_target: true,
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnOne,
                    variant: 0,
                    pre_height: 5,
                    fallthrough: None,
                    alt_target: None,
                    has_target: false,
                },
            ],
            test_hot_locals([None; HOT_LOCAL_COUNT]),
            FAST_COMPILE_CONFIG,
            6,
        );

        match &lowered[1].kind {
            LoweredIrOpKind::BrIf {
                stack_drop,
                arity,
                height,
                operand_base_offset,
            } => {
                assert_eq!((*stack_drop, *arity), (2, 1));
                assert_eq!(*height, 7);
                assert_eq!(*operand_base_offset, FAST_COMPILE_CONFIG.operand_base_offset(6) as u32);
            }
            _ => panic!("expected lowered br_if"),
        }

        match &lowered[2].kind {
            LoweredIrOpKind::ReturnOne {
                frame_size,
                operand_base_offset,
                height,
            } => {
                assert_eq!(*frame_size, 6);
                assert_eq!(*height, 5);
                assert_eq!(*operand_base_offset, FAST_COMPILE_CONFIG.operand_base_offset(6) as u32);
            }
            _ => panic!("expected lowered return_one"),
        }
    }

    #[test]
    fn test_lower_specializes_zero_fixup_br_if_to_simple_form() {
        let lowered = lower_to_lowered_ir(
            vec![SemanticOp {
                kind: SemanticOpKind::BrIf {
                    stack_drop: 0,
                    arity: 0,
                },
                variant: 2,
                pre_height: 3,
                fallthrough: None,
                alt_target: Some(OpIndex::new(0)),
                has_target: true,
            }],
            test_hot_locals([None; HOT_LOCAL_COUNT]),
            FAST_COMPILE_CONFIG,
            4,
        );

        assert!(matches!(lowered[1].kind, LoweredIrOpKind::BrIfSimple));
        assert_eq!(lowered[1].alt_target, Some(OpIndex::new(1)));
    }
}
