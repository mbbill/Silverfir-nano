//! Semantic Wasm decode IR before backend-local placement.
//!
//! This layer is intentionally incremental. It keeps control-flow metadata,
//! variants, and abstract TOS-cache management markers from the current
//! lowering pipeline, but leaves local placement, `InitLocals`, and concrete
//! lowered spill/fill opcodes to a separate lowering step driven by backend
//! compile planning.

use alloc::vec::Vec;

use super::{
    CompileConfig, HotLocalPlan, HOT_LOCAL_COUNT,
    lowered_ir::{self, IrOp as LoweredIrOp, IrOpKind as LoweredIrOpKind, OpIndex},
};

#[derive(Debug, Clone)]
pub struct SemanticOp {
    pub kind: SemanticOpKind,
    pub variant: u8,
    pub pre_height: u16,
    pub fallthrough: Option<OpIndex>,
    pub alt_target: Option<OpIndex>,
    pub has_target: bool,
}

#[derive(Debug, Clone)]
pub enum SemanticOpKind {
    Lowered(LoweredIrOpKind),
    LocalGet { idx: u16 },
    LocalSet { idx: u16 },
    LocalTee { idx: u16 },
    CacheSpill { slot: u16, count: u8 },
    CacheFill { slot: u16, count: u8 },
}

impl From<LoweredIrOpKind> for SemanticOpKind {
    #[inline]
    fn from(kind: LoweredIrOpKind) -> Self {
        Self::Lowered(kind)
    }
}

#[inline]
fn remap_index(index_map: &[usize], index: Option<OpIndex>) -> Option<OpIndex> {
    index.map(|idx| OpIndex::new(index_map[idx.as_usize()]))
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

fn lower_local_kind(
    kind: SemanticOpKind,
    hot_locals: HotLocalPlan,
    config: CompileConfig,
) -> Option<LoweredIrOpKind> {
    match kind {
        SemanticOpKind::Lowered(lowered) => Some(lowered),
        SemanticOpKind::LocalGet { idx } => {
            let remapped = hot_locals.remap_local(idx as u32);
            match remapped {
                0 if hot_slot_enabled(hot_locals, config, 0) => Some(LoweredIrOpKind::LocalGetHot { reg: 0 }),
                1 if hot_slot_enabled(hot_locals, config, 1) => Some(LoweredIrOpKind::LocalGetHot { reg: 1 }),
                2 if hot_slot_enabled(hot_locals, config, 2) => Some(LoweredIrOpKind::LocalGetHot { reg: 2 }),
                _ => Some(LoweredIrOpKind::LocalGetFrame { idx: remapped as u16 }),
            }
        }
        SemanticOpKind::LocalSet { idx } => {
            let remapped = hot_locals.remap_local(idx as u32);
            match remapped {
                0 if hot_slot_enabled(hot_locals, config, 0) => Some(LoweredIrOpKind::LocalSetHot { reg: 0 }),
                1 if hot_slot_enabled(hot_locals, config, 1) => Some(LoweredIrOpKind::LocalSetHot { reg: 1 }),
                2 if hot_slot_enabled(hot_locals, config, 2) => Some(LoweredIrOpKind::LocalSetHot { reg: 2 }),
                _ => Some(LoweredIrOpKind::LocalSetFrame { idx: remapped as u16 }),
            }
        }
        SemanticOpKind::LocalTee { idx } => {
            let remapped = hot_locals.remap_local(idx as u32);
            match remapped {
                0 if hot_slot_enabled(hot_locals, config, 0) => Some(LoweredIrOpKind::LocalTeeHot { reg: 0 }),
                1 if hot_slot_enabled(hot_locals, config, 1) => Some(LoweredIrOpKind::LocalTeeHot { reg: 1 }),
                2 if hot_slot_enabled(hot_locals, config, 2) => Some(LoweredIrOpKind::LocalTeeHot { reg: 2 }),
                _ => Some(LoweredIrOpKind::LocalTeeFrame { idx: remapped as u16 }),
            }
        }
        SemanticOpKind::CacheSpill { slot, count } => Some(LoweredIrOpKind::Spill { slot, count }),
        SemanticOpKind::CacheFill { slot, count } => Some(LoweredIrOpKind::Fill { slot, count }),
    }
}

pub fn lower_to_lowered_ir(
    ops: Vec<SemanticOp>,
    hot_locals: HotLocalPlan,
    config: CompileConfig,
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
        if let Some(kind) = lower_local_kind(op.kind.clone(), hot_locals, config) {
            lowered.push(LoweredIrOp {
                kind,
                variant: op.variant,
                pre_height: op.pre_height,
                fallthrough: None,
                alt_target: None,
                has_target: op.has_target,
            });
        }
    }

    for (semantic_op, lowered_op) in ops.into_iter().zip(lowered.iter_mut().skip(1)) {
        lowered_op.fallthrough = remap_index(&index_map, semantic_op.fallthrough);
        lowered_op.alt_target = remap_index(&index_map, semantic_op.alt_target);
        remap_kind_targets(&mut lowered_op.kind, &index_map);
    }

    lowered
}

pub fn stack_effect(kind: &SemanticOpKind) -> (u8, u8) {
    match kind {
        SemanticOpKind::Lowered(kind) => lowered_ir::stack_effect(kind),
        SemanticOpKind::LocalGet { .. } => (0, 1),
        SemanticOpKind::LocalSet { .. } => (1, 0),
        SemanticOpKind::LocalTee { .. } => (0, 0),
        SemanticOpKind::CacheSpill { .. } | SemanticOpKind::CacheFill { .. } => (0, 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use crate::vm::compile::FAST_COMPILE_CONFIG;

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
                    kind: SemanticOpKind::Lowered(LoweredIrOpKind::If),
                    variant: 1,
                    pre_height: 1,
                    fallthrough: Some(OpIndex::new(1)),
                    alt_target: Some(OpIndex::new(2)),
                    has_target: true,
                },
                SemanticOp {
                    kind: SemanticOpKind::Lowered(LoweredIrOpKind::Else),
                    variant: 0,
                    pre_height: 0,
                    fallthrough: Some(OpIndex::new(2)),
                    alt_target: Some(OpIndex::new(3)),
                    has_target: true,
                },
                SemanticOp {
                    kind: SemanticOpKind::Lowered(LoweredIrOpKind::BrTable {
                        entries: vec![
                            lowered_ir::BrTableEntry {
                                target_idx: Some(OpIndex::new(0)),
                                stack_offset: 0,
                                arity: 0,
                            },
                            lowered_ir::BrTableEntry {
                                target_idx: Some(OpIndex::new(2)),
                                stack_offset: 0,
                                arity: 0,
                            },
                        ],
                        entry_count: 2,
                        data_slot_count: 0,
                        height: 1,
                        operand_base_offset: 0,
                    }),
                    variant: 1,
                    pre_height: 1,
                    fallthrough: None,
                    alt_target: None,
                    has_target: true,
                },
            ],
            test_hot_locals([None; HOT_LOCAL_COUNT]),
            FAST_COMPILE_CONFIG,
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
        );

        assert!(matches!(lowered[1].kind, LoweredIrOpKind::Spill { slot: 8, count: 1 }));
        assert!(matches!(lowered[2].kind, LoweredIrOpKind::Fill { slot: 8, count: 1 }));
        assert_eq!(lowered[1].fallthrough, Some(OpIndex::new(2)));
    }
}
