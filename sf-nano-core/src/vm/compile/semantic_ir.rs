//! Semantic Wasm decode IR before backend-local placement.
//!
//! This layer is intentionally small for now. It keeps control-flow metadata,
//! variants, and stack-management ops from the current lowering pipeline, but
//! leaves local placement and the `InitLocals` prologue to a separate lowering
//! step driven by backend compile planning.

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
}

impl From<LoweredIrOpKind> for SemanticOpKind {
    #[inline]
    fn from(kind: LoweredIrOpKind) -> Self {
        Self::Lowered(kind)
    }
}

#[inline]
fn shifted_index(index: Option<OpIndex>) -> Option<OpIndex> {
    index.map(|idx| OpIndex::new(idx.as_usize() + 1))
}

#[inline]
fn shift_kind_targets(kind: &mut LoweredIrOpKind) {
    if let LoweredIrOpKind::BrTable { entries, .. } = kind {
        for entry in entries.iter_mut() {
            if let Some(target_idx) = entry.target_idx {
                entry.target_idx = Some(OpIndex::new(target_idx.as_usize() + 1));
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
) -> LoweredIrOpKind {
    match kind {
        SemanticOpKind::Lowered(mut lowered) => {
            shift_kind_targets(&mut lowered);
            lowered
        }
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

    for op in ops {
        lowered.push(LoweredIrOp {
            kind: lower_local_kind(op.kind, hot_locals, config),
            variant: op.variant,
            pre_height: op.pre_height,
            fallthrough: shifted_index(op.fallthrough),
            alt_target: shifted_index(op.alt_target),
            has_target: op.has_target,
        });
    }

    lowered
}

pub fn stack_effect(kind: &SemanticOpKind) -> (u8, u8) {
    match kind {
        SemanticOpKind::Lowered(kind) => lowered_ir::stack_effect(kind),
        SemanticOpKind::LocalGet { .. } => (0, 1),
        SemanticOpKind::LocalSet { .. } => (1, 0),
        SemanticOpKind::LocalTee { .. } => (0, 0),
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
}
