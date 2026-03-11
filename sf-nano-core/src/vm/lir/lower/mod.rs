//! Lower semantic/planning data into CFG + SSA LIR.

mod block;
mod cfg;
mod edge;
mod input;
mod ops;
mod stack;
mod state;

use alloc::vec::Vec;

use crate::error::WasmError;
use crate::vm::{
    lir::{
        ir::{LirAbi, LirBlock, LirProgram},
        target::LirTarget,
    },
    plan::{config::PlanConfig, PlannedProgram},
    wasm::semantic_ir::SemanticProgram,
};

use self::{
    block::lower_block_range,
    cfg::{build_block_ranges, build_semantic_to_block_map, retain_reachable_blocks},
    input::map_semantic_to_planned,
    state::{make_block_params, BlockState, ValueAlloc},
};

pub fn lower_to_lir(
    semantic: &SemanticProgram,
    planned: &PlannedProgram,
    config: PlanConfig,
) -> Result<LirProgram, WasmError> {
    semantic.validate()?;
    planned.validate(semantic, config)?;

    if semantic.ops.is_empty() {
        return Ok(LirProgram {
            entry: LirTarget(0),
            blocks: Vec::new(),
            abi: LirAbi {
                tos_register_count: config.tos_register_count,
                hot_local_count: config.hot_local_count,
            },
            hot_locals: planned.hot_locals.clone(),
        });
    }

    let mapped = map_semantic_to_planned(semantic, planned)?;
    if mapped.is_empty() {
        return Ok(LirProgram {
            entry: LirTarget(0),
            blocks: Vec::new(),
            abi: LirAbi {
                tos_register_count: config.tos_register_count,
                hot_local_count: config.hot_local_count,
            },
            hot_locals: planned.hot_locals.clone(),
        });
    }

    let block_ranges = retain_reachable_blocks(semantic, build_block_ranges(semantic, planned));
    let semantic_to_block = build_semantic_to_block_map(mapped.len(), &block_ranges);
    let semantic_entry_heights = compute_semantic_entry_heights(semantic, &mapped);
    let mut values = ValueAlloc::default();

    let mut blocks = Vec::with_capacity(block_ranges.len());
    for (block_index, semantic_range) in block_ranges.into_iter().enumerate() {
        let entry_height = semantic_entry_heights[semantic_range.start];
        let params = make_block_params(entry_height, config.tos_register_count, &mut values);
        let state = BlockState::from_params(entry_height, &params, planned.frame);
        let block = lower_block_range(
            semantic_range.clone(),
            state,
            &mapped,
            planned,
            config,
            &semantic_to_block,
            &mut values,
        )?;
        blocks.push(LirBlock {
            id: LirTarget(block_index as u32),
            params,
            ops: block.ops,
            terminator: block.terminator,
        });
    }

    let program = LirProgram {
        entry: semantic_to_block[0],
        blocks,
        abi: LirAbi {
            tos_register_count: config.tos_register_count,
            hot_local_count: config.hot_local_count,
        },
        hot_locals: planned.hot_locals.clone(),
    };
    program.validate()?;
    Ok(program)
}

#[derive(Clone, Copy, Debug)]
struct ControlFrame {
    start_height: u16,
    params: u16,
    results: u16,
}

fn compute_semantic_entry_heights(
    semantic: &SemanticProgram,
    mapped: &[input::SemanticPlannedOp<'_>],
) -> Vec<u16> {
    let mut heights = Vec::with_capacity(mapped.len());
    let mut control = alloc::vec![ControlFrame {
        start_height: 0,
        params: 0,
        results: semantic.results,
    }];

    for op in mapped {
        let entry_height = match &op.semantic.kind {
            crate::vm::wasm::semantic_ir::SemanticOpKind::Else => {
                control.last().map_or(0, |frame| frame.start_height + frame.params)
            }
            crate::vm::wasm::semantic_ir::SemanticOpKind::End => {
                control.last().map_or(0, |frame| frame.start_height + frame.results)
            }
            _ => op.planned.height,
        };
        heights.push(entry_height);

        match &op.semantic.kind {
            crate::vm::wasm::semantic_ir::SemanticOpKind::Block { params, results }
            | crate::vm::wasm::semantic_ir::SemanticOpKind::Loop { params, results } => {
                control.push(ControlFrame {
                    start_height: entry_height.saturating_sub(*params),
                    params: *params,
                    results: *results,
                });
            }
            crate::vm::wasm::semantic_ir::SemanticOpKind::If { params, results } => {
                control.push(ControlFrame {
                    start_height: entry_height.saturating_sub(1).saturating_sub(*params),
                    params: *params,
                    results: *results,
                });
            }
            crate::vm::wasm::semantic_ir::SemanticOpKind::End => {
                let _ = control.pop();
            }
            _ => {}
        }
    }

    heights
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::{
        backend::BackendKind,
        lir::{
            ir::{LirInstKind, LirTerminator},
            leaf::LirLeafOp,
            target::LirTarget,
        },
        plan::{
            config::PlanConfig,
            frame::{FramePlanner, FrameSpan},
            group::GroupPlan,
            hot_local::HotLocalPlan,
            tos::{SpillArtifact, TosRotation},
            PlannedHotLocalInit, PlannedLocal, PlannedOp, PlannedOpKind, PlannedProgram,
        },
        wasm::{
            common::SemanticIndex,
            primitive_op::PrimitiveOpKind,
            semantic_ir::{SemanticOp, SemanticOpKind, SemanticProgram},
        },
    };

    #[test]
    fn lowers_straight_line_primitive_and_hot_local_into_cfg_lir() {
        let semantic = SemanticProgram {
            params: 0,
            results: 1,
            local_count: 1,
            max_stack_height: 2,
            ops: alloc::vec![
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 0 },
                    next: Some(SemanticIndex::new(1)),
                    alt: None,
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 1 }),
                    next: Some(SemanticIndex::new(2)),
                    alt: None,
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Add),
                    next: Some(SemanticIndex::new(3)),
                    alt: None,
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnOne,
                    next: None,
                    alt: None,
                },
            ],
        };

        let frame = FramePlanner::new(1).reserve_operands(2).0.finish();
        let planned = PlannedProgram {
            frame,
            hot_locals: Some(HotLocalPlan::new(alloc::vec![Some(0)])),
            ops: alloc::vec![
                PlannedOp {
                    kind: PlannedOpKind::InitHotLocals {
                        inits: alloc::vec![PlannedHotLocalInit {
                            reg: 0,
                            frame_slot: frame.local_slot(0),
                        }],
                    },
                    rotation: TosRotation::new(0),
                    height: 0,
                    alt: None,
                },
                PlannedOp {
                    kind: PlannedOpKind::LocalGet {
                        local: PlannedLocal::Hot(0),
                    },
                    rotation: TosRotation::new(0),
                    height: 0,
                    alt: None,
                },
                PlannedOp {
                    kind: PlannedOpKind::Primitive(PrimitiveOpKind::I32Const { value: 1 }),
                    rotation: TosRotation::new(0),
                    height: 1,
                    alt: None,
                },
                PlannedOp {
                    kind: PlannedOpKind::Primitive(PrimitiveOpKind::I32Add),
                    rotation: TosRotation::new(0),
                    height: 2,
                    alt: None,
                },
                PlannedOp {
                    kind: PlannedOpKind::Return { results: None },
                    rotation: TosRotation::new(0),
                    height: 1,
                    alt: None,
                },
            ],
            groups: GroupPlan::default(),
        };

        let lir = lower_to_lir(
            &semantic,
            &planned,
            PlanConfig::for_backend(
                BackendKind::Base,
                crate::vm::backend::BackendConfig {
                    ctx_register_count: 1,
                    fp_register_count: 1,
                    tmp_register_count: 4,
                    hot_local_count: 1,
                    tos_register_count: 4,
                },
            ),
        )
        .expect("lowered");

        assert_eq!(lir.abi.hot_local_count, 1);
        assert_eq!(lir.abi.tos_register_count, 4);
        assert_eq!(lir.blocks.len(), 1);
        assert_eq!(lir.entry, LirTarget(0));

        let block0 = &lir.blocks[0];
        assert_eq!(block0.params.tos.len(), 0);
        assert!(matches!(
            block0.ops[0].kind,
            LirInstKind::ReadFrameLocal { .. }
        ));
        assert!(matches!(
            block0.ops[1].kind,
            LirInstKind::WriteHotLocal { reg: 0, .. }
        ));
        assert!(matches!(
            block0.ops[2].kind,
            LirInstKind::ReadHotLocal { reg: 0, .. }
        ));
        assert!(matches!(
            block0.ops[4].kind,
            LirInstKind::Leaf {
                op: LirLeafOp::I32Add,
                ..
            }
        ));
        assert!(matches!(block0.terminator, LirTerminator::Return { .. }));
        assert_eq!(
            lir.hot_locals
                .as_ref()
                .expect("hot locals present")
                .mapping(),
            &[Some(0)]
        );
    }

    #[test]
    fn lowers_explicit_planned_spill_artifacts_into_operand_slot_effects() {
        let semantic = SemanticProgram {
            params: 0,
            results: 2,
            local_count: 0,
            max_stack_height: 2,
            ops: alloc::vec![
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 10 }),
                    next: Some(SemanticIndex::new(1)),
                    alt: None,
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 20 }),
                    next: Some(SemanticIndex::new(2)),
                    alt: None,
                },
                SemanticOp {
                    kind: SemanticOpKind::Return { arity: 2 },
                    next: None,
                    alt: None,
                },
            ],
        };

        let frame = FramePlanner::new(0).reserve_operands(2).0.finish();
        let planned = PlannedProgram {
            frame,
            hot_locals: None,
            ops: alloc::vec![
                PlannedOp {
                    kind: PlannedOpKind::Primitive(PrimitiveOpKind::I32Const { value: 10 }),
                    rotation: TosRotation::new(0),
                    height: 0,
                    alt: None,
                },
                PlannedOp {
                    kind: PlannedOpKind::Spill(SpillArtifact::spill(FrameSpan::single(
                        frame.operand_slot(0),
                    ))),
                    rotation: TosRotation::new(0),
                    height: 1,
                    alt: None,
                },
                PlannedOp {
                    kind: PlannedOpKind::Primitive(PrimitiveOpKind::I32Const { value: 20 }),
                    rotation: TosRotation::new(0),
                    height: 1,
                    alt: None,
                },
                PlannedOp {
                    kind: PlannedOpKind::Return {
                        results: Some(FrameSpan::new(frame.operand_slot(0), 2)),
                    },
                    rotation: TosRotation::new(0),
                    height: 2,
                    alt: None,
                },
            ],
            groups: GroupPlan::default(),
        };

        let lir = lower_to_lir(
            &semantic,
            &planned,
            PlanConfig::for_backend(
                BackendKind::Base,
                crate::vm::backend::BackendConfig {
                    ctx_register_count: 1,
                    fp_register_count: 1,
                    tmp_register_count: 4,
                    hot_local_count: 0,
                    tos_register_count: 1,
                },
            ),
        )
        .expect("lowered");

        let block0 = &lir.blocks[0];
        assert!(matches!(
            block0.ops[0].kind,
            LirInstKind::Leaf {
                op: LirLeafOp::I32Const { value: 10 },
                ..
            }
        ));
        assert!(matches!(
            block0.ops[1].kind,
            LirInstKind::WriteOperandSlot { .. }
        ));
        assert!(matches!(
            block0.ops[2].kind,
            LirInstKind::Leaf {
                op: LirLeafOp::I32Const { value: 20 },
                ..
            }
        ));
        assert!(matches!(
            block0.ops[3].kind,
            LirInstKind::ReadOperandSlot { .. }
        ));
        assert!(matches!(
            block0.terminator,
            LirTerminator::Return { ref values } if values.len() == 2
        ));
    }
}
