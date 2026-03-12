//! Lower semantic/planning data into CFG + SSA LIR.
//!
//! The lowering pipeline is intentionally split into:
//! 1. semantic/planned alignment
//! 2. block-boundary shaping
//! 3. straight-line body lowering
//! 4. terminator lowering

mod body;
mod boundary;
mod block;
mod cfg;
mod edge;
mod input;
mod ops;
mod state;
mod terminator;

use alloc::vec::Vec;

use crate::error::WasmError;
use crate::vm::{
    lir::{
        ir::{LirBlock, LirProgram},
        target::LirTarget,
    },
    plan::{config::PlanConfig, PlannedProgram},
    wasm::semantic_ir::SemanticProgram,
};

use self::{
    boundary::{compute_entry_heights, make_block_params},
    block::lower_block_range,
    cfg::{build_block_ranges, build_semantic_to_block_map, retain_reachable_blocks},
    input::map_semantic_to_planned,
    state::{BlockState, ValueAlloc},
};

pub fn lower_to_lir(
    semantic: &SemanticProgram,
    planned: &PlannedProgram,
    config: PlanConfig,
) -> Result<LirProgram, WasmError> {
    semantic.validate()?;
    // Planning config still validates the stack-aware input bundle, but it no
    // longer changes the shape of semantic LIR.
    planned.validate(semantic, config)?;

    if semantic.ops.is_empty() {
        return Ok(LirProgram {
            entry: LirTarget(0),
            blocks: Vec::new(),
        });
    }

    let mapped = map_semantic_to_planned(semantic, planned)?;
    if mapped.is_empty() {
        return Ok(LirProgram {
            entry: LirTarget(0),
            blocks: Vec::new(),
        });
    }

    let block_ranges = retain_reachable_blocks(semantic, build_block_ranges(semantic, planned));
    let semantic_to_block = build_semantic_to_block_map(mapped.len(), &block_ranges);
    let semantic_entry_heights = compute_entry_heights(semantic, &mapped);
    let mut values = ValueAlloc::default();

    let mut blocks = Vec::with_capacity(block_ranges.len());
    for (block_index, semantic_range) in block_ranges.into_iter().enumerate() {
        let entry_height = semantic_entry_heights[semantic_range.start];
        let params = make_block_params(entry_height, &mut values);
        let state = BlockState::from_params(&params);
        let block = lower_block_range(
            semantic_range.clone(),
            state,
            &mapped,
            planned.frame,
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
    };
    program.validate()?;
    Ok(program)
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::vm::{
        backend::{BackendConfig, BackendKind},
        lir::{ir::LirTerminator, runtime::LirRuntimeOp},
        plan::{
            config::PlanConfig,
            frame::FramePlanner,
            group::GroupPlan,
            PlannedOp, PlannedOpKind, PlannedProgram,
        },
        wasm::{
            common::{SemanticIndex, SemanticTarget},
            primitive_op::PrimitiveOpKind,
            semantic_ir::{SemanticOp, SemanticOpKind, SemanticProgram},
        },
    };

    #[test]
    fn lowers_memory_grow_as_runtime_op() {
        let semantic = SemanticProgram {
            params: 0,
            results: 1,
            local_count: 0,
            max_stack_height: 1,
            ops: alloc::vec![
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 1 }),
                    next: Some(SemanticIndex::new(1)),
                    alt: None,
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::MemoryGrow { mem_idx: 0 }),
                    next: Some(SemanticIndex::new(2)),
                    alt: None,
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnOne,
                    next: None,
                    alt: None,
                },
            ],
        };
        let frame = FramePlanner::new(0).reserve_operands(1).0.finish();
        let planned = PlannedProgram {
            frame,
            hot_locals: None,
            ops: alloc::vec![
                PlannedOp {
                    kind: PlannedOpKind::Primitive(PrimitiveOpKind::I32Const { value: 1 }),
                    rotation: Default::default(),
                    height: 0,
                    alt: None,
                },
                PlannedOp {
                    kind: PlannedOpKind::Primitive(PrimitiveOpKind::MemoryGrow { mem_idx: 0 }),
                    rotation: Default::default(),
                    height: 1,
                    alt: None,
                },
                PlannedOp {
                    kind: PlannedOpKind::Return { results: None },
                    rotation: Default::default(),
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
                BackendConfig {
                    ctx_register_count: 1,
                    fp_register_count: 1,
                    tmp_register_count: 1,
                    hot_local_count: 0,
                    tos_register_count: 0,
                },
            ),
        )
        .expect("lowered");
        assert!(matches!(
            lir.blocks[0].ops[1].kind,
            crate::vm::lir::ir::LirInstKind::Runtime {
                op: LirRuntimeOp::MemoryGrow { mem_idx: 0 },
                ..
            }
        ));
        assert!(matches!(lir.blocks[0].terminator, LirTerminator::Return { .. }));
    }

    #[test]
    fn computes_else_block_entry_height_from_semantic_stack_shape() {
        let semantic_ops = alloc::vec![
            SemanticOp {
                kind: SemanticOpKind::If {
                    params: 0,
                    results: 1,
                },
                next: Some(SemanticIndex::new(1)),
                alt: Some(SemanticTarget::new(1)),
            },
            SemanticOp {
                kind: SemanticOpKind::Else,
                next: Some(SemanticIndex::new(2)),
                alt: Some(SemanticTarget::new(2)),
            },
            SemanticOp {
                kind: SemanticOpKind::End,
                next: None,
                alt: None,
            },
        ];
        let planned = [
            PlannedOp {
                kind: PlannedOpKind::Marker(crate::vm::plan::PlannedMarkerKind::If {
                    params: 0,
                    results: 1,
                }),
                rotation: Default::default(),
                height: 1,
                alt: Some(SemanticTarget::new(1)),
            },
            PlannedOp {
                kind: PlannedOpKind::Marker(crate::vm::plan::PlannedMarkerKind::Else),
                rotation: Default::default(),
                height: 0,
                alt: Some(SemanticTarget::new(2)),
            },
            PlannedOp {
                kind: PlannedOpKind::Marker(crate::vm::plan::PlannedMarkerKind::End),
                rotation: Default::default(),
                height: 1,
                alt: None,
            },
        ];
        let mapped = semantic_ops
            .iter()
            .zip(planned.iter())
            .map(|(semantic, planned)| input::SemanticPlannedOp { semantic, planned })
            .collect::<Vec<_>>();

        assert_eq!(
            compute_entry_heights(
                &SemanticProgram {
                    params: 0,
                    results: 1,
                    local_count: 0,
                    max_stack_height: 1,
                    ops: semantic_ops,
                },
                &mapped,
            )[1],
            1
        );
    }
}
