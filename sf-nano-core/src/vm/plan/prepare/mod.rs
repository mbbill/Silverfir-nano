//! Direct preparation from decoded Wasm semantics into prepared LIR.

mod block;
mod cfg;
mod edge;
mod optimize;
mod ops;
mod state;
mod steps;
mod terminator;

use alloc::vec::Vec;

use crate::{
    error::WasmError,
    vm::{
        lir::{
            ir::{LirBlock, LirProgram},
            target::LirTarget,
            validate::validate_program,
        },
        wasm::semantic_ir::SemanticProgram,
    },
};

use self::{
    block::lower_block_range,
    cfg::{build_block_ranges, build_semantic_to_block_map, retain_reachable_blocks},
    state::{BlockState, EntryState, ValueAlloc},
    steps::prepare_semantic_ops,
};
use super::{
    analyze_local_cache_prefs,
    config::PlanConfig,
    frame::{plan_frame_layout, FrameLayoutPlan},
};

/// Preparation input bundle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrepareInput {
    pub config: PlanConfig,
}

/// Shared frontend output consumed by interpreter and native backends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedFunction {
    pub frame: FrameLayoutPlan,
    pub lir: LirProgram,
}

pub fn prepare_function(
    input: PrepareInput,
    semantic: &SemanticProgram,
) -> Result<PreparedFunction, WasmError> {
    semantic.validate()?;

    let frame = plan_frame_layout(
        semantic.local_count,
        semantic.max_stack_height,
        input.config.call_scratch_slots,
    );
    let local_cache = analyze_local_cache_prefs(semantic, input.config.cached_locals, frame);
    let prepared = prepare_semantic_ops(semantic, frame, input.config)?;

    if semantic.ops.is_empty() {
        return Ok(PreparedFunction {
            frame,
            lir: LirProgram {
                entry: LirTarget(0),
                local_cache,
                blocks: Vec::new(),
            },
        });
    }

    let block_ranges = retain_reachable_blocks(semantic, build_block_ranges(semantic));
    let semantic_to_block = build_semantic_to_block_map(semantic.ops.len(), &block_ranges);
    let mut values = ValueAlloc::default();
    let block_params = block_ranges
        .iter()
        .map(|range| make_block_params(prepared.entry_states[range.start], &mut values))
        .collect::<Vec<_>>();

    let original_block_count = block_ranges.len();
    let mut blocks = Vec::with_capacity(block_ranges.len());
    let mut extra_blocks = Vec::new();
    for (block_index, semantic_range) in block_ranges.into_iter().enumerate() {
        let params = block_params[block_index].clone();
        let state = BlockState::from_entry(
            prepared.entry_states[semantic_range.start],
            &params,
            input.config.lir_lanes,
        )?;
        let block = lower_block_range(
            semantic_range.clone(),
            state,
            &prepared.ops,
            frame,
            &semantic_to_block,
            &block_params,
            &prepared.entry_states,
            &mut values,
            original_block_count,
            extra_blocks.len(),
        )?;
        blocks.push(LirBlock {
            id: LirTarget(block_index as u32),
            params,
            ops: block.ops,
            terminator: block.terminator,
        });
        extra_blocks.extend(block.extra_blocks);
    }
    blocks.extend(extra_blocks);

    let lir = LirProgram {
        entry: semantic_to_block[0],
        local_cache,
        blocks,
    };
    let mut lir = lir;
    optimize::optimize_lir(&mut lir, frame);
    validate_program(&lir)?;

    Ok(PreparedFunction { frame, lir })
}

#[inline]
fn make_block_params(
    entry: EntryState,
    values: &mut ValueAlloc,
) -> alloc::vec::Vec<crate::vm::lir::ir::LirValue> {
    values.many(entry.live_value_count() as usize)
}

#[cfg(test)]
mod tests {
    use std::eprintln;

    use crate::vm::{
        lir::ir::{LirBoundaryOp, LirInstKind},
        plan::{
            config::PlanConfig,
            prepare::{prepare_function, PrepareInput},
        },
        wasm::{
            primitive_op::PrimitiveOpKind,
            semantic_ir::{SemanticOp, SemanticOpKind, SemanticProgram},
        },
    };

    #[test]
    fn prepares_memory_copy_as_boundary_op() {
        let semantic = SemanticProgram {
            params: 0,
            results: 0,
            local_count: 0,
            max_stack_height: 3,
            ops: alloc::vec![
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 1 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 2 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 3 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::MemoryCopy {
                        imm0: 0,
                        imm1: 1,
                    }),
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnVoid,
                },
            ],
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 3),
            },
            &semantic,
        )
        .expect("memory.copy preparation should succeed");

        assert!(prepared
            .lir
            .blocks
            .iter()
            .any(|block| block.ops.iter().any(|inst| matches!(
                inst.kind,
                LirInstKind::Boundary(LirBoundaryOp::MemoryCopy {
                    dst_mem_idx: 0,
                    src_mem_idx: 1,
                    ..
                })
            ))));
    }

    #[test]
    fn prepares_table_fill_as_boundary_op() {
        let semantic = SemanticProgram {
            params: 0,
            results: 0,
            local_count: 0,
            max_stack_height: 3,
            ops: alloc::vec![
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 1 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::RefNull),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 3 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::TableFill {
                        imm0: 2,
                        imm1: 0
                    }),
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnVoid,
                },
            ],
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 3),
            },
            &semantic,
        )
        .expect("table.fill preparation should succeed");

        assert!(prepared
            .lir
            .blocks
            .iter()
            .any(|block| block.ops.iter().any(|inst| matches!(
                inst.kind,
                LirInstKind::Boundary(LirBoundaryOp::TableFill { table_idx: 2, .. })
            ))));
    }

    #[test]
    fn prepares_memory_init_with_data_and_memory_indices_in_spec_order() {
        let semantic = SemanticProgram {
            params: 0,
            results: 0,
            local_count: 0,
            max_stack_height: 3,
            ops: alloc::vec![
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 1 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 2 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 3 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::MemoryInit {
                        imm0: 4,
                        imm1: 7,
                    }),
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnVoid,
                },
            ],
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 3),
            },
            &semantic,
        )
        .expect("memory.init preparation should succeed");

        assert!(prepared
            .lir
            .blocks
            .iter()
            .any(|block| block.ops.iter().any(|inst| matches!(
                inst.kind,
                LirInstKind::Boundary(LirBoundaryOp::MemoryInit {
                    data_idx: 7,
                    mem_idx: 4,
                    ..
                })
            ))));
    }

    #[test]
    fn prepares_table_init_with_element_and_table_indices_in_spec_order() {
        let semantic = SemanticProgram {
            params: 0,
            results: 0,
            local_count: 0,
            max_stack_height: 3,
            ops: alloc::vec![
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 1 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 2 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 3 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::TableInit {
                        imm0: 5,
                        imm1: 8,
                    }),
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnVoid,
                },
            ],
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 3),
            },
            &semantic,
        )
        .expect("table.init preparation should succeed");

        assert!(prepared
            .lir
            .blocks
            .iter()
            .any(|block| block.ops.iter().any(|inst| matches!(
                inst.kind,
                LirInstKind::Boundary(LirBoundaryOp::TableInit {
                    elem_idx: 8,
                    table_idx: 5,
                    ..
                })
            ))));
    }

    #[test]
    fn merges_end_into_enclosing_block_for_empty_if() {
        let semantic = SemanticProgram {
            params: 1,
            results: 0,
            local_count: 1,
            max_stack_height: 1,
            ops: alloc::vec![
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 0 },
                },
                SemanticOp {
                    kind: SemanticOpKind::If {
                        params: 0,
                        results: 0,
                        else_target: crate::vm::wasm::common::SemanticTarget::new(2),
                    },
                },
                SemanticOp {
                    kind: SemanticOpKind::End,
                },
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 0 },
                },
                SemanticOp {
                    kind: SemanticOpKind::If {
                        params: 0,
                        results: 0,
                        else_target: crate::vm::wasm::common::SemanticTarget::new(5),
                    },
                },
                SemanticOp {
                    kind: SemanticOpKind::End,
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnVoid,
                },
            ],
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 3),
            },
            &semantic,
        )
        .expect("empty-if preparation should succeed");

        // End is no longer split into its own block — it merges with the
        // following code. Two empty-if sequences + return = 3 blocks
        // (entry, End+LocalGet+If, End+ReturnVoid).
        assert_eq!(prepared.lir.blocks.len(), 3);
    }

    #[test]
    fn prepares_result_if_without_transient_underflow() {
        let semantic = SemanticProgram {
            params: 1,
            results: 1,
            local_count: 1,
            max_stack_height: 1,
            ops: alloc::vec![
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 0 },
                },
                SemanticOp {
                    kind: SemanticOpKind::If {
                        params: 0,
                        results: 1,
                        else_target: crate::vm::wasm::common::SemanticTarget::new(4),
                    },
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 7 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Else {
                        end_target: crate::vm::wasm::common::SemanticTarget::new(6),
                    },
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 8 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::End,
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnOne,
                },
            ],
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 3),
            },
            &semantic,
        )
        .expect("result-if preparation should succeed");

        assert!(prepared.lir.blocks.iter().any(|block| matches!(
            block.terminator,
            crate::vm::lir::ir::LirTerminator::Return { .. }
        )));
    }

    #[test]
    fn prepares_br_if_with_block_result_payload() {
        let semantic = SemanticProgram {
            params: 1,
            results: 1,
            local_count: 1,
            max_stack_height: 2,
            ops: alloc::vec![
                SemanticOp {
                    kind: SemanticOpKind::Block {
                        params: 0,
                        results: 1,
                    },
                },
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 0 },
                },
                SemanticOp {
                    kind: SemanticOpKind::If {
                        params: 0,
                        results: 1,
                        else_target: crate::vm::wasm::common::SemanticTarget::new(5),
                    },
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 1 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Else {
                        end_target: crate::vm::wasm::common::SemanticTarget::new(7),
                    },
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 0 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::End,
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 2 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::BrIf {
                        stack_drop: 0,
                        arity: 1,
                        target: crate::vm::wasm::common::SemanticTarget::new(11),
                    },
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 3 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnOne,
                },
                SemanticOp {
                    kind: SemanticOpKind::End,
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnOne,
                },
            ],
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 3),
            },
            &semantic,
        )
        .expect("br_if block-result preparation should succeed");

        assert!(prepared.lir.blocks.iter().any(|block| matches!(
            block.terminator,
            crate::vm::lir::ir::LirTerminator::Branch { .. }
        )));
        let final_return = prepared
            .lir
            .blocks
            .iter()
            .find_map(|block| match block.terminator {
                crate::vm::lir::ir::LirTerminator::Return {
                    results: Some(span),
                } => Some(span),
                _ => None,
            })
            .expect("final return span");
        assert_eq!(final_return.start, prepared.frame.operand_slot(0));
        assert_eq!(final_return.count, 1);
    }

    #[test]
    fn prepares_if_with_block_param_and_result() {
        let semantic = SemanticProgram {
            params: 3,
            results: 1,
            local_count: 3,
            max_stack_height: 2,
            ops: alloc::vec![
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 0 },
                },
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 1 },
                },
                SemanticOp {
                    kind: SemanticOpKind::CallInternal {
                        callee: 0,
                        params: 2,
                        results: 2,
                    },
                },
                SemanticOp {
                    kind: SemanticOpKind::If {
                        params: 1,
                        results: 1,
                        else_target: crate::vm::wasm::common::SemanticTarget::new(7),
                    },
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::Drop),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I64Const { value: u64::MAX }),
                },
                SemanticOp {
                    kind: SemanticOpKind::End,
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnOne,
                },
            ],
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 3),
            },
            &semantic,
        )
        .expect("if param/result preparation should succeed");

        assert!(prepared.lir.blocks.iter().any(|block| matches!(
            block.terminator,
            crate::vm::lir::ir::LirTerminator::Return { .. }
        )));
    }

    #[test]
    fn prepares_if_param_passthrough_break_with_canonical_join_publish() {
        let semantic = SemanticProgram {
            params: 1,
            results: 1,
            local_count: 1,
            max_stack_height: 3,
            ops: alloc::vec![
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 1 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 2 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 0 },
                },
                SemanticOp {
                    kind: SemanticOpKind::If {
                        params: 2,
                        results: 2,
                        else_target: crate::vm::wasm::common::SemanticTarget::new(5),
                    },
                },
                SemanticOp {
                    kind: SemanticOpKind::Br {
                        stack_drop: 0,
                        arity: 2,
                        target: crate::vm::wasm::common::SemanticTarget::new(5),
                    },
                },
                SemanticOp {
                    kind: SemanticOpKind::End,
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Add),
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnOne,
                },
            ],
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 3),
            },
            &semantic,
        )
        .expect("if param passthrough break preparation should succeed");

        let if_block = prepared
            .lir
            .blocks
            .iter()
            .find(|block| {
                matches!(
                    block.terminator,
                    crate::vm::lir::ir::LirTerminator::Branch { .. }
                )
            })
            .expect("if block");
        let store_count = if_block
            .ops
            .iter()
            .filter(|inst| matches!(inst.kind, LirInstKind::StoreSlot { .. }))
            .count();

        assert!(
            store_count >= 2,
            "if join should publish live block values into canonical frame slots before branching to a canonical-only end block"
        );
    }

    #[test]
    fn prepares_unreachable_if_condition_without_phantom_result_growth() {
        let semantic = SemanticProgram {
            params: 0,
            results: 1,
            local_count: 0,
            max_stack_height: 2,
            ops: alloc::vec![
                SemanticOp {
                    kind: SemanticOpKind::Block {
                        params: 0,
                        results: 1,
                    },
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 2 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 0 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::BrTable {
                        entries: alloc::vec![crate::vm::wasm::common::BrTableEntry {
                            target: crate::vm::wasm::common::SemanticTarget::new(9),
                            stack_drop: 0,
                            arity: 1,
                        }],
                    },
                },
                SemanticOp {
                    kind: SemanticOpKind::If {
                        params: 0,
                        results: 1,
                        else_target: crate::vm::wasm::common::SemanticTarget::new(7),
                    },
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 0 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Else {
                        end_target: crate::vm::wasm::common::SemanticTarget::new(8),
                    },
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 1 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::End,
                },
                SemanticOp {
                    kind: SemanticOpKind::End,
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnOne,
                },
            ],
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 3),
            },
            &semantic,
        )
        .expect("unreachable folded-if preparation should succeed");

        assert!(prepared.lir.blocks.iter().any(|block| matches!(
            block.terminator,
            crate::vm::lir::ir::LirTerminator::Return { .. }
        )));
    }

    #[test]
    fn prepares_block_result_fallthrough_with_mixed_spilled_and_live_values() {
        let semantic = SemanticProgram {
            params: 0,
            results: 1,
            local_count: 0,
            max_stack_height: 3,
            ops: alloc::vec![
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 2 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Block {
                        params: 0,
                        results: 1,
                    },
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 1 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::End,
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 3 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::Select),
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnOne,
                },
            ],
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 3),
            },
            &semantic,
        )
        .expect("block-result fallthrough preparation should succeed");

        assert!(
            prepared.lir.blocks.iter().any(|block| {
                block.ops.iter().any(|inst| {
                    matches!(
                        inst.kind,
                        LirInstKind::StoreSlot { slot, .. }
                            if slot == prepared.frame.operand_slot(0)
                    )
                })
            }),
            "fallthrough from a block result must publish the older stack prefix to canonical slots before entering a mixed spill/live successor"
        );
    }

    #[test]
    fn prepares_block_result_used_as_select_operand_after_end() {
        let semantic = SemanticProgram {
            params: 0,
            results: 1,
            local_count: 0,
            max_stack_height: 3,
            ops: alloc::vec![
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 2 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 3 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Block {
                        params: 0,
                        results: 1,
                    },
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 1 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::End,
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::Select),
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnOne,
                },
            ],
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 3),
            },
            &semantic,
        )
        .expect("block result select preparation should succeed");

        assert!(prepared.lir.blocks.iter().any(|block| matches!(
            block.terminator,
            crate::vm::lir::ir::LirTerminator::Return { .. }
        )));
    }

    #[test]
    fn debug_prepares_nested_br_table_value_index_shape() {
        let semantic = SemanticProgram {
            params: 1,
            results: 1,
            local_count: 1,
            max_stack_height: 4,
            ops: alloc::vec![
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 1 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Block {
                        params: 0,
                        results: 1,
                    },
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 2 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::Drop),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 4 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Block {
                        params: 0,
                        results: 1,
                    },
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 8 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 0 },
                },
                SemanticOp {
                    kind: SemanticOpKind::BrIf {
                        stack_drop: 1,
                        arity: 1,
                        target: crate::vm::wasm::common::SemanticTarget::new(14),
                    },
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::Drop),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 1 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::End,
                },
                SemanticOp {
                    kind: SemanticOpKind::BrTable {
                        entries: alloc::vec![
                            crate::vm::wasm::common::BrTableEntry {
                                target: crate::vm::wasm::common::SemanticTarget::new(14),
                                stack_drop: 0,
                                arity: 1,
                            },
                            crate::vm::wasm::common::BrTableEntry {
                                target: crate::vm::wasm::common::SemanticTarget::new(14),
                                stack_drop: 0,
                                arity: 1,
                            },
                        ],
                    },
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 16 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::End,
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Add),
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnOne,
                },
            ],
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 3),
            },
            &semantic,
        )
        .expect("nested br_table index preparation should succeed");

        eprintln!("{:#?}", prepared.lir);
    }

    #[test]
    fn debug_prepares_break_br_table_nested_num_shape() {
        let semantic = SemanticProgram {
            params: 1,
            results: 1,
            local_count: 1,
            max_stack_height: 2,
            ops: alloc::vec![
                SemanticOp {
                    kind: SemanticOpKind::Block {
                        params: 0,
                        results: 1,
                    },
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 50 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 0 },
                },
                SemanticOp {
                    kind: SemanticOpKind::BrTable {
                        entries: alloc::vec![
                            crate::vm::wasm::common::BrTableEntry {
                                target: crate::vm::wasm::common::SemanticTarget::new(5),
                                stack_drop: 0,
                                arity: 1,
                            },
                            crate::vm::wasm::common::BrTableEntry {
                                target: crate::vm::wasm::common::SemanticTarget::new(8),
                                stack_drop: 0,
                                arity: 1,
                            },
                            crate::vm::wasm::common::BrTableEntry {
                                target: crate::vm::wasm::common::SemanticTarget::new(5),
                                stack_drop: 0,
                                arity: 1,
                            },
                        ],
                    },
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 51 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::End,
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 2 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Add),
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnOne,
                },
            ],
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 3),
            },
            &semantic,
        )
        .expect("nested br_table num preparation should succeed");

        eprintln!("{:#?}", prepared.lir);
    }

    #[test]
    fn debug_prepares_large_sig_shape() {
        let mut ops = alloc::vec![
            SemanticOp {
                kind: SemanticOpKind::LocalGet { idx: 5 },
            },
            SemanticOp {
                kind: SemanticOpKind::LocalGet { idx: 2 },
            },
            SemanticOp {
                kind: SemanticOpKind::LocalGet { idx: 0 },
            },
            SemanticOp {
                kind: SemanticOpKind::LocalGet { idx: 8 },
            },
            SemanticOp {
                kind: SemanticOpKind::LocalGet { idx: 7 },
            },
            SemanticOp {
                kind: SemanticOpKind::LocalGet { idx: 1 },
            },
            SemanticOp {
                kind: SemanticOpKind::LocalGet { idx: 3 },
            },
            SemanticOp {
                kind: SemanticOpKind::LocalGet { idx: 9 },
            },
            SemanticOp {
                kind: SemanticOpKind::LocalGet { idx: 4 },
            },
            SemanticOp {
                kind: SemanticOpKind::LocalGet { idx: 6 },
            },
            SemanticOp {
                kind: SemanticOpKind::LocalGet { idx: 13 },
            },
            SemanticOp {
                kind: SemanticOpKind::LocalGet { idx: 11 },
            },
            SemanticOp {
                kind: SemanticOpKind::LocalGet { idx: 15 },
            },
            SemanticOp {
                kind: SemanticOpKind::LocalGet { idx: 16 },
            },
            SemanticOp {
                kind: SemanticOpKind::LocalGet { idx: 14 },
            },
            SemanticOp {
                kind: SemanticOpKind::LocalGet { idx: 12 },
            },
        ];
        ops.push(SemanticOp {
            kind: SemanticOpKind::Return { arity: 16 },
        });

        let semantic = SemanticProgram {
            params: 17,
            results: 16,
            local_count: 17,
            max_stack_height: 16,
            ops,
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 3),
            },
            &semantic,
        )
        .expect("large signature preparation should succeed");

        eprintln!("{:#?}", prepared.lir);
    }
}
