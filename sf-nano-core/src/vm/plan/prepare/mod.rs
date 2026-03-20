//! Direct preparation from decoded Wasm semantics into prepared LIR.

mod block;
mod cfg;
mod edge;
mod ops;
mod optimize;
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
    let (local_cache, continuation_skip_reload) = analyze_local_cache_prefs(
        semantic,
        input.config.gp_unit_bytes,
        input.config.gp_local_cache_budget,
        input.config.fp_local_cache_budget,
        frame,
    );
    let gp_unused_cache = input
        .config
        .gp_local_cache_budget
        .saturating_sub(local_cache.gp_preferred_slots.len() as u8);
    let effective_config = PlanConfig {
        gp_transient_budget: input
            .config
            .gp_transient_budget
            .saturating_add(gp_unused_cache),
        ..input.config
    };
    let prepared = prepare_semantic_ops(semantic, frame, effective_config)?;

    if semantic.ops.is_empty() {
        return Ok(PreparedFunction {
            frame,
            lir: LirProgram {
                entry: LirTarget(0),
                local_cache,
                blocks: Vec::new(),
                value_types: Vec::new(),
            },
        });
    }

    let block_ranges = retain_reachable_blocks(semantic, build_block_ranges(semantic));
    let semantic_to_block = build_semantic_to_block_map(semantic.ops.len(), &block_ranges);
    let mut values = ValueAlloc::default();
    let local_types = &semantic.local_types;
    let op_result_types = &semantic.op_result_types;
    let block_params = block_ranges
        .iter()
        .map(|range| make_block_params(&prepared.entry_states[range.start], &mut values))
        .collect::<Vec<_>>();

    let original_block_count = block_ranges.len();
    let mut blocks = Vec::with_capacity(block_ranges.len());
    let mut extra_blocks = Vec::new();
    let mut skip_reload_iter = continuation_skip_reload.into_iter();
    for (block_index, semantic_range) in block_ranges.into_iter().enumerate() {
        let params = block_params[block_index].clone();
        let state = BlockState::from_entry(
            &prepared.entry_states[semantic_range.start],
            &params,
            effective_config.gp_unit_bytes,
            effective_config.gp_transient_budget,
            effective_config.fp_transient_budget,
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
            local_types,
            &semantic.result_types,
            op_result_types,
            &mut skip_reload_iter,
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
        value_types: values.take_types(),
    };
    let mut lir = lir;
    optimize::optimize_lir(&mut lir, frame);
    validate_program(&lir)?;

    Ok(PreparedFunction { frame, lir })
}

#[inline]
fn make_block_params(
    entry: &EntryState,
    values: &mut ValueAlloc,
) -> alloc::vec::Vec<crate::vm::lir::ir::LirValue> {
    debug_assert_eq!(
        entry.live_types.len(),
        entry.live_value_count() as usize,
        "EntryState.live_types length must match live_value_count",
    );
    values.many_typed(&entry.live_types)
}

#[cfg(test)]
mod tests {
    use crate::vm::{
        lir::ir::{LirBoundaryOp, LirInstKind},
        plan::{
            config::PlanConfig,
            prepare::{prepare_function, PrepareInput, PreparedFunction},
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
            local_types: alloc::vec![],
            result_types: alloc::vec![],
            op_result_types: alloc::collections::BTreeMap::from([(
                1usize,
                alloc::vec![crate::value_type::ValueType::I32],
            )]),
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 0, 2, 3),
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
        use crate::value_type::ValueType;

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
            local_types: alloc::vec![],
            result_types: alloc::vec![],
            op_result_types: alloc::collections::BTreeMap::from([(
                1usize,
                alloc::vec![ValueType::funcref()],
            )]),
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 0, 2, 3),
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
            local_types: alloc::vec![],
            result_types: alloc::vec![],
            op_result_types: alloc::collections::BTreeMap::from([(
                1usize,
                alloc::vec![crate::value_type::ValueType::I32],
            )]),
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 0, 2, 3),
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
            local_types: alloc::vec![],
            result_types: alloc::vec![],
            op_result_types: alloc::collections::BTreeMap::new(),
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 0, 2, 3),
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
            local_types: alloc::vec![],
            result_types: alloc::vec![],
            op_result_types: alloc::collections::BTreeMap::new(),
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 0, 2, 3),
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
            local_types: alloc::vec![],
            result_types: alloc::vec![],
            op_result_types: alloc::collections::BTreeMap::from([(
                1usize,
                alloc::vec![crate::value_type::ValueType::I32],
            )]),
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 0, 2, 3),
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
            local_types: alloc::vec![],
            result_types: alloc::vec![],
            op_result_types: alloc::collections::BTreeMap::from([
                (0usize, alloc::vec![crate::value_type::ValueType::I32]),
                (2usize, alloc::vec![crate::value_type::ValueType::I32]),
            ]),
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 0, 2, 3),
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
        use crate::value_type::ValueType;

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
            local_types: alloc::vec![ValueType::I64, ValueType::I64, ValueType::I64],
            result_types: alloc::vec![],
            op_result_types: alloc::collections::BTreeMap::from([(
                2usize,
                alloc::vec![ValueType::I64, ValueType::I32],
            )]),
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 0, 2, 3),
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
            local_types: alloc::vec![],
            result_types: alloc::vec![],
            op_result_types: alloc::collections::BTreeMap::new(),
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 0, 2, 3),
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
                        end_target: crate::vm::wasm::common::SemanticTarget::new(7),
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
            local_types: alloc::vec![],
            result_types: alloc::vec![],
            op_result_types: alloc::collections::BTreeMap::new(),
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 0, 2, 3),
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
            local_types: alloc::vec![],
            result_types: alloc::vec![],
            op_result_types: alloc::collections::BTreeMap::new(),
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 0, 2, 3),
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
            local_types: alloc::vec![],
            result_types: alloc::vec![],
            op_result_types: alloc::collections::BTreeMap::new(),
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 0, 2, 3),
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
            local_types: alloc::vec![],
            result_types: alloc::vec![],
            op_result_types: alloc::collections::BTreeMap::new(),
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 0, 2, 3),
            },
            &semantic,
        )
        .expect("nested br_table index preparation should succeed");

        assert!(!prepared.lir.blocks.is_empty());
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
            local_types: alloc::vec![],
            result_types: alloc::vec![],
            op_result_types: alloc::collections::BTreeMap::new(),
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 0, 2, 3),
            },
            &semantic,
        )
        .expect("nested br_table num preparation should succeed");

        assert!(!prepared.lir.blocks.is_empty());
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
            local_types: alloc::vec![],
            result_types: alloc::vec![],
            op_result_types: alloc::collections::BTreeMap::new(),
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 0, 2, 3),
            },
            &semantic,
        )
        .expect("large signature preparation should succeed");

        assert!(!prepared.lir.blocks.is_empty());
    }

    #[test]
    fn typed_pipeline_assigns_float_types_to_float_values() {
        use crate::value_type::ValueType;

        let semantic = SemanticProgram {
            params: 0,
            results: 1,
            local_count: 2,
            max_stack_height: 2,
            ops: alloc::vec![
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 0 },
                },
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 1 },
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::F64Add),
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnOne,
                },
            ],
            local_types: alloc::vec![ValueType::F64, ValueType::F64],
            result_types: alloc::vec![],
            op_result_types: alloc::collections::BTreeMap::new(),
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 0, 2, 3),
            },
            &semantic,
        )
        .expect("typed float pipeline should succeed");

        assert!(
            !prepared.lir.value_types.is_empty(),
            "value_types side table must be populated"
        );

        // Find LoadSlot results (from local.get) — they should be F64.
        for block in &prepared.lir.blocks {
            for inst in &block.ops {
                if let LirInstKind::LoadSlot { dst, .. } = &inst.kind {
                    let ty = prepared.lir.value_types.get(dst.0 as usize);
                    assert_eq!(
                        ty.copied(),
                        Some(ValueType::F64),
                        "local.get of an f64 local must produce an F64-typed LirValue"
                    );
                }
            }
        }
    }

    #[test]
    fn typed_pipeline_assigns_i32_types_to_int_ops() {
        use crate::value_type::ValueType;

        let semantic = SemanticProgram {
            params: 0,
            results: 1,
            local_count: 0,
            max_stack_height: 2,
            ops: alloc::vec![
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 1 }),
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
            local_types: alloc::vec![],
            result_types: alloc::vec![],
            op_result_types: alloc::collections::BTreeMap::new(),
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 0, 2, 3),
            },
            &semantic,
        )
        .expect("typed i32 pipeline should succeed");

        // All values should be I32 (consts produce I32, add produces I32).
        for ty in &prepared.lir.value_types {
            assert!(
                matches!(ty, ValueType::I32),
                "i32 ops should produce I32-typed values, got {:?}",
                ty,
            );
        }
    }

    #[test]
    fn i64_transient_pressure_counts_as_two_gp_units_on_32_bit() {
        let semantic = SemanticProgram {
            params: 0,
            results: 1,
            local_count: 0,
            max_stack_height: 4,
            ops: alloc::vec![
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I64Const { value: 1 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I64Const { value: 2 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I64Const { value: 3 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I64Const { value: 4 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I64Add),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I64Add),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I64Add),
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnOne,
                },
            ],
            local_types: alloc::vec![],
            result_types: alloc::vec![],
            op_result_types: alloc::collections::BTreeMap::new(),
        };

        let prepared_64 = prepare_function(
            PrepareInput {
                config: PlanConfig::new_with_gp_unit_bytes(0, 6, 0, 0, 3, 8),
            },
            &semantic,
        )
        .expect("64-bit gp pressure preparation should succeed");
        let prepared_32 = prepare_function(
            PrepareInput {
                config: PlanConfig::new_with_gp_unit_bytes(0, 6, 0, 0, 3, 4),
            },
            &semantic,
        )
        .expect("32-bit gp pressure preparation should succeed");

        let count_loads = |prepared: &PreparedFunction| {
            prepared
                .lir
                .blocks
                .iter()
                .flat_map(|block| block.ops.iter())
                .filter(|inst| matches!(inst.kind, LirInstKind::LoadSlot { .. }))
                .count()
        };

        assert_eq!(
            count_loads(&prepared_64),
            0,
            "four i64 values should fit in six 64-bit GP lanes without spilling",
        );
        assert!(
            count_loads(&prepared_32) >= 1,
            "four i64 values should require a spill/fill under a six-register 32-bit GP budget",
        );
    }

    #[test]
    fn unused_gp_cache_budget_extends_32bit_transient_capacity() {
        let semantic = SemanticProgram {
            params: 0,
            results: 1,
            local_count: 0,
            max_stack_height: 3,
            ops: alloc::vec![
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I64Const { value: 1 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I64Const { value: 2 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 1 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::Select),
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnOne,
                },
            ],
            local_types: alloc::vec![],
            result_types: alloc::vec![],
            op_result_types: alloc::collections::BTreeMap::new(),
        };

        prepare_function(
            PrepareInput {
                config: PlanConfig::new_with_gp_unit_bytes(4, 4, 0, 0, 3, 4),
            },
            &semantic,
        )
        .expect(
            "functions with no cached GP locals should be able to borrow that unused budget for 32-bit transient stack pressure",
        );
    }

    #[test]
    fn typed_pipeline_fills_preserve_float_types() {
        use crate::value_type::ValueType;

        // Push 4 f64 values to force a spill, then use them in an add.
        // The fill should preserve the f64 type.
        let semantic = SemanticProgram {
            params: 0,
            results: 1,
            local_count: 0,
            max_stack_height: 4,
            ops: alloc::vec![
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::F64Const {
                        value: 1.0f64.to_bits(),
                    }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::F64Const {
                        value: 2.0f64.to_bits(),
                    }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::F64Const {
                        value: 3.0f64.to_bits(),
                    }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::F64Const {
                        value: 4.0f64.to_bits(),
                    }),
                },
                // This add will need to fill the first spilled value.
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::F64Add),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::F64Add),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::F64Add),
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnOne,
                },
            ],
            local_types: alloc::vec![],
            result_types: alloc::vec![],
            op_result_types: alloc::collections::BTreeMap::new(),
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 0, 2, 3),
            },
            &semantic,
        )
        .expect("typed fill pipeline should succeed");

        // Every value in this program should be F64.
        for (idx, ty) in prepared.lir.value_types.iter().enumerate() {
            assert!(
                matches!(ty, ValueType::F64),
                "value {} should be F64 (fills must preserve float type), got {:?}",
                idx,
                ty,
            );
        }
    }

    #[test]
    fn typed_select_preserves_operand_type() {
        use crate::value_type::ValueType;

        // select(f64, f64, i32) should produce f64, not i64.
        let semantic = SemanticProgram {
            params: 0,
            results: 1,
            local_count: 2,
            max_stack_height: 3,
            ops: alloc::vec![
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 0 },
                },
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 1 },
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 1 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::Select),
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnOne,
                },
            ],
            local_types: alloc::vec![ValueType::F64, ValueType::F64],
            result_types: alloc::vec![],
            op_result_types: alloc::collections::BTreeMap::new(),
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 0, 2, 3),
            },
            &semantic,
        )
        .expect("typed select pipeline should succeed");

        // Find the Select result value and verify it's F64.
        let select_result = prepared
            .lir
            .blocks
            .iter()
            .flat_map(|b| b.ops.iter())
            .find_map(|inst| {
                if let LirInstKind::Value { op, results, .. } = &inst.kind {
                    if matches!(op.primitive(), PrimitiveOpKind::Select) {
                        results.first().copied()
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .expect("select instruction must exist");
        let ty = prepared.lir.value_types[select_result.0 as usize];
        assert_eq!(
            ty,
            ValueType::F64,
            "select of f64 operands must produce F64-typed value, got {:?}",
            ty,
        );
    }

    #[test]
    fn typed_if_else_preserves_param_types_across_arms() {
        use crate::value_type::ValueType;

        // if (param f64) (result f64): then-arm drops and pushes i32,
        // else-arm should still see f64 block param, not the i32 from then.
        //
        // (local.get 0)  ;; f64
        // (local.get 1)  ;; i32 condition
        // (if (param f64) (result f64)
        //   (then (drop) (f64.const 1.0))
        //   (else (nop))  ;; param f64 passes through
        // )
        // (return)
        let semantic = SemanticProgram {
            params: 2,
            results: 1,
            local_count: 2,
            max_stack_height: 2,
            ops: alloc::vec![
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 0 },
                },
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 1 },
                },
                SemanticOp {
                    kind: SemanticOpKind::If {
                        params: 1,
                        results: 1,
                        else_target: crate::vm::wasm::common::SemanticTarget::new(6),
                    },
                },
                // then: drop the f64 param, push f64.const
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::Drop),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::F64Const {
                        value: 1.0f64.to_bits(),
                    }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Else {
                        end_target: crate::vm::wasm::common::SemanticTarget::new(7),
                    },
                },
                // else: nop — pass through the f64 param
                SemanticOp {
                    kind: SemanticOpKind::End,
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnOne,
                },
            ],
            local_types: alloc::vec![ValueType::F64, ValueType::I32],
            result_types: alloc::vec![],
            op_result_types: alloc::collections::BTreeMap::from([(
                2usize,
                alloc::vec![ValueType::F64],
            )]),
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 0, 2, 3),
            },
            &semantic,
        )
        .expect("typed if/else param pipeline should succeed");

        // All block params across both arms should be F64 for the block
        // that corresponds to the else entry. Check that no block param
        // is typed as I32 (which would indicate type corruption from the
        // then arm's drop+push).
        for block in &prepared.lir.blocks {
            for param in &block.params {
                let ty = prepared.lir.value_types[param.0 as usize];
                // Block params for the if block carry the f64 param.
                // They must not be I32.
                assert_ne!(
                    ty,
                    ValueType::I32,
                    "block param LirValue({}) should not be I32 — if/else must preserve f64 param type",
                    param.0,
                );
            }
        }
    }

    #[test]
    fn typed_ref_is_null_preserves_ref_operand_type_across_if_else() {
        use crate::value_type::ValueType;

        let semantic = SemanticProgram {
            params: 2,
            results: 1,
            local_count: 2,
            max_stack_height: 2,
            ops: alloc::vec![
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 0 },
                },
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 1 },
                },
                SemanticOp {
                    kind: SemanticOpKind::If {
                        params: 1,
                        results: 1,
                        else_target: crate::vm::wasm::common::SemanticTarget::new(6),
                    },
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::Drop),
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::RefNull),
                },
                SemanticOp {
                    kind: SemanticOpKind::Else {
                        end_target: crate::vm::wasm::common::SemanticTarget::new(7),
                    },
                },
                SemanticOp {
                    kind: SemanticOpKind::End,
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::RefIsNull),
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnOne,
                },
            ],
            local_types: alloc::vec![ValueType::funcref(), ValueType::I32],
            result_types: alloc::vec![ValueType::I32],
            op_result_types: alloc::collections::BTreeMap::from([
                (2usize, alloc::vec![ValueType::funcref()]),
                (4usize, alloc::vec![ValueType::funcref()]),
            ]),
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 0, 2, 3),
            },
            &semantic,
        )
        .expect("typed ref.is_null pipeline should succeed");

        let ref_is_null_arg = prepared
            .lir
            .blocks
            .iter()
            .flat_map(|block| block.ops.iter())
            .find_map(|inst| {
                let LirInstKind::Value { op, args, .. } = &inst.kind else {
                    return None;
                };
                matches!(op.primitive(), PrimitiveOpKind::RefIsNull)
                    .then(|| args.first().copied())
                    .flatten()
            })
            .expect("ref.is_null instruction must exist");
        let arg_ty = prepared.lir.value_types[ref_is_null_arg.0 as usize];
        assert_eq!(
            arg_ty,
            ValueType::funcref(),
            "ref.is_null operand must keep funcref type across if/else, got {:?}",
            arg_ty,
        );
    }

    #[test]
    fn prepares_result_if_with_returning_arms_without_typed_stack_mismatch() {
        use crate::value_type::ValueType;

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
                        else_target: crate::vm::wasm::common::SemanticTarget::new(5),
                    },
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 7 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnOne,
                },
                SemanticOp {
                    kind: SemanticOpKind::Else {
                        end_target: crate::vm::wasm::common::SemanticTarget::new(7),
                    },
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 8 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnOne,
                },
                SemanticOp {
                    kind: SemanticOpKind::End,
                },
            ],
            local_types: alloc::vec![ValueType::I32],
            result_types: alloc::vec![ValueType::I32],
            op_result_types: alloc::collections::BTreeMap::from([(
                1usize,
                alloc::vec![ValueType::I32],
            )]),
        };

        let prepared = prepare_function(
            PrepareInput {
                config: PlanConfig::new(0, 4, 0, 2, 3),
            },
            &semantic,
        )
        .expect("result if with returning arms should prepare cleanly");

        assert!(prepared.lir.blocks.iter().any(|block| matches!(
            block.terminator,
            crate::vm::lir::ir::LirTerminator::Return { .. }
        )));
    }
}
