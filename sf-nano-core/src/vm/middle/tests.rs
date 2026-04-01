use crate::vm::{
    backend::BackendConfig,
    middle::{
        ssa_ir::ir::{SsaInstKind, SsaTerminator},
        PrepareInput, PreparedFunction,
    },
    wasm::{
        primitive_op::PrimitiveOpKind,
        semantic_ir::{SemanticOp, SemanticOpKind, SemanticProgram},
    },
};

use super::prepare_function;

fn test_backend_config() -> BackendConfig {
    BackendConfig::new(
        0,
        4,
        0,
        2,
        core::mem::size_of::<usize>() as u8,
        if core::mem::size_of::<usize>() == 4 {
            8
        } else {
            3
        },
    )
}

fn test_backend_config_with_gp_unit_bytes(gp_unit_bytes: u8) -> BackendConfig {
    BackendConfig::new(0, 6, 0, 0, gp_unit_bytes, 3)
}

#[test]
fn prepares_memory_copy_as_leaf_op() {
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
                kind: SemanticOpKind::Primitive(PrimitiveOpKind::MemoryCopy { imm0: 0, imm1: 1 }),
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
            config: test_backend_config(),
        },
        &semantic,
    )
    .expect("memory.copy preparation should succeed");

    assert!(prepared
        .ssa
        .blocks
        .iter()
        .any(|block| block.ops.iter().any(|inst| matches!(
            inst.kind,
            SsaInstKind::Value { ref op, .. }
                if matches!(op.primitive(), PrimitiveOpKind::MemoryCopy { imm0: 0, imm1: 1 })
        ))));
}

#[test]
fn prepares_table_fill_as_leaf_op() {
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
                kind: SemanticOpKind::Primitive(PrimitiveOpKind::TableFill { imm0: 2, imm1: 0 }),
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
            config: test_backend_config(),
        },
        &semantic,
    )
    .expect("table.fill preparation should succeed");

    assert!(prepared
        .ssa
        .blocks
        .iter()
        .any(|block| block.ops.iter().any(|inst| matches!(
            inst.kind,
            SsaInstKind::Value { ref op, .. }
                if matches!(op.primitive(), PrimitiveOpKind::TableFill { imm0: 2, .. })
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
                kind: SemanticOpKind::Primitive(PrimitiveOpKind::MemoryInit { imm0: 4, imm1: 7 }),
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
            config: test_backend_config(),
        },
        &semantic,
    )
    .expect("memory.init preparation should succeed");

    assert!(prepared
        .ssa
        .blocks
        .iter()
        .any(|block| block.ops.iter().any(|inst| matches!(
            inst.kind,
            SsaInstKind::Value { ref op, .. }
                if matches!(op.primitive(), PrimitiveOpKind::MemoryInit { imm0: 4, imm1: 7 })
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
                kind: SemanticOpKind::Primitive(PrimitiveOpKind::TableInit { imm0: 5, imm1: 8 }),
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
            config: test_backend_config(),
        },
        &semantic,
    )
    .expect("table.init preparation should succeed");

    assert!(prepared
        .ssa
        .blocks
        .iter()
        .any(|block| block.ops.iter().any(|inst| matches!(
            inst.kind,
            SsaInstKind::Value { ref op, .. }
                if matches!(op.primitive(), PrimitiveOpKind::TableInit { imm0: 5, imm1: 8 })
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
            config: test_backend_config(),
        },
        &semantic,
    )
    .expect("empty-if preparation should succeed");

    // End is no longer split into its own block — it merges with the
    // following code. Two empty-if sequences + return = 3 blocks
    // (entry, End+LocalGet+If, End+ReturnVoid).
    assert_eq!(prepared.ssa.blocks.len(), 3);
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
        result_types: alloc::vec![crate::value_type::ValueType::I32],
        op_result_types: alloc::collections::BTreeMap::from([(
            1usize,
            alloc::vec![crate::value_type::ValueType::I32],
        )]),
    };

    let prepared = prepare_function(
        PrepareInput {
            config: test_backend_config(),
        },
        &semantic,
    )
    .expect("result-if preparation should succeed");

    assert!(prepared
        .ssa
        .blocks
        .iter()
        .any(|block| matches!(block.terminator, SsaTerminator::Return { .. })));
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
        result_types: alloc::vec![crate::value_type::ValueType::I32],
        op_result_types: alloc::collections::BTreeMap::from([
            (0usize, alloc::vec![crate::value_type::ValueType::I32]),
            (2usize, alloc::vec![crate::value_type::ValueType::I32]),
        ]),
    };

    let prepared = prepare_function(
        PrepareInput {
            config: test_backend_config(),
        },
        &semantic,
    )
    .expect("br_if block-result preparation should succeed");

    assert!(prepared
        .ssa
        .blocks
        .iter()
        .any(|block| matches!(block.terminator, SsaTerminator::Branch { .. })));
    let final_return = prepared
        .ssa
        .blocks
        .iter()
        .find_map(|block| match block.terminator {
            SsaTerminator::Return {
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
                kind: SemanticOpKind::CallDirect {
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
        result_types: alloc::vec![ValueType::I64],
        op_result_types: alloc::collections::BTreeMap::from([
            (2usize, alloc::vec![ValueType::I64, ValueType::I32]),
            (3usize, alloc::vec![ValueType::I64]),
        ]),
    };

    let prepared = prepare_function(
        PrepareInput {
            config: test_backend_config(),
        },
        &semantic,
    )
    .expect("if param/result preparation should succeed");

    assert!(prepared
        .ssa
        .blocks
        .iter()
        .any(|block| matches!(block.terminator, SsaTerminator::Return { .. })));
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
        result_types: alloc::vec![crate::value_type::ValueType::I32],
        op_result_types: alloc::collections::BTreeMap::from([(
            3usize,
            alloc::vec![
                crate::value_type::ValueType::I32,
                crate::value_type::ValueType::I32
            ],
        )]),
    };

    let prepared = prepare_function(
        PrepareInput {
            config: test_backend_config(),
        },
        &semantic,
    )
    .expect("if param passthrough break preparation should succeed");

    let if_block = prepared
        .ssa
        .blocks
        .iter()
        .find(|block| matches!(block.terminator, SsaTerminator::Branch { .. }))
        .expect("if block");
    let store_count = if_block
        .ops
        .iter()
        .filter(|inst| matches!(inst.kind, SsaInstKind::Spill { .. }))
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
        result_types: alloc::vec![crate::value_type::ValueType::I32],
        op_result_types: alloc::collections::BTreeMap::from([
            (0usize, alloc::vec![crate::value_type::ValueType::I32]),
            (4usize, alloc::vec![crate::value_type::ValueType::I32]),
        ]),
    };

    let prepared = prepare_function(
        PrepareInput {
            config: test_backend_config(),
        },
        &semantic,
    )
    .expect("unreachable folded-if preparation should succeed");

    assert!(prepared
        .ssa
        .blocks
        .iter()
        .any(|block| matches!(block.terminator, SsaTerminator::Return { .. })));
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
        result_types: alloc::vec![crate::value_type::ValueType::I32],
        op_result_types: alloc::collections::BTreeMap::from([(
            1usize,
            alloc::vec![crate::value_type::ValueType::I32],
        )]),
    };

    let prepared = prepare_function(
        PrepareInput {
            config: test_backend_config(),
        },
        &semantic,
    )
    .expect("block-result fallthrough preparation should succeed");

    assert!(
        prepared.ssa.blocks.iter().any(|block| {
            block.ops.iter().any(|inst| {
                matches!(
                    inst.kind,
                    SsaInstKind::Spill { slot, .. }
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
        result_types: alloc::vec![crate::value_type::ValueType::I32],
        op_result_types: alloc::collections::BTreeMap::from([(
            2usize,
            alloc::vec![crate::value_type::ValueType::I32],
        )]),
    };

    let prepared = prepare_function(
        PrepareInput {
            config: test_backend_config(),
        },
        &semantic,
    )
    .expect("block result select preparation should succeed");

    assert!(prepared
        .ssa
        .blocks
        .iter()
        .any(|block| matches!(block.terminator, SsaTerminator::Return { .. })));
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
        result_types: alloc::vec![crate::value_type::ValueType::I32],
        op_result_types: alloc::collections::BTreeMap::from([
            (1usize, alloc::vec![crate::value_type::ValueType::I32]),
            (5usize, alloc::vec![crate::value_type::ValueType::I32]),
        ]),
    };

    let prepared = prepare_function(
        PrepareInput {
            config: test_backend_config(),
        },
        &semantic,
    )
    .expect("nested br_table index preparation should succeed");

    assert!(!prepared.ssa.blocks.is_empty());
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
        result_types: alloc::vec![crate::value_type::ValueType::I32],
        op_result_types: alloc::collections::BTreeMap::from([(
            0usize,
            alloc::vec![crate::value_type::ValueType::I32],
        )]),
    };

    let prepared = prepare_function(
        PrepareInput {
            config: test_backend_config(),
        },
        &semantic,
    )
    .expect("nested br_table num preparation should succeed");

    assert!(!prepared.ssa.blocks.is_empty());
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
        result_types: alloc::vec![crate::value_type::ValueType::I32; 16],
        op_result_types: alloc::collections::BTreeMap::new(),
    };

    let prepared = prepare_function(
        PrepareInput {
            config: test_backend_config(),
        },
        &semantic,
    )
    .expect("large signature preparation should succeed");

    assert!(!prepared.ssa.blocks.is_empty());
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
        result_types: alloc::vec![ValueType::F64],
        op_result_types: alloc::collections::BTreeMap::new(),
    };

    let prepared = prepare_function(
        PrepareInput {
            config: test_backend_config(),
        },
        &semantic,
    )
    .expect("typed float pipeline should succeed");

    assert!(
        !prepared.ssa.value_types.is_empty(),
        "value_types side table must be populated"
    );

    // Find LocalGet results (from local.get) — they should be F64.
    for block in &prepared.ssa.blocks {
        for inst in &block.ops {
            if let SsaInstKind::LocalGet { dst, .. } = &inst.kind {
                let ty = prepared.ssa.value_types.get(dst.0 as usize);
                assert_eq!(
                    ty.copied(),
                    Some(ValueType::F64),
                    "local.get of an f64 local must produce an F64-typed SsaValue"
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
        result_types: alloc::vec![ValueType::I32],
        op_result_types: alloc::collections::BTreeMap::new(),
    };

    let prepared = prepare_function(
        PrepareInput {
            config: test_backend_config(),
        },
        &semantic,
    )
    .expect("typed i32 pipeline should succeed");

    // All values should be I32 (consts produce I32, add produces I32).
    for ty in &prepared.ssa.value_types {
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
        result_types: alloc::vec![crate::value_type::ValueType::I64],
        op_result_types: alloc::collections::BTreeMap::new(),
    };

    let prepared_64 = prepare_function(
        PrepareInput {
            config: test_backend_config_with_gp_unit_bytes(8),
        },
        &semantic,
    )
    .expect("64-bit gp pressure preparation should succeed");
    let prepared_32 = prepare_function(
        PrepareInput {
            config: test_backend_config_with_gp_unit_bytes(4),
        },
        &semantic,
    )
    .expect("32-bit gp pressure preparation should succeed");

    let count_loads = |prepared: &PreparedFunction| {
        prepared
            .ssa
            .blocks
            .iter()
            .flat_map(|block| block.ops.iter())
            .filter(|inst| matches!(inst.kind, SsaInstKind::Fill { .. }))
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
        result_types: alloc::vec![ValueType::F64],
        op_result_types: alloc::collections::BTreeMap::new(),
    };

    let prepared = prepare_function(
        PrepareInput {
            config: test_backend_config(),
        },
        &semantic,
    )
    .expect("typed fill pipeline should succeed");

    // Every value in this program should be F64.
    for (idx, ty) in prepared.ssa.value_types.iter().enumerate() {
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
        result_types: alloc::vec![ValueType::F64],
        op_result_types: alloc::collections::BTreeMap::new(),
    };

    let prepared = prepare_function(
        PrepareInput {
            config: test_backend_config(),
        },
        &semantic,
    )
    .expect("typed select pipeline should succeed");

    // Find the Select result value and verify it's F64.
    let select_result = prepared
        .ssa
        .blocks
        .iter()
        .flat_map(|b| b.ops.iter())
        .find_map(|inst| {
            if let SsaInstKind::Value { op, results, .. } = &inst.kind {
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
    let ty = prepared.ssa.value_types[select_result.0 as usize];
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
        result_types: alloc::vec![ValueType::F64],
        op_result_types: alloc::collections::BTreeMap::from([(
            2usize,
            alloc::vec![ValueType::F64],
        )]),
    };

    let prepared = prepare_function(
        PrepareInput {
            config: test_backend_config(),
        },
        &semantic,
    )
    .expect("typed if/else param pipeline should succeed");

    // All block params across both arms should be F64 for the block
    // that corresponds to the else entry. Check that no block param
    // is typed as I32 (which would indicate type corruption from the
    // then arm's drop+push).
    for block in &prepared.ssa.blocks {
        for param in &block.params {
            let ty = prepared.ssa.value_types[param.0 as usize];
            // Block params for the if block carry the f64 param.
            // They must not be I32.
            assert_ne!(
                ty,
                ValueType::I32,
                "block param SsaValue({}) should not be I32 — if/else must preserve f64 param type",
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
            config: test_backend_config(),
        },
        &semantic,
    )
    .expect("typed ref.is_null pipeline should succeed");

    let ref_is_null_arg = prepared
        .ssa
        .blocks
        .iter()
        .flat_map(|block| block.ops.iter())
        .find_map(|inst| {
            let SsaInstKind::Value { op, args, .. } = &inst.kind else {
                return None;
            };
            matches!(op.primitive(), PrimitiveOpKind::RefIsNull)
                .then(|| args.first().copied())
                .flatten()
        })
        .expect("ref.is_null instruction must exist");
    let arg_ty = prepared.ssa.value_types[ref_is_null_arg.unwrap_value().0 as usize];
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
            config: test_backend_config(),
        },
        &semantic,
    )
    .expect("result if with returning arms should prepare cleanly");

    assert!(prepared
        .ssa
        .blocks
        .iter()
        .any(|block| matches!(block.terminator, SsaTerminator::Return { .. })));
}
