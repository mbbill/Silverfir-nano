#![cfg(sf_has_simd)]

use crate::collections;
use crate::{
    value_type::ValueType,
    vm::{
        backend::BackendConfig,
        middle::{prepare_function, ModuleFacts, PrepareInput},
        wasm::{
            primitive_op::PrimitiveOpKind,
            semantic_ir::{SemanticOp, SemanticOpKind, SemanticProgram},
        },
    },
};
use tracked_alloc::collections::BTreeMap;

#[test]
fn prepare_accepts_v128_locals() {
    let semantic = SemanticProgram {
        params: 0,
        results: 0,
        local_count: 1,
        max_stack_height: 0,
        ops: collections::Vec::new(),
        local_types: collections::vec![ValueType::V128],
        result_types: collections::Vec::new(),
        op_result_types: BTreeMap::new(),
    };

    let prepared = prepare_function(
        PrepareInput {
            config: BackendConfig::new(8, 8, 8, 3),
            function_index: None,
        },
        ModuleFacts { is_local_func: &[] },
        semantic,
    )
    .expect("v128 locals should prepare cleanly when SIMD is enabled");

    assert_eq!(prepared.ssa.cell_types, collections::vec![ValueType::V128]);
}

#[test]
fn prepare_accepts_v128_primitives() {
    let semantic = SemanticProgram {
        params: 0,
        results: 1,
        local_count: 0,
        max_stack_height: 1,
        ops: collections::vec![
            SemanticOp {
                kind: SemanticOpKind::Primitive(PrimitiveOpKind::V128Const { value: [0; 16] }),
            },
            SemanticOp {
                kind: SemanticOpKind::ReturnOne,
            },
        ],
        local_types: collections::Vec::new(),
        result_types: collections::vec![ValueType::V128],
        op_result_types: BTreeMap::new(),
    };

    let prepared = prepare_function(
        PrepareInput {
            config: BackendConfig::new(8, 8, 8, 3),
            function_index: None,
        },
        ModuleFacts { is_local_func: &[] },
        semantic,
    )
    .expect("v128 primitives should prepare cleanly when SIMD is enabled");

    assert!(
        prepared
            .ssa
            .primitive_pool
            .iter()
            .any(|kind| matches!(kind, PrimitiveOpKind::V128Const { value } if *value == [0; 16])),
        "prepared SSA should retain the v128.const primitive",
    );
    assert!(
        prepared
            .ssa
            .value_types
            .iter()
            .any(|value_type| matches!(value_type, ValueType::V128)),
        "prepared SSA should track a v128 value type",
    );
}
