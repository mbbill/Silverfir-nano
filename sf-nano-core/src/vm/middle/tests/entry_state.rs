use crate::value_type::ValueType;
use crate::vm::wasm::{primitive_op::PrimitiveOpKind, semantic_ir::SemanticOpKind};

use super::helpers::{
    block_for_semantic_index, i32_program, op, plan_i32_program, plan_program, prim, typed_program,
};

#[test]
fn block_open_prefers_hotter_local_when_only_one_cache_slot_remains_after_entry_stack_pressure() {
    // The then-block starts with one meaningfully-used entry stack value, so
    // with a 2-unit GP budget only one cached local can fit at entry. local1
    // is both earlier and more heavily reused than local0 inside the entry
    // region, so block_open should keep local1.
    let semantic = i32_program(
        2,
        3,
        0,
        alloc::vec![
            prim(PrimitiveOpKind::I32Const { value: 5 }),
            prim(PrimitiveOpKind::I32Const { value: 1 }),
            op(SemanticOpKind::If {
                params: 1,
                results: 0,
                else_target: super::helpers::target(11),
            }),
            op(SemanticOpKind::LocalGet { idx: 1 }),
            prim(PrimitiveOpKind::I32Add),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::LocalGet { idx: 1 }),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::Else {
                end_target: super::helpers::target(13),
            }),
            prim(PrimitiveOpKind::Nop),
            op(SemanticOpKind::End),
            op(SemanticOpKind::ReturnVoid),
        ],
    );

    let pipeline = plan_i32_program(&semantic, 2, 0);
    let block = block_for_semantic_index(&pipeline.cfg, 3);
    let slot0 = pipeline.frame.local_slot(0);
    let slot1 = pipeline.frame.local_slot(1);
    let block_open = pipeline.planner.block_open(block);

    assert_eq!(block_open.transient.spill_depth, 0);
    assert_eq!(block_open.cached_locals, &[slot1]);
    assert!(!block_open.cached_locals.contains(&slot0));
}

#[test]
fn block_open_spills_cold_bottom_entry_stack_value_to_keep_hot_top_and_local() {
    // Two entry-stack values arrive at the then-block, but only the top one is
    // used meaningfully before the barrier. Under a 2-unit GP budget the
    // planner should spill the cold bottom entry value so the hot local can
    // still be cached.
    let semantic = i32_program(
        1,
        4,
        0,
        alloc::vec![
            prim(PrimitiveOpKind::I32Const { value: 1 }),
            prim(PrimitiveOpKind::I32Const { value: 2 }),
            prim(PrimitiveOpKind::I32Const { value: 1 }),
            op(SemanticOpKind::If {
                params: 2,
                results: 0,
                else_target: super::helpers::target(8),
            }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            prim(PrimitiveOpKind::I32Add),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::Else {
                end_target: super::helpers::target(10),
            }),
            prim(PrimitiveOpKind::Nop),
            op(SemanticOpKind::End),
            op(SemanticOpKind::ReturnVoid),
        ],
    );

    let pipeline = plan_i32_program(&semantic, 2, 0);
    let block = block_for_semantic_index(&pipeline.cfg, 4);
    let slot0 = pipeline.frame.local_slot(0);
    let block_open = pipeline.planner.block_open(block);

    assert_eq!(block_open.transient.stack_height, 2);
    assert_eq!(block_open.transient.spill_depth, 1);
    assert_eq!(block_open.cached_locals, &[slot0]);
}

#[test]
fn block_open_uses_per_bank_budget_so_gp_pressure_does_not_block_hot_fp_local() {
    // The entry GP stack value is used meaningfully, so it really consumes the
    // only GP dynamic unit. The hot local is F32 and should still be admitted
    // into the separate FP budget.
    let semantic = typed_program(
        alloc::vec![ValueType::F32],
        alloc::vec![],
        4,
        alloc::vec![
            prim(PrimitiveOpKind::I32Const { value: 1 }),
            prim(PrimitiveOpKind::I32Const { value: 1 }),
            op(SemanticOpKind::If {
                params: 1,
                results: 0,
                else_target: super::helpers::target(10),
            }),
            prim(PrimitiveOpKind::I32Const { value: 4 }),
            prim(PrimitiveOpKind::I32Add),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            prim(PrimitiveOpKind::F32Add),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::Else {
                end_target: super::helpers::target(12),
            }),
            prim(PrimitiveOpKind::Nop),
            op(SemanticOpKind::End),
            op(SemanticOpKind::ReturnVoid),
        ],
    );

    let pipeline = plan_program(&semantic, 1, 1);
    let block = block_for_semantic_index(&pipeline.cfg, 3);
    let slot0 = pipeline.frame.local_slot(0);
    let block_open = pipeline.planner.block_open(block);

    assert_eq!(block_open.transient.spill_depth, 0);
    assert_eq!(block_open.cached_locals, &[slot0]);
}
