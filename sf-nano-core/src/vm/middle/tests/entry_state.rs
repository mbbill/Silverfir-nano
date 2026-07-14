use crate::collections;
use crate::value_type::ValueType;
use crate::vm::middle::cell::CellId;

use crate::vm::wasm::{primitive_op::PrimitiveOpKind, semantic_ir::SemanticOpKind};

use super::helpers::{
    block_for_semantic_index, i32_program, is_admitted, op, plan_i32_program, plan_program, prim,
    typed_program,
};

#[test]
fn block_open_prefers_hotter_local_when_only_one_cache_slot_remains_after_entry_stack_pressure() {
    // The then-block peaks at two live GP transients (entry value + local1 for
    // the add), so under the exact-headroom solver a 3-unit GP budget leaves
    // room for exactly one cached local. local1 is both earlier and more
    // heavily reused than local0 inside the entry region, so block_open should
    // keep local1.
    let semantic = i32_program(
        2,
        3,
        0,
        collections::vec![
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

    let pipeline = plan_i32_program(&semantic, 3, 0);
    let block = block_for_semantic_index(&pipeline.cfg, 3);
    let slot0 = CellId(0);
    let slot1 = CellId(1);
    let block_open = pipeline.planner.block_open(block);

    assert_eq!(block_open.transient.spill_depth, 0);
    assert!(
        is_admitted(&pipeline.planner, block, slot1),
        "the solver should admit the hotter local1 into the one remaining cache slot"
    );
    assert!(
        !is_admitted(&pipeline.planner, block, slot0),
        "the colder local0 should lose the admission contest"
    );
}

#[test]
fn block_open_keeps_structural_entry_stack_even_when_that_leaves_no_room_for_local_cache() {
    // The later plan simplification treats stack/SSA join shape as structural:
    // block_open may choose cached locals, but it must not rewrite the entry
    // spill depth just to make room for them. Here both entry stack values are
    // structurally live at the branch target, so under a 2-unit GP budget the
    // hot local simply cannot be admitted at entry.
    let semantic = i32_program(
        1,
        4,
        0,
        collections::vec![
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
    let slot0 = CellId(0);
    let block_open = pipeline.planner.block_open(block);

    assert_eq!(block_open.transient.stack_height, 2);
    assert_eq!(block_open.transient.spill_depth, 0);
    assert!(
        !is_admitted(&pipeline.planner, block, slot0),
        "when structural entry stack already consumes the whole budget, the hot local must not be admitted rather than changing stack join shape"
    );
}

#[test]
fn block_open_uses_per_bank_budget_so_gp_pressure_does_not_block_hot_fp_local() {
    // The entry GP stack value is used meaningfully, so it really consumes the
    // only GP dynamic unit. The hot local is F32 and should still be admitted
    // into the separate FP budget when that bank has its own spare headroom.
    let semantic = typed_program(
        collections::vec![ValueType::F32],
        collections::vec![],
        4,
        collections::vec![
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

    let pipeline = plan_program(&semantic, 1, 3);
    let block = block_for_semantic_index(&pipeline.cfg, 3);
    let slot0 = CellId(0);
    let block_open = pipeline.planner.block_open(block);

    assert_eq!(block_open.transient.spill_depth, 0);
    assert!(
        is_admitted(&pipeline.planner, block, slot0),
        "the hot F32 local should still be admitted into the separate FP budget"
    );
}
