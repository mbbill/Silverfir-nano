use alloc::collections::BTreeSet;

use crate::collections;

use crate::value_type::ValueType;
use crate::vm::middle::tests::helpers::{
    block_for_semantic_index, host_config, op, plan_program, prepare_program, prim, target,
    typed_program,
};
use crate::vm::{
    middle::{cfg, frame::plan_frame_layout, joint_plan, slot_ssa},
    wasm::{primitive_op::PrimitiveOpKind, semantic_ir::SemanticOpKind},
};

#[test]
fn typed_block_passthrough_keeps_all_param_values_live_on_identity_edge() {
    // Spectest `block.wast` exercises empty typed blocks whose params become
    // their results unchanged. The identity edge into the block must therefore
    // keep the full param stack live; otherwise rewrite underflows while
    // binding target params at the block boundary.
    let mut semantic = typed_program(
        collections::vec![],
        collections::vec![],
        3,
        collections::vec![
            prim(PrimitiveOpKind::I32Const { value: 0 }),
            prim(PrimitiveOpKind::F64Const { value: 0 }),
            prim(PrimitiveOpKind::I32Const { value: 0 }),
            op(SemanticOpKind::Block {
                params: 3,
                results: 3,
            }),
            op(SemanticOpKind::End),
            prim(PrimitiveOpKind::Drop),
            prim(PrimitiveOpKind::Drop),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::ReturnVoid),
        ],
    );
    semantic.op_result_types.insert(
        3,
        collections::vec![ValueType::I32, ValueType::F64, ValueType::I32],
    );

    let _prepared = prepare_program(&semantic, 7, 13);
}

#[test]
fn typed_block_sequence_from_block_wast_type_use_prepares_successfully() {
    // This mirrors the spec test `block.wast` export `type-use`. The failure
    // only showed up under the richer sequence of typed blocks plus drops, not
    // in the minimal passthrough block above.
    let mut semantic = typed_program(
        collections::vec![],
        collections::vec![],
        3,
        collections::vec![
            op(SemanticOpKind::Block {
                params: 0,
                results: 0,
            }),
            op(SemanticOpKind::End),
            op(SemanticOpKind::Block {
                params: 0,
                results: 1,
            }),
            prim(PrimitiveOpKind::I32Const { value: 0 }),
            op(SemanticOpKind::End),
            op(SemanticOpKind::Block {
                params: 1,
                results: 0,
            }),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::End),
            prim(PrimitiveOpKind::I32Const { value: 0 }),
            prim(PrimitiveOpKind::F64Const { value: 0 }),
            prim(PrimitiveOpKind::I32Const { value: 0 }),
            op(SemanticOpKind::Block {
                params: 3,
                results: 3,
            }),
            op(SemanticOpKind::End),
            prim(PrimitiveOpKind::Drop),
            prim(PrimitiveOpKind::Drop),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::Block {
                params: 0,
                results: 1,
            }),
            prim(PrimitiveOpKind::I32Const { value: 0 }),
            op(SemanticOpKind::End),
            op(SemanticOpKind::Block {
                params: 1,
                results: 0,
            }),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::End),
            prim(PrimitiveOpKind::I32Const { value: 0 }),
            prim(PrimitiveOpKind::F64Const { value: 0 }),
            prim(PrimitiveOpKind::I32Const { value: 0 }),
            op(SemanticOpKind::Block {
                params: 3,
                results: 3,
            }),
            op(SemanticOpKind::End),
            prim(PrimitiveOpKind::Drop),
            prim(PrimitiveOpKind::Drop),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::ReturnVoid),
        ],
    );
    semantic
        .op_result_types
        .insert(2, collections::vec![ValueType::I32]);
    semantic.op_result_types.insert(
        11,
        collections::vec![ValueType::I32, ValueType::F64, ValueType::I32],
    );
    semantic
        .op_result_types
        .insert(16, collections::vec![ValueType::I32]);
    semantic.op_result_types.insert(
        25,
        collections::vec![ValueType::I32, ValueType::F64, ValueType::I32],
    );

    let _prepared = prepare_program(&semantic, 7, 13);
}

#[test]
fn typed_block_param_identity_preserves_two_live_values_for_following_binary_op() {
    // Spectest `block.wast` export `params-id`:
    //
    //   (i32.const 1)
    //   (i32.const 2)
    //   (block (param i32 i32) (result i32 i32))
    //   (i32.add)
    //
    // The identity edge into the typed block must preserve both live values so
    // the following binary op still sees two operands.
    let mut semantic = typed_program(
        collections::vec![],
        collections::vec![ValueType::I32],
        2,
        collections::vec![
            prim(PrimitiveOpKind::I32Const { value: 1 }),
            prim(PrimitiveOpKind::I32Const { value: 2 }),
            op(SemanticOpKind::Block {
                params: 2,
                results: 2,
            }),
            op(SemanticOpKind::End),
            prim(PrimitiveOpKind::I32Add),
            op(SemanticOpKind::ReturnOne),
        ],
    );
    semantic
        .op_result_types
        .insert(2, collections::vec![ValueType::I32, ValueType::I32]);

    let _prepared = prepare_program(&semantic, 7, 13);
}

#[test]
fn structured_branch_target_keeps_exact_transient_entry_at_end_block_open() {
    // This is the smallest value-carrying `br` to an outer block end that
    // still needs two operands after the join:
    //
    //   local.get 0
    //   block (result i32)
    //     block (result i32)
    //       br 1 (i32.const 1)
    //   end
    //   i32.add
    //
    // The join block starts at `End`. Its structural entry contract is fixed by
    // the branch payload shape and should stay fully spilled there. If
    // `block_open` eagerly fills the values at the block boundary, taken-branch
    // edge binding underflows before rewrite even reaches the following add.
    let mut semantic = typed_program(
        collections::vec![ValueType::I32],
        collections::vec![ValueType::I32],
        2,
        collections::vec![
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::Block {
                params: 0,
                results: 1,
            }),
            op(SemanticOpKind::Block {
                params: 0,
                results: 1,
            }),
            prim(PrimitiveOpKind::I32Const { value: 1 }),
            op(SemanticOpKind::Br {
                stack_drop: 0,
                arity: 1,
                target: target(6),
            }),
            op(SemanticOpKind::End),
            op(SemanticOpKind::End),
            prim(PrimitiveOpKind::I32Add),
            op(SemanticOpKind::ReturnOne),
        ],
    );
    semantic
        .op_result_types
        .insert(1, collections::vec![ValueType::I32]);
    semantic
        .op_result_types
        .insert(2, collections::vec![ValueType::I32]);

    let frame = plan_frame_layout(semantic.local_count, semantic.max_stack_height, 3);
    let cfg = cfg::build_semantic_cfg(&semantic);
    let slot = slot_ssa::lower_slot_only_ssa(&semantic, &cfg, frame)
        .expect("slot-only SSA should build for structured branch target regression");
    let plan = joint_plan::build::build_plan(&semantic, &cfg, &slot, frame, host_config(7, 13))
        .expect("joint plan should build for structured branch target regression");
    let join_block = block_for_semantic_index(&cfg, 6);
    let join_entry = plan.blocks[join_block.as_usize()].entry.clone();

    assert_eq!(join_entry.stack_height, 2);
    assert_eq!(
        join_entry.spill_depth, 2,
        "block_open must not change the structural transient contract at an End-start join block",
    );
    assert!(
        join_entry.live_types.is_empty(),
        "the End-start join block should not request live transient values at the boundary",
    );
}

#[test]
fn typed_block_break_inner_sequence_prepares_successfully() {
    // This is a hand-built semantic version of spectest `block.wast` export
    // `break-inner`. It keeps the regression at the middle layer instead of
    // depending on WAT parsing in `sf-nano-core` tests.
    //
    // The shape that matters is repeated:
    // - `local.get 0`
    // - nested value-carrying blocks
    // - `br` to an outer block end
    // - continued arithmetic after the block result rejoins
    //
    // If block entry/live-window accounting drops one of those carried values,
    // rewrite underflows on the following `i32.add`.
    let mut semantic = typed_program(
        collections::vec![ValueType::I32],
        collections::vec![ValueType::I32],
        3,
        collections::vec![
            prim(PrimitiveOpKind::I32Const { value: 0 }),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::Block {
                params: 0,
                results: 1,
            }),
            op(SemanticOpKind::Block {
                params: 0,
                results: 1,
            }),
            prim(PrimitiveOpKind::I32Const { value: 0x1 }),
            op(SemanticOpKind::Br {
                stack_drop: 0,
                arity: 1,
                target: target(8),
            }),
            op(SemanticOpKind::End),
            op(SemanticOpKind::End),
            prim(PrimitiveOpKind::I32Add),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::Block {
                params: 0,
                results: 1,
            }),
            op(SemanticOpKind::Block {
                params: 0,
                results: 0,
            }),
            op(SemanticOpKind::Br {
                stack_drop: 0,
                arity: 0,
                target: target(15),
            }),
            op(SemanticOpKind::End),
            prim(PrimitiveOpKind::I32Const { value: 0x2 }),
            op(SemanticOpKind::End),
            prim(PrimitiveOpKind::I32Add),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::Block {
                params: 0,
                results: 1,
            }),
            prim(PrimitiveOpKind::I32Const { value: 0x4 }),
            op(SemanticOpKind::Br {
                stack_drop: 0,
                arity: 1,
                target: target(25),
            }),
            prim(PrimitiveOpKind::I32Ctz),
            op(SemanticOpKind::End),
            prim(PrimitiveOpKind::I32Add),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::Block {
                params: 0,
                results: 1,
            }),
            op(SemanticOpKind::Block {
                params: 0,
                results: 1,
            }),
            prim(PrimitiveOpKind::I32Const { value: 0x8 }),
            op(SemanticOpKind::Br {
                stack_drop: 0,
                arity: 1,
                target: target(35),
            }),
            op(SemanticOpKind::End),
            prim(PrimitiveOpKind::I32Ctz),
            op(SemanticOpKind::End),
            prim(PrimitiveOpKind::I32Add),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::ReturnOne),
        ],
    );
    semantic
        .op_result_types
        .insert(3, collections::vec![ValueType::I32]);
    semantic
        .op_result_types
        .insert(4, collections::vec![ValueType::I32]);
    semantic
        .op_result_types
        .insert(12, collections::vec![ValueType::I32]);
    semantic
        .op_result_types
        .insert(21, collections::vec![ValueType::I32]);
    semantic
        .op_result_types
        .insert(29, collections::vec![ValueType::I32]);
    semantic
        .op_result_types
        .insert(30, collections::vec![ValueType::I32]);

    let _prepared = prepare_program(&semantic, 7, 13);
}

#[test]
fn typed_loop_br_if_backedge_keeps_loop_payload_live_for_fallthrough_and_reentry() {
    // Spectest `loop.wast` compiles the whole module before the first invoke,
    // and one later helper function has this shape:
    //
    //   i32.const 1
    //   i32.const 2
    //   loop (param i32 i32) (result i32)
    //     i32.add
    //     local.tee 0
    //     i32.const 3
    //     local.get 0
    //     i32.const 10
    //     i32.lt_u
    //     br_if 0
    //     drop
    //   end
    //
    // The `br_if` carries the two loop params back to the loop header when the
    // branch is taken, but the fallthrough path also continues with the same
    // loop payload shape until the trailing `drop`. If the planner spills that
    // payload at the branch point, rewrite underflows while binding the typed
    // loop header or the fallthrough path.
    let mut semantic = typed_program(
        collections::vec![ValueType::I32],
        collections::vec![ValueType::I32],
        4,
        collections::vec![
            prim(PrimitiveOpKind::I32Const { value: 1 }),
            prim(PrimitiveOpKind::I32Const { value: 2 }),
            op(SemanticOpKind::Loop {
                params: 2,
                results: 1,
            }),
            prim(PrimitiveOpKind::I32Add),
            op(SemanticOpKind::LocalTee { idx: 0 }),
            prim(PrimitiveOpKind::I32Const { value: 3 }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            prim(PrimitiveOpKind::I32Const { value: 10 }),
            prim(PrimitiveOpKind::I32LtU),
            op(SemanticOpKind::BrIf {
                stack_drop: 0,
                arity: 2,
                target: target(3),
            }),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::End),
            op(SemanticOpKind::End),
            op(SemanticOpKind::ReturnOne),
        ],
    );
    semantic
        .op_result_types
        .insert(2, collections::vec![ValueType::I32]);

    let _prepared = prepare_program(&semantic, 7, 13);
}

#[test]
fn typed_loop_block_start_before_op_refills_loop_params_for_first_body_op() {
    let mut semantic = typed_program(
        collections::vec![ValueType::I32],
        collections::vec![ValueType::I32],
        4,
        collections::vec![
            prim(PrimitiveOpKind::I32Const { value: 1 }),
            prim(PrimitiveOpKind::I32Const { value: 2 }),
            op(SemanticOpKind::Loop {
                params: 2,
                results: 1,
            }),
            prim(PrimitiveOpKind::I32Add),
            op(SemanticOpKind::LocalTee { idx: 0 }),
            prim(PrimitiveOpKind::I32Const { value: 3 }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            prim(PrimitiveOpKind::I32Const { value: 10 }),
            prim(PrimitiveOpKind::I32LtU),
            op(SemanticOpKind::BrIf {
                stack_drop: 0,
                arity: 2,
                target: target(3),
            }),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::End),
            op(SemanticOpKind::End),
            op(SemanticOpKind::ReturnOne),
        ],
    );
    semantic
        .op_result_types
        .insert(2, collections::vec![ValueType::I32]);

    let planned = plan_program(&semantic, 7, 13);
    let loop_block = block_for_semantic_index(&planned.cfg, 3);
    let block_open = planned.planner.block_open(loop_block);
    let before_add = planned.planner.before_op(joint_plan::BeforeOpQuery {
        semantic_index: 3,
        resident_cache: &BTreeSet::from([planned.frame.local_slot(0)]),
    });

    assert_eq!(block_open.transient.stack_height, 2);
    assert_eq!(block_open.transient.spill_depth, 2);
    assert_eq!(
        before_add.transient.live_value_count(),
        2,
        "the structural loop entry may stay fully spilled, but the first body op must refill the loop params before the i32.add",
    );
}

#[test]
fn br_if_value_inside_void_block_keeps_fallthrough_value_live_for_following_unary_op() {
    // Spectest `br_if.wast` export `type-i32`:
    //
    //   (block
    //     (drop (i32.ctz (br_if 0 (i32.const 0) (i32.const 1)))))
    //
    // Exact decoded semantic shape from spectest `br_if.wast` export
    // `type-i32`:
    //
    //   block
    //     i32.const 0
    //     i32.const 1
    //     br_if 0       ;; stack_drop = 1, arity = 0
    //     i32.ctz
    //     drop
    //   end
    //   end             ;; function end marker
    //
    // The `stack_drop = 1` means the branch consumes the earlier `i32.const 0`
    // on the taken edge, while the fallthrough path still needs that same
    // value for `i32.ctz`. The spectest failure shows that current live-window
    // accounting loses it and underflows before rewrite.
    let semantic = typed_program(
        collections::vec![],
        collections::vec![],
        2,
        collections::vec![
            op(SemanticOpKind::Block {
                params: 0,
                results: 0,
            }),
            prim(PrimitiveOpKind::I32Const { value: 0 }),
            prim(PrimitiveOpKind::I32Const { value: 1 }),
            op(SemanticOpKind::BrIf {
                stack_drop: 1,
                arity: 0,
                target: target(6),
            }),
            prim(PrimitiveOpKind::I32Ctz),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::End),
            op(SemanticOpKind::End),
            op(SemanticOpKind::ReturnVoid),
        ],
    );

    let _prepared = prepare_program(&semantic, 7, 13);
}
