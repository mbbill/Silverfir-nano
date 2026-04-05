use crate::vm::middle::joint_plan::EdgeRepairQuery;
use crate::vm::wasm::{primitive_op::PrimitiveOpKind, semantic_ir::SemanticOpKind};

use super::helpers::{
    block_for_semantic_index, contains_ensure_cache, i32_program, op, plan_i32_program,
    prepare_i32_program, prim, target,
};

#[test]
fn prepared_ssa_trims_carried_local_when_branch_block_never_uses_it_and_pressure_drops_it() {
    // local0 is cached before the `if`, but the then-block only needs local1.
    // With one GP dynamic unit, the carried local0 must disappear before the
    // then-block can run, so the finalized entry for that block must be empty.
    let semantic = i32_program(
        2,
        1,
        0,
        alloc::vec![
            prim(PrimitiveOpKind::I32Const { value: 7 }),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            prim(PrimitiveOpKind::I32Const { value: 1 }),
            op(SemanticOpKind::If {
                params: 0,
                results: 0,
                else_target: target(7),
            }),
            op(SemanticOpKind::LocalGet { idx: 1 }),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::Else {
                end_target: target(9),
            }),
            prim(PrimitiveOpKind::Nop),
            op(SemanticOpKind::End),
            op(SemanticOpKind::ReturnVoid),
        ],
    );

    let pipeline = plan_i32_program(&semantic, 1, 0);
    let then_block = block_for_semantic_index(&pipeline.cfg, 4);
    let prepared = prepare_i32_program(&semantic, 1, 0);
    let slot0 = prepared.frame.local_slot(0);

    assert!(
        !prepared.ssa.block_entry_cached_slots[then_block.as_usize()].contains(&slot0),
        "finalized block entry should trim the wrong carried local from the pressured branch block; then_block={}, entries={:?}",
        then_block.as_usize(),
        prepared.ssa.block_entry_cached_slots,
    );
    assert!(
        !contains_ensure_cache(&prepared.ssa, slot0),
        "a trimmed carried local must not be re-ensured on the incoming repair path"
    );
}

#[test]
fn prepared_ssa_keeps_unused_carried_local_when_branch_block_can_carry_it_through() {
    // The branch bodies do not touch local0, but they also do not need to drop
    // it. The join block reads local0, so the finalized branch entry should
    // keep the carried cache because it survives the empty branch body.
    let semantic = i32_program(
        1,
        1,
        1,
        alloc::vec![
            prim(PrimitiveOpKind::I32Const { value: 7 }),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            prim(PrimitiveOpKind::I32Const { value: 1 }),
            op(SemanticOpKind::If {
                params: 0,
                results: 0,
                else_target: target(6),
            }),
            prim(PrimitiveOpKind::Nop),
            op(SemanticOpKind::Else {
                end_target: target(8),
            }),
            prim(PrimitiveOpKind::Nop),
            op(SemanticOpKind::End),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::ReturnOne),
        ],
    );

    let pipeline = plan_i32_program(&semantic, 2, 0);
    let prepared = prepare_i32_program(&semantic, 2, 0);
    let slot0 = prepared.frame.local_slot(0);
    let then_block = block_for_semantic_index(&pipeline.cfg, 4);
    let else_block = block_for_semantic_index(&pipeline.cfg, 6);

    assert_eq!(
        prepared.ssa.block_entry_cached_slots[then_block.as_usize()],
        alloc::vec![slot0]
    );
    assert_eq!(
        prepared.ssa.block_entry_cached_slots[else_block.as_usize()],
        alloc::vec![slot0]
    );
}

#[test]
fn prepared_ssa_uses_trimmed_final_entry_for_branch_block() {
    let semantic = i32_program(
        2,
        1,
        0,
        alloc::vec![
            prim(PrimitiveOpKind::I32Const { value: 7 }),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            prim(PrimitiveOpKind::I32Const { value: 1 }),
            op(SemanticOpKind::If {
                params: 0,
                results: 0,
                else_target: target(7),
            }),
            op(SemanticOpKind::LocalGet { idx: 1 }),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::Else {
                end_target: target(9),
            }),
            prim(PrimitiveOpKind::Nop),
            op(SemanticOpKind::End),
            op(SemanticOpKind::ReturnVoid),
        ],
    );

    let pipeline = plan_i32_program(&semantic, 1, 0);
    let prepared = prepare_i32_program(&semantic, 1, 0);
    let slot0 = prepared.frame.local_slot(0);
    let then_block = block_for_semantic_index(&pipeline.cfg, 4).as_usize();

    assert!(
        !prepared.ssa.block_entry_cached_slots[then_block].contains(&slot0),
        "prepared SSA should use the finalized, trimmed entry set for the pressured branch block; then_block={}, entries={:?}",
        then_block,
        prepared.ssa.block_entry_cached_slots,
    );
    assert!(
        !contains_ensure_cache(&prepared.ssa, slot0),
        "a trimmed carried local must not be re-ensured on the incoming repair path"
    );
}

#[test]
fn prepared_ssa_keeps_surviving_carried_local_in_branch_entry() {
    let semantic = i32_program(
        1,
        1,
        1,
        alloc::vec![
            prim(PrimitiveOpKind::I32Const { value: 7 }),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            prim(PrimitiveOpKind::I32Const { value: 1 }),
            op(SemanticOpKind::If {
                params: 0,
                results: 0,
                else_target: target(6),
            }),
            prim(PrimitiveOpKind::Nop),
            op(SemanticOpKind::Else {
                end_target: target(8),
            }),
            prim(PrimitiveOpKind::Nop),
            op(SemanticOpKind::End),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::ReturnOne),
        ],
    );

    let pipeline = plan_i32_program(&semantic, 2, 0);
    let prepared = prepare_i32_program(&semantic, 2, 0);
    let slot0 = prepared.frame.local_slot(0);
    let then_block = block_for_semantic_index(&pipeline.cfg, 4).as_usize();
    let else_block = block_for_semantic_index(&pipeline.cfg, 6).as_usize();

    assert_eq!(prepared.ssa.block_entry_cached_slots[then_block], alloc::vec![slot0]);
    assert_eq!(prepared.ssa.block_entry_cached_slots[else_block], alloc::vec![slot0]);
}

#[test]
fn write_first_branch_block_can_claim_entry_cache_residency() {
    // This is the core write-first case from PLAN.md: the target block starts
    // by writing the local. The planner should still be free to reserve cache
    // residency for that slot on block entry.
    let semantic = i32_program(
        1,
        1,
        0,
        alloc::vec![
            prim(PrimitiveOpKind::I32Const { value: 1 }),
            op(SemanticOpKind::If {
                params: 0,
                results: 0,
                else_target: target(5),
            }),
            prim(PrimitiveOpKind::I32Const { value: 9 }),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            op(SemanticOpKind::Else {
                end_target: target(7),
            }),
            prim(PrimitiveOpKind::Nop),
            op(SemanticOpKind::End),
            op(SemanticOpKind::ReturnVoid),
        ],
    );

    let pipeline = plan_i32_program(&semantic, 1, 0);
    let slot0 = pipeline.frame.local_slot(0);
    let then_block = block_for_semantic_index(&pipeline.cfg, 2);

    assert_eq!(pipeline.planner.block_open(then_block).cached_locals, &[slot0]);
}

#[test]
fn edge_repair_is_trivial_diff_against_finalized_entry() {
    let semantic = i32_program(
        1,
        1,
        1,
        alloc::vec![
            prim(PrimitiveOpKind::I32Const { value: 7 }),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            prim(PrimitiveOpKind::I32Const { value: 1 }),
            op(SemanticOpKind::If {
                params: 0,
                results: 0,
                else_target: target(6),
            }),
            prim(PrimitiveOpKind::Nop),
            op(SemanticOpKind::Else {
                end_target: target(8),
            }),
            prim(PrimitiveOpKind::Nop),
            op(SemanticOpKind::End),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::ReturnOne),
        ],
    );

    let pipeline = plan_i32_program(&semantic, 2, 0);
    let slot0 = pipeline.frame.local_slot(0);
    let then_block = block_for_semantic_index(&pipeline.cfg, 4);
    let repair = pipeline.planner.edge_repair(EdgeRepairQuery {
        succ_block: Some(then_block),
        pred_exit: &[],
        succ_entry: pipeline.planner.block_open(then_block).cached_locals,
    });

    assert_eq!(repair.ensure_cached_locals, alloc::vec![slot0]);
    assert!(repair.reserve_cached_locals.is_empty());
    assert!(repair.drop_cached_locals.is_empty());
}
