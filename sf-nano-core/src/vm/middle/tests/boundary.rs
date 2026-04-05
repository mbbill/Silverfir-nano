use crate::vm::middle::joint_plan::EdgeRepairQuery;
use crate::vm::middle::ssa_ir::ir::SsaInstKind;
use crate::vm::wasm::{primitive_op::PrimitiveOpKind, semantic_ir::SemanticOpKind};

use super::helpers::{
    block_for_semantic_index, contains_ensure_cache, i32_program, incoming_cache_repair_blocks, op,
    plan_i32_program, prepare_i32_program, prim, target,
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

    assert_eq!(
        prepared.ssa.block_entry_cached_slots[then_block],
        alloc::vec![slot0]
    );
    assert_eq!(
        prepared.ssa.block_entry_cached_slots[else_block],
        alloc::vec![slot0]
    );
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

    assert_eq!(
        pipeline.planner.block_open(then_block).cached_locals,
        &[slot0]
    );
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

#[test]
fn prepared_ssa_trims_multi_carried_entry_down_to_the_hot_survivor() {
    // local0 and local1 are both carried into the branch block, so the
    // tentative entry may start with both. The block only reads local0, and
    // under a 2-unit budget the local.get result room should force local1 out.
    // Finalized entry must therefore keep only local0.
    let semantic = i32_program(
        2,
        2,
        0,
        alloc::vec![
            prim(PrimitiveOpKind::I32Const { value: 7 }),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            prim(PrimitiveOpKind::I32Const { value: 8 }),
            op(SemanticOpKind::LocalSet { idx: 1 }),
            prim(PrimitiveOpKind::I32Const { value: 1 }),
            op(SemanticOpKind::If {
                params: 0,
                results: 0,
                else_target: target(9),
            }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::Else {
                end_target: target(11),
            }),
            prim(PrimitiveOpKind::Nop),
            op(SemanticOpKind::End),
            op(SemanticOpKind::ReturnVoid),
        ],
    );

    let pipeline = plan_i32_program(&semantic, 2, 0);
    let prepared = prepare_i32_program(&semantic, 2, 0);
    let slot0 = prepared.frame.local_slot(0);
    let slot1 = prepared.frame.local_slot(1);
    let then_block = block_for_semantic_index(&pipeline.cfg, 6).as_usize();

    assert_eq!(
        prepared.ssa.block_entry_cached_slots[then_block],
        alloc::vec![slot0],
        "finalized entry should trim the colder carried local after one lowering pass"
    );
    assert!(
        !prepared.ssa.block_entry_cached_slots[then_block].contains(&slot1),
        "the colder carried local should not survive finalized entry once the block drops it under pressure"
    );
}

#[test]
fn ensured_entry_local_is_not_dropped_before_its_first_cache_read_under_pressure() {
    // This pins the PLAN.md failure condition as observable behavior.
    // local0 is read-first and must be ensured on the cold incoming edge. A
    // colder carried local1 also enters the target block, so the first const
    // push creates real pressure before local0's first read. The planner may
    // drop local1, but it must not ensure local0 and then drop it before the
    // first LocalGetCache.
    let semantic = i32_program(
        2,
        1,
        0,
        alloc::vec![
            prim(PrimitiveOpKind::I32Const { value: 9 }),
            op(SemanticOpKind::LocalSet { idx: 1 }),
            prim(PrimitiveOpKind::I32Const { value: 1 }),
            op(SemanticOpKind::If {
                params: 0,
                results: 0,
                else_target: target(10),
            }),
            prim(PrimitiveOpKind::I32Const { value: 22 }),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            prim(PrimitiveOpKind::I32Add),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::Else {
                end_target: target(12),
            }),
            prim(PrimitiveOpKind::Nop),
            op(SemanticOpKind::End),
            op(SemanticOpKind::ReturnVoid),
        ],
    );

    let pipeline = plan_i32_program(&semantic, 2, 0);
    let prepared = prepare_i32_program(&semantic, 2, 0);
    let slot0 = prepared.frame.local_slot(0);
    let slot1 = prepared.frame.local_slot(1);
    let then_block = block_for_semantic_index(&pipeline.cfg, 4).as_usize();
    let first_use_block = prepared
        .ssa
        .blocks
        .iter()
        .find(|block| {
            block.ops.iter().any(
                |inst| matches!(inst.kind, SsaInstKind::LocalGetCache { slot, .. } if slot == slot0),
            )
        })
        .expect("the ensured local should be read from cache in one cleaned SSA block");
    let first_get_idx = first_use_block
        .ops
        .iter()
        .position(
            |inst| matches!(inst.kind, SsaInstKind::LocalGetCache { slot, .. } if slot == slot0),
        )
        .expect("the ensured local should be read from cache in the target block");

    assert!(
        first_use_block.ops[..first_get_idx]
            .iter()
            .any(|inst| matches!(inst.kind, SsaInstKind::LocalEnsureCache { slot } if slot == slot0)),
        "the cold path should still materialize the read-first local into cache before its first cache read; ops={:?}",
        first_use_block.ops
    );
    assert!(
        first_use_block.ops[..first_get_idx]
            .iter()
            .all(|inst| !matches!(inst.kind, SsaInstKind::LocalDropCache { slot } if slot == slot0)),
        "the planner must not ensure a local on entry and then drop it before its first cache read; ops={:?}",
        first_use_block.ops
    );
    assert!(
        first_use_block.ops[..first_get_idx]
            .iter()
            .any(|inst| matches!(inst.kind, SsaInstKind::LocalDropCache { slot } if slot == slot1)),
        "this scenario should include real pre-use cache pressure, and the colder carried local should lose before the ensured local; ops={:?}",
        first_use_block.ops
    );
    assert!(
        incoming_cache_repair_blocks(&prepared.ssa, then_block).len() <= 1,
        "cleanup may merge the repair into the target block, but it must not leave more than one explicit incoming repair block"
    );
}
