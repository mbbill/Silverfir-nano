use crate::vm::middle::ssa_ir::ir::SsaInstKind;
use crate::vm::wasm::{primitive_op::PrimitiveOpKind, semantic_ir::SemanticOpKind};

use super::helpers::{
    block_for_semantic_index, count_ensure_cache, first_local_get_for, i32_program, op,
    plan_i32_program, prepare_i32_program, prim, target,
};

#[test]
fn write_first_branch_residency_does_not_force_entry_ensure() {
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
    let prepared = prepare_i32_program(&semantic, 1, 0);
    let slot0 = prepared.frame.local_slot(0);
    let then_block = block_for_semantic_index(&pipeline.cfg, 2).as_usize();
    let incoming_repairs = prepared
        .ssa
        .blocks
        .iter()
        .filter(|block| {
            matches!(
                &block.terminator,
                crate::vm::middle::ssa_ir::ir::SsaTerminator::Goto(edge)
                    if edge.target.as_usize() == then_block
            )
        })
        .collect::<alloc::vec::Vec<_>>();

    assert_eq!(prepared.ssa.block_entry_cached_slots[then_block], alloc::vec![slot0]);
    assert!(
        incoming_repairs.iter().any(|block| {
            block.ops.iter().any(|inst| {
                matches!(inst.kind, SsaInstKind::LocalReserveCache { slot } if slot == slot0)
            }) && block.ops.iter().all(|inst| {
                !matches!(inst.kind, SsaInstKind::LocalEnsureCache { slot } if slot == slot0)
            })
        }),
        "the repaired edge into the write-first block should reserve cache residency without ensuring the old slot value; blocks={:?}",
        prepared.ssa.blocks
    );
}

#[test]
fn entry_block_hot_local_preload_uses_one_synthetic_entry_repair() {
    let semantic = i32_program(
        1,
        2,
        1,
        alloc::vec![
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            prim(PrimitiveOpKind::I32Add),
            op(SemanticOpKind::ReturnOne),
        ],
    );

    let prepared = prepare_i32_program(&semantic, 2, 0);
    let slot0 = prepared.frame.local_slot(0);

    assert_eq!(prepared.ssa.block_entry_cached_slots[0], alloc::vec![slot0]);
    assert!(
        count_ensure_cache(&prepared.ssa, slot0) == 1,
        "entry-block preload should materialize once via the synthetic entry repair path"
    );
}

#[test]
fn local_get_uses_cache_when_local_is_already_resident_and_budget_has_result_room() {
    let semantic = i32_program(
        1,
        1,
        1,
        alloc::vec![
            prim(PrimitiveOpKind::I32Const { value: 7 }),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::ReturnOne),
        ],
    );

    let prepared = prepare_i32_program(&semantic, 2, 0);
    let slot0 = prepared.frame.local_slot(0);
    let first_get = first_local_get_for(&prepared.ssa, slot0)
        .expect("expected one local.get for the already-resident local");

    assert!(
        matches!(first_get, SsaInstKind::LocalGetCache { .. }),
        "a local that is already resident from LocalSetCache should be read back through LocalGetCache"
    );
}

#[test]
fn hot_loop_body_keeps_hot_local_in_final_entry_with_at_most_one_cold_ensure() {
    // This is the steady-state loop case from PLAN.md:
    // - local0 is hot inside the loop body
    // - non-backedge entry may need one cold materialization
    // - the backedge should then keep the loop header hot
    let semantic = i32_program(
        2,
        1,
        0,
        alloc::vec![
            prim(PrimitiveOpKind::I32Const { value: 7 }),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            prim(PrimitiveOpKind::I32Const { value: 1 }),
            op(SemanticOpKind::LocalSet { idx: 1 }),
            op(SemanticOpKind::Block {
                params: 0,
                results: 0,
            }),
            op(SemanticOpKind::Loop {
                params: 0,
                results: 0,
            }),
            op(SemanticOpKind::LocalGet { idx: 1 }),
            op(SemanticOpKind::BrIf {
                stack_drop: 0,
                arity: 0,
                target: target(12),
            }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            op(SemanticOpKind::Br {
                stack_drop: 0,
                arity: 0,
                target: target(5),
            }),
            op(SemanticOpKind::End),
            op(SemanticOpKind::ReturnVoid),
        ],
    );

    let pipeline = plan_i32_program(&semantic, 2, 0);
    let prepared = prepare_i32_program(&semantic, 2, 0);
    let slot0 = prepared.frame.local_slot(0);
    let hot_loop_body = block_for_semantic_index(&pipeline.cfg, 8).as_usize();

    assert!(
        prepared.ssa.block_entry_cached_slots[hot_loop_body].contains(&slot0),
        "the hot loop body block should keep the carried local in its finalized entry boundary; entries={:?}",
        prepared.ssa.block_entry_cached_slots
    );
    assert!(
        count_ensure_cache(&prepared.ssa, slot0) <= 1,
        "loop steady state should need at most one cold ensure for the hot local"
    );
}
