use crate::vm::jit::middle::cell::CellId;
use crate::vm::jit::wasm::{primitive_op::PrimitiveOpKind, semantic_ir::SemanticOpKind};

use crate::collections;

use super::helpers::{
    branch_edge_targets, contains_ensure_cache, i32_program, op, prepare_i32_program, prim, target,
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
        collections::vec![
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

    let prepared = prepare_i32_program(&semantic, 1, 0);
    let slot0 = CellId(0);
    let then_block = branch_edge_targets(&prepared.ssa).0;

    assert!(
        !prepared.ssa.block_entry_cached_cells[then_block].contains(&slot0),
        "finalized block entry should trim the wrong carried local from the pressured branch block; then_block={}, entries={:?}",
        then_block,
        prepared.ssa.block_entry_cached_cells,
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
        collections::vec![
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

    let prepared = prepare_i32_program(&semantic, 2, 0);
    let slot0 = CellId(0);
    let (then_block, else_block) = branch_edge_targets(&prepared.ssa);

    assert_eq!(
        prepared.ssa.block_entry_cached_cells[then_block],
        collections::vec![slot0]
    );
    assert_eq!(
        prepared.ssa.block_entry_cached_cells[else_block],
        collections::vec![slot0]
    );
}

#[test]
fn prepared_ssa_uses_trimmed_final_entry_for_branch_block() {
    let semantic = i32_program(
        2,
        1,
        0,
        collections::vec![
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

    let prepared = prepare_i32_program(&semantic, 1, 0);
    let slot0 = CellId(0);
    let then_block = branch_edge_targets(&prepared.ssa).0;

    assert!(
        !prepared.ssa.block_entry_cached_cells[then_block].contains(&slot0),
        "prepared SSA should use the finalized, trimmed entry set for the pressured branch block; then_block={}, entries={:?}",
        then_block,
        prepared.ssa.block_entry_cached_cells,
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
        collections::vec![
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

    let prepared = prepare_i32_program(&semantic, 2, 0);
    let slot0 = CellId(0);
    let (then_block, else_block) = branch_edge_targets(&prepared.ssa);

    assert_eq!(
        prepared.ssa.block_entry_cached_cells[then_block],
        collections::vec![slot0]
    );
    assert_eq!(
        prepared.ssa.block_entry_cached_cells[else_block],
        collections::vec![slot0]
    );
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
        collections::vec![
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

    let prepared = prepare_i32_program(&semantic, 2, 0);
    let slot0 = CellId(0);
    let slot1 = CellId(1);
    let then_block = branch_edge_targets(&prepared.ssa).0;

    assert_eq!(
        prepared.ssa.block_entry_cached_cells[then_block],
        collections::vec![slot0],
        "finalized entry should trim the colder carried local after one lowering pass"
    );
    assert!(
        !prepared.ssa.block_entry_cached_cells[then_block].contains(&slot1),
        "the colder carried local should not survive finalized entry once the block drops it under pressure"
    );
}
