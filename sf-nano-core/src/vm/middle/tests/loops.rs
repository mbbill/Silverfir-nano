use crate::collections;
use crate::vm::middle::cell::CellId;
use crate::vm::middle::ssa_ir::ir::SsaOp;

use crate::vm::wasm::{primitive_op::PrimitiveOpKind, semantic_ir::SemanticOpKind};

use super::helpers::{
    block_for_semantic_index, count_ensure_cache, first_local_get_for, i32_program,
    incoming_cache_repair_blocks, is_admitted, loop_body_block, loop_header_block,
    natural_loop_blocks, op, plan_i32_program, prepare_i32_program, prim, target,
};

#[test]
fn entry_block_hot_local_preload_uses_one_synthetic_entry_repair() {
    let semantic = i32_program(
        1,
        2,
        1,
        collections::vec![
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            prim(PrimitiveOpKind::I32Add),
            op(SemanticOpKind::ReturnOne),
        ],
    );

    let prepared = prepare_i32_program(&semantic, 3, 0);
    let slot0 = CellId(0);

    assert_eq!(prepared.ssa.blocks.len(), 1);
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
        collections::vec![
            prim(PrimitiveOpKind::I32Const { value: 7 }),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::ReturnOne),
        ],
    );

    let prepared = prepare_i32_program(&semantic, 2, 0);
    let slot0 = CellId(0);
    let first_get = first_local_get_for(&prepared.ssa, slot0)
        .expect("expected one local.get for the already-resident local");

    assert!(
        first_get.op == SsaOp::CELL_GET_CACHE,
        "a local that is already resident from CellSetCache should be read back through CellGetCache"
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
        collections::vec![
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
    let slot0 = CellId(0);
    let hot_loop_body_cfg = block_for_semantic_index(&pipeline.cfg, 8);

    assert!(
        prepared.ssa.block_entry_cached_cells[hot_loop_body_cfg.as_usize()]
            .contains(&slot0),
        "the hot loop body block should keep the carried local in its finalized entry boundary; entries={:?}",
        prepared.ssa.block_entry_cached_cells[hot_loop_body_cfg.as_usize()]
    );
    assert!(
        count_ensure_cache(&prepared.ssa, slot0) <= 1,
        "loop steady state should need at most one cold ensure for the hot local"
    );
}

#[test]
fn hot_loop_header_needs_repair_on_at_most_one_incoming_edge() {
    // The hot loop body is entered from a cold preheader and from the hot
    // backedge. Once the finalized entry is chosen, only one of those edges
    // should need cached-local repair. If both do, the loop merge is still
    // churning on the hot path.
    let semantic = i32_program(
        2,
        1,
        0,
        collections::vec![
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
    let slot0 = CellId(0);
    let hot_loop_body_cfg = block_for_semantic_index(&pipeline.cfg, 8);
    let hot_loop_body = loop_body_block(&prepared.ssa);
    let repair_blocks = incoming_cache_repair_blocks(&prepared.ssa, hot_loop_body);

    assert!(
        prepared.ssa.block_entry_cached_cells[hot_loop_body_cfg.as_usize()].contains(&slot0),
        "the loop header should still admit the hot carried local into finalized entry"
    );
    assert!(
        repair_blocks.len() <= 1,
        "the hot loop merge should converge so that at most one incoming edge needs cache repair; repair_blocks={:?}",
        repair_blocks
    );
}

#[test]
fn loop_dispatch_header_keeps_hot_pass_through_locals_across_backedge() {
    // Reduced func10-style case:
    // - local3 is the current dispatch character, used in the loop header
    // - local0/local1/local2 are hot loop-carried values used only in the
    //   body after the header branch
    // - a good steady-state boundary should keep those carried locals hot
    //   through the tiny dispatch header instead of forcing the backedge to
    //   spill and reload them every iteration
    let semantic = i32_program(
        4,
        1,
        0,
        collections::vec![
            prim(PrimitiveOpKind::I32Const { value: 7 }),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            prim(PrimitiveOpKind::I32Const { value: 11 }),
            op(SemanticOpKind::LocalSet { idx: 1 }),
            prim(PrimitiveOpKind::I32Const { value: 13 }),
            op(SemanticOpKind::LocalSet { idx: 2 }),
            prim(PrimitiveOpKind::I32Const { value: 65 }),
            op(SemanticOpKind::LocalSet { idx: 3 }),
            op(SemanticOpKind::Block {
                params: 0,
                results: 0,
            }),
            op(SemanticOpKind::Loop {
                params: 0,
                results: 0,
            }),
            op(SemanticOpKind::LocalGet { idx: 3 }),
            prim(PrimitiveOpKind::I32Const { value: 44 }),
            prim(PrimitiveOpKind::I32Eq),
            op(SemanticOpKind::BrIf {
                stack_drop: 0,
                arity: 0,
                target: target(20),
            }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::LocalGet { idx: 1 }),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::LocalGet { idx: 2 }),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::Br {
                stack_drop: 0,
                arity: 0,
                target: target(9),
            }),
            op(SemanticOpKind::End),
            op(SemanticOpKind::ReturnVoid),
        ],
    );

    let pipeline = plan_i32_program(&semantic, 5, 0);
    let prepared = prepare_i32_program(&semantic, 5, 0);
    let slot0 = CellId(0);
    let slot1 = CellId(1);
    let slot2 = CellId(2);
    let header_block = block_for_semantic_index(&pipeline.cfg, 10);
    let header_block_ssa = loop_header_block(&prepared.ssa);
    let repair_blocks = incoming_cache_repair_blocks(&prepared.ssa, header_block_ssa);

    let entry = &prepared.ssa.block_entry_cached_cells[header_block.as_usize()];
    assert!(
        entry.contains(&slot0) && entry.contains(&slot1) && entry.contains(&slot2),
        "a tiny dispatch header should keep the hot pass-through loop locals hot across the backedge; entry={entry:?}"
    );
    assert!(
        repair_blocks.len() <= 1,
        "once the loop header keeps the carried locals, at most one cold incoming edge should still need repair; repair_blocks={repair_blocks:?}"
    );
}

#[test]
fn loop_interior_state_blocks_keep_hot_locals_for_later_dispatch_bodies() {
    // Reduced func10-style interior chain:
    // - locals 0/1 are hot loop-carried data, used before and after the tiny
    //   state-only chain
    // - local3 drives an interior branch block that should behave as pure
    //   pass-through for the hot data locals
    // - local2 drives the dispatch block immediately after that branch
    //
    // A good steady-state boundary should keep locals 0/1 cached through both
    // the interior state block and the dispatch block so the later bodies do
    // not need to repair them back in.
    let semantic = i32_program(
        4,
        1,
        0,
        collections::vec![
            prim(PrimitiveOpKind::I32Const { value: 7 }),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            prim(PrimitiveOpKind::I32Const { value: 11 }),
            op(SemanticOpKind::LocalSet { idx: 1 }),
            prim(PrimitiveOpKind::I32Const { value: 1 }),
            op(SemanticOpKind::LocalSet { idx: 2 }),
            prim(PrimitiveOpKind::I32Const { value: 0 }),
            op(SemanticOpKind::LocalSet { idx: 3 }),
            op(SemanticOpKind::Block {
                params: 0,
                results: 0,
            }),
            op(SemanticOpKind::Loop {
                params: 0,
                results: 0,
            }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::LocalGet { idx: 1 }),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::LocalGet { idx: 3 }),
            op(SemanticOpKind::BrIf {
                stack_drop: 0,
                arity: 0,
                target: target(29),
            }),
            op(SemanticOpKind::LocalGet { idx: 2 }),
            op(SemanticOpKind::If {
                params: 0,
                results: 0,
                else_target: target(23),
            }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::LocalGet { idx: 1 }),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::Else {
                end_target: target(27),
            }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::LocalGet { idx: 1 }),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::End),
            op(SemanticOpKind::Br {
                stack_drop: 0,
                arity: 0,
                target: target(9),
            }),
            op(SemanticOpKind::End),
            op(SemanticOpKind::ReturnVoid),
        ],
    );

    let prepared = prepare_i32_program(&semantic, 4, 0);
    let slot0 = CellId(0);
    let slot1 = CellId(1);
    let header = loop_header_block(&prepared.ssa);
    let loop_blocks = natural_loop_blocks(&prepared.ssa);

    // Positive property: the hot loop-carried data locals are admitted to the
    // loop-entry boundary in the first place.
    assert!(
        prepared.ssa.block_entry_cached_cells[header].contains(&slot0)
            && prepared.ssa.block_entry_cached_cells[header].contains(&slot1),
        "hot loop-carried data locals should be cached at the loop-header entry; entries={:?}",
        prepared.ssa.block_entry_cached_cells,
    );

    // Whole-loop-body property (subsumes the old two-block checks): no block
    // inside the loop drops a hot data local, and no block re-ensures a slot it
    // already holds at entry — so the interior state/dispatch chain is a true
    // pass-through and later bodies never repair the hot locals back in.
    for &block_index in &loop_blocks {
        let block = &prepared.ssa.blocks[block_index];
        assert!(
            block.ops.iter().all(|inst| {
                !(inst.op == SsaOp::CELL_DROP_CACHE
                    && (CellId(inst.meta) == slot0 || CellId(inst.meta) == slot1))
            }),
            "no interior loop block should drop the hot loop-carried locals; block b{block_index}={block:?}"
        );
        let entry = &prepared.ssa.block_entry_cached_cells[block_index];
        assert!(
            block.ops.iter().all(|inst| {
                !(inst.op == SsaOp::CELL_ENSURE_CACHE && entry.contains(&CellId(inst.meta)))
            }),
            "no loop block should re-ensure a slot already cached at its entry; block b{block_index}={block:?}, entry={entry:?}"
        );
    }
}

#[test]
fn region_solver_keeps_higher_value_stable_layout_when_child_capacity_competes() {
    // Two child loop regions share one effective cache lane. The second loop
    // reads local2 more often than local1. The solver must let capacity prices
    // select the higher-value stable resident instead of reconstructing a
    // weaker local from a zero-price extraction.
    let semantic = i32_program(
        3,
        1,
        0,
        collections::vec![
            op(SemanticOpKind::Loop {
                params: 0,
                results: 0,
            }),
            op(SemanticOpKind::End),
            op(SemanticOpKind::Loop {
                params: 0,
                results: 0,
            }),
            op(SemanticOpKind::LocalGet { idx: 1 }),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::LocalGet { idx: 1 }),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::LocalGet { idx: 2 }),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::LocalGet { idx: 2 }),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::LocalGet { idx: 2 }),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::End),
            op(SemanticOpKind::ReturnVoid),
        ],
    );

    let pipeline = plan_i32_program(&semantic, 2, 0);
    let slot1 = CellId(1);
    let slot2 = CellId(2);
    let second_loop_body = block_for_semantic_index(&pipeline.cfg, 3);

    assert!(
        is_admitted(&pipeline.planner, second_loop_body, slot2)
            && !is_admitted(&pipeline.planner, second_loop_body, slot1),
        "the effective one-lane layout should admit the higher-value local2, not the weaker local1"
    );
}

#[test]
fn write_first_loop_header_uses_reserve_on_cold_entry_and_no_hot_backedge_repair() {
    // The loop body writes local0 before any read and then reads it later in
    // the same iteration. Finalized entry should therefore keep local0 hot,
    // but the cold incoming edge should still reserve it rather than ensuring
    // an old slot value. The hot backedge should match that boundary already.
    let semantic = i32_program(
        2,
        2,
        0,
        collections::vec![
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
            prim(PrimitiveOpKind::I32Const { value: 9 }),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::LocalGet { idx: 1 }),
            op(SemanticOpKind::BrIf {
                stack_drop: 0,
                arity: 0,
                target: target(12),
            }),
            op(SemanticOpKind::Br {
                stack_drop: 0,
                arity: 0,
                target: target(3),
            }),
            op(SemanticOpKind::End),
            op(SemanticOpKind::ReturnVoid),
        ],
    );

    let prepared = prepare_i32_program(&semantic, 2, 0);
    let slot0 = CellId(0);
    let loop_body = loop_header_block(&prepared.ssa);
    let repair_blocks = incoming_cache_repair_blocks(&prepared.ssa, loop_body);

    assert!(
        prepared.ssa.block_entry_cached_cells[loop_body].contains(&slot0),
        "a surviving write-first loop local should stay in the final loop-entry boundary as a reserved hot lane"
    );
    assert!(
        repair_blocks
            .iter()
            .filter(|block| {
                block.ops.iter().any(|inst| {
                    matches!(
                        inst.op,
                        SsaOp::CELL_RESERVE_CACHE
                            | SsaOp::CELL_ENSURE_CACHE
                            | SsaOp::CELL_DROP_CACHE
                    ) && CellId(inst.meta) == slot0
                })
            })
            .count()
            <= 1,
        "the write-first local itself should need repair on at most one incoming edge; loop_body={}, entries={:?}, repair_blocks={:?}",
        loop_body,
        prepared.ssa.block_entry_cached_cells[loop_body].clone(),
        repair_blocks
    );
    assert!(
        repair_blocks.iter().all(|block| {
            block
                .ops
                .iter()
                .all(|inst| !(inst.op == SsaOp::CELL_ENSURE_CACHE && CellId(inst.meta) == slot0))
        }),
        "the cold incoming edge should reserve the write-first local, not ensure an old value"
    );
    assert!(
        prepared.ssa.blocks.iter().any(|block| {
            block.ops.iter().any(|inst| {
                inst.op == SsaOp::CELL_RESERVE_CACHE && CellId(inst.meta) == slot0
            })
        }),
        "the cold incoming path should reserve the hot write-first local so the loop can keep that lane hot without loading an old slot value; repair_blocks={repair_blocks:?}, blocks={:?}",
        prepared.ssa.blocks
    );
}

#[test]
fn hot_loop_header_needs_no_cache_repair_when_all_incoming_edges_already_match() {
    // local0 is cached before entering the loop and remains the only hot cached
    // local in the loop body. Both the preheader and the backedge should
    // therefore already match the finalized loop-header entry, so no cache
    // repair block should be inserted at all.
    let semantic = i32_program(
        1,
        2,
        0,
        collections::vec![
            prim(PrimitiveOpKind::I32Const { value: 7 }),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            op(SemanticOpKind::Block {
                params: 0,
                results: 0,
            }),
            op(SemanticOpKind::Loop {
                params: 0,
                results: 0,
            }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            prim(PrimitiveOpKind::Drop),
            prim(PrimitiveOpKind::I32Const { value: 0 }),
            op(SemanticOpKind::BrIf {
                stack_drop: 0,
                arity: 0,
                target: target(10),
            }),
            op(SemanticOpKind::Br {
                stack_drop: 0,
                arity: 0,
                target: target(3),
            }),
            op(SemanticOpKind::End),
            op(SemanticOpKind::ReturnVoid),
        ],
    );

    let pipeline = plan_i32_program(&semantic, 2, 0);
    let prepared = prepare_i32_program(&semantic, 2, 0);
    let slot0 = CellId(0);
    let loop_body_cfg = block_for_semantic_index(&pipeline.cfg, 4);
    let loop_body = loop_header_block(&prepared.ssa);
    let repair_blocks = incoming_cache_repair_blocks(&prepared.ssa, loop_body);

    assert_eq!(
        prepared.ssa.block_entry_cached_cells[loop_body_cfg.as_usize()].as_slice(),
        &[slot0]
    );
    assert!(
        repair_blocks.is_empty(),
        "matching preheader and backedge exits should not create any cache repair for the loop header; entries={:?}, repair_blocks={:?}",
        prepared.ssa.block_entry_cached_cells,
        repair_blocks
    );
}

#[test]
fn loop_header_trims_cold_carried_local_so_only_the_cold_edge_needs_repair() {
    // local0 and local1 are both cached before the loop, but only local0 is
    // used in the loop body. The loop header should trim local1 from finalized
    // entry so the hot backedge no longer repairs it. At most one incoming
    // repair block should still mention local1, and that block should come from
    // the colder preheader exit that still carried both locals.
    let semantic = i32_program(
        2,
        2,
        0,
        collections::vec![
            prim(PrimitiveOpKind::I32Const { value: 7 }),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            prim(PrimitiveOpKind::I32Const { value: 8 }),
            op(SemanticOpKind::LocalSet { idx: 1 }),
            op(SemanticOpKind::Block {
                params: 0,
                results: 0,
            }),
            op(SemanticOpKind::Loop {
                params: 0,
                results: 0,
            }),
            prim(PrimitiveOpKind::I32Const { value: 0 }),
            op(SemanticOpKind::BrIf {
                stack_drop: 0,
                arity: 0,
                target: target(11),
            }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            prim(PrimitiveOpKind::Drop),
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
    let slot0 = CellId(0);
    let slot1 = CellId(1);
    let loop_body_cfg = block_for_semantic_index(&pipeline.cfg, 8);
    let loop_body = loop_body_block(&prepared.ssa);
    let repair_blocks = incoming_cache_repair_blocks(&prepared.ssa, loop_body);
    let slot1_repairs = repair_blocks
        .iter()
        .copied()
        .filter(|block| {
            block.ops.iter().any(|inst| {
                matches!(
                    inst.op,
                    SsaOp::CELL_DROP_CACHE | SsaOp::CELL_ENSURE_CACHE | SsaOp::CELL_RESERVE_CACHE
                ) && CellId(inst.meta) == slot1
            })
        })
        .collect::<collections::Vec<_>>();

    assert!(
        prepared.ssa.block_entry_cached_cells[loop_body_cfg.as_usize()].contains(&slot0),
        "the hot local should stay in the finalized loop-header entry"
    );
    assert!(
        !prepared.ssa.block_entry_cached_cells[loop_body_cfg.as_usize()].contains(&slot1),
        "the cold carried local should be trimmed from the finalized loop-header entry"
    );
    assert!(
        slot1_repairs.len() <= 1,
        "only the colder preheader edge should still mention the trimmed local in repair; entries={:?}, repair_blocks={:?}",
        prepared.ssa.block_entry_cached_cells,
        repair_blocks
    );
    if let Some(block) = slot1_repairs.first() {
        let pred_exit = &prepared.ssa.block_entry_cached_cells[block.id.as_usize()];
        assert!(
            pred_exit.contains(&slot1),
            "if a repair block still mentions the trimmed local, it should come from the predecessor that actually carried it"
        );
    }
}
