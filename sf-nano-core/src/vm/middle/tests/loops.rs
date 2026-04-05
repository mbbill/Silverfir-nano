use crate::vm::middle::ssa_ir::ir::SsaInstKind;
use crate::vm::wasm::{primitive_op::PrimitiveOpKind, semantic_ir::SemanticOpKind};

use super::helpers::{
    block_for_semantic_index, count_ensure_cache, first_local_get_for, i32_program,
    incoming_cache_repair_blocks, op, plan_i32_program, prepare_i32_program, prim, target,
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

    assert_eq!(
        pipeline.planner.block_open(crate::vm::middle::cfg::CfgBlockId(then_block as u32)).cached_locals,
        alloc::vec![slot0]
    );
    assert!(
        prepared.ssa.blocks.iter().any(|block| {
            let first_set = block.ops.iter().position(
                |inst| matches!(inst.kind, SsaInstKind::LocalSetCache { slot, .. } if slot == slot0),
            );
            let Some(first_set) = first_set else {
                return false;
            };
            block.ops[..first_set].iter().all(|inst| {
                !matches!(inst.kind, SsaInstKind::LocalEnsureCache { slot } if slot == slot0)
            })
        }),
        "cleanup may erase an explicit reserve when the boundary disappears, but it must not introduce an old-value ensure before the first LocalSetCache; blocks={:?}",
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
        pipeline
            .planner
            .block_open(crate::vm::middle::cfg::CfgBlockId(hot_loop_body as u32))
            .cached_locals
            .contains(&slot0),
        "the hot loop body block should keep the carried local in its finalized entry boundary; entries={:?}",
        pipeline
            .planner
            .block_open(crate::vm::middle::cfg::CfgBlockId(hot_loop_body as u32))
            .cached_locals
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
    let repair_blocks = incoming_cache_repair_blocks(&prepared.ssa, hot_loop_body);

    assert!(
        pipeline
            .planner
            .block_open(crate::vm::middle::cfg::CfgBlockId(hot_loop_body as u32))
            .cached_locals
            .contains(&slot0),
        "the loop header should still admit the hot carried local into finalized entry"
    );
    assert!(
        repair_blocks.len() <= 1,
        "the hot loop merge should converge so that at most one incoming edge needs cache repair; repair_blocks={:?}",
        repair_blocks
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
        alloc::vec![
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

    let pipeline = plan_i32_program(&semantic, 2, 0);
    let prepared = prepare_i32_program(&semantic, 2, 0);
    let slot0 = prepared.frame.local_slot(0);
    let loop_body = block_for_semantic_index(&pipeline.cfg, 6).as_usize();
    let repair_blocks = incoming_cache_repair_blocks(&prepared.ssa, loop_body);

    assert!(
        pipeline
            .planner
            .block_open(crate::vm::middle::cfg::CfgBlockId(loop_body as u32))
            .cached_locals
            .contains(&slot0),
        "the write-first loop header should keep the hot local in finalized entry"
    );
    assert!(
        repair_blocks
            .iter()
            .filter(|block| {
                block.ops.iter().any(|inst| {
                    matches!(
                        inst.kind,
                        SsaInstKind::LocalReserveCache { slot }
                            | SsaInstKind::LocalEnsureCache { slot }
                            | SsaInstKind::LocalDropCache { slot }
                            if slot == slot0
                    )
                })
            })
            .count()
            <= 1,
        "the write-first local itself should need repair on at most one incoming edge; loop_body={}, entries={:?}, repair_blocks={:?}",
        loop_body,
        pipeline
            .planner
            .block_open(crate::vm::middle::cfg::CfgBlockId(loop_body as u32))
            .cached_locals,
        repair_blocks
    );
    assert!(
        prepared.ssa.blocks.iter().any(|block| {
            block.ops.iter().any(|inst| {
                matches!(inst.kind, SsaInstKind::LocalReserveCache { slot } if slot == slot0)
            })
        }),
        "the cleaned SSA should still reserve loop-header residency for the write-first local on the cold path"
    );
    assert!(
        prepared.ssa.blocks.iter().all(|block| {
            block.ops.iter().all(|inst| {
                !matches!(inst.kind, SsaInstKind::LocalEnsureCache { slot } if slot == slot0)
            })
        }),
        "write-first loop entry should not force an old-value ensure on any incoming repair edge"
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
        alloc::vec![
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
    let slot0 = prepared.frame.local_slot(0);
    let loop_body = block_for_semantic_index(&pipeline.cfg, 4).as_usize();
    let repair_blocks = incoming_cache_repair_blocks(&prepared.ssa, loop_body);

    assert_eq!(
        pipeline.planner.block_open(crate::vm::middle::cfg::CfgBlockId(loop_body as u32)).cached_locals,
        alloc::vec![slot0]
    );
    assert!(
        repair_blocks.is_empty(),
        "matching preheader and backedge exits should not create any cache repair for the loop header; entries={:?}, repair_blocks={:?}",
        prepared.ssa.block_entry_cached_slots,
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
        alloc::vec![
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
    let slot0 = prepared.frame.local_slot(0);
    let slot1 = prepared.frame.local_slot(1);
    let loop_body = block_for_semantic_index(&pipeline.cfg, 8).as_usize();
    let repair_blocks = incoming_cache_repair_blocks(&prepared.ssa, loop_body);
    let slot1_repairs = repair_blocks
        .iter()
        .copied()
        .filter(|block| {
            block.ops.iter().any(|inst| {
                matches!(
                    inst.kind,
                    SsaInstKind::LocalDropCache { slot }
                        | SsaInstKind::LocalEnsureCache { slot }
                        | SsaInstKind::LocalReserveCache { slot }
                        if slot == slot1
                )
            })
        })
        .collect::<alloc::vec::Vec<_>>();

    assert!(
        pipeline
            .planner
            .block_open(crate::vm::middle::cfg::CfgBlockId(loop_body as u32))
            .cached_locals
            .contains(&slot0),
        "the hot local should stay in the finalized loop-header entry"
    );
    assert!(
        !pipeline
            .planner
            .block_open(crate::vm::middle::cfg::CfgBlockId(loop_body as u32))
            .cached_locals
            .contains(&slot1),
        "the cold carried local should be trimmed from the finalized loop-header entry"
    );
    assert!(
        slot1_repairs.len() <= 1,
        "only the colder preheader edge should still mention the trimmed local in repair; entries={:?}, repair_blocks={:?}",
        prepared.ssa.block_entry_cached_slots,
        repair_blocks
    );
    if let Some(block) = slot1_repairs.first() {
        let pred_exit = &prepared.ssa.block_entry_cached_slots[block.id.as_usize()];
        assert!(
            pred_exit.contains(&slot1),
            "if a repair block still mentions the trimmed local, it should come from the predecessor that actually carried it"
        );
    }
}
