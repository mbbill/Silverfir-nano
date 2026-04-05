use crate::vm::middle::ssa_ir::ir::SsaInstKind;
use crate::vm::wasm::semantic_ir::SemanticOpKind;

use super::helpers::{
    all_inst_kinds, block_for_semantic_index, contains_drop_cache, contains_ensure_cache,
    first_local_get_for, i32_program, incoming_cache_repair_blocks, op, plan_i32_program,
    prepare_i32_program, prim, target,
};

#[test]
fn entry_block_does_not_preload_local_used_only_after_call_barrier() {
    // Entry-region planning must stop at the call barrier. Even if local0 is
    // used repeatedly after the call, the entry block should not preload it.
    let semantic = i32_program(
        1,
        1,
        1,
        alloc::vec![
            op(SemanticOpKind::CallDirect {
                callee: 0,
                params: 0,
                results: 0,
            }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::Primitive(
                crate::vm::wasm::primitive_op::PrimitiveOpKind::I32Add,
            )),
            op(SemanticOpKind::ReturnOne),
        ],
    );

    let prepared = prepare_i32_program(&semantic, 2, 0);
    let slot0 = prepared.frame.local_slot(0);

    assert!(
        prepared
            .ssa
            .block_entry_cached_slots
            .first()
            .map(|slots| !slots.contains(&slot0))
            .unwrap_or(true),
        "entry-block cached locals should not preload values first used only after a call barrier"
    );
    assert!(
        !contains_ensure_cache(&prepared.ssa, slot0),
        "entry-block preload stays outside middle SSA; a post-call-only local should not be ensured on entry"
    );
}

#[test]
fn call_barrier_drops_cached_local_before_call() {
    let semantic = i32_program(
        1,
        1,
        0,
        alloc::vec![
            op(SemanticOpKind::Primitive(
                crate::vm::wasm::primitive_op::PrimitiveOpKind::I32Const { value: 7 },
            )),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            op(SemanticOpKind::CallDirect {
                callee: 0,
                params: 0,
                results: 0,
            }),
            op(SemanticOpKind::ReturnVoid),
        ],
    );

    let prepared = prepare_i32_program(&semantic, 1, 0);
    let slot0 = prepared.frame.local_slot(0);
    let ops = all_inst_kinds(&prepared.ssa);
    let call_index = ops
        .iter()
        .position(|kind| matches!(kind, SsaInstKind::Call(_)))
        .expect("expected one call in the prepared SSA");

    assert!(
        contains_drop_cache(&prepared.ssa, slot0),
        "call barriers should flush resident cached locals in middle SSA"
    );
    assert!(
        ops[..call_index]
            .iter()
            .any(|kind| matches!(kind, SsaInstKind::LocalDropCache { slot } if *slot == slot0)),
        "the cached local should be dropped before the call executes"
    );
}

#[test]
fn call_barrier_rebuilds_local_access_after_flush() {
    let semantic = i32_program(
        1,
        1,
        1,
        alloc::vec![
            op(SemanticOpKind::Primitive(
                crate::vm::wasm::primitive_op::PrimitiveOpKind::I32Const { value: 7 },
            )),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            op(SemanticOpKind::CallDirect {
                callee: 0,
                params: 0,
                results: 0,
            }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::ReturnOne),
        ],
    );

    let prepared = prepare_i32_program(&semantic, 1, 0);
    let slot0 = prepared.frame.local_slot(0);
    let ops = all_inst_kinds(&prepared.ssa);
    let post_call_get = ops
        .iter()
        .find(|kind| {
            matches!(
                kind,
                SsaInstKind::LocalGetSlot { slot, .. } | SsaInstKind::LocalGetCache { slot, .. }
                    if *slot == slot0
            )
        })
        .expect("expected one local.get after the call");

    assert!(
        matches!(post_call_get, SsaInstKind::LocalGetSlot { .. }),
        "after a forced call flush, a one-shot local.get should rebuild from slot form"
    );
}

#[test]
fn call_barrier_allows_hot_repeated_local_to_repromote_after_flush() {
    let semantic = i32_program(
        1,
        2,
        1,
        alloc::vec![
            op(SemanticOpKind::Primitive(
                crate::vm::wasm::primitive_op::PrimitiveOpKind::I32Const { value: 7 },
            )),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            op(SemanticOpKind::CallDirect {
                callee: 0,
                params: 0,
                results: 0,
            }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::Primitive(
                crate::vm::wasm::primitive_op::PrimitiveOpKind::I32Add,
            )),
            op(SemanticOpKind::ReturnOne),
        ],
    );

    let prepared = prepare_i32_program(&semantic, 2, 0);
    let slot0 = prepared.frame.local_slot(0);
    let first_get =
        first_local_get_for(&prepared.ssa, slot0).expect("expected one local.get after the call");

    assert!(
        matches!(first_get, SsaInstKind::LocalGetCache { .. }),
        "a repeated local.get after a call barrier should be allowed to re-promote into cache form"
    );
}

#[test]
fn loop_header_does_not_repair_local_whose_first_use_is_only_after_call_barrier() {
    // Even on a hot loop path, entry planning must stop at the call barrier.
    // The loop header should not keep repairing local0 into entry when the
    // first real use is only after the call.
    let semantic = i32_program(
        1,
        2,
        0,
        alloc::vec![
            prim(crate::vm::wasm::primitive_op::PrimitiveOpKind::I32Const { value: 7 }),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            op(SemanticOpKind::Block {
                params: 0,
                results: 0,
            }),
            op(SemanticOpKind::Loop {
                params: 0,
                results: 0,
            }),
            op(SemanticOpKind::CallDirect {
                callee: 0,
                params: 0,
                results: 0,
            }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::Primitive(
                crate::vm::wasm::primitive_op::PrimitiveOpKind::I32Add,
            )),
            op(SemanticOpKind::Primitive(
                crate::vm::wasm::primitive_op::PrimitiveOpKind::Drop,
            )),
            op(SemanticOpKind::Primitive(
                crate::vm::wasm::primitive_op::PrimitiveOpKind::I32Const { value: 0 },
            )),
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
    let loop_body = block_for_semantic_index(&pipeline.cfg, 4).as_usize();
    let repair_blocks = incoming_cache_repair_blocks(&prepared.ssa, loop_body);

    assert!(
        !prepared.ssa.block_entry_cached_slots[loop_body].contains(&slot0),
        "loop entry should not keep a local hot when its first use is only after the call barrier"
    );
    assert!(
        repair_blocks.iter().all(|block| {
            block.ops.iter().all(|inst| {
                !matches!(
                    inst.kind,
                    SsaInstKind::LocalEnsureCache { slot } | SsaInstKind::LocalReserveCache { slot }
                        if slot == slot0
                )
            })
        }),
        "the loop header should not keep repairing that post-call-only local on the hot path; entries={:?}, repair_blocks={:?}",
        prepared.ssa.block_entry_cached_slots,
        repair_blocks
    );
}
