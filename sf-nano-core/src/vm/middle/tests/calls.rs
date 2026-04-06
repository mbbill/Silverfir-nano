use crate::vm::middle::ssa_ir::ir::SsaInstKind;
use crate::vm::wasm::semantic_ir::SemanticOpKind;

use super::helpers::{
    all_inst_kinds, contains_ensure_cache, first_local_get_for, i32_program, op,
    prepare_i32_program,
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
        "a one-shot local.get after a call should stay slot-based when the public set does not keep that local resident"
    );
}

#[test]
fn hot_repeated_local_can_stay_public_across_call() {
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
        "repeated post-call uses should still be allowed to use the public cache form"
    );
}
