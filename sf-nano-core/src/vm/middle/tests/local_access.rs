use crate::vm::middle::ssa_ir::ir::SsaInstKind;
use crate::vm::wasm::{primitive_op::PrimitiveOpKind, semantic_ir::SemanticOpKind};

use super::helpers::{
    all_inst_kinds, contains_drop_cache, first_local_get_for, i32_program, local_set_kinds_for,
    op, prepare_i32_program, prim,
};

#[test]
fn local_set_lowers_to_cache_even_at_minimal_budget() {
    let semantic = i32_program(
        1,
        1,
        0,
        alloc::vec![
            prim(PrimitiveOpKind::I32Const { value: 7 }),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            op(SemanticOpKind::ReturnVoid),
        ],
    );

    let prepared = prepare_i32_program(&semantic, 1, 0);
    let slot0 = prepared.frame.local_slot(0);
    let set_kinds = local_set_kinds_for(&prepared.ssa, slot0);

    assert_eq!(set_kinds.len(), 1, "expected exactly one local.set lowering");
    assert!(matches!(set_kinds[0], SsaInstKind::LocalSetCache { .. }));
    assert!(
        all_inst_kinds(&prepared.ssa)
            .into_iter()
            .all(|kind| !matches!(kind, SsaInstKind::LocalSetSlot { .. })),
        "plain local.set should no longer lower through LocalSetSlot"
    );
}

#[test]
fn local_tee_uses_cache_form_when_budget_has_spare_capacity() {
    let semantic = i32_program(
        1,
        1,
        1,
        alloc::vec![
            prim(PrimitiveOpKind::I32Const { value: 1 }),
            op(SemanticOpKind::LocalTee { idx: 0 }),
            op(SemanticOpKind::ReturnOne),
        ],
    );

    let prepared = prepare_i32_program(&semantic, 2, 0);
    let slot0 = prepared.frame.local_slot(0);
    let ops = all_inst_kinds(&prepared.ssa);

    let set_idx = ops
        .iter()
        .position(|kind| matches!(kind, SsaInstKind::LocalSetCache { slot, .. } if *slot == slot0))
        .expect("local.tee with spare capacity should set through cache");
    assert!(
        matches!(ops.get(set_idx + 1), Some(SsaInstKind::LocalGetCache { slot, .. }) if *slot == slot0),
        "cache-form local.tee should immediately read back through LocalGetCache"
    );
}

#[test]
fn local_tee_uses_slot_form_when_existing_hot_cache_should_stay() {
    // local0 becomes hot and must stay cached for two later reads.
    // local1 is only teed once for the immediate expression result. With a
    // tight budget, `local.tee 1` should stay slot-based instead of evicting
    // the hotter resident cache.
    let semantic = i32_program(
        2,
        2,
        1,
        alloc::vec![
            prim(PrimitiveOpKind::I32Const { value: 1 }),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            prim(PrimitiveOpKind::I32Const { value: 2 }),
            op(SemanticOpKind::LocalTee { idx: 1 }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            prim(PrimitiveOpKind::I32Add),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            prim(PrimitiveOpKind::I32Add),
            op(SemanticOpKind::ReturnOne),
        ],
    );

    let prepared = prepare_i32_program(&semantic, 2, 0);
    let slot1 = prepared.frame.local_slot(1);
    let ops = all_inst_kinds(&prepared.ssa);

    let set_idx = ops
        .iter()
        .position(|kind| matches!(kind, SsaInstKind::LocalSetSlot { slot, .. } if *slot == slot1))
        .expect("tight-budget local.tee should use LocalSetSlot for the colder target local");
    assert!(
        matches!(ops.get(set_idx + 1), Some(SsaInstKind::LocalGetSlot { slot, .. }) if *slot == slot1),
        "slot-form local.tee should immediately read back through LocalGetSlot"
    );
}

#[test]
fn local_get_promotes_hot_local_and_drops_cold_cache() {
    // local0 is cached but dead after the set. local1 is read twice. Under a
    // tight budget the planner should drop the cold cache and admit local1.
    let semantic = i32_program(
        2,
        2,
        1,
        alloc::vec![
            prim(PrimitiveOpKind::I32Const { value: 1 }),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            op(SemanticOpKind::LocalGet { idx: 1 }),
            op(SemanticOpKind::LocalGet { idx: 1 }),
            prim(PrimitiveOpKind::I32Add),
            op(SemanticOpKind::ReturnOne),
        ],
    );

    let prepared = prepare_i32_program(&semantic, 2, 0);
    let slot0 = prepared.frame.local_slot(0);
    let slot1 = prepared.frame.local_slot(1);
    let first_get = first_local_get_for(&prepared.ssa, slot1)
        .expect("expected one local.get for slot1 in the prepared SSA");

    assert!(
        matches!(first_get, SsaInstKind::LocalGetCache { .. }),
        "a hotter repeated local.get should be promoted to cache form"
    );
    assert!(
        contains_drop_cache(&prepared.ssa, slot0),
        "promoting a hotter local.get under pressure should evict the colder cached local"
    );
}

#[test]
fn local_get_uses_slot_when_hot_resident_cache_should_stay() {
    // local0 stays hot because it is read later in the block. local1 is only
    // read once. The first local.get for slot1 should therefore stay slot-based
    // instead of evicting the hotter resident local0 cache.
    let semantic = i32_program(
        2,
        2,
        1,
        alloc::vec![
            prim(PrimitiveOpKind::I32Const { value: 1 }),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            op(SemanticOpKind::LocalGet { idx: 1 }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            prim(PrimitiveOpKind::I32Add),
            op(SemanticOpKind::ReturnOne),
        ],
    );

    let prepared = prepare_i32_program(&semantic, 2, 0);
    let slot0 = prepared.frame.local_slot(0);
    let slot1 = prepared.frame.local_slot(1);
    let first_get = first_local_get_for(&prepared.ssa, slot1)
        .expect("expected one local.get for slot1 in the prepared SSA");
    let ops = all_inst_kinds(&prepared.ssa);
    let first_get_idx = ops
        .iter()
        .position(|kind| matches!(kind, SsaInstKind::LocalGetSlot { slot, .. } if *slot == slot1))
        .expect("expected the first slot-based local.get for slot1");

    assert!(
        matches!(first_get, SsaInstKind::LocalGetSlot { .. }),
        "a colder one-shot local.get should stay slot-based when a hotter cache is already resident"
    );
    assert!(
        ops[..first_get_idx]
            .iter()
            .all(|kind| !matches!(kind, SsaInstKind::LocalDropCache { slot } if *slot == slot0)),
        "keeping the hot resident cache should avoid dropping local0 before the colder slot-based get"
    );
}
