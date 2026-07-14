use crate::collections;
use crate::vm::middle::cell::CellId;
use crate::vm::middle::ssa_ir::ir::SsaInstKind;

use crate::vm::wasm::primitive_op::PrimitiveOpKind;
use crate::vm::wasm::semantic_ir::SemanticOpKind;

use super::helpers::{
    all_inst_kinds, contains_drop_cache, i32_program, op, prepare_i32_program, prim,
};

#[test]
fn mid_block_pressure_spills_colder_bottom_transient_to_keep_hot_cache() {
    // One transient stays live across the block. A later local is cached and
    // then read twice. When another const push happens under a 2-unit budget,
    // the planner should spill the colder bottom transient instead of dropping
    // the hotter cache.
    let semantic = i32_program(
        1,
        3,
        0,
        collections::vec![
            prim(PrimitiveOpKind::I32Const { value: 5 }),
            prim(PrimitiveOpKind::I32Const { value: 7 }),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            prim(PrimitiveOpKind::I32Const { value: 22 }),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            prim(PrimitiveOpKind::I32Add),
            prim(PrimitiveOpKind::I32Add),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::ReturnVoid),
        ],
    );

    let prepared = prepare_i32_program(&semantic, 2, 0);
    let slot0 = CellId(0);
    let ops = all_inst_kinds(&prepared.ssa);
    let set_cache_idx = ops
        .iter()
        .position(|kind| matches!(kind, SsaInstKind::CellSetCache { slot, .. } if *slot == slot0))
        .expect("the written local should become cached at the set point");
    let first_cache_get_idx = ops
        .iter()
        .position(|kind| matches!(kind, SsaInstKind::CellGetCache { slot, .. } if *slot == slot0))
        .expect("the hot cached local should stay in cache form");
    let spill_idx = ops
        .iter()
        .position(|kind| matches!(kind, SsaInstKind::Spill { .. }))
        .expect("tight mid-block pressure should spill the colder bottom transient");
    let fill_idx = ops
        .iter()
        .position(|kind| matches!(kind, SsaInstKind::Fill { .. }))
        .expect("the spilled transient should be filled back before its later use");

    assert!(
        spill_idx < first_cache_get_idx,
        "the transient should spill before the block reaches the hot cache reads; ops={:?}",
        ops
    );
    assert!(
        fill_idx > first_cache_get_idx,
        "the spilled transient should only be restored near its later use; ops={:?}",
        ops
    );
    assert!(
        ops[..first_cache_get_idx]
            .iter()
            .skip(set_cache_idx + 1)
            .all(|kind| !matches!(kind, SsaInstKind::CellDropCache { slot } if *slot == slot0)),
        "the newly cached local should survive the pressure point instead of being dropped before its hot reads; ops={:?}",
        ops
    );
}

#[test]
fn mid_block_pressure_spills_truly_unused_transient_before_dropping_hot_cache() {
    // The bottom transient is never used anywhere in the block after it is
    // created. Under pressure, that makes it the first eviction candidate.
    // The hot cached local should survive until its later cache reads.
    let semantic = i32_program(
        1,
        3,
        0,
        collections::vec![
            prim(PrimitiveOpKind::I32Const { value: 5 }),
            prim(PrimitiveOpKind::I32Const { value: 7 }),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            prim(PrimitiveOpKind::I32Const { value: 22 }),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            prim(PrimitiveOpKind::I32Add),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::ReturnVoid),
        ],
    );

    let prepared = prepare_i32_program(&semantic, 2, 0);
    let slot0 = CellId(0);
    let ops = all_inst_kinds(&prepared.ssa);
    let set_cache_idx = ops
        .iter()
        .position(|kind| matches!(kind, SsaInstKind::CellSetCache { slot, .. } if *slot == slot0))
        .expect("the local should become cached at the set point");
    let first_cache_get_idx = ops
        .iter()
        .position(|kind| matches!(kind, SsaInstKind::CellGetCache { slot, .. } if *slot == slot0))
        .expect("the later reads should still use the hot cached local");
    let spill_idx = ops
        .iter()
        .position(|kind| matches!(kind, SsaInstKind::Spill { .. }))
        .expect("the truly unused bottom transient should spill first");
    let spilled_slot = match ops[spill_idx] {
        SsaInstKind::Spill { slot, .. } => *slot,
        _ => unreachable!("expected the recorded spill op"),
    };

    assert!(
        spill_idx < first_cache_get_idx,
        "the dead transient should spill before the block reaches the hot cache reads; ops={:?}",
        ops
    );
    assert!(
        ops.iter()
            .all(|kind| !matches!(kind, SsaInstKind::Fill { slot, .. } if *slot == spilled_slot)),
        "the specific dead transient that spilled first should never be filled back; ops={:?}",
        ops
    );
    assert!(
        ops[..first_cache_get_idx]
            .iter()
            .skip(set_cache_idx + 1)
            .all(|kind| !matches!(kind, SsaInstKind::CellDropCache { slot } if *slot == slot0)),
        "the hot cache should survive while the truly unused transient is discarded; ops={:?}",
        ops
    );
}

#[test]
fn mid_block_pressure_drops_cold_cache_to_keep_hot_transient_live() {
    // The transient is used immediately by the add, while the cached local is
    // never touched again. Under the same tight budget, the planner should
    // drop the cold cache instead of spilling the live transient.
    let semantic = i32_program(
        2,
        2,
        0,
        collections::vec![
            prim(PrimitiveOpKind::I32Const { value: 7 }),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            op(SemanticOpKind::LocalGet { idx: 1 }),
            prim(PrimitiveOpKind::I32Const { value: 22 }),
            prim(PrimitiveOpKind::I32Add),
            prim(PrimitiveOpKind::Drop),
            op(SemanticOpKind::ReturnVoid),
        ],
    );

    let prepared = prepare_i32_program(&semantic, 2, 0);
    let slot0 = CellId(0);
    let ops = all_inst_kinds(&prepared.ssa);
    let add_idx = ops
        .iter()
        .position(|kind| {
            matches!(
                kind,
                SsaInstKind::Value { op, .. }
                    if matches!(op.primitive(), PrimitiveOpKind::I32Add)
            )
        })
        .expect("expected the pressured add");

    assert!(
        contains_drop_cache(&prepared.ssa, slot0),
        "the unused cached local should be dropped under pressure"
    );
    assert!(
        ops[..add_idx]
            .iter()
            .any(|kind| matches!(kind, SsaInstKind::CellDropCache { slot } if *slot == slot0)),
        "the cache drop should happen before the pressured add; ops={:?}",
        ops
    );
    assert!(
        ops[..=add_idx]
            .iter()
            .all(|kind| !matches!(kind, SsaInstKind::Spill { .. })),
        "the live transient should stay resident instead of being spilled; ops={:?}",
        ops
    );
}

#[test]
fn mid_block_pressure_can_drop_multiple_cold_caches_to_admit_one_hot_local() {
    // Two colder caches occupy the whole GP budget. The later local2 is read
    // twice, so promoting it to cache form should be worth dropping both older
    // caches before the first get.
    let semantic = i32_program(
        3,
        2,
        1,
        collections::vec![
            prim(PrimitiveOpKind::I32Const { value: 1 }),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            prim(PrimitiveOpKind::I32Const { value: 2 }),
            op(SemanticOpKind::LocalSet { idx: 1 }),
            op(SemanticOpKind::LocalGet { idx: 2 }),
            op(SemanticOpKind::LocalGet { idx: 2 }),
            prim(PrimitiveOpKind::I32Add),
            op(SemanticOpKind::ReturnOne),
        ],
    );

    let prepared = prepare_i32_program(&semantic, 2, 0);
    let slot0 = CellId(0);
    let slot1 = CellId(1);
    let slot2 = CellId(2);
    let ops = all_inst_kinds(&prepared.ssa);
    let first_get2_idx = ops
        .iter()
        .position(|kind| matches!(kind, SsaInstKind::CellGetCache { slot, .. } if *slot == slot2))
        .unwrap_or_else(|| {
            panic!(
                "the hotter repeated local2 get should be promoted to cache form even if two colder caches must be dropped; ops={:?}",
                ops
            )
        });

    assert!(
        ops[..first_get2_idx]
            .iter()
            .any(|kind| matches!(kind, SsaInstKind::CellDropCache { slot } if *slot == slot0)),
        "the first colder cache should be dropped before the promoted local2 read"
    );
    assert!(
        ops[..first_get2_idx]
            .iter()
            .any(|kind| matches!(kind, SsaInstKind::CellDropCache { slot } if *slot == slot1)),
        "the second colder cache should also be dropped before the promoted local2 read"
    );
}
