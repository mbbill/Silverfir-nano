use crate::collections;
use crate::vm::middle::frame::FrameSlot;
use crate::vm::middle::ssa_ir::ir::SsaOp;

use crate::vm::wasm::primitive_op::PrimitiveOpKind;
use crate::vm::wasm::semantic_ir::SemanticOpKind;

use crate::vm::backend::BackendConfig;

use super::helpers::{
    all_inst_kinds, contains_ensure_cache, count_ensure_cache, first_local_get_for, i32_program,
    op, prepare_i32_program, prepare_program_with, target,
};

/// A config with preserved GP lanes and a zero nomination overhead, so any
/// trip-weighted survivable crossing nominates its residents.
fn preserved_config() -> BackendConfig {
    BackendConfig::with_volatility(8, 6, 4, 1, 4, 0, 4, 4, false, 3)
        .with_preserved_lane_save_overhead(0)
}

fn hot_loop_with_call(direct: bool) -> crate::vm::wasm::semantic_ir::SemanticProgram {
    use crate::vm::wasm::primitive_op::PrimitiveOpKind;
    let call = if direct {
        op(SemanticOpKind::CallDirect {
            callee: 0,
            params: 0,
            results: 0,
        })
    } else {
        op(SemanticOpKind::CallIndirect {
            type_idx: 0,
            table_idx: 0,
            params: 0,
            results: 0,
        })
    };
    i32_program(
        2,
        2,
        0,
        collections::vec![
            op(SemanticOpKind::Primitive(PrimitiveOpKind::I32Const {
                value: 7,
            })),
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
            op(SemanticOpKind::LocalSet { idx: 0 }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            call,
            op(SemanticOpKind::LocalGet { idx: 1 }),
            op(SemanticOpKind::BrIf {
                stack_drop: 0,
                arity: 0,
                target: target(13),
            }),
            op(SemanticOpKind::Br {
                stack_drop: 0,
                arity: 0,
                target: target(3),
            }),
            op(SemanticOpKind::End),
            op(SemanticOpKind::ReturnVoid),
        ],
    )
}

#[test]
fn nominated_resident_survives_a_direct_local_call_in_the_plan() {
    let semantic = hot_loop_with_call(true);

    // Preserved-capable config + a local callee: the loop-hot local is
    // nominated and the plan keeps it resident across the call, so the loop
    // steady state needs no per-iteration re-ensure (at most the cold entry
    // preload remains).
    let surviving = prepare_program_with(&semantic, preserved_config(), &[true]);
    let slot0 = surviving.frame.local_slot(0);
    let surviving_ensures = count_ensure_cache(&surviving.ssa, slot0);

    // Same program through a preserved-free config: every call kills the
    // cache and the backedge must re-ensure per iteration.
    let killed = prepare_program_with(
        &semantic,
        BackendConfig::with_volatility(8, 6, 0, 1, 4, 0, 4, 4, false, 3),
        &[true],
    );
    let killed_ensures = count_ensure_cache(&killed.ssa, slot0);

    assert!(
        surviving_ensures < killed_ensures,
        "plan-level call survival should remove the per-iteration re-ensure: surviving={surviving_ensures} killed={killed_ensures}"
    );
}

#[test]
fn barrier_calls_still_kill_nominated_residents() {
    let direct = prepare_program_with(&hot_loop_with_call(true), preserved_config(), &[true]);
    let indirect = prepare_program_with(&hot_loop_with_call(false), preserved_config(), &[true]);
    let slot0 = indirect.frame.local_slot(0);

    assert!(
        count_ensure_cache(&indirect.ssa, slot0)
            > count_ensure_cache(&direct.ssa, direct.frame.local_slot(0)),
        "an indirect call is a barrier: nominated residents must not survive it in the plan"
    );
}

#[test]
fn entry_block_does_not_preload_local_used_only_after_call_barrier() {
    // Entry-region planning must stop at the call barrier. Even if local0 is
    // used repeatedly after the call, the entry block should not preload it.
    let semantic = i32_program(
        1,
        1,
        1,
        collections::vec![
            op(SemanticOpKind::CallDirect {
                callee: 0,
                params: 0,
                results: 0,
            }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::Primitive(PrimitiveOpKind::I32Add,)),
            op(SemanticOpKind::ReturnOne),
        ],
    );

    let prepared = prepare_i32_program(&semantic, 3, 0);
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
        collections::vec![
            op(SemanticOpKind::Primitive(PrimitiveOpKind::I32Const {
                value: 7
            },)),
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
        .find(|inst| {
            matches!(inst.op, SsaOp::LOCAL_GET_SLOT | SsaOp::LOCAL_GET_CACHE)
                && FrameSlot(inst.meta) == slot0
        })
        .expect("expected one local.get after the call");

    assert!(
        post_call_get.op == SsaOp::LOCAL_GET_SLOT,
        "a one-shot local.get after a call should stay slot-based when the public set does not keep that local resident"
    );
}

#[test]
fn hot_repeated_local_can_stay_public_across_call() {
    let semantic = i32_program(
        1,
        4,
        1,
        collections::vec![
            op(SemanticOpKind::Primitive(PrimitiveOpKind::I32Const {
                value: 7
            },)),
            op(SemanticOpKind::LocalSet { idx: 0 }),
            op(SemanticOpKind::CallDirect {
                callee: 0,
                params: 0,
                results: 0,
            }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::Primitive(PrimitiveOpKind::I32Add,)),
            op(SemanticOpKind::Primitive(PrimitiveOpKind::Drop,)),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::Primitive(PrimitiveOpKind::I32Add,)),
            op(SemanticOpKind::ReturnOne),
        ],
    );

    let prepared = prepare_i32_program(&semantic, 4, 0);
    let slot0 = prepared.frame.local_slot(0);
    let first_get =
        first_local_get_for(&prepared.ssa, slot0).expect("expected one local.get after the call");

    assert!(
        first_get.op == SsaOp::LOCAL_GET_CACHE,
        "repeated post-call uses should still be allowed to use the public cache form"
    );
}

/// The bounded-counter forwarding pattern: an in-memory counter stored as
/// `k+1`, an intervening counter-indexed byte store, and the same-memloc reload
/// feeding the `!= B` exit compare. The pass must forward (recompute) the
/// reload; breaking the bound (byte-store base too close to the memloc) must
/// leave the load alone.
#[test]
fn bounded_counter_reload_is_forwarded() {
    use crate::vm::wasm::primitive_op::PrimitiveOpKind as P;

    // locals: 0=ctx base, 2=counter, 3=buffer base alias (ctx+16).
    // Layout: buffer at ctx+16..ctx+79 (64 bytes), memloc at ctx+80.
    let build = |memloc_off: u32| {
        i32_program(
            4,
            3,
            0,
            collections::vec![
                // ctx = 1024; buffer alias = ctx+16; counter = 0.
                op(SemanticOpKind::Primitive(P::I32Const { value: 1024 })),
                op(SemanticOpKind::LocalSet { idx: 0 }),
                op(SemanticOpKind::LocalGet { idx: 0 }),
                op(SemanticOpKind::Primitive(P::I32Const { value: 16 })),
                op(SemanticOpKind::Primitive(P::I32Add)),
                op(SemanticOpKind::LocalSet { idx: 3 }),
                op(SemanticOpKind::Primitive(P::I32Const { value: 0 })),
                op(SemanticOpKind::LocalSet { idx: 2 }),
                op(SemanticOpKind::Block {
                    params: 0,
                    results: 0,
                }),
                op(SemanticOpKind::Loop {
                    params: 0,
                    results: 0,
                }),
                // ctx->memloc = k + 1
                op(SemanticOpKind::LocalGet { idx: 0 }),
                op(SemanticOpKind::LocalGet { idx: 2 }),
                op(SemanticOpKind::Primitive(P::I32Const { value: 1 })),
                op(SemanticOpKind::Primitive(P::I32Add)),
                op(SemanticOpKind::Primitive(P::I32Store {
                    offset: memloc_off,
                    memidx: 0,
                })),
                // buffer[k] = 7
                op(SemanticOpKind::LocalGet { idx: 3 }),
                op(SemanticOpKind::LocalGet { idx: 2 }),
                op(SemanticOpKind::Primitive(P::I32Add)),
                op(SemanticOpKind::Primitive(P::I32Const { value: 7 })),
                op(SemanticOpKind::Primitive(P::I32Store8 {
                    offset: 0,
                    memidx: 0,
                })),
                // k = ctx->memloc; if k != 64 continue
                op(SemanticOpKind::LocalGet { idx: 0 }),
                op(SemanticOpKind::Primitive(P::I32Load {
                    offset: memloc_off,
                    memidx: 0,
                })),
                op(SemanticOpKind::LocalSet { idx: 2 }),
                op(SemanticOpKind::LocalGet { idx: 2 }),
                op(SemanticOpKind::Primitive(P::I32Const { value: 64 })),
                op(SemanticOpKind::Primitive(P::I32Ne)),
                op(SemanticOpKind::BrIf {
                    stack_drop: 0,
                    arity: 0,
                    target: target(9),
                }),
                // reset: k = 0; ctx->memloc = 0
                op(SemanticOpKind::Primitive(P::I32Const { value: 0 })),
                op(SemanticOpKind::LocalSet { idx: 2 }),
                op(SemanticOpKind::LocalGet { idx: 0 }),
                op(SemanticOpKind::Primitive(P::I32Const { value: 0 })),
                op(SemanticOpKind::Primitive(P::I32Store {
                    offset: memloc_off,
                    memidx: 0,
                })),
                op(SemanticOpKind::Primitive(P::I32Const { value: 1 })),
                op(SemanticOpKind::BrIf {
                    stack_drop: 0,
                    arity: 0,
                    target: target(35),
                }),
                op(SemanticOpKind::Br {
                    stack_drop: 0,
                    arity: 0,
                    target: target(9),
                }),
                op(SemanticOpKind::End),
                op(SemanticOpKind::ReturnVoid),
            ],
        )
    };

    let count_memloc_loads = |ssa: &crate::vm::middle::ssa_ir::ir::SsaProgram, off: u32| {
        let mut count = 0;
        for block in &ssa.blocks {
            for inst in &block.ops {
                if let Some(pool_idx) = inst.op.as_primitive_idx() {
                    if matches!(
                        ssa.primitive_pool[pool_idx as usize],
                        P::I32Load { offset, memidx: 0 } if offset == off
                    ) {
                        count += 1;
                    }
                }
            }
        }
        count
    };

    // memloc at ctx+80: buffer 16..79, delta 16 + B 64 = 80 <= 80 — forwards.
    let forwarded = prepare_i32_program(&build(80), 6, 0);
    assert_eq!(
        count_memloc_loads(&forwarded.ssa, 80),
        0,
        "the bounded-counter reload should be forwarded away"
    );

    // memloc at ctx+76 (inside the byte-store range): must NOT forward.
    let kept = prepare_i32_program(&build(76), 6, 0);
    assert!(
        count_memloc_loads(&kept.ssa, 76) > 0,
        "an in-range counter store must block forwarding"
    );
}
