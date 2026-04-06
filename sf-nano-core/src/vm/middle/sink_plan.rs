//! Backward per-block sink planner.
//!
//! For each `LocalSetCache { slot, src }`, this pass checks whether the
//! producer of `src` can write its result directly into the local's cache
//! register instead of a transient linear-value register. The machine
//! lowering layer uses the resulting `value_sink_local` annotations via
//! `apply_sink_premap` to place results directly into cache registers.
//!
//! The sink is legal when:
//! - `src` is produced by a single-result `Value` instruction in the same block
//! - The producer is not a call
//! - No barrier (Call) exists between the producer and the LocalSetCache
//! - No `LocalGetCache` or `LocalGetSlot` of the same slot exists between
//!   the producer and the set (i.e., the old cached value is not read)

use alloc::vec;
use alloc::vec::Vec;

use crate::vm::middle::{
    frame::FrameSlot,
    ssa_ir::ir::{SsaBlock, SsaInstKind, SsaProgram},
};

/// Run the sink planner over all blocks in the program.
/// Populates `program.value_sink_local` with sink annotations.
pub(super) fn plan_sinks(program: &mut SsaProgram) {
    let value_count = program.value_types.len();
    if value_count == 0 {
        return;
    }
    let mut sinks: Vec<Option<FrameSlot>> = vec![None; value_count];

    for block in &program.blocks {
        plan_block_sinks(block, &mut sinks);
    }

    program.value_sink_local = sinks;
}

/// Analyze one block for sink opportunities.
fn plan_block_sinks(block: &SsaBlock, sinks: &mut [Option<FrameSlot>]) {
    let ops = &block.ops;
    if ops.is_empty() {
        return;
    }

    // Step 1: Record producer positions for single-result Value ops.
    let mut producer_pos: Vec<Option<u32>> = Vec::new();
    let mut max_val: u32 = 0;

    for inst in ops.iter() {
        match &inst.kind {
            SsaInstKind::Value { results, .. } => {
                for r in results {
                    if r.0 >= max_val {
                        max_val = r.0 + 1;
                    }
                }
            }
            SsaInstKind::LocalGetCache { dst, .. }
            | SsaInstKind::LocalGetSlot { dst, .. }
            | SsaInstKind::Fill { dst, .. } => {
                if dst.0 >= max_val {
                    max_val = dst.0 + 1;
                }
            }
            _ => {}
        }
    }

    producer_pos.resize(max_val as usize, None);
    let mut is_single_result = vec![false; max_val as usize];

    for (pos, inst) in ops.iter().enumerate() {
        let produced = match &inst.kind {
            SsaInstKind::Value { results, .. } if results.len() == 1 => Some(results[0]),
            _ => None,
        };
        if let Some(r) = produced {
            if (r.0 as usize) < producer_pos.len() {
                producer_pos[r.0 as usize] = Some(pos as u32);
                is_single_result[r.0 as usize] = true;
            }
        }
    }

    // Step 2: For each LocalSetCache, check sink legality.
    for (set_pos, inst) in ops.iter().enumerate() {
        let (slot, src) = match &inst.kind {
            SsaInstKind::LocalSetCache { slot, src } => (*slot, *src),
            _ => continue,
        };

        let src_idx = src.0 as usize;
        if src_idx >= producer_pos.len() || !is_single_result[src_idx] {
            continue;
        }
        let Some(prod_pos) = producer_pos[src_idx] else {
            continue;
        };
        let prod_pos = prod_pos as usize;

        if prod_pos >= set_pos {
            continue;
        }

        // No call barrier between producer and set.
        let has_barrier = ops[prod_pos + 1..set_pos]
            .iter()
            .any(|i| matches!(i.kind, SsaInstKind::Call(_)));
        if has_barrier {
            continue;
        }

        // Value must not already be sunk elsewhere.
        if sinks.get(src_idx).copied().flatten().is_some() {
            continue;
        }

        // Old value of this local must not be read between producer and set.
        let old_value_live = ops[prod_pos + 1..set_pos].iter().any(|i| match &i.kind {
            SsaInstKind::LocalGetCache { slot: s, .. }
            | SsaInstKind::LocalGetSlot { slot: s, .. } => *s == slot,
            _ => false,
        });
        if old_value_live {
            continue;
        }

        if src_idx < sinks.len() {
            sinks[src_idx] = Some(slot);
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use crate::vm::middle::{
        frame::FrameSlot,
        ssa_ir::{
            ir::{
                LocalSlotInfo, SsaBlock, SsaInst, SsaInstKind, SsaOperand, SsaProgram,
                SsaTerminator, SsaValue,
            },
            leaf::SsaLeafOp,
            target::SsaTarget,
        },
    };
    use crate::vm::wasm::primitive_op::PrimitiveOpKind;

    use super::plan_sinks;

    fn make_program(ops: alloc::vec::Vec<SsaInst>, value_count: usize) -> SsaProgram {
        SsaProgram {
            entry: SsaTarget(0),
            local_slot_types: vec![crate::value_type::ValueType::I32; 2],
            local_slot_info: vec![
                LocalSlotInfo {
                    is_param: true,
                    reads_before_write: true,
                };
                2
            ],
            blocks: vec![SsaBlock {
                id: SsaTarget(0),
                params: vec![],
                ops,
                terminator: SsaTerminator::Return { results: None },
            }],
            value_types: vec![crate::value_type::ValueType::I32; value_count],
            value_sink_local: vec![None; value_count],
            block_entry_cached_slots: vec![],
        block_cfg_origins: alloc::vec![],
        }
    }

    #[test]
    fn sinks_single_result_value_into_local_set_cache() {
        let mut program = make_program(
            vec![
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 42 })
                            .unwrap(),
                        args: vec![],
                        results: vec![SsaValue(0)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::LocalSetCache {
                        slot: FrameSlot(0),
                        src: SsaValue(0),
                    },
                },
            ],
            1,
        );
        plan_sinks(&mut program);
        assert_eq!(program.value_sink_local[0], Some(FrameSlot(0)));
    }

    #[test]
    fn does_not_sink_when_old_value_is_read_between_producer_and_set() {
        let mut program = make_program(
            vec![
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 42 })
                            .unwrap(),
                        args: vec![],
                        results: vec![SsaValue(0)],
                    },
                },
                // Read of fp[0] between producer and set — old value is live.
                SsaInst {
                    kind: SsaInstKind::LocalGetCache {
                        slot: FrameSlot(0),
                        dst: SsaValue(1),
                    },
                },
                SsaInst {
                    kind: SsaInstKind::LocalSetCache {
                        slot: FrameSlot(0),
                        src: SsaValue(0),
                    },
                },
            ],
            2,
        );
        plan_sinks(&mut program);
        assert_eq!(program.value_sink_local[0], None);
    }

    #[test]
    fn does_not_sink_across_call_barrier() {
        let mut program = make_program(
            vec![
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 42 })
                            .unwrap(),
                        args: vec![],
                        results: vec![SsaValue(0)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Call(
                        crate::vm::middle::ssa_ir::ir::SsaCallOp::CallDirect {
                            callee: 1,
                            args: crate::vm::middle::frame::FrameSpan {
                                start: FrameSlot(0),
                                count: 0,
                            },
                            results: crate::vm::middle::frame::FrameSpan {
                                start: FrameSlot(0),
                                count: 0,
                            },
                        },
                    ),
                },
                SsaInst {
                    kind: SsaInstKind::LocalSetCache {
                        slot: FrameSlot(0),
                        src: SsaValue(0),
                    },
                },
            ],
            1,
        );
        plan_sinks(&mut program);
        assert_eq!(program.value_sink_local[0], None);
    }

    #[test]
    fn sinks_when_different_slot_is_read_between_producer_and_set() {
        let mut program = make_program(
            vec![
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 42 })
                            .unwrap(),
                        args: vec![],
                        results: vec![SsaValue(0)],
                    },
                },
                // Read of fp[1] (different slot) — does not block sinking into fp[0].
                SsaInst {
                    kind: SsaInstKind::LocalGetCache {
                        slot: FrameSlot(1),
                        dst: SsaValue(1),
                    },
                },
                SsaInst {
                    kind: SsaInstKind::LocalSetCache {
                        slot: FrameSlot(0),
                        src: SsaValue(0),
                    },
                },
            ],
            2,
        );
        plan_sinks(&mut program);
        assert_eq!(program.value_sink_local[0], Some(FrameSlot(0)));
    }

    #[test]
    fn sinks_leaf_op_consuming_cache_value() {
        // get_cache fp[0] → v0, i32.add(v0, #1) → v1, set_cache fp[0] ← v1
        // v1 should sink into fp[0].
        let mut program = make_program(
            vec![
                SsaInst {
                    kind: SsaInstKind::LocalGetCache {
                        slot: FrameSlot(0),
                        dst: SsaValue(0),
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Add).unwrap(),
                        args: vec![
                            SsaOperand::Value(SsaValue(0)),
                            SsaOperand::Const(1),
                        ],
                        results: vec![SsaValue(1)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::LocalSetCache {
                        slot: FrameSlot(0),
                        src: SsaValue(1),
                    },
                },
            ],
            2,
        );
        plan_sinks(&mut program);
        // v0 is a LocalGetCache, not a single-result Value — not sinkable.
        assert_eq!(program.value_sink_local[0], None);
        // v1 is the add result — sinkable into fp[0]. The LocalGetCache of
        // fp[0] before the producer reads the OLD value of fp[0], but it's
        // before the producer, not between producer and set.
        assert_eq!(program.value_sink_local[1], Some(FrameSlot(0)));
    }
}
