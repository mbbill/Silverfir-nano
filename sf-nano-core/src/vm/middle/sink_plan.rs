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

use crate::collections;

use crate::vm::middle::{
    frame::FrameSlot,
    ssa_ir::ir::{SsaBlock, SsaOp, SsaProgram},
};

/// Run the sink planner over all blocks in the program.
/// Populates `program.value_sink_local` with sink annotations.
pub(super) fn plan_sinks(program: &mut SsaProgram) {
    let value_count = program.value_types.len();
    if value_count == 0 {
        return;
    }
    let mut sinks: collections::Vec<Option<FrameSlot>> = collections::vec![None; value_count];

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
    let mut producer_pos: collections::Vec<Option<u32>> = collections::Vec::new();
    let mut max_val: u32 = 0;

    for inst in ops.iter() {
        if inst.op.is_primitive() {
            if inst.result.is_some() && inst.result.0 >= max_val {
                max_val = inst.result.0 + 1;
            }
        } else {
            match inst.op {
                SsaOp::LOCAL_GET_CACHE | SsaOp::LOCAL_GET_SLOT | SsaOp::FILL => {
                    let dst = inst.result;
                    if dst.is_some() && dst.0 >= max_val {
                        max_val = dst.0 + 1;
                    }
                }
                _ => {}
            }
        }
    }

    producer_pos.resize(max_val as usize, None);
    let mut is_single_result = collections::vec![false; max_val as usize];

    for (pos, inst) in ops.iter().enumerate() {
        // A primitive op always produces at most one result in the new flat
        // IR (tri-arg ops still have a single `result` slot), so treating
        // any primitive with a non-NONE result as single-result matches the
        // old `results.len() == 1` predicate.
        let produced = if inst.op.is_primitive() && inst.result.is_some() {
            Some(inst.result)
        } else {
            None
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
        if inst.op != SsaOp::LOCAL_SET_CACHE {
            continue;
        }
        let slot = FrameSlot(inst.meta);
        let src = inst.args[0]
            .as_value()
            .expect("LocalSetCache src must be an SsaValue");

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
            .any(|i| i.op == SsaOp::CALL);
        if has_barrier {
            continue;
        }

        // Value must not already be sunk elsewhere.
        if sinks.get(src_idx).copied().flatten().is_some() {
            continue;
        }

        // Old value of this local must not be read between producer and set.
        let old_value_live = ops[prod_pos + 1..set_pos].iter().any(|i| {
            matches!(i.op, SsaOp::LOCAL_GET_CACHE | SsaOp::LOCAL_GET_SLOT)
                && FrameSlot(i.meta) == slot
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
    use crate::collections;

    use crate::vm::middle::{
        frame::FrameSlot,
        ssa_ir::{
            ir::{
                LocalSlotInfo, SsaBlock, SsaInst, SsaOperand, SsaProgram, SsaTerminator, SsaValue,
            },
            target::SsaTarget,
        },
    };
    use crate::vm::wasm::primitive_op::PrimitiveOpKind;

    use super::plan_sinks;

    fn make_program(ops: collections::Vec<SsaInst>, value_count: usize) -> SsaProgram {
        SsaProgram {
            entry: SsaTarget(0),
            local_slot_types: collections::vec![crate::value_type::ValueType::I32; 2],
            local_slot_info: collections::vec![
                LocalSlotInfo {
                    is_param: true,
                    reads_before_write: true,
                };
                2
            ],
            blocks: collections::vec![SsaBlock {
                id: SsaTarget(0),
                params: collections::vec![],
                ops,
                extra_args: collections::Vec::new(),
                terminator: SsaTerminator::Return { results: None },
            }],
            value_types: collections::vec![crate::value_type::ValueType::I32; value_count],
            value_sink_local: collections::vec![None; value_count],
            block_entry_cached_slots: collections::vec![],
            block_cfg_origins: collections::vec![],
            const_pool: collections::Vec::new(),
            primitive_pool: collections::Vec::new(),
            call_ops: collections::Vec::new(),
        }
    }

    /// Build a primitive Value instruction from raw parts, interning the op
    /// kind into the program's primitive pool.
    fn prim_inst(
        program: &mut SsaProgram,
        kind: PrimitiveOpKind,
        result: SsaValue,
        args: [SsaOperand; 2],
    ) -> SsaInst {
        let pool_idx = program.intern_primitive(kind).unwrap();
        SsaInst::primitive(pool_idx, result, args, 0)
    }

    #[test]
    fn sinks_single_result_value_into_local_set_cache() {
        let mut program = make_program(collections::Vec::new(), 1);
        let i32_const = prim_inst(
            &mut program,
            PrimitiveOpKind::I32Const { value: 42 },
            SsaValue(0),
            [SsaOperand::NONE, SsaOperand::NONE],
        );
        let set_cache = SsaInst::local_set_cache(FrameSlot(0), SsaValue(0));
        program.blocks[0].ops = collections::vec![i32_const, set_cache];

        plan_sinks(&mut program);
        assert_eq!(program.value_sink_local[0], Some(FrameSlot(0)));
    }

    #[test]
    fn does_not_sink_when_old_value_is_read_between_producer_and_set() {
        let mut program = make_program(collections::Vec::new(), 2);
        let i32_const = prim_inst(
            &mut program,
            PrimitiveOpKind::I32Const { value: 42 },
            SsaValue(0),
            [SsaOperand::NONE, SsaOperand::NONE],
        );
        // Read of fp[0] between producer and set — old value is live.
        let get_cache = SsaInst::local_get_cache(FrameSlot(0), SsaValue(1));
        let set_cache = SsaInst::local_set_cache(FrameSlot(0), SsaValue(0));
        program.blocks[0].ops = collections::vec![i32_const, get_cache, set_cache];

        plan_sinks(&mut program);
        assert_eq!(program.value_sink_local[0], None);
    }

    #[test]
    fn does_not_sink_across_call_barrier() {
        let mut program = make_program(collections::Vec::new(), 1);
        let i32_const = prim_inst(
            &mut program,
            PrimitiveOpKind::I32Const { value: 42 },
            SsaValue(0),
            [SsaOperand::NONE, SsaOperand::NONE],
        );
        let call_idx = program.push_call_op(crate::vm::middle::ssa_ir::ir::SsaCallOp::CallDirect {
            callee: 1,
            args: crate::vm::middle::frame::FrameSpan {
                start: FrameSlot(0),
                count: 0,
            },
            results: crate::vm::middle::frame::FrameSpan {
                start: FrameSlot(0),
                count: 0,
            },
        });
        let call = SsaInst::call(call_idx);
        let set_cache = SsaInst::local_set_cache(FrameSlot(0), SsaValue(0));
        program.blocks[0].ops = collections::vec![i32_const, call, set_cache];

        plan_sinks(&mut program);
        assert_eq!(program.value_sink_local[0], None);
    }

    #[test]
    fn sinks_when_different_slot_is_read_between_producer_and_set() {
        let mut program = make_program(collections::Vec::new(), 2);
        let i32_const = prim_inst(
            &mut program,
            PrimitiveOpKind::I32Const { value: 42 },
            SsaValue(0),
            [SsaOperand::NONE, SsaOperand::NONE],
        );
        // Read of fp[1] (different slot) — does not block sinking into fp[0].
        let get_cache = SsaInst::local_get_cache(FrameSlot(1), SsaValue(1));
        let set_cache = SsaInst::local_set_cache(FrameSlot(0), SsaValue(0));
        program.blocks[0].ops = collections::vec![i32_const, get_cache, set_cache];

        plan_sinks(&mut program);
        assert_eq!(program.value_sink_local[0], Some(FrameSlot(0)));
    }

    #[test]
    fn sinks_leaf_op_consuming_cache_value() {
        // get_cache fp[0] → v0, i32.add(v0, #1) → v1, set_cache fp[0] ← v1
        // v1 should sink into fp[0].
        let mut program = make_program(collections::Vec::new(), 2);
        let get_cache = SsaInst::local_get_cache(FrameSlot(0), SsaValue(0));
        let const_one = program.intern_const(1_u64);
        let add = prim_inst(
            &mut program,
            PrimitiveOpKind::I32Add,
            SsaValue(1),
            [SsaOperand::value(SsaValue(0)), const_one],
        );
        let set_cache = SsaInst::local_set_cache(FrameSlot(0), SsaValue(1));
        program.blocks[0].ops = collections::vec![get_cache, add, set_cache];

        plan_sinks(&mut program);
        // v0 is a LocalGetCache, not a single-result Value — not sinkable.
        assert_eq!(program.value_sink_local[0], None);
        // v1 is the add result — sinkable into fp[0]. The LocalGetCache of
        // fp[0] before the producer reads the OLD value of fp[0], but it's
        // before the producer, not between producer and set.
        assert_eq!(program.value_sink_local[1], Some(FrameSlot(0)));
    }
}
