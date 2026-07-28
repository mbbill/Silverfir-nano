//! Backward per-block sink planner.
//!
//! For each `CellSetCache { slot, src }`, this pass checks whether the
//! producer of `src` can write its result directly into the local's cache
//! register instead of a transient linear-value register. The machine
//! lowering layer uses the resulting `value_sink_cell` annotations via
//! `apply_sink_premap` to place results directly into cache registers.
//!
//! The sink is legal when:
//! - `src` is produced by a single-result `Value` or scalar-result `Call`
//!   instruction in the same block
//! - No barrier (Call) exists between the producer and the CellSetCache
//! - No `CellGetCache` or `CellGetSlot` of the same slot exists between
//!   the producer and the set (i.e., the old cached value is not read)

use crate::collections;

use crate::vm::jit::middle::{
    cell::CellId,
    ssa_ir::ir::{SsaBlock, SsaInst, SsaOp, SsaProgram, SsaValue},
};

/// Run the sink planner over all blocks in the program.
/// Populates `program.value_sink_cell` with sink annotations.
pub(super) fn plan_sinks(program: &mut SsaProgram) {
    let value_count = program.value_types.len();
    if value_count == 0 {
        return;
    }
    let mut sinks: collections::Vec<Option<CellId>> = collections::vec![None; value_count];

    for block in &program.blocks {
        plan_block_sinks(block, &mut sinks);
    }

    program.value_sink_cell = sinks;
}

/// Analyze one block for sink opportunities.
fn plan_block_sinks(block: &SsaBlock, sinks: &mut [Option<CellId>]) {
    let ops = &block.ops;
    if ops.is_empty() {
        return;
    }
    // Step 1: Record producer positions for single-result Value/Call ops. SSA
    // values are numbered across the whole function, but this analysis is
    // block-local, so offset the scratch array by this block's first producer.
    // Indexing by the absolute value number made every later block allocate and
    // clear storage for all values produced by earlier blocks.
    let Some((value_base, value_count)) = sinkable_result_range(ops) else {
        return;
    };
    let mut cache_aliases: collections::Vec<(SsaValue, CellId)> = ops
        .iter()
        .filter(|inst| inst.op == SsaOp::CELL_GET_CACHE && inst.result.is_some())
        .map(|inst| (inst.result, CellId(inst.meta)))
        .collect();
    cache_aliases.sort_unstable_by_key(|entry| entry.0);
    let mut producer_pos: collections::Vec<Option<u32>> = collections::vec![None; value_count];

    for (pos, inst) in ops.iter().enumerate() {
        // A primitive op always produces at most one result in the new flat
        // IR (tri-arg ops still have a single `result` slot), so treating
        // any primitive with a non-NONE result as single-result matches the
        // old `results.len() == 1` predicate.
        if let Some(result) = sinkable_result(inst) {
            if let Some(index) = local_value_index(value_base, result, producer_pos.len()) {
                producer_pos[index] = Some(pos as u32);
            }
        }
    }

    // Step 2: For each CellSetCache, check sink legality.
    for (set_pos, inst) in ops.iter().enumerate() {
        if inst.op != SsaOp::CELL_SET_CACHE {
            continue;
        }
        let slot = CellId(inst.meta);
        let src = inst.args[0]
            .as_value()
            .expect("CellSetCache src must be an SsaValue");

        let Some(local_src_idx) = local_value_index(value_base, src, producer_pos.len()) else {
            continue;
        };
        let Some(prod_pos) = producer_pos[local_src_idx] else {
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
        let src_idx = src.0 as usize;
        if sinks.get(src_idx).copied().flatten().is_some() {
            continue;
        }

        // Old value of this local must not be read between producer and set.
        let old_value_live = ops[prod_pos + 1..set_pos].iter().any(|i| {
            matches!(i.op, SsaOp::CELL_GET_CACHE | SsaOp::CELL_GET_SLOT) && CellId(i.meta) == slot
        });
        if old_value_live {
            continue;
        }

        // The discipline pass publishes a cached local's older live snapshots
        // immediately before overwriting that cache lane. Sinking the producer
        // would move the overwrite earlier than those Spill ops and destroy
        // the value before it reaches its frame slot. Keep the ordinary
        // transient result in this case; machine lowering can then execute the
        // planned spills before the later CellSetCache.
        let old_alias_spilled = ops[prod_pos + 1..set_pos].iter().any(|i| {
            if i.op != SsaOp::SPILL {
                return false;
            }
            i.args[0].as_value().and_then(|value| {
                cache_aliases
                    .binary_search_by_key(&value, |entry| entry.0)
                    .ok()
                    .map(|index| cache_aliases[index].1)
            }) == Some(slot)
        });
        if old_alias_spilled {
            continue;
        }

        if src_idx < sinks.len() {
            sinks[src_idx] = Some(slot);
        }
    }
}

#[inline]
fn sinkable_result(inst: &SsaInst) -> Option<SsaValue> {
    ((inst.op.is_primitive() || inst.op == SsaOp::CALL) && inst.result.is_some())
        .then_some(inst.result)
}

fn sinkable_result_range(ops: &[SsaInst]) -> Option<(SsaValue, usize)> {
    let mut min = u32::MAX;
    let mut max = 0u32;
    for result in ops.iter().filter_map(sinkable_result) {
        min = min.min(result.0);
        max = max.max(result.0);
    }
    (min != u32::MAX).then(|| (SsaValue(min), (max - min) as usize + 1))
}

#[inline]
fn local_value_index(base: SsaValue, value: SsaValue, len: usize) -> Option<usize> {
    let index = value.0.checked_sub(base.0)? as usize;
    (index < len).then_some(index)
}

#[cfg(test)]
mod tests {
    use crate::collections;
    use crate::value_type::ValueType;

    use crate::vm::jit::middle::{
        cell::CellId,
        frame::FrameSpan,
        ssa_ir::{
            ir::{
                CellInfo, SsaBlock, SsaCallArgs, SsaCallOp, SsaInst, SsaOperand, SsaProgram,
                SsaTerminator, SsaValue,
            },
            target::SsaTarget,
        },
    };
    use crate::vm::jit::wasm::primitive_op::PrimitiveOpKind;

    use super::plan_sinks;
    use crate::vm::jit::middle::frame::FrameSlot;

    fn make_program(ops: collections::Vec<SsaInst>, value_count: usize) -> SsaProgram {
        SsaProgram {
            cell_homes: collections::Vec::new(),
            entry: SsaTarget(0),
            cell_types: collections::vec![ValueType::I32; 2],
            result_types: collections::Vec::new(),
            cell_info: collections::vec![
                CellInfo {
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
            value_types: collections::vec![ValueType::I32; value_count],
            value_sink_cell: collections::vec![None; value_count],
            block_entry_cached_cells: collections::vec![],
            block_entry_cache_requirements: collections::vec![],
            preferred_preserved: collections::Vec::new(),
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
        let set_cache = SsaInst::cell_set_cache(CellId(0), SsaValue(0));
        program.blocks[0].ops = collections::vec![i32_const, set_cache];

        plan_sinks(&mut program);
        assert_eq!(program.value_sink_cell[0], Some(CellId(0)));
    }

    #[test]
    fn sinks_scalar_call_result_into_local_set_cache() {
        let mut program = make_program(collections::Vec::new(), 1);
        let call_idx = program.push_call_op(SsaCallOp::CallDirect {
            callee: 1,
            args: SsaCallArgs {
                frame_base: FrameSlot(0),
                total_params: 0,
                param_types: collections::Vec::new(),
                stack_prefix_count: 0,
                live_suffix: collections::Vec::new(),
            },
            results: FrameSpan {
                start: FrameSlot(0),
                count: 1,
            },
            result_types: collections::vec![ValueType::I32],
        });
        let call = SsaInst::call_result(call_idx, SsaValue(0));
        let set_cache = SsaInst::cell_set_cache(CellId(0), SsaValue(0));
        program.blocks[0].ops = collections::vec![call, set_cache];

        plan_sinks(&mut program);
        assert_eq!(program.value_sink_cell[0], Some(CellId(0)));
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
        let get_cache = SsaInst::cell_get_cache(CellId(0), SsaValue(1));
        let set_cache = SsaInst::cell_set_cache(CellId(0), SsaValue(0));
        program.blocks[0].ops = collections::vec![i32_const, get_cache, set_cache];

        plan_sinks(&mut program);
        assert_eq!(program.value_sink_cell[0], None);
    }

    #[test]
    fn does_not_sink_before_an_older_cache_alias_is_spilled() {
        let mut program = make_program(collections::Vec::new(), 2);
        let old_value = SsaInst::cell_get_cache(CellId(0), SsaValue(0));
        let replacement = prim_inst(
            &mut program,
            PrimitiveOpKind::I32Const { value: 42 },
            SsaValue(1),
            [SsaOperand::NONE, SsaOperand::NONE],
        );
        let spill_old = SsaInst::spill(FrameSlot(0), SsaValue(0));
        let set_cache = SsaInst::cell_set_cache(CellId(0), SsaValue(1));
        program.blocks[0].ops = collections::vec![old_value, replacement, spill_old, set_cache];

        plan_sinks(&mut program);
        assert_eq!(program.value_sink_cell[1], None);
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
        let call_idx = program.push_call_op(SsaCallOp::CallDirect {
            callee: 1,
            args: SsaCallArgs {
                frame_base: FrameSlot(0),
                total_params: 0,
                param_types: collections::Vec::new(),
                stack_prefix_count: 0,
                live_suffix: collections::Vec::new(),
            },
            results: FrameSpan {
                start: FrameSlot(0),
                count: 0,
            },
            result_types: collections::Vec::new(),
        });
        let call = SsaInst::call(call_idx);
        let set_cache = SsaInst::cell_set_cache(CellId(0), SsaValue(0));
        program.blocks[0].ops = collections::vec![i32_const, call, set_cache];

        plan_sinks(&mut program);
        assert_eq!(program.value_sink_cell[0], None);
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
        let get_cache = SsaInst::cell_get_cache(CellId(1), SsaValue(1));
        let set_cache = SsaInst::cell_set_cache(CellId(0), SsaValue(0));
        program.blocks[0].ops = collections::vec![i32_const, get_cache, set_cache];

        plan_sinks(&mut program);
        assert_eq!(program.value_sink_cell[0], Some(CellId(0)));
    }

    #[test]
    fn sinks_leaf_op_consuming_cache_value() {
        // get_cache fp[0] → v0, i32.add(v0, #1) → v1, set_cache fp[0] ← v1
        // v1 should sink into fp[0].
        let mut program = make_program(collections::Vec::new(), 2);
        let get_cache = SsaInst::cell_get_cache(CellId(0), SsaValue(0));
        let const_one = program.intern_const(1_u64);
        let add = prim_inst(
            &mut program,
            PrimitiveOpKind::I32Add,
            SsaValue(1),
            [SsaOperand::value(SsaValue(0)), const_one],
        );
        let set_cache = SsaInst::cell_set_cache(CellId(0), SsaValue(1));
        program.blocks[0].ops = collections::vec![get_cache, add, set_cache];

        plan_sinks(&mut program);
        // v0 is a CellGetCache, not a single-result Value — not sinkable.
        assert_eq!(program.value_sink_cell[0], None);
        // v1 is the add result — sinkable into fp[0]. The CellGetCache of
        // fp[0] before the producer reads the OLD value of fp[0], but it's
        // before the producer, not between producer and set.
        assert_eq!(program.value_sink_cell[1], Some(CellId(0)));
    }
}
