//! Backward per-block sink planner.
//!
//! For each `LocalSet { slot, src, version }`, this pass checks whether the
//! producer of `src` can write its result directly into the local's home
//! instead of a transient register. The `LocalSet` can then be elided.
//!
//! The sink is legal when:
//! - `src` is produced by a single-result `Value` instruction in the same block
//! - The producer is "targetable" (not a boundary, not already sunk)
//! - No barrier (Boundary) exists between the producer and the LocalSet
//! - The old version of the local (version - 1) is dead at the producer —
//!   meaning no `LocalGet` reads the old version between the producer and
//!   the LocalSet
//!
//! This pass is register-agnostic. The machine lowering layer decides whether
//! to exploit a sink annotation based on whether the target local is cached
//! in a register.

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

    // Step 1: Build a map from SsaValue -> (producer_position, is_targetable).
    // A value is targetable if it is the sole result of a Value instruction.
    let mut producer_pos: Vec<Option<u32>> = Vec::new();
    let mut max_val: u32 = 0;

    for (pos, inst) in ops.iter().enumerate() {
        match &inst.kind {
            SsaInstKind::Value { results, .. } => {
                for r in results {
                    if r.0 >= max_val {
                        max_val = r.0 + 1;
                    }
                }
            }
            SsaInstKind::LocalGet { dst, .. } | SsaInstKind::Fill { dst, .. } => {
                if dst.0 >= max_val {
                    max_val = dst.0 + 1;
                }
            }
            _ => {}
        }
        let _ = pos; // used below
    }

    producer_pos.resize(max_val as usize, None);

    // Record producer positions and whether each value is a single-result Value op.
    let mut is_single_result = vec![false; max_val as usize];

    for (pos, inst) in ops.iter().enumerate() {
        let produced = match &inst.kind {
            SsaInstKind::Value { results, .. } if results.len() == 1 => Some(results[0]),
            SsaInstKind::Fill { dst, .. } => Some(*dst),
            _ => None,
        };
        if let Some(r) = produced {
            if (r.0 as usize) < producer_pos.len() {
                producer_pos[r.0 as usize] = Some(pos as u32);
                is_single_result[r.0 as usize] = true;
            }
        }
    }

    // Step 2: Identify barrier positions. A barrier is a Boundary instruction.
    // We'll check for barriers in ranges on the fly.

    // Step 3: For each LocalSet (walking forward is fine — we check liveness
    // of the old version between producer and set), check sink legality.
    //
    // For the old-version liveness check, we need to know if any LocalGet
    // in the range (producer_pos, local_set_pos) reads the same slot with
    // the old version. Since versions are monotonic within a block, the old
    // version is `new_version - 1`. Any LocalGet of this slot between
    // producer and set must have been emitted before the set, so it reads
    // whatever version was current at that point. If version > 0, the old
    // version was `version - 1`, and any LocalGet of the same slot between
    // producer and set reads that old version.

    for (set_pos, inst) in ops.iter().enumerate() {
        let (slot, src, _version) = match &inst.kind {
            SsaInstKind::LocalSet { slot, src, version } => (*slot, *src, *version),
            _ => continue,
        };

        // src must be an SSA value produced by a targetable Value in this block.
        let src_idx = src.0 as usize;
        if src_idx >= producer_pos.len() || !is_single_result[src_idx] {
            continue;
        }
        let Some(prod_pos) = producer_pos[src_idx] else {
            continue;
        };
        let prod_pos = prod_pos as usize;

        // Producer must be before the set (same block, forward order).
        if prod_pos >= set_pos {
            continue;
        }

        // Check: no barrier between producer and set.
        let has_barrier = ops[prod_pos + 1..set_pos]
            .iter()
            .any(|i| matches!(i.kind, SsaInstKind::Boundary(_)));
        if has_barrier {
            continue;
        }

        // Check: the value must not already be sunk elsewhere.
        if sinks.get(src_idx).copied().flatten().is_some() {
            continue;
        }

        // Check: old version of this local is dead between producer and set.
        // The old version is `version - 1`. If version == 0, this is the first
        // write — there's no prior version to worry about from within this block,
        // but there may be a live value from a preceding block. We conservatively
        // skip version 0 sinking since we can't prove the entry version is dead
        // without cross-block analysis.
        //
        // Actually: version 0 means this is the first local.set in the block.
        // The "old version" is whatever was in the local at block entry. If any
        // LocalGet of this slot exists between producer and set, it reads that
        // entry version, so the old value is live and we cannot sink. If none
        // exists, sinking is safe regardless of version number.
        let old_version_live = ops[prod_pos + 1..set_pos].iter().any(|i| {
            matches!(
                &i.kind,
                SsaInstKind::LocalGet { slot: s, .. } if *s == slot
            )
        });
        if old_version_live {
            continue;
        }

        // All checks passed — annotate the sink.
        // Note: we do NOT reject sinks where the producer reads old x as an
        // input operand. In linear SSA, the producer consumes its inputs, so
        // old x dies at the producer's def point. The producer can safely
        // overwrite x's home with its result.
        if src_idx < sinks.len() {
            sinks[src_idx] = Some(slot);
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use crate::vm::middle::{
        frame::FrameSlot,
        ssa_ir::{
            ir::{
                SsaBlock, SsaInst, SsaInstKind, SsaOperand, SsaProgram, SsaTerminator, SsaValue,
            },
            leaf::SsaLeafOp,
            target::SsaTarget,
        },
    };
    use crate::vm::wasm::primitive_op::PrimitiveOpKind;

    use super::plan_sinks;

    fn make_program(ops: Vec<SsaInst>) -> SsaProgram {
        // Determine the max value index for type/home arrays.
        let mut max_val: u32 = 0;
        for inst in &ops {
            match &inst.kind {
                SsaInstKind::Value { results, args, .. } => {
                    for r in results {
                        max_val = max_val.max(r.0 + 1);
                    }
                    for a in args {
                        if let SsaOperand::Value(v) = a {
                            max_val = max_val.max(v.0 + 1);
                        }
                    }
                }
                SsaInstKind::LocalGet { dst, .. } | SsaInstKind::Fill { dst, .. } => {
                    max_val = max_val.max(dst.0 + 1);
                }
                SsaInstKind::LocalSet { src, .. } | SsaInstKind::Spill { src, .. } => {
                    max_val = max_val.max(src.0 + 1);
                }
                SsaInstKind::Boundary(_) => {}
            }
        }
        let n = max_val as usize;
        SsaProgram {
            entry: SsaTarget(0),
            local_cache: Default::default(),
            blocks: vec![SsaBlock {
                id: SsaTarget(0),
                params: Vec::new(),
                ops,
                terminator: SsaTerminator::Return { results: None },
            }],
            value_types: vec![crate::value_type::ValueType::I32; n],
            value_homes: vec![Default::default(); n],
            value_sink_local: vec![None; n],
        }
    }

    #[test]
    fn sinks_simple_add_into_local() {
        // local.get x -> v0
        // i32.const 1 -> v1
        // i32.add v0, v1 -> v2
        // local.set x, v2 (version 1)
        //
        // old x (version 0) is read at prod_pos-1 (LocalGet), but the
        // producer of v2 is the i32.add at pos 2. Between pos 2 and set
        // at pos 3, there's no LocalGet of slot 0. So sinking is legal.
        let mut program = make_program(vec![
            SsaInst {
                kind: SsaInstKind::LocalGet {
                    slot: FrameSlot(0),
                    dst: SsaValue(0),
                },
            },
            SsaInst {
                kind: SsaInstKind::Value {
                    op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 1 }).unwrap(),
                    args: Vec::new(),
                    results: vec![SsaValue(1)],
                },
            },
            SsaInst {
                kind: SsaInstKind::Value {
                    op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Add).unwrap(),
                    args: vec![
                        SsaOperand::Value(SsaValue(0)),
                        SsaOperand::Value(SsaValue(1)),
                    ],
                    results: vec![SsaValue(2)],
                },
            },
            SsaInst {
                kind: SsaInstKind::LocalSet {
                    slot: FrameSlot(0),
                    src: SsaValue(2),
                    version: 1,
                },
            },
        ]);

        plan_sinks(&mut program);
        assert_eq!(program.value_sink_local[2], Some(FrameSlot(0)));
    }

    #[test]
    fn does_not_sink_when_old_version_live() {
        // i32.const 1 -> v0
        // i32.add v0, v0 -> v1
        // local.get x -> v2          <-- reads old x between producer and set
        // local.set x, v1 (version 1)
        let mut program = make_program(vec![
            SsaInst {
                kind: SsaInstKind::Value {
                    op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 1 }).unwrap(),
                    args: Vec::new(),
                    results: vec![SsaValue(0)],
                },
            },
            SsaInst {
                kind: SsaInstKind::Value {
                    op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Add).unwrap(),
                    args: vec![
                        SsaOperand::Value(SsaValue(0)),
                        SsaOperand::Value(SsaValue(0)),
                    ],
                    results: vec![SsaValue(1)],
                },
            },
            SsaInst {
                kind: SsaInstKind::LocalGet {
                    slot: FrameSlot(0),
                    dst: SsaValue(2),
                },
            },
            SsaInst {
                kind: SsaInstKind::LocalSet {
                    slot: FrameSlot(0),
                    src: SsaValue(1),
                    version: 1,
                },
            },
        ]);

        plan_sinks(&mut program);
        assert_eq!(program.value_sink_local[1], None);
    }

    #[test]
    fn does_not_sink_across_barrier() {
        // i32.const 1 -> v0
        // boundary (e.g., memory.grow)
        // local.set x, v0 (version 1)
        let mut program = make_program(vec![
            SsaInst {
                kind: SsaInstKind::Value {
                    op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 1 }).unwrap(),
                    args: Vec::new(),
                    results: vec![SsaValue(0)],
                },
            },
            SsaInst {
                kind: SsaInstKind::Boundary(
                    crate::vm::middle::ssa_ir::ir::SsaBoundaryOp::DataDrop { data_idx: 0 },
                ),
            },
            SsaInst {
                kind: SsaInstKind::LocalSet {
                    slot: FrameSlot(0),
                    src: SsaValue(0),
                    version: 1,
                },
            },
        ]);

        plan_sinks(&mut program);
        assert_eq!(program.value_sink_local[0], None);
    }

    #[test]
    fn sinks_when_producer_reads_same_local_as_input() {
        // local.get x -> v0
        // i32.add v0, v0 -> v1  (producer reads old x via v0)
        // local.set x, v1 (version 1)
        //
        // This IS legal: the producer consumes v0 (old x) as an input and
        // produces v1. Even though old x is in the same local's home, the
        // machine lowering handles read-before-write for the same register.
        // No LocalGet of x exists between producer (pos 1) and set (pos 2).
        let mut program = make_program(vec![
            SsaInst {
                kind: SsaInstKind::LocalGet {
                    slot: FrameSlot(0),
                    dst: SsaValue(0),
                },
            },
            SsaInst {
                kind: SsaInstKind::Value {
                    op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Add).unwrap(),
                    args: vec![
                        SsaOperand::Value(SsaValue(0)),
                        SsaOperand::Value(SsaValue(0)),
                    ],
                    results: vec![SsaValue(1)],
                },
            },
            SsaInst {
                kind: SsaInstKind::LocalSet {
                    slot: FrameSlot(0),
                    src: SsaValue(1),
                    version: 1,
                },
            },
        ]);

        plan_sinks(&mut program);
        assert_eq!(program.value_sink_local[1], Some(FrameSlot(0)));
    }
}
