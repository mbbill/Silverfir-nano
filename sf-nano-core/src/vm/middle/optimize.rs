use alloc::vec;
use alloc::vec::Vec;

use crate::vm::middle::{
    ssa_ir::ir::{SsaBlock, SsaEdge, SsaInstKind, SsaOperand, SsaProgram, SsaTerminator, SsaValue},
    frame::FrameLayoutPlan,
};

pub(super) fn optimize_ssa(program: &mut SsaProgram, frame: FrameLayoutPlan) {
    for block in &mut program.blocks {
        forward_slot_values(block, frame, &program.value_types);
    }
    for block in &mut program.blocks {
        fold_constants_into_operands(block);
    }
}

/// Absorb single-use constant definitions into the consuming instruction's
/// operand slot, replacing `SsaOperand::Value(v)` with `SsaOperand::Const(bits)`.
///
/// The spill planner already budgeted a transient register for the absorbed
/// constant.  Backends that cannot encode the immediate natively **must**
/// materialize it into a transient register — guaranteed available from the
/// pre-optimization register budget.
fn fold_constants_into_operands(block: &mut SsaBlock) {
    use crate::vm::wasm::primitive_op::PrimitiveOpKind;

    let max_val = max_value_index(block)
        .map(|v| v.0 as usize + 1)
        .unwrap_or(0);
    if max_val == 0 {
        return;
    }

    // Identify constants and values used in terminator edge bindings.
    let mut known_const: Vec<Option<u64>> = vec![None; max_val];
    let mut used_in_terminator = vec![false; max_val];
    for inst in &block.ops {
        if let SsaInstKind::Value { op, results, .. } = &inst.kind {
            if let Some(r) = results.first() {
                let bits = match op.primitive() {
                    PrimitiveOpKind::I32Const { value } => Some(*value as u64),
                    PrimitiveOpKind::I64Const { value } => Some(*value),
                    PrimitiveOpKind::F32Const { value } => Some(*value as u64),
                    PrimitiveOpKind::F64Const { value } => Some(*value),
                    _ => None,
                };
                if let Some(bits) = bits {
                    if let Some(slot) = known_const.get_mut(r.0 as usize) {
                        *slot = Some(bits);
                    }
                }
            }
        }
    }
    mark_terminator_uses(&block.terminator, &mut used_in_terminator);

    // Rewrite operands in arithmetic/comparison ops only.
    // A constant can be folded if it has no terminator uses (linear SSA
    // guarantees exactly one use in the op stream, so no terminator use
    // means total uses == 1).
    for inst in &mut block.ops {
        if let SsaInstKind::Value { op, args, .. } = &mut inst.kind {
            if !can_accept_const_operand(op.primitive()) {
                continue;
            }
            for operand in args.iter_mut() {
                if let SsaOperand::Value(v) = operand {
                    let idx = v.0 as usize;
                    if let Some(Some(bits)) = known_const.get(idx) {
                        if !used_in_terminator.get(idx).copied().unwrap_or(true) {
                            *operand = SsaOperand::Const(*bits);
                        }
                    }
                }
            }
        }
    }

    // Remove dead const definitions: a const is dead when no
    // SsaOperand::Value reference to it remains in the block.
    let mut still_used = vec![false; max_val];
    for inst in &block.ops {
        match &inst.kind {
            SsaInstKind::Value { args, .. } => {
                for operand in args {
                    if let SsaOperand::Value(v) = operand {
                        if let Some(slot) = still_used.get_mut(v.0 as usize) {
                            *slot = true;
                        }
                    }
                }
            }
            SsaInstKind::StoreSlot { src, .. } => {
                if let Some(slot) = still_used.get_mut(src.0 as usize) {
                    *slot = true;
                }
            }
            SsaInstKind::LoadSlot { .. } | SsaInstKind::Boundary(_) => {}
        }
    }
    // Terminator uses also keep the const alive.
    for (i, used) in used_in_terminator.iter().enumerate() {
        if *used {
            if let Some(slot) = still_used.get_mut(i) {
                *slot = true;
            }
        }
    }
    block.ops.retain(|inst| {
        if let SsaInstKind::Value { args, results, .. } = &inst.kind {
            if args.is_empty() && results.len() == 1 {
                let idx = results[0].0 as usize;
                if known_const.get(idx).copied().flatten().is_some()
                    && !still_used.get(idx).copied().unwrap_or(true)
                {
                    return false;
                }
            }
        }
        true
    });
}

/// Returns true if the given primitive op's machine lowering supports
/// `SsaOperand::Const` in its argument slots.  Only pure arithmetic,
/// bitwise, comparison, unary, and conversion ops qualify — NOT memory,
/// table, global, select, drop, or reference ops.
fn can_accept_const_operand(kind: &crate::vm::wasm::primitive_op::PrimitiveOpKind) -> bool {
    use crate::vm::wasm::primitive_op::PrimitiveOpKind as P;
    matches!(
        kind,
        // i32 binary
        P::I32Add | P::I32Sub | P::I32Mul | P::I32DivS | P::I32DivU
        | P::I32RemS | P::I32RemU | P::I32And | P::I32Or | P::I32Xor
        | P::I32Shl | P::I32ShrS | P::I32ShrU | P::I32Rotl | P::I32Rotr
        // i64 binary
        | P::I64Add | P::I64Sub | P::I64Mul | P::I64DivS | P::I64DivU
        | P::I64RemS | P::I64RemU | P::I64And | P::I64Or | P::I64Xor
        | P::I64Shl | P::I64ShrS | P::I64ShrU | P::I64Rotl | P::I64Rotr
        // f32/f64 binary
        | P::F32Add | P::F32Sub | P::F32Mul | P::F32Div | P::F32Min | P::F32Max | P::F32Copysign
        | P::F64Add | P::F64Sub | P::F64Mul | P::F64Div | P::F64Min | P::F64Max | P::F64Copysign
        // i32/i64 compare
        | P::I32Eq | P::I32Ne | P::I32LtS | P::I32LtU | P::I32GtS | P::I32GtU
        | P::I32LeS | P::I32LeU | P::I32GeS | P::I32GeU
        | P::I64Eq | P::I64Ne | P::I64LtS | P::I64LtU | P::I64GtS | P::I64GtU
        | P::I64LeS | P::I64LeU | P::I64GeS | P::I64GeU
        // f32/f64 compare
        | P::F32Eq | P::F32Ne | P::F32Lt | P::F32Gt | P::F32Le | P::F32Ge
        | P::F64Eq | P::F64Ne | P::F64Lt | P::F64Gt | P::F64Le | P::F64Ge
        // i32/i64 unary
        | P::I32Eqz | P::I32Clz | P::I32Ctz | P::I32Popcnt
        | P::I64Eqz | P::I64Clz | P::I64Ctz | P::I64Popcnt
        | P::I32Extend8S | P::I32Extend16S
        | P::I64Extend8S | P::I64Extend16S | P::I64Extend32S
        // f32/f64 unary
        | P::F32Abs | P::F32Neg | P::F32Ceil | P::F32Floor | P::F32Trunc | P::F32Nearest | P::F32Sqrt
        | P::F64Abs | P::F64Neg | P::F64Ceil | P::F64Floor | P::F64Trunc | P::F64Nearest | P::F64Sqrt
        // conversions
        | P::I32WrapI64 | P::I64ExtendI32S | P::I64ExtendI32U
        | P::I32TruncF32S | P::I32TruncF32U | P::I32TruncF64S | P::I32TruncF64U
        | P::I64TruncF32S | P::I64TruncF32U | P::I64TruncF64S | P::I64TruncF64U
        | P::I32TruncSatF32S | P::I32TruncSatF32U | P::I32TruncSatF64S | P::I32TruncSatF64U
        | P::I64TruncSatF32S | P::I64TruncSatF32U | P::I64TruncSatF64S | P::I64TruncSatF64U
        | P::F32ConvertI32S | P::F32ConvertI32U | P::F32ConvertI64S | P::F32ConvertI64U
        | P::F64ConvertI32S | P::F64ConvertI32U | P::F64ConvertI64S | P::F64ConvertI64U
        | P::F32DemoteF64 | P::F64PromoteF32
        | P::I32ReinterpretF32 | P::I64ReinterpretF64 | P::F32ReinterpretI32 | P::F64ReinterpretI64
    )
}

/// Mark values that appear in terminator edge bindings or as branch/table
/// conditions.  These values need registers and cannot be folded to Const.
fn mark_terminator_uses(term: &SsaTerminator, used: &mut [bool]) {
    let mut mark = |v: SsaValue| {
        if let Some(slot) = used.get_mut(v.0 as usize) {
            *slot = true;
        }
    };
    match term {
        SsaTerminator::Goto(edge) => {
            for b in &edge.bindings { mark(b.value); }
        }
        SsaTerminator::Branch { cond, then_edge, else_edge } => {
            mark(*cond);
            for b in &then_edge.bindings { mark(b.value); }
            for b in &else_edge.bindings { mark(b.value); }
        }
        SsaTerminator::BrTable { index, entries } => {
            mark(*index);
            for edge in entries {
                for b in &edge.bindings { mark(b.value); }
            }
        }
        SsaTerminator::Return { .. } | SsaTerminator::TrapUnreachable => {}
    }
}

fn forward_slot_values(
    block: &mut SsaBlock,
    frame: FrameLayoutPlan,
    value_types: &[crate::value_type::ValueType],
) {
    let alias_len = max_value_index(block)
        .map(|value| value.0 as usize + 1)
        .unwrap_or(0);
    if alias_len == 0 {
        return;
    }

    let last_uses = compute_last_uses(block, alias_len);
    let mut aliases = vec![None; alias_len];
    let mut slot_values = vec![None; frame.total_slots() as usize];
    let original_ops = core::mem::take(&mut block.ops);
    let mut rewritten = Vec::with_capacity(original_ops.len());

    for mut inst in original_ops {
        rewrite_inst_uses(&mut inst.kind, &aliases);

        match &inst.kind {
            SsaInstKind::LoadSlot { slot, dst } => {
                if let Some(src) = slot_values.get(slot.0 as usize).copied().flatten() {
                    let resolved_src = resolve_alias(src, &aliases);
                    if can_forward_load(resolved_src, *dst, &last_uses)
                        && can_forward_load_type(resolved_src, *dst, value_types)
                    {
                        aliases[dst.0 as usize] = Some(resolved_src);
                        continue;
                    }
                }
            }
            SsaInstKind::StoreSlot { slot, src } => {
                slot_values[slot.0 as usize] = Some(*src);
            }
            SsaInstKind::Boundary(_) => {
                slot_values.fill(None);
            }
            SsaInstKind::Value { .. } => {}
        }

        rewritten.push(inst);
    }

    rewrite_terminator_uses(&mut block.terminator, &aliases);
    block.ops = rewritten;
}

fn max_value_index(block: &SsaBlock) -> Option<SsaValue> {
    let mut max_value = block.params.iter().copied().max();

    for inst in &block.ops {
        match &inst.kind {
            SsaInstKind::Value { args, results, .. } => {
                max_value = max_value.max(
                    args.iter()
                        .filter_map(|op| match op {
                            SsaOperand::Value(v) => Some(*v),
                            SsaOperand::Const(_) => None,
                        })
                        .max(),
                );
                max_value = max_value.max(results.iter().copied().max());
            }
            SsaInstKind::LoadSlot { dst, .. } => {
                max_value = max_value.max(Some(*dst));
            }
            SsaInstKind::StoreSlot { src, .. } => {
                max_value = max_value.max(Some(*src));
            }
            SsaInstKind::Boundary(_) => {}
        }
    }

    match &block.terminator {
        SsaTerminator::Goto(edge) => {
            max_value = max_value.max(edge.bindings.iter().map(|binding| binding.value).max());
        }
        SsaTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => {
            max_value = max_value.max(Some(*cond));
            max_value = max_value.max(then_edge.bindings.iter().map(|binding| binding.value).max());
            max_value = max_value.max(else_edge.bindings.iter().map(|binding| binding.value).max());
        }
        SsaTerminator::BrTable { index, entries } => {
            max_value = max_value.max(Some(*index));
            for edge in entries {
                max_value = max_value.max(edge.bindings.iter().map(|binding| binding.value).max());
            }
        }
        SsaTerminator::Return { .. } | SsaTerminator::TrapUnreachable => {}
    }

    max_value
}

fn compute_last_uses(block: &SsaBlock, len: usize) -> Vec<Option<u32>> {
    let mut last_uses = vec![None; len];

    for (index, inst) in block.ops.iter().enumerate() {
        let pos = index as u32;
        match &inst.kind {
            SsaInstKind::Value { args, .. } => {
                for operand in args {
                    if let SsaOperand::Value(value) = operand {
                        last_uses[value.0 as usize] = Some(pos);
                    }
                }
            }
            SsaInstKind::StoreSlot { src, .. } => {
                last_uses[src.0 as usize] = Some(pos);
            }
            SsaInstKind::LoadSlot { .. } | SsaInstKind::Boundary(_) => {}
        }
    }

    let term_pos = block.ops.len() as u32;
    match &block.terminator {
        SsaTerminator::Goto(edge) => mark_edge_uses(edge, term_pos, &mut last_uses),
        SsaTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => {
            last_uses[cond.0 as usize] = Some(term_pos);
            mark_edge_uses(then_edge, term_pos, &mut last_uses);
            mark_edge_uses(else_edge, term_pos, &mut last_uses);
        }
        SsaTerminator::BrTable { index, entries } => {
            last_uses[index.0 as usize] = Some(term_pos);
            for edge in entries {
                mark_edge_uses(edge, term_pos, &mut last_uses);
            }
        }
        SsaTerminator::Return { .. } | SsaTerminator::TrapUnreachable => {}
    }

    last_uses
}

fn mark_edge_uses(edge: &SsaEdge, pos: u32, last_uses: &mut [Option<u32>]) {
    for binding in &edge.bindings {
        last_uses[binding.value.0 as usize] = Some(pos);
    }
}

fn can_forward_load(src: SsaValue, dst: SsaValue, last_uses: &[Option<u32>]) -> bool {
    let Some(dst_last_use) = last_uses.get(dst.0 as usize).copied().flatten() else {
        return false;
    };
    let Some(src_last_use) = last_uses.get(src.0 as usize).copied().flatten() else {
        return false;
    };
    dst_last_use <= src_last_use
}

fn can_forward_load_type(
    src: SsaValue,
    dst: SsaValue,
    value_types: &[crate::value_type::ValueType],
) -> bool {
    match (
        value_types.get(src.0 as usize).copied(),
        value_types.get(dst.0 as usize).copied(),
    ) {
        (Some(src_ty), Some(dst_ty)) => src_ty == dst_ty,
        _ => true,
    }
}

fn rewrite_inst_uses(kind: &mut SsaInstKind, aliases: &[Option<SsaValue>]) {
    match kind {
        SsaInstKind::Value { args, .. } => {
            for operand in args {
                if let SsaOperand::Value(value) = operand {
                    *value = resolve_alias(*value, aliases);
                }
            }
        }
        SsaInstKind::StoreSlot { src, .. } => {
            *src = resolve_alias(*src, aliases);
        }
        SsaInstKind::LoadSlot { .. } | SsaInstKind::Boundary(_) => {}
    }
}

fn rewrite_terminator_uses(term: &mut SsaTerminator, aliases: &[Option<SsaValue>]) {
    match term {
        SsaTerminator::Goto(edge) => rewrite_edge_uses(edge, aliases),
        SsaTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => {
            *cond = resolve_alias(*cond, aliases);
            rewrite_edge_uses(then_edge, aliases);
            rewrite_edge_uses(else_edge, aliases);
        }
        SsaTerminator::BrTable { index, entries } => {
            *index = resolve_alias(*index, aliases);
            for edge in entries {
                rewrite_edge_uses(edge, aliases);
            }
        }
        SsaTerminator::Return { .. } | SsaTerminator::TrapUnreachable => {}
    }
}

fn rewrite_edge_uses(edge: &mut SsaEdge, aliases: &[Option<SsaValue>]) {
    for binding in &mut edge.bindings {
        binding.value = resolve_alias(binding.value, aliases);
    }
}

fn resolve_alias(value: SsaValue, aliases: &[Option<SsaValue>]) -> SsaValue {
    let mut resolved = value;
    while let Some(Some(next)) = aliases.get(resolved.0 as usize) {
        if *next == resolved {
            break;
        }
        resolved = *next;
    }
    resolved
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use crate::value_type::ValueType;
    use crate::vm::{
        middle::{
            frame::{plan_frame_layout, FrameSlot},
            ssa_ir::{
                ir::{SsaBlock, SsaInstKind, SsaProgram, SsaTerminator, SsaValue},
                leaf::SsaLeafOp,
                target::SsaTarget,
            },
        },
        wasm::primitive_op::PrimitiveOpKind,
    };
    use super::optimize_ssa;

    #[test]
    fn forwards_store_reload_when_source_already_lives_long_enough() {
        let mut program = SsaProgram {
            entry: SsaTarget(0),
            local_cache: Default::default(),
            blocks: alloc::vec![SsaBlock {
                id: SsaTarget(0),
                params: Vec::new(),
                ops: alloc::vec![
                    crate::vm::middle::ssa_ir::ir::SsaInst {
                        kind: SsaInstKind::Value {
                            op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 7 })
                                .unwrap(),
                            args: Vec::new(),
                            results: alloc::vec![SsaValue(0)],
                        },
                    },
                    crate::vm::middle::ssa_ir::ir::SsaInst {
                        kind: SsaInstKind::StoreSlot {
                            slot: FrameSlot(0),
                            src: SsaValue(0),
                        },
                    },
                    crate::vm::middle::ssa_ir::ir::SsaInst {
                        kind: SsaInstKind::LoadSlot {
                            slot: FrameSlot(0),
                            dst: SsaValue(1),
                        },
                    },
                    crate::vm::middle::ssa_ir::ir::SsaInst {
                        kind: SsaInstKind::Value {
                            op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Add).unwrap(),
                            args: alloc::vec![SsaValue(0), SsaValue(1)],
                            results: alloc::vec![SsaValue(2)],
                        },
                    },
                ],
                terminator: SsaTerminator::Return { results: None },
            }],
            value_types: alloc::vec![],
        };

        optimize_ssa(&mut program, plan_frame_layout(1, 2, 0));

        let block = &program.blocks[0];
        assert_eq!(block.ops.len(), 3);
        assert!(matches!(
            &block.ops[2].kind,
            SsaInstKind::Value {
                args,
                ..
            } if args == &alloc::vec![SsaValue(0), SsaValue(0)]
        ));
    }

    #[test]
    fn does_not_forward_duplicate_slot_loads_without_same_block_store() {
        let mut program = SsaProgram {
            entry: SsaTarget(0),
            local_cache: Default::default(),
            blocks: alloc::vec![SsaBlock {
                id: SsaTarget(0),
                params: Vec::new(),
                ops: alloc::vec![
                    crate::vm::middle::ssa_ir::ir::SsaInst {
                        kind: SsaInstKind::LoadSlot {
                            slot: FrameSlot(0),
                            dst: SsaValue(0),
                        },
                    },
                    crate::vm::middle::ssa_ir::ir::SsaInst {
                        kind: SsaInstKind::LoadSlot {
                            slot: FrameSlot(0),
                            dst: SsaValue(1),
                        },
                    },
                    crate::vm::middle::ssa_ir::ir::SsaInst {
                        kind: SsaInstKind::Value {
                            op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Add).unwrap(),
                            args: alloc::vec![SsaValue(0), SsaValue(1)],
                            results: alloc::vec![SsaValue(2)],
                        },
                    },
                ],
                terminator: SsaTerminator::Return { results: None },
            }],
            value_types: alloc::vec![],
        };

        optimize_ssa(&mut program, plan_frame_layout(1, 2, 0));

        let block = &program.blocks[0];
        assert!(matches!(block.ops[1].kind, SsaInstKind::LoadSlot { .. }));
        assert!(matches!(
            &block.ops[2].kind,
            SsaInstKind::Value {
                args,
                ..
            } if args == &alloc::vec![SsaValue(0), SsaValue(1)]
        ));
    }

    #[test]
    fn keeps_reload_when_forwarding_would_extend_the_source_lifetime() {
        let mut program = SsaProgram {
            entry: SsaTarget(0),
            local_cache: Default::default(),
            blocks: alloc::vec![SsaBlock {
                id: SsaTarget(0),
                params: Vec::new(),
                ops: alloc::vec![
                    crate::vm::middle::ssa_ir::ir::SsaInst {
                        kind: SsaInstKind::LoadSlot {
                            slot: FrameSlot(0),
                            dst: SsaValue(0),
                        },
                    },
                    crate::vm::middle::ssa_ir::ir::SsaInst {
                        kind: SsaInstKind::Value {
                            op: SsaLeafOp::from_primitive(PrimitiveOpKind::Drop).unwrap(),
                            args: alloc::vec![SsaValue(0)],
                            results: Vec::new(),
                        },
                    },
                    crate::vm::middle::ssa_ir::ir::SsaInst {
                        kind: SsaInstKind::LoadSlot {
                            slot: FrameSlot(0),
                            dst: SsaValue(1),
                        },
                    },
                    crate::vm::middle::ssa_ir::ir::SsaInst {
                        kind: SsaInstKind::Value {
                            op: SsaLeafOp::from_primitive(PrimitiveOpKind::Drop).unwrap(),
                            args: alloc::vec![SsaValue(1)],
                            results: Vec::new(),
                        },
                    },
                ],
                terminator: SsaTerminator::Return { results: None },
            }],
            value_types: alloc::vec![],
        };

        optimize_ssa(&mut program, plan_frame_layout(1, 2, 0));

        let block = &program.blocks[0];
        assert!(matches!(block.ops[2].kind, SsaInstKind::LoadSlot { .. }));
        assert!(matches!(
            &block.ops[3].kind,
            SsaInstKind::Value {
                args,
                ..
            } if args == &alloc::vec![SsaValue(1)]
        ));
    }

    #[test]
    fn does_not_forward_slot_reload_across_mismatched_value_types() {
        let mut program = SsaProgram {
            entry: SsaTarget(0),
            local_cache: Default::default(),
            blocks: alloc::vec![SsaBlock {
                id: SsaTarget(0),
                params: Vec::new(),
                ops: alloc::vec![
                    crate::vm::middle::ssa_ir::ir::SsaInst {
                        kind: SsaInstKind::Value {
                            op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 1 })
                                .unwrap(),
                            args: Vec::new(),
                            results: alloc::vec![SsaValue(0)],
                        },
                    },
                    crate::vm::middle::ssa_ir::ir::SsaInst {
                        kind: SsaInstKind::StoreSlot {
                            slot: FrameSlot(0),
                            src: SsaValue(0),
                        },
                    },
                    crate::vm::middle::ssa_ir::ir::SsaInst {
                        kind: SsaInstKind::LoadSlot {
                            slot: FrameSlot(0),
                            dst: SsaValue(1),
                        },
                    },
                    crate::vm::middle::ssa_ir::ir::SsaInst {
                        kind: SsaInstKind::Value {
                            op: SsaLeafOp::from_primitive(PrimitiveOpKind::Drop).unwrap(),
                            args: alloc::vec![SsaValue(1)],
                            results: Vec::new(),
                        },
                    },
                ],
                terminator: SsaTerminator::Return { results: None },
            }],
            value_types: alloc::vec![ValueType::I32, ValueType::I64],
        };

        optimize_ssa(&mut program, plan_frame_layout(1, 2, 0));

        let block = &program.blocks[0];
        assert!(matches!(block.ops[2].kind, SsaInstKind::LoadSlot { .. }));
        assert!(matches!(
            &block.ops[3].kind,
            SsaInstKind::Value {
                args,
                ..
            } if args == &alloc::vec![SsaValue(1)]
        ));
    }
}
