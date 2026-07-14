//! Middle SSA optimization passes.
//!
//! These passes are still middle-layer transforms: they rewrite prepared
//! SSA-IR without relying on backend-specific peepholes. The current pass
//! restores the old inline-constant absorption path:
//!
//! - single-use constant producers can become `SsaOperand::Const(bits)` in a
//!   later leaf op
//! - fully constant pure leaf ops can fold to one constant producer
//! - dead constant producers are then removed from the block
//!
//! This keeps immediates explicit in SSA so the backend can use native
//! immediates directly instead of materializing them as transient registers.

use crate::collections;

use crate::{
    value_type::ValueType,
    vm::{
        middle::ssa_ir::ir::{
            SsaCallArgs, SsaCallOp, SsaCallOperandLoc, SsaInst, SsaOp, SsaOperand, SsaProgram,
            SsaScalarResultLoc, SsaTerminator, SsaValue,
        },
        wasm::primitive_op::{self, PrimitiveOpKind},
    },
};

pub(crate) fn optimize_program(program: &mut SsaProgram) {
    for block_idx in 0..program.blocks.len() {
        fold_constants_into_operands(program, block_idx);
    }
}

/// Absorb single-use const producers into later leaf operands and fold fully
/// constant pure leaf ops.
///
/// This is intentionally block-local. Once cleanup has merged trivial CFG
/// structure, the profitable constant chains we care about are visible within
/// one prepared SSA block.
fn fold_constants_into_operands(program: &mut SsaProgram, block_idx: usize) {
    // Detach the mutable block pieces so we can freely borrow the rest of the
    // program (const_pool, primitive_pool) while rewriting them.
    let mut ops = core::mem::take(&mut program.blocks[block_idx].ops);
    let extra_args = core::mem::take(&mut program.blocks[block_idx].extra_args);
    let params = core::mem::take(&mut program.blocks[block_idx].params);
    let terminator = core::mem::replace(
        &mut program.blocks[block_idx].terminator,
        SsaTerminator::TrapUnreachable,
    );

    let max_val = max_value_index_parts(&params, &ops, &extra_args, &terminator, program)
        .map(|value| value.0 as usize + 1)
        .unwrap_or(0);
    if max_val == 0 {
        program.blocks[block_idx].ops = ops;
        program.blocks[block_idx].extra_args = extra_args;
        program.blocks[block_idx].params = params;
        program.blocks[block_idx].terminator = terminator;
        return;
    }

    let mut known_const: collections::Vec<Option<u64>> = collections::vec![None; max_val];
    let mut used_in_terminator = collections::vec![false; max_val];

    // Pass 1: collect Const producers.
    for inst in &ops {
        if !inst.op.is_primitive() {
            continue;
        }
        if !inst.args[0].is_none() || !inst.args[1].is_none() {
            continue;
        }
        if inst.result.is_none() {
            continue;
        }
        let pool_idx = inst.op.as_primitive_idx().expect("primitive op") as usize;
        let kind = &program.primitive_pool[pool_idx];
        if let Some(bits) = const_bits_of_primitive(kind) {
            let value = inst.result;
            if let Some(slot) = known_const.get_mut(value.0 as usize) {
                *slot = Some(bits);
            }
        }
    }
    mark_terminator_uses(&terminator, &mut used_in_terminator);
    for inst in &ops {
        if inst.op == SsaOp::CALL {
            if let Some(call) = program.call_ops.get(inst.meta as usize) {
                mark_call_op_uses(call, &mut used_in_terminator);
            }
        }
    }

    // Pass 2: try to fold fully-constant ops and absorb const operands.
    for inst in ops.iter_mut() {
        let Some(pool_idx) = inst.op.as_primitive_idx() else {
            continue;
        };

        let current_kind = program.primitive_pool[pool_idx as usize].clone();
        let args_len = inline_arg_count(inst);

        if args_len > 0 && can_accept_const_operand(&current_kind) {
            // Gather const bit values for each arg (up to 2 inline; this path
            // never involves 3-arg primitives because `can_accept_const_operand`
            // does not include them).
            let mut const_args = collections::Vec::with_capacity(args_len);
            let mut all_const = true;
            for operand in inst.args.iter().take(args_len) {
                match operand.decode() {
                    super::ssa_ir::ir::DecodedOperand::Value(value) => {
                        match known_const.get(value.0 as usize).copied().flatten() {
                            Some(bits) => const_args.push(bits),
                            None => {
                                all_const = false;
                                break;
                            }
                        }
                    }
                    super::ssa_ir::ir::DecodedOperand::Const(idx) => {
                        const_args.push(program.const_pool[idx as usize]);
                    }
                    super::ssa_ir::ir::DecodedOperand::None => {
                        all_const = false;
                        break;
                    }
                }
            }
            if all_const && const_args.len() == args_len {
                if let Some((result_bits, const_primitive)) = try_eval(&current_kind, &const_args) {
                    // Folding is best-effort: if the primitive pool is full
                    // (a u16-encoded SsaOp cannot address more entries), we
                    // leave the original op in place rather than miscompile.
                    if let Ok(new_pool_idx) = program.intern_primitive(const_primitive) {
                        let result = inst.result;
                        if result.is_some() {
                            if let Some(slot) = known_const.get_mut(result.0 as usize) {
                                *slot = Some(result_bits);
                            }
                        }
                        inst.op = SsaOp::primitive(new_pool_idx);
                        inst.args = [SsaOperand::NONE, SsaOperand::NONE];
                        inst.meta = 0;
                        continue;
                    }
                }
            }
        }

        if can_accept_const_operand(&current_kind) {
            for operand in inst.args.iter_mut() {
                let Some(value) = operand.as_value() else {
                    continue;
                };
                let index = value.0 as usize;
                if let Some(bits) = known_const.get(index).copied().flatten() {
                    if !used_in_terminator.get(index).copied().unwrap_or(true) {
                        *operand = program.intern_const(bits);
                    }
                }
            }
        }
    }

    // Pass 3: compute which SSA values are still used, including terminator,
    // then drop dead Const producers.
    let mut still_used = collections::vec![false; max_val];
    for inst in &ops {
        if inst.op.is_primitive() {
            for operand in inst.args.iter() {
                if let Some(value) = operand.as_value() {
                    still_used[value.0 as usize] = true;
                }
            }
            let pool_idx = inst.op.as_primitive_idx().expect("primitive op") as usize;
            let kind = &program.primitive_pool[pool_idx];
            let extra_count = primitive_op::stack_effect(kind).0.saturating_sub(2);
            if extra_count != 0 {
                let start = inst.meta as usize;
                let end = start
                    .checked_add(extra_count)
                    .expect("primitive extra_args index overflow");
                if let Some(operands) = extra_args.get(start..end) {
                    for operand in operands {
                        if let Some(value) = operand.as_value() {
                            still_used[value.0 as usize] = true;
                        }
                    }
                }
            }
        } else {
            match inst.op {
                SsaOp::SPILL | SsaOp::CELL_SET_SLOT | SsaOp::CELL_SET_CACHE => {
                    if let Some(value) = inst.args[0].as_value() {
                        still_used[value.0 as usize] = true;
                    }
                }
                SsaOp::CALL => {
                    if let Some(call) = program.call_ops.get(inst.meta as usize) {
                        mark_call_op_uses(call, &mut still_used);
                    }
                }
                _ => {}
            }
        }
    }
    for (index, used) in used_in_terminator.iter().copied().enumerate() {
        if used {
            still_used[index] = true;
        }
    }

    ops.retain(|inst| {
        let Some(pool_idx) = inst.op.as_primitive_idx() else {
            return true;
        };
        // Candidate for removal only if this is a 0-arg 1-result op.
        if !inst.args[0].is_none() || !inst.args[1].is_none() {
            return true;
        }
        if inst.result.is_none() {
            return true;
        }
        let kind = &program.primitive_pool[pool_idx as usize];
        if !matches!(
            kind,
            PrimitiveOpKind::I32Const { .. }
                | PrimitiveOpKind::I64Const { .. }
                | PrimitiveOpKind::F32Const { .. }
                | PrimitiveOpKind::F64Const { .. }
        ) {
            return true;
        }
        let index = inst.result.0 as usize;
        // Keep the producer if the value is still consumed anywhere.
        still_used.get(index).copied().unwrap_or(true)
    });

    // Reattach the block pieces.
    program.blocks[block_idx].ops = ops;
    program.blocks[block_idx].extra_args = extra_args;
    program.blocks[block_idx].params = params;
    program.blocks[block_idx].terminator = terminator;
}

/// Count the number of present inline operands for a primitive op (0, 1, or 2).
///
/// This function intentionally only inspects the two inline slots: every
/// candidate for constant-folding in this pass (see `can_accept_const_operand`)
/// is a 0/1/2-arg op, so the 3rd operand (in `block.extra_args`) is
/// irrelevant here.
#[inline]
fn inline_arg_count(inst: &SsaInst) -> usize {
    let mut n = 0;
    for arg in &inst.args {
        if !arg.is_none() {
            n += 1;
        } else {
            break;
        }
    }
    n
}

#[inline]
fn const_bits_of_primitive(kind: &PrimitiveOpKind) -> Option<u64> {
    match kind {
        PrimitiveOpKind::I32Const { value } => Some(*value as u64),
        PrimitiveOpKind::I64Const { value } => Some(*value),
        PrimitiveOpKind::F32Const { value } => Some(*value as u64),
        PrimitiveOpKind::F64Const { value } => Some(*value),
        _ => None,
    }
}

/// Only fold/absorb constants into leaf ops whose machine lowering already
/// accepts `SsaOperand::Const`.
fn can_accept_const_operand(kind: &PrimitiveOpKind) -> bool {
    use PrimitiveOpKind as P;
    matches!(
        kind,
        P::I32Add
            | P::I32Sub
            | P::I32Mul
            | P::I32DivS
            | P::I32DivU
            | P::I32RemS
            | P::I32RemU
            | P::I32And
            | P::I32Or
            | P::I32Xor
            | P::I32Shl
            | P::I32ShrS
            | P::I32ShrU
            | P::I32Rotl
            | P::I32Rotr
            | P::I64Add
            | P::I64Sub
            | P::I64Mul
            | P::I64DivS
            | P::I64DivU
            | P::I64RemS
            | P::I64RemU
            | P::I64And
            | P::I64Or
            | P::I64Xor
            | P::I64Shl
            | P::I64ShrS
            | P::I64ShrU
            | P::I64Rotl
            | P::I64Rotr
            | P::F32Add
            | P::F32Sub
            | P::F32Mul
            | P::F32Div
            | P::F32Min
            | P::F32Max
            | P::F32Copysign
            | P::F64Add
            | P::F64Sub
            | P::F64Mul
            | P::F64Div
            | P::F64Min
            | P::F64Max
            | P::F64Copysign
            | P::I32Eq
            | P::I32Ne
            | P::I32LtS
            | P::I32LtU
            | P::I32GtS
            | P::I32GtU
            | P::I32LeS
            | P::I32LeU
            | P::I32GeS
            | P::I32GeU
            | P::I64Eq
            | P::I64Ne
            | P::I64LtS
            | P::I64LtU
            | P::I64GtS
            | P::I64GtU
            | P::I64LeS
            | P::I64LeU
            | P::I64GeS
            | P::I64GeU
            | P::F32Eq
            | P::F32Ne
            | P::F32Lt
            | P::F32Gt
            | P::F32Le
            | P::F32Ge
            | P::F64Eq
            | P::F64Ne
            | P::F64Lt
            | P::F64Gt
            | P::F64Le
            | P::F64Ge
            | P::I32Eqz
            | P::I32Clz
            | P::I32Ctz
            | P::I32Popcnt
            | P::I64Eqz
            | P::I64Clz
            | P::I64Ctz
            | P::I64Popcnt
            | P::I32Extend8S
            | P::I32Extend16S
            | P::I64Extend8S
            | P::I64Extend16S
            | P::I64Extend32S
            | P::F32Abs
            | P::F32Neg
            | P::F32Ceil
            | P::F32Floor
            | P::F32Trunc
            | P::F32Nearest
            | P::F32Sqrt
            | P::F64Abs
            | P::F64Neg
            | P::F64Ceil
            | P::F64Floor
            | P::F64Trunc
            | P::F64Nearest
            | P::F64Sqrt
            | P::I32WrapI64
            | P::I64ExtendI32S
            | P::I64ExtendI32U
            | P::I32TruncF32S
            | P::I32TruncF32U
            | P::I32TruncF64S
            | P::I32TruncF64U
            | P::I64TruncF32S
            | P::I64TruncF32U
            | P::I64TruncF64S
            | P::I64TruncF64U
            | P::I32TruncSatF32S
            | P::I32TruncSatF32U
            | P::I32TruncSatF64S
            | P::I32TruncSatF64U
            | P::I64TruncSatF32S
            | P::I64TruncSatF32U
            | P::I64TruncSatF64S
            | P::I64TruncSatF64U
            | P::F32ConvertI32S
            | P::F32ConvertI32U
            | P::F32ConvertI64S
            | P::F32ConvertI64U
            | P::F64ConvertI32S
            | P::F64ConvertI32U
            | P::F64ConvertI64S
            | P::F64ConvertI64U
            | P::F32DemoteF64
            | P::F64PromoteF32
            | P::I32ReinterpretF32
            | P::I64ReinterpretF64
            | P::F32ReinterpretI32
            | P::F64ReinterpretI64
    )
}

/// Evaluate a pure primitive op at compile time when all operands are known
/// constants. Return `None` for trapping or backend-unsupported cases.
fn try_eval(kind: &PrimitiveOpKind, args: &[u64]) -> Option<(u64, PrimitiveOpKind)> {
    use PrimitiveOpKind as P;

    let result_ty = primitive_op::result_type(kind)?;
    let bits = match kind {
        P::I32Add => ((args[0] as u32).wrapping_add(args[1] as u32)) as u64,
        P::I32Sub => ((args[0] as u32).wrapping_sub(args[1] as u32)) as u64,
        P::I32Mul => ((args[0] as u32).wrapping_mul(args[1] as u32)) as u64,
        P::I32DivS => {
            let (a, b) = (args[0] as u32 as i32, args[1] as u32 as i32);
            if b == 0 || (a == i32::MIN && b == -1) {
                return None;
            }
            (a.wrapping_div(b) as u32) as u64
        }
        P::I32DivU => {
            let (a, b) = (args[0] as u32, args[1] as u32);
            if b == 0 {
                return None;
            }
            (a / b) as u64
        }
        P::I32RemS => {
            let (a, b) = (args[0] as u32 as i32, args[1] as u32 as i32);
            if b == 0 {
                return None;
            }
            (a.wrapping_rem(b) as u32) as u64
        }
        P::I32RemU => {
            let (a, b) = (args[0] as u32, args[1] as u32);
            if b == 0 {
                return None;
            }
            (a % b) as u64
        }
        P::I32And => (args[0] as u32 & args[1] as u32) as u64,
        P::I32Or => (args[0] as u32 | args[1] as u32) as u64,
        P::I32Xor => (args[0] as u32 ^ args[1] as u32) as u64,
        P::I32Shl => ((args[0] as u32).wrapping_shl(args[1] as u32 & 31)) as u64,
        P::I32ShrS => ((args[0] as u32 as i32).wrapping_shr(args[1] as u32 & 31) as u32) as u64,
        P::I32ShrU => ((args[0] as u32).wrapping_shr(args[1] as u32 & 31)) as u64,
        P::I32Rotl => ((args[0] as u32).rotate_left(args[1] as u32 & 31)) as u64,
        P::I32Rotr => ((args[0] as u32).rotate_right(args[1] as u32 & 31)) as u64,
        P::I64Add => args[0].wrapping_add(args[1]),
        P::I64Sub => args[0].wrapping_sub(args[1]),
        P::I64Mul => args[0].wrapping_mul(args[1]),
        P::I64DivS => {
            let (a, b) = (args[0] as i64, args[1] as i64);
            if b == 0 || (a == i64::MIN && b == -1) {
                return None;
            }
            a.wrapping_div(b) as u64
        }
        P::I64DivU => {
            if args[1] == 0 {
                return None;
            }
            args[0] / args[1]
        }
        P::I64RemS => {
            let (a, b) = (args[0] as i64, args[1] as i64);
            if b == 0 {
                return None;
            }
            a.wrapping_rem(b) as u64
        }
        P::I64RemU => {
            if args[1] == 0 {
                return None;
            }
            args[0] % args[1]
        }
        P::I64And => args[0] & args[1],
        P::I64Or => args[0] | args[1],
        P::I64Xor => args[0] ^ args[1],
        P::I64Shl => args[0].wrapping_shl((args[1] & 63) as u32),
        P::I64ShrS => (args[0] as i64).wrapping_shr((args[1] & 63) as u32) as u64,
        P::I64ShrU => args[0].wrapping_shr((args[1] & 63) as u32),
        P::I64Rotl => args[0].rotate_left((args[1] & 63) as u32),
        P::I64Rotr => args[0].rotate_right((args[1] & 63) as u32),
        P::F32Add => {
            canon_f32(f32::from_bits(args[0] as u32) + f32::from_bits(args[1] as u32)) as u64
        }
        P::F32Sub => {
            canon_f32(f32::from_bits(args[0] as u32) - f32::from_bits(args[1] as u32)) as u64
        }
        P::F32Mul => {
            canon_f32(f32::from_bits(args[0] as u32) * f32::from_bits(args[1] as u32)) as u64
        }
        P::F32Div => {
            canon_f32(f32::from_bits(args[0] as u32) / f32::from_bits(args[1] as u32)) as u64
        }
        P::F32Min => wasm_f32_min(args[0] as u32, args[1] as u32) as u64,
        P::F32Max => wasm_f32_max(args[0] as u32, args[1] as u32) as u64,
        P::F32Copysign => f32::from_bits(args[0] as u32)
            .copysign(f32::from_bits(args[1] as u32))
            .to_bits() as u64,
        P::F64Add => canon_f64(f64::from_bits(args[0]) + f64::from_bits(args[1])),
        P::F64Sub => canon_f64(f64::from_bits(args[0]) - f64::from_bits(args[1])),
        P::F64Mul => canon_f64(f64::from_bits(args[0]) * f64::from_bits(args[1])),
        P::F64Div => canon_f64(f64::from_bits(args[0]) / f64::from_bits(args[1])),
        P::F64Min => wasm_f64_min(args[0], args[1]),
        P::F64Max => wasm_f64_max(args[0], args[1]),
        P::F64Copysign => f64::from_bits(args[0])
            .copysign(f64::from_bits(args[1]))
            .to_bits(),
        P::I32Eq => bool32(args[0] as u32 == args[1] as u32),
        P::I32Ne => bool32(args[0] as u32 != args[1] as u32),
        P::I32LtS => bool32((args[0] as u32 as i32) < (args[1] as u32 as i32)),
        P::I32LtU => bool32((args[0] as u32) < (args[1] as u32)),
        P::I32GtS => bool32((args[0] as u32 as i32) > (args[1] as u32 as i32)),
        P::I32GtU => bool32((args[0] as u32) > (args[1] as u32)),
        P::I32LeS => bool32((args[0] as u32 as i32) <= (args[1] as u32 as i32)),
        P::I32LeU => bool32((args[0] as u32) <= (args[1] as u32)),
        P::I32GeS => bool32((args[0] as u32 as i32) >= (args[1] as u32 as i32)),
        P::I32GeU => bool32((args[0] as u32) >= (args[1] as u32)),
        P::I64Eq => bool32(args[0] == args[1]),
        P::I64Ne => bool32(args[0] != args[1]),
        P::I64LtS => bool32((args[0] as i64) < (args[1] as i64)),
        P::I64LtU => bool32(args[0] < args[1]),
        P::I64GtS => bool32((args[0] as i64) > (args[1] as i64)),
        P::I64GtU => bool32(args[0] > args[1]),
        P::I64LeS => bool32((args[0] as i64) <= (args[1] as i64)),
        P::I64LeU => bool32(args[0] <= args[1]),
        P::I64GeS => bool32((args[0] as i64) >= (args[1] as i64)),
        P::I64GeU => bool32(args[0] >= args[1]),
        P::F32Eq => bool32(f32::from_bits(args[0] as u32) == f32::from_bits(args[1] as u32)),
        P::F32Ne => bool32(f32::from_bits(args[0] as u32) != f32::from_bits(args[1] as u32)),
        P::F32Lt => bool32(f32::from_bits(args[0] as u32) < f32::from_bits(args[1] as u32)),
        P::F32Gt => bool32(f32::from_bits(args[0] as u32) > f32::from_bits(args[1] as u32)),
        P::F32Le => bool32(f32::from_bits(args[0] as u32) <= f32::from_bits(args[1] as u32)),
        P::F32Ge => bool32(f32::from_bits(args[0] as u32) >= f32::from_bits(args[1] as u32)),
        P::F64Eq => bool32(f64::from_bits(args[0]) == f64::from_bits(args[1])),
        P::F64Ne => bool32(f64::from_bits(args[0]) != f64::from_bits(args[1])),
        P::F64Lt => bool32(f64::from_bits(args[0]) < f64::from_bits(args[1])),
        P::F64Gt => bool32(f64::from_bits(args[0]) > f64::from_bits(args[1])),
        P::F64Le => bool32(f64::from_bits(args[0]) <= f64::from_bits(args[1])),
        P::F64Ge => bool32(f64::from_bits(args[0]) >= f64::from_bits(args[1])),
        P::I32Eqz => bool32(args[0] as u32 == 0),
        P::I64Eqz => bool32(args[0] == 0),
        P::I32Clz => (args[0] as u32).leading_zeros() as u64,
        P::I32Ctz => (args[0] as u32).trailing_zeros() as u64,
        P::I32Popcnt => (args[0] as u32).count_ones() as u64,
        P::I64Clz => args[0].leading_zeros() as u64,
        P::I64Ctz => args[0].trailing_zeros() as u64,
        P::I64Popcnt => args[0].count_ones() as u64,
        P::F32Abs => f32::from_bits(args[0] as u32).abs().to_bits() as u64,
        P::F32Neg => (-f32::from_bits(args[0] as u32)).to_bits() as u64,
        P::F32Ceil => soft_f32_ceil(args[0] as u32) as u64,
        P::F32Floor => soft_f32_floor(args[0] as u32) as u64,
        P::F32Trunc => soft_f32_trunc(args[0] as u32) as u64,
        P::F32Nearest => soft_f32_nearest(args[0] as u32) as u64,
        P::F32Sqrt => return None,
        P::F64Abs => f64::from_bits(args[0]).abs().to_bits(),
        P::F64Neg => (-f64::from_bits(args[0])).to_bits(),
        P::F64Ceil => soft_f64_ceil(args[0]),
        P::F64Floor => soft_f64_floor(args[0]),
        P::F64Trunc => soft_f64_trunc(args[0]),
        P::F64Nearest => soft_f64_nearest(args[0]),
        P::F64Sqrt => return None,
        P::I32Extend8S => ((args[0] as u32 as i8) as i32 as u32) as u64,
        P::I32Extend16S => ((args[0] as u32 as i16) as i32 as u32) as u64,
        P::I64Extend8S => (args[0] as i8 as i64) as u64,
        P::I64Extend16S => (args[0] as i16 as i64) as u64,
        P::I64Extend32S => (args[0] as i32 as i64) as u64,
        P::I32WrapI64 => (args[0] as u32) as u64,
        P::I64ExtendI32S => ((args[0] as u32 as i32) as i64) as u64,
        P::I64ExtendI32U => (args[0] as u32) as u64,
        P::I32ReinterpretF32 => args[0] & 0xFFFF_FFFF,
        P::I64ReinterpretF64 => args[0],
        P::F32ReinterpretI32 => args[0] & 0xFFFF_FFFF,
        P::F64ReinterpretI64 => args[0],
        P::F32ConvertI32S => (((args[0] as u32 as i32) as f32).to_bits()) as u64,
        P::F32ConvertI32U => (((args[0] as u32) as f32).to_bits()) as u64,
        P::F32ConvertI64S => (((args[0] as i64) as f32).to_bits()) as u64,
        P::F32ConvertI64U => ((args[0] as f32).to_bits()) as u64,
        P::F64ConvertI32S => ((args[0] as u32 as i32) as f64).to_bits(),
        P::F64ConvertI32U => ((args[0] as u32) as f64).to_bits(),
        P::F64ConvertI64S => ((args[0] as i64) as f64).to_bits(),
        P::F64ConvertI64U => (args[0] as f64).to_bits(),
        P::F32DemoteF64 => canon_f32(f64::from_bits(args[0]) as f32) as u64,
        P::F64PromoteF32 => canon_f64(f32::from_bits(args[0] as u32) as f64),
        P::I32TruncF32S => {
            let truncated = soft_f32_trunc(args[0] as u32);
            let f = f32::from_bits(truncated);
            if f.is_nan() || f < i32::MIN as f32 || f > i32::MAX as f32 {
                return None;
            }
            (f as i32 as u32) as u64
        }
        P::I32TruncF32U => {
            let truncated = soft_f32_trunc(args[0] as u32);
            let f = f32::from_bits(truncated);
            if f.is_nan() || f < 0.0 || f > u32::MAX as f32 {
                return None;
            }
            f as u32 as u64
        }
        P::I32TruncF64S => {
            let truncated = soft_f64_trunc(args[0]);
            let f = f64::from_bits(truncated);
            if f.is_nan() || f < i32::MIN as f64 || f > i32::MAX as f64 {
                return None;
            }
            (f as i32 as u32) as u64
        }
        P::I32TruncF64U => {
            let truncated = soft_f64_trunc(args[0]);
            let f = f64::from_bits(truncated);
            if f.is_nan() || f < 0.0 || f > u32::MAX as f64 {
                return None;
            }
            f as u32 as u64
        }
        P::I64TruncF32S => {
            let truncated = soft_f32_trunc(args[0] as u32);
            let f = f32::from_bits(truncated);
            if f.is_nan() || f < i64::MIN as f32 || f > i64::MAX as f32 {
                return None;
            }
            f as i64 as u64
        }
        P::I64TruncF32U => {
            let truncated = soft_f32_trunc(args[0] as u32);
            let f = f32::from_bits(truncated);
            if f.is_nan() || f < 0.0 || f > u64::MAX as f32 {
                return None;
            }
            f as u64
        }
        P::I64TruncF64S => {
            let truncated = soft_f64_trunc(args[0]);
            let f = f64::from_bits(truncated);
            if f.is_nan() || f < i64::MIN as f64 || f > i64::MAX as f64 {
                return None;
            }
            f as i64 as u64
        }
        P::I64TruncF64U => {
            let truncated = soft_f64_trunc(args[0]);
            let f = f64::from_bits(truncated);
            if f.is_nan() || f < 0.0 || f > u64::MAX as f64 {
                return None;
            }
            f as u64
        }
        P::I32TruncSatF32S => {
            (if f32::from_bits(args[0] as u32).is_nan() {
                0
            } else {
                (f32::from_bits(args[0] as u32) as i32)
                    .max(i32::MIN)
                    .min(i32::MAX)
            } as u32) as u64
        }
        P::I32TruncSatF32U => {
            let f = f32::from_bits(args[0] as u32);
            (if f.is_nan() || f < 0.0 {
                0u32
            } else if f >= u32::MAX as f32 {
                u32::MAX
            } else {
                f as u32
            }) as u64
        }
        P::I32TruncSatF64S => {
            let f = f64::from_bits(args[0]);
            (if f.is_nan() {
                0
            } else if f <= i32::MIN as f64 - 1.0 {
                i32::MIN
            } else if f >= i32::MAX as f64 + 1.0 {
                i32::MAX
            } else {
                f as i32
            } as u32) as u64
        }
        P::I32TruncSatF64U => {
            let f = f64::from_bits(args[0]);
            (if f.is_nan() || f < 0.0 {
                0u32
            } else if f >= u32::MAX as f64 + 1.0 {
                u32::MAX
            } else {
                f as u32
            }) as u64
        }
        P::I64TruncSatF32S => {
            let f = f32::from_bits(args[0] as u32);
            (if f.is_nan() {
                0i64
            } else {
                (f as i64).max(i64::MIN).min(i64::MAX)
            }) as u64
        }
        P::I64TruncSatF32U => {
            let f = f32::from_bits(args[0] as u32);
            if f.is_nan() || f < 0.0 {
                0
            } else if f >= u64::MAX as f32 {
                u64::MAX
            } else {
                f as u64
            }
        }
        P::I64TruncSatF64S => {
            let f = f64::from_bits(args[0]);
            (if f.is_nan() {
                0i64
            } else {
                (f as i64).max(i64::MIN).min(i64::MAX)
            }) as u64
        }
        P::I64TruncSatF64U => {
            let f = f64::from_bits(args[0]);
            if f.is_nan() || f < 0.0 {
                0
            } else if f >= u64::MAX as f64 {
                u64::MAX
            } else {
                f as u64
            }
        }
        _ => return None,
    };

    let const_primitive = match result_ty {
        ValueType::I32 => P::I32Const { value: bits as u32 },
        ValueType::I64 => P::I64Const { value: bits },
        ValueType::F32 => P::F32Const { value: bits as u32 },
        ValueType::F64 => P::F64Const { value: bits },
        _ => return None,
    };
    Some((bits, const_primitive))
}

#[inline]
fn bool32(value: bool) -> u64 {
    if value {
        1
    } else {
        0
    }
}

#[inline]
fn canon_f32(value: f32) -> u32 {
    if value.is_nan() {
        0x7fc0_0000
    } else {
        value.to_bits()
    }
}

#[inline]
fn canon_f64(value: f64) -> u64 {
    if value.is_nan() {
        0x7ff8_0000_0000_0000
    } else {
        value.to_bits()
    }
}

fn wasm_f32_min(a: u32, b: u32) -> u32 {
    let (fa, fb) = (f32::from_bits(a), f32::from_bits(b));
    if fa.is_nan() || fb.is_nan() {
        return 0x7fc0_0000;
    }
    if fa == 0.0 && fb == 0.0 {
        return if (a | b) & 0x8000_0000 != 0 {
            0x8000_0000
        } else {
            0
        };
    }
    canon_f32(fa.min(fb))
}

fn wasm_f32_max(a: u32, b: u32) -> u32 {
    let (fa, fb) = (f32::from_bits(a), f32::from_bits(b));
    if fa.is_nan() || fb.is_nan() {
        return 0x7fc0_0000;
    }
    if fa == 0.0 && fb == 0.0 {
        return if (a & b) & 0x8000_0000 != 0 {
            0x8000_0000
        } else {
            0
        };
    }
    canon_f32(fa.max(fb))
}

fn wasm_f64_min(a: u64, b: u64) -> u64 {
    let (fa, fb) = (f64::from_bits(a), f64::from_bits(b));
    if fa.is_nan() || fb.is_nan() {
        return 0x7ff8_0000_0000_0000;
    }
    if fa == 0.0 && fb == 0.0 {
        return if (a | b) & 0x8000_0000_0000_0000 != 0 {
            0x8000_0000_0000_0000
        } else {
            0
        };
    }
    canon_f64(fa.min(fb))
}

fn wasm_f64_max(a: u64, b: u64) -> u64 {
    let (fa, fb) = (f64::from_bits(a), f64::from_bits(b));
    if fa.is_nan() || fb.is_nan() {
        return 0x7ff8_0000_0000_0000;
    }
    if fa == 0.0 && fb == 0.0 {
        return if (a & b) & 0x8000_0000_0000_0000 != 0 {
            0x8000_0000_0000_0000
        } else {
            0
        };
    }
    canon_f64(fa.max(fb))
}

fn soft_f32_trunc(bits: u32) -> u32 {
    let f = f32::from_bits(bits);
    if f.is_nan() || f.is_infinite() || f == 0.0 {
        return bits;
    }
    let exp = ((bits >> 23) & 0xFF) as i32 - 127;
    if exp < 0 {
        return bits & 0x8000_0000;
    }
    if exp >= 23 {
        return bits;
    }
    f32::from_bits(bits & !(0x007F_FFFFu32 >> exp)).to_bits()
}

fn soft_f64_trunc(bits: u64) -> u64 {
    let f = f64::from_bits(bits);
    if f.is_nan() || f.is_infinite() || f == 0.0 {
        return bits;
    }
    let exp = ((bits >> 52) & 0x7FF) as i32 - 1023;
    if exp < 0 {
        return bits & 0x8000_0000_0000_0000;
    }
    if exp >= 52 {
        return bits;
    }
    f64::from_bits(bits & !(0x000F_FFFF_FFFF_FFFFu64 >> exp)).to_bits()
}

fn soft_f32_floor(bits: u32) -> u32 {
    let truncated = soft_f32_trunc(bits);
    let f = f32::from_bits(bits);
    let ft = f32::from_bits(truncated);
    if f < ft {
        (ft - 1.0).to_bits()
    } else {
        truncated
    }
}

fn soft_f64_floor(bits: u64) -> u64 {
    let truncated = soft_f64_trunc(bits);
    let f = f64::from_bits(bits);
    let ft = f64::from_bits(truncated);
    if f < ft {
        (ft - 1.0).to_bits()
    } else {
        truncated
    }
}

fn soft_f32_ceil(bits: u32) -> u32 {
    let truncated = soft_f32_trunc(bits);
    let f = f32::from_bits(bits);
    let ft = f32::from_bits(truncated);
    if f > ft {
        (ft + 1.0).to_bits()
    } else {
        truncated
    }
}

fn soft_f64_ceil(bits: u64) -> u64 {
    let truncated = soft_f64_trunc(bits);
    let f = f64::from_bits(bits);
    let ft = f64::from_bits(truncated);
    if f > ft {
        (ft + 1.0).to_bits()
    } else {
        truncated
    }
}

fn soft_f32_nearest(bits: u32) -> u32 {
    let f = f32::from_bits(bits);
    if f.is_nan() || f.is_infinite() || f == 0.0 {
        return bits;
    }
    let truncated = f32::from_bits(soft_f32_trunc(bits));
    let delta = (f - truncated).abs();
    if delta < 0.5 {
        return truncated.to_bits();
    }
    if delta > 0.5 {
        return (truncated + f.signum()).to_bits();
    }
    let rounded = truncated + f.signum();
    if (rounded as i64) % 2 == 0 {
        rounded.to_bits()
    } else {
        truncated.to_bits()
    }
}

fn soft_f64_nearest(bits: u64) -> u64 {
    let f = f64::from_bits(bits);
    if f.is_nan() || f.is_infinite() || f == 0.0 {
        return bits;
    }
    let truncated = f64::from_bits(soft_f64_trunc(bits));
    let delta = (f - truncated).abs();
    if delta < 0.5 {
        return truncated.to_bits();
    }
    if delta > 0.5 {
        return (truncated + f.signum()).to_bits();
    }
    let rounded = truncated + f.signum();
    if (rounded as i64) % 2 == 0 {
        rounded.to_bits()
    } else {
        truncated.to_bits()
    }
}

fn mark_terminator_uses(term: &SsaTerminator, used: &mut [bool]) {
    let mut mark = |value: SsaValue| {
        if let Some(slot) = used.get_mut(value.0 as usize) {
            *slot = true;
        }
    };
    match term {
        SsaTerminator::Goto(edge) => {
            for binding in &edge.bindings {
                mark(binding.value);
            }
        }
        SsaTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => {
            mark(*cond);
            for binding in &then_edge.bindings {
                mark(binding.value);
            }
            for binding in &else_edge.bindings {
                mark(binding.value);
            }
        }
        SsaTerminator::BrTable { index, entries } => {
            mark(*index);
            for edge in entries {
                for binding in &edge.bindings {
                    mark(binding.value);
                }
            }
        }
        SsaTerminator::Return { .. }
        | SsaTerminator::TrapUnreachable
        | SsaTerminator::EhThrow { .. }
        | SsaTerminator::EhThrowRef { .. } => {}
        SsaTerminator::ReturnScalar { result, .. } => {
            mark_scalar_result_uses(result, used);
        }
        SsaTerminator::TailCallDirect { args, .. } => {
            mark_call_args_uses(args, used);
        }
        SsaTerminator::TailCallIndirect { index, args, .. } => {
            mark_call_operand_uses(index, used);
            mark_call_args_uses(args, used);
        }
        SsaTerminator::TailCallRef {
            callee_ref, args, ..
        } => {
            mark_call_operand_uses(callee_ref, used);
            mark_call_args_uses(args, used);
        }
    }
}

fn mark_call_op_uses(call: &SsaCallOp, used: &mut [bool]) {
    match call {
        SsaCallOp::CallDirect { args, .. } => mark_call_args_uses(args, used),
        SsaCallOp::CallIndirect { index, args, .. } => {
            mark_call_operand_uses(index, used);
            mark_call_args_uses(args, used);
        }
        SsaCallOp::CallRef {
            callee_ref, args, ..
        } => {
            mark_call_operand_uses(callee_ref, used);
            mark_call_args_uses(args, used);
        }
    }
}

fn mark_call_args_uses(args: &SsaCallArgs, used: &mut [bool]) {
    for arg in &args.live_suffix {
        if let Some(slot) = used.get_mut(arg.value.0 as usize) {
            *slot = true;
        }
    }
}

fn mark_call_operand_uses(loc: &SsaCallOperandLoc, used: &mut [bool]) {
    if let SsaCallOperandLoc::Live { value, .. } = loc {
        if let Some(slot) = used.get_mut(value.0 as usize) {
            *slot = true;
        }
    }
}

fn mark_scalar_result_uses(loc: &SsaScalarResultLoc, used: &mut [bool]) {
    if let SsaScalarResultLoc::Live { value, .. } = loc {
        if let Some(slot) = used.get_mut(value.0 as usize) {
            *slot = true;
        }
    }
}

/// Scan every place a value may appear (params, op args & results, extra_args
/// operands, terminator) and return the highest-indexed `SsaValue` seen.
fn max_value_index_parts(
    params: &[SsaValue],
    ops: &[SsaInst],
    extra_args: &[SsaOperand],
    terminator: &SsaTerminator,
    program: &SsaProgram,
) -> Option<SsaValue> {
    let mut max_value = params.iter().copied().max();

    for inst in ops {
        if inst.op.is_primitive() {
            for operand in inst.args.iter() {
                if let Some(value) = operand.as_value() {
                    max_value = max_value.max(Some(value));
                }
            }
            if inst.result.is_some() {
                max_value = max_value.max(Some(inst.result));
            }
            // Primitive overflow operands live in `extra_args`; covered by the
            // blanket scan below.
            let _ = program;
        } else {
            match inst.op {
                SsaOp::FILL | SsaOp::CELL_GET_SLOT | SsaOp::CELL_GET_CACHE => {
                    max_value = max_value.max(Some(inst.result));
                }
                SsaOp::SPILL | SsaOp::CELL_SET_SLOT | SsaOp::CELL_SET_CACHE => {
                    if let Some(value) = inst.args[0].as_value() {
                        max_value = max_value.max(Some(value));
                    }
                }
                SsaOp::CELL_ENSURE_CACHE | SsaOp::CELL_RESERVE_CACHE | SsaOp::CELL_DROP_CACHE => {}
                SsaOp::CALL => {
                    if inst.result.is_some() {
                        max_value = max_value.max(Some(inst.result));
                    }
                    if let Some(call) = program.call_ops.get(inst.meta as usize) {
                        max_value = max_value.max(max_call_op_value(call));
                    }
                }
                _ => {}
            }
        }
    }

    for operand in extra_args {
        if let Some(value) = operand.as_value() {
            max_value = max_value.max(Some(value));
        }
    }

    match terminator {
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
        SsaTerminator::Return { .. }
        | SsaTerminator::TrapUnreachable
        | SsaTerminator::EhThrow { .. }
        | SsaTerminator::EhThrowRef { .. } => {}
        SsaTerminator::ReturnScalar { result, .. } => {
            max_value = max_value.max(max_scalar_result_value(result));
        }
        SsaTerminator::TailCallDirect { args, .. } => {
            max_value = max_value.max(max_call_args_value(args));
        }
        SsaTerminator::TailCallIndirect { index, args, .. } => {
            max_value = max_value.max(max_call_operand_value(index));
            max_value = max_value.max(max_call_args_value(args));
        }
        SsaTerminator::TailCallRef {
            callee_ref, args, ..
        } => {
            max_value = max_value.max(max_call_operand_value(callee_ref));
            max_value = max_value.max(max_call_args_value(args));
        }
    }

    max_value
}

fn max_call_op_value(call: &SsaCallOp) -> Option<SsaValue> {
    match call {
        SsaCallOp::CallDirect { args, .. } => max_call_args_value(args),
        SsaCallOp::CallIndirect { index, args, .. } => {
            max_call_operand_value(index).max(max_call_args_value(args))
        }
        SsaCallOp::CallRef {
            callee_ref, args, ..
        } => max_call_operand_value(callee_ref).max(max_call_args_value(args)),
    }
}

fn max_call_args_value(args: &SsaCallArgs) -> Option<SsaValue> {
    args.live_suffix.iter().map(|arg| arg.value).max()
}

fn max_call_operand_value(loc: &SsaCallOperandLoc) -> Option<SsaValue> {
    match loc {
        SsaCallOperandLoc::Stack { .. } => None,
        SsaCallOperandLoc::Live { value, .. } => Some(*value),
    }
}

fn max_scalar_result_value(loc: &SsaScalarResultLoc) -> Option<SsaValue> {
    match loc {
        SsaScalarResultLoc::Stack { .. } => None,
        SsaScalarResultLoc::Live { value, .. } => Some(*value),
    }
}

#[cfg(test)]
mod tests {
    use crate::collections;

    use super::{fold_constants_into_operands, SsaProgram};
    use crate::vm::{
        middle::frame::{FrameSlot, FrameSpan},
        middle::ssa_ir::{
            ir::{
                SsaBlock, SsaCallArgs, SsaCallLiveArg, SsaCallOp, SsaEdge, SsaInst, SsaOp,
                SsaOperand, SsaTerminator, SsaValue,
            },
            target::SsaTarget,
        },
        wasm::primitive_op::PrimitiveOpKind,
    };

    fn empty_program() -> SsaProgram {
        SsaProgram {
            cell_homes: collections::Vec::new(),
            entry: SsaTarget(0),
            blocks: collections::Vec::new(),
            cell_types: collections::Vec::new(),
            result_types: collections::Vec::new(),
            cell_info: collections::Vec::new(),
            block_entry_cached_cells: collections::Vec::new(),
            block_entry_cache_requirements: collections::Vec::new(),
            preferred_preserved: collections::Vec::new(),
            value_types: collections::Vec::new(),
            value_sink_cell: collections::Vec::new(),
            const_pool: collections::Vec::new(),
            primitive_pool: collections::Vec::new(),
            call_ops: collections::Vec::new(),
        }
    }

    #[test]
    fn folds_single_use_const_into_value_operand() {
        let mut program = empty_program();
        let const_pool_idx = program
            .intern_primitive(PrimitiveOpKind::I32Const { value: 7 })
            .unwrap();
        let add_pool_idx = program.intern_primitive(PrimitiveOpKind::I32Add).unwrap();
        let block = SsaBlock {
            id: SsaTarget(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                SsaInst::primitive(
                    const_pool_idx,
                    SsaValue(0),
                    [SsaOperand::NONE, SsaOperand::NONE],
                    0,
                ),
                SsaInst::primitive(
                    add_pool_idx,
                    SsaValue(2),
                    [
                        SsaOperand::value(SsaValue(1)),
                        SsaOperand::value(SsaValue(0))
                    ],
                    0,
                ),
            ],
            extra_args: collections::Vec::new(),
            terminator: SsaTerminator::Return { results: None },
        };
        program.blocks.push(block);
        program
            .block_entry_cached_cells
            .push(collections::Vec::new());
        program
            .block_entry_cache_requirements
            .push(collections::Vec::new());

        fold_constants_into_operands(&mut program, 0);

        let block = &program.blocks[0];
        assert_eq!(
            block.ops.len(),
            1,
            "dead const producer should be removed after folding"
        );
        let add_inst = &block.ops[0];
        assert!(add_inst.op.is_primitive());
        let absorbed_const = add_inst.args.iter().any(|operand| {
            matches!(
                operand.decode(),
                crate::vm::middle::ssa_ir::ir::DecodedOperand::Const(idx)
                    if program.const_pool[idx as usize] == 7
            )
        });
        assert!(
            absorbed_const,
            "mixed arithmetic should absorb the const operand"
        );
    }

    #[test]
    fn keeps_const_producer_when_value_is_used_by_terminator() {
        let mut program = empty_program();
        let const_pool_idx = program
            .intern_primitive(PrimitiveOpKind::I32Const { value: 1 })
            .unwrap();
        program.blocks.push(SsaBlock {
            id: SsaTarget(0),
            params: collections::Vec::new(),
            ops: collections::vec![SsaInst::primitive(
                const_pool_idx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0,
            )],
            extra_args: collections::Vec::new(),
            terminator: SsaTerminator::Branch {
                cond: SsaValue(0),
                then_edge: SsaEdge {
                    target: SsaTarget(1),
                    bindings: collections::Vec::new(),
                },
                else_edge: SsaEdge {
                    target: SsaTarget(2),
                    bindings: collections::Vec::new(),
                },
            },
        });
        program
            .block_entry_cached_cells
            .push(collections::Vec::new());
        program
            .block_entry_cache_requirements
            .push(collections::Vec::new());

        fold_constants_into_operands(&mut program, 0);

        let block = &program.blocks[0];
        assert_eq!(
            block.ops.len(),
            1,
            "terminator users must keep the const producer live"
        );
        let producer = &block.ops[0];
        assert_eq!(producer.op, SsaOp::primitive(const_pool_idx));
        assert!(producer.args[0].is_none() && producer.args[1].is_none());
        assert_eq!(producer.result, SsaValue(0));
        assert!(matches!(
            program.primitive_pool[const_pool_idx as usize],
            PrimitiveOpKind::I32Const { value: 1 }
        ));
    }

    #[test]
    fn keeps_const_producer_when_value_is_used_by_call_args() {
        let mut program = empty_program();
        let const_pool_idx = program
            .intern_primitive(PrimitiveOpKind::I32Const { value: 2 })
            .unwrap();
        program.call_ops.push(SsaCallOp::CallDirect {
            callee: 1,
            args: SsaCallArgs {
                frame_base: FrameSlot(0),
                total_params: 1,
                param_types: collections::vec![crate::value_type::ValueType::I32],
                stack_prefix_count: 0,
                live_suffix: collections::vec![SsaCallLiveArg {
                    param_index: 0,
                    value: SsaValue(0),
                    ty: crate::value_type::ValueType::I32,
                    frame_slot: FrameSlot(0),
                }],
            },
            results: FrameSpan::new(FrameSlot(0), 0),
            result_types: collections::Vec::new(),
        });
        program.blocks.push(SsaBlock {
            id: SsaTarget(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                SsaInst::primitive(
                    const_pool_idx,
                    SsaValue(0),
                    [SsaOperand::NONE, SsaOperand::NONE],
                    0,
                ),
                SsaInst::call(0),
            ],
            extra_args: collections::Vec::new(),
            terminator: SsaTerminator::Return { results: None },
        });
        program
            .block_entry_cached_cells
            .push(collections::Vec::new());
        program
            .block_entry_cache_requirements
            .push(collections::Vec::new());

        fold_constants_into_operands(&mut program, 0);

        let block = &program.blocks[0];
        assert_eq!(
            block.ops.len(),
            2,
            "call live arguments must keep their const producers live"
        );
        assert_eq!(block.ops[0].op, SsaOp::primitive(const_pool_idx));
        assert_eq!(block.ops[0].result, SsaValue(0));
        assert_eq!(block.ops[1].op, SsaOp::CALL);
    }
}
