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
        backend::BackendConfig,
        middle::ssa_ir::ir::{
            SsaBinding, SsaEdge, SsaInst, SsaOp, SsaOperand, SsaProgram, SsaTerminator, SsaValue,
        },
        wasm::primitive_op::{self, PrimitiveOpKind},
    },
};

pub(crate) fn optimize_program(program: &mut SsaProgram, config: BackendConfig) {
    for block_idx in 0..program.blocks.len() {
        fold_constants_into_operands(program, block_idx);
        cse_local_slot_reads(program, block_idx, config);
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
            // 3-arg primitives pull their third operand from extra_args via
            // `meta`. We consult stack_effect to know whether this applies.
            let pool_idx = inst.op.as_primitive_idx().expect("primitive op") as usize;
            let kind = &program.primitive_pool[pool_idx];
            if primitive_op::stack_effect(kind).0 == 3 {
                if let Some(operand) = extra_args.get(inst.meta as usize) {
                    if let Some(value) = operand.as_value() {
                        still_used[value.0 as usize] = true;
                    }
                }
            }
        } else {
            match inst.op {
                SsaOp::SPILL | SsaOp::LOCAL_SET_SLOT | SsaOp::LOCAL_SET_CACHE => {
                    if let Some(value) = inst.args[0].as_value() {
                        still_used[value.0 as usize] = true;
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
        SsaTerminator::Return { .. } | SsaTerminator::TrapUnreachable => {}
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
            // 3-arg primitives' third operand is in extra_args[meta]. Covered
            // by the blanket scan below.
            let _ = program;
        } else {
            match inst.op {
                SsaOp::FILL | SsaOp::LOCAL_GET_SLOT | SsaOp::LOCAL_GET_CACHE => {
                    max_value = max_value.max(Some(inst.result));
                }
                SsaOp::SPILL | SsaOp::LOCAL_SET_SLOT | SsaOp::LOCAL_SET_CACHE => {
                    if let Some(value) = inst.args[0].as_value() {
                        max_value = max_value.max(Some(value));
                    }
                }
                SsaOp::LOCAL_ENSURE_CACHE
                | SsaOp::LOCAL_RESERVE_CACHE
                | SsaOp::LOCAL_DROP_CACHE
                | SsaOp::CALL => {}
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
        SsaTerminator::Return { .. } | SsaTerminator::TrapUnreachable => {}
    }

    max_value
}

// ── Block-local CSE for `LocalGetSlot` loads ────────────────────────────────
//
// ALGORITHM4 decides public residency at the region level by maximizing the
// total weighted frame-op benefit within capacity. When capacity is tight
// (small `gp_dynamic_budget`, e.g. x86_64) a hot read-only local can be
// priced out of residency even though it is read many times within a single
// straight-line block. Without this pass every lexical `local.get` of such a
// slot lowers to a fresh memory load, and the post-regalloc MIR peephole
// cannot recover: by the time `reuse_loaded_values` sees the redundant
// loads, the backend allocator has already reused the destination register
// for follow-on ops that consume the loaded value, clobbering the tracked
// load source before the next read appears.
//
// This is the intra-block complement `ALGORITHM4.md` calls "private flex
// promotion": dedupe repeated slot reads into a single SSA value within one
// block, leaving region-level decisions untouched. Downstream, the machine
// allocator observes one SSA value with many uses and naturally keeps it in
// a register across those uses.

#[derive(Clone, Copy)]
struct ValueSubstitution {
    from: SsaValue,
    to: SsaValue,
}

fn cse_local_slot_reads(program: &mut SsaProgram, block_idx: usize, config: BackendConfig) {
    // Pressure guard: the allocator downstream sees +1 concurrent live
    // SsaValue for every hoist whose window covers a given op. Over-
    // hoisting pushes the per-op live count past the dynamic GP budget and
    // the lowerer then fails with "prepared SSA-IR exceeded dynamic
    // register budget during native lowering in block b for value".
    //
    // Compute pre-CSE max live count across the block and cap the number
    // of simultaneously open hoists at (allocatable_budget - peak_live -
    // margin). margin=1 keeps one scratch slot for the peephole/reg-alloc
    // plumbing. When no slack exists, skip CSE entirely for the block.
    let gp_budget = usize::from(config.allocatable_gp_dynamic_budget());
    let fp_budget = usize::from(config.fp_dynamic_budget);
    // Use the tighter of the two banks for the cap: a hoist may target
    // either GP or FP depending on the slot's value type, and v1 does not
    // distinguish per-bank pressure. min(gp, fp) is the safe conservative
    // choice.
    let budget = gp_budget.min(fp_budget);
    // Reserve a margin for: scratch reg needs (call arg setup reserves
    // N free dynamic regs beyond the arg values themselves on some
    // ABIs), and any transient pressure the estimator may under-count
    // (values reused across the terminator, values flowing in from
    // block params that are also produced inside, pair-register
    // demands for 32-bit `i64` on ARMv7, etc.).
    const PRESSURE_MARGIN: usize = 3;

    let params = core::mem::take(&mut program.blocks[block_idx].params);
    let peak_live = block_peak_live_count(
        &program.blocks[block_idx].ops,
        &program.blocks[block_idx].extra_args,
        &program.blocks[block_idx].terminator,
        &params,
    );
    // Cached locals hold dynamic-register lanes across the whole block, so
    // their count is additive on top of SSA transient pressure from the
    // regalloc's perspective. `block_entry_cached_slots[block_idx]` is the
    // set ALGORITHM4 chose to keep resident here.
    let cached_slots = program
        .block_entry_cached_slots
        .get(block_idx)
        .map(|slots| slots.len())
        .unwrap_or(0);
    let hoist_cap = budget
        .saturating_sub(peak_live)
        .saturating_sub(cached_slots)
        .saturating_sub(PRESSURE_MARGIN);
    if hoist_cap == 0 {
        program.blocks[block_idx].params = params;
        return;
    }

    let mut ops = core::mem::take(&mut program.blocks[block_idx].ops);
    let mut extra_args = core::mem::take(&mut program.blocks[block_idx].extra_args);
    let terminator = core::mem::replace(
        &mut program.blocks[block_idx].terminator,
        SsaTerminator::TrapUnreachable,
    );

    // hoisted[slot_idx] = Some(v_first) when `v_first` is the in-block
    // canonical SsaValue currently holding fp[slot_idx]'s memory contents.
    // None when the slot has been invalidated (writer seen, cache state
    // changed, or a call barrier was crossed).
    let mut hoisted: collections::Vec<Option<SsaValue>> = collections::Vec::new();

    // Accumulated substitutions: uses of `from` get rewritten to `to`.
    // Only `from` values that correspond to removed `LocalGetSlot` results
    // are ever inserted, and the `to` side is always a still-live prior
    // result, so no chained lookup is needed.
    let mut subst: collections::Vec<ValueSubstitution> = collections::Vec::new();

    // Active hoists counter — caps concurrent open hoist windows to
    // `hoist_cap`. Over-approximation (never decremented) keeps the check
    // correct regardless of actual overlap and avoids a second pass.
    let mut active_hoists: usize = 0;

    let mut new_ops = collections::Vec::with_capacity(ops.len());

    for mut inst in ops.drain(..) {
        // Rewrite all operand-bearing fields on this inst first, so every
        // preserved op sees the canonical values. Args[0..1] cover every
        // inline operand surface (primitive args, set-op sources). The
        // third arg of 3-arg primitives lives in `extra_args` and is
        // substituted in the post-pass sweep below.
        inst.args[0] = substitute_operand(inst.args[0], &subst);
        inst.args[1] = substitute_operand(inst.args[1], &subst);

        match inst.op {
            SsaOp::LOCAL_GET_SLOT => {
                let slot_idx = inst.meta as usize;
                if slot_idx >= hoisted.len() {
                    hoisted.resize(slot_idx + 1, None);
                }
                if let Some(existing) = hoisted[slot_idx] {
                    // Redundant load: drop the inst and substitute all
                    // later uses of its result with the canonical value.
                    subst.push(ValueSubstitution {
                        from: inst.result,
                        to: existing,
                    });
                    continue;
                }
                // First read of this slot in the block. Only open a hoist
                // window if pressure headroom remains.
                if active_hoists < hoist_cap {
                    hoisted[slot_idx] = Some(inst.result);
                    active_hoists += 1;
                }
                new_ops.push(inst);
            }
            SsaOp::LOCAL_SET_SLOT
            | SsaOp::LOCAL_SET_CACHE
            | SsaOp::SPILL
            | SsaOp::LOCAL_ENSURE_CACHE
            | SsaOp::LOCAL_RESERVE_CACHE
            | SsaOp::LOCAL_DROP_CACHE => {
                // Any writer or cache-state change on a slot invalidates
                // the canonical load for that slot.
                let slot_idx = inst.meta as usize;
                if slot_idx < hoisted.len() {
                    hoisted[slot_idx] = None;
                }
                new_ops.push(inst);
            }
            SsaOp::CALL => {
                // v1 conservative: a call boundary may force the machine
                // allocator to spill/reload anything it was holding; treat
                // every hoisted slot as invalidated and re-load lazily on
                // the far side.
                for h in hoisted.iter_mut() {
                    *h = None;
                }
                new_ops.push(inst);
            }
            _ => {
                // Primitives, Fill, LocalGetCache: no frame-slot writes.
                // Primitive args are already substituted above.
                new_ops.push(inst);
            }
        }
    }

    // Third arg of 3-arg primitives (Select, bulk memory/table ops) lives
    // in `extra_args` indexed by `inst.meta`. Sweep once to substitute.
    for operand in extra_args.iter_mut() {
        *operand = substitute_operand(*operand, &subst);
    }

    let terminator = substitute_terminator(terminator, &subst);

    program.blocks[block_idx].ops = new_ops;
    program.blocks[block_idx].extra_args = extra_args;
    program.blocks[block_idx].terminator = terminator;
    program.blocks[block_idx].params = params;
}

/// Forward-liveness sweep over one block: returns the maximum concurrent
/// live SsaValue count at any point in the block.
///
/// Considers block params (live at entry) and op results (defined on
/// emission). A value dies at its last use within the block; uses in the
/// terminator or outgoing edge bindings count as uses at the terminator
/// position (i.e. at `ops.len()`).
fn block_peak_live_count(
    ops: &[SsaInst],
    extra_args: &[SsaOperand],
    terminator: &SsaTerminator,
    params: &[SsaValue],
) -> usize {
    // Find the highest SsaValue id touched in this block so we can size
    // the last-use table tightly. `SsaValue::NONE` (== u32::MAX) is never
    // a real id, so cap at a value we can actually observe.
    let mut max_id: u32 = 0;
    let bump = |max_id: &mut u32, v: SsaValue| {
        if !v.is_none() && v.0 > *max_id {
            *max_id = v.0;
        }
    };
    for v in params {
        bump(&mut max_id, *v);
    }
    for inst in ops {
        bump(&mut max_id, inst.result);
        for operand in inst.args {
            if let Some(v) = operand.as_value() {
                bump(&mut max_id, v);
            }
        }
    }
    for operand in extra_args {
        if let Some(v) = operand.as_value() {
            bump(&mut max_id, v);
        }
    }
    for_each_terminator_value(terminator, |v| bump(&mut max_id, v));

    let table_len = (max_id as usize).saturating_add(1);
    // last_use[v] = index into a logical timeline; ops.len() == "terminator".
    let mut last_use: collections::Vec<Option<usize>> =
        collections::vec![None; table_len];

    let term_idx = ops.len();
    for (i, inst) in ops.iter().enumerate() {
        for operand in inst.args {
            if let Some(v) = operand.as_value() {
                let slot = &mut last_use[v.0 as usize];
                if slot.map_or(true, |prev| prev < i) {
                    *slot = Some(i);
                }
            }
        }
    }
    for operand in extra_args {
        if let Some(v) = operand.as_value() {
            let slot = &mut last_use[v.0 as usize];
            if slot.map_or(true, |prev| prev < term_idx) {
                *slot = Some(term_idx);
            }
        }
    }
    for_each_terminator_value(terminator, |v| {
        let slot = &mut last_use[v.0 as usize];
        if slot.map_or(true, |prev| prev < term_idx) {
            *slot = Some(term_idx);
        }
    });

    // Forward sweep: start with params live. On each op, add result to
    // live set, then drop any value whose last use was this op.
    let mut live = 0usize;
    let mut peak = 0usize;
    // Track which values are currently live via a bitset-like Vec<bool>.
    // Sized to max_id+1; not memory-critical for the per-block pass.
    let mut is_live: collections::Vec<bool> = collections::vec![false; table_len];

    for v in params {
        if !v.is_none() {
            let idx = v.0 as usize;
            if !is_live[idx] {
                is_live[idx] = true;
                live += 1;
            }
        }
    }
    peak = peak.max(live);

    for (i, inst) in ops.iter().enumerate() {
        if !inst.result.is_none() {
            let idx = inst.result.0 as usize;
            if !is_live[idx] {
                is_live[idx] = true;
                live += 1;
            }
        }
        peak = peak.max(live);
        // Drop args whose last use was at this op. Also drop the result
        // if the value is defined and never used (last_use == None).
        for operand in inst.args {
            if let Some(v) = operand.as_value() {
                let idx = v.0 as usize;
                if let Some(last) = last_use[idx] {
                    if last == i && is_live[idx] {
                        is_live[idx] = false;
                        live -= 1;
                    }
                }
            }
        }
        if !inst.result.is_none() {
            let idx = inst.result.0 as usize;
            if last_use[idx].is_none() && is_live[idx] {
                // Dead define: dies immediately after emission.
                is_live[idx] = false;
                live -= 1;
            }
        }
    }

    peak
}

fn for_each_terminator_value(terminator: &SsaTerminator, mut visit: impl FnMut(SsaValue)) {
    match terminator {
        SsaTerminator::Goto(edge) => {
            for binding in &edge.bindings {
                visit(binding.value);
            }
        }
        SsaTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => {
            visit(*cond);
            for binding in &then_edge.bindings {
                visit(binding.value);
            }
            for binding in &else_edge.bindings {
                visit(binding.value);
            }
        }
        SsaTerminator::BrTable { index, entries } => {
            visit(*index);
            for edge in entries {
                for binding in &edge.bindings {
                    visit(binding.value);
                }
            }
        }
        SsaTerminator::Return { .. } | SsaTerminator::TrapUnreachable => {}
    }
}

fn substitute_operand(operand: SsaOperand, subst: &[ValueSubstitution]) -> SsaOperand {
    match operand.as_value() {
        Some(value) => {
            let substituted = substitute_value(value, subst);
            if substituted == value {
                operand
            } else {
                SsaOperand::value(substituted)
            }
        }
        None => operand,
    }
}

#[inline]
fn substitute_value(value: SsaValue, subst: &[ValueSubstitution]) -> SsaValue {
    subst
        .iter()
        .find(|entry| entry.from == value)
        .map_or(value, |entry| entry.to)
}

fn substitute_terminator(
    terminator: SsaTerminator,
    subst: &[ValueSubstitution],
) -> SsaTerminator {
    match terminator {
        SsaTerminator::Goto(edge) => SsaTerminator::Goto(substitute_edge(edge, subst)),
        SsaTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => SsaTerminator::Branch {
            cond: substitute_value(cond, subst),
            then_edge: substitute_edge(then_edge, subst),
            else_edge: substitute_edge(else_edge, subst),
        },
        SsaTerminator::BrTable { index, entries } => SsaTerminator::BrTable {
            index: substitute_value(index, subst),
            entries: entries
                .into_iter()
                .map(|edge| substitute_edge(edge, subst))
                .collect(),
        },
        SsaTerminator::Return { results } => SsaTerminator::Return { results },
        SsaTerminator::TrapUnreachable => SsaTerminator::TrapUnreachable,
    }
}

fn substitute_edge(edge: SsaEdge, subst: &[ValueSubstitution]) -> SsaEdge {
    SsaEdge {
        target: edge.target,
        bindings: edge
            .bindings
            .into_iter()
            .map(|binding| SsaBinding {
                param: binding.param,
                value: substitute_value(binding.value, subst),
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use crate::collections;

    use super::{fold_constants_into_operands, SsaProgram};
    use crate::vm::{
        middle::ssa_ir::{
            ir::{SsaBlock, SsaEdge, SsaInst, SsaOp, SsaOperand, SsaTerminator, SsaValue},
            target::SsaTarget,
        },
        wasm::primitive_op::PrimitiveOpKind,
    };

    fn empty_program() -> SsaProgram {
        SsaProgram {
            entry: SsaTarget(0),
            blocks: collections::Vec::new(),
            local_slot_types: collections::Vec::new(),
            local_slot_info: collections::Vec::new(),
            block_entry_cached_slots: collections::Vec::new(),
            block_cfg_origins: collections::Vec::new(),
            value_types: collections::Vec::new(),
            value_sink_local: collections::Vec::new(),
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
            .block_entry_cached_slots
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
            .block_entry_cached_slots
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
}
