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

use alloc::vec;
use alloc::vec::Vec;

use crate::{
    value_type::ValueType,
    vm::{
        middle::ssa_ir::{
            ir::{SsaBlock, SsaInstKind, SsaOperand, SsaProgram, SsaTerminator, SsaValue},
            leaf::SsaLeafOp,
        },
        wasm::primitive_op::{self, PrimitiveOpKind},
    },
};

pub(crate) fn optimize_program(program: &mut SsaProgram) {
    for block in &mut program.blocks {
        fold_constants_into_operands(block);
    }
}

/// Absorb single-use const producers into later leaf operands and fold fully
/// constant pure leaf ops.
///
/// This is intentionally block-local. Once cleanup has merged trivial CFG
/// structure, the profitable constant chains we care about are visible within
/// one prepared SSA block.
fn fold_constants_into_operands(block: &mut SsaBlock) {
    let max_val = max_value_index(block)
        .map(|value| value.0 as usize + 1)
        .unwrap_or(0);
    if max_val == 0 {
        return;
    }

    let mut known_const: Vec<Option<u64>> = vec![None; max_val];
    let mut used_in_terminator = vec![false; max_val];

    for inst in &block.ops {
        if let SsaInstKind::Value { op, args, results } = &inst.kind {
            if !args.is_empty() || results.len() != 1 {
                continue;
            }
            if let Some(bits) = const_bits_of_primitive(op.primitive()) {
                let value = results[0];
                if let Some(slot) = known_const.get_mut(value.0 as usize) {
                    *slot = Some(bits);
                }
            }
        }
    }
    mark_terminator_uses(&block.terminator, &mut used_in_terminator);

    for inst in &mut block.ops {
        let SsaInstKind::Value { op, args, results } = &mut inst.kind else {
            continue;
        };

        if !args.is_empty() && can_accept_const_operand(op.primitive()) {
            let const_args = args
                .iter()
                .filter_map(|operand| match operand {
                    SsaOperand::Value(value) => {
                        known_const.get(value.0 as usize).copied().flatten()
                    }
                    SsaOperand::Const(bits) => Some(*bits),
                })
                .collect::<Vec<_>>();
            if const_args.len() == args.len() {
                if let Some((result_bits, const_primitive)) = try_eval(op.primitive(), &const_args)
                {
                    if let Some(result) = results.first().copied() {
                        if let Some(slot) = known_const.get_mut(result.0 as usize) {
                            *slot = Some(result_bits);
                        }
                    }
                    *op = SsaLeafOp::from_primitive(const_primitive)
                        .expect("folded constant primitive must stay a valid leaf op");
                    args.clear();
                    continue;
                }
            }
        }

        if can_accept_const_operand(op.primitive()) {
            for operand in args.iter_mut() {
                let SsaOperand::Value(value) = operand else {
                    continue;
                };
                let index = value.0 as usize;
                if let Some(Some(bits)) = known_const.get(index) {
                    if !used_in_terminator.get(index).copied().unwrap_or(true) {
                        *operand = SsaOperand::Const(*bits);
                    }
                }
            }
        }
    }

    let mut still_used = vec![false; max_val];
    for inst in &block.ops {
        match &inst.kind {
            SsaInstKind::Value { args, .. } => {
                for operand in args {
                    if let SsaOperand::Value(value) = operand {
                        still_used[value.0 as usize] = true;
                    }
                }
            }
            SsaInstKind::LocalSetSlot { src, .. }
            | SsaInstKind::LocalSetCache { src, .. }
            | SsaInstKind::Spill { src, .. } => {
                still_used[src.0 as usize] = true;
            }
            SsaInstKind::Fill { .. }
            | SsaInstKind::LocalGetSlot { .. }
            | SsaInstKind::LocalGetCache { .. }
            | SsaInstKind::LocalEnsureCache { .. }
            | SsaInstKind::LocalReserveCache { .. }
            | SsaInstKind::LocalDropCache { .. }
            | SsaInstKind::Call(_) => {}
        }
    }
    for (index, used) in used_in_terminator.iter().copied().enumerate() {
        if used {
            still_used[index] = true;
        }
    }

    block.ops.retain(|inst| {
        let SsaInstKind::Value { args, results, .. } = &inst.kind else {
            return true;
        };
        if !args.is_empty() || results.len() != 1 {
            return true;
        }
        let index = results[0].0 as usize;
        known_const.get(index).copied().flatten().is_none()
            || still_used.get(index).copied().unwrap_or(true)
    });
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
        P::I32Add | P::I32Sub | P::I32Mul | P::I32DivS | P::I32DivU
            | P::I32RemS | P::I32RemU | P::I32And | P::I32Or | P::I32Xor
            | P::I32Shl | P::I32ShrS | P::I32ShrU | P::I32Rotl | P::I32Rotr
            | P::I64Add | P::I64Sub | P::I64Mul | P::I64DivS | P::I64DivU
            | P::I64RemS | P::I64RemU | P::I64And | P::I64Or | P::I64Xor
            | P::I64Shl | P::I64ShrS | P::I64ShrU | P::I64Rotl | P::I64Rotr
            | P::F32Add | P::F32Sub | P::F32Mul | P::F32Div | P::F32Min | P::F32Max
            | P::F32Copysign | P::F64Add | P::F64Sub | P::F64Mul | P::F64Div
            | P::F64Min | P::F64Max | P::F64Copysign
            | P::I32Eq | P::I32Ne | P::I32LtS | P::I32LtU | P::I32GtS | P::I32GtU
            | P::I32LeS | P::I32LeU | P::I32GeS | P::I32GeU
            | P::I64Eq | P::I64Ne | P::I64LtS | P::I64LtU | P::I64GtS | P::I64GtU
            | P::I64LeS | P::I64LeU | P::I64GeS | P::I64GeU
            | P::F32Eq | P::F32Ne | P::F32Lt | P::F32Gt | P::F32Le | P::F32Ge
            | P::F64Eq | P::F64Ne | P::F64Lt | P::F64Gt | P::F64Le | P::F64Ge
            | P::I32Eqz | P::I32Clz | P::I32Ctz | P::I32Popcnt
            | P::I64Eqz | P::I64Clz | P::I64Ctz | P::I64Popcnt
            | P::I32Extend8S | P::I32Extend16S
            | P::I64Extend8S | P::I64Extend16S | P::I64Extend32S
            | P::F32Abs | P::F32Neg | P::F32Ceil | P::F32Floor | P::F32Trunc
            | P::F32Nearest | P::F32Sqrt | P::F64Abs | P::F64Neg | P::F64Ceil
            | P::F64Floor | P::F64Trunc | P::F64Nearest | P::F64Sqrt
            | P::I32WrapI64 | P::I64ExtendI32S | P::I64ExtendI32U
            | P::I32TruncF32S | P::I32TruncF32U | P::I32TruncF64S | P::I32TruncF64U
            | P::I64TruncF32S | P::I64TruncF32U | P::I64TruncF64S | P::I64TruncF64U
            | P::I32TruncSatF32S | P::I32TruncSatF32U
            | P::I32TruncSatF64S | P::I32TruncSatF64U
            | P::I64TruncSatF32S | P::I64TruncSatF32U
            | P::I64TruncSatF64S | P::I64TruncSatF64U
            | P::F32ConvertI32S | P::F32ConvertI32U | P::F32ConvertI64S
            | P::F32ConvertI64U | P::F64ConvertI32S | P::F64ConvertI32U
            | P::F64ConvertI64S | P::F64ConvertI64U | P::F32DemoteF64
            | P::F64PromoteF32 | P::I32ReinterpretF32 | P::I64ReinterpretF64
            | P::F32ReinterpretI32 | P::F64ReinterpretI64
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
        P::F32Add => canon_f32(f32::from_bits(args[0] as u32) + f32::from_bits(args[1] as u32)) as u64,
        P::F32Sub => canon_f32(f32::from_bits(args[0] as u32) - f32::from_bits(args[1] as u32)) as u64,
        P::F32Mul => canon_f32(f32::from_bits(args[0] as u32) * f32::from_bits(args[1] as u32)) as u64,
        P::F32Div => canon_f32(f32::from_bits(args[0] as u32) / f32::from_bits(args[1] as u32)) as u64,
        P::F32Min => wasm_f32_min(args[0] as u32, args[1] as u32) as u64,
        P::F32Max => wasm_f32_max(args[0] as u32, args[1] as u32) as u64,
        P::F32Copysign => f32::from_bits(args[0] as u32).copysign(f32::from_bits(args[1] as u32)).to_bits() as u64,
        P::F64Add => canon_f64(f64::from_bits(args[0]) + f64::from_bits(args[1])),
        P::F64Sub => canon_f64(f64::from_bits(args[0]) - f64::from_bits(args[1])),
        P::F64Mul => canon_f64(f64::from_bits(args[0]) * f64::from_bits(args[1])),
        P::F64Div => canon_f64(f64::from_bits(args[0]) / f64::from_bits(args[1])),
        P::F64Min => wasm_f64_min(args[0], args[1]),
        P::F64Max => wasm_f64_max(args[0], args[1]),
        P::F64Copysign => f64::from_bits(args[0]).copysign(f64::from_bits(args[1])).to_bits(),
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
        P::I32TruncSatF32S => (if f32::from_bits(args[0] as u32).is_nan() {
            0
        } else {
            (f32::from_bits(args[0] as u32) as i32).max(i32::MIN).min(i32::MAX)
        } as u32) as u64,
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
            (if f.is_nan() { 0i64 } else { (f as i64).max(i64::MIN).min(i64::MAX) }) as u64
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
            (if f.is_nan() { 0i64 } else { (f as i64).max(i64::MIN).min(i64::MAX) }) as u64
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
    if value { 1 } else { 0 }
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

fn max_value_index(block: &SsaBlock) -> Option<SsaValue> {
    let mut max_value = block.params.iter().copied().max();

    for inst in &block.ops {
        match &inst.kind {
            SsaInstKind::Value { args, results, .. } => {
                max_value = max_value.max(
                    args.iter()
                        .filter_map(|operand| match operand {
                            SsaOperand::Value(value) => Some(*value),
                            SsaOperand::Const(_) => None,
                        })
                        .max(),
                );
                max_value = max_value.max(results.iter().copied().max());
            }
            SsaInstKind::LocalGetSlot { dst, .. }
            | SsaInstKind::LocalGetCache { dst, .. }
            | SsaInstKind::Fill { dst, .. } => {
                max_value = max_value.max(Some(*dst));
            }
            SsaInstKind::LocalSetSlot { src, .. }
            | SsaInstKind::LocalSetCache { src, .. }
            | SsaInstKind::Spill { src, .. } => {
                max_value = max_value.max(Some(*src));
            }
            SsaInstKind::LocalEnsureCache { .. }
            | SsaInstKind::LocalReserveCache { .. }
            | SsaInstKind::LocalDropCache { .. }
            | SsaInstKind::Call(_) => {}
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
