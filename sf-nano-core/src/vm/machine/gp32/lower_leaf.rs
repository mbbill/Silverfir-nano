//! 32-bit i64 pair lowering — splitting i64 operations into lo/hi word pairs.

use crate::{
    error::WasmError,
    vm::{
        machine::machine_ir::{
            MachineCompareKind, MachineFloatWidth, MachineInst, MachineInstKind,
            MachineIntBinaryOp, MachineSign, MachineStorageType, MachineValue,
        },
        middle::ssa_ir::ir::{SsaOperand, SsaValue},
        wasm::primitive_op::PrimitiveOpKind,
    },
};

use super::super::{
    lower_leaf_arith::{machine_convert, machine_int_binary, machine_int_compare, machine_int_unary},
    lower_context::BlockLowerContext,
    lower_util::{single_arg, single_result, two_args},
};

impl<'a> BlockLowerContext<'a> {
    pub(super) fn lower_i64_pair_leaf(
        &mut self,
        primitive: &PrimitiveOpKind,
        args: &[SsaOperand],
        results: &[SsaValue],
    ) -> Result<bool, WasmError> {
        use MachineCompareKind as Cmp;
        use MachineFloatWidth as Fw;
        use MachineIntBinaryOp as BinOp;
        use MachineSign as Sign;
        use MachineStorageType as Ty;
        use PrimitiveOpKind as P;

        let result_ty = results
            .first()
            .copied()
            .map(|value| self.value_storage_type(value));

        match primitive {
            P::I64Const { value } => {
                let (dst_lo, dst_hi) = self.alloc_i64_value_pair(single_result(results)?)?;
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Move {
                        ty: Ty::GpWord,
                        dst: dst_lo,
                        src: MachineValue::Imm64(*value as u32 as u64),
                    },
                });
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Move {
                        ty: Ty::GpWord,
                        dst: dst_hi,
                        src: MachineValue::Imm64((*value >> 32) as u32 as u64),
                    },
                });
                Ok(true)
            }
            P::Select if matches!(result_ty, Some(Ty::GpI64)) => {
                if args.len() != 3 {
                    return Err(WasmError::internal("select expects three arguments".into()));
                }
                let true_val = args[0].unwrap_value();
                let false_val = args[1].unwrap_value();
                let cond_val = args[2].unwrap_value();
                let (true_lo, true_hi) = self.use_i64_value_pair(true_val)?;
                let (false_lo, false_hi) = self.use_i64_value_pair(false_val)?;
                let cond = self.use_value(cond_val)?;
                let (dst_lo, dst_hi) =
                    self.alloc_i64_value_pair_reusing_dead_inputs(single_result(results)?, &[true_val, false_val, cond_val])?;
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Select {
                        ty: Ty::GpWord,
                        dst: dst_lo,
                        on_true: MachineValue::Reg(true_lo),
                        on_false: MachineValue::Reg(false_lo),
                        cond: MachineValue::Reg(cond),
                    },
                });
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Select {
                        ty: Ty::GpWord,
                        dst: dst_hi,
                        on_true: MachineValue::Reg(true_hi),
                        on_false: MachineValue::Reg(false_hi),
                        cond: MachineValue::Reg(cond),
                    },
                });
                Ok(true)
            }
            P::I64Add | P::I64Sub | P::I64Mul | P::I64And | P::I64Or | P::I64Xor => {
                let (lhs_value, rhs_value) = {
                    let (a, b) = two_args(args)?;
                    (a.unwrap_value(), b.unwrap_value())
                };
                let (lhs_lo, lhs_hi) = self.use_i64_value_pair(lhs_value)?;
                let (rhs_lo, rhs_hi) = self.use_i64_value_pair(rhs_value)?;
                let (dst_lo, dst_hi) =
                    self.alloc_i64_value_pair_reusing_dead_inputs(single_result(results)?, &[lhs_value, rhs_value])?;
                let op = machine_int_binary(primitive)
                    .ok_or_else(|| WasmError::internal("missing i64 binary lowering".into()))?
                    .1;
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Int64PairBinary {
                        op,
                        dst_lo,
                        dst_hi,
                        lhs_lo: MachineValue::Reg(lhs_lo),
                        lhs_hi: MachineValue::Reg(lhs_hi),
                        rhs_lo: MachineValue::Reg(rhs_lo),
                        rhs_hi: MachineValue::Reg(rhs_hi),
                    },
                });
                Ok(true)
            }
            P::I64DivS | P::I64DivU | P::I64RemS | P::I64RemU => {
                let (lhs_value, rhs_value) = {
                    let (a, b) = two_args(args)?;
                    (a.unwrap_value(), b.unwrap_value())
                };
                let (lhs_lo, lhs_hi) = self.use_i64_value_pair(lhs_value)?;
                let (rhs_lo, rhs_hi) = self.use_i64_value_pair(rhs_value)?;
                let (dst_lo, dst_hi) =
                    self.alloc_i64_value_pair_reusing_dead_inputs(single_result(results)?, &[lhs_value, rhs_value])?;
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Int64PairDivRem {
                        sign: match primitive {
                            P::I64DivS | P::I64RemS => Sign::Signed,
                            _ => Sign::Unsigned,
                        },
                        rem: matches!(primitive, P::I64RemS | P::I64RemU),
                        dst_lo,
                        dst_hi,
                        lhs_lo: MachineValue::Reg(lhs_lo),
                        lhs_hi: MachineValue::Reg(lhs_hi),
                        rhs_lo: MachineValue::Reg(rhs_lo),
                        rhs_hi: MachineValue::Reg(rhs_hi),
                    },
                });
                Ok(true)
            }
            P::I64Shl | P::I64ShrS | P::I64ShrU | P::I64Rotl | P::I64Rotr => {
                let (lhs_value, rhs_value) = {
                    let (a, b) = two_args(args)?;
                    (a.unwrap_value(), b.unwrap_value())
                };
                let (lhs_lo, lhs_hi) = self.use_i64_value_pair(lhs_value)?;
                let (rhs_lo, _) = self.use_i64_value_pair(rhs_value)?;
                let (dst_lo, dst_hi) =
                    self.alloc_i64_value_pair_reusing_dead_inputs(single_result(results)?, &[lhs_value, rhs_value])?;
                let op = machine_int_binary(primitive)
                    .ok_or_else(|| WasmError::internal("missing i64 shift lowering".into()))?
                    .1;
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Int64PairShift {
                        op,
                        dst_lo,
                        dst_hi,
                        lhs_lo: MachineValue::Reg(lhs_lo),
                        lhs_hi: MachineValue::Reg(lhs_hi),
                        rhs: MachineValue::Reg(rhs_lo),
                    },
                });
                Ok(true)
            }
            P::I64Eq
            | P::I64Ne
            | P::I64LtS
            | P::I64LtU
            | P::I64GtS
            | P::I64GtU
            | P::I64LeS
            | P::I64LeU
            | P::I64GeS
            | P::I64GeU => {
                let (lhs_value, rhs_value) = {
                    let (a, b) = two_args(args)?;
                    (a.unwrap_value(), b.unwrap_value())
                };
                let (lhs_lo, lhs_hi) = self.use_i64_value_pair(lhs_value)?;
                let (rhs_lo, rhs_hi) = self.use_i64_value_pair(rhs_value)?;
                let dst = self.alloc_value_reusing_dead_inputs(
                    single_result(results)?,
                    &[lhs_value, rhs_value],
                )?;
                let (_, kind, sign) = machine_int_compare(primitive)
                    .ok_or_else(|| WasmError::internal("missing i64 compare lowering".into()))?;
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Int64PairCompare {
                        kind,
                        sign,
                        dst,
                        lhs_lo: MachineValue::Reg(lhs_lo),
                        lhs_hi: MachineValue::Reg(lhs_hi),
                        rhs_lo: MachineValue::Reg(rhs_lo),
                        rhs_hi: MachineValue::Reg(rhs_hi),
                    },
                });
                Ok(true)
            }
            P::I64Eqz => {
                let src_value = single_arg(args)?.unwrap_value();
                let (src_lo, src_hi) = self.use_i64_value_pair(src_value)?;
                let dst =
                    self.alloc_value_reusing_dead_inputs(single_result(results)?, &[src_value])?;
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Int64PairCompare {
                        kind: Cmp::Eq,
                        sign: Sign::Unsigned,
                        dst,
                        lhs_lo: MachineValue::Reg(src_lo),
                        lhs_hi: MachineValue::Reg(src_hi),
                        rhs_lo: MachineValue::Imm64(0),
                        rhs_hi: MachineValue::Imm64(0),
                    },
                });
                Ok(true)
            }
            P::I64Clz
            | P::I64Ctz
            | P::I64Popcnt
            | P::I64Extend8S
            | P::I64Extend16S
            | P::I64Extend32S => {
                let src_value = single_arg(args)?.unwrap_value();
                let (src_lo, src_hi) = self.use_i64_value_pair(src_value)?;
                let (dst_lo, dst_hi) = self.alloc_i64_value_pair_reusing_dead_inputs(
                    single_result(results)?,
                    &[src_value],
                )?;
                let op = machine_int_unary(primitive)
                    .ok_or_else(|| WasmError::internal("missing i64 unary lowering".into()))?
                    .1;
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Int64PairUnary {
                        op,
                        dst_lo,
                        dst_hi,
                        src_lo: MachineValue::Reg(src_lo),
                        src_hi: MachineValue::Reg(src_hi),
                    },
                });
                Ok(true)
            }
            P::I32WrapI64 => {
                let src_value = single_arg(args)?.unwrap_value();
                let (src_lo, _) = self.use_i64_value_pair(src_value)?;
                let dst =
                    self.alloc_value_reusing_dead_inputs(single_result(results)?, &[src_value])?;
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Move {
                        ty: Ty::GpWord,
                        dst,
                        src: MachineValue::Reg(src_lo),
                    },
                });
                Ok(true)
            }
            P::I64ExtendI32S | P::I64ExtendI32U => {
                let src_value = single_arg(args)?.unwrap_value();
                let src = self.use_value(src_value)?;
                let (dst_lo, dst_hi) = self.alloc_i64_value_pair_reusing_dead_inputs(
                    single_result(results)?,
                    &[src_value],
                )?;
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Move {
                        ty: Ty::GpWord,
                        dst: dst_lo,
                        src: MachineValue::Reg(src),
                    },
                });
                match primitive {
                    P::I64ExtendI32S => {
                        self.emit_machine_inst(MachineInst {
                            kind: MachineInstKind::IntBinary {
                                width: self.gp_word_int_width(),
                                op: BinOp::ShrS,
                                dst: dst_hi,
                                lhs: MachineValue::Reg(src),
                                rhs: MachineValue::Imm64(31),
                            },
                        });
                    }
                    P::I64ExtendI32U => {
                        self.emit_machine_inst(MachineInst {
                            kind: MachineInstKind::Move {
                                ty: Ty::GpWord,
                                dst: dst_hi,
                                src: MachineValue::Imm64(0),
                            },
                        });
                    }
                    _ => unreachable!("filtered i64 extend"),
                }
                Ok(true)
            }
            P::F32ConvertI64S | P::F32ConvertI64U | P::F64ConvertI64S | P::F64ConvertI64U => {
                let src_value = single_arg(args)?.unwrap_value();
                let (src_lo, src_hi) = self.use_i64_value_pair(src_value)?;
                let width = match primitive {
                    P::F32ConvertI64S | P::F32ConvertI64U => Fw::F32,
                    _ => Fw::F64,
                };
                let dst = self.alloc_float_value_reusing_dead_inputs(
                    single_result(results)?,
                    &[src_value],
                    width,
                )?;
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::ConvertI64PairToFloat {
                        width,
                        sign: match primitive {
                            P::F32ConvertI64S | P::F64ConvertI64S => Sign::Signed,
                            _ => Sign::Unsigned,
                        },
                        dst,
                        src_lo: MachineValue::Reg(src_lo),
                        src_hi: MachineValue::Reg(src_hi),
                    },
                });
                Ok(true)
            }
            P::I64TruncF32S
            | P::I64TruncF32U
            | P::I64TruncF64S
            | P::I64TruncF64U
            | P::I64TruncSatF32S
            | P::I64TruncSatF32U
            | P::I64TruncSatF64S
            | P::I64TruncSatF64U => {
                let src_value = single_arg(args)?.unwrap_value();
                let src = self.use_value(src_value)?;
                let (dst_lo, dst_hi) = self.alloc_i64_value_pair_reusing_dead_inputs(
                    single_result(results)?,
                    &[src_value],
                )?;
                let op = machine_convert(primitive)
                    .ok_or_else(|| WasmError::internal("missing i64 trunc lowering".into()))?;
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::ConvertFloatToI64Pair {
                        op,
                        dst_lo,
                        dst_hi,
                        src: MachineValue::Reg(src),
                    },
                });
                Ok(true)
            }
            P::I64ReinterpretF64 => {
                let src_value = single_arg(args)?.unwrap_value();
                let src = self.use_value(src_value)?;
                let (dst_lo, dst_hi) = self.alloc_i64_value_pair_reusing_dead_inputs(
                    single_result(results)?,
                    &[src_value],
                )?;
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::ReinterpretF64ToI64Pair {
                        dst_lo,
                        dst_hi,
                        src: MachineValue::Reg(src),
                    },
                });
                Ok(true)
            }
            P::F64ReinterpretI64 => {
                let src_value = single_arg(args)?.unwrap_value();
                let (src_lo, src_hi) = self.use_i64_value_pair(src_value)?;
                let dst = self.alloc_float_value_reusing_dead_inputs(
                    single_result(results)?,
                    &[src_value],
                    Fw::F64,
                )?;
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::ReinterpretI64PairToF64 {
                        dst,
                        src_lo: MachineValue::Reg(src_lo),
                        src_hi: MachineValue::Reg(src_hi),
                    },
                });
                Ok(true)
            }
            _ => Ok(false),
        }
    }
}
