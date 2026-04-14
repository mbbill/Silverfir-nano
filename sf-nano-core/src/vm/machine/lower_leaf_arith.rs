//! Arithmetic, compare, convert, and select lowering.

use crate::collections;

use crate::{
    error::WasmError,
    vm::{
        machine::machine_ir::{
            MachineCompareKind, MachineConvertOp, MachineFloatBinaryOp, MachineFloatUnaryOp,
            MachineFloatWidth, MachineInst, MachineInstKind, MachineIntBinaryOp, MachineIntUnaryOp,
            MachineIntWidth, MachineRegOwner, MachineSign, MachineValue,
        },
        middle::ssa_ir::ir::{DecodedOperand, SsaOperand, SsaValue},
        wasm::primitive_op::PrimitiveOpKind,
    },
};

use super::{
    lower_context::BlockLowerContext,
    lower_regalloc::{convert_result_float_width, lir_value_storage_type},
    lower_util::{single_arg, single_result, two_args},
};

impl<'a> BlockLowerContext<'a> {
    /// Resolve an operand to a `MachineValue`.
    ///
    /// - `SsaOperand::Value(v)` → allocate/lookup a register via `use_value`.
    /// - `SsaOperand::Const(bits)` → `MachineValue::Imm64(bits)` with no register
    ///   allocation.  The architecture backend encodes this as a native
    ///   immediate when possible, or materializes into a scratch register.
    pub(super) fn lower_operand(&mut self, operand: SsaOperand) -> Result<MachineValue, WasmError> {
        match operand.decode() {
            DecodedOperand::Value(v) => Ok(MachineValue::Reg(self.use_value(v)?)),
            DecodedOperand::Const(idx) => {
                let bits = self.program().const_pool[idx as usize];
                Ok(MachineValue::Imm64(bits))
            }
            DecodedOperand::None => Err(WasmError::internal(
                "lower_operand called with a NONE SsaOperand",
            )),
        }
    }

    pub(super) fn lower_const(&mut self, results: &[SsaValue], imm: u64) -> Result<(), WasmError> {
        let dst = single_result(results)?;
        let dst_reg = self.alloc_value(dst)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Move {
                owner: MachineRegOwner::LinearValue,
                ty: lir_value_storage_type(self.program(), dst),
                dst: dst_reg,
                src: MachineValue::Imm64(imm),
            },
        });
        Ok(())
    }

    pub(super) fn lower_float_const(
        &mut self,
        results: &[SsaValue],
        width: MachineFloatWidth,
        bits: u64,
    ) -> Result<(), WasmError> {
        let dst = self.alloc_float_value(single_result(results)?, width)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::FloatConst { width, dst, bits },
        });
        Ok(())
    }

    pub(super) fn lower_int_unary(
        &mut self,
        args: &[SsaOperand],
        results: &[SsaValue],
        width: MachineIntWidth,
        op: MachineIntUnaryOp,
    ) -> Result<(), WasmError> {
        let src_op = single_arg(args)?;
        let src = self.lower_operand(src_op)?;
        let dead: collections::Vec<SsaValue> = args
            .iter()
            .filter_map(|a| match a.decode() {
                DecodedOperand::Value(v) => Some(v),
                DecodedOperand::Const(_) | DecodedOperand::None => None,
            })
            .collect();
        let dst = self.alloc_value_reusing_dead_inputs(single_result(results)?, &dead)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::IntUnary {
                width,
                op,
                dst,
                src,
            },
        });
        Ok(())
    }

    pub(super) fn lower_int_binary(
        &mut self,
        args: &[SsaOperand],
        results: &[SsaValue],
        width: MachineIntWidth,
        op: MachineIntBinaryOp,
    ) -> Result<(), WasmError> {
        let (lhs_op, rhs_op) = two_args(args)?;
        let lhs = self.lower_operand(lhs_op)?;
        let rhs = self.lower_operand(rhs_op)?;
        let dead: collections::Vec<SsaValue> = args
            .iter()
            .filter_map(|a| match a.decode() {
                DecodedOperand::Value(v) => Some(v),
                DecodedOperand::Const(_) | DecodedOperand::None => None,
            })
            .collect();
        let dst = self.alloc_value_reusing_dead_inputs(single_result(results)?, &dead)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::IntBinary {
                width,
                op,
                dst,
                lhs,
                rhs,
            },
        });
        Ok(())
    }

    pub(super) fn lower_int_compare(
        &mut self,
        args: &[SsaOperand],
        results: &[SsaValue],
        width: MachineIntWidth,
        kind: MachineCompareKind,
        sign: MachineSign,
    ) -> Result<(), WasmError> {
        let (lhs_op, rhs_op) = two_args(args)?;
        let lhs = self.lower_operand(lhs_op)?;
        let rhs = self.lower_operand(rhs_op)?;
        let dead: collections::Vec<SsaValue> = args
            .iter()
            .filter_map(|a| match a.decode() {
                DecodedOperand::Value(v) => Some(v),
                DecodedOperand::Const(_) | DecodedOperand::None => None,
            })
            .collect();
        let dst = self.alloc_value_reusing_dead_inputs(single_result(results)?, &dead)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::IntCompare {
                width,
                kind,
                sign,
                dst,
                lhs,
                rhs,
            },
        });
        Ok(())
    }

    /// Lower wasm `i32.eqz` / `i64.eqz` as `IntCompare { Eq, rhs: Imm64(0) }`.
    ///
    /// Eqz is bit-for-bit identical to comparing against zero, but emitting it
    /// as `IntCompare` lets `fuse_compare_branch` collapse the common
    /// `i32.eqz; br_if` pattern into a single `cbz`/`cbnz`. The MIR layer no
    /// longer carries a separate `Eqz` opcode.
    pub(super) fn lower_int_eqz(
        &mut self,
        args: &[SsaOperand],
        results: &[SsaValue],
        width: MachineIntWidth,
    ) -> Result<(), WasmError> {
        let src_op = single_arg(args)?;
        let lhs = self.lower_operand(src_op)?;
        let dead: collections::Vec<SsaValue> = args
            .iter()
            .filter_map(|a| match a.decode() {
                DecodedOperand::Value(v) => Some(v),
                DecodedOperand::Const(_) | DecodedOperand::None => None,
            })
            .collect();
        let dst = self.alloc_value_reusing_dead_inputs(single_result(results)?, &dead)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::IntCompare {
                width,
                kind: MachineCompareKind::Eq,
                sign: MachineSign::Unsigned,
                dst,
                lhs,
                rhs: MachineValue::Imm64(0),
            },
        });
        Ok(())
    }

    pub(super) fn lower_float_binary(
        &mut self,
        args: &[SsaOperand],
        results: &[SsaValue],
        width: MachineFloatWidth,
        op: MachineFloatBinaryOp,
    ) -> Result<(), WasmError> {
        let (lhs_op, rhs_op) = two_args(args)?;
        let lhs = self.lower_operand(lhs_op)?;
        let rhs = self.lower_operand(rhs_op)?;
        let dead: collections::Vec<SsaValue> = args
            .iter()
            .filter_map(|a| match a.decode() {
                DecodedOperand::Value(v) => Some(v),
                DecodedOperand::Const(_) | DecodedOperand::None => None,
            })
            .collect();
        let dst =
            self.alloc_float_value_reusing_dead_inputs(single_result(results)?, &dead, width)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::FloatBinary {
                width,
                op,
                dst,
                lhs,
                rhs,
            },
        });
        Ok(())
    }

    pub(super) fn lower_float_compare(
        &mut self,
        args: &[SsaOperand],
        results: &[SsaValue],
        width: MachineFloatWidth,
        kind: MachineCompareKind,
    ) -> Result<(), WasmError> {
        let (lhs_op, rhs_op) = two_args(args)?;
        let lhs = self.lower_operand(lhs_op)?;
        let rhs = self.lower_operand(rhs_op)?;
        let dead: collections::Vec<SsaValue> = args
            .iter()
            .filter_map(|a| match a.decode() {
                DecodedOperand::Value(v) => Some(v),
                DecodedOperand::Const(_) | DecodedOperand::None => None,
            })
            .collect();
        let dst = self.alloc_value_reusing_dead_inputs(single_result(results)?, &dead)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::FloatCompare {
                width,
                kind,
                dst,
                lhs,
                rhs,
            },
        });
        Ok(())
    }

    pub(super) fn lower_float_unary(
        &mut self,
        args: &[SsaOperand],
        results: &[SsaValue],
        width: MachineFloatWidth,
        op: MachineFloatUnaryOp,
    ) -> Result<(), WasmError> {
        let src_op = single_arg(args)?;
        let src = self.lower_operand(src_op)?;
        let dead: collections::Vec<SsaValue> = args
            .iter()
            .filter_map(|a| match a.decode() {
                DecodedOperand::Value(v) => Some(v),
                DecodedOperand::Const(_) | DecodedOperand::None => None,
            })
            .collect();
        let dst =
            self.alloc_float_value_reusing_dead_inputs(single_result(results)?, &dead, width)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::FloatUnary {
                width,
                op,
                dst,
                src,
            },
        });
        Ok(())
    }

    pub(super) fn lower_convert(
        &mut self,
        args: &[SsaOperand],
        results: &[SsaValue],
        op: MachineConvertOp,
    ) -> Result<(), WasmError> {
        let src_op = single_arg(args)?;
        let src = self.lower_operand(src_op)?;
        let dead: collections::Vec<SsaValue> = args
            .iter()
            .filter_map(|a| match a.decode() {
                DecodedOperand::Value(v) => Some(v),
                DecodedOperand::Const(_) | DecodedOperand::None => None,
            })
            .collect();
        let dst = if let Some(width) = convert_result_float_width(op) {
            self.alloc_float_value_reusing_dead_inputs(single_result(results)?, &dead, width)?
        } else {
            self.alloc_value_reusing_dead_inputs(single_result(results)?, &dead)?
        };
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Convert { op, dst, src },
        });
        Ok(())
    }

    pub(super) fn lower_select(
        &mut self,
        args: &[SsaOperand],
        results: &[SsaValue],
    ) -> Result<(), WasmError> {
        if args.len() != 3 {
            return Err(WasmError::internal("select expects three arguments"));
        }
        let on_true = self.use_operand(args[0])?;
        let on_false = self.use_operand(args[1])?;
        let cond = self.use_operand(args[2])?;
        let dead_inputs: collections::Vec<_> = args
            .iter()
            .filter_map(|a| match a.decode() {
                DecodedOperand::Value(v) => Some(v),
                DecodedOperand::Const(_) | DecodedOperand::None => None,
            })
            .collect();
        let dst = self.alloc_value_reusing_dead_inputs(single_result(results)?, &dead_inputs)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Select {
                ty: lir_value_storage_type(self.program(), single_result(results)?),
                dst,
                on_true: MachineValue::Reg(on_true),
                on_false: MachineValue::Reg(on_false),
                cond: MachineValue::Reg(cond),
            },
        });
        Ok(())
    }

    pub(super) fn lower_ref_is_null(
        &mut self,
        args: &[SsaOperand],
        results: &[SsaValue],
    ) -> Result<(), WasmError> {
        let src_value = single_arg(args)?.unwrap_value();
        let src = self.use_value(src_value)?;
        let dst = self.alloc_value_reusing_dead_inputs(single_result(results)?, &[src_value])?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::IntCompare {
                width: self.gp_word_int_width(),
                kind: MachineCompareKind::Eq,
                sign: MachineSign::Unsigned,
                dst,
                lhs: MachineValue::Reg(src),
                rhs: MachineValue::Imm64(self.gp_word_max_imm()),
            },
        });
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Mapping table functions (primitive op -> machine IR op)
// ---------------------------------------------------------------------------

pub(super) fn machine_int_binary(
    primitive: &PrimitiveOpKind,
) -> Option<(MachineIntWidth, MachineIntBinaryOp)> {
    use MachineIntBinaryOp as Op;
    use MachineIntWidth as W;
    use PrimitiveOpKind as P;

    Some(match primitive {
        P::I32Add => (W::I32, Op::Add),
        P::I32Sub => (W::I32, Op::Sub),
        P::I32Mul => (W::I32, Op::Mul),
        P::I32DivS => (W::I32, Op::DivS),
        P::I32DivU => (W::I32, Op::DivU),
        P::I32RemS => (W::I32, Op::RemS),
        P::I32RemU => (W::I32, Op::RemU),
        P::I32And => (W::I32, Op::And),
        P::I32Or => (W::I32, Op::Or),
        P::I32Xor => (W::I32, Op::Xor),
        P::I32Shl => (W::I32, Op::Shl),
        P::I32ShrS => (W::I32, Op::ShrS),
        P::I32ShrU => (W::I32, Op::ShrU),
        P::I32Rotl => (W::I32, Op::Rotl),
        P::I32Rotr => (W::I32, Op::Rotr),
        P::I64Add => (W::I64, Op::Add),
        P::I64Sub => (W::I64, Op::Sub),
        P::I64Mul => (W::I64, Op::Mul),
        P::I64DivS => (W::I64, Op::DivS),
        P::I64DivU => (W::I64, Op::DivU),
        P::I64RemS => (W::I64, Op::RemS),
        P::I64RemU => (W::I64, Op::RemU),
        P::I64And => (W::I64, Op::And),
        P::I64Or => (W::I64, Op::Or),
        P::I64Xor => (W::I64, Op::Xor),
        P::I64Shl => (W::I64, Op::Shl),
        P::I64ShrS => (W::I64, Op::ShrS),
        P::I64ShrU => (W::I64, Op::ShrU),
        P::I64Rotl => (W::I64, Op::Rotl),
        P::I64Rotr => (W::I64, Op::Rotr),
        _ => return None,
    })
}

pub(super) fn machine_int_compare(
    primitive: &PrimitiveOpKind,
) -> Option<(MachineIntWidth, MachineCompareKind, MachineSign)> {
    use MachineCompareKind as K;
    use MachineIntWidth as W;
    use MachineSign as S;
    use PrimitiveOpKind as P;

    Some(match primitive {
        P::I32Eq => (W::I32, K::Eq, S::Unsigned),
        P::I32Ne => (W::I32, K::Ne, S::Unsigned),
        P::I32LtS => (W::I32, K::Lt, S::Signed),
        P::I32LtU => (W::I32, K::Lt, S::Unsigned),
        P::I32GtS => (W::I32, K::Gt, S::Signed),
        P::I32GtU => (W::I32, K::Gt, S::Unsigned),
        P::I32LeS => (W::I32, K::Le, S::Signed),
        P::I32LeU => (W::I32, K::Le, S::Unsigned),
        P::I32GeS => (W::I32, K::Ge, S::Signed),
        P::I32GeU => (W::I32, K::Ge, S::Unsigned),
        P::I64Eq => (W::I64, K::Eq, S::Unsigned),
        P::I64Ne => (W::I64, K::Ne, S::Unsigned),
        P::I64LtS => (W::I64, K::Lt, S::Signed),
        P::I64LtU => (W::I64, K::Lt, S::Unsigned),
        P::I64GtS => (W::I64, K::Gt, S::Signed),
        P::I64GtU => (W::I64, K::Gt, S::Unsigned),
        P::I64LeS => (W::I64, K::Le, S::Signed),
        P::I64LeU => (W::I64, K::Le, S::Unsigned),
        P::I64GeS => (W::I64, K::Ge, S::Signed),
        P::I64GeU => (W::I64, K::Ge, S::Unsigned),
        _ => return None,
    })
}

pub(super) fn machine_int_unary(
    primitive: &PrimitiveOpKind,
) -> Option<(MachineIntWidth, MachineIntUnaryOp)> {
    use MachineIntUnaryOp as Op;
    use MachineIntWidth as W;
    use PrimitiveOpKind as P;

    Some(match primitive {
        P::I32Clz => (W::I32, Op::Clz),
        P::I32Ctz => (W::I32, Op::Ctz),
        P::I32Popcnt => (W::I32, Op::Popcnt),
        P::I64Clz => (W::I64, Op::Clz),
        P::I64Ctz => (W::I64, Op::Ctz),
        P::I64Popcnt => (W::I64, Op::Popcnt),
        P::I32Extend8S => (W::I32, Op::Extend8S),
        P::I32Extend16S => (W::I32, Op::Extend16S),
        P::I64Extend8S => (W::I64, Op::Extend8S),
        P::I64Extend16S => (W::I64, Op::Extend16S),
        P::I64Extend32S => (W::I64, Op::Extend32S),
        _ => return None,
    })
}

pub(super) fn machine_float_binary(
    primitive: &PrimitiveOpKind,
) -> Option<(MachineFloatWidth, MachineFloatBinaryOp)> {
    use MachineFloatBinaryOp as Op;
    use MachineFloatWidth as W;
    use PrimitiveOpKind as P;

    Some(match primitive {
        P::F32Add => (W::F32, Op::Add),
        P::F32Sub => (W::F32, Op::Sub),
        P::F32Mul => (W::F32, Op::Mul),
        P::F32Div => (W::F32, Op::Div),
        P::F32Min => (W::F32, Op::Min),
        P::F32Max => (W::F32, Op::Max),
        P::F32Copysign => (W::F32, Op::Copysign),
        P::F64Add => (W::F64, Op::Add),
        P::F64Sub => (W::F64, Op::Sub),
        P::F64Mul => (W::F64, Op::Mul),
        P::F64Div => (W::F64, Op::Div),
        P::F64Min => (W::F64, Op::Min),
        P::F64Max => (W::F64, Op::Max),
        P::F64Copysign => (W::F64, Op::Copysign),
        _ => return None,
    })
}

pub(super) fn machine_float_compare(
    primitive: &PrimitiveOpKind,
) -> Option<(MachineFloatWidth, MachineCompareKind)> {
    use MachineCompareKind as K;
    use MachineFloatWidth as W;
    use PrimitiveOpKind as P;

    Some(match primitive {
        P::F32Eq => (W::F32, K::Eq),
        P::F32Ne => (W::F32, K::Ne),
        P::F32Lt => (W::F32, K::Lt),
        P::F32Gt => (W::F32, K::Gt),
        P::F32Le => (W::F32, K::Le),
        P::F32Ge => (W::F32, K::Ge),
        P::F64Eq => (W::F64, K::Eq),
        P::F64Ne => (W::F64, K::Ne),
        P::F64Lt => (W::F64, K::Lt),
        P::F64Gt => (W::F64, K::Gt),
        P::F64Le => (W::F64, K::Le),
        P::F64Ge => (W::F64, K::Ge),
        _ => return None,
    })
}

pub(super) fn machine_float_unary(
    primitive: &PrimitiveOpKind,
) -> Option<(MachineFloatWidth, MachineFloatUnaryOp)> {
    use MachineFloatUnaryOp as Op;
    use MachineFloatWidth as W;
    use PrimitiveOpKind as P;

    Some(match primitive {
        P::F32Abs => (W::F32, Op::Abs),
        P::F32Neg => (W::F32, Op::Neg),
        P::F32Ceil => (W::F32, Op::Ceil),
        P::F32Floor => (W::F32, Op::Floor),
        P::F32Trunc => (W::F32, Op::Trunc),
        P::F32Nearest => (W::F32, Op::Nearest),
        P::F32Sqrt => (W::F32, Op::Sqrt),
        P::F64Abs => (W::F64, Op::Abs),
        P::F64Neg => (W::F64, Op::Neg),
        P::F64Ceil => (W::F64, Op::Ceil),
        P::F64Floor => (W::F64, Op::Floor),
        P::F64Trunc => (W::F64, Op::Trunc),
        P::F64Nearest => (W::F64, Op::Nearest),
        P::F64Sqrt => (W::F64, Op::Sqrt),
        _ => return None,
    })
}

pub(super) fn machine_convert(primitive: &PrimitiveOpKind) -> Option<MachineConvertOp> {
    use MachineConvertOp as Op;
    use PrimitiveOpKind as P;

    Some(match primitive {
        P::I32WrapI64 => Op::I32WrapI64,
        P::I64ExtendI32S => Op::I64ExtendI32S,
        P::I64ExtendI32U => Op::I64ExtendI32U,
        P::I32TruncF32S => Op::I32TruncF32S,
        P::I32TruncF32U => Op::I32TruncF32U,
        P::I32TruncF64S => Op::I32TruncF64S,
        P::I32TruncF64U => Op::I32TruncF64U,
        P::I64TruncF32S => Op::I64TruncF32S,
        P::I64TruncF32U => Op::I64TruncF32U,
        P::I64TruncF64S => Op::I64TruncF64S,
        P::I64TruncF64U => Op::I64TruncF64U,
        P::I32TruncSatF32S => Op::I32TruncSatF32S,
        P::I32TruncSatF32U => Op::I32TruncSatF32U,
        P::I32TruncSatF64S => Op::I32TruncSatF64S,
        P::I32TruncSatF64U => Op::I32TruncSatF64U,
        P::I64TruncSatF32S => Op::I64TruncSatF32S,
        P::I64TruncSatF32U => Op::I64TruncSatF32U,
        P::I64TruncSatF64S => Op::I64TruncSatF64S,
        P::I64TruncSatF64U => Op::I64TruncSatF64U,
        P::F32ConvertI32S => Op::F32ConvertI32S,
        P::F32ConvertI32U => Op::F32ConvertI32U,
        P::F32ConvertI64S => Op::F32ConvertI64S,
        P::F32ConvertI64U => Op::F32ConvertI64U,
        P::F64ConvertI32S => Op::F64ConvertI32S,
        P::F64ConvertI32U => Op::F64ConvertI32U,
        P::F64ConvertI64S => Op::F64ConvertI64S,
        P::F64ConvertI64U => Op::F64ConvertI64U,
        P::F32DemoteF64 => Op::F32DemoteF64,
        P::F64PromoteF32 => Op::F64PromoteF32,
        P::I32ReinterpretF32 => Op::I32ReinterpretF32,
        P::I64ReinterpretF64 => Op::I64ReinterpretF64,
        P::F32ReinterpretI32 => Op::F32ReinterpretI32,
        P::F64ReinterpretI64 => Op::F64ReinterpretI64,
        _ => return None,
    })
}
