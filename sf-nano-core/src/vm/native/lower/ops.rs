use crate::{
    error::WasmError,
    vm::{
        lir::{ir::LirValue, leaf::LirLeafOp},
        wasm::primitive_op::PrimitiveOpKind,
    },
};

use super::context::BlockLowerContext;

impl<'a> BlockLowerContext<'a> {
    pub(super) fn lower_leaf(
        &mut self,
        op: &LirLeafOp,
        args: &[LirValue],
        results: &[LirValue],
    ) -> Result<(), WasmError> {
        use PrimitiveOpKind as P;
        let primitive = op.primitive();

        match primitive {
            P::Drop | P::Nop => {
                for arg in args {
                    let _ = self.use_value(*arg)?;
                }
                Ok(())
            }
            P::I32Const { value } => self.lower_const(results, *value as u64),
            P::I64Const { value } => self.lower_const(results, *value),
            P::F32Const { value } => self.lower_const(results, *value as u64),
            P::F64Const { value } => self.lower_const(results, *value),
            P::RefNull => self.lower_const(results, usize::MAX as u64),
            P::RefFunc { func_idx } => self.lower_const(results, *func_idx as u64),
            P::RefIsNull => self.lower_ref_is_null(args, results),
            P::Select => self.lower_select(args, results),
            primitive => {
                if let Some((width, op)) = machine_int_binary(primitive) {
                    return self.lower_int_binary(args, results, width, op);
                }
                if let Some((width, kind, sign)) = machine_int_compare(primitive) {
                    return self.lower_int_compare(args, results, width, kind, sign);
                }
                if let Some((width, op)) = machine_int_unary(primitive) {
                    return self.lower_int_unary(args, results, width, op);
                }
                if let Some((width, op)) = machine_float_binary(primitive) {
                    return self.lower_float_binary(args, results, width, op);
                }
                if let Some((width, kind)) = machine_float_compare(primitive) {
                    return self.lower_float_compare(args, results, width, kind);
                }
                if let Some((width, op)) = machine_float_unary(primitive) {
                    return self.lower_float_unary(args, results, width, op);
                }
                if let Some(op) = machine_convert(primitive) {
                    return self.lower_convert(args, results, op);
                }

                Err(WasmError::internal(alloc::format!(
                    "primitive {:?} is not lowered to MachineIR yet",
                    primitive
                )))
            }
        }
    }
}

fn machine_int_binary(
    primitive: &PrimitiveOpKind,
) -> Option<(
    crate::vm::native::ir::machine::MachineIntWidth,
    crate::vm::native::ir::machine::MachineIntBinaryOp,
)> {
    use crate::vm::native::ir::machine::{MachineIntBinaryOp as Op, MachineIntWidth as W};
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

fn machine_int_compare(
    primitive: &PrimitiveOpKind,
) -> Option<(
    crate::vm::native::ir::machine::MachineIntWidth,
    crate::vm::native::ir::machine::MachineCompareKind,
    crate::vm::native::ir::machine::MachineSign,
)> {
    use crate::vm::native::ir::machine::{
        MachineCompareKind as K, MachineIntWidth as W, MachineSign as S,
    };
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

fn machine_int_unary(
    primitive: &PrimitiveOpKind,
) -> Option<(
    crate::vm::native::ir::machine::MachineIntWidth,
    crate::vm::native::ir::machine::MachineIntUnaryOp,
)> {
    use crate::vm::native::ir::machine::{MachineIntUnaryOp as Op, MachineIntWidth as W};
    use PrimitiveOpKind as P;

    Some(match primitive {
        P::I32Eqz => (W::I32, Op::Eqz),
        P::I32Clz => (W::I32, Op::Clz),
        P::I32Ctz => (W::I32, Op::Ctz),
        P::I32Popcnt => (W::I32, Op::Popcnt),
        P::I64Eqz => (W::I64, Op::Eqz),
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

fn machine_float_binary(
    primitive: &PrimitiveOpKind,
) -> Option<(
    crate::vm::native::ir::machine::MachineFloatWidth,
    crate::vm::native::ir::machine::MachineFloatBinaryOp,
)> {
    use crate::vm::native::ir::machine::{MachineFloatBinaryOp as Op, MachineFloatWidth as W};
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

fn machine_float_compare(
    primitive: &PrimitiveOpKind,
) -> Option<(
    crate::vm::native::ir::machine::MachineFloatWidth,
    crate::vm::native::ir::machine::MachineCompareKind,
)> {
    use crate::vm::native::ir::machine::{MachineCompareKind as K, MachineFloatWidth as W};
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

fn machine_float_unary(
    primitive: &PrimitiveOpKind,
) -> Option<(
    crate::vm::native::ir::machine::MachineFloatWidth,
    crate::vm::native::ir::machine::MachineFloatUnaryOp,
)> {
    use crate::vm::native::ir::machine::{MachineFloatUnaryOp as Op, MachineFloatWidth as W};
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

fn machine_convert(
    primitive: &PrimitiveOpKind,
) -> Option<crate::vm::native::ir::machine::MachineConvertOp> {
    use crate::vm::native::ir::machine::MachineConvertOp as Op;
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
