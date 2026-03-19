use alloc::vec;
use alloc::vec::Vec;

use crate::error::WasmError;

use super::{
    machine_ptr_width, machine_word_int_width, MachineBranchCond, MachineConvertOp,
    MachineFloatWidth, MachineFunction, MachineInstKind, MachineIntWidth, MachineMemWidth,
    MachineModule, MachineProgram, MachineReg, MachineStorageType, MachineTerminator, MachineValue,
    MACHINE_CTX_REG, MACHINE_FP_REG, MACHINE_MEM0_BASE_REG, MACHINE_MEM0_SIZE_REG,
};

impl MachineModule {
    /// Run 32-bit legalization prep.
    ///
    /// This is intentionally a scaffold at first: on 32-bit targets it proves
    /// that MachineIR carries enough storage information to drive a later
    /// register-pair rewrite without backend-specific guessing. The actual
    /// rewrite lands in follow-up patches.
    pub fn legalize(&mut self, gp_reg_width: u8) -> Result<(), WasmError> {
        if gp_reg_width != 4 {
            return Ok(());
        }

        for MachineFunction { program, .. } in &self.functions {
            let _ = program.infer_32bit_reg_storage_types()?;
        }

        Ok(())
    }
}

impl MachineProgram {
    pub(crate) fn infer_32bit_reg_storage_types(
        &self,
    ) -> Result<Vec<Option<MachineStorageType>>, WasmError> {
        let mut types = vec![None; self.reg_count as usize];
        let gp_word_mem_width = machine_ptr_width(4);
        let gp_word_int_width = machine_word_int_width(4);

        for reg in [
            MACHINE_CTX_REG,
            MACHINE_FP_REG,
            MACHINE_MEM0_BASE_REG,
            MACHINE_MEM0_SIZE_REG,
        ] {
            if reg.0 < self.reg_count {
                assign_reg_type(
                    &mut types,
                    reg,
                    MachineStorageType::GpWord,
                    "fixed register",
                )?;
            }
        }

        for (index, width) in self.fp_reg_init_widths.iter().copied().enumerate() {
            let Some(width) = width else {
                continue;
            };
            let reg = MachineReg(self.first_fp_reg.checked_add(index as u16).ok_or_else(|| {
                WasmError::internal(
                    "machine FP init register index overflowed register space".into(),
                )
            })?);
            if reg.0 >= self.reg_count {
                return Err(WasmError::internal(alloc::format!(
                    "machine fp_reg_init_widths entry {} references out-of-range register {}",
                    index,
                    reg.0,
                )));
            }
            assign_reg_type(
                &mut types,
                reg,
                float_storage_type(width),
                "FP init storage",
            )?;
        }

        for block in &self.blocks {
            for param in &block.params {
                assign_reg_type(&mut types, param.reg, param.ty, "block parameter storage")?;
            }

            for inst in &block.ops {
                infer_inst_types(
                    &mut types,
                    &inst.kind,
                    self.first_fp_reg,
                    gp_word_mem_width,
                    gp_word_int_width,
                )?;
            }

            infer_term_types(
                &mut types,
                &block.terminator,
                self.first_fp_reg,
                gp_word_mem_width,
                gp_word_int_width,
            )?;
        }

        Ok(types)
    }
}

fn infer_inst_types(
    types: &mut [Option<MachineStorageType>],
    inst: &MachineInstKind,
    first_fp_reg: u16,
    gp_word_mem_width: MachineMemWidth,
    gp_word_int_width: MachineIntWidth,
) -> Result<(), WasmError> {
    match inst {
        MachineInstKind::Move { ty, dst, src } => {
            assign_reg_type(types, *dst, *ty, "move destination")?;
            assign_move_like_value_type(types, *src, *ty, first_fp_reg, "move source")?;
        }
        MachineInstKind::FloatConst { width, dst, .. } => {
            assign_reg_type(
                types,
                *dst,
                float_storage_type(*width),
                "float const destination",
            )?;
        }
        MachineInstKind::Lea { dst, addr } => {
            assign_reg_type(types, *dst, MachineStorageType::GpWord, "lea destination")?;
            assign_reg_type(types, addr.base, MachineStorageType::GpWord, "address base")?;
        }
        MachineInstKind::Load {
            dst, addr, width, ..
        } => {
            assign_reg_type(types, addr.base, MachineStorageType::GpWord, "address base")?;
            assign_reg_type(
                types,
                *dst,
                storage_type_for_load(*width, *dst, first_fp_reg, gp_word_mem_width)?,
                "load destination",
            )?;
        }
        MachineInstKind::Store { addr, width, src } => {
            assign_reg_type(types, addr.base, MachineStorageType::GpWord, "address base")?;
            assign_store_value_type(
                types,
                *src,
                *width,
                first_fp_reg,
                gp_word_mem_width,
                "store source",
            )?;
        }
        MachineInstKind::IntUnary {
            width, dst, src, ..
        } => {
            let operand_ty = storage_type_for_int_width(*width, gp_word_int_width)?;
            assign_reg_type(types, *dst, operand_ty, "integer unary destination")?;
            assign_value_type_if_reg(types, *src, operand_ty, "integer unary source")?;
        }
        MachineInstKind::IntBinary {
            width,
            dst,
            lhs,
            rhs,
            ..
        } => {
            let operand_ty = storage_type_for_int_width(*width, gp_word_int_width)?;
            assign_reg_type(types, *dst, operand_ty, "integer binary destination")?;
            assign_value_type_if_reg(types, *lhs, operand_ty, "integer binary lhs")?;
            assign_value_type_if_reg(types, *rhs, operand_ty, "integer binary rhs")?;
        }
        MachineInstKind::IntCompare {
            width,
            dst,
            lhs,
            rhs,
            ..
        } => {
            let operand_ty = storage_type_for_int_width(*width, gp_word_int_width)?;
            assign_reg_type(
                types,
                *dst,
                MachineStorageType::GpWord,
                "integer compare destination",
            )?;
            assign_value_type_if_reg(types, *lhs, operand_ty, "integer compare lhs")?;
            assign_value_type_if_reg(types, *rhs, operand_ty, "integer compare rhs")?;
        }
        MachineInstKind::FloatUnary {
            width, dst, src, ..
        } => {
            let operand_ty = float_storage_type(*width);
            assign_reg_type(types, *dst, operand_ty, "float unary destination")?;
            assign_value_type_if_reg(types, *src, operand_ty, "float unary source")?;
        }
        MachineInstKind::FloatBinary {
            width,
            dst,
            lhs,
            rhs,
            ..
        } => {
            let operand_ty = float_storage_type(*width);
            assign_reg_type(types, *dst, operand_ty, "float binary destination")?;
            assign_value_type_if_reg(types, *lhs, operand_ty, "float binary lhs")?;
            assign_value_type_if_reg(types, *rhs, operand_ty, "float binary rhs")?;
        }
        MachineInstKind::FloatCompare {
            width,
            dst,
            lhs,
            rhs,
            ..
        } => {
            let operand_ty = float_storage_type(*width);
            assign_reg_type(
                types,
                *dst,
                MachineStorageType::GpWord,
                "float compare destination",
            )?;
            assign_value_type_if_reg(types, *lhs, operand_ty, "float compare lhs")?;
            assign_value_type_if_reg(types, *rhs, operand_ty, "float compare rhs")?;
        }
        MachineInstKind::Convert { op, dst, src } => {
            let (src_ty, dst_ty) = storage_types_for_convert(*op);
            assign_reg_type(types, *dst, dst_ty, "convert destination")?;
            assign_value_type_if_reg(types, *src, src_ty, "convert source")?;
        }
        MachineInstKind::Select {
            ty,
            dst,
            on_true,
            on_false,
            cond,
        } => {
            assign_reg_type(types, *dst, *ty, "select destination")?;
            assign_move_like_value_type(types, *on_true, *ty, first_fp_reg, "select on_true")?;
            assign_move_like_value_type(types, *on_false, *ty, first_fp_reg, "select on_false")?;
            assign_value_type_if_reg(types, *cond, MachineStorageType::GpWord, "select condition")?;
        }
        MachineInstKind::TrapIf { cond, .. } => {
            infer_branch_cond_types(types, *cond, first_fp_reg, gp_word_int_width)?;
        }
        MachineInstKind::CallHelper(_) => {}
    }

    Ok(())
}

fn infer_term_types(
    types: &mut [Option<MachineStorageType>],
    term: &MachineTerminator,
    first_fp_reg: u16,
    gp_word_mem_width: MachineMemWidth,
    gp_word_int_width: MachineIntWidth,
) -> Result<(), WasmError> {
    match term {
        MachineTerminator::Jump(edge) => {
            for arg in &edge.args {
                let _ = arg;
            }
        }
        MachineTerminator::Branch { cond, .. } => {
            infer_branch_cond_types(types, *cond, first_fp_reg, gp_word_int_width)?;
        }
        MachineTerminator::JumpTable { index, .. } => {
            assign_value_type_if_reg(
                types,
                *index,
                MachineStorageType::GpWord,
                "jump table index",
            )?;
        }
        MachineTerminator::CallDirect {
            callee_frame_base, ..
        } => {
            assign_reg_type(
                types,
                *callee_frame_base,
                MachineStorageType::GpWord,
                "call frame base",
            )?;
        }
        MachineTerminator::CallIndirect {
            callee_target,
            callee_frame_base,
            ..
        } => {
            assign_value_type_if_reg(
                types,
                *callee_target,
                storage_type_for_mem_width(gp_word_mem_width, false, gp_word_mem_width)?,
                "indirect callee target",
            )?;
            assign_reg_type(
                types,
                *callee_frame_base,
                MachineStorageType::GpWord,
                "call frame base",
            )?;
        }
        MachineTerminator::Return | MachineTerminator::Trap { .. } => {}
    }

    Ok(())
}

fn infer_branch_cond_types(
    types: &mut [Option<MachineStorageType>],
    cond: MachineBranchCond,
    first_fp_reg: u16,
    gp_word_int_width: MachineIntWidth,
) -> Result<(), WasmError> {
    match cond {
        MachineBranchCond::Value(value) => {
            assign_value_type_if_reg(types, value, MachineStorageType::GpWord, "branch value")?;
        }
        MachineBranchCond::IntCompare {
            width, lhs, rhs, ..
        } => {
            let operand_ty = storage_type_for_int_width(width, gp_word_int_width)?;
            assign_value_type_if_reg(types, lhs, operand_ty, "branch integer lhs")?;
            assign_value_type_if_reg(types, rhs, operand_ty, "branch integer rhs")?;
        }
        MachineBranchCond::FloatCompare {
            width, lhs, rhs, ..
        } => {
            let operand_ty = float_storage_type(width);
            assign_value_type_if_reg(types, lhs, operand_ty, "branch float lhs")?;
            assign_value_type_if_reg(types, rhs, operand_ty, "branch float rhs")?;
        }
    }

    let _ = first_fp_reg;
    Ok(())
}

fn assign_move_like_value_type(
    types: &mut [Option<MachineStorageType>],
    value: MachineValue,
    ty: MachineStorageType,
    first_fp_reg: u16,
    context: &str,
) -> Result<(), WasmError> {
    let MachineValue::Reg(reg) = value else {
        return Ok(());
    };
    assign_reg_type(
        types,
        reg,
        move_like_storage_type_for_reg(ty, reg.0 >= first_fp_reg),
        context,
    )
}

fn assign_store_value_type(
    types: &mut [Option<MachineStorageType>],
    value: MachineValue,
    width: MachineMemWidth,
    first_fp_reg: u16,
    gp_word_mem_width: MachineMemWidth,
    context: &str,
) -> Result<(), WasmError> {
    let MachineValue::Reg(reg) = value else {
        return Ok(());
    };
    assign_reg_type(
        types,
        reg,
        storage_type_for_mem_width(width, reg.0 >= first_fp_reg, gp_word_mem_width)?,
        context,
    )
}

fn assign_value_type_if_reg(
    types: &mut [Option<MachineStorageType>],
    value: MachineValue,
    ty: MachineStorageType,
    context: &str,
) -> Result<(), WasmError> {
    let MachineValue::Reg(reg) = value else {
        return Ok(());
    };
    assign_reg_type(types, reg, ty, context)
}

fn assign_reg_type(
    types: &mut [Option<MachineStorageType>],
    reg: MachineReg,
    ty: MachineStorageType,
    context: &str,
) -> Result<(), WasmError> {
    let slot = types.get_mut(reg.0 as usize).ok_or_else(|| {
        WasmError::internal(alloc::format!(
            "{context} references out-of-range machine register {}",
            reg.0,
        ))
    })?;

    match slot {
        Some(existing) if *existing != ty => Err(WasmError::internal(alloc::format!(
            "machine register {} has conflicting 32-bit storage types {:?} and {:?} ({context})",
            reg.0,
            *existing,
            ty,
        ))),
        Some(_) => Ok(()),
        None => {
            *slot = Some(ty);
            Ok(())
        }
    }
}

fn float_storage_type(width: MachineFloatWidth) -> MachineStorageType {
    match width {
        MachineFloatWidth::F32 => MachineStorageType::Fp32,
        MachineFloatWidth::F64 => MachineStorageType::Fp64,
    }
}

fn move_like_storage_type_for_reg(ty: MachineStorageType, is_fp_reg: bool) -> MachineStorageType {
    match (ty, is_fp_reg) {
        (MachineStorageType::Fp32, true) => MachineStorageType::Fp32,
        (MachineStorageType::Fp64, true) => MachineStorageType::Fp64,
        (MachineStorageType::GpWord, false) => MachineStorageType::GpWord,
        (MachineStorageType::GpI64, false) => MachineStorageType::GpI64,
        (MachineStorageType::Fp32, false) | (MachineStorageType::GpWord, true) => {
            if is_fp_reg {
                MachineStorageType::Fp32
            } else {
                MachineStorageType::GpWord
            }
        }
        (MachineStorageType::Fp64, false) | (MachineStorageType::GpI64, true) => {
            if is_fp_reg {
                MachineStorageType::Fp64
            } else {
                MachineStorageType::GpI64
            }
        }
    }
}

fn storage_type_for_load(
    width: MachineMemWidth,
    dst: MachineReg,
    first_fp_reg: u16,
    gp_word_mem_width: MachineMemWidth,
) -> Result<MachineStorageType, WasmError> {
    storage_type_for_mem_width(width, dst.0 >= first_fp_reg, gp_word_mem_width)
}

fn storage_type_for_mem_width(
    width: MachineMemWidth,
    is_fp_reg: bool,
    gp_word_mem_width: MachineMemWidth,
) -> Result<MachineStorageType, WasmError> {
    if is_fp_reg {
        return match width {
            MachineMemWidth::U32 => Ok(MachineStorageType::Fp32),
            MachineMemWidth::U64 => Ok(MachineStorageType::Fp64),
            other => Err(WasmError::internal(alloc::format!(
                "unsupported FP memory width {:?} during 32-bit legalizer prep",
                other,
            ))),
        };
    }

    Ok(if width == gp_word_mem_width {
        MachineStorageType::GpWord
    } else {
        match width {
            MachineMemWidth::U8 | MachineMemWidth::U16 | MachineMemWidth::U32 => {
                MachineStorageType::GpWord
            }
            MachineMemWidth::U64 => MachineStorageType::GpI64,
        }
    })
}

fn storage_type_for_int_width(
    width: MachineIntWidth,
    gp_word_int_width: MachineIntWidth,
) -> Result<MachineStorageType, WasmError> {
    Ok(
        if width == gp_word_int_width || matches!(width, MachineIntWidth::I32) {
            MachineStorageType::GpWord
        } else {
            match width {
                MachineIntWidth::I64 => MachineStorageType::GpI64,
                MachineIntWidth::I32 => MachineStorageType::GpWord,
            }
        },
    )
}

fn storage_types_for_convert(op: MachineConvertOp) -> (MachineStorageType, MachineStorageType) {
    match op {
        MachineConvertOp::I32WrapI64 => (MachineStorageType::GpI64, MachineStorageType::GpWord),
        MachineConvertOp::I64ExtendI32S | MachineConvertOp::I64ExtendI32U => {
            (MachineStorageType::GpWord, MachineStorageType::GpI64)
        }
        MachineConvertOp::I32TruncF32S
        | MachineConvertOp::I32TruncF32U
        | MachineConvertOp::I32TruncSatF32S
        | MachineConvertOp::I32TruncSatF32U
        | MachineConvertOp::I32ReinterpretF32 => {
            (MachineStorageType::Fp32, MachineStorageType::GpWord)
        }
        MachineConvertOp::I32TruncF64S
        | MachineConvertOp::I32TruncF64U
        | MachineConvertOp::I32TruncSatF64S
        | MachineConvertOp::I32TruncSatF64U => {
            (MachineStorageType::Fp64, MachineStorageType::GpWord)
        }
        MachineConvertOp::I64TruncF32S
        | MachineConvertOp::I64TruncF32U
        | MachineConvertOp::I64TruncSatF32S
        | MachineConvertOp::I64TruncSatF32U => {
            (MachineStorageType::Fp32, MachineStorageType::GpI64)
        }
        MachineConvertOp::I64TruncF64S
        | MachineConvertOp::I64TruncF64U
        | MachineConvertOp::I64TruncSatF64S
        | MachineConvertOp::I64TruncSatF64U
        | MachineConvertOp::I64ReinterpretF64 => {
            (MachineStorageType::Fp64, MachineStorageType::GpI64)
        }
        MachineConvertOp::F32ConvertI32S
        | MachineConvertOp::F32ConvertI32U
        | MachineConvertOp::F32ReinterpretI32 => {
            (MachineStorageType::GpWord, MachineStorageType::Fp32)
        }
        MachineConvertOp::F32ConvertI64S | MachineConvertOp::F32ConvertI64U => {
            (MachineStorageType::GpI64, MachineStorageType::Fp32)
        }
        MachineConvertOp::F64ConvertI32S | MachineConvertOp::F64ConvertI32U => {
            (MachineStorageType::GpWord, MachineStorageType::Fp64)
        }
        MachineConvertOp::F64ConvertI64S
        | MachineConvertOp::F64ConvertI64U
        | MachineConvertOp::F64ReinterpretI64 => {
            (MachineStorageType::GpI64, MachineStorageType::Fp64)
        }
        MachineConvertOp::F32DemoteF64 => (MachineStorageType::Fp64, MachineStorageType::Fp32),
        MachineConvertOp::F64PromoteF32 => (MachineStorageType::Fp32, MachineStorageType::Fp64),
    }
}
