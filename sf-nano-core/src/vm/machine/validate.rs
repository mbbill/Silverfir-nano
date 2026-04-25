use crate::error::WasmError;
use crate::vm::backend::BackendConfig;

#[cfg(any(debug_assertions, test))]
use super::machine_ir::is_fp_reg;
#[cfg(any(debug_assertions, test))]
use super::machine_ir::{
    is_dynamic_reg, MachineAddr, MachineCallTarget, MachineConstId, MachineEdge, MachineFloatWidth,
    MachineFuncId, MachineInst, MachineReg, MachineRegOwner, MachineValue,
};
use super::machine_ir::{
    MachineBlockId, MachineBlockParam, MachineBranchCond, MachineConvertOp, MachineInstKind,
    MachineIntWidth, MachineModule, MachineProgram, MachineStorageType, MachineTerminator,
};
#[cfg(any(debug_assertions, test))]
use super::machine_ir::{MachineIntBinaryOp, MachineIntUnaryOp};

type ValidateResult = Result<(), WasmError>;

impl MachineProgram {
    pub(crate) fn validate_32bit_gp_target(
        &self,
        max_gp_regs: u16,
        config: BackendConfig,
    ) -> Result<(), WasmError> {
        let first_fp = config.first_fp_reg();
        let total = config.total_reg_count();
        if first_fp != max_gp_regs {
            return Err(WasmError::internal(
                "32-bit GP target MachineIR has mismatched first_fp_reg boundary",
            ));
        }
        if total < first_fp {
            return Err(WasmError::internal(
                "machine reg_count is below 32-bit GP-target fp boundary",
            ));
        }

        for block in &self.blocks {
            for (param_index, param) in block.params.iter().enumerate() {
                validate_32bit_gp_target_param(block.id, param_index, *param)?;
            }
            for (inst_index, inst) in block.ops.iter().enumerate() {
                validate_32bit_gp_target_inst(block.id, inst_index, &inst.kind)?;
            }
            validate_32bit_gp_target_term(block.id, &block.terminator)?;
        }

        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    pub(crate) fn validate(&self, config: BackendConfig) -> Result<(), WasmError> {
        if self.blocks.is_empty() {
            if self.entry.as_usize() != 0 {
                return Err(WasmError::internal(
                    "empty machine program must use entry block 0".into(),
                ));
            }
            return Ok(());
        }

        if self.entry.as_usize() >= self.blocks.len() {
            return Err(WasmError::internal(
                "machine entry block is out of range for blocks",
            ));
        }

        let reg_count = config.total_reg_count();
        let first_fp = config.first_fp_reg();
        let fp_bank_count = reg_count.saturating_sub(first_fp) as usize;
        if !self.fp_reg_init_widths.is_empty() && self.fp_reg_init_widths.len() != fp_bank_count {
            return Err(WasmError::internal(
                "machine fp_reg_init_widths length does not match fp bank size",
            ));
        }

        for (index, block) in self.blocks.iter().enumerate() {
            if block.id.as_usize() != index {
                return Err(WasmError::internal("machine block has mismatched id"));
            }
            for param in &block.params {
                self.validate_param(*param, config)?;
            }
            for inst in &block.ops {
                self.validate_inst(inst, config)?;
            }
            self.validate_term(&block.terminator, index, config)?;
        }

        Ok(())
    }

    #[cfg(not(any(debug_assertions, test)))]
    #[inline]
    pub(crate) fn validate(&self, _config: BackendConfig) -> Result<(), WasmError> {
        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_param(&self, param: MachineBlockParam, config: BackendConfig) -> ValidateResult {
        self.validate_reg(param.reg, config)?;
        if is_fp_reg(param.reg, config) != param.ty.is_fp() {
            return Err(WasmError::internal(
                "machine block param has mismatched storage type for its register bank",
            ));
        }
        if matches!(param.owner, MachineRegOwner::CachedLocal) && !is_dynamic_reg(param.reg, config)
        {
            return Err(WasmError::internal(
                "cached-local block param must use a dynamic register",
            ));
        }
        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_inst(&self, inst: &MachineInst, config: BackendConfig) -> ValidateResult {
        match &inst.kind {
            MachineInstKind::Move { ty, dst, src, .. } => {
                self.validate_reg(*dst, config)?;
                self.validate_value(*src, config)?;
                self.validate_reg_storage_type(*dst, *ty, config)?;
                if matches!(
                    &inst.kind,
                    MachineInstKind::Move {
                        owner: MachineRegOwner::CachedLocal,
                        ..
                    }
                ) && !is_dynamic_reg(*dst, config)
                {
                    return Err(WasmError::internal(
                        "cached-local move destination must use a dynamic register",
                    ));
                }
            }
            MachineInstKind::FloatConst { dst, .. } => {
                self.validate_reg(*dst, config)?;
                if !is_fp_reg(*dst, config) {
                    return Err(WasmError::internal(
                        "machine FloatConst destination must be an FP register",
                    ));
                }
            }
            #[cfg(sf_has_simd)]
            MachineInstKind::V128Const { dst, .. } => {
                self.validate_reg(*dst, config)?;
                self.validate_reg_storage_type(*dst, MachineStorageType::V128, config)?;
            }
            #[cfg(sf_has_simd)]
            MachineInstKind::V128FromRaw { dst, raw, .. } => {
                self.validate_reg(*dst, config)?;
                self.validate_reg_storage_type(*dst, MachineStorageType::V128, config)?;
                self.validate_typed_value(*raw, MachineStorageType::GpWord, config)?;
            }
            #[cfg(sf_has_simd)]
            MachineInstKind::V128ToRaw { dst, src } => {
                self.validate_reg(*dst, config)?;
                self.validate_reg_storage_type(*dst, MachineStorageType::GpWord, config)?;
                self.validate_typed_value(*src, MachineStorageType::V128, config)?;
            }
            MachineInstKind::Load { ty, dst, addr, .. } => {
                self.validate_reg(*dst, config)?;
                self.validate_addr(*addr, config)?;
                self.validate_reg_storage_type(*dst, *ty, config)?;
                if matches!(
                    &inst.kind,
                    MachineInstKind::Load {
                        owner: MachineRegOwner::CachedLocal,
                        ..
                    }
                ) && !is_dynamic_reg(*dst, config)
                {
                    return Err(WasmError::internal(
                        "cached-local load destination must use a dynamic register",
                    ));
                }
            }
            MachineInstKind::Store { ty, addr, src, .. } => {
                self.validate_addr(*addr, config)?;
                self.validate_value(*src, config)?;
                if let MachineValue::Reg(src_reg) = src {
                    self.validate_reg_storage_type(*src_reg, *ty, config)?;
                }
            }
            #[cfg(sf_has_simd)]
            MachineInstKind::SimdLoad { dst, addr, .. } => {
                self.validate_reg(*dst, config)?;
                self.validate_reg_storage_type(*dst, MachineStorageType::V128, config)?;
                self.validate_addr(*addr, config)?;
            }
            #[cfg(sf_has_simd)]
            MachineInstKind::SimdStore { addr, src, .. } => {
                self.validate_addr(*addr, config)?;
                self.validate_value(*src, config)?;
            }
            #[cfg(sf_has_simd)]
            MachineInstKind::SimdLoadLane {
                dst, addr, vector, ..
            } => {
                self.validate_reg(*dst, config)?;
                self.validate_reg_storage_type(*dst, MachineStorageType::V128, config)?;
                self.validate_addr(*addr, config)?;
                self.validate_value(*vector, config)?;
            }
            #[cfg(sf_has_simd)]
            MachineInstKind::SimdStoreLane { addr, vector, .. } => {
                self.validate_addr(*addr, config)?;
                self.validate_value(*vector, config)?;
            }
            MachineInstKind::IntUnary { dst, src, .. }
            | MachineInstKind::FloatUnary { dst, src, .. }
            | MachineInstKind::Convert { dst, src, .. } => {
                self.validate_reg(*dst, config)?;
                self.validate_value(*src, config)?;
            }
            #[cfg(sf_has_simd)]
            MachineInstKind::SimdUnary {
                dst_ty, dst, src, ..
            }
            | MachineInstKind::SimdExtractLane {
                dst_ty, dst, src, ..
            } => {
                self.validate_reg(*dst, config)?;
                self.validate_reg_storage_type(*dst, *dst_ty, config)?;
                self.validate_value(*src, config)?;
            }
            MachineInstKind::IntBinary { dst, lhs, rhs, .. }
            | MachineInstKind::IntCompare { dst, lhs, rhs, .. }
            | MachineInstKind::FloatBinary { dst, lhs, rhs, .. } => {
                self.validate_reg(*dst, config)?;
                self.validate_value(*lhs, config)?;
                self.validate_value(*rhs, config)?;
            }
            #[cfg(sf_has_simd)]
            MachineInstKind::SimdBinary {
                dst_ty,
                dst,
                lhs,
                rhs,
                ..
            } => {
                self.validate_reg(*dst, config)?;
                self.validate_reg_storage_type(*dst, *dst_ty, config)?;
                self.validate_value(*lhs, config)?;
                self.validate_value(*rhs, config)?;
            }
            #[cfg(sf_has_simd)]
            MachineInstKind::SimdTernary {
                dst_ty,
                dst,
                a,
                b,
                c,
                ..
            } => {
                self.validate_reg(*dst, config)?;
                self.validate_reg_storage_type(*dst, *dst_ty, config)?;
                self.validate_value(*a, config)?;
                self.validate_value(*b, config)?;
                self.validate_value(*c, config)?;
            }
            #[cfg(sf_has_simd)]
            MachineInstKind::SimdShift {
                dst, vector, shift, ..
            } => {
                self.validate_reg(*dst, config)?;
                self.validate_reg_storage_type(*dst, MachineStorageType::V128, config)?;
                self.validate_value(*vector, config)?;
                self.validate_value(*shift, config)?;
            }
            #[cfg(sf_has_simd)]
            MachineInstKind::SimdReplaceLane {
                dst,
                vector,
                scalar,
                ..
            } => {
                self.validate_reg(*dst, config)?;
                self.validate_reg_storage_type(*dst, MachineStorageType::V128, config)?;
                self.validate_value(*vector, config)?;
                self.validate_value(*scalar, config)?;
            }
            #[cfg(sf_has_simd)]
            MachineInstKind::SimdShuffle { dst, lhs, rhs, .. } => {
                self.validate_reg(*dst, config)?;
                self.validate_reg_storage_type(*dst, MachineStorageType::V128, config)?;
                self.validate_value(*lhs, config)?;
                self.validate_value(*rhs, config)?;
            }
            MachineInstKind::BitfieldExtractU { dst, src, .. } => {
                self.validate_reg(*dst, config)?;
                self.validate_value(MachineValue::Reg(*src), config)?;
            }
            MachineInstKind::IntBinaryShifted { dst, lhs, rhs, .. } => {
                self.validate_reg(*dst, config)?;
                self.validate_value(MachineValue::Reg(*lhs), config)?;
                self.validate_value(MachineValue::Reg(*rhs), config)?;
            }
            MachineInstKind::TestBits { dst, src, mask, .. } => {
                self.validate_reg(*dst, config)?;
                self.validate_value(MachineValue::Reg(*src), config)?;
                self.validate_value(*mask, config)?;
            }
            MachineInstKind::Int64PairBinary {
                op,
                dst_lo,
                dst_hi,
                lhs_lo,
                lhs_hi,
                rhs_lo,
                rhs_hi,
            } => {
                self.validate_reg(*dst_lo, config)?;
                self.validate_reg(*dst_hi, config)?;
                self.validate_value(*lhs_lo, config)?;
                self.validate_value(*lhs_hi, config)?;
                self.validate_value(*rhs_lo, config)?;
                self.validate_value(*rhs_hi, config)?;
                self.validate_reg_storage_type(*dst_lo, MachineStorageType::GpWord, config)?;
                self.validate_reg_storage_type(*dst_hi, MachineStorageType::GpWord, config)?;
                if !matches!(
                    op,
                    MachineIntBinaryOp::Add
                        | MachineIntBinaryOp::Sub
                        | MachineIntBinaryOp::Mul
                        | MachineIntBinaryOp::And
                        | MachineIntBinaryOp::Or
                        | MachineIntBinaryOp::Xor
                ) {
                    return Err(WasmError::internal(
                        "machine Int64PairBinary requires a supported i64 binary op".into(),
                    ));
                }
                if dst_lo == dst_hi {
                    return Err(WasmError::internal(
                        "machine Int64PairBinary requires distinct low/high destinations".into(),
                    ));
                }
            }
            MachineInstKind::Int64MulFromSignExt32 {
                dst_lo,
                dst_hi,
                lhs,
                rhs,
            } => {
                self.validate_reg(*dst_lo, config)?;
                self.validate_reg(*dst_hi, config)?;
                self.validate_value(*lhs, config)?;
                self.validate_value(*rhs, config)?;
                self.validate_reg_storage_type(*dst_lo, MachineStorageType::GpWord, config)?;
                self.validate_reg_storage_type(*dst_hi, MachineStorageType::GpWord, config)?;
                if dst_lo == dst_hi {
                    return Err(WasmError::internal(
                        "machine Int64MulFromSignExt32 requires distinct low/high destinations"
                            .into(),
                    ));
                }
            }
            MachineInstKind::Int64PairDivRem {
                dst_lo,
                dst_hi,
                lhs_lo,
                lhs_hi,
                rhs_lo,
                rhs_hi,
                ..
            } => {
                self.validate_reg(*dst_lo, config)?;
                self.validate_reg(*dst_hi, config)?;
                self.validate_value(*lhs_lo, config)?;
                self.validate_value(*lhs_hi, config)?;
                self.validate_value(*rhs_lo, config)?;
                self.validate_value(*rhs_hi, config)?;
                self.validate_reg_storage_type(*dst_lo, MachineStorageType::GpWord, config)?;
                self.validate_reg_storage_type(*dst_hi, MachineStorageType::GpWord, config)?;
                if dst_lo == dst_hi {
                    return Err(WasmError::internal(
                        "machine Int64PairDivRem requires distinct low/high destinations".into(),
                    ));
                }
            }
            MachineInstKind::Int64PairUnary {
                op,
                dst_lo,
                dst_hi,
                src_lo,
                src_hi,
            } => {
                self.validate_reg(*dst_lo, config)?;
                self.validate_reg(*dst_hi, config)?;
                self.validate_value(*src_lo, config)?;
                self.validate_value(*src_hi, config)?;
                self.validate_reg_storage_type(*dst_lo, MachineStorageType::GpWord, config)?;
                self.validate_reg_storage_type(*dst_hi, MachineStorageType::GpWord, config)?;
                if !matches!(
                    op,
                    MachineIntUnaryOp::Clz
                        | MachineIntUnaryOp::Ctz
                        | MachineIntUnaryOp::Popcnt
                        | MachineIntUnaryOp::Extend8S
                        | MachineIntUnaryOp::Extend16S
                        | MachineIntUnaryOp::Extend32S
                ) {
                    return Err(WasmError::internal(
                        "machine Int64PairUnary requires a supported i64 unary op".into(),
                    ));
                }
                if dst_lo == dst_hi {
                    return Err(WasmError::internal(
                        "machine Int64PairUnary requires distinct low/high destinations".into(),
                    ));
                }
            }
            MachineInstKind::Int64PairShift {
                op,
                dst_lo,
                dst_hi,
                lhs_lo,
                lhs_hi,
                rhs,
            } => {
                self.validate_reg(*dst_lo, config)?;
                self.validate_reg(*dst_hi, config)?;
                self.validate_value(*lhs_lo, config)?;
                self.validate_value(*lhs_hi, config)?;
                self.validate_value(*rhs, config)?;
                self.validate_reg_storage_type(*dst_lo, MachineStorageType::GpWord, config)?;
                self.validate_reg_storage_type(*dst_hi, MachineStorageType::GpWord, config)?;
                if !matches!(
                    op,
                    MachineIntBinaryOp::Shl
                        | MachineIntBinaryOp::ShrS
                        | MachineIntBinaryOp::ShrU
                        | MachineIntBinaryOp::Rotl
                        | MachineIntBinaryOp::Rotr
                ) {
                    return Err(WasmError::internal(
                        "machine Int64PairShift requires a shift/rotate op".into(),
                    ));
                }
                if dst_lo == dst_hi {
                    return Err(WasmError::internal(
                        "machine Int64PairShift requires distinct low/high destinations".into(),
                    ));
                }
            }
            MachineInstKind::FloatCompare { dst, lhs, rhs, .. } => {
                self.validate_reg(*dst, config)?;
                self.validate_value(*lhs, config)?;
                self.validate_value(*rhs, config)?;
            }
            MachineInstKind::Int64PairCompare {
                dst,
                lhs_lo,
                lhs_hi,
                rhs_lo,
                rhs_hi,
                ..
            } => {
                self.validate_reg(*dst, config)?;
                self.validate_value(*lhs_lo, config)?;
                self.validate_value(*lhs_hi, config)?;
                self.validate_value(*rhs_lo, config)?;
                self.validate_value(*rhs_hi, config)?;
                self.validate_reg_storage_type(*dst, MachineStorageType::GpWord, config)?;
            }
            MachineInstKind::ConvertI64PairToFloat {
                width,
                dst,
                src_lo,
                src_hi,
                ..
            } => {
                self.validate_reg(*dst, config)?;
                self.validate_value(*src_lo, config)?;
                self.validate_value(*src_hi, config)?;
                self.validate_reg_storage_type(
                    *dst,
                    match width {
                        MachineFloatWidth::F32 => MachineStorageType::Fp32,
                        MachineFloatWidth::F64 => MachineStorageType::Fp64,
                    },
                    config,
                )?;
            }
            MachineInstKind::ConvertFloatToI64Pair {
                op,
                dst_lo,
                dst_hi,
                src,
            } => {
                self.validate_reg(*dst_lo, config)?;
                self.validate_reg(*dst_hi, config)?;
                self.validate_value(*src, config)?;
                self.validate_reg_storage_type(*dst_lo, MachineStorageType::GpWord, config)?;
                self.validate_reg_storage_type(*dst_hi, MachineStorageType::GpWord, config)?;
                if !matches!(
                    op,
                    MachineConvertOp::I64TruncF32S
                        | MachineConvertOp::I64TruncF32U
                        | MachineConvertOp::I64TruncF64S
                        | MachineConvertOp::I64TruncF64U
                        | MachineConvertOp::I64TruncSatF32S
                        | MachineConvertOp::I64TruncSatF32U
                        | MachineConvertOp::I64TruncSatF64S
                        | MachineConvertOp::I64TruncSatF64U
                ) {
                    return Err(WasmError::internal(
                        "machine ConvertFloatToI64Pair requires an i64 trunc/trunc_sat op".into(),
                    ));
                }
                if let MachineValue::Reg(src_reg) = src {
                    self.validate_reg_storage_type(
                        *src_reg,
                        match op {
                            MachineConvertOp::I64TruncF32S
                            | MachineConvertOp::I64TruncF32U
                            | MachineConvertOp::I64TruncSatF32S
                            | MachineConvertOp::I64TruncSatF32U => MachineStorageType::Fp32,
                            _ => MachineStorageType::Fp64,
                        },
                        config,
                    )?;
                }
                if dst_lo == dst_hi {
                    return Err(WasmError::internal(
                        "machine ConvertFloatToI64Pair requires distinct low/high destinations"
                            .into(),
                    ));
                }
            }
            MachineInstKind::ReinterpretF64ToI64Pair {
                dst_lo,
                dst_hi,
                src,
            } => {
                self.validate_reg(*dst_lo, config)?;
                self.validate_reg(*dst_hi, config)?;
                self.validate_value(*src, config)?;
                self.validate_reg_storage_type(*dst_lo, MachineStorageType::GpWord, config)?;
                self.validate_reg_storage_type(*dst_hi, MachineStorageType::GpWord, config)?;
                if let MachineValue::Reg(src_reg) = src {
                    self.validate_reg_storage_type(*src_reg, MachineStorageType::Fp64, config)?;
                }
                if dst_lo == dst_hi {
                    return Err(WasmError::internal(
                        "machine ReinterpretF64ToI64Pair requires distinct low/high destinations"
                            .into(),
                    ));
                }
            }
            MachineInstKind::ReinterpretI64PairToF64 {
                dst,
                src_lo,
                src_hi,
            } => {
                self.validate_reg(*dst, config)?;
                self.validate_value(*src_lo, config)?;
                self.validate_value(*src_hi, config)?;
                self.validate_reg_storage_type(*dst, MachineStorageType::Fp64, config)?;
                if let MachineValue::Reg(src_lo_reg) = src_lo {
                    self.validate_reg_storage_type(
                        *src_lo_reg,
                        MachineStorageType::GpWord,
                        config,
                    )?;
                }
                if let MachineValue::Reg(src_hi_reg) = src_hi {
                    self.validate_reg_storage_type(
                        *src_hi_reg,
                        MachineStorageType::GpWord,
                        config,
                    )?;
                }
            }
            MachineInstKind::Select {
                ty,
                dst,
                on_true,
                on_false,
                cond,
            } => {
                self.validate_reg(*dst, config)?;
                self.validate_value(*on_true, config)?;
                self.validate_value(*on_false, config)?;
                self.validate_value(*cond, config)?;
                self.validate_reg_storage_type(*dst, *ty, config)?;
            }
            MachineInstKind::IndexedLoad {
                dst, base, index, ..
            } => {
                self.validate_reg(*dst, config)?;
                self.validate_reg(*base, config)?;
                self.validate_reg(*index, config)?;
            }
            MachineInstKind::IndexedStore {
                base, index, src, ..
            } => {
                self.validate_reg(*base, config)?;
                self.validate_reg(*index, config)?;
                self.validate_value(*src, config)?;
            }
            MachineInstKind::TrapIf { cond, .. } => {
                self.validate_branch_cond(*cond, config)?;
            }
            MachineInstKind::CallRuntime(_) => {}
            MachineInstKind::RefFunc { dst, .. }
            | MachineInstKind::StructNewDefault { dst, .. }
            | MachineInstKind::ArrayNewDefault { dst, .. } => {
                self.validate_reg(*dst, config)?;
                self.validate_reg_storage_type(*dst, MachineStorageType::GpWord, config)?;
            }
            MachineInstKind::RefAsNonNull { src, dst }
            | MachineInstKind::RefI31 { src, dst }
            | MachineInstKind::I31GetS { src, dst }
            | MachineInstKind::I31GetU { src, dst }
            | MachineInstKind::AnyConvertExtern { src, dst }
            | MachineInstKind::ExternConvertAny { src, dst }
            | MachineInstKind::RefTest { src, dst, .. }
            | MachineInstKind::RefCast { src, dst, .. } => {
                self.validate_reg(*dst, config)?;
                self.validate_value(*src, config)?;
                self.validate_reg_storage_type(*dst, MachineStorageType::GpWord, config)?;
            }
            MachineInstKind::RefEq { lhs, rhs, dst } => {
                self.validate_reg(*dst, config)?;
                self.validate_value(*lhs, config)?;
                self.validate_value(*rhs, config)?;
                self.validate_reg_storage_type(*dst, MachineStorageType::GpWord, config)?;
            }
            MachineInstKind::StructGet {
                src, dst, dst_hi, ..
            } => {
                self.validate_value(*src, config)?;
                self.validate_reg(*dst, config)?;
                if let Some(dst_hi) = dst_hi {
                    self.validate_reg(*dst_hi, config)?;
                }
            }
            MachineInstKind::ArrayGet {
                ref_src,
                index,
                dst,
                dst_hi,
                ..
            } => {
                self.validate_value(*ref_src, config)?;
                self.validate_value(*index, config)?;
                self.validate_reg(*dst, config)?;
                if let Some(dst_hi) = dst_hi {
                    self.validate_reg(*dst_hi, config)?;
                }
            }
            MachineInstKind::ArraySet {
                ref_src,
                index,
                value_lo,
                value_hi,
                ..
            } => {
                self.validate_value(*ref_src, config)?;
                self.validate_value(*index, config)?;
                self.validate_value(*value_lo, config)?;
                if let Some(value_hi) = value_hi {
                    self.validate_value(*value_hi, config)?;
                }
            }
            MachineInstKind::StructNew { dst, fields, .. } => {
                self.validate_reg(*dst, config)?;
                self.validate_reg_storage_type(*dst, MachineStorageType::GpWord, config)?;
                for (value_lo, value_hi) in fields {
                    self.validate_value(*value_lo, config)?;
                    if let Some(value_hi) = value_hi {
                        self.validate_value(*value_hi, config)?;
                    }
                }
            }
            MachineInstKind::StructSet {
                ref_src,
                value_lo,
                value_hi,
                ..
            } => {
                self.validate_value(*ref_src, config)?;
                self.validate_value(*value_lo, config)?;
                if let Some(value_hi) = value_hi {
                    self.validate_value(*value_hi, config)?;
                }
            }
            MachineInstKind::ArrayNew {
                dst,
                init_lo,
                init_hi,
                length,
                ..
            } => {
                self.validate_reg(*dst, config)?;
                self.validate_reg_storage_type(*dst, MachineStorageType::GpWord, config)?;
                self.validate_value(*init_lo, config)?;
                if let Some(init_hi) = init_hi {
                    self.validate_value(*init_hi, config)?;
                }
                self.validate_value(*length, config)?;
            }
            MachineInstKind::ArrayNewFixed { dst, elements, .. } => {
                self.validate_reg(*dst, config)?;
                self.validate_reg_storage_type(*dst, MachineStorageType::GpWord, config)?;
                for (value_lo, value_hi) in elements {
                    self.validate_value(*value_lo, config)?;
                    if let Some(value_hi) = value_hi {
                        self.validate_value(*value_hi, config)?;
                    }
                }
            }
            MachineInstKind::ArrayNewData { dst, src, len, .. }
            | MachineInstKind::ArrayNewElem { dst, src, len, .. } => {
                self.validate_reg(*dst, config)?;
                self.validate_reg_storage_type(*dst, MachineStorageType::GpWord, config)?;
                self.validate_value(*src, config)?;
                self.validate_value(*len, config)?;
            }
            MachineInstKind::ArrayLen { src, dst } => {
                self.validate_reg(*dst, config)?;
                self.validate_reg_storage_type(*dst, MachineStorageType::GpWord, config)?;
                self.validate_value(*src, config)?;
            }
            MachineInstKind::ArrayFill {
                ref_src,
                index,
                value_lo,
                value_hi,
                len,
                ..
            } => {
                self.validate_value(*ref_src, config)?;
                self.validate_value(*index, config)?;
                self.validate_value(*value_lo, config)?;
                if let Some(value_hi) = value_hi {
                    self.validate_value(*value_hi, config)?;
                }
                self.validate_value(*len, config)?;
            }
            MachineInstKind::ArrayCopy {
                dst_ref,
                dst_index,
                src_ref,
                src_index,
                len,
                ..
            } => {
                self.validate_value(*dst_ref, config)?;
                self.validate_value(*dst_index, config)?;
                self.validate_value(*src_ref, config)?;
                self.validate_value(*src_index, config)?;
                self.validate_value(*len, config)?;
            }
            MachineInstKind::ArrayInitData {
                ref_src,
                dst_index,
                src_index,
                len,
                ..
            }
            | MachineInstKind::ArrayInitElem {
                ref_src,
                dst_index,
                src_index,
                len,
                ..
            } => {
                self.validate_value(*ref_src, config)?;
                self.validate_value(*dst_index, config)?;
                self.validate_value(*src_index, config)?;
                self.validate_value(*len, config)?;
            }
            MachineInstKind::MemoryGrow { dst, delta, .. } => {
                self.validate_reg(*dst, config)?;
                self.validate_value(*delta, config)?;
            }
            MachineInstKind::EhAllocExnRef { dst, .. } => {
                self.validate_reg(*dst, config)?;
                self.validate_reg_storage_type(*dst, MachineStorageType::GpWord, config)?;
            }
            MachineInstKind::EhThrow { .. } | MachineInstKind::EhThrowRef { .. } => {}
            MachineInstKind::MemoryFill { dest, val, len, .. }
            | MachineInstKind::TableFill {
                start: dest,
                val,
                len,
                ..
            } => {
                self.validate_value(*dest, config)?;
                self.validate_value(*val, config)?;
                self.validate_value(*len, config)?;
            }
            MachineInstKind::MemoryCopy { dest, src, len, .. }
            | MachineInstKind::MemoryInit { dest, src, len, .. }
            | MachineInstKind::TableCopy { dest, src, len, .. }
            | MachineInstKind::TableInit { dest, src, len, .. } => {
                self.validate_value(*dest, config)?;
                self.validate_value(*src, config)?;
                self.validate_value(*len, config)?;
            }
            MachineInstKind::TableGrow {
                dst,
                init_val,
                delta,
                ..
            } => {
                self.validate_reg(*dst, config)?;
                self.validate_value(*init_val, config)?;
                self.validate_value(*delta, config)?;
            }
            MachineInstKind::DataDrop { .. } | MachineInstKind::ElemDrop { .. } => {}
        }
        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_term(
        &self,
        term: &MachineTerminator,
        source_block: usize,
        config: BackendConfig,
    ) -> ValidateResult {
        match term {
            MachineTerminator::Jump(edge) => self.validate_edge(edge, source_block, config),
            MachineTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            } => {
                self.validate_branch_cond(*cond, config)?;
                self.validate_edge(then_edge, source_block, config)?;
                self.validate_edge(else_edge, source_block, config)
            }
            MachineTerminator::JumpTable { index, entries } => {
                self.validate_value(*index, config)?;
                for edge in entries {
                    self.validate_edge(edge, source_block, config)?;
                }
                Ok(())
            }
            MachineTerminator::Call {
                target,
                callee_frame_base,
                caller_result_base,
                continuation,
                ..
            } => {
                if let MachineCallTarget::Indirect {
                    callee_target,
                    callee_entry,
                } = target
                {
                    self.validate_reg(*callee_target, config)?;
                    self.validate_reg(*callee_entry, config)?;
                }
                self.validate_reg(*callee_frame_base, config)?;
                self.validate_reg(*caller_result_base, config)?;
                self.validate_block_id(*continuation, source_block, "continuation")
            }
            MachineTerminator::TailCall {
                target,
                callee_frame_base,
            } => {
                if let MachineCallTarget::Indirect {
                    callee_target,
                    callee_entry,
                } = target
                {
                    self.validate_reg(*callee_target, config)?;
                    self.validate_reg(*callee_entry, config)?;
                }
                self.validate_reg(*callee_frame_base, config)
            }
            MachineTerminator::Return => Ok(()),
            MachineTerminator::Trap { .. } => Ok(()),
        }
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_branch_cond(
        &self,
        cond: MachineBranchCond,
        config: BackendConfig,
    ) -> ValidateResult {
        match cond {
            MachineBranchCond::Value(value) => self.validate_value(value, config),
            MachineBranchCond::IntCompare { lhs, rhs, .. } => {
                self.validate_value(lhs, config)?;
                self.validate_value(rhs, config)
            }
            MachineBranchCond::TestBits { src, mask, .. } => {
                self.validate_value(src, config)?;
                self.validate_value(mask, config)
            }
        }
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_edge(
        &self,
        edge: &MachineEdge,
        source_block: usize,
        config: BackendConfig,
    ) -> ValidateResult {
        self.validate_block_id(edge.target, source_block, "edge target")?;
        let target = &self.blocks[edge.target.as_usize()];
        if edge.args.len() != target.params.len() {
            return Err(WasmError::internal(
                "machine edge supplies the wrong number of args",
            ));
        }
        for value in &edge.args {
            self.validate_value(*value, config)?;
        }
        for (param, arg) in self.blocks[edge.target.as_usize()]
            .params
            .iter()
            .zip(edge.args.iter())
        {
            if let MachineValue::ReservedReg(reg) = arg {
                if !matches!(param.owner, MachineRegOwner::CachedLocal) {
                    return Err(WasmError::internal(
                        "machine block -> uses ReservedReg for non-cached-local param",
                    ));
                }
                if *reg != param.reg {
                    return Err(WasmError::internal(
                        "machine block -> must reserve cached-local param in-place, got",
                    ));
                }
            }
        }
        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_block_id(
        &self,
        block: MachineBlockId,
        _source_block: usize,
        _role: &str,
    ) -> ValidateResult {
        if block.as_usize() >= self.blocks.len() {
            return Err(WasmError::internal("machine block has out-of-range"));
        }
        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_value(&self, value: MachineValue, config: BackendConfig) -> ValidateResult {
        match value {
            MachineValue::Reg(reg) => self.validate_reg(reg, config),
            MachineValue::ReservedReg(reg) => self.validate_reg(reg, config),
            MachineValue::Imm64(_) => Ok(()),
        }
    }

    #[cfg(all(any(debug_assertions, test), sf_has_simd))]
    fn validate_typed_value(
        &self,
        value: MachineValue,
        ty: MachineStorageType,
        config: BackendConfig,
    ) -> ValidateResult {
        match value {
            MachineValue::Reg(reg) | MachineValue::ReservedReg(reg) => {
                self.validate_reg(reg, config)?;
                self.validate_reg_storage_type(reg, ty, config)
            }
            MachineValue::Imm64(_) => Ok(()),
        }
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_addr(&self, addr: MachineAddr, config: BackendConfig) -> ValidateResult {
        self.validate_reg(addr.base, config)
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_reg(&self, reg: MachineReg, config: BackendConfig) -> ValidateResult {
        let reg_count = config.total_reg_count();
        let first_fp = config.first_fp_reg();
        if reg.0 >= reg_count {
            return Err(WasmError::internal(
                "machine register exceeds declared register count",
            ));
        }
        if first_fp > reg_count {
            return Err(WasmError::internal(
                "machine first_fp_reg exceeds declared register count",
            ));
        }
        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_reg_storage_type(
        &self,
        reg: MachineReg,
        ty: MachineStorageType,
        config: BackendConfig,
    ) -> ValidateResult {
        if is_fp_reg(reg, config) != ty.is_fp() {
            return Err(WasmError::internal(
                "machine register has storage type in the wrong bank",
            ));
        }
        Ok(())
    }
}

impl MachineModule {
    pub(crate) fn validate_32bit_gp_target(&self, max_gp_regs: u16) -> Result<(), WasmError> {
        for func in &self.functions {
            func.program
                .validate_32bit_gp_target(max_gp_regs, self.config)
                .map_err(|err| {
                    WasmError::internal(match err.message() {
                        "32-bit GP target MachineIR has mismatched first_fp_reg boundary" => {
                            "machine function is not valid 32-bit GP-target MachineIR: mismatched first_fp_reg boundary"
                        }
                        "block param still uses GpI64 on a 32-bit GP target" => {
                            "machine function is not valid 32-bit GP-target MachineIR: block param still uses GpI64 on a 32-bit GP target"
                        }
                        "still uses GpI64 storage" => {
                            "machine function is not valid 32-bit GP-target MachineIR: still uses GpI64 storage"
                        }
                        "still uses scalar i64 integer width" => {
                            "machine function is not valid 32-bit GP-target MachineIR: still uses scalar i64 integer width"
                        }
                        "still uses an unsplit i64 convert/reinterpret op" => {
                            "machine function is not valid 32-bit GP-target MachineIR: still uses an unsplit i64 convert/reinterpret op"
                        }
                        "still uses an i64 trap condition" => {
                            "machine function is not valid 32-bit GP-target MachineIR: still uses an i64 trap condition"
                        }
                        _ => "machine function is not valid 32-bit GP-target MachineIR",
                    })
                })?;
        }
        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    pub(crate) fn validate(&self) -> Result<(), WasmError> {
        for (index, konst) in self.consts.iter().enumerate() {
            if konst.id.0 as usize != index {
                return Err(WasmError::internal("machine const has mismatched id"));
            }
        }

        for (index, func) in self.functions.iter().enumerate() {
            if func.id.0 as usize != index {
                return Err(WasmError::internal("machine function has mismatched id"));
            }
            func.program.validate(self.config)?;
            self.validate_function_refs(func.id, &func.program)?;
        }

        Ok(())
    }

    #[cfg(not(any(debug_assertions, test)))]
    #[inline]
    pub(crate) fn validate(&self) -> Result<(), WasmError> {
        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_function_refs(
        &self,
        func: MachineFuncId,
        program: &MachineProgram,
    ) -> Result<(), WasmError> {
        for (block_idx, block) in program.blocks.iter().enumerate() {
            for inst in &block.ops {
                if let MachineInstKind::CallRuntime(call) = &inst.kind {
                    self.validate_const_id(func, block_idx, call.metadata)?;
                }
            }
            match block.terminator {
                MachineTerminator::Call {
                    target: MachineCallTarget::Direct(callee),
                    ..
                }
                | MachineTerminator::TailCall {
                    target: MachineCallTarget::Direct(callee),
                    ..
                } => {
                    self.validate_func_id(func, block_idx, callee)?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_const_id(
        &self,
        _func: MachineFuncId,
        _block_idx: usize,
        konst: MachineConstId,
    ) -> Result<(), WasmError> {
        if konst.0 as usize >= self.consts.len() {
            return Err(WasmError::internal(
                "machine function block has out-of-range const",
            ));
        }
        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_func_id(
        &self,
        _func: MachineFuncId,
        _block_idx: usize,
        callee: MachineFuncId,
    ) -> Result<(), WasmError> {
        if callee.0 as usize >= self.functions.len() {
            return Err(WasmError::internal(
                "machine function block has out-of-range callee",
            ));
        }
        Ok(())
    }
}

fn validate_32bit_gp_target_param(
    _block_id: MachineBlockId,
    _param_index: usize,
    param: MachineBlockParam,
) -> ValidateResult {
    if matches!(param.ty, MachineStorageType::GpI64) {
        return Err(WasmError::internal(
            "block param still uses GpI64 on a 32-bit GP target",
        ));
    }
    Ok(())
}

fn validate_32bit_gp_target_inst(
    _block_id: MachineBlockId,
    _inst_index: usize,
    inst: &MachineInstKind,
) -> ValidateResult {
    let detail = match inst {
        MachineInstKind::Move { ty, .. }
        | MachineInstKind::Load { ty, .. }
        | MachineInstKind::Store { ty, .. }
        | MachineInstKind::Select { ty, .. }
            if matches!(ty, MachineStorageType::GpI64) =>
        {
            Some("still uses GpI64 storage")
        }
        MachineInstKind::IntUnary { width, .. }
        | MachineInstKind::IntBinary { width, .. }
        | MachineInstKind::IntCompare { width, .. }
        | MachineInstKind::BitfieldExtractU { width, .. }
        | MachineInstKind::IntBinaryShifted { width, .. }
        | MachineInstKind::TestBits { width, .. }
            if matches!(width, MachineIntWidth::I64) =>
        {
            Some("still uses scalar i64 integer width")
        }
        MachineInstKind::Convert { op, .. } if convert_requires_32bit_finalization(*op) => {
            Some("still uses an unsplit i64 convert/reinterpret op")
        }
        MachineInstKind::TrapIf { cond, .. } if branch_cond_requires_32bit_finalization(*cond) => {
            Some("still uses an i64 trap condition")
        }
        _ => None,
    };

    if let Some(detail) = detail {
        return Err(WasmError::internal(detail));
    }

    Ok(())
}

fn validate_32bit_gp_target_term(
    _block_id: MachineBlockId,
    term: &MachineTerminator,
) -> ValidateResult {
    match term {
        MachineTerminator::Branch { cond, .. }
            if branch_cond_requires_32bit_finalization(*cond) =>
        {
            Err(WasmError::internal(
                "block terminator still uses an i64 branch condition",
            ))
        }
        _ => Ok(()),
    }
}

fn branch_cond_requires_32bit_finalization(cond: MachineBranchCond) -> bool {
    matches!(
        cond,
        MachineBranchCond::IntCompare {
            width: MachineIntWidth::I64,
            ..
        } | MachineBranchCond::TestBits {
            width: MachineIntWidth::I64,
            ..
        }
    )
}

fn convert_requires_32bit_finalization(op: MachineConvertOp) -> bool {
    matches!(
        op,
        MachineConvertOp::I32WrapI64
            | MachineConvertOp::I64ExtendI32S
            | MachineConvertOp::I64ExtendI32U
            | MachineConvertOp::I64TruncF32S
            | MachineConvertOp::I64TruncF32U
            | MachineConvertOp::I64TruncF64S
            | MachineConvertOp::I64TruncF64U
            | MachineConvertOp::I64TruncSatF32S
            | MachineConvertOp::I64TruncSatF32U
            | MachineConvertOp::I64TruncSatF64S
            | MachineConvertOp::I64TruncSatF64U
            | MachineConvertOp::F32ConvertI64S
            | MachineConvertOp::F32ConvertI64U
            | MachineConvertOp::F64ConvertI64S
            | MachineConvertOp::F64ConvertI64U
            | MachineConvertOp::I64ReinterpretF64
            | MachineConvertOp::F64ReinterpretI64
    )
}
