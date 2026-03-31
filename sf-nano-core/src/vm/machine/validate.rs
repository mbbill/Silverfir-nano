use crate::error::WasmError;
use crate::vm::backend::BackendConfig;

use super::machine_ir::{
    MachineBlockId, MachineBlockParam, MachineBranchCond,
    MachineConvertOp, MachineInstKind, MachineIntWidth,
    MachineModule, MachineProgram, MachineStorageType, MachineTerminator,
};
#[cfg(any(debug_assertions, test))]
use super::machine_ir::{MachineIntBinaryOp, MachineIntUnaryOp};
#[cfg(any(debug_assertions, test))]
use super::machine_ir::{
    MachineAddr, MachineConstId, MachineEdge, MachineExternId, MachineFloatWidth,
    MachineFuncId, MachineInst, MachineReg, MachineValue,
};
#[cfg(any(debug_assertions, test))]
use super::machine_ir::is_fp_reg;

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
            return Err(WasmError::internal(alloc::format!(
                "expected first_fp_reg {} for 32-bit GP target MachineIR, found {}",
                max_gp_regs,
                first_fp,
            )));
        }
        if total < first_fp {
            return Err(WasmError::internal(alloc::format!(
                "machine reg_count {} is below 32-bit GP-target fp boundary {}",
                total,
                first_fp,
            )));
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
            return Err(WasmError::internal(alloc::format!(
                "machine entry block {} is out of range for {} blocks",
                self.entry.as_usize(),
                self.blocks.len(),
            )));
        }

        let reg_count = config.total_reg_count();
        let first_fp = config.first_fp_reg();
        let fp_bank_count = reg_count.saturating_sub(first_fp) as usize;
        if config.fp_transient_budget as usize > fp_bank_count {
            return Err(WasmError::internal(alloc::format!(
                "machine fp_transient_count {} exceeds fp bank size {}",
                config.fp_transient_budget,
                fp_bank_count,
            )));
        }
        if !self.fp_reg_init_widths.is_empty() && self.fp_reg_init_widths.len() != fp_bank_count {
            return Err(WasmError::internal(alloc::format!(
                "machine fp_reg_init_widths length {} does not match fp bank size {}",
                self.fp_reg_init_widths.len(),
                fp_bank_count,
            )));
        }

        for (index, block) in self.blocks.iter().enumerate() {
            if block.id.as_usize() != index {
                return Err(WasmError::internal(alloc::format!(
                    "machine block {} has mismatched id {}",
                    index,
                    block.id.as_usize(),
                )));
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
            return Err(WasmError::internal(alloc::format!(
                "machine block param {} has mismatched storage type {:?} for its register bank",
                param.reg.0,
                param.ty,
            )));
        }
        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_inst(&self, inst: &MachineInst, config: BackendConfig) -> ValidateResult {
        match &inst.kind {
            MachineInstKind::Move { ty, dst, src } => {
                self.validate_reg(*dst, config)?;
                self.validate_value(*src, config)?;
                self.validate_reg_storage_type(*dst, *ty, config)?;
            }
            MachineInstKind::FloatConst { dst, .. } => {
                self.validate_reg(*dst, config)?;
                if !is_fp_reg(*dst, config) {
                    return Err(WasmError::internal(alloc::format!(
                        "machine FloatConst destination {} must be an FP register",
                        dst.0,
                    )));
                }
            }
            MachineInstKind::Load { ty, dst, addr, .. } => {
                self.validate_reg(*dst, config)?;
                self.validate_addr(*addr, config)?;
                self.validate_reg_storage_type(*dst, *ty, config)?;
            }
            MachineInstKind::Store { ty, addr, src, .. } => {
                self.validate_addr(*addr, config)?;
                self.validate_value(*src, config)?;
                if let MachineValue::Reg(src_reg) = src {
                    self.validate_reg_storage_type(*src_reg, *ty, config)?;
                }
            }
            MachineInstKind::IntUnary { dst, src, .. }
            | MachineInstKind::FloatUnary { dst, src, .. }
            | MachineInstKind::Convert { dst, src, .. } => {
                self.validate_reg(*dst, config)?;
                self.validate_value(*src, config)?;
            }
            MachineInstKind::IntBinary { dst, lhs, rhs, .. }
            | MachineInstKind::IntCompare { dst, lhs, rhs, .. }
            | MachineInstKind::FloatBinary { dst, lhs, rhs, .. } => {
                self.validate_reg(*dst, config)?;
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
                            | MachineConvertOp::I64TruncSatF32U => {
                                MachineStorageType::Fp32
                            }
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
                    self.validate_reg_storage_type(*src_lo_reg, MachineStorageType::GpWord, config)?;
                }
                if let MachineValue::Reg(src_hi_reg) = src_hi {
                    self.validate_reg_storage_type(*src_hi_reg, MachineStorageType::GpWord, config)?;
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
                dst,
                base,
                index,
                ..
            } => {
                self.validate_reg(*dst, config)?;
                self.validate_reg(*base, config)?;
                self.validate_reg(*index, config)?;
            }
            MachineInstKind::IndexedStore {
                base,
                index,
                src,
                ..
            } => {
                self.validate_reg(*base, config)?;
                self.validate_reg(*index, config)?;
                self.validate_value(*src, config)?;
            }
            MachineInstKind::TrapIf { cond, .. } => {
                self.validate_branch_cond(*cond, config)?;
            }
            MachineInstKind::CallHelper(_) => {}
            MachineInstKind::MemoryGrow {
                dst, delta, ..
            } => {
                self.validate_reg(*dst, config)?;
                self.validate_value(*delta, config)?;
            }
        }
        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_term(&self, term: &MachineTerminator, source_block: usize, config: BackendConfig) -> ValidateResult {
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
            MachineTerminator::CallDirect {
                callee_frame_base,
                continuation,
                ..
            } => {
                self.validate_reg(*callee_frame_base, config)?;
                self.validate_block_id(*continuation, source_block, "continuation")
            }
            MachineTerminator::CallIndirect {
                callee_target,
                callee_frame_base,
                continuation,
                ..
            } => {
                self.validate_value(*callee_target, config)?;
                self.validate_reg(*callee_frame_base, config)?;
                self.validate_block_id(*continuation, source_block, "continuation")
            }
            MachineTerminator::Return => Ok(()),
            MachineTerminator::Trap { .. } => Ok(()),
        }
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_branch_cond(&self, cond: MachineBranchCond, config: BackendConfig) -> ValidateResult {
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
    fn validate_edge(&self, edge: &MachineEdge, source_block: usize, config: BackendConfig) -> ValidateResult {
        self.validate_block_id(edge.target, source_block, "edge target")?;
        let target = &self.blocks[edge.target.as_usize()];
        if edge.args.len() != target.params.len() {
            return Err(WasmError::internal(alloc::format!(
                "machine block {} -> {} supplies {} args, but target expects {}",
                source_block,
                edge.target.as_usize(),
                edge.args.len(),
                target.params.len(),
            )));
        }
        for value in &edge.args {
            self.validate_value(*value, config)?;
        }
        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_block_id(
        &self,
        block: MachineBlockId,
        source_block: usize,
        role: &str,
    ) -> ValidateResult {
        if block.as_usize() >= self.blocks.len() {
            return Err(WasmError::internal(alloc::format!(
                "machine block {} has out-of-range {} {}",
                source_block,
                role,
                block.as_usize(),
            )));
        }
        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_value(&self, value: MachineValue, config: BackendConfig) -> ValidateResult {
        match value {
            MachineValue::Reg(reg) => self.validate_reg(reg, config),
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
            return Err(WasmError::internal(alloc::format!(
                "machine register {} exceeds declared register count {}",
                reg.0,
                reg_count,
            )));
        }
        if first_fp > reg_count {
            return Err(WasmError::internal(alloc::format!(
                "machine first_fp_reg {} exceeds declared register count {}",
                first_fp,
                reg_count,
            )));
        }
        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_reg_storage_type(&self, reg: MachineReg, ty: MachineStorageType, config: BackendConfig) -> ValidateResult {
        if is_fp_reg(reg, config) != ty.is_fp() {
            return Err(WasmError::internal(alloc::format!(
                "machine register {} has storage type {:?} in the wrong bank",
                reg.0,
                ty,
            )));
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
                    WasmError::internal(alloc::format!(
                        "machine function {} is not valid 32-bit GP-target MachineIR: {}",
                        func.id.0,
                        err
                    ))
                })?;
        }
        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    pub(crate) fn validate(&self) -> Result<(), WasmError> {
        for (index, konst) in self.consts.iter().enumerate() {
            if konst.id.0 as usize != index {
                return Err(WasmError::internal(alloc::format!(
                    "machine const {} has mismatched id {}",
                    index,
                    konst.id.0,
                )));
            }
        }

        for (index, func) in self.functions.iter().enumerate() {
            if func.id.0 as usize != index {
                return Err(WasmError::internal(alloc::format!(
                    "machine function {} has mismatched id {}",
                    index,
                    func.id.0,
                )));
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
                if let MachineInstKind::CallHelper(call) = &inst.kind {
                    self.validate_const_id(func, block_idx, call.metadata)?;
                    self.validate_extern_id(func, block_idx, call.target)?;
                }
            }
            if let MachineTerminator::CallDirect { callee, .. } = block.terminator {
                self.validate_func_id(func, block_idx, callee)?;
            }
        }
        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_const_id(
        &self,
        func: MachineFuncId,
        block_idx: usize,
        konst: MachineConstId,
    ) -> Result<(), WasmError> {
        if konst.0 as usize >= self.consts.len() {
            return Err(WasmError::internal(alloc::format!(
                "machine function {} block {} has out-of-range const {}",
                func.0,
                block_idx,
                konst.0,
            )));
        }
        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_func_id(
        &self,
        func: MachineFuncId,
        block_idx: usize,
        callee: MachineFuncId,
    ) -> Result<(), WasmError> {
        if callee.0 as usize >= self.functions.len() {
            return Err(WasmError::internal(alloc::format!(
                "machine function {} block {} has out-of-range callee {}",
                func.0,
                block_idx,
                callee.0,
            )));
        }
        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_extern_id(
        &self,
        func: MachineFuncId,
        block_idx: usize,
        target: MachineExternId,
    ) -> Result<(), WasmError> {
        if target.0 as usize >= self.externs.len() {
            return Err(WasmError::internal(alloc::format!(
                "machine function {} block {} has out-of-range extern {}",
                func.0,
                block_idx,
                target.0,
            )));
        }
        Ok(())
    }
}

fn validate_32bit_gp_target_param(
    block_id: MachineBlockId,
    param_index: usize,
    param: MachineBlockParam,
) -> ValidateResult {
    if matches!(param.ty, MachineStorageType::GpI64) {
        return Err(WasmError::internal(alloc::format!(
            "block {} param {} still uses GpI64 on a 32-bit GP target",
            block_id.0,
            param_index,
        )));
    }
    Ok(())
}

fn validate_32bit_gp_target_inst(
    block_id: MachineBlockId,
    inst_index: usize,
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
        return Err(WasmError::internal(alloc::format!(
            "block {} op {} {:?} {}",
            block_id.0,
            inst_index,
            inst,
            detail,
        )));
    }

    Ok(())
}

fn validate_32bit_gp_target_term(
    block_id: MachineBlockId,
    term: &MachineTerminator,
) -> ValidateResult {
    match term {
        MachineTerminator::Branch { cond, .. }
            if branch_cond_requires_32bit_finalization(*cond) =>
        {
            Err(WasmError::internal(alloc::format!(
                "block {} terminator {:?} still uses an i64 branch condition",
                block_id.0,
                term,
            )))
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
