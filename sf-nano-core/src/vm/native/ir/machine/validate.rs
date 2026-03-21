use crate::error::WasmError;

use super::cfg::{MachineBlockParam, MachineBranchCond, MachineEdge, MachineTerminator};
use super::inst::{MachineInst, MachineInstKind};
use super::module::{MachineModule, MachineProgram};
use super::types::{
    MachineAddr, MachineBlockId, MachineConstId, MachineExternId, MachineFuncId, MachineReg,
    MachineValue,
};

type ValidateResult = Result<(), WasmError>;

impl MachineProgram {
    #[cfg(any(debug_assertions, test))]
    pub fn validate(&self) -> Result<(), WasmError> {
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

        let fp_bank_count = self.reg_count.saturating_sub(self.first_fp_reg) as usize;
        if self.fp_transient_count as usize > fp_bank_count {
            return Err(WasmError::internal(alloc::format!(
                "machine fp_transient_count {} exceeds fp bank size {}",
                self.fp_transient_count,
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
                self.validate_param(*param)?;
            }
            for inst in &block.ops {
                self.validate_inst(inst)?;
            }
            self.validate_term(&block.terminator, index)?;
        }

        Ok(())
    }

    #[cfg(not(any(debug_assertions, test)))]
    #[inline]
    pub fn validate(&self) -> Result<(), WasmError> {
        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_param(&self, param: MachineBlockParam) -> ValidateResult {
        self.validate_reg(param.reg)?;
        if self.is_fp_reg(param.reg) {
            if param.float_width.is_none() {
                return Err(WasmError::internal(alloc::format!(
                    "machine FP block param {} is missing its float width",
                    param.reg.0,
                )));
            }
        } else if param.float_width.is_some() {
            return Err(WasmError::internal(alloc::format!(
                "machine GP block param {} must not declare an FP width",
                param.reg.0,
            )));
        }
        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_inst(&self, inst: &MachineInst) -> ValidateResult {
        match &inst.kind {
            MachineInstKind::Move { dst, src } => {
                self.validate_reg(*dst)?;
                self.validate_value(*src)?;
            }
            MachineInstKind::FloatConst { dst, .. } => {
                self.validate_reg(*dst)?;
                if !self.is_fp_reg(*dst) {
                    return Err(WasmError::internal(alloc::format!(
                        "machine FloatConst destination {} must be an FP register",
                        dst.0,
                    )));
                }
            }
            MachineInstKind::Lea { dst, addr } => {
                self.validate_reg(*dst)?;
                self.validate_addr(*addr)?;
            }
            MachineInstKind::Load { dst, addr, .. } => {
                self.validate_reg(*dst)?;
                self.validate_addr(*addr)?;
            }
            MachineInstKind::Store { addr, src, .. } => {
                self.validate_addr(*addr)?;
                self.validate_value(*src)?;
            }
            MachineInstKind::IntUnary { dst, src, .. }
            | MachineInstKind::FloatUnary { dst, src, .. }
            | MachineInstKind::Convert { dst, src, .. } => {
                self.validate_reg(*dst)?;
                self.validate_value(*src)?;
            }
            MachineInstKind::IntBinary { dst, lhs, rhs, .. }
            | MachineInstKind::IntCompare { dst, lhs, rhs, .. }
            | MachineInstKind::FloatBinary { dst, lhs, rhs, .. } => {
                self.validate_reg(*dst)?;
                self.validate_value(*lhs)?;
                self.validate_value(*rhs)?;
            }
            MachineInstKind::FloatCompare { dst, lhs, rhs, .. } => {
                self.validate_reg(*dst)?;
                self.validate_value(*lhs)?;
                self.validate_value(*rhs)?;
            }
            MachineInstKind::Select {
                dst,
                on_true,
                on_false,
                cond,
            } => {
                self.validate_reg(*dst)?;
                self.validate_value(*on_true)?;
                self.validate_value(*on_false)?;
                self.validate_value(*cond)?;
            }
            MachineInstKind::TrapIf { cond, .. } => {
                self.validate_branch_cond(*cond)?;
            }
            MachineInstKind::IntAddCarryOut { dst, carry_out, lhs, rhs } => {
                self.validate_reg(*dst)?;
                self.validate_reg(*carry_out)?;
                self.validate_value(*lhs)?;
                self.validate_value(*rhs)?;
            }
            MachineInstKind::IntAddWithCarry { dst, lhs, rhs, carry_in } => {
                self.validate_reg(*dst)?;
                self.validate_value(*lhs)?;
                self.validate_value(*rhs)?;
                self.validate_value(*carry_in)?;
            }
            MachineInstKind::IntSubBorrowOut { dst, borrow_out, lhs, rhs } => {
                self.validate_reg(*dst)?;
                self.validate_reg(*borrow_out)?;
                self.validate_value(*lhs)?;
                self.validate_value(*rhs)?;
            }
            MachineInstKind::IntSubWithBorrow { dst, lhs, rhs, borrow_in } => {
                self.validate_reg(*dst)?;
                self.validate_value(*lhs)?;
                self.validate_value(*rhs)?;
                self.validate_value(*borrow_in)?;
            }
            MachineInstKind::IntMulWide { dst_lo, dst_hi, lhs, rhs, .. } => {
                self.validate_reg(*dst_lo)?;
                self.validate_reg(*dst_hi)?;
                self.validate_value(*lhs)?;
                self.validate_value(*rhs)?;
            }
            MachineInstKind::CallHelper(_) => {}
        }
        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_term(&self, term: &MachineTerminator, source_block: usize) -> ValidateResult {
        match term {
            MachineTerminator::Jump(edge) => self.validate_edge(edge, source_block),
            MachineTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            } => {
                self.validate_branch_cond(*cond)?;
                self.validate_edge(then_edge, source_block)?;
                self.validate_edge(else_edge, source_block)
            }
            MachineTerminator::JumpTable { index, entries } => {
                self.validate_value(*index)?;
                for edge in entries {
                    self.validate_edge(edge, source_block)?;
                }
                Ok(())
            }
            MachineTerminator::CallDirect {
                callee_frame_base,
                continuation,
                ..
            } => {
                self.validate_reg(*callee_frame_base)?;
                self.validate_block_id(*continuation, source_block, "continuation")
            }
            MachineTerminator::CallIndirect {
                callee_target,
                callee_frame_base,
                continuation,
                ..
            } => {
                self.validate_value(*callee_target)?;
                self.validate_reg(*callee_frame_base)?;
                self.validate_block_id(*continuation, source_block, "continuation")
            }
            MachineTerminator::Return => Ok(()),
            MachineTerminator::Trap { .. } => Ok(()),
        }
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_branch_cond(&self, cond: MachineBranchCond) -> ValidateResult {
        match cond {
            MachineBranchCond::Value(value) => self.validate_value(value),
            MachineBranchCond::IntCompare { lhs, rhs, .. }
            | MachineBranchCond::FloatCompare { lhs, rhs, .. } => {
                self.validate_value(lhs)?;
                self.validate_value(rhs)
            }
        }
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_edge(&self, edge: &MachineEdge, source_block: usize) -> ValidateResult {
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
            self.validate_value(*value)?;
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
    fn validate_value(&self, value: MachineValue) -> ValidateResult {
        match value {
            MachineValue::Reg(reg) => self.validate_reg(reg),
            MachineValue::Imm64(_) => Ok(()),
        }
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_addr(&self, addr: MachineAddr) -> ValidateResult {
        self.validate_reg(addr.base)
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_reg(&self, reg: MachineReg) -> ValidateResult {
        if reg.0 >= self.reg_count {
            return Err(WasmError::internal(alloc::format!(
                "machine register {} exceeds declared register count {}",
                reg.0,
                self.reg_count,
            )));
        }
        if self.first_fp_reg > self.reg_count {
            return Err(WasmError::internal(alloc::format!(
                "machine first_fp_reg {} exceeds declared register count {}",
                self.first_fp_reg,
                self.reg_count,
            )));
        }
        Ok(())
    }
}

impl MachineModule {
    #[cfg(any(debug_assertions, test))]
    pub fn validate(&self) -> Result<(), WasmError> {
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
            func.program.validate()?;
            self.validate_function_refs(func.id, &func.program)?;
        }

        Ok(())
    }

    #[cfg(not(any(debug_assertions, test)))]
    #[inline]
    pub fn validate(&self) -> Result<(), WasmError> {
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
