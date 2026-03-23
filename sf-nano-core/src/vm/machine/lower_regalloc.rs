//! Register allocation and register-file partition for SSA-IR -> MachineIR lowering.

use alloc::vec::Vec;

use crate::{
    error::WasmError,
    value_type::ValueType,
    vm::{
        backend::BackendConfig,
        machine::machine_ir::{
            machine_ptr_width, machine_word_int_width, MachineBlockParam, MachineBranchCond,
            MachineConvertOp, MachineEdge, MachineFloatWidth, MachineInst,
            MachineInstKind, MachineIntWidth, MachineMemWidth, MachineReg,
            MachineStorageType, MachineTerminator, MachineValue, MACHINE_CTX_REG,
            MACHINE_FIXED_REG_COUNT, MACHINE_FP_REG, MACHINE_MEM0_BASE_REG, MACHINE_MEM0_SIZE_REG,
        },
        middle::ssa_ir::ir::{SsaProgram, SsaValue},
    },
};

use super::lower_context::BlockLowerContext;

// ---------------------------------------------------------------------------
// MachineRegFile — fixed machine-register partition used by lowering
// ---------------------------------------------------------------------------

/// Fixed machine-register partition used by lowering.
///
/// `ctx`, `fp`, and the pinned `mem0` view regs are fixed MachineIR roles.
/// The remaining cache and lane partitions are a logical ownership model chosen
/// for lowering; they may be reused for other temporary purposes when the
/// owning values are proven dead.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct MachineRegFile {
    gp_local_cache: Vec<MachineReg>,
    gp_transient: Vec<MachineReg>,
    fp_transient: Vec<MachineReg>,
    fp_local_cache: Vec<MachineReg>,
    first_fp_reg: u16,
    reg_count: u16,
}

impl MachineRegFile {
    pub(super) fn new(config: BackendConfig) -> Result<Self, WasmError> {
        if config.gp_transient_budget == 0 {
            return Err(WasmError::internal(
                "native lowering requires at least one GP transient register".into(),
            ));
        }

        let mut next = MACHINE_FIXED_REG_COUNT;
        let gp_local_cache = collect_regs(&mut next, config.gp_local_cache_budget);
        let gp_transient = collect_regs(&mut next, config.gp_transient_budget);
        let first_fp_reg = next;
        let fp_transient = collect_regs(&mut next, config.fp_transient_budget);
        let fp_local_cache = collect_regs(&mut next, config.fp_local_cache_budget);

        // Layout: [fixed | gp_local_cache | gp_transient | fp_transient | fp_local_cache]
        //                                                              ^ first_fp_reg
        Ok(Self {
            gp_local_cache,
            gp_transient,
            fp_transient,
            fp_local_cache,
            first_fp_reg,
            reg_count: next,
        })
    }

    #[inline]
    pub(super) const fn runtime_base(&self) -> MachineReg {
        MACHINE_CTX_REG
    }

    #[inline]
    pub(super) const fn frame_base(&self) -> MachineReg {
        MACHINE_FP_REG
    }

    #[inline]
    pub(super) const fn mem0_base(&self) -> MachineReg {
        MACHINE_MEM0_BASE_REG
    }

    #[inline]
    pub(super) const fn mem0_size(&self) -> MachineReg {
        MACHINE_MEM0_SIZE_REG
    }

    #[inline]
    pub(super) fn gp_local_cache(&self, index: usize) -> Option<MachineReg> {
        self.gp_local_cache.get(index).copied()
    }

    #[inline]
    pub(super) fn gp_transient(&self, index: usize) -> Option<MachineReg> {
        self.gp_transient.get(index).copied()
    }

    pub(super) fn gp_transient_count(&self) -> usize {
        self.gp_transient.len()
    }

    #[inline]
    pub(super) fn fp_transient(&self, index: usize) -> Option<MachineReg> {
        self.fp_transient.get(index).copied()
    }

    pub(super) fn fp_transient_count(&self) -> usize {
        self.fp_transient.len()
    }

    #[inline]
    pub(super) fn fp_local_cache(&self, index: usize) -> Option<MachineReg> {
        self.fp_local_cache.get(index).copied()
    }

    pub(super) fn fp_local_cache_count(&self) -> usize {
        self.fp_local_cache.len()
    }

    #[inline]
    pub(super) fn first_fp_reg(&self) -> u16 {
        self.first_fp_reg
    }

    #[inline]
    pub(super) fn reg_count(&self) -> u16 {
        self.reg_count
    }
}

fn collect_regs(next: &mut u16, count: u8) -> Vec<MachineReg> {
    let mut regs = Vec::with_capacity(count as usize);
    for _ in 0..count {
        regs.push(MachineReg(*next));
        *next += 1;
    }
    regs
}

// ---------------------------------------------------------------------------
// BlockLowerContext register allocation methods
// ---------------------------------------------------------------------------

impl<'a> BlockLowerContext<'a> {
    pub(super) fn alloc_value(&mut self, value: SsaValue) -> Result<MachineReg, WasmError> {
        self.alloc_value_in_bank(value, lir_value_storage_type(self.program(), value))
    }

    pub(super) fn alloc_result_value(&mut self, value: SsaValue) -> Result<MachineReg, WasmError> {
        self.alloc_value_in_bank(value, lir_value_storage_type(self.program(), value))
    }

    /// Allocate a LoadSlot destination in the correct bank based on the type table.
    pub(super) fn alloc_slot_load_value(
        &mut self,
        value: SsaValue,
    ) -> Result<MachineReg, WasmError> {
        self.alloc_value_in_bank(value, lir_value_storage_type(self.program(), value))
    }

    pub(super) fn alloc_value_reusing_dead_inputs(
        &mut self,
        value: SsaValue,
        candidates: &[SsaValue],
    ) -> Result<MachineReg, WasmError> {
        self.alloc_value_in_bank_reusing_dead_inputs(
            value,
            candidates,
            lir_value_storage_type(self.program(), value),
        )
    }

    pub(super) fn alloc_result_value_reusing_dead_inputs(
        &mut self,
        value: SsaValue,
        candidates: &[SsaValue],
    ) -> Result<MachineReg, WasmError> {
        self.alloc_value_in_bank_reusing_dead_inputs(
            value,
            candidates,
            lir_value_storage_type(self.program(), value),
        )
    }

    pub(super) fn alloc_float_value(
        &mut self,
        value: SsaValue,
        width: MachineFloatWidth,
    ) -> Result<MachineReg, WasmError> {
        self.alloc_value_in_bank(value, float_storage_type(width))
    }

    pub(super) fn alloc_float_value_reusing_dead_inputs(
        &mut self,
        value: SsaValue,
        candidates: &[SsaValue],
        width: MachineFloatWidth,
    ) -> Result<MachineReg, WasmError> {
        self.alloc_value_in_bank_reusing_dead_inputs(value, candidates, float_storage_type(width))
    }

    pub(super) fn use_value(&mut self, value: SsaValue) -> Result<MachineReg, WasmError> {
        let reg = self.value_reg(value)?;
        if let Some(remaining) = self.remaining_uses_mut().get_mut(&value) {
            *remaining = remaining.saturating_sub(1);
        }
        Ok(reg)
    }

    pub(super) fn use_value_regs(
        &mut self,
        value: SsaValue,
    ) -> Result<(MachineReg, Option<MachineReg>), WasmError> {
        let regs = self.value_regs(value)?;
        if let Some(remaining) = self.remaining_uses_mut().get_mut(&value) {
            *remaining = remaining.saturating_sub(1);
        }
        Ok(regs)
    }

    pub(super) fn use_i64_value_pair(
        &mut self,
        value: SsaValue,
    ) -> Result<(MachineReg, MachineReg), WasmError> {
        let (lo, hi) = self.use_value_regs(value)?;
        hi.map(|hi| (lo, hi)).ok_or_else(|| {
            WasmError::internal(alloc::format!(
                "SSA-IR i64 value {:?} does not have a paired machine-register mapping",
                value
            ))
        })
    }

    pub(super) fn split_continuation_params(
        &self,
        continuation_ops: &[MachineInst],
        continuation_term: &MachineTerminator,
    ) -> Vec<MachineBlockParam> {
        let mut params = Vec::new();
        let all_defined = continuation_ops
            .iter()
            .filter_map(|inst| inst_defined_reg(&inst.kind))
            .filter(|reg| self.is_transient_reg(*reg))
            .collect::<Vec<_>>();

        for entry in self.values_iter() {
            let remaining = self.remaining_use_count(entry.value);
            if remaining != 0
                && self.is_transient_reg(entry.reg)
                && !all_defined.contains(&entry.reg)
            {
                push_unique_param(
                    &mut params,
                    machine_block_param(entry.reg, self.storage_type_for_reg(entry.reg)),
                );
                if let Some(hi_reg) = entry.hi_reg {
                    if self.is_transient_reg(hi_reg) && !all_defined.contains(&hi_reg) {
                        push_unique_param(
                            &mut params,
                            machine_block_param(hi_reg, self.storage_type_for_reg(hi_reg)),
                        );
                    }
                }
            }
        }

        let mut defined_so_far = Vec::new();
        for inst in continuation_ops {
            visit_inst_source_regs(&inst.kind, |reg| {
                if self.is_transient_reg(reg) && !defined_so_far.contains(&reg) {
                    push_unique_param(
                        &mut params,
                        machine_block_param(reg, self.storage_type_for_reg(reg)),
                    );
                }
            });
            if let Some(dst) = inst_defined_reg(&inst.kind) {
                if self.is_transient_reg(dst) && !defined_so_far.contains(&dst) {
                    defined_so_far.push(dst);
                }
            }
        }
        visit_term_source_regs(continuation_term, |reg| {
            if self.is_transient_reg(reg) && !defined_so_far.contains(&reg) {
                push_unique_param(
                    &mut params,
                    machine_block_param(reg, self.storage_type_for_reg(reg)),
                );
            }
        });

        params.sort_by_key(|param| param.reg.0);
        params
    }

    pub(super) fn try_value_reg(&self, value: SsaValue) -> Option<MachineReg> {
        self.values_iter()
            .find(|entry| entry.value == value)
            .map(|entry| entry.reg)
    }

    pub(super) fn dead_value_reg(&self, value: SsaValue) -> Option<MachineReg> {
        if self.remaining_use_count(value) != 0 {
            return None;
        }
        self.try_value_reg(value)
    }

    pub(super) fn try_value_regs(
        &self,
        value: SsaValue,
    ) -> Option<(MachineReg, Option<MachineReg>)> {
        self.values_iter()
            .find(|entry| entry.value == value)
            .map(|entry| (entry.reg, entry.hi_reg))
    }

    fn value_reg(&self, value: SsaValue) -> Result<MachineReg, WasmError> {
        self.try_value_reg(value).ok_or_else(|| {
            WasmError::internal(alloc::format!(
                "no machine register assigned for SSA-IR value {:?}",
                value
            ))
        })
    }

    fn value_regs(&self, value: SsaValue) -> Result<(MachineReg, Option<MachineReg>), WasmError> {
        self.try_value_regs(value).ok_or_else(|| {
            WasmError::internal(alloc::format!(
                "no machine register pair assigned for SSA-IR value {:?}",
                value
            ))
        })
    }

    pub(super) fn alloc_value_in_bank(
        &mut self,
        value: SsaValue,
        ty: MachineStorageType,
    ) -> Result<MachineReg, WasmError> {
        if let Some(reg) = self.try_value_reg(value) {
            return Ok(reg);
        }
        let Some(reg) = self.first_free_transient(ty) else {
            return Err(WasmError::internal(alloc::format!(
                "prepared SSA-IR exceeded {} transient register budget during native lowering in block b{} for value {}",
                if ty.is_fp() { "FP" } else { "GP" },
                self.block_id(),
                value.0,
            )));
        };
        self.push_value_location(value, reg, None);
        self.set_transient(reg, Some(value), Some(ty))?;
        Ok(reg)
    }

    pub(super) fn alloc_i64_value_pair(
        &mut self,
        value: SsaValue,
    ) -> Result<(MachineReg, MachineReg), WasmError> {
        if let Some((lo, Some(hi))) = self.try_value_regs(value) {
            return Ok((lo, hi));
        }
        if self.try_value_reg(value).is_some() {
            return Err(WasmError::internal(alloc::format!(
                "SSA-IR value {:?} already has a scalar machine-register mapping; cannot also allocate a pair",
                value
            )));
        }

        let Some((lo, hi)) = self.first_free_gp_pair_transient() else {
            return Err(WasmError::internal(alloc::format!(
                "prepared SSA-IR exceeded GP transient pair budget during native lowering in block b{} for value {}",
                self.block_id(),
                value.0,
            )));
        };
        self.push_value_location(value, lo, Some(hi));
        // Pair-aware 32-bit lowering treats both halves as GP-word registers.
        self.set_transient(lo, Some(value), Some(MachineStorageType::GpWord))?;
        self.set_transient(hi, Some(value), Some(MachineStorageType::GpWord))?;
        Ok((lo, hi))
    }

    pub(super) fn alloc_i64_value_pair_reusing_dead_inputs(
        &mut self,
        value: SsaValue,
        candidates: &[SsaValue],
    ) -> Result<(MachineReg, MachineReg), WasmError> {
        if let Some((lo, Some(hi))) = self.try_value_regs(value) {
            return Ok((lo, hi));
        }

        for candidate in candidates {
            if self.remaining_use_count(*candidate) != 0 {
                continue;
            }
            if let Some((lo, hi)) = self.try_reuse_pair_candidate(value, *candidate)? {
                return Ok((lo, hi));
            }
        }

        for candidate in candidates {
            if self.remaining_use_count(*candidate) != 0 {
                continue;
            }
            if let Some((lo, hi)) = self.try_reuse_scalar_for_pair(value, *candidate)? {
                return Ok((lo, hi));
            }
        }

        self.alloc_i64_value_pair(value)
    }

    pub(super) fn alloc_value_in_bank_reusing_dead_inputs(
        &mut self,
        value: SsaValue,
        candidates: &[SsaValue],
        ty: MachineStorageType,
    ) -> Result<MachineReg, WasmError> {
        if let Some(reg) = self.try_value_reg(value) {
            return Ok(reg);
        }

        for candidate in candidates {
            if self.remaining_use_count(*candidate) != 0 {
                continue;
            }
            if let Some(reg) = self.try_reuse_scalar_candidate(value, *candidate, ty)? {
                return Ok(reg);
            }
        }

        self.alloc_value_in_bank(value, ty)
    }

    pub(super) fn first_free_transient(&self, ty: MachineStorageType) -> Option<MachineReg> {
        let regfile = self.regfile();
        let start = if ty.is_fp() {
            regfile.gp_transient_count()
        } else {
            0
        };
        let count = if ty.is_fp() {
            regfile.fp_transient_count()
        } else {
            regfile.gp_transient_count()
        };
        for index in start..start + count {
            if !self.transient_occupied(index) {
                return if ty.is_fp() {
                    regfile.fp_transient(index - start)
                } else {
                    regfile.gp_transient(index - start)
                };
            }
        }
        None
    }

    pub(super) fn first_free_gp_pair_transient(&self) -> Option<(MachineReg, MachineReg)> {
        let regfile = self.regfile();
        let mut first = None;
        for index in 0..regfile.gp_transient_count() {
            if self.transient_occupied(index) {
                continue;
            }
            let reg = regfile.gp_transient(index)?;
            if let Some(first_reg) = first {
                return Some((first_reg, reg));
            }
            first = Some(reg);
        }
        None
    }

    pub(super) fn set_transient(
        &mut self,
        reg: MachineReg,
        value: Option<SsaValue>,
        ty: Option<MachineStorageType>,
    ) -> Result<(), WasmError> {
        let index = self.transient_index(reg)?;
        self.set_transient_state(index, value, ty)
    }

    pub(super) fn clear_transient(&mut self, reg: MachineReg) -> Result<(), WasmError> {
        self.set_transient(reg, None, None)
    }

    pub(super) fn is_transient_reg(&self, reg: MachineReg) -> bool {
        self.transient_index(reg).is_ok()
    }

    pub(super) fn is_fp_reg(&self, reg: MachineReg) -> bool {
        let regfile = self.regfile();
        reg.0 >= regfile.first_fp_reg() && reg.0 < regfile.reg_count()
    }

    pub(super) fn transient_index(&self, reg: MachineReg) -> Result<usize, WasmError> {
        let regfile = self.regfile();
        if let Some(first) = regfile.gp_transient(0) {
            let start = first.0;
            let end = start + regfile.gp_transient_count() as u16;
            if reg.0 >= start && reg.0 < end {
                return Ok((reg.0 - start) as usize);
            }
        }

        if let Some(first) = regfile.fp_transient(0) {
            let start = first.0;
            let end = start + regfile.fp_transient_count() as u16;
            if reg.0 >= start && reg.0 < end {
                return Ok(regfile.gp_transient_count() + (reg.0 - start) as usize);
            }
        }

        Err(WasmError::internal(
            "machine register is not in transient partition".into(),
        ))
    }

    pub(super) fn storage_type_for_reg(&self, reg: MachineReg) -> MachineStorageType {
        if let Ok(index) = self.transient_index(reg) {
            return self
                .transient_state_ty(index)
                .unwrap_or(MachineStorageType::GpWord);
        }
        self.cached_local_storage_type_for_reg(reg)
            .unwrap_or(MachineStorageType::GpWord)
    }

    pub(super) fn transient_reg(&self, index: usize) -> Result<MachineReg, WasmError> {
        self.regfile().gp_transient(index).ok_or_else(|| {
            WasmError::internal("native lowering requires one transient register".into())
        })
    }

    pub(super) fn borrow_free_transients(
        &self,
        count: usize,
    ) -> Result<Vec<MachineReg>, WasmError> {
        let regfile = self.regfile();
        let mut regs = Vec::with_capacity(count);
        for index in 0..regfile.gp_transient_count() {
            if !self.transient_occupied(index) {
                regs.push(regfile.gp_transient(index).ok_or_else(|| {
                    WasmError::internal("free transient register index is out of range".into())
                })?);
                if regs.len() == count {
                    return Ok(regs);
                }
            }
        }
        Err(WasmError::internal(alloc::format!(
            "native lowering requires {count} free transient registers"
        )))
    }

    /// Try to coalesce a transient-to-cache-local move by patching the previous
    /// instruction's destination register directly. Returns true if successful.
    ///
    /// This eliminates patterns like `load r_transient <- [addr]; move r_cache <- r_transient`
    /// by rewriting to `load r_cache <- [addr]`.
    ///
    /// Only safe when:
    /// - src is a transient defined by the immediately preceding instruction
    /// - src and target are in the same register bank
    /// - src and target use the same machine storage type
    /// - the last instruction doesn't also read target_reg (would clobber input)
    pub(super) fn try_coalesce_last_dst(
        &mut self,
        src_value: SsaValue,
        src_reg: MachineReg,
        target_reg: MachineReg,
    ) -> bool {
        if src_reg == target_reg {
            return true;
        }
        if self.is_fp_reg(src_reg) != self.is_fp_reg(target_reg) {
            return false;
        }
        if !self.is_transient_reg(src_reg) {
            return false;
        }
        if self.storage_type_for_reg(src_reg) != self.storage_type_for_reg(target_reg) {
            return false;
        }
        // The value must have 0 remaining uses after this store consumes it.
        // This ensures no other instruction between the def and store reads it.
        let remaining = self.remaining_use_count(src_value);
        if remaining != 0 {
            return false;
        }
        let Some(last) = self.ops_last_mut() else {
            return false;
        };
        if !machine_inst_dst_eq(&last.kind, src_reg) {
            return false;
        }
        // Make sure the last instruction doesn't read target_reg as an input
        // (that would mean we'd clobber an input by redirecting the output).
        if machine_inst_uses_reg(&last.kind, target_reg) {
            return false;
        }
        patch_machine_inst_dst(&mut last.kind, target_reg);
        true
    }

    /// Try to fold a dead constant-producing instruction directly into an
    /// uncached frame store so 32-bit lowering does not keep unnecessary GP
    /// temporaries alive across long argument setup.
    pub(super) fn try_coalesce_last_store_immediate(
        &mut self,
        src_value: SsaValue,
        src_reg: MachineReg,
        ty: MachineStorageType,
        addr: crate::vm::machine::machine_ir::MachineAddr,
        width: MachineMemWidth,
    ) -> bool {
        if !self.is_transient_reg(src_reg) {
            return false;
        }
        let remaining = self.remaining_use_count(src_value);
        if remaining != 0 {
            return false;
        }

        let imm = match self.ops_last().map(|inst| &inst.kind) {
            Some(MachineInstKind::Move {
                ty: inst_ty,
                dst,
                src: MachineValue::Imm64(imm),
            }) if *dst == src_reg && *inst_ty == ty => *imm,
            Some(MachineInstKind::FloatConst {
                width: inst_width,
                dst,
                bits,
            }) if *dst == src_reg
                && matches!(
                    (ty, inst_width),
                    (MachineStorageType::Fp32, MachineFloatWidth::F32)
                        | (MachineStorageType::Fp64, MachineFloatWidth::F64)
                ) =>
            {
                *bits
            }
            _ => return false,
        };

        self.ops_pop();
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Store {
                ty,
                addr,
                width,
                src: MachineValue::Imm64(imm),
            },
        });
        true
    }
}

// ---------------------------------------------------------------------------
// Free functions used by register allocation
// ---------------------------------------------------------------------------

pub(super) fn canonical_storage_mem_width(ty: MachineStorageType) -> MachineMemWidth {
    match ty {
        MachineStorageType::GpWord | MachineStorageType::GpI64 | MachineStorageType::Fp64 => {
            MachineMemWidth::U64
        }
        MachineStorageType::Fp32 => MachineMemWidth::U32,
    }
}

pub(super) fn canonical_cached_local_mem_width(ty: MachineStorageType) -> MachineMemWidth {
    canonical_storage_mem_width(ty)
}

pub(super) fn lir_value_storage_type(
    program: &SsaProgram,
    value: SsaValue,
) -> MachineStorageType {
    program
        .value_types
        .get(value.0 as usize)
        .copied()
        .map(value_type_storage_type)
        .unwrap_or(MachineStorageType::GpWord)
}

pub(super) fn canonical_value_mem_width_for_value(
    program: &SsaProgram,
    value: SsaValue,
) -> MachineMemWidth {
    canonical_storage_mem_width(lir_value_storage_type(program, value))
}

fn push_unique_param(params: &mut Vec<MachineBlockParam>, param: MachineBlockParam) {
    if !params.iter().any(|candidate| candidate.reg == param.reg) {
        params.push(param);
    }
}

pub(super) fn machine_block_params_for_value(
    regs: super::lower_context::ValueRegs,
    ty: MachineStorageType,
) -> alloc::vec::Vec<MachineBlockParam> {
    match (ty, regs.hi) {
        (MachineStorageType::GpI64, Some(hi)) => {
            alloc::vec![
                MachineBlockParam::gp_word(regs.lo),
                MachineBlockParam::gp_word(hi)
            ]
        }
        _ => alloc::vec![machine_block_param(regs.lo, ty)],
    }
}

pub(super) fn inst_defined_reg(kind: &MachineInstKind) -> Option<MachineReg> {
    match kind {
        MachineInstKind::Move { dst, .. }
        | MachineInstKind::FloatConst { dst, .. }
        | MachineInstKind::Load { dst, .. }
        | MachineInstKind::IntUnary { dst, .. }
        | MachineInstKind::IntBinary { dst, .. }
        | MachineInstKind::IntCompare { dst, .. }
        | MachineInstKind::FloatUnary { dst, .. }
        | MachineInstKind::FloatBinary { dst, .. }
        | MachineInstKind::FloatCompare { dst, .. }
        | MachineInstKind::Convert { dst, .. }
        | MachineInstKind::Select { dst, .. }
        | MachineInstKind::IndexedLoad { dst, .. } => Some(*dst),
        MachineInstKind::Int64PairBinary { .. } => None,
        MachineInstKind::Int64PairUnary { .. } => None,
        MachineInstKind::Int64PairDivRem { .. } => None,
        MachineInstKind::Int64PairShift { .. } => None,
        MachineInstKind::Int64PairCompare { dst, .. } => Some(*dst),
        MachineInstKind::ConvertFloatToI64Pair { .. } => None,
        MachineInstKind::ConvertI64PairToFloat { dst, .. }
        | MachineInstKind::ReinterpretI64PairToF64 { dst, .. } => Some(*dst),
        MachineInstKind::ReinterpretF64ToI64Pair { .. } => None,
        MachineInstKind::Store { .. }
        | MachineInstKind::IndexedStore { .. }
        | MachineInstKind::TrapIf { .. }
        | MachineInstKind::CallHelper(_) => None,
    }
}

fn visit_inst_source_regs(kind: &MachineInstKind, mut visit: impl FnMut(MachineReg)) {
    match kind {
        MachineInstKind::Move { src, .. } => visit_value_reg(src, &mut visit),
        MachineInstKind::FloatConst { .. } => {}
        MachineInstKind::Load { addr, .. } => visit(addr.base),
        MachineInstKind::Store { addr, src, .. } => {
            visit(addr.base);
            visit_value_reg(src, &mut visit);
        }
        MachineInstKind::IndexedLoad { base, index, .. } => {
            visit(*base);
            visit(*index);
        }
        MachineInstKind::IndexedStore { base, index, src, .. } => {
            visit(*base);
            visit(*index);
            visit_value_reg(src, &mut visit);
        }
        MachineInstKind::IntUnary { src, .. }
        | MachineInstKind::FloatUnary { src, .. }
        | MachineInstKind::Convert { src, .. } => visit_value_reg(src, &mut visit),
        MachineInstKind::IntBinary { lhs, rhs, .. }
        | MachineInstKind::IntCompare { lhs, rhs, .. }
        | MachineInstKind::FloatBinary { lhs, rhs, .. }
        | MachineInstKind::FloatCompare { lhs, rhs, .. } => {
            visit_value_reg(lhs, &mut visit);
            visit_value_reg(rhs, &mut visit);
        }
        MachineInstKind::Int64PairBinary {
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        } => {
            visit_value_reg(lhs_lo, &mut visit);
            visit_value_reg(lhs_hi, &mut visit);
            visit_value_reg(rhs_lo, &mut visit);
            visit_value_reg(rhs_hi, &mut visit);
        }
        MachineInstKind::Int64PairUnary { src_lo, src_hi, .. } => {
            visit_value_reg(src_lo, &mut visit);
            visit_value_reg(src_hi, &mut visit);
        }
        MachineInstKind::Int64PairDivRem {
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        } => {
            visit_value_reg(lhs_lo, &mut visit);
            visit_value_reg(lhs_hi, &mut visit);
            visit_value_reg(rhs_lo, &mut visit);
            visit_value_reg(rhs_hi, &mut visit);
        }
        MachineInstKind::Int64PairShift {
            lhs_lo,
            lhs_hi,
            rhs,
            ..
        } => {
            visit_value_reg(lhs_lo, &mut visit);
            visit_value_reg(lhs_hi, &mut visit);
            visit_value_reg(rhs, &mut visit);
        }
        MachineInstKind::Int64PairCompare {
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        } => {
            visit_value_reg(lhs_lo, &mut visit);
            visit_value_reg(lhs_hi, &mut visit);
            visit_value_reg(rhs_lo, &mut visit);
            visit_value_reg(rhs_hi, &mut visit);
        }
        MachineInstKind::ConvertI64PairToFloat { src_lo, src_hi, .. } => {
            visit_value_reg(src_lo, &mut visit);
            visit_value_reg(src_hi, &mut visit);
        }
        MachineInstKind::ConvertFloatToI64Pair { src, .. }
        | MachineInstKind::ReinterpretF64ToI64Pair { src, .. } => {
            visit_value_reg(src, &mut visit);
        }
        MachineInstKind::ReinterpretI64PairToF64 { src_lo, src_hi, .. } => {
            visit_value_reg(src_lo, &mut visit);
            visit_value_reg(src_hi, &mut visit);
        }
        MachineInstKind::Select {
            on_true,
            on_false,
            cond,
            ..
        } => {
            visit_value_reg(on_true, &mut visit);
            visit_value_reg(on_false, &mut visit);
            visit_value_reg(cond, &mut visit);
        }
        MachineInstKind::TrapIf { cond, .. } => {
            visit_branch_cond_regs(cond, &mut visit);
        }
        MachineInstKind::CallHelper(_) => {}
    }
}

fn visit_branch_cond_regs(cond: &MachineBranchCond, visit: &mut impl FnMut(MachineReg)) {
    match cond {
        MachineBranchCond::Value(value) => visit_value_reg(value, visit),
        MachineBranchCond::IntCompare { lhs, rhs, .. } => {
            visit_value_reg(lhs, visit);
            visit_value_reg(rhs, visit);
        }
    }
}

fn visit_value_reg(value: &MachineValue, visit: &mut impl FnMut(MachineReg)) {
    if let MachineValue::Reg(reg) = value {
        visit(*reg);
    }
}

fn visit_term_source_regs(term: &MachineTerminator, mut visit: impl FnMut(MachineReg)) {
    match term {
        MachineTerminator::Jump(edge) => visit_edge_regs(edge, &mut visit),
        MachineTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => {
            visit_branch_cond_regs(cond, &mut visit);
            visit_edge_regs(then_edge, &mut visit);
            visit_edge_regs(else_edge, &mut visit);
        }
        MachineTerminator::JumpTable { index, entries } => {
            visit_value_reg(index, &mut visit);
            for edge in entries {
                visit_edge_regs(edge, &mut visit);
            }
        }
        MachineTerminator::CallDirect {
            callee_frame_base, ..
        } => visit(*callee_frame_base),
        MachineTerminator::CallIndirect {
            callee_target,
            callee_frame_base,
            ..
        } => {
            visit_value_reg(callee_target, &mut visit);
            visit(*callee_frame_base);
        }
        MachineTerminator::Return | MachineTerminator::Trap { .. } => {}
    }
}

fn visit_edge_regs(edge: &MachineEdge, visit: &mut impl FnMut(MachineReg)) {
    for arg in &edge.args {
        visit_value_reg(arg, visit);
    }
}

pub(super) fn convert_result_float_width(op: MachineConvertOp) -> Option<MachineFloatWidth> {
    use MachineConvertOp as Op;

    Some(match op {
        Op::F32ConvertI32S
        | Op::F32ConvertI32U
        | Op::F32ConvertI64S
        | Op::F32ConvertI64U
        | Op::F32DemoteF64
        | Op::F32ReinterpretI32 => MachineFloatWidth::F32,
        Op::F64ConvertI32S
        | Op::F64ConvertI32U
        | Op::F64ConvertI64S
        | Op::F64ConvertI64U
        | Op::F64PromoteF32
        | Op::F64ReinterpretI64 => MachineFloatWidth::F64,
        _ => return None,
    })
}

/// Check if the instruction defines (writes to) the given register.
fn machine_inst_dst_eq(kind: &MachineInstKind, reg: MachineReg) -> bool {
    match kind {
        MachineInstKind::Move { dst, .. }
        | MachineInstKind::FloatConst { dst, .. }
        | MachineInstKind::Load { dst, .. }
        | MachineInstKind::IntUnary { dst, .. }
        | MachineInstKind::IntBinary { dst, .. }
        | MachineInstKind::IntCompare { dst, .. }
        | MachineInstKind::FloatUnary { dst, .. }
        | MachineInstKind::FloatBinary { dst, .. }
        | MachineInstKind::FloatCompare { dst, .. }
        | MachineInstKind::Convert { dst, .. }
        | MachineInstKind::Select { dst, .. }
        | MachineInstKind::IndexedLoad { dst, .. } => *dst == reg,
        MachineInstKind::Int64PairBinary { dst_lo, dst_hi, .. } => *dst_lo == reg || *dst_hi == reg,
        MachineInstKind::Int64PairUnary { dst_lo, dst_hi, .. } => *dst_lo == reg || *dst_hi == reg,
        MachineInstKind::Int64PairDivRem { dst_lo, dst_hi, .. } => *dst_lo == reg || *dst_hi == reg,
        MachineInstKind::Int64PairShift { dst_lo, dst_hi, .. } => *dst_lo == reg || *dst_hi == reg,
        MachineInstKind::Int64PairCompare { dst, .. } => *dst == reg,
        MachineInstKind::ConvertFloatToI64Pair { dst_lo, dst_hi, .. } => {
            *dst_lo == reg || *dst_hi == reg
        }
        MachineInstKind::ReinterpretF64ToI64Pair { dst_lo, dst_hi, .. } => {
            *dst_lo == reg || *dst_hi == reg
        }
        MachineInstKind::ConvertI64PairToFloat { dst, .. }
        | MachineInstKind::ReinterpretI64PairToF64 { dst, .. } => *dst == reg,
        MachineInstKind::Store { .. }
        | MachineInstKind::IndexedStore { .. }
        | MachineInstKind::TrapIf { .. }
        | MachineInstKind::CallHelper(_) => false,
    }
}

/// Check if the instruction reads the given register as an input.
fn machine_inst_uses_reg(kind: &MachineInstKind, reg: MachineReg) -> bool {
    let is = |v: &MachineValue| matches!(v, MachineValue::Reg(r) if *r == reg);
    match kind {
        MachineInstKind::Move { src, .. } => is(src),
        MachineInstKind::FloatConst { .. } => false,
        MachineInstKind::Load { addr, .. } => addr.base == reg,
        MachineInstKind::Store { addr, src, .. } => addr.base == reg || is(src),
        MachineInstKind::IndexedLoad { base, index, .. } => *base == reg || *index == reg,
        MachineInstKind::IndexedStore { base, index, src, .. } => {
            *base == reg || *index == reg || is(src)
        }
        MachineInstKind::IntUnary { src, .. }
        | MachineInstKind::FloatUnary { src, .. }
        | MachineInstKind::Convert { src, .. } => is(src),
        MachineInstKind::IntBinary { lhs, rhs, .. }
        | MachineInstKind::IntCompare { lhs, rhs, .. }
        | MachineInstKind::FloatBinary { lhs, rhs, .. }
        | MachineInstKind::FloatCompare { lhs, rhs, .. } => is(lhs) || is(rhs),
        MachineInstKind::Int64PairBinary {
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        } => is(lhs_lo) || is(lhs_hi) || is(rhs_lo) || is(rhs_hi),
        MachineInstKind::Int64PairUnary { src_lo, src_hi, .. } => is(src_lo) || is(src_hi),
        MachineInstKind::Int64PairDivRem {
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        } => is(lhs_lo) || is(lhs_hi) || is(rhs_lo) || is(rhs_hi),
        MachineInstKind::Int64PairShift {
            lhs_lo,
            lhs_hi,
            rhs,
            ..
        } => is(lhs_lo) || is(lhs_hi) || is(rhs),
        MachineInstKind::Int64PairCompare {
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        } => is(lhs_lo) || is(lhs_hi) || is(rhs_lo) || is(rhs_hi),
        MachineInstKind::ConvertI64PairToFloat { src_lo, src_hi, .. } => is(src_lo) || is(src_hi),
        MachineInstKind::ConvertFloatToI64Pair { src, .. }
        | MachineInstKind::ReinterpretF64ToI64Pair { src, .. } => is(src),
        MachineInstKind::ReinterpretI64PairToF64 { src_lo, src_hi, .. } => is(src_lo) || is(src_hi),
        MachineInstKind::Select {
            on_true,
            on_false,
            cond,
            ..
        } => is(on_true) || is(on_false) || is(cond),
        MachineInstKind::TrapIf { .. } | MachineInstKind::CallHelper(_) => false,
    }
}

/// Patch the destination register of an instruction in place.
fn patch_machine_inst_dst(kind: &mut MachineInstKind, new_dst: MachineReg) {
    match kind {
        MachineInstKind::Move { dst, .. }
        | MachineInstKind::FloatConst { dst, .. }
        | MachineInstKind::Load { dst, .. }
        | MachineInstKind::IntUnary { dst, .. }
        | MachineInstKind::IntBinary { dst, .. }
        | MachineInstKind::IntCompare { dst, .. }
        | MachineInstKind::FloatUnary { dst, .. }
        | MachineInstKind::FloatBinary { dst, .. }
        | MachineInstKind::FloatCompare { dst, .. }
        | MachineInstKind::Convert { dst, .. }
        | MachineInstKind::Select { dst, .. }
        | MachineInstKind::IndexedLoad { dst, .. } => *dst = new_dst,
        MachineInstKind::ConvertI64PairToFloat { dst, .. }
        | MachineInstKind::ReinterpretI64PairToF64 { dst, .. } => *dst = new_dst,
        MachineInstKind::Int64PairBinary { dst_lo, dst_hi, .. }
        | MachineInstKind::Int64PairDivRem { dst_lo, dst_hi, .. }
        | MachineInstKind::Int64PairUnary { dst_lo, dst_hi, .. } => {
            if *dst_lo == new_dst || *dst_hi == new_dst {
                *dst_lo = new_dst;
            }
        }
        MachineInstKind::ConvertFloatToI64Pair { dst_lo, dst_hi, .. }
        | MachineInstKind::ReinterpretF64ToI64Pair { dst_lo, dst_hi, .. } => {
            if *dst_lo == new_dst || *dst_hi == new_dst {
                *dst_lo = new_dst;
            }
        }
        MachineInstKind::Int64PairShift { dst_lo, dst_hi, .. } => {
            if *dst_lo == new_dst || *dst_hi == new_dst {
                *dst_lo = new_dst;
            }
        }
        MachineInstKind::Int64PairCompare { dst, .. } => *dst = new_dst,
        MachineInstKind::Store { .. }
        | MachineInstKind::IndexedStore { .. }
        | MachineInstKind::TrapIf { .. }
        | MachineInstKind::CallHelper(_) => {}
    }
}

fn machine_block_param(reg: MachineReg, ty: MachineStorageType) -> MachineBlockParam {
    match ty {
        MachineStorageType::GpWord => MachineBlockParam::gp_word(reg),
        MachineStorageType::GpI64 => MachineBlockParam::gp_i64(reg),
        MachineStorageType::Fp32 => MachineBlockParam::fp(reg, MachineFloatWidth::F32),
        MachineStorageType::Fp64 => MachineBlockParam::fp(reg, MachineFloatWidth::F64),
    }
}

pub(super) fn value_type_storage_type(ty: ValueType) -> MachineStorageType {
    match ty {
        ValueType::F32 => MachineStorageType::Fp32,
        ValueType::F64 => MachineStorageType::Fp64,
        ValueType::I64 => MachineStorageType::GpI64,
        _ => MachineStorageType::GpWord,
    }
}

pub(super) fn float_storage_type(width: MachineFloatWidth) -> MachineStorageType {
    match width {
        MachineFloatWidth::F32 => MachineStorageType::Fp32,
        MachineFloatWidth::F64 => MachineStorageType::Fp64,
    }
}

pub(super) fn gp_reg_mem_width(gp_reg_width: u8) -> MachineMemWidth {
    machine_ptr_width(gp_reg_width)
}

pub(super) fn gp_reg_int_width(gp_reg_width: u8) -> MachineIntWidth {
    machine_word_int_width(gp_reg_width)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use alloc::boxed::Box;

    use crate::{
        value_type::ValueType,
        vm::{
            backend::BackendConfig,
            machine::machine_ir::{
                MachineFunctionRuntime, MachineCallLinkLayout, MachineStorageType,
            },
            middle::{
                ssa_ir::{
                    ir::{SsaBlock, SsaLocalCachePrefs, SsaProgram, SsaTerminator, SsaValue},
                    target::SsaTarget,
                },
            },
        },
    };

    use super::*;
    use crate::vm::machine::lower_context::BlockLowerContext;

    fn make_test_context(value_types: Vec<ValueType>) -> BlockLowerContext<'static> {
        let program = Box::leak(Box::new(SsaProgram {
            entry: SsaTarget(0),
            local_cache: SsaLocalCachePrefs::default(),
            blocks: alloc::vec![SsaBlock {
                id: SsaTarget(0),
                params: alloc::vec![],
                ops: alloc::vec![],
                terminator: SsaTerminator::Return { results: None },
            }],
            value_types,
        }));
        let regfile = Box::leak(Box::new(
            MachineRegFile::new(BackendConfig::new_with_gp_unit_bytes(0, 4, 0, 0, 4))
                .expect("regfile"),
        ));
        let runtime = MachineFunctionRuntime::default();
        let all_runtime = Box::leak(Box::new(alloc::vec![runtime]));
        let call_link = MachineCallLinkLayout {
            continuation_offset: 0,
            caller_frame_offset: 8,
            caller_result_base_offset: 16,
            slot_count: 3,
        };

        BlockLowerContext::new(
            regfile,
            program,
            &program.local_cache,
            &program.blocks[0],
            all_runtime,
            call_link,
            4,
            &super::super::gp32::Gp32Lowering,
            true,
            #[cfg(has_guard_pages)]
            false,
        )
        .expect("lower context")
    }

    #[test]
    fn alloc_i64_value_pair_reserves_two_gp_word_transients() {
        let mut lower = make_test_context(alloc::vec![ValueType::I64]);
        let (lo, hi) = lower.alloc_i64_value_pair(SsaValue(0)).expect("pair alloc");
        assert_ne!(lo, hi);
        assert_eq!(
            lower.use_i64_value_pair(SsaValue(0)).expect("pair use"),
            (lo, hi)
        );
        assert_eq!(lower.storage_type_for_reg(lo), MachineStorageType::GpWord);
        assert_eq!(lower.storage_type_for_reg(hi), MachineStorageType::GpWord);

        let lo_index = lower.transient_index(lo).expect("lo transient");
        let hi_index = lower.transient_index(hi).expect("hi transient");
        assert!(lower.transient_occupied(lo_index));
        assert!(lower.transient_occupied(hi_index));

        lower.release_dead_values().expect("release pair");
        assert!(lower.try_value_regs(SsaValue(0)).is_none());
        assert!(!lower.transient_occupied(lo_index));
        assert!(!lower.transient_occupied(hi_index));
    }

    #[test]
    fn scalar_reuse_can_claim_low_half_of_dead_pair_and_frees_high_half() {
        let mut lower = make_test_context(alloc::vec![ValueType::I64, ValueType::I32]);
        let (pair_lo, pair_hi) = lower.alloc_i64_value_pair(SsaValue(0)).expect("pair alloc");
        let scalar = lower
            .alloc_value_in_bank_reusing_dead_inputs(
                SsaValue(1),
                &[SsaValue(0)],
                MachineStorageType::GpWord,
            )
            .expect("scalar alloc");

        assert_eq!(scalar, pair_lo);
        assert_eq!(lower.try_value_regs(SsaValue(0)), None);
        assert_eq!(lower.try_value_regs(SsaValue(1)), Some((pair_lo, None)));
        let hi_index = lower.transient_index(pair_hi).expect("hi transient");
        assert!(!lower.transient_occupied(hi_index));
    }

    #[test]
    fn pair_reuse_can_claim_low_half_of_dead_scalar_and_allocate_only_high_half() {
        let mut lower = make_test_context(alloc::vec![ValueType::I32, ValueType::I64]);
        let scalar = lower
            .alloc_value_in_bank(SsaValue(0), MachineStorageType::GpWord)
            .expect("scalar alloc");
        let (pair_lo, pair_hi) = lower
            .alloc_i64_value_pair_reusing_dead_inputs(SsaValue(1), &[SsaValue(0)])
            .expect("pair alloc reusing dead scalar");

        assert_eq!(pair_lo, scalar);
        assert_ne!(pair_lo, pair_hi);
        assert_eq!(lower.try_value_regs(SsaValue(0)), None);
        assert_eq!(
            lower.try_value_regs(SsaValue(1)),
            Some((pair_lo, Some(pair_hi)))
        );
        assert_eq!(
            lower.storage_type_for_reg(pair_lo),
            MachineStorageType::GpWord
        );
        assert_eq!(
            lower.storage_type_for_reg(pair_hi),
            MachineStorageType::GpWord
        );
    }
}
