//! Register allocation and register-file layout for SSA-IR -> MachineIR lowering.

use crate::collections;

use crate::{
    error::WasmError,
    value_type::ValueType,
    vm::{
        backend::BackendConfig,
        machine::machine_ir::{
            machine_ptr_width, machine_word_int_width, MachineAddr, MachineBlockParam,
            MachineBranchCond, MachineCallTarget, MachineConvertOp, MachineEdge, MachineFloatWidth,
            MachineInst, MachineInstKind, MachineIntWidth, MachineMemWidth, MachineReg,
            MachineRegOwner, MachineStorageType, MachineTerminator, MachineValue, MACHINE_CTX_REG,
            MACHINE_FIXED_REG_COUNT, MACHINE_FP_REG, MACHINE_MEM0_BASE_REG, MACHINE_MEM0_SIZE_REG,
        },
        middle::ssa_ir::ir::{DecodedOperand, SsaOperand, SsaProgram, SsaValue},
    },
};

use super::lower_context::BlockLowerContext;

// ---------------------------------------------------------------------------
// MachineRegFile — unified dynamic register banks used by lowering
// ---------------------------------------------------------------------------

/// Fixed machine-register layout used by lowering.
///
/// `ctx`, `fp`, and the pinned `mem0` view regs are fixed MachineIR roles.
/// The remaining GP/FP registers form two ordered dynamic banks. The order is
/// a backend preference only: semantic linear-value ownership is tracked by
/// live lowering state in `BlockLowerContext`, not by register number.
///
/// Some lowering helpers still borrow free GP dynamic regs for short-lived
/// control plumbing. Those helpers must be explicit at the callsite because
/// borrowing a dynamic reg for non-SSA work is the exception rather than the
/// rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct MachineRegFile {
    gp_dynamic: collections::Vec<MachineReg>,
    fp_dynamic: collections::Vec<MachineReg>,
    gp_allocatable_count: usize,
    first_fp_reg: u16,
    reg_count: u16,
}

impl MachineRegFile {
    pub(super) fn new(config: BackendConfig) -> Result<Self, WasmError> {
        if config.gp_dynamic_budget == 0 {
            return Err(WasmError::internal(
                "native lowering requires at least one GP dynamic register".into(),
            ));
        }

        let mut next = MACHINE_FIXED_REG_COUNT;
        let gp_dynamic = collect_regs(&mut next, config.gp_dynamic_budget);
        let gp_allocatable_count = usize::from(config.allocatable_gp_dynamic_budget());
        let first_fp_reg = next;
        let fp_dynamic = collect_regs(&mut next, config.fp_dynamic_budget);

        // Layout: [fixed | gp_dynamic | fp_dynamic]
        Ok(Self {
            gp_dynamic,
            fp_dynamic,
            gp_allocatable_count,
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
    pub(super) fn gp_dynamic(&self, index: usize) -> Option<MachineReg> {
        self.gp_dynamic.get(index).copied()
    }

    #[inline]
    pub(super) fn gp_dynamic_count(&self) -> usize {
        self.gp_dynamic.len()
    }

    #[inline]
    pub(super) fn ordered_gp_dynamic(&self, index: usize) -> Option<MachineReg> {
        self.gp_dynamic.get(index).copied()
    }

    #[inline]
    pub(super) fn gp_allocatable_count(&self) -> usize {
        self.gp_allocatable_count
    }

    #[inline]
    pub(super) fn ordered_gp_allocatable(&self, index: usize) -> Option<MachineReg> {
        (index < self.gp_allocatable_count)
            .then(|| self.gp_dynamic.get(index).copied())
            .flatten()
    }

    #[inline]
    pub(super) fn fp_dynamic(&self, index: usize) -> Option<MachineReg> {
        self.fp_dynamic.get(index).copied()
    }

    #[inline]
    pub(super) fn fp_dynamic_count(&self) -> usize {
        self.fp_dynamic.len()
    }

    #[inline]
    pub(super) fn ordered_fp_dynamic(&self, index: usize) -> Option<MachineReg> {
        self.fp_dynamic.get(index).copied()
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

fn collect_regs(next: &mut u16, count: u8) -> collections::Vec<MachineReg> {
    let mut regs = collections::Vec::with_capacity(count as usize);
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

    /// Allocate a Fill/LocalGet destination in the correct bank based on the type table.
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

    /// Resolve an operand to a machine register.
    ///
    /// - `SsaOperand::Value(v)` → `use_value(v)`
    /// - `SsaOperand::Const(bits)` → allocate a linear-value reg, emit a Move with
    ///   the constant, and return the register.
    pub(super) fn use_operand(&mut self, operand: SsaOperand) -> Result<MachineReg, WasmError> {
        match operand.decode() {
            DecodedOperand::Value(v) => self.use_value(v),
            DecodedOperand::Const(idx) => {
                let bits = self.program().const_pool[idx as usize];
                let ty = MachineStorageType::GpWord;
                let Some(reg) = self.first_free_linear_value_reg(ty) else {
                    return Err(WasmError::internal(
                        "dynamic register budget exhausted while materializing constant operand"
                            .into(),
                    ));
                };
                self.set_linear_value_reg(reg, None, Some(ty))?;
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::LinearValue,
                        ty,
                        dst: reg,
                        src: MachineValue::Imm64(bits),
                    },
                });
                Ok(reg)
            }
            DecodedOperand::None => Err(WasmError::internal(
                "use_operand called with a NONE SsaOperand",
            )),
        }
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
            WasmError::internal("SSA-IR i64 value does not have a paired machine-register mapping")
        })
    }

    /// Resolve an operand into an i64 (lo, hi) machine-value pair.
    ///
    /// - `Value(v)` → call `use_i64_value_pair` → two register values
    /// - `Const(bits)` → split into lo/hi immediates
    pub(super) fn use_i64_operand_pair(
        &mut self,
        operand: &SsaOperand,
    ) -> Result<(MachineValue, MachineValue), WasmError> {
        match operand.decode() {
            DecodedOperand::Value(v) => {
                let (lo, hi) = self.use_i64_value_pair(v)?;
                Ok((MachineValue::Reg(lo), MachineValue::Reg(hi)))
            }
            DecodedOperand::Const(idx) => {
                let bits = self.program().const_pool[idx as usize];
                Ok((
                    MachineValue::Imm64(bits as u32 as u64),
                    MachineValue::Imm64((bits >> 32) as u64),
                ))
            }
            DecodedOperand::None => Err(WasmError::internal(
                "use_i64_operand_pair called with a NONE SsaOperand",
            )),
        }
    }

    pub(super) fn split_continuation_params(
        &self,
        continuation_ops: &[MachineInst],
        continuation_term: &MachineTerminator,
    ) -> collections::Vec<MachineBlockParam> {
        let mut params = collections::Vec::new();
        let mut all_defined = collections::Vec::new();
        for inst in continuation_ops {
            for_each_inst_defined_reg(&inst.kind, |reg| {
                if self.is_linear_value_reg(reg) && !all_defined.contains(&reg) {
                    all_defined.push(reg);
                }
            });
        }

        for entry in self.values_iter() {
            let remaining = self.remaining_use_count(entry.value);
            if remaining != 0
                && self.is_linear_value_reg(entry.reg)
                && !all_defined.contains(&entry.reg)
            {
                push_unique_param(
                    &mut params,
                    machine_block_param(entry.reg, self.storage_type_for_reg(entry.reg)),
                );
                if let Some(hi_reg) = entry.hi_reg {
                    if self.is_linear_value_reg(hi_reg) && !all_defined.contains(&hi_reg) {
                        push_unique_param(
                            &mut params,
                            machine_block_param(hi_reg, self.storage_type_for_reg(hi_reg)),
                        );
                    }
                }
            }
        }

        let mut defined_so_far = collections::Vec::new();
        for inst in continuation_ops {
            visit_inst_source_regs(&inst.kind, |reg| {
                if self.is_linear_value_reg(reg) && !defined_so_far.contains(&reg) {
                    push_unique_param(
                        &mut params,
                        machine_block_param(reg, self.storage_type_for_reg(reg)),
                    );
                }
            });
            for_each_inst_defined_reg(&inst.kind, |dst| {
                if self.is_linear_value_reg(dst) && !defined_so_far.contains(&dst) {
                    defined_so_far.push(dst);
                }
            });
        }
        visit_term_source_regs(continuation_term, |reg| {
            if self.is_linear_value_reg(reg) && !defined_so_far.contains(&reg) {
                push_unique_param(
                    &mut params,
                    machine_block_param(reg, self.storage_type_for_reg(reg)),
                );
            }
        });

        for index in 0..self.cached_locals().len() {
            if !self.is_cache_live(index) {
                continue;
            }
            let Some(bound) = self.bound_cached_local(index) else {
                continue;
            };
            push_unique_param(
                &mut params,
                machine_block_param_with_owner(
                    bound.reg,
                    self.storage_type_for_reg(bound.reg),
                    MachineRegOwner::CachedLocal,
                ),
            );
            if let Some(hi_reg) = bound.hi_reg {
                push_unique_param(
                    &mut params,
                    machine_block_param_with_owner(
                        hi_reg,
                        self.storage_type_for_reg(hi_reg),
                        MachineRegOwner::CachedLocal,
                    ),
                );
            }
        }

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
        let reg = self.try_value_reg(value)?;
        // Cache registers belong to cached locals and must not be reused as
        // scratch — they may still be needed by later source-alias lookups.
        if !self.is_linear_value_reg(reg) {
            return None;
        }
        Some(reg)
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
        self.try_value_reg(value)
            .ok_or_else(|| WasmError::internal("no machine register assigned for SSA-IR value"))
    }

    fn value_regs(&self, value: SsaValue) -> Result<(MachineReg, Option<MachineReg>), WasmError> {
        self.try_value_regs(value).ok_or_else(|| {
            WasmError::internal("no machine register pair assigned for SSA-IR value")
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
        let Some(reg) = self.first_free_linear_value_reg(ty) else {
            return Err(WasmError::internal("prepared SSA-IR exceeded dynamic register budget during native lowering in block b for value"));
        };
        self.push_value_location(value, reg, None);
        self.set_linear_value_reg(reg, Some(value), Some(ty))?;
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
            return Err(WasmError::internal("SSA-IR value already has a scalar machine-register mapping; cannot also allocate a pair"));
        }

        let Some((lo, hi)) = self.first_free_gp_linear_value_pair() else {
            return Err(WasmError::internal("prepared SSA-IR exceeded GP dynamic pair budget during native lowering in block b for value"));
        };
        self.push_value_location(value, lo, Some(hi));
        // Pair-aware 32-bit lowering treats both halves as GP-word registers.
        self.set_linear_value_reg(lo, Some(value), Some(MachineStorageType::GpWord))?;
        self.set_linear_value_reg(hi, Some(value), Some(MachineStorageType::GpWord))?;
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

    pub(super) fn first_free_linear_value_reg(&self, ty: MachineStorageType) -> Option<MachineReg> {
        let regfile = self.regfile();
        let count = if ty.is_fp() {
            regfile.fp_dynamic_count()
        } else {
            regfile.gp_allocatable_count()
        };
        for ordinal in 0..count {
            let reg = if ty.is_fp() {
                preferred_fp_dynamic_reg(regfile, ordinal)?
            } else {
                preferred_gp_dynamic_reg(regfile, ordinal)?
            };
            if self.dynamic_reg_available(reg) {
                return Some(reg);
            }
        }
        None
    }

    pub(super) fn first_free_gp_linear_value_pair(&self) -> Option<(MachineReg, MachineReg)> {
        let regfile = self.regfile();
        let mut first = None;
        for ordinal in 0..regfile.gp_allocatable_count() {
            let reg = preferred_gp_dynamic_reg(regfile, ordinal)?;
            if !self.dynamic_reg_available(reg) {
                continue;
            }
            if let Some(first_reg) = first {
                return Some((first_reg, reg));
            }
            first = Some(reg);
        }
        None
    }

    pub(super) fn set_linear_value_reg(
        &mut self,
        reg: MachineReg,
        value: Option<SsaValue>,
        ty: Option<MachineStorageType>,
    ) -> Result<(), WasmError> {
        let index = self.dynamic_index(reg)?;
        self.set_linear_value_state(index, value, ty)
    }

    pub(super) fn clear_linear_value_reg(&mut self, reg: MachineReg) -> Result<(), WasmError> {
        self.set_linear_value_reg(reg, None, None)
    }

    pub(super) fn is_linear_value_reg(&self, reg: MachineReg) -> bool {
        self.dynamic_index(reg).is_ok() && !self.is_bound_cache_reg(reg)
    }

    pub(super) fn is_fp_reg(&self, reg: MachineReg) -> bool {
        let regfile = self.regfile();
        reg.0 >= regfile.first_fp_reg() && reg.0 < regfile.reg_count()
    }

    pub(super) fn dynamic_index(&self, reg: MachineReg) -> Result<usize, WasmError> {
        let regfile = self.regfile();
        if let Some(first) = regfile.gp_dynamic(0) {
            let start = first.0;
            let end = start + regfile.gp_dynamic_count() as u16;
            if reg.0 >= start && reg.0 < end {
                return Ok((reg.0 - start) as usize);
            }
        }

        if let Some(first) = regfile.fp_dynamic(0) {
            let start = first.0;
            let end = start + regfile.fp_dynamic_count() as u16;
            if reg.0 >= start && reg.0 < end {
                return Ok(regfile.gp_dynamic_count() + (reg.0 - start) as usize);
            }
        }

        Err(WasmError::internal(
            "machine register is not in the dynamic bank".into(),
        ))
    }

    pub(super) fn storage_type_for_reg(&self, reg: MachineReg) -> MachineStorageType {
        if let Ok(index) = self.dynamic_index(reg) {
            return self
                .linear_value_storage_type(index)
                .unwrap_or(MachineStorageType::GpWord);
        }
        self.cached_local_storage_type_for_reg(reg)
            .unwrap_or(MachineStorageType::GpWord)
    }

    /// Reserve a specific GP dynamic lane for structured lowering-internal
    /// control flow. This is the narrow escape hatch for non-SSA use of the
    /// dynamic bank; callers must name the purpose so the exception is
    /// explicit at the call site.
    pub(super) fn reserved_gp_dynamic(
        &self,
        index: usize,
        _purpose: &'static str,
    ) -> Result<MachineReg, WasmError> {
        self.regfile()
            .ordered_gp_dynamic(index)
            .ok_or_else(|| WasmError::internal("native lowering requires GP dynamic register for"))
    }

    pub(super) fn borrow_free_gp_dynamic_regs(
        &self,
        count: usize,
    ) -> Result<collections::Vec<MachineReg>, WasmError> {
        let regfile = self.regfile();
        let mut regs = collections::Vec::with_capacity(count);
        for ordinal in 0..regfile.gp_dynamic_count() {
            let Some(reg) = preferred_gp_dynamic_reg(regfile, ordinal) else {
                continue;
            };
            if self.dynamic_reg_available(reg) {
                regs.push(reg);
                if regs.len() == count {
                    return Ok(regs);
                }
            }
        }
        Err(WasmError::internal(
            "native lowering requires free GP dynamic registers",
        ))
    }

    /// Try to fold a dead constant-producing instruction directly into an
    /// uncached frame store so 32-bit lowering does not keep unnecessary GP
    /// temporaries alive across long argument setup.
    pub(super) fn try_coalesce_last_store_immediate(
        &mut self,
        src_value: SsaValue,
        src_reg: MachineReg,
        ty: MachineStorageType,
        addr: MachineAddr,
        width: MachineMemWidth,
    ) -> bool {
        if !self.is_linear_value_reg(src_reg) {
            return false;
        }
        let remaining = self.remaining_use_count(src_value);
        if remaining != 0 {
            return false;
        }

        let imm = match self.ops_last().map(|inst| &inst.kind) {
            Some(MachineInstKind::Move {
                owner: MachineRegOwner::LinearValue,
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

impl<'a> BlockLowerContext<'a> {
    fn dynamic_reg_available(&self, reg: MachineReg) -> bool {
        self.dynamic_index(reg)
            .ok()
            .map(|index| !self.linear_value_occupied(index) && !self.is_bound_cache_reg(reg))
            .unwrap_or(false)
    }
}

// ---------------------------------------------------------------------------
// Free functions used by register allocation
// ---------------------------------------------------------------------------

pub(super) fn canonical_storage_mem_width(ty: MachineStorageType) -> MachineMemWidth {
    match ty {
        MachineStorageType::GpWord
        | MachineStorageType::GpI64
        | MachineStorageType::Fp64
        | MachineStorageType::V128 => MachineMemWidth::U64,
        MachineStorageType::Fp32 => MachineMemWidth::U32,
    }
}

fn preferred_gp_dynamic_reg(regfile: &MachineRegFile, ordinal: usize) -> Option<MachineReg> {
    regfile.ordered_gp_allocatable(ordinal)
}

fn preferred_fp_dynamic_reg(regfile: &MachineRegFile, ordinal: usize) -> Option<MachineReg> {
    regfile.ordered_fp_dynamic(ordinal)
}

pub(super) fn canonical_cached_local_mem_width(ty: MachineStorageType) -> MachineMemWidth {
    canonical_storage_mem_width(ty)
}

pub(super) fn lir_value_storage_type(program: &SsaProgram, value: SsaValue) -> MachineStorageType {
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

fn push_unique_param(params: &mut collections::Vec<MachineBlockParam>, param: MachineBlockParam) {
    if !params.iter().any(|candidate| candidate.reg == param.reg) {
        params.push(param);
    }
}

pub(super) fn machine_block_params_for_value(
    regs: super::lower_context::ValueRegs,
    ty: MachineStorageType,
) -> collections::Vec<MachineBlockParam> {
    match (ty, regs.hi) {
        (MachineStorageType::GpI64, Some(hi)) => {
            collections::vec![
                MachineBlockParam::gp_word(regs.lo),
                MachineBlockParam::gp_word(hi)
            ]
        }
        _ => collections::vec![machine_block_param(regs.lo, ty)],
    }
}

/// Visit all registers defined by an instruction, including both halves of i64 pair ops.
fn for_each_inst_defined_reg(kind: &MachineInstKind, mut f: impl FnMut(MachineReg)) {
    if let Some(dst) = inst_defined_reg(kind) {
        f(dst);
    }
    match kind {
        MachineInstKind::Int64PairBinary { dst_lo, dst_hi, .. }
        | MachineInstKind::Int64PairUnary { dst_lo, dst_hi, .. }
        | MachineInstKind::Int64PairDivRem { dst_lo, dst_hi, .. }
        | MachineInstKind::Int64PairShift { dst_lo, dst_hi, .. }
        | MachineInstKind::Int64MulFromSignExt32 { dst_lo, dst_hi, .. }
        | MachineInstKind::ConvertFloatToI64Pair { dst_lo, dst_hi, .. }
        | MachineInstKind::ReinterpretF64ToI64Pair { dst_lo, dst_hi, .. }
        | MachineInstKind::StructGet {
            dst: dst_lo,
            dst_hi: Some(dst_hi),
            ..
        }
        | MachineInstKind::ArrayGet {
            dst: dst_lo,
            dst_hi: Some(dst_hi),
            ..
        } => {
            f(*dst_lo);
            f(*dst_hi);
        }
        _ => {}
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
        | MachineInstKind::IndexedLoad { dst, .. }
        | MachineInstKind::BitfieldExtractU { dst, .. }
        | MachineInstKind::IntBinaryShifted { dst, .. }
        | MachineInstKind::TestBits { dst, .. }
        | MachineInstKind::EhAllocExnRef { dst, .. }
        | MachineInstKind::RefFunc { dst, .. }
        | MachineInstKind::RefAsNonNull { dst, .. }
        | MachineInstKind::RefEq { dst, .. }
        | MachineInstKind::RefI31 { dst, .. }
        | MachineInstKind::I31GetS { dst, .. }
        | MachineInstKind::I31GetU { dst, .. }
        | MachineInstKind::AnyConvertExtern { dst, .. }
        | MachineInstKind::ExternConvertAny { dst, .. }
        | MachineInstKind::RefTest { dst, .. }
        | MachineInstKind::RefCast { dst, .. }
        | MachineInstKind::StructNew { dst, .. }
        | MachineInstKind::StructNewDefault { dst, .. }
        | MachineInstKind::ArrayNew { dst, .. }
        | MachineInstKind::ArrayNewDefault { dst, .. }
        | MachineInstKind::ArrayNewFixed { dst, .. }
        | MachineInstKind::ArrayNewData { dst, .. }
        | MachineInstKind::ArrayNewElem { dst, .. }
        | MachineInstKind::ArrayLen { dst, .. } => Some(*dst),
        #[cfg(sf_has_simd)]
        MachineInstKind::V128Const { dst, .. }
        | MachineInstKind::V128FromRaw { dst, .. }
        | MachineInstKind::V128ToRaw { dst, .. }
        | MachineInstKind::SimdUnary { dst, .. }
        | MachineInstKind::SimdBinary { dst, .. }
        | MachineInstKind::SimdTernary { dst, .. }
        | MachineInstKind::SimdShift { dst, .. }
        | MachineInstKind::SimdExtractLane { dst, .. }
        | MachineInstKind::SimdReplaceLane { dst, .. }
        | MachineInstKind::SimdShuffle { dst, .. }
        | MachineInstKind::SimdLoad { dst, .. }
        | MachineInstKind::SimdLoadLane { dst, .. } => Some(*dst),
        MachineInstKind::StructGet { dst, dst_hi, .. }
        | MachineInstKind::ArrayGet { dst, dst_hi, .. } => dst_hi.is_none().then_some(*dst),
        MachineInstKind::MemoryGrow { dst, .. } | MachineInstKind::TableGrow { dst, .. } => {
            Some(*dst)
        }
        MachineInstKind::MemoryFill { .. }
        | MachineInstKind::MemoryCopy { .. }
        | MachineInstKind::MemoryInit { .. }
        | MachineInstKind::DataDrop { .. }
        | MachineInstKind::TableFill { .. }
        | MachineInstKind::TableCopy { .. }
        | MachineInstKind::TableInit { .. }
        | MachineInstKind::ElemDrop { .. }
        | MachineInstKind::StructSet { .. }
        | MachineInstKind::ArraySet { .. }
        | MachineInstKind::ArrayFill { .. }
        | MachineInstKind::ArrayCopy { .. }
        | MachineInstKind::ArrayInitData { .. }
        | MachineInstKind::ArrayInitElem { .. } => None,
        MachineInstKind::Int64PairBinary { .. } => None,
        MachineInstKind::Int64PairUnary { .. } => None,
        MachineInstKind::Int64PairDivRem { .. } => None,
        MachineInstKind::Int64PairShift { .. } => None,
        MachineInstKind::Int64MulFromSignExt32 { .. } => None,
        MachineInstKind::Int64PairCompare { dst, .. } => Some(*dst),
        MachineInstKind::ConvertFloatToI64Pair { .. } => None,
        MachineInstKind::ConvertI64PairToFloat { dst, .. }
        | MachineInstKind::ReinterpretI64PairToF64 { dst, .. } => Some(*dst),
        MachineInstKind::ReinterpretF64ToI64Pair { .. } => None,
        MachineInstKind::Store { .. }
        | MachineInstKind::IndexedStore { .. }
        | MachineInstKind::TrapIf { .. }
        | MachineInstKind::CallRuntime(_)
        | MachineInstKind::EhThrow { .. }
        | MachineInstKind::EhThrowRef { .. } => None,
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdStore { .. } | MachineInstKind::SimdStoreLane { .. } => None,
    }
}

fn visit_inst_source_regs(kind: &MachineInstKind, mut visit: impl FnMut(MachineReg)) {
    match kind {
        MachineInstKind::Move { src, .. } => visit_value_reg(src, &mut visit),
        MachineInstKind::FloatConst { .. } => {}
        #[cfg(sf_has_simd)]
        MachineInstKind::V128Const { .. } => {}
        #[cfg(sf_has_simd)]
        MachineInstKind::V128FromRaw { raw, .. } => visit_value_reg(raw, &mut visit),
        #[cfg(sf_has_simd)]
        MachineInstKind::V128ToRaw { src, .. } => visit_value_reg(src, &mut visit),
        MachineInstKind::Load { addr, .. } => visit(addr.base),
        MachineInstKind::Store { addr, src, .. } => {
            visit(addr.base);
            visit_value_reg(src, &mut visit);
        }
        MachineInstKind::IndexedLoad { base, index, .. } => {
            visit(*base);
            visit(*index);
        }
        MachineInstKind::IndexedStore {
            base, index, src, ..
        } => {
            visit(*base);
            visit(*index);
            visit_value_reg(src, &mut visit);
        }
        MachineInstKind::IntUnary { src, .. }
        | MachineInstKind::FloatUnary { src, .. }
        | MachineInstKind::Convert { src, .. } => visit_value_reg(src, &mut visit),
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdUnary { src, .. } | MachineInstKind::SimdExtractLane { src, .. } => {
            visit_value_reg(src, &mut visit)
        }
        MachineInstKind::IntBinary { lhs, rhs, .. }
        | MachineInstKind::IntCompare { lhs, rhs, .. }
        | MachineInstKind::FloatBinary { lhs, rhs, .. }
        | MachineInstKind::FloatCompare { lhs, rhs, .. } => {
            visit_value_reg(lhs, &mut visit);
            visit_value_reg(rhs, &mut visit);
        }
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdBinary { lhs, rhs, .. } => {
            visit_value_reg(lhs, &mut visit);
            visit_value_reg(rhs, &mut visit);
        }
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdTernary { a, b, c, .. } => {
            visit_value_reg(a, &mut visit);
            visit_value_reg(b, &mut visit);
            visit_value_reg(c, &mut visit);
        }
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdShift { vector, shift, .. } => {
            visit_value_reg(vector, &mut visit);
            visit_value_reg(shift, &mut visit);
        }
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdReplaceLane { vector, scalar, .. } => {
            visit_value_reg(vector, &mut visit);
            visit_value_reg(scalar, &mut visit);
        }
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdShuffle { lhs, rhs, .. } => {
            visit_value_reg(lhs, &mut visit);
            visit_value_reg(rhs, &mut visit);
        }
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdLoad { addr, .. } => visit(addr.base),
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdStore { addr, src, .. } => {
            visit(addr.base);
            visit_value_reg(src, &mut visit);
        }
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdLoadLane { addr, vector, .. } => {
            visit(addr.base);
            visit_value_reg(vector, &mut visit);
        }
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdStoreLane { addr, vector, .. } => {
            visit(addr.base);
            visit_value_reg(vector, &mut visit);
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
        MachineInstKind::Int64MulFromSignExt32 { lhs, rhs, .. } => {
            visit_value_reg(lhs, &mut visit);
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
        MachineInstKind::BitfieldExtractU { src, .. } => visit(*src),
        MachineInstKind::IntBinaryShifted { lhs, rhs, .. } => {
            visit(*lhs);
            visit(*rhs);
        }
        MachineInstKind::TestBits { src, mask, .. } => {
            visit(*src);
            visit_value_reg(mask, &mut visit);
        }
        MachineInstKind::TrapIf { cond, .. } => {
            visit_branch_cond_regs(cond, &mut visit);
        }
        MachineInstKind::CallRuntime(_) => {}
        MachineInstKind::RefFunc { .. } => {}
        MachineInstKind::StructNew { fields, .. } => {
            for (value_lo, value_hi) in fields {
                visit_value_reg(value_lo, &mut visit);
                if let Some(value_hi) = value_hi {
                    visit_value_reg(value_hi, &mut visit);
                }
            }
        }
        MachineInstKind::StructNewDefault { .. } => {}
        MachineInstKind::RefAsNonNull { src, .. }
        | MachineInstKind::RefI31 { src, .. }
        | MachineInstKind::I31GetS { src, .. }
        | MachineInstKind::I31GetU { src, .. }
        | MachineInstKind::AnyConvertExtern { src, .. }
        | MachineInstKind::ExternConvertAny { src, .. }
        | MachineInstKind::RefTest { src, .. }
        | MachineInstKind::RefCast { src, .. }
        | MachineInstKind::StructGet { src, .. }
        | MachineInstKind::ArrayNewDefault { length: src, .. }
        | MachineInstKind::ArrayLen { src, .. } => {
            visit_value_reg(src, &mut visit);
        }
        MachineInstKind::ArrayNew {
            init_lo,
            init_hi,
            length,
            ..
        } => {
            visit_value_reg(init_lo, &mut visit);
            if let Some(init_hi) = init_hi {
                visit_value_reg(init_hi, &mut visit);
            }
            visit_value_reg(length, &mut visit);
        }
        MachineInstKind::ArrayNewFixed { elements, .. } => {
            for (value_lo, value_hi) in elements {
                visit_value_reg(value_lo, &mut visit);
                if let Some(value_hi) = value_hi {
                    visit_value_reg(value_hi, &mut visit);
                }
            }
        }
        MachineInstKind::ArrayNewData { src, len, .. }
        | MachineInstKind::ArrayNewElem { src, len, .. } => {
            visit_value_reg(src, &mut visit);
            visit_value_reg(len, &mut visit);
        }
        MachineInstKind::RefEq { lhs, rhs, .. } => {
            visit_value_reg(lhs, &mut visit);
            visit_value_reg(rhs, &mut visit);
        }
        MachineInstKind::ArrayGet { ref_src, index, .. } => {
            visit_value_reg(ref_src, &mut visit);
            visit_value_reg(index, &mut visit);
        }
        MachineInstKind::ArraySet {
            ref_src,
            index,
            value_lo,
            value_hi,
            ..
        } => {
            visit_value_reg(ref_src, &mut visit);
            visit_value_reg(index, &mut visit);
            visit_value_reg(value_lo, &mut visit);
            if let Some(value_hi) = value_hi {
                visit_value_reg(value_hi, &mut visit);
            }
        }
        MachineInstKind::ArrayFill {
            ref_src,
            index,
            value_lo,
            value_hi,
            len,
            ..
        } => {
            visit_value_reg(ref_src, &mut visit);
            visit_value_reg(index, &mut visit);
            visit_value_reg(value_lo, &mut visit);
            if let Some(value_hi) = value_hi {
                visit_value_reg(value_hi, &mut visit);
            }
            visit_value_reg(len, &mut visit);
        }
        MachineInstKind::ArrayCopy {
            dst_ref,
            dst_index,
            src_ref,
            src_index,
            len,
            ..
        } => {
            visit_value_reg(dst_ref, &mut visit);
            visit_value_reg(dst_index, &mut visit);
            visit_value_reg(src_ref, &mut visit);
            visit_value_reg(src_index, &mut visit);
            visit_value_reg(len, &mut visit);
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
            visit_value_reg(ref_src, &mut visit);
            visit_value_reg(dst_index, &mut visit);
            visit_value_reg(src_index, &mut visit);
            visit_value_reg(len, &mut visit);
        }
        MachineInstKind::StructSet {
            ref_src,
            value_lo,
            value_hi,
            ..
        } => {
            visit_value_reg(ref_src, &mut visit);
            visit_value_reg(value_lo, &mut visit);
            if let Some(value_hi) = value_hi {
                visit_value_reg(value_hi, &mut visit);
            }
        }
        MachineInstKind::MemoryGrow { delta, .. } => {
            visit_value_reg(delta, &mut visit);
        }
        MachineInstKind::MemoryFill { dest, val, len, .. }
        | MachineInstKind::TableFill {
            start: dest,
            val,
            len,
            ..
        } => {
            visit_value_reg(dest, &mut visit);
            visit_value_reg(val, &mut visit);
            visit_value_reg(len, &mut visit);
        }
        MachineInstKind::MemoryCopy { dest, src, len, .. }
        | MachineInstKind::MemoryInit { dest, src, len, .. }
        | MachineInstKind::TableCopy { dest, src, len, .. }
        | MachineInstKind::TableInit { dest, src, len, .. } => {
            visit_value_reg(dest, &mut visit);
            visit_value_reg(src, &mut visit);
            visit_value_reg(len, &mut visit);
        }
        MachineInstKind::TableGrow {
            init_val, delta, ..
        } => {
            visit_value_reg(init_val, &mut visit);
            visit_value_reg(delta, &mut visit);
        }
        MachineInstKind::DataDrop { .. }
        | MachineInstKind::ElemDrop { .. }
        | MachineInstKind::EhThrow { .. }
        | MachineInstKind::EhThrowRef { .. }
        | MachineInstKind::EhAllocExnRef { .. } => {}
    }
}

fn visit_branch_cond_regs(cond: &MachineBranchCond, visit: &mut impl FnMut(MachineReg)) {
    match cond {
        MachineBranchCond::Value(value) => visit_value_reg(value, visit),
        MachineBranchCond::IntCompare { lhs, rhs, .. } => {
            visit_value_reg(lhs, visit);
            visit_value_reg(rhs, visit);
        }
        MachineBranchCond::TestBits { src, mask, .. } => {
            visit_value_reg(src, visit);
            visit_value_reg(mask, visit);
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
        MachineTerminator::Call {
            target,
            callee_frame_base,
            caller_result_base,
            ..
        } => {
            if let MachineCallTarget::Indirect {
                callee_target,
                callee_entry,
            } = target
            {
                visit(*callee_target);
                visit(*callee_entry);
            }
            visit(*callee_frame_base);
            visit(*caller_result_base);
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
                visit(*callee_target);
                visit(*callee_entry);
            }
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

fn machine_block_param(reg: MachineReg, ty: MachineStorageType) -> MachineBlockParam {
    machine_block_param_with_owner(reg, ty, MachineRegOwner::LinearValue)
}

fn machine_block_param_with_owner(
    reg: MachineReg,
    ty: MachineStorageType,
    owner: MachineRegOwner,
) -> MachineBlockParam {
    match ty {
        MachineStorageType::GpWord => MachineBlockParam::gp_word(reg).with_owner(owner),
        MachineStorageType::GpI64 => MachineBlockParam::gp_i64(reg).with_owner(owner),
        MachineStorageType::Fp32 => {
            MachineBlockParam::fp(reg, MachineFloatWidth::F32).with_owner(owner)
        }
        MachineStorageType::Fp64 => {
            MachineBlockParam::fp(reg, MachineFloatWidth::F64).with_owner(owner)
        }
        MachineStorageType::V128 => MachineBlockParam::v128(reg).with_owner(owner),
    }
}

pub(super) fn value_type_storage_type(ty: ValueType) -> MachineStorageType {
    match ty {
        ValueType::F32 => MachineStorageType::Fp32,
        ValueType::F64 => MachineStorageType::Fp64,
        ValueType::I64 => MachineStorageType::GpI64,
        ValueType::V128 => MachineStorageType::V128,
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
    use tracked_alloc::boxed::Box;

    use crate::{
        value_type::ValueType,
        vm::{
            backend::BackendConfig,
            machine::machine_ir::{MachineFunctionAbi, MachineStorageType},
            middle::ssa_ir::{
                ir::{SsaBlock, SsaProgram, SsaTerminator, SsaValue},
                target::SsaTarget,
            },
        },
    };

    use super::*;
    use crate::vm::machine::lower_context::BlockLowerContext;

    fn make_test_context(value_types: collections::Vec<ValueType>) -> BlockLowerContext<'static> {
        let program = Box::leak(Box::new(SsaProgram {
            entry: SsaTarget(0),
            blocks: collections::vec![SsaBlock {
                id: SsaTarget(0),
                params: collections::vec![],
                ops: collections::vec![],
                extra_args: collections::Vec::new(),
                terminator: SsaTerminator::Return { results: None },
            }],
            local_slot_types: collections::Vec::new(),
            local_slot_info: collections::Vec::new(),
            block_entry_cached_slots: collections::vec![collections::vec![]],
            block_cfg_origins: collections::vec![],
            value_types,
            value_sink_local: collections::vec![],
            const_pool: collections::Vec::new(),
            primitive_pool: collections::Vec::new(),
            call_ops: collections::Vec::new(),
            uniform_boundary: false,
        }));
        let regfile = Box::leak(Box::new(
            MachineRegFile::new(BackendConfig::new(4, 5, 0, 8)).expect("regfile"),
        ));
        let runtime = MachineFunctionAbi::default();
        let all_runtime = Box::leak(Box::new(collections::vec![runtime]));
        let explicit_cache = Box::leak(Box::new(
            super::super::lower_context::explicit_cached_locals(program),
        ));
        let entry_cache_params = Box::leak(Box::new(
            super::super::lower_cache_layout::compute_block_entry_cache_params(
                regfile,
                program,
                explicit_cache,
                4,
            )
            .expect("entry cache layouts"),
        ));

        BlockLowerContext::new(
            regfile,
            program,
            explicit_cache,
            entry_cache_params,
            &program.blocks[0],
            all_runtime,
            4,
            &super::super::gp32::Gp32Lowering,
            true,
            None,
            #[cfg(sf_has_guard_pages)]
            false,
        )
        .expect("lower context")
    }

    #[test]
    fn alloc_i64_value_pair_reserves_two_gp_word_linear_value_regs() {
        let mut lower = make_test_context(collections::vec![ValueType::I64]);
        let (lo, hi) = lower.alloc_i64_value_pair(SsaValue(0)).expect("pair alloc");
        assert_ne!(lo, hi);
        assert_eq!(
            lower.use_i64_value_pair(SsaValue(0)).expect("pair use"),
            (lo, hi)
        );
        assert_eq!(lower.storage_type_for_reg(lo), MachineStorageType::GpWord);
        assert_eq!(lower.storage_type_for_reg(hi), MachineStorageType::GpWord);

        let lo_index = lower.dynamic_index(lo).expect("lo linear value");
        let hi_index = lower.dynamic_index(hi).expect("hi linear value");
        assert!(lower.linear_value_occupied(lo_index));
        assert!(lower.linear_value_occupied(hi_index));

        lower.release_dead_values().expect("release pair");
        assert!(lower.try_value_regs(SsaValue(0)).is_none());
        assert!(!lower.linear_value_occupied(lo_index));
        assert!(!lower.linear_value_occupied(hi_index));
    }

    #[test]
    fn scalar_reuse_can_claim_low_half_of_dead_pair_and_frees_high_half() {
        let mut lower = make_test_context(collections::vec![ValueType::I64, ValueType::I32]);
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
        let hi_index = lower.dynamic_index(pair_hi).expect("hi linear value");
        assert!(!lower.linear_value_occupied(hi_index));
    }

    #[test]
    fn pair_reuse_can_claim_low_half_of_dead_scalar_and_allocate_only_high_half() {
        let mut lower = make_test_context(collections::vec![ValueType::I32, ValueType::I64]);
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
