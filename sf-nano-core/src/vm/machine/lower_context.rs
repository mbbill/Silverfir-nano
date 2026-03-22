use alloc::vec::Vec;
use core::mem;

use crate::{
    error::WasmError,
    value_type::ValueType,
    vm::{
        machine::{
            machine_ir::{
                machine_ptr_width, machine_word_int_width, MachineAddr, MachineBlockId,
                MachineBlockParam, MachineBranchCond, MachineCallLinkLayout, MachineCompareKind,
                MachineConvertOp, MachineEdge, MachineFloatBinaryOp, MachineFloatUnaryOp,
                MachineFloatWidth, MachineFrameRegion, MachineFuncId, MachineFunctionRuntime,
                MachineInst, MachineInstKind, MachineIntBinaryOp, MachineIntUnaryOp,
                MachineIntWidth, MachineLoadExtension, MachineMemWidth, MachineReg, MachineSign,
                MachineStorageType, MachineTerminator, MachineTrapKind, MachineValue,
            },
        },
        middle::{
            frame::{FrameLayoutPlan, FrameSlot},
            ssa_ir::{
                ir::{
                    SsaBlock, SsaEdge, SsaInst, SsaInstKind, SsaLocalCachePrefs, SsaProgram,
                    SsaTerminator, SsaValue,
                },
                leaf::SsaLeafOp,
            },
        },
        runtime::layout::{native_runtime_abi_layout, NativeRuntimeAbiLayout},
    },
};

use super::{
    lower_module::{slot_offset_bytes, target_param_regs},
    lower_regalloc::{
        canonical_cached_local_mem_width, canonical_value_mem_width_for_value, gp_reg_int_width,
        gp_reg_mem_width, lir_value_storage_type, value_type_storage_type, MachineRegFile,
    },
    lower_util::compute_remaining_uses,
};

use crate::vm::middle::ssa_ir::ir::CachedLocalInfo;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CachedLocal {
    pub slot: FrameSlot,
    pub reg: MachineReg,
    pub hi_reg: Option<MachineReg>,
    pub ty: MachineStorageType,
    pub info: CachedLocalInfo,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ValueRegs {
    pub lo: MachineReg,
    pub hi: Option<MachineReg>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ValueLocation {
    pub value: SsaValue,
    pub reg: MachineReg,
    pub hi_reg: Option<MachineReg>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct TransientState {
    value: Option<SsaValue>,
    ty: Option<MachineStorageType>,
}

pub(super) struct BlockLowerContext<'a> {
    regfile: &'a MachineRegFile,
    frame: FrameLayoutPlan,
    program: &'a SsaProgram,
    block: &'a SsaBlock,
    runtime: MachineFunctionRuntime,
    all_runtime: &'a [MachineFunctionRuntime],
    call_link: MachineCallLinkLayout,
    machine_params: Vec<ValueRegs>,
    gp_reg_width: u8,
    ops: Vec<MachineInst>,
    cached_locals: Vec<CachedLocal>,
    values: Vec<ValueLocation>,
    remaining_uses: alloc::collections::BTreeMap<SsaValue, u32>,
    transient_state: Vec<TransientState>,
    #[cfg(has_guard_pages)]
    guard_pages: bool,
}

impl<'a> BlockLowerContext<'a> {
    pub(super) fn new(
        regfile: &'a MachineRegFile,
        frame: FrameLayoutPlan,
        program: &'a SsaProgram,
        cache_prefs: &SsaLocalCachePrefs,
        block: &'a SsaBlock,
        runtime: MachineFunctionRuntime,
        all_runtime: &'a [MachineFunctionRuntime],
        call_link: MachineCallLinkLayout,
        gp_reg_width: u8,
        is_entry: bool,
        #[cfg(has_guard_pages)] guard_pages: bool,
    ) -> Result<Self, WasmError> {
        if cache_prefs.gp_preferred_slots.len() != cache_prefs.gp_preferred_types.len() {
            return Err(WasmError::internal(
                "GP cached-local slot/type metadata length mismatch".into(),
            ));
        }
        let machine_params = target_param_regs(&block.params, program, regfile, gp_reg_width)?;
        let mut cached_locals = Vec::new();
        let mut gp_cache_index = 0usize;
        for (index, slot) in cache_prefs.gp_preferred_slots.iter().copied().enumerate() {
            let Some(reg) = regfile.gp_local_cache(gp_cache_index) else {
                break;
            };
            let ty = cache_prefs
                .gp_preferred_types
                .get(index)
                .copied()
                .map(value_type_storage_type)
                .ok_or_else(|| {
                    WasmError::internal(alloc::format!(
                        "GP cached local {:?} is missing a type entry",
                        slot
                    ))
                })?;
            if ty.is_fp() {
                return Err(WasmError::internal(alloc::format!(
                    "GP cached local {:?} must not have float storage type {:?}",
                    slot,
                    ty
                )));
            }
            let info = cache_prefs
                .gp_local_info
                .get(index)
                .copied()
                .unwrap_or_default();
            let hi_reg = if gp_reg_width == 4 && matches!(ty, MachineStorageType::GpI64) {
                Some(regfile.gp_local_cache(gp_cache_index + 1).ok_or_else(|| {
                    WasmError::internal(alloc::format!(
                        "GP cached local {:?} requires a second cache register on 32-bit targets",
                        slot
                    ))
                })?)
            } else {
                None
            };
            cached_locals.push(CachedLocal {
                slot,
                reg,
                hi_reg,
                ty,
                info,
            });
            gp_cache_index += if hi_reg.is_some() { 2 } else { 1 };
        }
        for (index, slot) in cache_prefs.fp_preferred_slots.iter().copied().enumerate() {
            let Some(reg) = regfile.fp_local_cache(index) else {
                break;
            };
            let ty = cache_prefs
                .fp_preferred_types
                .get(index)
                .copied()
                .map(value_type_storage_type)
                .ok_or_else(|| {
                    WasmError::internal(alloc::format!(
                        "FP cached local {:?} is missing a float type entry",
                        slot
                    ))
                })?;
            let Some(_width) = ty.float_width() else {
                return Err(WasmError::internal(alloc::format!(
                    "FP cached local {:?} must have float storage type, got {:?}",
                    slot,
                    ty
                )));
            };
            let info = cache_prefs
                .fp_local_info
                .get(index)
                .copied()
                .unwrap_or_default();
            cached_locals.push(CachedLocal {
                slot,
                reg,
                hi_reg: None,
                ty,
                info,
            });
        }

        let mut lower = Self {
            regfile,
            frame,
            program,
            block,
            runtime,
            all_runtime,
            call_link,
            machine_params,
            gp_reg_width,
            ops: Vec::new(),
            cached_locals,
            values: Vec::new(),
            remaining_uses: compute_remaining_uses(block),
            transient_state: alloc::vec![
                TransientState::default();
                regfile.gp_transient_count() + regfile.fp_transient_count()
            ],
            #[cfg(has_guard_pages)]
            guard_pages,
        };

        let machine_params = lower.machine_params.clone();
        for (param, regs) in block
            .params
            .iter()
            .copied()
            .zip(machine_params.iter().copied())
        {
            lower.values.push(ValueLocation {
                value: param,
                reg: regs.lo,
                hi_reg: regs.hi,
            });
            let ty = lir_value_storage_type(lower.program, param);
            if lower.gp_reg_width == 4 && matches!(ty, MachineStorageType::GpI64) {
                lower.set_transient(regs.lo, Some(param), Some(MachineStorageType::GpWord))?;
                if let Some(hi) = regs.hi {
                    lower.set_transient(hi, Some(param), Some(MachineStorageType::GpWord))?;
                }
            } else {
                lower.set_transient(regs.lo, Some(param), Some(ty))?;
            }
        }
        lower.release_dead_values()?;

        if is_entry {
            lower.emit_entry_cached_locals()?;
        }

        Ok(lower)
    }

    pub(super) fn machine_params(&self) -> &[ValueRegs] {
        &self.machine_params
    }

    pub(super) fn take_ops(&mut self) -> Vec<MachineInst> {
        mem::take(&mut self.ops)
    }

    // -----------------------------------------------------------------------
    // Address / frame helpers
    // -----------------------------------------------------------------------

    pub(super) fn cached_local_index(&self, slot: FrameSlot) -> Option<usize> {
        self.cached_locals
            .iter()
            .position(|cached| cached.slot == slot)
    }

    pub(super) fn frame_addr(&self, slot: FrameSlot) -> Result<MachineAddr, WasmError> {
        self.frame_addr_from(self.regfile.frame_base(), slot)
    }

    pub(super) fn frame_addr_offset(
        &self,
        slot: FrameSlot,
        byte_offset: i32,
    ) -> Result<MachineAddr, WasmError> {
        let mut addr = self.frame_addr(slot)?;
        addr.offset = addr
            .offset
            .checked_add(byte_offset)
            .ok_or_else(|| WasmError::internal("frame byte offset overflow".into()))?;
        Ok(addr)
    }

    pub(super) fn runtime_addr(&self, offset: u32) -> MachineAddr {
        MachineAddr {
            base: self.regfile.runtime_base(),
            offset: offset as i32,
        }
    }

    pub(super) fn frame_addr_from(
        &self,
        base: MachineReg,
        slot: FrameSlot,
    ) -> Result<MachineAddr, WasmError> {
        Ok(MachineAddr {
            base,
            offset: slot_offset_bytes(slot)?,
        })
    }

    pub(super) fn frame_region_addr(
        &self,
        base: MachineReg,
        region: MachineFrameRegion,
        byte_offset: i32,
    ) -> Result<MachineAddr, WasmError> {
        let region_offset = slot_offset_bytes(FrameSlot(region.base_slot))?;
        let offset = region_offset
            .checked_add(byte_offset)
            .ok_or_else(|| WasmError::internal("frame region byte offset overflow".into()))?;
        Ok(MachineAddr { base, offset })
    }

    pub(super) fn runtime_for_func(
        &self,
        func: MachineFuncId,
    ) -> Result<MachineFunctionRuntime, WasmError> {
        self.all_runtime
            .get(func.0 as usize)
            .copied()
            .ok_or_else(|| {
                WasmError::internal("machine runtime metadata missing for callee".into())
            })
    }

    // -----------------------------------------------------------------------
    // Canonical width / storage helpers
    // -----------------------------------------------------------------------

    pub(super) fn canonical_value_mem_width_for_value(&self, value: SsaValue) -> MachineMemWidth {
        canonical_value_mem_width_for_value(self.program, value)
    }

    pub(super) fn canonical_gp_word_mem_width(&self) -> MachineMemWidth {
        super::lower_regalloc::canonical_storage_mem_width(MachineStorageType::GpWord)
    }

    pub(super) fn value_storage_type(&self, value: SsaValue) -> MachineStorageType {
        lir_value_storage_type(self.program, value)
    }

    // -----------------------------------------------------------------------
    // Accessors and emit helpers
    // -----------------------------------------------------------------------

    pub(super) fn ensure_no_live_values(&self, message: &'static str) -> Result<(), WasmError> {
        if self.values.is_empty() {
            Ok(())
        } else {
            Err(WasmError::internal(message.into()))
        }
    }

    #[cfg(has_guard_pages)]
    pub(super) fn use_guard_pages(&self) -> bool {
        self.guard_pages
    }

    pub(super) fn current_runtime(&self) -> MachineFunctionRuntime {
        self.runtime
    }

    pub(super) fn call_link_layout(&self) -> MachineCallLinkLayout {
        self.call_link
    }

    pub(super) fn frame_base_reg(&self) -> MachineReg {
        self.regfile.frame_base()
    }

    pub(super) fn runtime_base_reg(&self) -> MachineReg {
        self.regfile.runtime_base()
    }

    pub(super) fn temp_reg(&self, index: usize) -> Result<MachineReg, WasmError> {
        self.borrow_free_transients(index + 1)?
            .get(index)
            .copied()
            .ok_or_else(|| {
                WasmError::internal("native lowering requires one free transient register".into())
            })
    }

    pub(super) fn mem0_base_reg(&self) -> MachineReg {
        self.regfile.mem0_base()
    }

    pub(super) fn mem0_size_reg(&self) -> MachineReg {
        self.regfile.mem0_size()
    }

    pub(super) fn gp_reg_width(&self) -> u8 {
        self.gp_reg_width
    }

    pub(super) fn runtime_abi_layout(&self) -> NativeRuntimeAbiLayout {
        native_runtime_abi_layout(self.gp_reg_width)
    }

    pub(super) fn gp_word_mem_width(&self) -> MachineMemWidth {
        gp_reg_mem_width(self.gp_reg_width)
    }

    pub(super) fn gp_word_int_width(&self) -> MachineIntWidth {
        gp_reg_int_width(self.gp_reg_width)
    }

    pub(super) fn gp_word_max_imm(&self) -> u64 {
        match self.gp_reg_width {
            4 => u32::MAX as u64,
            8 => u64::MAX,
            other => panic!("unsupported GP register width {other}"),
        }
    }

    pub(super) fn emit_machine_inst(&mut self, inst: MachineInst) {
        self.ops.push(inst);
    }

    pub(super) fn emit_machine_ops<I>(&mut self, insts: I)
    where
        I: IntoIterator<Item = MachineInst>,
    {
        self.ops.extend(insts);
    }

    // -----------------------------------------------------------------------
    // Internal accessors used by sibling impl blocks in other files
    // -----------------------------------------------------------------------

    pub(super) fn regfile(&self) -> &MachineRegFile {
        self.regfile
    }

    pub(super) fn program(&self) -> &SsaProgram {
        self.program
    }

    pub(super) fn block(&self) -> &SsaBlock {
        self.block
    }

    pub(super) fn block_id(&self) -> u32 {
        self.block.id.0
    }

    pub(super) fn cached_locals(&self) -> &[CachedLocal] {
        &self.cached_locals
    }

    pub(super) fn values_iter(&self) -> core::slice::Iter<'_, ValueLocation> {
        self.values.iter()
    }

    pub(super) fn remaining_use_count(&self, value: SsaValue) -> u32 {
        self.remaining_uses.get(&value).copied().unwrap_or(0)
    }

    pub(super) fn remaining_uses_mut(
        &mut self,
    ) -> &mut alloc::collections::BTreeMap<SsaValue, u32> {
        &mut self.remaining_uses
    }

    pub(super) fn transient_count(&self) -> usize {
        self.transient_state.len()
    }

    pub(super) fn transient_occupied(&self, index: usize) -> bool {
        self.transient_state
            .get(index)
            .map(|state| state.value.is_some())
            .unwrap_or(false)
    }

    pub(super) fn transient_state_ty(&self, index: usize) -> Option<MachineStorageType> {
        self.transient_state.get(index).and_then(|state| state.ty)
    }

    pub(super) fn set_transient_state(
        &mut self,
        index: usize,
        value: Option<SsaValue>,
        ty: Option<MachineStorageType>,
    ) -> Result<(), WasmError> {
        let slot = self.transient_state.get_mut(index).ok_or_else(|| {
            WasmError::internal("transient register index is out of range".into())
        })?;
        *slot = TransientState { value, ty };
        Ok(())
    }

    pub(super) fn push_value_location(
        &mut self,
        value: SsaValue,
        reg: MachineReg,
        hi_reg: Option<MachineReg>,
    ) {
        self.values.push(ValueLocation {
            value,
            reg,
            hi_reg,
        });
    }

    /// Free transient registers for values with no remaining uses.
    pub(super) fn release_dead_values(&mut self) -> Result<(), WasmError> {
        let mut index = 0;
        while index < self.values.len() {
            let value = self.values[index].value;
            let remaining = self.remaining_uses.get(&value).copied().unwrap_or(0);
            if remaining == 0 {
                let reg = self.values[index].reg;
                let hi_reg = self.values[index].hi_reg;
                self.values.swap_remove(index);
                self.clear_transient(reg)?;
                if let Some(hi_reg) = hi_reg {
                    self.clear_transient(hi_reg)?;
                }
            } else {
                index += 1;
            }
        }
        Ok(())
    }

    pub(super) fn cached_local_storage_type_for_reg(
        &self,
        reg: MachineReg,
    ) -> Option<MachineStorageType> {
        self.cached_locals.iter().find_map(|cached| {
            if cached.reg == reg || cached.hi_reg == Some(reg) {
                Some(if cached.hi_reg.is_some() {
                    MachineStorageType::GpWord
                } else {
                    cached.ty
                })
            } else {
                None
            }
        })
    }

    /// Try to reuse a dead pair candidate's registers for a new i64 value.
    pub(super) fn try_reuse_pair_candidate(
        &mut self,
        value: SsaValue,
        candidate: SsaValue,
    ) -> Result<Option<(MachineReg, MachineReg)>, WasmError> {
        if let Some(index) = self
            .values
            .iter()
            .position(|entry| entry.value == candidate && entry.hi_reg.is_some())
        {
            let lo = self.values[index].reg;
            let hi = self.values[index]
                .hi_reg
                .expect("pair candidate must have hi reg");
            self.values[index].value = value;
            self.set_transient(lo, Some(value), Some(MachineStorageType::GpWord))?;
            self.set_transient(hi, Some(value), Some(MachineStorageType::GpWord))?;
            return Ok(Some((lo, hi)));
        }
        Ok(None)
    }

    /// Try to reuse a dead scalar candidate's register as the lo half of a new pair.
    pub(super) fn try_reuse_scalar_for_pair(
        &mut self,
        value: SsaValue,
        candidate: SsaValue,
    ) -> Result<Option<(MachineReg, MachineReg)>, WasmError> {
        if let Some(index) = self
            .values
            .iter()
            .position(|entry| entry.value == candidate && entry.hi_reg.is_none())
        {
            let lo = self.values[index].reg;
            if self.is_fp_reg(lo) {
                return Ok(None);
            }
            let Some(hi) = self.first_free_transient(MachineStorageType::GpWord) else {
                return Err(WasmError::internal(alloc::format!(
                    "prepared LIR exceeded GP transient pair budget during native lowering in block b{} for value {}",
                    self.block.id.0,
                    value.0,
                )));
            };
            self.values[index].value = value;
            self.values[index].hi_reg = Some(hi);
            self.set_transient(lo, Some(value), Some(MachineStorageType::GpWord))?;
            self.set_transient(hi, Some(value), Some(MachineStorageType::GpWord))?;
            return Ok(Some((lo, hi)));
        }
        Ok(None)
    }

    /// Try to reuse a dead scalar candidate's register for a new scalar value.
    pub(super) fn try_reuse_scalar_candidate(
        &mut self,
        value: SsaValue,
        candidate: SsaValue,
        ty: MachineStorageType,
    ) -> Result<Option<MachineReg>, WasmError> {
        if let Some(index) = self.values.iter().position(|entry| {
            entry.value == candidate && self.is_fp_reg(entry.reg) == ty.is_fp()
        }) {
            let reg = self.values[index].reg;
            if let Some(hi_reg) = self.values[index].hi_reg {
                if ty.is_fp() {
                    return Ok(None);
                }
                self.clear_transient(hi_reg)?;
                self.values[index].hi_reg = None;
            }
            self.values[index].value = value;
            self.set_transient(reg, Some(value), Some(ty))?;
            return Ok(Some(reg));
        }
        Ok(None)
    }

    pub(super) fn ops_last_mut(&mut self) -> Option<&mut MachineInst> {
        self.ops.last_mut()
    }

    pub(super) fn ops_last(&self) -> Option<&MachineInst> {
        self.ops.last()
    }

    pub(super) fn ops_pop(&mut self) -> Option<MachineInst> {
        self.ops.pop()
    }
}
