use crate::collections;
use tracked_alloc::collections::BTreeMap;

use core::mem;

use crate::{
    error::WasmError,
    value_type::ValueType,
    vm::{
        machine::machine_ir::{
            MachineAddr, MachineFuncId, MachineFunctionAbi, MachineInst, MachineInstKind,
            MachineIntWidth, MachineLoadExtension, MachineMemWidth, MachineReg, MachineRegOwner,
            MachineStorageType, MachineValue,
        },
        middle::{
            frame::FrameSlot,
            ssa_ir::ir::{
                LocalSlotInfo, SsaBlock, SsaInstView, SsaProgram, SsaTerminator, SsaValue,
            },
        },
        runtime::layout::{native_runtime_abi_layout, NativeRuntimeAbiLayout},
    },
};

use super::{
    lower_i64::I64Lowering,
    lower_module::{slot_offset_bytes, target_param_regs},
    lower_regalloc::{
        canonical_cached_local_mem_width, canonical_value_mem_width_for_value, gp_reg_int_width,
        gp_reg_mem_width, lir_value_storage_type, value_type_storage_type, MachineRegFile,
    },
    lower_util::compute_remaining_uses,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CachedLocal {
    pub slot: FrameSlot,
    pub ty: MachineStorageType,
    pub info: LocalSlotInfo,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CachedLocalBinding {
    pub reg: MachineReg,
    pub hi_reg: Option<MachineReg>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BoundCachedLocal {
    pub slot: FrameSlot,
    pub reg: MachineReg,
    pub hi_reg: Option<MachineReg>,
    pub ty: MachineStorageType,
    pub info: LocalSlotInfo,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ValueRegs {
    pub lo: MachineReg,
    pub hi: Option<MachineReg>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct EntryCacheParam {
    pub cached_index: u16,
    pub regs: ValueRegs,
    /// True when the target block entry needs the actual local value already
    /// materialized in the cache register. False means the lane is reserved
    /// for a write-first cached local and may arrive without a valid value.
    pub needs_value: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ValueLocation {
    pub value: SsaValue,
    pub reg: MachineReg,
    pub hi_reg: Option<MachineReg>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct LinearValueState {
    value: Option<SsaValue>,
    ty: Option<MachineStorageType>,
}

pub(super) struct BlockLowerContext<'a> {
    regfile: &'a MachineRegFile,
    program: &'a SsaProgram,
    block: &'a SsaBlock,
    all_runtime: &'a [MachineFunctionAbi],
    machine_params: collections::Vec<ValueRegs>,
    entry_cache_params: collections::Vec<EntryCacheParam>,
    all_entry_cache_params: &'a [collections::Vec<EntryCacheParam>],
    gp_reg_width: u8,
    i64_ops: &'static dyn I64Lowering,
    ops: collections::Vec<MachineInst>,
    cached_locals: collections::Vec<CachedLocal>,
    cache_bindings: collections::Vec<Option<CachedLocalBinding>>,
    /// Per cached-local dirty bit: `true` means the register has been written
    /// since the last call save. Only dirty locals need saving before the next
    /// call. Entry blocks start clean; non-entry blocks receive their carried
    /// dirty state from cross-block analysis.
    cache_live: collections::Vec<bool>,
    /// Per cached-local validity bit: `true` means the bound cache register
    /// currently holds the logical local value. `false` means the cache lane
    /// is merely reserved for a write-first local and must not be threaded as
    /// a real incoming edge value.
    cache_has_value: collections::Vec<bool>,
    cache_dirty: collections::Vec<bool>,
    values: collections::Vec<ValueLocation>,
    remaining_uses: tracked_alloc::collections::BTreeMap<SsaValue, u32>,
    /// Dynamic-register occupancy for linear SSA-like values.
    ///
    /// Cached locals are tracked separately through `cache_bindings`. A dynamic
    /// register is therefore in exactly one of three semantic states at any
    /// program point:
    /// - free
    /// - occupied by one linear value
    /// - bound to one cached local
    linear_value_state: collections::Vec<LinearValueState>,
    #[cfg(sf_has_guard_pages)]
    guard_pages: bool,
}

#[derive(Clone, Copy, Debug)]
struct CachePrefEntry {
    ty: ValueType,
    info: LocalSlotInfo,
}

#[derive(Clone, Copy, Debug)]
struct ExplicitCachedLocalEntry {
    order: usize,
    cached: CachedLocal,
    typed: bool,
}

impl<'a> BlockLowerContext<'a> {
    pub(super) fn new(
        regfile: &'a MachineRegFile,
        program: &'a SsaProgram,
        cached_locals: &'a [CachedLocal],
        all_entry_cache_params: &'a [collections::Vec<EntryCacheParam>],
        block: &'a SsaBlock,
        all_runtime: &'a [MachineFunctionAbi],
        gp_reg_width: u8,
        i64_ops: &'static dyn I64Lowering,
        is_entry: bool,
        initial_cache_dirty: Option<&[bool]>,
        #[cfg(sf_has_guard_pages)] guard_pages: bool,
    ) -> Result<Self, WasmError> {
        let machine_params = target_param_regs(&block.params, program, regfile, gp_reg_width)?;
        let entry_cache_params = all_entry_cache_params
            .get(block.id.as_usize())
            .cloned()
            .unwrap_or_default();
        let cache_live = collections::vec![false; cached_locals.len()];
        let cache_has_value = collections::vec![false; cached_locals.len()];
        let cache_dirty = collections::vec![false; cached_locals.len()];
        let mut lower = Self {
            regfile,
            program,
            block,
            all_runtime,
            machine_params,
            entry_cache_params,
            all_entry_cache_params,
            gp_reg_width,
            i64_ops,
            ops: collections::Vec::new(),
            cached_locals: cached_locals.to_vec().into(),
            cache_bindings: collections::vec![None; cache_live.len()],
            cache_live,
            cache_has_value,
            cache_dirty,
            values: collections::Vec::new(),
            remaining_uses: compute_remaining_uses(block, program),
            linear_value_state: collections::vec![
                LinearValueState::default();
                regfile.gp_dynamic_count() + regfile.fp_dynamic_count()
            ],
            #[cfg(sf_has_guard_pages)]
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
                lower.set_linear_value_reg(
                    regs.lo,
                    Some(param),
                    Some(MachineStorageType::GpWord),
                )?;
                if let Some(hi) = regs.hi {
                    lower.set_linear_value_reg(
                        hi,
                        Some(param),
                        Some(MachineStorageType::GpWord),
                    )?;
                }
            } else {
                lower.set_linear_value_reg(regs.lo, Some(param), Some(ty))?;
            }
        }
        let entry_cache_params = lower.entry_cache_params.clone();
        for entry in entry_cache_params {
            let cached_index = usize::from(entry.cached_index);
            if is_entry {
                if entry.needs_value {
                    lower.materialize_entry_cached_local(cached_index, entry.regs)?;
                } else {
                    lower.bind_cached_local_to_regs(cached_index, entry.regs.lo, entry.regs.hi)?;
                    lower.set_cache_live(cached_index, true);
                    lower.set_cache_has_value(cached_index, false);
                    lower.set_cache_dirty(cached_index, false);
                }
            } else {
                lower.bind_cached_local_to_regs(cached_index, entry.regs.lo, entry.regs.hi)?;
                lower.set_cache_live(cached_index, true);
                lower.set_cache_has_value(cached_index, entry.needs_value);
                let initial_dirty = if entry.needs_value {
                    initial_cache_dirty
                        .and_then(|bits| bits.get(cached_index))
                        .copied()
                        .unwrap_or(true)
                } else {
                    false
                };
                lower.set_cache_dirty(cached_index, initial_dirty);
            }
        }
        lower.release_dead_values()?;

        Ok(lower)
    }

    pub(super) fn machine_params(&self) -> &[ValueRegs] {
        &self.machine_params
    }

    pub(super) fn entry_cache_params(&self) -> &[EntryCacheParam] {
        &self.entry_cache_params
    }

    pub(super) fn block_entry_cache_params(&self, block_id: u32) -> &[EntryCacheParam] {
        self.all_entry_cache_params
            .get(block_id as usize)
            .map(|params| params.as_slice())
            .unwrap_or(&[])
    }

    pub(super) fn take_ops(&mut self) -> collections::Vec<MachineInst> {
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
            .ok_or_else(|| WasmError::internal("frame byte offset overflow"))?;
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

    pub(super) fn runtime_for_func(
        &self,
        func: MachineFuncId,
    ) -> Result<&MachineFunctionAbi, WasmError> {
        self.all_runtime
            .get(func.0 as usize)
            .ok_or_else(|| WasmError::internal("machine runtime metadata missing for callee"))
    }

    // -----------------------------------------------------------------------
    // Canonical width / storage helpers
    // -----------------------------------------------------------------------

    pub(super) fn canonical_value_mem_width_for_value(&self, value: SsaValue) -> MachineMemWidth {
        canonical_value_mem_width_for_value(self.program, value)
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
            Err(WasmError::internal(message))
        }
    }

    #[cfg(sf_has_guard_pages)]
    pub(super) fn use_guard_pages(&self) -> bool {
        self.guard_pages
    }

    pub(super) fn frame_base_reg(&self) -> MachineReg {
        self.regfile.frame_base()
    }

    pub(super) fn runtime_base_reg(&self) -> MachineReg {
        self.regfile.runtime_base()
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

    pub(super) fn i64_ops(&self) -> &'static dyn I64Lowering {
        self.i64_ops
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

    pub(super) fn cached_locals(&self) -> &[CachedLocal] {
        &self.cached_locals
    }

    pub(super) fn bound_cached_local(&self, index: usize) -> Option<BoundCachedLocal> {
        let cached = self.cached_locals.get(index).copied()?;
        let binding = self.cache_bindings.get(index).copied().flatten()?;
        Some(BoundCachedLocal {
            slot: cached.slot,
            reg: binding.reg,
            hi_reg: binding.hi_reg,
            ty: cached.ty,
            info: cached.info,
        })
    }

    pub(super) fn ensure_bound_cached_local(
        &mut self,
        index: usize,
    ) -> Result<BoundCachedLocal, WasmError> {
        if self.cache_bindings.get(index).copied().flatten().is_none() {
            let preferred = if let Some(preferred) = self.preferred_cache_binding(index) {
                preferred
            } else {
                self.allocate_cache_binding(index)?
            };
            let slot = self
                .cache_bindings
                .get_mut(index)
                .ok_or_else(|| WasmError::internal("cached local binding is out of range"))?;
            *slot = Some(preferred);
        }
        self.bound_cached_local(index)
            .ok_or_else(|| WasmError::internal("cached local binding missing after assignment"))
    }

    pub(super) fn bind_cached_local_to_regs(
        &mut self,
        index: usize,
        reg: MachineReg,
        hi_reg: Option<MachineReg>,
    ) -> Result<BoundCachedLocal, WasmError> {
        let slot = self
            .cache_bindings
            .get_mut(index)
            .ok_or_else(|| WasmError::internal("cached local binding is out of range"))?;
        *slot = Some(CachedLocalBinding { reg, hi_reg });
        self.bound_cached_local(index)
            .ok_or_else(|| WasmError::internal("cached local binding missing after assignment"))
    }

    pub(super) fn is_cache_live(&self, index: usize) -> bool {
        self.cache_live.get(index).copied().unwrap_or(false)
    }

    pub(super) fn set_cache_live(&mut self, index: usize, live: bool) {
        if index < self.cache_live.len() {
            self.cache_live[index] = live;
        }
    }

    pub(super) fn cache_has_value(&self, index: usize) -> bool {
        self.cache_has_value.get(index).copied().unwrap_or(false)
    }

    pub(super) fn set_cache_has_value(&mut self, index: usize, has_value: bool) {
        if index < self.cache_has_value.len() {
            self.cache_has_value[index] = has_value;
        }
    }

    pub(super) fn clear_cache_binding(&mut self, index: usize) {
        if let Some(binding) = self.cache_bindings.get_mut(index) {
            *binding = None;
        }
    }

    pub(super) fn clear_cache_live(&mut self) {
        for live in &mut self.cache_live {
            *live = false;
        }
        for has_value in &mut self.cache_has_value {
            *has_value = false;
        }
        for binding in &mut self.cache_bindings {
            *binding = None;
        }
    }

    /// Mark a cached local as dirty (register was written by a LocalSet).
    pub(super) fn mark_cache_dirty(&mut self, index: usize) {
        if index < self.cache_dirty.len() {
            self.cache_dirty[index] = true;
        }
    }

    pub(super) fn set_cache_dirty(&mut self, index: usize, dirty: bool) {
        if index < self.cache_dirty.len() {
            self.cache_dirty[index] = dirty;
        }
    }

    /// Check if a cached local is dirty.
    pub(super) fn is_cache_dirty(&self, index: usize) -> bool {
        self.cache_dirty.get(index).copied().unwrap_or(true)
    }

    /// Clear all dirty flags (called after saving all dirty locals).
    pub(super) fn clear_cache_dirty(&mut self) {
        for d in &mut self.cache_dirty {
            *d = false;
        }
    }

    pub(super) fn values_iter(&self) -> core::slice::Iter<'_, ValueLocation> {
        self.values.iter()
    }

    pub(super) fn remaining_use_count(&self, value: SsaValue) -> u32 {
        self.remaining_uses.get(&value).copied().unwrap_or(0)
    }

    pub(super) fn remaining_uses_mut(
        &mut self,
    ) -> &mut tracked_alloc::collections::BTreeMap<SsaValue, u32> {
        &mut self.remaining_uses
    }

    pub(super) fn linear_value_occupied(&self, index: usize) -> bool {
        self.linear_value_state
            .get(index)
            .map(|state| state.value.is_some())
            .unwrap_or(false)
    }

    pub(super) fn linear_value_storage_type(&self, index: usize) -> Option<MachineStorageType> {
        self.linear_value_state
            .get(index)
            .and_then(|state| state.ty)
    }

    pub(super) fn set_linear_value_state(
        &mut self,
        index: usize,
        value: Option<SsaValue>,
        ty: Option<MachineStorageType>,
    ) -> Result<(), WasmError> {
        let slot = self
            .linear_value_state
            .get_mut(index)
            .ok_or_else(|| WasmError::internal("linear-value register index is out of range"))?;
        *slot = LinearValueState { value, ty };
        Ok(())
    }

    pub(super) fn push_value_location(
        &mut self,
        value: SsaValue,
        reg: MachineReg,
        hi_reg: Option<MachineReg>,
    ) {
        self.values.push(ValueLocation { value, reg, hi_reg });
    }

    /// Free linear-value registers for values with no remaining uses.
    /// Values in non-linear registers (e.g. cache registers from source
    /// aliasing or sink allocation) are removed from the value list but their
    /// registers are not cleared — they belong to the cached local.
    pub(super) fn release_dead_values(&mut self) -> Result<(), WasmError> {
        let mut index = 0;
        while index < self.values.len() {
            let value = self.values[index].value;
            let remaining = self.remaining_uses.get(&value).copied().unwrap_or(0);
            if remaining == 0 {
                let reg = self.values[index].reg;
                let hi_reg = self.values[index].hi_reg;
                self.values.swap_remove(index);
                if self.is_linear_value_reg(reg) {
                    self.clear_linear_value_reg(reg)?;
                }
                if let Some(hi_reg) = hi_reg {
                    if self.is_linear_value_reg(hi_reg) {
                        self.clear_linear_value_reg(hi_reg)?;
                    }
                }
            } else {
                index += 1;
            }
        }
        Ok(())
    }

    pub(super) fn try_bind_cached_local_from_dying_value(
        &mut self,
        index: usize,
        value: SsaValue,
        ty: MachineStorageType,
    ) -> Result<Option<BoundCachedLocal>, WasmError> {
        if self.cache_bindings.get(index).copied().flatten().is_some() {
            return self.bound_cached_local(index).map(Some).ok_or_else(|| {
                WasmError::internal("cached local binding missing after assignment")
            });
        }
        if self.remaining_use_count(value) != 1 {
            return Ok(None);
        }
        let Some((reg, hi_reg)) = self.try_value_regs(value) else {
            return Ok(None);
        };
        if !self.is_linear_value_reg(reg) || self.is_fp_reg(reg) != ty.is_fp() {
            return Ok(None);
        }
        match ty {
            MachineStorageType::GpI64 if self.gp_reg_width == 4 => {
                let Some(hi_reg) = hi_reg else {
                    return Ok(None);
                };
                if !self.is_linear_value_reg(hi_reg) || self.is_fp_reg(hi_reg) {
                    return Ok(None);
                }
                self.clear_linear_value_reg(reg)?;
                self.clear_linear_value_reg(hi_reg)?;
                return self
                    .bind_cached_local_to_regs(index, reg, Some(hi_reg))
                    .map(Some);
            }
            _ => {
                if hi_reg.is_some() {
                    return Ok(None);
                }
            }
        }
        self.clear_linear_value_reg(reg)?;
        self.bind_cached_local_to_regs(index, reg, None).map(Some)
    }

    /// Materialize all live values aliased to `cache_reg` into linear-value
    /// registers. Values in `except` are skipped — they are instruction args
    /// that will be consumed (read) before the cache register is overwritten.
    pub(super) fn materialize_cache_aliases(
        &mut self,
        cache_reg: MachineReg,
        except: &[SsaValue],
    ) -> Result<(), WasmError> {
        let ty = self
            .cached_local_storage_type_for_reg(cache_reg)
            .unwrap_or(MachineStorageType::GpWord);
        let mut i = 0;
        while i < self.values.len() {
            let vr = self.values[i].reg;
            let vv = self.values[i].value;
            if vr == cache_reg && !except.contains(&vv) {
                let Some(t) = self.first_free_linear_value_reg(ty) else {
                    return Err(WasmError::internal(
                        "linear-value budget exhausted during cache alias materialization".into(),
                    ));
                };
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::LinearValue,
                        ty,
                        dst: t,
                        src: MachineValue::Reg(cache_reg),
                    },
                });
                self.set_linear_value_reg(t, Some(vv), Some(ty))?;
                self.values[i].reg = t;
            }
            i += 1;
        }
        Ok(())
    }

    pub(super) fn cached_local_storage_type_for_reg(
        &self,
        reg: MachineReg,
    ) -> Option<MachineStorageType> {
        self.cached_locals
            .iter()
            .zip(self.cache_bindings.iter())
            .find_map(|(cached, binding)| {
                let binding = (*binding)?;
                if binding.reg == reg || binding.hi_reg == Some(reg) {
                    Some(if binding.hi_reg.is_some() {
                        MachineStorageType::GpWord
                    } else {
                        cached.ty
                    })
                } else {
                    None
                }
            })
    }

    pub(super) fn is_bound_cache_reg(&self, reg: MachineReg) -> bool {
        self.cache_bindings
            .iter()
            .flatten()
            .any(|binding| binding.reg == reg || binding.hi_reg == Some(reg))
    }

    fn preferred_cache_binding(&self, index: usize) -> Option<CachedLocalBinding> {
        self.entry_cache_params
            .iter()
            .find(|entry| usize::from(entry.cached_index) == index)
            .and_then(|entry| self.try_binding_from_regs(index, entry.regs))
            .or_else(|| {
                let target = self.single_successor_target()?;
                self.block_entry_cache_params(target)
                    .iter()
                    .find(|entry| usize::from(entry.cached_index) == index)
                    .and_then(|entry| self.try_binding_from_regs(index, entry.regs))
            })
    }

    fn single_successor_target(&self) -> Option<u32> {
        match &self.block.terminator {
            SsaTerminator::Goto(edge) => Some(edge.target.0),
            SsaTerminator::Branch {
                then_edge,
                else_edge,
                ..
            } if then_edge.target == else_edge.target => Some(then_edge.target.0),
            SsaTerminator::BrTable { entries, .. }
                if entries
                    .first()
                    .map(|first| entries.iter().all(|entry| entry.target == first.target))
                    .unwrap_or(false) =>
            {
                entries.first().map(|entry| entry.target.0)
            }
            SsaTerminator::Branch { .. }
            | SsaTerminator::BrTable { .. }
            | SsaTerminator::Return { .. }
            | SsaTerminator::TrapUnreachable => None,
        }
    }

    fn try_binding_from_regs(&self, index: usize, regs: ValueRegs) -> Option<CachedLocalBinding> {
        let cached = self.cached_locals.get(index)?;
        if cached.ty.is_fp() != self.is_fp_reg(regs.lo) {
            return None;
        }
        if self.dynamic_reg_unavailable(regs.lo) {
            return None;
        }
        match (cached.ty, regs.hi) {
            (MachineStorageType::GpI64, Some(hi)) if self.gp_reg_width == 4 => {
                if self.is_fp_reg(hi) || self.dynamic_reg_unavailable(hi) {
                    return None;
                }
                Some(CachedLocalBinding {
                    reg: regs.lo,
                    hi_reg: Some(hi),
                })
            }
            (MachineStorageType::GpI64, None) if self.gp_reg_width == 4 => None,
            (_, hi_reg) => Some(CachedLocalBinding {
                reg: regs.lo,
                hi_reg,
            }),
        }
    }

    fn allocate_cache_binding(&self, index: usize) -> Result<CachedLocalBinding, WasmError> {
        let cached =
            self.cached_locals.get(index).copied().ok_or_else(|| {
                WasmError::internal("cached local binding request is out of range")
            })?;

        if cached.ty.is_fp() {
            for fp_index in 0..self.regfile.fp_dynamic_count() {
                let Some(reg) = self.regfile.fp_dynamic(fp_index) else {
                    continue;
                };
                if self.dynamic_reg_unavailable(reg) {
                    continue;
                }
                return Ok(CachedLocalBinding { reg, hi_reg: None });
            }
            return Err(WasmError::internal("no free FP cache register available for cached local in block b (live_values=, bound_caches=)"));
        }

        if self.gp_reg_width == 4 && matches!(cached.ty, MachineStorageType::GpI64) {
            let dynamic_count = self.regfile.gp_allocatable_count();
            for gp_index in 0..dynamic_count {
                let Some(lo) =
                    super::lower_module::preferred_gp_dynamic_reg(self.regfile, gp_index)
                else {
                    continue;
                };
                if self.dynamic_reg_unavailable(lo) {
                    continue;
                }
                let Some(hi) =
                    super::lower_module::preferred_gp_dynamic_reg(self.regfile, gp_index + 1)
                else {
                    break;
                };
                if self.dynamic_reg_unavailable(hi) {
                    continue;
                }
                return Ok(CachedLocalBinding {
                    reg: lo,
                    hi_reg: Some(hi),
                });
            }
            return Err(WasmError::internal("no free GP cache register pair available for cached local in block b (live_values=, bound_caches=)"));
        }

        for gp_index in 0..self.regfile.gp_allocatable_count() {
            let Some(reg) = super::lower_module::preferred_gp_dynamic_reg(self.regfile, gp_index)
            else {
                continue;
            };
            if self.dynamic_reg_unavailable(reg) {
                continue;
            }
            return Ok(CachedLocalBinding { reg, hi_reg: None });
        }
        Err(WasmError::internal("no free GP cache register available for cached local in block b (live_values=, bound_caches=)"))
    }

    fn dynamic_reg_unavailable(&self, reg: MachineReg) -> bool {
        self.is_bound_cache_reg(reg)
            || self
                .dynamic_index(reg)
                .ok()
                .map(|index| self.linear_value_occupied(index))
                .unwrap_or(true)
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
            if !self.is_linear_value_reg(lo) {
                return Ok(None);
            }
            let hi = self.values[index]
                .hi_reg
                .expect("pair candidate must have hi reg");
            self.values[index].value = value;
            self.set_linear_value_reg(lo, Some(value), Some(MachineStorageType::GpWord))?;
            self.set_linear_value_reg(hi, Some(value), Some(MachineStorageType::GpWord))?;
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
            if self.is_fp_reg(lo) || !self.is_linear_value_reg(lo) {
                return Ok(None);
            }
            let Some(hi) = self.first_free_linear_value_reg(MachineStorageType::GpWord) else {
                return Err(WasmError::internal("prepared SSA-IR exceeded GP dynamic pair budget during native lowering in block b for value"));
            };
            self.values[index].value = value;
            self.values[index].hi_reg = Some(hi);
            self.set_linear_value_reg(lo, Some(value), Some(MachineStorageType::GpWord))?;
            self.set_linear_value_reg(hi, Some(value), Some(MachineStorageType::GpWord))?;
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
        if let Some(index) = self
            .values
            .iter()
            .position(|entry| entry.value == candidate && self.is_fp_reg(entry.reg) == ty.is_fp())
        {
            let reg = self.values[index].reg;
            // Cannot reuse non-linear registers (e.g. cache registers from
            // source aliasing) — they belong to cached locals.
            if !self.is_linear_value_reg(reg) {
                return Ok(None);
            }
            if let Some(hi_reg) = self.values[index].hi_reg {
                if ty.is_fp() {
                    return Ok(None);
                }
                self.clear_linear_value_reg(hi_reg)?;
                self.values[index].hi_reg = None;
            }
            self.values[index].value = value;
            self.set_linear_value_reg(reg, Some(value), Some(ty))?;
            return Ok(Some(reg));
        }
        Ok(None)
    }

    pub(super) fn ops_last(&self) -> Option<&MachineInst> {
        self.ops.last()
    }

    pub(super) fn ops_pop(&mut self) -> Option<MachineInst> {
        self.ops.pop()
    }

    fn materialize_entry_cached_local(
        &mut self,
        cached_index: usize,
        regs: ValueRegs,
    ) -> Result<(), WasmError> {
        // Function entry has no predecessor edge to carry cached locals. If the
        // planner wants them resident immediately, materialize them here from
        // their frame slots instead of faking extra entry block params.
        let cached = self.bind_cached_local_to_regs(cached_index, regs.lo, regs.hi)?;
        if matches!(cached.ty, MachineStorageType::GpI64) {
            let ops = self.i64_ops();
            ops.emit_reload_cached_i64(self, &cached)?;
        } else {
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Load {
                    owner: MachineRegOwner::CachedLocal,
                    ty: cached.ty,
                    dst: cached.reg,
                    addr: self.frame_addr(cached.slot)?,
                    width: canonical_cached_local_mem_width(cached.ty),
                    extension: MachineLoadExtension::None,
                },
            });
        }
        self.set_cache_live(cached_index, true);
        self.set_cache_has_value(cached_index, true);
        self.set_cache_dirty(cached_index, false);
        Ok(())
    }
}

pub(super) fn explicit_cached_locals(program: &SsaProgram) -> collections::Vec<CachedLocal> {
    let pref_map = explicit_cached_local_pref_map(program);
    let mut explicit = BTreeMap::<FrameSlot, ExplicitCachedLocalEntry>::new();
    let mut order = 0usize;

    for block in &program.blocks {
        for inst_idx in 0..block.ops.len() {
            match block.view(inst_idx, program) {
                SsaInstView::LocalGetCache { slot, dst } => {
                    record_explicit_cached_local(
                        &mut explicit,
                        &pref_map,
                        slot,
                        Some(value_type(program, dst)),
                        &mut order,
                    );
                }
                SsaInstView::LocalSetCache { slot, src } => {
                    record_explicit_cached_local(
                        &mut explicit,
                        &pref_map,
                        slot,
                        Some(value_type(program, src)),
                        &mut order,
                    );
                }
                SsaInstView::LocalEnsureCache { slot }
                | SsaInstView::LocalReserveCache { slot }
                | SsaInstView::LocalDropCache { slot } => {
                    record_explicit_cached_local(&mut explicit, &pref_map, slot, None, &mut order);
                }
                _ => {}
            }
        }
    }

    let mut entries = explicit.into_values().collect::<collections::Vec<_>>();
    entries.sort_by_key(|entry| entry.order);
    entries.into_iter().map(|entry| entry.cached).collect()
}

fn explicit_cached_local_pref_map(program: &SsaProgram) -> BTreeMap<FrameSlot, CachePrefEntry> {
    let mut map = BTreeMap::new();
    for (slot_index, ty) in program.local_slot_types.iter().copied().enumerate() {
        map.insert(
            FrameSlot(slot_index as u16),
            CachePrefEntry {
                ty,
                info: program
                    .local_slot_info
                    .get(slot_index)
                    .copied()
                    .unwrap_or_default(),
            },
        );
    }
    map
}

fn record_explicit_cached_local(
    explicit: &mut BTreeMap<FrameSlot, ExplicitCachedLocalEntry>,
    pref_map: &BTreeMap<FrameSlot, CachePrefEntry>,
    slot: FrameSlot,
    ty: Option<ValueType>,
    order: &mut usize,
) {
    let pref = pref_map.get(&slot).copied().unwrap_or(CachePrefEntry {
        ty: ValueType::I32,
        info: LocalSlotInfo::default(),
    });
    let typed = ty.is_some();
    let ty = value_type_storage_type(ty.unwrap_or(pref.ty));

    explicit
        .entry(slot)
        .and_modify(|entry| {
            if typed && !entry.typed {
                entry.cached.ty = ty;
                entry.typed = true;
            }
        })
        .or_insert_with(|| {
            let entry = ExplicitCachedLocalEntry {
                order: *order,
                cached: CachedLocal {
                    slot,
                    ty,
                    info: pref.info,
                },
                typed,
            };
            *order += 1;
            entry
        });
}

fn value_type(program: &SsaProgram, value: SsaValue) -> ValueType {
    program
        .value_types
        .get(value.0 as usize)
        .copied()
        .unwrap_or(ValueType::I32)
}
