use crate::collections;
use tracked_alloc::collections::BTreeMap;

use core::mem;

use crate::{
    error::WasmError,
    value_type::ValueType,
    vm::{
        machine::machine_ir::{
            MachineAddr, MachineFuncId, MachineFunctionAbi, MachineInst, MachineInstKind,
            MachineIntWidth, MachineLoadExtension, MachineMemWidth, MachineParamLoc, MachineReg,
            MachineRegOwner, MachineStorageType, MachineValue,
        },
        middle::{
            cell::CellId,
            frame::FrameSlot,
            ssa_ir::ir::{CellInfo, SsaBlock, SsaInstView, SsaProgram, SsaTerminator, SsaValue},
        },
        runtime::layout::{native_runtime_abi_layout, NativeRuntimeAbiLayout},
    },
};

use super::{
    lower_i64::I64Lowering,
    lower_module::{slot_offset_bytes, target_param_regs},
    lower_regalloc::{
        canonical_cached_cell_mem_width, gp_reg_int_width, gp_reg_mem_width,
        lir_value_storage_type, reserve_nonlinear_dynamic_reg, value_type_storage_type,
        MachineRegFile,
    },
    lower_util::{compute_remaining_uses, RemainingUses},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CachedCell {
    pub cell: CellId,
    pub home: FrameSlot,
    pub value_ty: ValueType,
    pub ty: MachineStorageType,
    pub info: CellInfo,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CachedCellBinding {
    pub reg: MachineReg,
    pub hi_reg: Option<MachineReg>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BoundCachedCell {
    pub cell: CellId,
    pub home: FrameSlot,
    pub value_ty: ValueType,
    pub reg: MachineReg,
    pub hi_reg: Option<MachineReg>,
    pub ty: MachineStorageType,
    pub info: CellInfo,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ParamSlotState {
    FrameAuthoritative,
    RegisterOnly {
        lo: MachineReg,
        hi: Option<MachineReg>,
        ty: MachineStorageType,
    },
}

pub(super) struct BlockLowerContext<'a> {
    regfile: &'a MachineRegFile,
    program: &'a SsaProgram,
    block: &'a SsaBlock,
    current_abi: &'a MachineFunctionAbi,
    all_runtime: &'a [MachineFunctionAbi],
    machine_params: collections::Vec<ValueRegs>,
    entry_cache_params: collections::Vec<EntryCacheParam>,
    all_entry_cache_params: &'a [collections::Vec<EntryCacheParam>],
    gp_reg_width: u8,
    i64_ops: &'static dyn I64Lowering,
    ops: collections::Vec<MachineInst>,
    cached_cells: collections::Vec<CachedCell>,
    cache_bindings: collections::Vec<Option<CachedCellBinding>>,
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
    call_preserved_cache_candidates: &'a [bool],
    values: collections::Vec<ValueLocation>,
    remaining_uses: RemainingUses,
    /// Values whose last use has just been consumed. Keeping this worklist
    /// lets op-end cleanup visit only values that can actually be released.
    dead_values: collections::Vec<SsaValue>,
    /// Dynamic-register occupancy for linear SSA-like values.
    ///
    /// Cached locals are tracked separately through `cache_bindings`. A dynamic
    /// register is therefore in exactly one of three semantic states at any
    /// program point:
    /// - free
    /// - occupied by one linear value
    /// - bound to one cached local
    linear_value_state: collections::Vec<LinearValueState>,
    /// Number of non-linear owners reserving each dynamic register.
    ///
    /// This mirrors cached-local bindings and still-unpublished incoming
    /// parameters so allocator queries do not repeatedly scan both vectors.
    /// A count is used because entry materialization can briefly transfer a
    /// parameter directly into a cache binding on the same register.
    nonlinear_dynamic_reg_owners: collections::Vec<u8>,
    param_slot_state: collections::Vec<ParamSlotState>,
    #[cfg(sf_has_guard_pages)]
    guard_pages: bool,
    #[cfg(sf_has_guard_pages)]
    stack_guard_pages: bool,
}

#[derive(Clone, Copy, Debug)]
struct CachePrefEntry {
    ty: ValueType,
    info: CellInfo,
}

#[derive(Clone, Copy, Debug)]
struct ExplicitCachedCellEntry {
    order: usize,
    cached: CachedCell,
    typed: bool,
}

fn initial_param_slot_state(
    abi: &MachineFunctionAbi,
    regfile: &MachineRegFile,
    program: &SsaProgram,
) -> Result<collections::Vec<ParamSlotState>, WasmError> {
    let state_len = program
        .cell_types
        .len()
        .max(usize::from(abi.frame_prefix_slots));
    let mut states = collections::vec![ParamSlotState::FrameAuthoritative; state_len];
    for loc in &abi.param_locs {
        let (param_index, lo, hi, ty) = match *loc {
            MachineParamLoc::Frame { .. } => continue,
            MachineParamLoc::GpArg {
                param_index,
                lane,
                ty,
            } => {
                let lo = regfile
                    .ordered_gp_allocatable(lane as usize)
                    .ok_or_else(|| {
                        WasmError::internal(
                            "internal GP argument lane exceeds dynamic register budget",
                        )
                    })?;
                (param_index, lo, None, ty)
            }
            MachineParamLoc::GpArgPair {
                param_index,
                lo_lane,
                hi_lane,
            } => {
                let lo = regfile
                    .ordered_gp_allocatable(lo_lane as usize)
                    .ok_or_else(|| {
                        WasmError::internal(
                            "internal GP argument lo lane exceeds dynamic register budget",
                        )
                    })?;
                let hi = regfile
                    .ordered_gp_allocatable(hi_lane as usize)
                    .ok_or_else(|| {
                        WasmError::internal(
                            "internal GP argument hi lane exceeds dynamic register budget",
                        )
                    })?;
                (param_index, lo, Some(hi), MachineStorageType::GpI64)
            }
            MachineParamLoc::FpArg {
                param_index,
                lane,
                ty,
            } => {
                let lo = regfile.fp_dynamic(lane as usize).ok_or_else(|| {
                    WasmError::internal("internal FP argument lane exceeds dynamic register budget")
                })?;
                (param_index, lo, None, ty)
            }
        };
        if let Some(state) = states.get_mut(param_index as usize) {
            *state = ParamSlotState::RegisterOnly { lo, hi, ty };
        }
    }
    Ok(states)
}

impl<'a> BlockLowerContext<'a> {
    pub(super) fn new(
        regfile: &'a MachineRegFile,
        program: &'a SsaProgram,
        cached_cells: &'a [CachedCell],
        all_entry_cache_params: &'a [collections::Vec<EntryCacheParam>],
        block: &'a SsaBlock,
        current_abi: &'a MachineFunctionAbi,
        all_runtime: &'a [MachineFunctionAbi],
        gp_reg_width: u8,
        i64_ops: &'static dyn I64Lowering,
        is_entry: bool,
        initial_cache_dirty: Option<&[bool]>,
        call_preserved_cache_candidates: Option<&'a [bool]>,
        #[cfg(sf_has_guard_pages)] guard_pages: bool,
        #[cfg(sf_has_guard_pages)] stack_guard_pages: bool,
    ) -> Result<Self, WasmError> {
        let machine_params = target_param_regs(&block.params, program, regfile, gp_reg_width)?;
        let entry_cache_params = all_entry_cache_params
            .get(block.id.as_usize())
            .cloned()
            .unwrap_or_default();
        let cache_live = collections::vec![false; cached_cells.len()];
        let cache_has_value = collections::vec![false; cached_cells.len()];
        let cache_dirty = collections::vec![false; cached_cells.len()];
        let param_slot_state = if is_entry {
            initial_param_slot_state(current_abi, regfile, program)?
        } else {
            collections::vec![
                ParamSlotState::FrameAuthoritative;
                program
                    .cell_types
                    .len()
                    .max(usize::from(current_abi.frame_prefix_slots))
            ]
        };
        let mut nonlinear_dynamic_reg_owners =
            collections::vec![0; regfile.gp_dynamic_count() + regfile.fp_dynamic_count()];
        for state in &param_slot_state {
            if let ParamSlotState::RegisterOnly { lo, hi, .. } = *state {
                reserve_nonlinear_dynamic_reg(regfile, &mut nonlinear_dynamic_reg_owners, lo)?;
                if let Some(hi) = hi {
                    reserve_nonlinear_dynamic_reg(regfile, &mut nonlinear_dynamic_reg_owners, hi)?;
                }
            }
        }

        let mut lower = Self {
            regfile,
            program,
            block,
            current_abi,
            all_runtime,
            machine_params,
            entry_cache_params,
            all_entry_cache_params,
            gp_reg_width,
            i64_ops,
            ops: collections::Vec::new(),
            cached_cells: cached_cells.to_vec().into(),
            cache_bindings: collections::vec![None; cache_live.len()],
            cache_live,
            cache_has_value,
            cache_dirty,
            call_preserved_cache_candidates: call_preserved_cache_candidates.unwrap_or(&[]),
            values: collections::Vec::new(),
            remaining_uses: compute_remaining_uses(block, program),
            dead_values: collections::Vec::new(),
            linear_value_state: collections::vec![
                LinearValueState::default();
                regfile.gp_dynamic_count() + regfile.fp_dynamic_count()
            ],
            nonlinear_dynamic_reg_owners,
            param_slot_state,
            #[cfg(sf_has_guard_pages)]
            guard_pages,
            #[cfg(sf_has_guard_pages)]
            stack_guard_pages,
        };

        let machine_params = lower.machine_params.clone();
        for (param, regs) in block
            .params
            .iter()
            .copied()
            .zip(machine_params.iter().copied())
        {
            lower.push_value_location(param, regs.lo, regs.hi);
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
                    lower.materialize_entry_cached_cell(cached_index, entry.regs)?;
                } else {
                    lower.bind_cached_cell_to_regs(cached_index, entry.regs.lo, entry.regs.hi)?;
                    lower.set_cache_live(cached_index, true);
                    lower.set_cache_has_value(cached_index, false);
                    lower.set_cache_dirty(cached_index, false);
                }
            } else {
                lower.bind_cached_cell_to_regs(cached_index, entry.regs.lo, entry.regs.hi)?;
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

    pub(super) fn cached_cell_index(&self, cell: CellId) -> Option<usize> {
        self.cached_cells
            .iter()
            .position(|cached| cached.cell == cell)
    }

    /// A cell's frame home. Cells mirror their home slot; every frame access
    /// for a cell goes through this lookup, never through the cell id itself.
    pub(super) fn cell_home(&self, cell: CellId) -> Result<FrameSlot, WasmError> {
        self.program
            .cell_homes
            .get(cell.0 as usize)
            .copied()
            .ok_or_else(|| WasmError::internal("cell has no frame home"))
    }

    pub(super) fn register_param_state(
        &self,
        cell: CellId,
    ) -> Option<(MachineReg, Option<MachineReg>, MachineStorageType)> {
        match self.param_slot_state.get(cell.0 as usize).copied() {
            Some(ParamSlotState::RegisterOnly { lo, hi, ty }) => Some((lo, hi, ty)),
            _ => None,
        }
    }

    pub(super) fn mark_param_frame_authoritative(&mut self, cell: CellId) {
        let Some(state) = self.param_slot_state.get_mut(cell.0 as usize) else {
            return;
        };
        let previous = core::mem::replace(state, ParamSlotState::FrameAuthoritative);
        if let ParamSlotState::RegisterOnly { lo, hi, .. } = previous {
            self.release_nonlinear_dynamic_reg(lo);
            if let Some(hi) = hi {
                self.release_nonlinear_dynamic_reg(hi);
            }
        }
    }

    pub(super) fn publish_register_params_to_frame(&mut self) -> Result<(), WasmError> {
        let mut params = collections::Vec::new();
        for (cell_index, state) in self.param_slot_state.iter().copied().enumerate() {
            if let ParamSlotState::RegisterOnly { lo, hi, ty } = state {
                params.push((CellId(cell_index as u16), lo, hi, ty));
            }
        }

        for (cell, lo, hi, ty) in params {
            let slot = self.cell_home(cell)?;
            if let Some(hi) = hi {
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: self.frame_addr_offset(slot, 0)?,
                        width: MachineMemWidth::U32,
                        src: MachineValue::Reg(lo),
                    },
                });
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: self.frame_addr_offset(slot, 4)?,
                        width: MachineMemWidth::U32,
                        src: MachineValue::Reg(hi),
                    },
                });
            } else {
                #[cfg(sf_has_simd)]
                if matches!(ty, MachineStorageType::V128) {
                    self.emit_store_frame_v128(slot, lo)?;
                    self.mark_param_frame_authoritative(cell);
                    continue;
                }
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Store {
                        ty,
                        addr: self.frame_addr(slot)?,
                        width: canonical_cached_cell_mem_width(ty),
                        src: MachineValue::Reg(lo),
                    },
                });
            }
            self.mark_param_frame_authoritative(cell);
        }

        Ok(())
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

    /// Address of the i-th inline global raw-ptr slot within the runtime
    /// context: `[runtime_base + globals_ptrs_inline_offset + idx * ptr_size]`.
    /// Used by `global.get`/`global.set` lowering to reach the precomputed
    /// raw_ptr without a second indirection through a view pointer.
    pub(super) fn globals_ptr_slot_addr(&self, idx: u32) -> Result<MachineAddr, WasmError> {
        let layout = self.runtime_abi_layout();
        let stride = layout.gp_unit_bytes as u64;
        let scaled = (idx as u64)
            .checked_mul(stride)
            .and_then(|v| v.checked_add(layout.context.globals_ptrs_inline_offset as u64))
            .ok_or_else(|| WasmError::internal("globals ptr slot offset overflow"))?;
        let offset = i32::try_from(scaled)
            .map_err(|_| WasmError::internal("globals ptr slot offset exceeds i32"))?;
        Ok(MachineAddr {
            base: self.regfile.runtime_base(),
            offset,
        })
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

    pub(super) fn current_return_abi(&self) -> &crate::vm::machine::machine_ir::MachineReturnAbi {
        &self.current_abi.return_abi
    }

    // -----------------------------------------------------------------------
    // Canonical width / storage helpers
    // -----------------------------------------------------------------------

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

    #[cfg(sf_has_guard_pages)]
    pub(super) fn use_stack_guard_pages(&self) -> bool {
        self.stack_guard_pages
    }

    pub(super) fn use_explicit_stack_prechecks(&self) -> bool {
        #[cfg(sf_has_guard_pages)]
        {
            !self.use_stack_guard_pages()
        }
        #[cfg(not(sf_has_guard_pages))]
        {
            true
        }
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

    pub(super) fn cached_cells(&self) -> &[CachedCell] {
        &self.cached_cells
    }

    pub(super) fn bound_cached_cell(&self, index: usize) -> Option<BoundCachedCell> {
        let cached = self.cached_cells.get(index).copied()?;
        let binding = self.cache_bindings.get(index).copied().flatten()?;
        Some(BoundCachedCell {
            cell: cached.cell,
            home: cached.home,
            value_ty: cached.value_ty,
            reg: binding.reg,
            hi_reg: binding.hi_reg,
            ty: cached.ty,
            info: cached.info,
        })
    }

    pub(super) fn ensure_bound_cached_cell(
        &mut self,
        index: usize,
    ) -> Result<BoundCachedCell, WasmError> {
        if self.cache_bindings.get(index).copied().flatten().is_none() {
            let preferred = self.allocate_cache_binding(index)?;
            self.bind_cached_cell_to_regs(index, preferred.reg, preferred.hi_reg)?;
        }
        self.bound_cached_cell(index)
            .ok_or_else(|| WasmError::internal("cached local binding missing after assignment"))
    }

    pub(super) fn bind_cached_cell_to_regs(
        &mut self,
        index: usize,
        reg: MachineReg,
        hi_reg: Option<MachineReg>,
    ) -> Result<BoundCachedCell, WasmError> {
        let next = CachedCellBinding { reg, hi_reg };
        let previous = self
            .cache_bindings
            .get(index)
            .copied()
            .ok_or_else(|| WasmError::internal("cached local binding is out of range"))?;
        if previous == Some(next) {
            return self.bound_cached_cell(index).ok_or_else(|| {
                WasmError::internal("cached local binding missing after assignment")
            });
        }
        self.reserve_nonlinear_dynamic_reg(reg)?;
        if let Some(hi_reg) = hi_reg {
            self.reserve_nonlinear_dynamic_reg(hi_reg)?;
        }
        if let Some(previous) = previous {
            self.release_nonlinear_dynamic_reg(previous.reg);
            if let Some(hi_reg) = previous.hi_reg {
                self.release_nonlinear_dynamic_reg(hi_reg);
            }
        }
        let slot = self
            .cache_bindings
            .get_mut(index)
            .ok_or_else(|| WasmError::internal("cached local binding is out of range"))?;
        *slot = Some(next);
        self.bound_cached_cell(index)
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
            let previous = binding.take();
            if let Some(previous) = previous {
                self.release_nonlinear_dynamic_reg(previous.reg);
                if let Some(hi_reg) = previous.hi_reg {
                    self.release_nonlinear_dynamic_reg(hi_reg);
                }
            }
        }
    }

    pub(super) fn clear_cache_live(&mut self) {
        for live in &mut self.cache_live {
            *live = false;
        }
        for has_value in &mut self.cache_has_value {
            *has_value = false;
        }
        for index in 0..self.cache_bindings.len() {
            self.clear_cache_binding(index);
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

    pub(super) fn prefers_preserved_cache_binding(&self, index: usize) -> bool {
        self.call_preserved_cache_candidates
            .get(index)
            .copied()
            .unwrap_or(false)
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

    pub(super) fn value_location_owns_reg(&self, reg: MachineReg) -> bool {
        self.values
            .iter()
            .any(|value| value.reg == reg || value.hi_reg == Some(reg))
    }

    pub(super) fn remaining_use_count(&self, value: SsaValue) -> u32 {
        self.remaining_uses.count(value)
    }

    pub(super) fn consume_value_use(&mut self, value: SsaValue) {
        if self.remaining_uses.consume(value) == 0 {
            self.dead_values.push(value);
        }
    }

    pub(super) fn linear_value_occupied(&self, index: usize) -> bool {
        self.linear_value_state
            .get(index)
            .map(|state| state.value.is_some())
            .unwrap_or(false)
    }

    pub(super) fn nonlinear_dynamic_reg_occupied(&self, index: usize) -> bool {
        self.nonlinear_dynamic_reg_owners
            .get(index)
            .copied()
            .unwrap_or(1)
            != 0
    }

    fn reserve_nonlinear_dynamic_reg(&mut self, reg: MachineReg) -> Result<(), WasmError> {
        let index = self.dynamic_index(reg)?;
        let owners = self
            .nonlinear_dynamic_reg_owners
            .get_mut(index)
            .ok_or_else(|| WasmError::internal("dynamic register owner index is out of range"))?;
        *owners = owners
            .checked_add(1)
            .ok_or_else(|| WasmError::internal("dynamic register owner count overflow"))?;
        Ok(())
    }

    fn release_nonlinear_dynamic_reg(&mut self, reg: MachineReg) {
        let Ok(index) = self.dynamic_index(reg) else {
            debug_assert!(false, "non-dynamic register had a dynamic owner");
            return;
        };
        let Some(owners) = self.nonlinear_dynamic_reg_owners.get_mut(index) else {
            debug_assert!(false, "dynamic register owner index is out of range");
            return;
        };
        debug_assert_ne!(*owners, 0, "dynamic register owner count underflow");
        *owners = owners.saturating_sub(1);
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
        self.queue_value_if_dead(value);
    }

    fn queue_value_if_dead(&mut self, value: SsaValue) {
        if self.remaining_use_count(value) == 0 {
            self.dead_values.push(value);
        }
    }

    /// Free linear-value registers for values with no remaining uses.
    /// Values in non-linear registers (e.g. cache registers from source
    /// aliasing or sink allocation) are removed from the value list but their
    /// registers are not cleared — they belong to the cached local.
    pub(super) fn release_dead_values(&mut self) -> Result<(), WasmError> {
        while let Some(value) = self.dead_values.pop() {
            if self.remaining_use_count(value) != 0 {
                continue;
            }
            let Some(index) = self.values.iter().position(|entry| entry.value == value) else {
                // A dead input may already have donated its register to the
                // current result. Its stale work item is then harmless.
                continue;
            };
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
        }
        Ok(())
    }

    pub(super) fn try_bind_cached_cell_from_dying_value(
        &mut self,
        index: usize,
        value: SsaValue,
        ty: MachineStorageType,
    ) -> Result<Option<BoundCachedCell>, WasmError> {
        if self.cache_bindings.get(index).copied().flatten().is_some() {
            return self.bound_cached_cell(index).map(Some).ok_or_else(|| {
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
        if self.prefers_preserved_cache_binding(index)
            && !self.binding_is_fully_preserved(reg, hi_reg)
            && self.find_cache_binding(index, Some(true))?.is_some()
        {
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
                    .bind_cached_cell_to_regs(index, reg, Some(hi_reg))
                    .map(Some);
            }
            _ => {
                if hi_reg.is_some() {
                    return Ok(None);
                }
            }
        }
        self.clear_linear_value_reg(reg)?;
        self.bind_cached_cell_to_regs(index, reg, None).map(Some)
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
            .cached_cell_storage_type_for_reg(cache_reg)
            .unwrap_or(MachineStorageType::GpWord);
        let mut i = 0;
        while i < self.values.len() {
            let vr = self.values[i].reg;
            let vv = self.values[i].value;
            if vr == cache_reg && !except.contains(&vv) {
                if let Some(hi_reg) = self.values[i].hi_reg {
                    let lo_tmp =
                        self.materialize_cache_alias_reg(vr, vv, MachineStorageType::GpWord)?;
                    let hi_tmp =
                        self.materialize_cache_alias_reg(hi_reg, vv, MachineStorageType::GpWord)?;
                    self.values[i].reg = lo_tmp;
                    self.values[i].hi_reg = Some(hi_tmp);
                    i += 1;
                    continue;
                }
                let t = self.materialize_cache_alias_reg(cache_reg, vv, ty)?;
                self.values[i].reg = t;
            }
            i += 1;
        }
        Ok(())
    }

    fn materialize_cache_alias_reg(
        &mut self,
        src: MachineReg,
        value: SsaValue,
        ty: MachineStorageType,
    ) -> Result<MachineReg, WasmError> {
        let Some(dst) = self.first_free_linear_value_reg(ty) else {
            return Err(WasmError::internal(
                "linear-value budget exhausted during cache alias materialization".into(),
            ));
        };
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Move {
                owner: MachineRegOwner::LinearValue,
                ty,
                dst,
                src: MachineValue::Reg(src),
            },
        });
        self.set_linear_value_reg(dst, Some(value), Some(ty))?;
        Ok(dst)
    }

    pub(super) fn cached_cell_storage_type_for_reg(
        &self,
        reg: MachineReg,
    ) -> Option<MachineStorageType> {
        self.cached_cells
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

    fn preferred_cache_binding(&self, index: usize) -> Option<CachedCellBinding> {
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
            | SsaTerminator::ReturnScalar { .. }
            | SsaTerminator::TailCallDirect { .. }
            | SsaTerminator::TailCallIndirect { .. }
            | SsaTerminator::TailCallRef { .. }
            | SsaTerminator::TrapUnreachable
            | SsaTerminator::EhThrow { .. }
            | SsaTerminator::EhThrowRef { .. } => None,
        }
    }

    fn try_binding_from_regs(&self, index: usize, regs: ValueRegs) -> Option<CachedCellBinding> {
        let cached = self.cached_cells.get(index)?;
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
                Some(CachedCellBinding {
                    reg: regs.lo,
                    hi_reg: Some(hi),
                })
            }
            (MachineStorageType::GpI64, None) if self.gp_reg_width == 4 => None,
            (_, hi_reg) => Some(CachedCellBinding {
                reg: regs.lo,
                hi_reg,
            }),
        }
    }

    fn allocate_cache_binding(&self, index: usize) -> Result<CachedCellBinding, WasmError> {
        let preferred_preserved = self.prefers_preserved_cache_binding(index);
        if let Some(binding) = self.preferred_cache_binding(index) {
            if self.cache_binding_matches_preference(
                binding.reg,
                binding.hi_reg,
                Some(preferred_preserved),
            ) {
                return Ok(binding);
            }
        }
        if let Some(binding) = self.find_cache_binding(index, Some(preferred_preserved))? {
            return Ok(binding);
        }
        if let Some(binding) = self.preferred_cache_binding(index) {
            return Ok(binding);
        }
        if let Some(binding) = self.find_cache_binding(index, None)? {
            return Ok(binding);
        }
        Err(WasmError::internal(
            "middle cache demand exceeded available dynamic lanes after canonical register params were frame-published",
        ))
    }

    fn find_cache_binding(
        &self,
        index: usize,
        preference: Option<bool>,
    ) -> Result<Option<CachedCellBinding>, WasmError> {
        let cached =
            self.cached_cells.get(index).copied().ok_or_else(|| {
                WasmError::internal("cached local binding request is out of range")
            })?;

        if cached.ty.is_fp() {
            for fp_index in 0..self.regfile.fp_allocatable_count() {
                let Some(reg) =
                    super::lower_module::preferred_fp_dynamic_reg(self.regfile, fp_index)
                else {
                    continue;
                };
                if !self.cache_binding_matches_preference(reg, None, preference) {
                    continue;
                }
                if self.dynamic_reg_unavailable(reg) {
                    continue;
                }
                return Ok(Some(CachedCellBinding { reg, hi_reg: None }));
            }
            return Ok(None);
        }

        if self.gp_reg_width == 4 && matches!(cached.ty, MachineStorageType::GpI64) {
            let dynamic_count = self.regfile.gp_allocatable_count();
            for gp_index in 0..dynamic_count {
                let Some(lo) =
                    super::lower_module::preferred_gp_dynamic_reg(self.regfile, gp_index)
                else {
                    continue;
                };
                let Some(hi) =
                    super::lower_module::preferred_gp_dynamic_reg(self.regfile, gp_index + 1)
                else {
                    break;
                };
                if !self.cache_binding_matches_preference(lo, Some(hi), preference) {
                    continue;
                }
                if self.dynamic_reg_unavailable(lo) || self.dynamic_reg_unavailable(hi) {
                    continue;
                }
                return Ok(Some(CachedCellBinding {
                    reg: lo,
                    hi_reg: Some(hi),
                }));
            }
            return Ok(None);
        }

        for gp_index in 0..self.regfile.gp_allocatable_count() {
            let Some(reg) = super::lower_module::preferred_gp_dynamic_reg(self.regfile, gp_index)
            else {
                continue;
            };
            if !self.cache_binding_matches_preference(reg, None, preference) {
                continue;
            }
            if self.dynamic_reg_unavailable(reg) {
                continue;
            }
            return Ok(Some(CachedCellBinding { reg, hi_reg: None }));
        }
        Ok(None)
    }

    fn cache_binding_matches_preference(
        &self,
        reg: MachineReg,
        hi_reg: Option<MachineReg>,
        preference: Option<bool>,
    ) -> bool {
        match preference {
            Some(true) => self.binding_is_fully_preserved(reg, hi_reg),
            Some(false) => !self.binding_uses_preserved(reg, hi_reg),
            None => true,
        }
    }

    fn binding_is_fully_preserved(&self, reg: MachineReg, hi_reg: Option<MachineReg>) -> bool {
        self.regfile.is_preserved_dynamic_reg(reg)
            && hi_reg
                .map(|reg| self.regfile.is_preserved_dynamic_reg(reg))
                .unwrap_or(true)
    }

    fn binding_uses_preserved(&self, reg: MachineReg, hi_reg: Option<MachineReg>) -> bool {
        self.regfile.is_preserved_dynamic_reg(reg)
            || hi_reg
                .map(|reg| self.regfile.is_preserved_dynamic_reg(reg))
                .unwrap_or(false)
    }

    pub(super) fn bind_register_param_cached_cell(
        &mut self,
        index: usize,
        src_lo: MachineReg,
        src_hi: Option<MachineReg>,
    ) -> Result<BoundCachedCell, WasmError> {
        if self.prefers_preserved_cache_binding(index) {
            if let Some(target) = self.find_cache_binding(index, Some(true))? {
                let cached = self.bind_cached_cell_to_regs(index, target.reg, target.hi_reg)?;
                self.emit_move_register_param_to_cached_cell(&cached, src_lo, src_hi)?;
                return Ok(cached);
            }
        }
        self.bind_cached_cell_to_regs(index, src_lo, src_hi)
    }

    pub(super) fn emit_move_register_param_to_cached_cell(
        &mut self,
        cached: &BoundCachedCell,
        src_lo: MachineReg,
        src_hi: Option<MachineReg>,
    ) -> Result<(), WasmError> {
        if let Some(dst_hi) = cached.hi_reg {
            let Some(src_hi) = src_hi else {
                return Err(WasmError::internal(
                    "cached i64 register-param source is missing high half",
                ));
            };
            if cached.reg != src_lo {
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::CachedCell,
                        ty: MachineStorageType::GpWord,
                        dst: cached.reg,
                        src: MachineValue::Reg(src_lo),
                    },
                });
            }
            if dst_hi != src_hi {
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::CachedCell,
                        ty: MachineStorageType::GpWord,
                        dst: dst_hi,
                        src: MachineValue::Reg(src_hi),
                    },
                });
            }
            return Ok(());
        }
        if cached.reg != src_lo {
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Move {
                    owner: MachineRegOwner::CachedCell,
                    ty: cached.ty,
                    dst: cached.reg,
                    src: MachineValue::Reg(src_lo),
                },
            });
        }
        Ok(())
    }

    fn dynamic_reg_unavailable(&self, reg: MachineReg) -> bool {
        self.dynamic_index(reg)
            .ok()
            .map(|index| {
                self.linear_value_occupied(index)
                    || self.nonlinear_dynamic_reg_occupied(index)
                    || self.value_location_owns_reg(reg)
            })
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
            self.queue_value_if_dead(value);
            return Ok(Some((lo, hi)));
        }
        Ok(None)
    }

    /// Try to reuse a dead scalar candidate's register as the lo half of a new pair.
    ///
    /// Returns `Ok(None)` when no free GP lane is available for the hi half,
    /// letting the caller fall through to `alloc_i64_value_pair`. That path
    /// owns the single canonical register-param publication retry; if that
    /// still cannot allocate, the middle-end budget invariant failed.
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
                return Ok(None);
            };
            self.values[index].value = value;
            self.values[index].hi_reg = Some(hi);
            self.set_linear_value_reg(lo, Some(value), Some(MachineStorageType::GpWord))?;
            self.set_linear_value_reg(hi, Some(value), Some(MachineStorageType::GpWord))?;
            self.queue_value_if_dead(value);
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
            self.queue_value_if_dead(value);
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

    fn materialize_entry_cached_cell(
        &mut self,
        cached_index: usize,
        regs: ValueRegs,
    ) -> Result<(), WasmError> {
        // Function entry has no predecessor edge to carry cached locals. If the
        // planner wants them resident immediately, materialize them here from
        // their frame slots instead of faking extra entry block params.
        if let Some(cached) = self.cached_cells.get(cached_index).copied() {
            if let Some((lo, hi, ty)) = self.register_param_state(cached.cell) {
                if ty == cached.ty {
                    let cached = self.bind_cached_cell_to_regs(cached_index, regs.lo, regs.hi)?;
                    self.emit_move_register_param_to_cached_cell(&cached, lo, hi)?;
                    self.set_cache_live(cached_index, true);
                    self.set_cache_has_value(cached_index, true);
                    self.set_cache_dirty(cached_index, true);
                    self.mark_param_frame_authoritative(cached.cell);
                    return Ok(());
                }
            }
        }
        let cached = self.bind_cached_cell_to_regs(cached_index, regs.lo, regs.hi)?;
        if matches!(cached.ty, MachineStorageType::GpI64) {
            let ops = self.i64_ops();
            ops.emit_reload_cached_i64(self, &cached)?;
        } else {
            #[cfg(sf_has_simd)]
            if matches!(cached.ty, MachineStorageType::V128) {
                self.emit_load_frame_v128(cached.home, cached.reg, MachineRegOwner::CachedCell)?;
            } else {
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::CachedCell,
                        ty: cached.ty,
                        dst: cached.reg,
                        addr: self.frame_addr(cached.home)?,
                        width: canonical_cached_cell_mem_width(cached.ty),
                        extension: MachineLoadExtension::None,
                    },
                });
            }
            #[cfg(not(sf_has_simd))]
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Load {
                    owner: MachineRegOwner::CachedCell,
                    ty: cached.ty,
                    dst: cached.reg,
                    addr: self.frame_addr(cached.home)?,
                    width: canonical_cached_cell_mem_width(cached.ty),
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

pub(super) fn explicit_cached_cells(
    program: &SsaProgram,
) -> Result<collections::Vec<CachedCell>, WasmError> {
    let pref_map = explicit_cached_cell_pref_map(program);
    let mut explicit = BTreeMap::<CellId, ExplicitCachedCellEntry>::new();
    let mut order = 0usize;

    for block in &program.blocks {
        for inst_idx in 0..block.ops.len() {
            match block.view(inst_idx, program) {
                SsaInstView::CellGetCache { cell: slot, dst } => {
                    record_explicit_cached_cell(
                        &mut explicit,
                        &pref_map,
                        cell_home_in(program, slot)?,
                        slot,
                        Some(value_type(program, dst)),
                        &mut order,
                    );
                }
                SsaInstView::CellSetCache { cell: slot, src } => {
                    record_explicit_cached_cell(
                        &mut explicit,
                        &pref_map,
                        cell_home_in(program, slot)?,
                        slot,
                        Some(value_type(program, src)),
                        &mut order,
                    );
                }
                SsaInstView::CellEnsureCache { cell: slot }
                | SsaInstView::CellReserveCache { cell: slot }
                | SsaInstView::CellDropCache { cell: slot } => {
                    record_explicit_cached_cell(
                        &mut explicit,
                        &pref_map,
                        cell_home_in(program, slot)?,
                        slot,
                        None,
                        &mut order,
                    );
                }
                _ => {}
            }
        }
    }

    let mut entries = explicit.into_values().collect::<collections::Vec<_>>();
    entries.sort_by_key(|entry| entry.order);
    Ok(entries.into_iter().map(|entry| entry.cached).collect())
}

/// A cell's frame home out of the published table (free-function form for
/// contexts without a `BlockLowerContext`).
pub(super) fn cell_home_in(program: &SsaProgram, cell: CellId) -> Result<FrameSlot, WasmError> {
    program
        .cell_homes
        .get(cell.0 as usize)
        .copied()
        .ok_or_else(|| WasmError::internal("cell has no frame home"))
}

fn explicit_cached_cell_pref_map(program: &SsaProgram) -> BTreeMap<CellId, CachePrefEntry> {
    let mut map = BTreeMap::new();
    for (slot_index, ty) in program.cell_types.iter().copied().enumerate() {
        map.insert(
            CellId(slot_index as u16),
            CachePrefEntry {
                ty,
                info: program
                    .cell_info
                    .get(slot_index)
                    .copied()
                    .unwrap_or_default(),
            },
        );
    }
    map
}

fn record_explicit_cached_cell(
    explicit: &mut BTreeMap<CellId, ExplicitCachedCellEntry>,
    pref_map: &BTreeMap<CellId, CachePrefEntry>,
    home: FrameSlot,
    cell: CellId,
    ty: Option<ValueType>,
    order: &mut usize,
) {
    let pref = pref_map.get(&cell).copied().unwrap_or(CachePrefEntry {
        ty: ValueType::I32,
        info: CellInfo::default(),
    });
    let typed = ty.is_some();
    let value_ty = ty.unwrap_or(pref.ty);
    let storage_ty = value_type_storage_type(value_ty);

    explicit
        .entry(cell)
        .and_modify(|entry| {
            if typed && !entry.typed {
                entry.cached.value_ty = value_ty;
                entry.cached.ty = storage_ty;
                entry.typed = true;
            }
        })
        .or_insert_with(|| {
            let entry = ExplicitCachedCellEntry {
                order: *order,
                cached: CachedCell {
                    cell,
                    home,
                    value_ty,
                    ty: storage_ty,
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
