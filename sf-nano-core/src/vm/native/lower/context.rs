use alloc::vec::Vec;
use core::mem;

use crate::{
    error::WasmError,
    vm::{
        lir::{
            ir::{
                LirBlock, LirEdge, LirInst, LirInstKind, LirLocalCachePrefs, LirProgram,
                LirTerminator, LirValue,
            },
            leaf::LirLeafOp,
            slot::FrameSlot,
        },
        native::{
            ir::machine::{
                MachineAddr, MachineBlockId, MachineBranchCond, MachineEdge, MachineFuncId,
                MachineInst, MachineInstKind, MachineLoadExtension, MachineMemWidth, MachineReg,
                MachineTerminator, MachineTrapKind, MachineValue,
            },
            ir::runtime::{MachineCallLinkLayout, MachineFrameRegion, MachineFunctionRuntime},
            lower::{slot_offset_bytes, target_param_regs},
            runtime::context::ctx_offset,
        },
        plan::frame::FrameLayoutPlan,
    },
};

use super::{
    regfile::MachineRegFile,
    util::{compute_remaining_uses, single_arg, single_result, two_args},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CachedLocal {
    slot: FrameSlot,
    reg: MachineReg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ValueLocation {
    value: LirValue,
    reg: MachineReg,
}

pub(super) struct BlockLowerContext<'a> {
    regfile: &'a MachineRegFile,
    frame: FrameLayoutPlan,
    program: &'a LirProgram,
    block: &'a LirBlock,
    runtime: MachineFunctionRuntime,
    all_runtime: &'a [MachineFunctionRuntime],
    call_link: MachineCallLinkLayout,
    machine_params: Vec<MachineReg>,
    ops: Vec<MachineInst>,
    cached_locals: Vec<CachedLocal>,
    values: Vec<ValueLocation>,
    remaining_uses: alloc::collections::BTreeMap<LirValue, u32>,
    transient_owner: Vec<Option<LirValue>>,
}

impl<'a> BlockLowerContext<'a> {
    pub(super) fn new(
        regfile: &'a MachineRegFile,
        frame: FrameLayoutPlan,
        program: &'a LirProgram,
        cache_prefs: &LirLocalCachePrefs,
        block: &'a LirBlock,
        runtime: MachineFunctionRuntime,
        all_runtime: &'a [MachineFunctionRuntime],
        call_link: MachineCallLinkLayout,
        is_entry: bool,
    ) -> Result<Self, WasmError> {
        let machine_params = target_param_regs(block.params.len(), regfile)?;
        let mut cached_locals = Vec::new();
        for (index, slot) in cache_prefs.preferred_slots.iter().copied().enumerate() {
            let Some(reg) = regfile.local_cache(index) else {
                break;
            };
            cached_locals.push(CachedLocal { slot, reg });
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
            ops: Vec::new(),
            cached_locals,
            values: Vec::new(),
            remaining_uses: compute_remaining_uses(block),
            transient_owner: alloc::vec![None; regfile.transient_count()],
        };

        let machine_params = lower.machine_params.clone();
        for (param, reg) in block.params.iter().copied().zip(machine_params.into_iter()) {
            lower.values.push(ValueLocation { value: param, reg });
            lower.mark_transient(reg, Some(param))?;
        }
        lower.release_dead_values()?;

        if is_entry {
            lower.emit_reload_cached_locals()?;
        }

        Ok(lower)
    }

    pub(super) fn machine_params(&self) -> &[MachineReg] {
        &self.machine_params
    }

    pub(super) fn take_ops(&mut self) -> Vec<MachineInst> {
        mem::take(&mut self.ops)
    }

    pub(super) fn lower_ops(&mut self) -> Result<(), WasmError> {
        for inst in &self.block.ops {
            self.lower_inst(inst)?;
        }
        Ok(())
    }

    pub(super) fn lower_terminator(&mut self) -> Result<MachineTerminator, WasmError> {
        match &self.block.terminator {
            LirTerminator::Goto(edge) => Ok(MachineTerminator::Jump(self.lower_edge(edge)?)),
            LirTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            } => {
                let cond = self.use_value(*cond)?;
                self.release_dead_values()?;
                Ok(MachineTerminator::Branch {
                    cond: MachineBranchCond::Value(MachineValue::Reg(cond)),
                    then_edge: self.lower_edge(then_edge)?,
                    else_edge: self.lower_edge(else_edge)?,
                })
            }
            LirTerminator::BrTable { index, entries } => {
                let index = self.use_value(*index)?;
                self.release_dead_values()?;
                let mut lowered = Vec::with_capacity(entries.len());
                for edge in entries {
                    lowered.push(self.lower_edge(edge)?);
                }
                Ok(MachineTerminator::JumpTable {
                    index: MachineValue::Reg(index),
                    entries: lowered,
                })
            }
            LirTerminator::Return { .. } => {
                if !self.values.is_empty() {
                    return Err(WasmError::internal(
                        "LIR return reached native lowering with live transient SSA values; results must be published before return".into(),
                    ));
                }
                Ok(MachineTerminator::Return)
            }
            LirTerminator::TrapUnreachable => Ok(MachineTerminator::Trap {
                kind: MachineTrapKind::Unreachable,
            }),
        }
    }

    pub(super) fn begin_continuation_block(&mut self) -> Result<(), WasmError> {
        self.emit_reload_cached_locals()
    }

    pub(super) fn lower_inst(&mut self, inst: &LirInst) -> Result<(), WasmError> {
        match &inst.kind {
            LirInstKind::LoadSlot { slot, dst } => {
                let dst_reg = self.alloc_value(*dst)?;
                if let Some(cached_index) = self.cached_local_index(*slot) {
                    self.emit_machine_inst(MachineInst {
                        kind: MachineInstKind::Move {
                            dst: dst_reg,
                            src: MachineValue::Reg(self.cached_locals[cached_index].reg),
                        },
                    });
                } else {
                    self.emit_machine_inst(MachineInst {
                        kind: MachineInstKind::Load {
                            dst: dst_reg,
                            addr: self.frame_addr(*slot)?,
                            width: MachineMemWidth::U64,
                            extension: MachineLoadExtension::None,
                        },
                    });
                }
            }
            LirInstKind::StoreSlot { slot, src } => {
                let src_reg = self.use_value(*src)?;
                if let Some(cached_index) = self.cached_local_index(*slot) {
                    self.emit_machine_inst(MachineInst {
                        kind: MachineInstKind::Move {
                            dst: self.cached_locals[cached_index].reg,
                            src: MachineValue::Reg(src_reg),
                        },
                    });
                } else {
                    self.emit_machine_inst(MachineInst {
                        kind: MachineInstKind::Store {
                            addr: self.frame_addr(*slot)?,
                            width: MachineMemWidth::U64,
                            src: MachineValue::Reg(src_reg),
                        },
                    });
                }
                self.release_dead_values()?;
            }
            LirInstKind::Value { op, args, results } => {
                self.lower_leaf(op, args, results)?;
                self.release_dead_values()?;
            }
            LirInstKind::Boundary(boundary) => {
                return Err(WasmError::internal(alloc::format!(
                    "boundary op {:?} must be lowered through its specialized native path",
                    boundary
                )));
            }
        }
        Ok(())
    }

    pub(super) fn lower_const(&mut self, results: &[LirValue], imm: u64) -> Result<(), WasmError> {
        let dst = single_result(results)?;
        let dst_reg = self.alloc_value(dst)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Move {
                dst: dst_reg,
                src: MachineValue::Imm64(imm),
            },
        });
        Ok(())
    }

    pub(super) fn lower_int_unary(
        &mut self,
        args: &[LirValue],
        results: &[LirValue],
        width: super::super::ir::machine::MachineIntWidth,
        op: super::super::ir::machine::MachineIntUnaryOp,
    ) -> Result<(), WasmError> {
        let src_value = single_arg(args)?;
        let src = self.use_value(src_value)?;
        let dst = self.alloc_value_reusing_dead_inputs(single_result(results)?, &[src_value])?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::IntUnary {
                width,
                op,
                dst,
                src: MachineValue::Reg(src),
            },
        });
        Ok(())
    }

    pub(super) fn lower_int_binary(
        &mut self,
        args: &[LirValue],
        results: &[LirValue],
        width: super::super::ir::machine::MachineIntWidth,
        op: super::super::ir::machine::MachineIntBinaryOp,
    ) -> Result<(), WasmError> {
        let (lhs_value, rhs_value) = two_args(args)?;
        let lhs = self.use_value(lhs_value)?;
        let rhs = self.use_value(rhs_value)?;
        let dst =
            self.alloc_value_reusing_dead_inputs(single_result(results)?, &[lhs_value, rhs_value])?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::IntBinary {
                width,
                op,
                dst,
                lhs: MachineValue::Reg(lhs),
                rhs: MachineValue::Reg(rhs),
            },
        });
        Ok(())
    }

    pub(super) fn lower_int_compare(
        &mut self,
        args: &[LirValue],
        results: &[LirValue],
        width: super::super::ir::machine::MachineIntWidth,
        kind: super::super::ir::machine::MachineCompareKind,
        sign: super::super::ir::machine::MachineSign,
    ) -> Result<(), WasmError> {
        let (lhs_value, rhs_value) = two_args(args)?;
        let lhs = self.use_value(lhs_value)?;
        let rhs = self.use_value(rhs_value)?;
        let dst =
            self.alloc_value_reusing_dead_inputs(single_result(results)?, &[lhs_value, rhs_value])?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::IntCompare {
                width,
                kind,
                sign,
                dst,
                lhs: MachineValue::Reg(lhs),
                rhs: MachineValue::Reg(rhs),
            },
        });
        Ok(())
    }

    pub(super) fn lower_float_binary(
        &mut self,
        args: &[LirValue],
        results: &[LirValue],
        width: super::super::ir::machine::MachineFloatWidth,
        op: super::super::ir::machine::MachineFloatBinaryOp,
    ) -> Result<(), WasmError> {
        let (lhs_value, rhs_value) = two_args(args)?;
        let lhs = self.use_value(lhs_value)?;
        let rhs = self.use_value(rhs_value)?;
        let dst =
            self.alloc_value_reusing_dead_inputs(single_result(results)?, &[lhs_value, rhs_value])?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::FloatBinary {
                width,
                op,
                dst,
                lhs: MachineValue::Reg(lhs),
                rhs: MachineValue::Reg(rhs),
            },
        });
        Ok(())
    }

    pub(super) fn lower_float_compare(
        &mut self,
        args: &[LirValue],
        results: &[LirValue],
        width: super::super::ir::machine::MachineFloatWidth,
        kind: super::super::ir::machine::MachineCompareKind,
    ) -> Result<(), WasmError> {
        let (lhs_value, rhs_value) = two_args(args)?;
        let lhs = self.use_value(lhs_value)?;
        let rhs = self.use_value(rhs_value)?;
        let dst =
            self.alloc_value_reusing_dead_inputs(single_result(results)?, &[lhs_value, rhs_value])?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::FloatCompare {
                width,
                kind,
                dst,
                lhs: MachineValue::Reg(lhs),
                rhs: MachineValue::Reg(rhs),
            },
        });
        Ok(())
    }

    pub(super) fn lower_float_unary(
        &mut self,
        args: &[LirValue],
        results: &[LirValue],
        width: super::super::ir::machine::MachineFloatWidth,
        op: super::super::ir::machine::MachineFloatUnaryOp,
    ) -> Result<(), WasmError> {
        let src_value = single_arg(args)?;
        let src = self.use_value(src_value)?;
        let dst = self.alloc_value_reusing_dead_inputs(single_result(results)?, &[src_value])?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::FloatUnary {
                width,
                op,
                dst,
                src: MachineValue::Reg(src),
            },
        });
        Ok(())
    }

    pub(super) fn lower_convert(
        &mut self,
        args: &[LirValue],
        results: &[LirValue],
        op: super::super::ir::machine::MachineConvertOp,
    ) -> Result<(), WasmError> {
        let src_value = single_arg(args)?;
        let src = self.use_value(src_value)?;
        let dst = self.alloc_value_reusing_dead_inputs(single_result(results)?, &[src_value])?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Convert {
                op,
                dst,
                src: MachineValue::Reg(src),
            },
        });
        Ok(())
    }

    pub(super) fn lower_select(
        &mut self,
        args: &[LirValue],
        results: &[LirValue],
    ) -> Result<(), WasmError> {
        if args.len() != 3 {
            return Err(WasmError::internal("select expects three arguments".into()));
        }
        let on_true = self.use_value(args[0])?;
        let on_false = self.use_value(args[1])?;
        let cond = self.use_value(args[2])?;
        let dst = self.alloc_value_reusing_dead_inputs(single_result(results)?, args)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Select {
                dst,
                on_true: MachineValue::Reg(on_true),
                on_false: MachineValue::Reg(on_false),
                cond: MachineValue::Reg(cond),
            },
        });
        Ok(())
    }

    pub(super) fn lower_ref_is_null(
        &mut self,
        args: &[LirValue],
        results: &[LirValue],
    ) -> Result<(), WasmError> {
        let src_value = single_arg(args)?;
        let src = self.use_value(src_value)?;
        let dst = self.alloc_value_reusing_dead_inputs(single_result(results)?, &[src_value])?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::IntCompare {
                width: super::super::ir::machine::MachineIntWidth::I64,
                kind: super::super::ir::machine::MachineCompareKind::Eq,
                sign: super::super::ir::machine::MachineSign::Unsigned,
                dst,
                lhs: MachineValue::Reg(src),
                rhs: MachineValue::Imm64(usize::MAX as u64),
            },
        });
        Ok(())
    }

    fn lower_edge(&self, edge: &LirEdge) -> Result<MachineEdge, WasmError> {
        let target_block = MachineBlockId(edge.target.as_u32());
        let target = self
            .program
            .blocks
            .get(edge.target.as_usize())
            .ok_or_else(|| {
                WasmError::internal("edge target is out of range during native lowering".into())
            })?;
        let mut args = Vec::with_capacity(target.params.len());
        for target_param in &target.params {
            let binding = edge
                .bindings
                .iter()
                .find(|binding| binding.param == *target_param)
                .ok_or_else(|| {
                    WasmError::internal(
                        "missing LIR edge binding for target param during native lowering".into(),
                    )
                })?;
            args.push(MachineValue::Reg(self.value_reg(binding.value)?));
        }
        Ok(MachineEdge {
            target: target_block,
            args,
        })
    }

    pub(super) fn emit_reload_cached_locals(&mut self) -> Result<(), WasmError> {
        for index in 0..self.cached_locals.len() {
            let reg = self.cached_locals[index].reg;
            let slot = self.cached_locals[index].slot;
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Load {
                    dst: reg,
                    addr: self.frame_addr(slot)?,
                    width: MachineMemWidth::U64,
                    extension: MachineLoadExtension::None,
                },
            });
        }
        Ok(())
    }

    pub(super) fn emit_save_all_cached_locals(&mut self) -> Result<(), WasmError> {
        for index in 0..self.cached_locals.len() {
            let cached = self.cached_locals[index];
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Store {
                    addr: self.frame_addr(cached.slot)?,
                    width: MachineMemWidth::U64,
                    src: MachineValue::Reg(cached.reg),
                },
            });
        }
        Ok(())
    }

    pub(super) fn emit_reload_mem0_cache_regs(&mut self) {
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                dst: self.regfile.mem0_base(),
                addr: self.runtime_addr(ctx_offset::MEM0_BASE),
                width: MachineMemWidth::U64,
                extension: MachineLoadExtension::None,
            },
        });
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                dst: self.regfile.mem0_size(),
                addr: self.runtime_addr(ctx_offset::MEM0_SIZE),
                width: MachineMemWidth::U64,
                extension: MachineLoadExtension::None,
            },
        });
    }

    fn cached_local_index(&self, slot: FrameSlot) -> Option<usize> {
        self.cached_locals
            .iter()
            .position(|cached| cached.slot == slot)
    }

    pub(super) fn frame_addr(&self, slot: FrameSlot) -> Result<MachineAddr, WasmError> {
        self.frame_addr_from(self.regfile.frame_base(), slot)
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

    pub(super) fn alloc_value(&mut self, value: LirValue) -> Result<MachineReg, WasmError> {
        if let Some(reg) = self.try_value_reg(value) {
            return Ok(reg);
        }
        let reg = self.first_free_transient().ok_or_else(|| {
            WasmError::internal(
                "prepared LIR exceeded transient register budget during native lowering".into(),
            )
        })?;
        self.values.push(ValueLocation { value, reg });
        self.mark_transient(reg, Some(value))?;
        Ok(reg)
    }

    pub(super) fn alloc_value_reusing_dead_inputs(
        &mut self,
        value: LirValue,
        candidates: &[LirValue],
    ) -> Result<MachineReg, WasmError> {
        if let Some(reg) = self.try_value_reg(value) {
            return Ok(reg);
        }

        for candidate in candidates {
            if self.remaining_uses.get(candidate).copied().unwrap_or(0) != 0 {
                continue;
            }
            if let Some(index) = self
                .values
                .iter()
                .position(|entry| entry.value == *candidate)
            {
                let reg = self.values[index].reg;
                self.values[index].value = value;
                self.mark_transient(reg, Some(value))?;
                return Ok(reg);
            }
        }

        self.alloc_value(value)
    }

    pub(super) fn use_value(&mut self, value: LirValue) -> Result<MachineReg, WasmError> {
        let reg = self.value_reg(value)?;
        if let Some(remaining) = self.remaining_uses.get_mut(&value) {
            *remaining = remaining.saturating_sub(1);
        }
        Ok(reg)
    }

    pub(super) fn release_dead_values(&mut self) -> Result<(), WasmError> {
        let mut index = 0;
        while index < self.values.len() {
            let value = self.values[index].value;
            let remaining = self.remaining_uses.get(&value).copied().unwrap_or(0);
            if remaining == 0 {
                let reg = self.values[index].reg;
                self.values.swap_remove(index);
                self.mark_transient(reg, None)?;
            } else {
                index += 1;
            }
        }
        Ok(())
    }

    fn try_value_reg(&self, value: LirValue) -> Option<MachineReg> {
        self.values
            .iter()
            .find(|entry| entry.value == value)
            .map(|entry| entry.reg)
    }

    pub(super) fn dead_value_reg(&self, value: LirValue) -> Option<MachineReg> {
        if self.remaining_uses.get(&value).copied().unwrap_or(0) != 0 {
            return None;
        }
        self.try_value_reg(value)
    }

    fn value_reg(&self, value: LirValue) -> Result<MachineReg, WasmError> {
        self.try_value_reg(value).ok_or_else(|| {
            WasmError::internal(alloc::format!(
                "no machine register assigned for LIR value {:?}",
                value
            ))
        })
    }

    fn first_free_transient(&self) -> Option<MachineReg> {
        for index in 0..self.transient_owner.len() {
            if self.transient_owner[index].is_none() {
                return self.regfile.transient(index);
            }
        }
        None
    }

    fn mark_transient(
        &mut self,
        reg: MachineReg,
        value: Option<LirValue>,
    ) -> Result<(), WasmError> {
        let index = reg
            .0
            .checked_sub(
                self.regfile
                    .transient(0)
                    .ok_or_else(|| {
                        WasmError::internal("native lowering has no transient registers".into())
                    })?
                    .0,
            )
            .ok_or_else(|| {
                WasmError::internal("machine register is not in transient partition".into())
            })? as usize;
        let slot = self.transient_owner.get_mut(index).ok_or_else(|| {
            WasmError::internal("transient register index is out of range".into())
        })?;
        *slot = value;
        Ok(())
    }

    pub(super) fn ensure_no_live_values(&self, message: &'static str) -> Result<(), WasmError> {
        if self.values.is_empty() {
            Ok(())
        } else {
            Err(WasmError::internal(message.into()))
        }
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

    pub(super) fn transient_reg(&self, index: usize) -> Result<MachineReg, WasmError> {
        self.regfile.transient(index).ok_or_else(|| {
            WasmError::internal("native lowering requires one transient register".into())
        })
    }

    pub(super) fn transient_in_use(&self, index: usize) -> Result<bool, WasmError> {
        self.transient_owner
            .get(index)
            .copied()
            .map(|value| value.is_some())
            .ok_or_else(|| WasmError::internal("transient register index is out of range".into()))
    }

    pub(super) fn borrow_free_transients(
        &self,
        count: usize,
    ) -> Result<Vec<MachineReg>, WasmError> {
        let mut regs = Vec::with_capacity(count);
        for index in 0..self.transient_owner.len() {
            if self.transient_owner[index].is_none() {
                regs.push(self.regfile.transient(index).ok_or_else(|| {
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

    pub(super) fn emit_machine_inst(&mut self, inst: MachineInst) {
        self.ops.push(inst);
    }

    pub(super) fn emit_machine_ops<I>(&mut self, insts: I)
    where
        I: IntoIterator<Item = MachineInst>,
    {
        self.ops.extend(insts);
    }
}
