use alloc::vec::Vec;
use core::mem;

use crate::{
    error::WasmError,
    value_type::ValueType,
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
                machine_ptr_width, machine_word_int_width, MachineAddr, MachineBlockId,
                MachineBlockParam, MachineBranchCond, MachineEdge, MachineFloatWidth,
                MachineFuncId, MachineInst, MachineInstKind, MachineIntWidth, MachineLoadExtension,
                MachineMemWidth, MachineReg, MachineStorageType, MachineTerminator,
                MachineTrapKind, MachineValue,
            },
            ir::runtime::{MachineCallLinkLayout, MachineFrameRegion, MachineFunctionRuntime},
            lower::{slot_offset_bytes, target_param_regs},
            runtime::layout::{native_runtime_abi_layout, NativeRuntimeAbiLayout},
        },
        plan::frame::FrameLayoutPlan,
    },
};

use super::{
    regfile::MachineRegFile,
    util::{compute_remaining_uses, single_arg, single_result, two_args},
};

use crate::vm::lir::ir::CachedLocalInfo;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CachedLocal {
    slot: FrameSlot,
    reg: MachineReg,
    hi_reg: Option<MachineReg>,
    ty: MachineStorageType,
    info: CachedLocalInfo,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ValueRegs {
    pub lo: MachineReg,
    pub hi: Option<MachineReg>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ValueLocation {
    value: LirValue,
    reg: MachineReg,
    hi_reg: Option<MachineReg>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct TransientState {
    value: Option<LirValue>,
    ty: Option<MachineStorageType>,
}

pub(super) struct BlockLowerContext<'a> {
    regfile: &'a MachineRegFile,
    frame: FrameLayoutPlan,
    program: &'a LirProgram,
    block: &'a LirBlock,
    runtime: MachineFunctionRuntime,
    all_runtime: &'a [MachineFunctionRuntime],
    call_link: MachineCallLinkLayout,
    machine_params: Vec<ValueRegs>,
    gp_reg_width: u8,
    ops: Vec<MachineInst>,
    cached_locals: Vec<CachedLocal>,
    values: Vec<ValueLocation>,
    remaining_uses: alloc::collections::BTreeMap<LirValue, u32>,
    transient_state: Vec<TransientState>,
    #[cfg(has_guard_pages)]
    guard_pages: bool,
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

    /// Begin a continuation block after a call, selectively skipping reloads
    /// for cached locals that are known to be written before read.
    pub(super) fn begin_continuation_block_selective(
        &mut self,
        skip_reload: Option<&[bool]>,
    ) -> Result<(), WasmError> {
        self.emit_reload_cached_locals_selective(skip_reload)
    }

    pub(super) fn lower_inst(&mut self, inst: &LirInst) -> Result<(), WasmError> {
        match &inst.kind {
            LirInstKind::LoadSlot { slot, dst } => {
                let ty = lir_value_storage_type(self.program, *dst);
                if self.gp_reg_width == 4 && matches!(ty, MachineStorageType::GpI64) {
                    let (dst_lo, dst_hi) = self.alloc_i64_value_pair(*dst)?;
                    if let Some(cached_index) = self.cached_local_index(*slot) {
                        let cached = self.cached_locals[cached_index];
                        if cached.ty != ty {
                            return Err(WasmError::internal(alloc::format!(
                                "typed LIR load from cached local slot {:?} expects {:?} for value {:?}, but cached local is {:?}",
                                slot,
                                ty,
                                dst,
                                cached.ty,
                            )));
                        }
                        let cached_hi = cached.hi_reg.ok_or_else(|| {
                            WasmError::internal(
                                "cached i64 local is missing a high-half register".into(),
                            )
                        })?;
                        self.emit_machine_inst(MachineInst {
                            kind: MachineInstKind::Move {
                                ty: MachineStorageType::GpWord,
                                dst: dst_lo,
                                src: MachineValue::Reg(cached.reg),
                            },
                        });
                        self.emit_machine_inst(MachineInst {
                            kind: MachineInstKind::Move {
                                ty: MachineStorageType::GpWord,
                                dst: dst_hi,
                                src: MachineValue::Reg(cached_hi),
                            },
                        });
                    } else {
                        self.emit_machine_inst(MachineInst {
                            kind: MachineInstKind::Load {
                                ty: MachineStorageType::GpWord,
                                dst: dst_lo,
                                addr: self.frame_addr_offset(*slot, 0)?,
                                width: MachineMemWidth::U32,
                                extension: MachineLoadExtension::None,
                            },
                        });
                        self.emit_machine_inst(MachineInst {
                            kind: MachineInstKind::Load {
                                ty: MachineStorageType::GpWord,
                                dst: dst_hi,
                                addr: self.frame_addr_offset(*slot, 4)?,
                                width: MachineMemWidth::U32,
                                extension: MachineLoadExtension::None,
                            },
                        });
                    }
                    return Ok(());
                }

                let dst_reg = self.alloc_slot_load_value(*dst)?;
                let width = canonical_value_mem_width_for_value(self.program, *dst);
                if let Some(cached_index) = self.cached_local_index(*slot) {
                    let cached = self.cached_locals[cached_index];
                    if cached.ty != ty {
                        return Err(WasmError::internal(alloc::format!(
                            "typed LIR load from cached local slot {:?} expects {:?} for value {:?}, but cached local is {:?}",
                            slot,
                            ty,
                            dst,
                            cached.ty,
                        )));
                    }
                    self.emit_machine_inst(MachineInst {
                        kind: MachineInstKind::Move {
                            ty: cached.ty,
                            dst: dst_reg,
                            src: MachineValue::Reg(cached.reg),
                        },
                    });
                } else {
                    self.emit_machine_inst(MachineInst {
                        kind: MachineInstKind::Load {
                            ty,
                            dst: dst_reg,
                            addr: self.frame_addr(*slot)?,
                            width,
                            extension: MachineLoadExtension::None,
                        },
                    });
                }
            }
            LirInstKind::StoreSlot { slot, src } => {
                let ty = lir_value_storage_type(self.program, *src);
                if self.gp_reg_width == 4 && matches!(ty, MachineStorageType::GpI64) {
                    let (src_lo, src_hi) = self.use_i64_value_pair(*src)?;
                    if let Some(cached_index) = self.cached_local_index(*slot) {
                        let cached = self.cached_locals[cached_index];
                        if cached.ty != ty {
                            return Err(WasmError::internal(alloc::format!(
                                "typed LIR store to cached local slot {:?} uses {:?} value {:?}, but cached local is {:?}",
                                slot,
                                ty,
                                src,
                                cached.ty,
                            )));
                        }
                        let cache_hi = cached.hi_reg.ok_or_else(|| {
                            WasmError::internal(
                                "cached i64 local is missing a high-half register".into(),
                            )
                        })?;
                        self.emit_machine_inst(MachineInst {
                            kind: MachineInstKind::Move {
                                ty: MachineStorageType::GpWord,
                                dst: cached.reg,
                                src: MachineValue::Reg(src_lo),
                            },
                        });
                        self.emit_machine_inst(MachineInst {
                            kind: MachineInstKind::Move {
                                ty: MachineStorageType::GpWord,
                                dst: cache_hi,
                                src: MachineValue::Reg(src_hi),
                            },
                        });
                    } else {
                        self.emit_machine_inst(MachineInst {
                            kind: MachineInstKind::Store {
                                ty: MachineStorageType::GpWord,
                                addr: self.frame_addr_offset(*slot, 0)?,
                                width: MachineMemWidth::U32,
                                src: MachineValue::Reg(src_lo),
                            },
                        });
                        self.emit_machine_inst(MachineInst {
                            kind: MachineInstKind::Store {
                                ty: MachineStorageType::GpWord,
                                addr: self.frame_addr_offset(*slot, 4)?,
                                width: MachineMemWidth::U32,
                                src: MachineValue::Reg(src_hi),
                            },
                        });
                    }
                    self.release_dead_values()?;
                    return Ok(());
                }

                let src_reg = self.use_value(*src)?;
                let width = canonical_value_mem_width_for_value(self.program, *src);
                if let Some(cached_index) = self.cached_local_index(*slot) {
                    let cached = self.cached_locals[cached_index];
                    if cached.ty != ty {
                        return Err(WasmError::internal(alloc::format!(
                            "typed LIR store to cached local slot {:?} uses {:?} value {:?}, but cached local is {:?}",
                            slot,
                            ty,
                            src,
                            cached.ty,
                        )));
                    }
                    let cache_reg = cached.reg;
                    if !self.try_coalesce_last_dst(*src, src_reg, cache_reg) {
                        self.emit_machine_inst(MachineInst {
                            kind: MachineInstKind::Move {
                                ty: cached.ty,
                                dst: cache_reg,
                                src: MachineValue::Reg(src_reg),
                            },
                        });
                    }
                } else {
                    let addr = self.frame_addr(*slot)?;
                    if !self.try_coalesce_last_store_immediate(*src, src_reg, ty, addr, width) {
                        self.emit_machine_inst(MachineInst {
                            kind: MachineInstKind::Store {
                                ty,
                                addr,
                                width,
                                src: MachineValue::Reg(src_reg),
                            },
                        });
                    }
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
                ty: lir_value_storage_type(self.program, dst),
                dst: dst_reg,
                src: MachineValue::Imm64(imm),
            },
        });
        Ok(())
    }

    pub(super) fn lower_float_const(
        &mut self,
        results: &[LirValue],
        width: MachineFloatWidth,
        bits: u64,
    ) -> Result<(), WasmError> {
        let dst = self.alloc_float_value(single_result(results)?, width)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::FloatConst { width, dst, bits },
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
        let dst = self.alloc_float_value_reusing_dead_inputs(
            single_result(results)?,
            &[lhs_value, rhs_value],
            width,
        )?;
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
        let dst = self.alloc_float_value_reusing_dead_inputs(
            single_result(results)?,
            &[src_value],
            width,
        )?;
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
        let dst = if let Some(width) = convert_result_float_width(op) {
            self.alloc_float_value_reusing_dead_inputs(
                single_result(results)?,
                &[src_value],
                width,
            )?
        } else {
            self.alloc_value_reusing_dead_inputs(single_result(results)?, &[src_value])?
        };
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
                ty: lir_value_storage_type(self.program, single_result(results)?),
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
                width: self.gp_word_int_width(),
                kind: super::super::ir::machine::MachineCompareKind::Eq,
                sign: super::super::ir::machine::MachineSign::Unsigned,
                dst,
                lhs: MachineValue::Reg(src),
                rhs: MachineValue::Imm64(self.gp_word_max_imm()),
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
            let regs = self.value_regs(binding.value)?;
            args.push(MachineValue::Reg(regs.0));
            if let Some(hi) = regs.1 {
                args.push(MachineValue::Reg(hi));
            }
        }
        Ok(MachineEdge {
            target: target_block,
            args,
        })
    }

    /// Reload all cached locals from the frame. Used after calls and at
    /// non-entry block boundaries where the cache may be stale.
    pub(super) fn emit_reload_cached_locals(&mut self) -> Result<(), WasmError> {
        self.emit_reload_cached_locals_selective(None)
    }

    /// Reload cached locals from the frame, optionally skipping locals that
    /// are known to be written before read at this continuation point.
    ///
    /// `skip_reload` is parallel to the cached_locals vec (GP then FP order).
    /// When `skip_reload[i]` is `true`, the reload for that cached local is
    /// elided because the local will be overwritten before anyone reads it.
    pub(super) fn emit_reload_cached_locals_selective(
        &mut self,
        skip_reload: Option<&[bool]>,
    ) -> Result<(), WasmError> {
        for index in 0..self.cached_locals.len() {
            if let Some(skip) = skip_reload {
                if index < skip.len() && skip[index] {
                    continue;
                }
            }
            let cached = self.cached_locals[index];
            if self.gp_reg_width == 4 && matches!(cached.ty, MachineStorageType::GpI64) {
                let cached_hi = cached.hi_reg.ok_or_else(|| {
                    WasmError::internal("cached i64 local is missing a high-half register".into())
                })?;
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Load {
                        ty: MachineStorageType::GpWord,
                        dst: cached.reg,
                        addr: self.frame_addr_offset(cached.slot, 0)?,
                        width: MachineMemWidth::U32,
                        extension: MachineLoadExtension::None,
                    },
                });
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Load {
                        ty: MachineStorageType::GpWord,
                        dst: cached_hi,
                        addr: self.frame_addr_offset(cached.slot, 4)?,
                        width: MachineMemWidth::U32,
                        extension: MachineLoadExtension::None,
                    },
                });
            } else {
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Load {
                        ty: cached.ty,
                        dst: cached.reg,
                        addr: self.frame_addr(cached.slot)?,
                        width: canonical_cached_local_mem_width(cached),
                        extension: MachineLoadExtension::None,
                    },
                });
            }
        }
        Ok(())
    }

    /// Initialize cached locals at function entry.
    ///
    /// Parameters are loaded from the frame (the caller already wrote them).
    /// Non-parameter locals that may be read before written need a zero
    /// materialisation (Wasm locals start at zero). Locals that are definitely
    /// written before any read can be left undefined; the `reads_before_write`
    /// analysis in `local_cache.rs` is a whole-function dataflow pass that is
    /// sound for this purpose on both 32-bit and 64-bit targets.
    fn emit_entry_cached_locals(&mut self) -> Result<(), WasmError> {
        for index in 0..self.cached_locals.len() {
            let cached = self.cached_locals[index];
            if cached.info.is_param {
                // Argument — caller wrote a real value, must load from frame.
                if self.gp_reg_width == 4 && matches!(cached.ty, MachineStorageType::GpI64) {
                    let cached_hi = cached.hi_reg.ok_or_else(|| {
                        WasmError::internal(
                            "cached i64 local is missing a high-half register".into(),
                        )
                    })?;
                    self.emit_machine_inst(MachineInst {
                        kind: MachineInstKind::Load {
                            ty: MachineStorageType::GpWord,
                            dst: cached.reg,
                            addr: self.frame_addr_offset(cached.slot, 0)?,
                            width: MachineMemWidth::U32,
                            extension: MachineLoadExtension::None,
                        },
                    });
                    self.emit_machine_inst(MachineInst {
                        kind: MachineInstKind::Load {
                            ty: MachineStorageType::GpWord,
                            dst: cached_hi,
                            addr: self.frame_addr_offset(cached.slot, 4)?,
                            width: MachineMemWidth::U32,
                            extension: MachineLoadExtension::None,
                        },
                    });
                } else {
                    self.emit_machine_inst(MachineInst {
                        kind: MachineInstKind::Load {
                            ty: cached.ty,
                            dst: cached.reg,
                            addr: self.frame_addr(cached.slot)?,
                            width: canonical_cached_local_mem_width(cached),
                            extension: MachineLoadExtension::None,
                        },
                    });
                }
            } else if cached.info.reads_before_write {
                // Non-param local that may be read before written (or 32-bit
                // target requiring type-defined registers) — zero the
                // register (Wasm locals are initialised to zero).
                if self.gp_reg_width == 4 && matches!(cached.ty, MachineStorageType::GpI64) {
                    let cached_hi = cached.hi_reg.ok_or_else(|| {
                        WasmError::internal(
                            "cached i64 local is missing a high-half register".into(),
                        )
                    })?;
                    self.emit_machine_inst(MachineInst {
                        kind: MachineInstKind::Move {
                            ty: MachineStorageType::GpWord,
                            dst: cached.reg,
                            src: MachineValue::Imm64(0),
                        },
                    });
                    self.emit_machine_inst(MachineInst {
                        kind: MachineInstKind::Move {
                            ty: MachineStorageType::GpWord,
                            dst: cached_hi,
                            src: MachineValue::Imm64(0),
                        },
                    });
                } else if let Some(width) = cached.ty.float_width() {
                    self.emit_machine_inst(MachineInst {
                        kind: MachineInstKind::FloatConst {
                            width,
                            dst: cached.reg,
                            bits: 0,
                        },
                    });
                } else {
                    self.emit_machine_inst(MachineInst {
                        kind: MachineInstKind::Move {
                            ty: cached.ty,
                            dst: cached.reg,
                            src: MachineValue::Imm64(0),
                        },
                    });
                }
            }
            // else: non-param, written before read — skip entirely.
        }
        Ok(())
    }

    pub(super) fn emit_save_all_cached_locals(&mut self) -> Result<(), WasmError> {
        for index in 0..self.cached_locals.len() {
            let cached = self.cached_locals[index];
            if self.gp_reg_width == 4 && matches!(cached.ty, MachineStorageType::GpI64) {
                let cached_hi = cached.hi_reg.ok_or_else(|| {
                    WasmError::internal("cached i64 local is missing a high-half register".into())
                })?;
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: self.frame_addr_offset(cached.slot, 0)?,
                        width: MachineMemWidth::U32,
                        src: MachineValue::Reg(cached.reg),
                    },
                });
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: self.frame_addr_offset(cached.slot, 4)?,
                        width: MachineMemWidth::U32,
                        src: MachineValue::Reg(cached_hi),
                    },
                });
            } else {
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Store {
                        ty: cached.ty,
                        addr: self.frame_addr(cached.slot)?,
                        width: canonical_cached_local_mem_width(cached),
                        src: MachineValue::Reg(cached.reg),
                    },
                });
            }
        }
        Ok(())
    }

    pub(super) fn emit_reload_mem0_cache_regs(&mut self) {
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                ty: MachineStorageType::GpWord,
                dst: self.regfile.mem0_base(),
                addr: self.runtime_addr(self.runtime_abi_layout().context.mem0_base_offset),
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                ty: MachineStorageType::GpWord,
                dst: self.regfile.mem0_size(),
                addr: self.runtime_addr(self.runtime_abi_layout().context.mem0_size_offset),
                width: self.gp_word_mem_width(),
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

    pub(super) fn alloc_value(&mut self, value: LirValue) -> Result<MachineReg, WasmError> {
        self.alloc_value_in_bank(value, lir_value_storage_type(self.program, value))
    }

    pub(super) fn alloc_result_value(&mut self, value: LirValue) -> Result<MachineReg, WasmError> {
        self.alloc_value_in_bank(value, lir_value_storage_type(self.program, value))
    }

    /// Allocate a LoadSlot destination in the correct bank based on the type table.
    fn alloc_slot_load_value(&mut self, value: LirValue) -> Result<MachineReg, WasmError> {
        self.alloc_value_in_bank(value, lir_value_storage_type(self.program, value))
    }

    pub(super) fn alloc_value_reusing_dead_inputs(
        &mut self,
        value: LirValue,
        candidates: &[LirValue],
    ) -> Result<MachineReg, WasmError> {
        self.alloc_value_in_bank_reusing_dead_inputs(
            value,
            candidates,
            lir_value_storage_type(self.program, value),
        )
    }

    pub(super) fn alloc_result_value_reusing_dead_inputs(
        &mut self,
        value: LirValue,
        candidates: &[LirValue],
    ) -> Result<MachineReg, WasmError> {
        self.alloc_value_in_bank_reusing_dead_inputs(
            value,
            candidates,
            lir_value_storage_type(self.program, value),
        )
    }

    pub(super) fn canonical_value_mem_width_for_value(&self, value: LirValue) -> MachineMemWidth {
        canonical_value_mem_width_for_value(self.program, value)
    }

    pub(super) fn canonical_gp_word_mem_width(&self) -> MachineMemWidth {
        canonical_storage_mem_width(MachineStorageType::GpWord)
    }

    pub(super) fn value_storage_type(&self, value: LirValue) -> MachineStorageType {
        lir_value_storage_type(self.program, value)
    }

    pub(super) fn alloc_float_value(
        &mut self,
        value: LirValue,
        width: MachineFloatWidth,
    ) -> Result<MachineReg, WasmError> {
        self.alloc_value_in_bank(value, float_storage_type(width))
    }

    pub(super) fn alloc_float_value_reusing_dead_inputs(
        &mut self,
        value: LirValue,
        candidates: &[LirValue],
        width: MachineFloatWidth,
    ) -> Result<MachineReg, WasmError> {
        self.alloc_value_in_bank_reusing_dead_inputs(value, candidates, float_storage_type(width))
    }

    pub(super) fn use_value(&mut self, value: LirValue) -> Result<MachineReg, WasmError> {
        let reg = self.value_reg(value)?;
        if let Some(remaining) = self.remaining_uses.get_mut(&value) {
            *remaining = remaining.saturating_sub(1);
        }
        Ok(reg)
    }

    pub(super) fn use_value_regs(
        &mut self,
        value: LirValue,
    ) -> Result<(MachineReg, Option<MachineReg>), WasmError> {
        let regs = self.value_regs(value)?;
        if let Some(remaining) = self.remaining_uses.get_mut(&value) {
            *remaining = remaining.saturating_sub(1);
        }
        Ok(regs)
    }

    pub(super) fn use_i64_value_pair(
        &mut self,
        value: LirValue,
    ) -> Result<(MachineReg, MachineReg), WasmError> {
        let (lo, hi) = self.use_value_regs(value)?;
        hi.map(|hi| (lo, hi)).ok_or_else(|| {
            WasmError::internal(alloc::format!(
                "LIR i64 value {:?} does not have a paired machine-register mapping",
                value
            ))
        })
    }

    /// Free transient registers for values with no remaining uses. With
    /// linear SSA, each op's inputs become dead after a single use, so this
    /// typically frees exactly the consumed operands. Must be called after
    /// the instruction's results are allocated to avoid reuse conflicts.
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

        for entry in &self.values {
            let remaining = self.remaining_uses.get(&entry.value).copied().unwrap_or(0);
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

    fn try_value_regs(&self, value: LirValue) -> Option<(MachineReg, Option<MachineReg>)> {
        self.values
            .iter()
            .find(|entry| entry.value == value)
            .map(|entry| (entry.reg, entry.hi_reg))
    }

    fn value_reg(&self, value: LirValue) -> Result<MachineReg, WasmError> {
        self.try_value_reg(value).ok_or_else(|| {
            WasmError::internal(alloc::format!(
                "no machine register assigned for LIR value {:?}",
                value
            ))
        })
    }

    fn value_regs(&self, value: LirValue) -> Result<(MachineReg, Option<MachineReg>), WasmError> {
        self.try_value_regs(value).ok_or_else(|| {
            WasmError::internal(alloc::format!(
                "no machine register pair assigned for LIR value {:?}",
                value
            ))
        })
    }

    fn alloc_value_in_bank(
        &mut self,
        value: LirValue,
        ty: MachineStorageType,
    ) -> Result<MachineReg, WasmError> {
        if let Some(reg) = self.try_value_reg(value) {
            return Ok(reg);
        }
        let Some(reg) = self.first_free_transient(ty) else {
            return Err(WasmError::internal(alloc::format!(
                "prepared LIR exceeded {} transient register budget during native lowering in block b{} for value {}",
                if ty.is_fp() { "FP" } else { "GP" },
                self.block.id.0,
                value.0,
            )));
        };
        self.values.push(ValueLocation {
            value,
            reg,
            hi_reg: None,
        });
        self.set_transient(reg, Some(value), Some(ty))?;
        Ok(reg)
    }

    pub(super) fn alloc_i64_value_pair(
        &mut self,
        value: LirValue,
    ) -> Result<(MachineReg, MachineReg), WasmError> {
        if let Some((lo, Some(hi))) = self.try_value_regs(value) {
            return Ok((lo, hi));
        }
        if self.try_value_reg(value).is_some() {
            return Err(WasmError::internal(alloc::format!(
                "LIR value {:?} already has a scalar machine-register mapping; cannot also allocate a pair",
                value
            )));
        }

        let Some((lo, hi)) = self.first_free_gp_pair_transient() else {
            return Err(WasmError::internal(alloc::format!(
                "prepared LIR exceeded GP transient pair budget during native lowering in block b{} for value {}",
                self.block.id.0,
                value.0,
            )));
        };
        self.values.push(ValueLocation {
            value,
            reg: lo,
            hi_reg: Some(hi),
        });
        // Pair-aware 32-bit lowering treats both halves as GP-word registers.
        self.set_transient(lo, Some(value), Some(MachineStorageType::GpWord))?;
        self.set_transient(hi, Some(value), Some(MachineStorageType::GpWord))?;
        Ok((lo, hi))
    }

    pub(super) fn alloc_i64_value_pair_reusing_dead_inputs(
        &mut self,
        value: LirValue,
        candidates: &[LirValue],
    ) -> Result<(MachineReg, MachineReg), WasmError> {
        if let Some((lo, Some(hi))) = self.try_value_regs(value) {
            return Ok((lo, hi));
        }

        for candidate in candidates {
            if self.remaining_uses.get(candidate).copied().unwrap_or(0) != 0 {
                continue;
            }
            if let Some(index) = self
                .values
                .iter()
                .position(|entry| entry.value == *candidate && entry.hi_reg.is_some())
            {
                let lo = self.values[index].reg;
                let hi = self.values[index]
                    .hi_reg
                    .expect("pair candidate must have hi reg");
                self.values[index].value = value;
                self.set_transient(lo, Some(value), Some(MachineStorageType::GpWord))?;
                self.set_transient(hi, Some(value), Some(MachineStorageType::GpWord))?;
                return Ok((lo, hi));
            }
        }

        for candidate in candidates {
            if self.remaining_uses.get(candidate).copied().unwrap_or(0) != 0 {
                continue;
            }
            if let Some(index) = self
                .values
                .iter()
                .position(|entry| entry.value == *candidate && entry.hi_reg.is_none())
            {
                let lo = self.values[index].reg;
                if self.is_fp_reg(lo) {
                    continue;
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
                return Ok((lo, hi));
            }
        }

        self.alloc_i64_value_pair(value)
    }

    fn alloc_value_in_bank_reusing_dead_inputs(
        &mut self,
        value: LirValue,
        candidates: &[LirValue],
        ty: MachineStorageType,
    ) -> Result<MachineReg, WasmError> {
        if let Some(reg) = self.try_value_reg(value) {
            return Ok(reg);
        }

        for candidate in candidates {
            if self.remaining_uses.get(candidate).copied().unwrap_or(0) != 0 {
                continue;
            }
            if let Some(index) = self.values.iter().position(|entry| {
                entry.value == *candidate && self.is_fp_reg(entry.reg) == ty.is_fp()
            }) {
                let reg = self.values[index].reg;
                if let Some(hi_reg) = self.values[index].hi_reg {
                    if ty.is_fp() {
                        continue;
                    }
                    self.clear_transient(hi_reg)?;
                    self.values[index].hi_reg = None;
                }
                self.values[index].value = value;
                self.set_transient(reg, Some(value), Some(ty))?;
                return Ok(reg);
            }
        }

        self.alloc_value_in_bank(value, ty)
    }

    fn first_free_transient(&self, ty: MachineStorageType) -> Option<MachineReg> {
        let start = if ty.is_fp() {
            self.regfile.gp_transient_count()
        } else {
            0
        };
        let count = if ty.is_fp() {
            self.regfile.fp_transient_count()
        } else {
            self.regfile.gp_transient_count()
        };
        for index in start..start + count {
            if self.transient_state[index].value.is_none() {
                return if ty.is_fp() {
                    self.regfile.fp_transient(index - start)
                } else {
                    self.regfile.gp_transient(index - start)
                };
            }
        }
        None
    }

    fn first_free_gp_pair_transient(&self) -> Option<(MachineReg, MachineReg)> {
        let mut first = None;
        for index in 0..self.regfile.gp_transient_count() {
            if self.transient_state[index].value.is_some() {
                continue;
            }
            let reg = self.regfile.gp_transient(index)?;
            if let Some(first_reg) = first {
                return Some((first_reg, reg));
            }
            first = Some(reg);
        }
        None
    }

    fn set_transient(
        &mut self,
        reg: MachineReg,
        value: Option<LirValue>,
        ty: Option<MachineStorageType>,
    ) -> Result<(), WasmError> {
        let index = self.transient_index(reg)?;
        let slot = self.transient_state.get_mut(index).ok_or_else(|| {
            WasmError::internal("transient register index is out of range".into())
        })?;
        *slot = TransientState { value, ty };
        Ok(())
    }

    fn clear_transient(&mut self, reg: MachineReg) -> Result<(), WasmError> {
        self.set_transient(reg, None, None)
    }

    fn is_transient_reg(&self, reg: MachineReg) -> bool {
        self.transient_index(reg).is_ok()
    }

    pub(super) fn is_fp_reg(&self, reg: MachineReg) -> bool {
        reg.0 >= self.regfile.first_fp_reg() && reg.0 < self.regfile.reg_count()
    }

    fn transient_index(&self, reg: MachineReg) -> Result<usize, WasmError> {
        if let Some(first) = self.regfile.gp_transient(0) {
            let start = first.0;
            let end = start + self.regfile.gp_transient_count() as u16;
            if reg.0 >= start && reg.0 < end {
                return Ok((reg.0 - start) as usize);
            }
        }

        if let Some(first) = self.regfile.fp_transient(0) {
            let start = first.0;
            let end = start + self.regfile.fp_transient_count() as u16;
            if reg.0 >= start && reg.0 < end {
                return Ok(self.regfile.gp_transient_count() + (reg.0 - start) as usize);
            }
        }

        Err(WasmError::internal(
            "machine register is not in transient partition".into(),
        ))
    }

    fn storage_type_for_reg(&self, reg: MachineReg) -> MachineStorageType {
        if let Ok(index) = self.transient_index(reg) {
            return self
                .transient_state
                .get(index)
                .and_then(|state| state.ty)
                .unwrap_or(MachineStorageType::GpWord);
        }
        self.cached_locals
            .iter()
            .find_map(|cached| {
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
            .unwrap_or(MachineStorageType::GpWord)
    }

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

    pub(super) fn transient_reg(&self, index: usize) -> Result<MachineReg, WasmError> {
        self.regfile.gp_transient(index).ok_or_else(|| {
            WasmError::internal("native lowering requires one transient register".into())
        })
    }

    pub(super) fn transient_in_use(&self, index: usize) -> Result<bool, WasmError> {
        self.transient_state
            .get(index)
            .map(|state| state.value.is_some())
            .ok_or_else(|| WasmError::internal("transient register index is out of range".into()))
    }

    pub(super) fn borrow_free_transients(
        &self,
        count: usize,
    ) -> Result<Vec<MachineReg>, WasmError> {
        let mut regs = Vec::with_capacity(count);
        for index in 0..self.regfile.gp_transient_count() {
            if self.transient_state[index].value.is_none() {
                regs.push(self.regfile.gp_transient(index).ok_or_else(|| {
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
    fn try_coalesce_last_dst(
        &mut self,
        src_value: LirValue,
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
        let remaining = self.remaining_uses.get(&src_value).copied().unwrap_or(0);
        if remaining != 0 {
            return false;
        }
        let Some(last) = self.ops.last_mut() else {
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
    fn try_coalesce_last_store_immediate(
        &mut self,
        src_value: LirValue,
        src_reg: MachineReg,
        ty: MachineStorageType,
        addr: MachineAddr,
        width: MachineMemWidth,
    ) -> bool {
        if !self.is_transient_reg(src_reg) {
            return false;
        }
        let remaining = self.remaining_uses.get(&src_value).copied().unwrap_or(0);
        if remaining != 0 {
            return false;
        }

        let imm = match self.ops.last().map(|inst| &inst.kind) {
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

        let _ = self.ops.pop();
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

fn machine_block_param(reg: MachineReg, ty: MachineStorageType) -> MachineBlockParam {
    match ty {
        MachineStorageType::GpWord => MachineBlockParam::gp_word(reg),
        MachineStorageType::GpI64 => MachineBlockParam::gp_i64(reg),
        MachineStorageType::Fp32 => MachineBlockParam::fp(reg, MachineFloatWidth::F32),
        MachineStorageType::Fp64 => MachineBlockParam::fp(reg, MachineFloatWidth::F64),
    }
}

fn value_type_storage_type(ty: ValueType) -> MachineStorageType {
    match ty {
        ValueType::F32 => MachineStorageType::Fp32,
        ValueType::F64 => MachineStorageType::Fp64,
        ValueType::I64 => MachineStorageType::GpI64,
        _ => MachineStorageType::GpWord,
    }
}

fn float_storage_type(width: MachineFloatWidth) -> MachineStorageType {
    match width {
        MachineFloatWidth::F32 => MachineStorageType::Fp32,
        MachineFloatWidth::F64 => MachineStorageType::Fp64,
    }
}

fn gp_reg_mem_width(gp_reg_width: u8) -> MachineMemWidth {
    machine_ptr_width(gp_reg_width)
}

fn gp_reg_int_width(gp_reg_width: u8) -> MachineIntWidth {
    machine_word_int_width(gp_reg_width)
}

fn canonical_storage_mem_width(ty: MachineStorageType) -> MachineMemWidth {
    match ty {
        MachineStorageType::GpWord | MachineStorageType::GpI64 | MachineStorageType::Fp64 => {
            MachineMemWidth::U64
        }
        MachineStorageType::Fp32 => MachineMemWidth::U32,
    }
}

fn canonical_cached_local_mem_width(cached: CachedLocal) -> MachineMemWidth {
    canonical_storage_mem_width(cached.ty)
}

fn lir_value_storage_type(program: &LirProgram, value: LirValue) -> MachineStorageType {
    program
        .value_types
        .get(value.0 as usize)
        .copied()
        .map(value_type_storage_type)
        .unwrap_or(MachineStorageType::GpWord)
}

fn canonical_value_mem_width_for_value(program: &LirProgram, value: LirValue) -> MachineMemWidth {
    canonical_storage_mem_width(lir_value_storage_type(program, value))
}

fn push_unique_param(params: &mut Vec<MachineBlockParam>, param: MachineBlockParam) {
    if !params.iter().any(|candidate| candidate.reg == param.reg) {
        params.push(param);
    }
}

pub(super) fn machine_block_params_for_value(
    regs: ValueRegs,
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

fn inst_defined_reg(kind: &MachineInstKind) -> Option<MachineReg> {
    match kind {
        MachineInstKind::Move { dst, .. }
        | MachineInstKind::FloatConst { dst, .. }
        | MachineInstKind::Lea { dst, .. }
        | MachineInstKind::Load { dst, .. }
        | MachineInstKind::IntUnary { dst, .. }
        | MachineInstKind::IntBinary { dst, .. }
        | MachineInstKind::IntCompare { dst, .. }
        | MachineInstKind::FloatUnary { dst, .. }
        | MachineInstKind::FloatBinary { dst, .. }
        | MachineInstKind::FloatCompare { dst, .. }
        | MachineInstKind::Convert { dst, .. }
        | MachineInstKind::Select { dst, .. } => Some(*dst),
        MachineInstKind::IntMulWide { .. } => None,
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
        | MachineInstKind::TrapIf { .. }
        | MachineInstKind::CallHelper(_) => None,
    }
}

fn visit_inst_source_regs(kind: &MachineInstKind, mut visit: impl FnMut(MachineReg)) {
    match kind {
        MachineInstKind::Move { src, .. } => visit_value_reg(src, &mut visit),
        MachineInstKind::FloatConst { .. } => {}
        MachineInstKind::Lea { addr, .. } | MachineInstKind::Load { addr, .. } => visit(addr.base),
        MachineInstKind::Store { addr, src, .. } => {
            visit(addr.base);
            visit_value_reg(src, &mut visit);
        }
        MachineInstKind::IntUnary { src, .. }
        | MachineInstKind::FloatUnary { src, .. }
        | MachineInstKind::Convert { src, .. } => visit_value_reg(src, &mut visit),
        MachineInstKind::IntBinary { lhs, rhs, .. }
        | MachineInstKind::IntMulWide { lhs, rhs, .. }
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
        MachineBranchCond::IntCompare { lhs, rhs, .. }
        | MachineBranchCond::FloatCompare { lhs, rhs, .. } => {
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

fn convert_result_float_width(
    op: super::super::ir::machine::MachineConvertOp,
) -> Option<MachineFloatWidth> {
    use super::super::ir::machine::MachineConvertOp as Op;

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
        | MachineInstKind::Lea { dst, .. }
        | MachineInstKind::Load { dst, .. }
        | MachineInstKind::IntUnary { dst, .. }
        | MachineInstKind::IntBinary { dst, .. }
        | MachineInstKind::IntCompare { dst, .. }
        | MachineInstKind::FloatUnary { dst, .. }
        | MachineInstKind::FloatBinary { dst, .. }
        | MachineInstKind::FloatCompare { dst, .. }
        | MachineInstKind::Convert { dst, .. }
        | MachineInstKind::Select { dst, .. } => *dst == reg,
        MachineInstKind::IntMulWide { dst_lo, dst_hi, .. } => *dst_lo == reg || *dst_hi == reg,
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
        MachineInstKind::Lea { addr, .. } | MachineInstKind::Load { addr, .. } => addr.base == reg,
        MachineInstKind::Store { addr, src, .. } => addr.base == reg || is(src),
        MachineInstKind::IntUnary { src, .. }
        | MachineInstKind::FloatUnary { src, .. }
        | MachineInstKind::Convert { src, .. } => is(src),
        MachineInstKind::IntBinary { lhs, rhs, .. }
        | MachineInstKind::IntMulWide { lhs, rhs, .. }
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
        | MachineInstKind::Lea { dst, .. }
        | MachineInstKind::Load { dst, .. }
        | MachineInstKind::IntUnary { dst, .. }
        | MachineInstKind::IntBinary { dst, .. }
        | MachineInstKind::IntCompare { dst, .. }
        | MachineInstKind::FloatUnary { dst, .. }
        | MachineInstKind::FloatBinary { dst, .. }
        | MachineInstKind::FloatCompare { dst, .. }
        | MachineInstKind::Convert { dst, .. }
        | MachineInstKind::Select { dst, .. } => *dst = new_dst,
        MachineInstKind::IntMulWide { .. }
        | MachineInstKind::Int64PairBinary { .. }
        | MachineInstKind::Int64PairUnary { .. }
        | MachineInstKind::Int64PairDivRem { .. }
        | MachineInstKind::Int64PairShift { .. }
        | MachineInstKind::Int64PairCompare { .. }
        | MachineInstKind::ConvertFloatToI64Pair { .. }
        | MachineInstKind::ReinterpretF64ToI64Pair { .. } => {}
        MachineInstKind::ConvertI64PairToFloat { dst, .. }
        | MachineInstKind::ReinterpretI64PairToF64 { dst, .. } => *dst = new_dst,
        MachineInstKind::Store { .. }
        | MachineInstKind::TrapIf { .. }
        | MachineInstKind::CallHelper(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use alloc::boxed::Box;

    use super::*;
    use crate::vm::{
        backend::BackendConfig,
        lir::{
            ir::{LirBlock, LirLocalCachePrefs, LirProgram, LirTerminator, LirValue},
            target::LirTarget,
        },
        native::ir::runtime::MachineCallLinkLayout,
        plan::frame::plan_frame_layout,
    };

    fn make_test_context(value_types: Vec<ValueType>) -> BlockLowerContext<'static> {
        let frame = plan_frame_layout(0, 4, 0);
        let program = Box::leak(Box::new(LirProgram {
            entry: LirTarget(0),
            local_cache: LirLocalCachePrefs::default(),
            blocks: alloc::vec![LirBlock {
                id: LirTarget(0),
                params: alloc::vec![],
                ops: alloc::vec![],
                terminator: LirTerminator::Return { results: None },
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
            frame,
            program,
            &program.local_cache,
            &program.blocks[0],
            runtime,
            all_runtime,
            call_link,
            4,
            true,
            #[cfg(has_guard_pages)]
            false,
        )
        .expect("lower context")
    }

    #[test]
    fn alloc_i64_value_pair_reserves_two_gp_word_transients() {
        let mut lower = make_test_context(alloc::vec![ValueType::I64]);
        let (lo, hi) = lower.alloc_i64_value_pair(LirValue(0)).expect("pair alloc");
        assert_ne!(lo, hi);
        assert_eq!(
            lower.use_i64_value_pair(LirValue(0)).expect("pair use"),
            (lo, hi)
        );
        assert_eq!(lower.storage_type_for_reg(lo), MachineStorageType::GpWord);
        assert_eq!(lower.storage_type_for_reg(hi), MachineStorageType::GpWord);

        let lo_index = lower.transient_index(lo).expect("lo transient");
        let hi_index = lower.transient_index(hi).expect("hi transient");
        assert_eq!(lower.transient_state[lo_index].value, Some(LirValue(0)));
        assert_eq!(lower.transient_state[hi_index].value, Some(LirValue(0)));

        lower.release_dead_values().expect("release pair");
        assert!(lower.try_value_regs(LirValue(0)).is_none());
        assert!(lower.transient_state[lo_index].value.is_none());
        assert!(lower.transient_state[hi_index].value.is_none());
    }

    #[test]
    fn scalar_reuse_can_claim_low_half_of_dead_pair_and_frees_high_half() {
        let mut lower = make_test_context(alloc::vec![ValueType::I64, ValueType::I32]);
        let (pair_lo, pair_hi) = lower.alloc_i64_value_pair(LirValue(0)).expect("pair alloc");
        let scalar = lower
            .alloc_value_in_bank_reusing_dead_inputs(
                LirValue(1),
                &[LirValue(0)],
                MachineStorageType::GpWord,
            )
            .expect("scalar alloc");

        assert_eq!(scalar, pair_lo);
        assert_eq!(lower.try_value_regs(LirValue(0)), None);
        assert_eq!(lower.try_value_regs(LirValue(1)), Some((pair_lo, None)));
        let hi_index = lower.transient_index(pair_hi).expect("hi transient");
        assert!(lower.transient_state[hi_index].value.is_none());
    }

    #[test]
    fn pair_reuse_can_claim_low_half_of_dead_scalar_and_allocate_only_high_half() {
        let mut lower = make_test_context(alloc::vec![ValueType::I32, ValueType::I64]);
        let scalar = lower
            .alloc_value_in_bank(LirValue(0), MachineStorageType::GpWord)
            .expect("scalar alloc");
        let (pair_lo, pair_hi) = lower
            .alloc_i64_value_pair_reusing_dead_inputs(LirValue(1), &[LirValue(0)])
            .expect("pair alloc reusing dead scalar");

        assert_eq!(pair_lo, scalar);
        assert_ne!(pair_lo, pair_hi);
        assert_eq!(lower.try_value_regs(LirValue(0)), None);
        assert_eq!(
            lower.try_value_regs(LirValue(1)),
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
