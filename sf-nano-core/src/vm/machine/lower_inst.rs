//! Instruction dispatch — lower_ops, lower_terminator, lower_inst, lower_leaf,
//! lower_leaf_special, lower_edge.

use crate::collections;

use crate::{
    error::WasmError,
    opcodes::OpcodeFD,
    vm::{
        machine::machine_ir::{
            MachineBlockId, MachineBranchCond, MachineEdge, MachineFloatWidth, MachineInst,
            MachineInstKind, MachineIntWidth, MachineLoadExtension, MachineMemWidth, MachineReg,
            MachineRegOwner, MachineResultSrc, MachineReturnAbi, MachineReturnValue,
            MachineStorageType, MachineTerminator, MachineTrapKind, MachineValue,
        },
        middle::{
            cell::CellId,
            frame::FrameSlot,
            ssa_ir::ir::{
                DecodedOperand, SsaEdge, SsaInst, SsaInstView, SsaOp, SsaOperand,
                SsaScalarResultLoc, SsaTerminator, SsaValue,
            },
        },
        wasm::primitive_op::PrimitiveOpKind,
    },
};

use super::{
    lower_context::BlockLowerContext,
    lower_regalloc::{canonical_value_mem_width_for_value, lir_value_storage_type},
    lower_util::{five_args, four_args, single_arg, single_result, three_args, two_args},
};

pub(super) enum LeafLowering {
    InPlace,
    Split {
        continuation: MachineBlockId,
        trap: MachineBlockId,
        trap_kind: MachineTrapKind,
        terminator: MachineTerminator,
        continuation_ops: collections::Vec<MachineInst>,
    },
}

impl<'a> BlockLowerContext<'a> {
    pub(super) fn lower_terminator(&mut self) -> Result<MachineTerminator, WasmError> {
        // Clone the terminator to avoid borrow conflict with self.
        let terminator = self.block().terminator.clone();
        match &terminator {
            SsaTerminator::Goto(edge) => Ok(MachineTerminator::Jump(self.lower_edge(edge)?)),
            SsaTerminator::Branch {
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
            SsaTerminator::BrTable { index, entries } => {
                let index = self.use_value(*index)?;
                self.release_dead_values()?;
                let mut lowered = collections::Vec::with_capacity(entries.len());
                for edge in entries {
                    lowered.push(self.lower_edge(edge)?);
                }
                Ok(MachineTerminator::JumpTable {
                    index: MachineValue::Reg(index),
                    entries: lowered,
                })
            }
            SsaTerminator::Return { .. } => {
                if self.values_iter().next().is_some() {
                    return Err(WasmError::internal(
                        "SSA-IR return reached native lowering with live linear SSA values; results must be published before return".into(),
                    ));
                }
                Ok(MachineTerminator::Return)
            }
            SsaTerminator::ReturnScalar { result, .. } => {
                if matches!(
                    self.current_return_abi(),
                    MachineReturnAbi::ScalarGp { .. }
                        | MachineReturnAbi::ScalarGpPair
                        | MachineReturnAbi::ScalarFp { .. }
                ) {
                    let return_value = self.lower_scalar_return_value(result)?;
                    self.release_dead_values()?;
                    Ok(MachineTerminator::ReturnScalar {
                        value: return_value,
                    })
                } else {
                    self.publish_scalar_return_to_frame(result)?;
                    self.release_dead_values()?;
                    Ok(MachineTerminator::Return)
                }
            }
            SsaTerminator::TailCallDirect { .. }
            | SsaTerminator::TailCallIndirect { .. }
            | SsaTerminator::TailCallRef { .. } => Err(WasmError::internal(
                "SSA-IR tail calls must be lowered in lower_module before generic terminator lowering",
            )),
            SsaTerminator::TrapUnreachable => Ok(MachineTerminator::Trap {
                kind: MachineTrapKind::Unreachable,
            }),
            SsaTerminator::EhThrow { .. } | SsaTerminator::EhThrowRef { .. } => {
                Err(WasmError::internal(
                    "EhThrow/EhThrowRef must be lowered by lower_module before generic terminator lowering",
                ))
            }
        }
    }

    fn lower_scalar_return_value(
        &mut self,
        result: &SsaScalarResultLoc,
    ) -> Result<MachineReturnValue, WasmError> {
        let ty = match result {
            SsaScalarResultLoc::Stack { ty, .. } | SsaScalarResultLoc::Live { ty, .. } => *ty,
        };
        let storage = crate::vm::machine::lower_regalloc::value_type_storage_type(ty);
        match storage {
            MachineStorageType::GpWord => Ok(MachineReturnValue::ScalarGp {
                src: self.scalar_result_src(result)?,
                ty: MachineStorageType::GpWord,
            }),
            MachineStorageType::GpI64 if self.gp_reg_width() == 4 => {
                let (lo, hi) = self.scalar_result_src_pair(result)?;
                Ok(MachineReturnValue::ScalarGpPair { lo, hi })
            }
            MachineStorageType::GpI64 => Ok(MachineReturnValue::ScalarGp {
                src: self.scalar_result_src(result)?,
                ty: MachineStorageType::GpI64,
            }),
            MachineStorageType::Fp32 => Ok(MachineReturnValue::ScalarFp {
                src: self.scalar_result_src(result)?,
                ty: MachineStorageType::Fp32,
            }),
            MachineStorageType::Fp64 => Ok(MachineReturnValue::ScalarFp {
                src: self.scalar_result_src(result)?,
                ty: MachineStorageType::Fp64,
            }),
            MachineStorageType::V128 => Err(WasmError::internal(
                "v128 scalar return reached machine lowering".into(),
            )),
        }
    }

    fn scalar_result_src(
        &mut self,
        result: &SsaScalarResultLoc,
    ) -> Result<MachineResultSrc, WasmError> {
        match result {
            SsaScalarResultLoc::Stack { slot, .. } => Ok(MachineResultSrc::FrameSlot(*slot)),
            SsaScalarResultLoc::Live { value, .. } => {
                Ok(MachineResultSrc::Reg(self.use_value(*value)?))
            }
        }
    }

    fn scalar_result_src_pair(
        &mut self,
        result: &SsaScalarResultLoc,
    ) -> Result<(MachineResultSrc, MachineResultSrc), WasmError> {
        match result {
            SsaScalarResultLoc::Stack { slot, .. } => Ok((
                MachineResultSrc::FrameSlotOffset {
                    slot: *slot,
                    byte_offset: 0,
                },
                MachineResultSrc::FrameSlotOffset {
                    slot: *slot,
                    byte_offset: 4,
                },
            )),
            SsaScalarResultLoc::Live { value, .. } => {
                let (lo, hi) = self.use_i64_value_pair(*value)?;
                Ok((MachineResultSrc::Reg(lo), MachineResultSrc::Reg(hi)))
            }
        }
    }

    fn publish_scalar_return_to_frame(
        &mut self,
        result: &SsaScalarResultLoc,
    ) -> Result<(), WasmError> {
        match result {
            SsaScalarResultLoc::Live { value, .. } => {
                self.publish_value_to_frame_slot(*value, FrameSlot(0))?;
            }
            SsaScalarResultLoc::Stack { slot, ty } if *slot != FrameSlot(0) => {
                let storage = crate::vm::machine::lower_regalloc::value_type_storage_type(*ty);
                let regs = self.borrow_free_gp_dynamic_regs(2)?;
                let tmp = regs[0];
                let tmp_hi = regs[1];
                if storage == MachineStorageType::GpI64 && self.gp_reg_width() == 4 {
                    self.emit_machine_inst(MachineInst {
                        kind: MachineInstKind::Load {
                            owner: MachineRegOwner::LinearValue,
                            ty: MachineStorageType::GpWord,
                            dst: tmp,
                            addr: self.frame_addr_offset(*slot, 0)?,
                            width: MachineMemWidth::U32,
                            extension: MachineLoadExtension::None,
                        },
                    });
                    self.emit_machine_inst(MachineInst {
                        kind: MachineInstKind::Load {
                            owner: MachineRegOwner::LinearValue,
                            ty: MachineStorageType::GpWord,
                            dst: tmp_hi,
                            addr: self.frame_addr_offset(*slot, 4)?,
                            width: MachineMemWidth::U32,
                            extension: MachineLoadExtension::None,
                        },
                    });
                    self.emit_machine_inst(MachineInst {
                        kind: MachineInstKind::Store {
                            ty: MachineStorageType::GpWord,
                            addr: self.frame_addr_offset(FrameSlot(0), 0)?,
                            width: MachineMemWidth::U32,
                            src: MachineValue::Reg(tmp),
                        },
                    });
                    self.emit_machine_inst(MachineInst {
                        kind: MachineInstKind::Store {
                            ty: MachineStorageType::GpWord,
                            addr: self.frame_addr_offset(FrameSlot(0), 4)?,
                            width: MachineMemWidth::U32,
                            src: MachineValue::Reg(tmp_hi),
                        },
                    });
                } else {
                    if matches!(storage, MachineStorageType::Fp32 | MachineStorageType::Fp64) {
                        return Err(WasmError::internal(
                            "stack-resident FP scalar return reached frame fallback".into(),
                        ));
                    }
                    let width = match storage {
                        _ => MachineMemWidth::U64,
                    };
                    self.emit_machine_inst(MachineInst {
                        kind: MachineInstKind::Load {
                            owner: MachineRegOwner::LinearValue,
                            ty: storage,
                            dst: tmp,
                            addr: self.frame_addr(*slot)?,
                            width,
                            extension: MachineLoadExtension::None,
                        },
                    });
                    self.emit_machine_inst(MachineInst {
                        kind: MachineInstKind::Store {
                            ty: storage,
                            addr: self.frame_addr(FrameSlot(0))?,
                            width,
                            src: MachineValue::Reg(tmp),
                        },
                    });
                }
            }
            SsaScalarResultLoc::Stack { .. } => {}
        }
        Ok(())
    }

    /// Continuation blocks start with no live cached locals in the explicit
    /// cache model; any re-caching must already be present in SSA-IR.
    pub(super) fn begin_continuation_block_selective(&mut self) -> Result<(), WasmError> {
        self.clear_cache_live();
        self.clear_cache_dirty();
        Ok(())
    }

    /// Pre-map a sunk result to its cache register so the lowering
    /// writes directly there instead of allocating a fresh linear-value
    /// register.
    ///
    /// Must be called before any register allocation for the result
    /// (i.e. before both `lower_leaf_special` and `lower_inst`).
    pub(super) fn apply_sink_premap(
        &mut self,
        args: &[SsaOperand],
        results: &[SsaValue],
    ) -> Result<(), WasmError> {
        if results.len() != 1 {
            return Ok(());
        }
        let result = results[0];
        let Some(sink_slot) = self.program().value_sink(result) else {
            return Ok(());
        };
        let Some(cached_index) = self.cached_cell_index(sink_slot) else {
            return Ok(());
        };
        // Only premap when the cache is already bound. If not yet bound,
        // skip — the later CellSetCache will handle binding via
        // try_bind_cached_cell_from_dying_value without needing an
        // extra register.
        let Some(cached) = self.bound_cached_cell(cached_index) else {
            return Ok(());
        };
        let cache_reg = cached.reg;
        let cache_hi_reg = cached.hi_reg;
        let mut arg_vals = [SsaValue(u32::MAX); 4];
        let mut n = 0;
        for a in args.iter() {
            if let DecodedOperand::Value(v) = a.decode() {
                if n < arg_vals.len() {
                    arg_vals[n] = v;
                    n += 1;
                }
            }
        }
        self.materialize_cache_aliases(cache_reg, &arg_vals[..n])?;
        if let Some(hi) = cache_hi_reg {
            self.materialize_cache_aliases(hi, &arg_vals[..n])?;
        }
        self.push_value_location(result, cache_reg, cache_hi_reg);
        Ok(())
    }

    pub(super) fn lower_inst(&mut self, inst: &SsaInst) -> Result<(), WasmError> {
        // Dispatch on the raw opcode so we can call `&mut self` methods in
        // each arm without fighting the view borrow (`SsaInstView` for `Call`
        // and `Value` borrows both `&SsaProgram` and `&SsaBlock`).
        match inst.op {
            SsaOp::CELL_GET_SLOT => {
                let cell = CellId(inst.meta);
                let dst = inst.result;
                self.lower_cell_get_slot(cell, dst)?;
            }
            SsaOp::CELL_GET_CACHE => {
                let cell = CellId(inst.meta);
                let dst = inst.result;
                self.lower_cell_get_cache(cell, dst)?;
            }
            SsaOp::CELL_SET_SLOT => {
                let cell = CellId(inst.meta);
                let src = inst.args[0]
                    .as_value()
                    .expect("CellSetSlot src must be a Value");
                self.lower_cell_set_slot(cell, src)?;
            }
            SsaOp::CELL_SET_CACHE => {
                let cell = CellId(inst.meta);
                let src = inst.args[0]
                    .as_value()
                    .expect("CellSetCache src must be a Value");
                self.lower_cell_set_cache(cell, src)?;
            }
            SsaOp::CELL_ENSURE_CACHE => {
                let cell = CellId(inst.meta);
                self.lower_cell_ensure_cache(cell)?;
            }
            SsaOp::CELL_RESERVE_CACHE => {
                let cell = CellId(inst.meta);
                self.lower_cell_reserve_cache(cell)?;
            }
            SsaOp::CELL_DROP_CACHE => {
                let cell = CellId(inst.meta);
                if let Some(index) = self.cached_cell_index(cell) {
                    self.emit_drop_cached_cell(index)?;
                }
            }
            SsaOp::FILL => {
                let slot = FrameSlot(inst.meta);
                let dst = inst.result;
                let ty = lir_value_storage_type(self.program(), dst);
                self.apply_sink_premap(&[], &[dst])?;
                if matches!(ty, MachineStorageType::GpI64) {
                    let ops = self.i64_ops();
                    ops.emit_load_slot_i64(self, slot, dst)?;
                    return Ok(());
                }
                let dst_reg = self.alloc_slot_load_value(dst)?;
                #[cfg(sf_has_simd)]
                if matches!(ty, MachineStorageType::V128) {
                    self.emit_load_frame_v128(slot, dst_reg, MachineRegOwner::LinearValue)?;
                    return Ok(());
                }
                let width = canonical_value_mem_width_for_value(self.program(), dst);
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty,
                        dst: dst_reg,
                        addr: self.frame_addr(slot)?,
                        width,
                        extension: MachineLoadExtension::None,
                    },
                });
            }
            SsaOp::SPILL => {
                let slot = FrameSlot(inst.meta);
                let src = inst.args[0].as_value().expect("Spill src must be a Value");
                let ty = lir_value_storage_type(self.program(), src);
                if matches!(ty, MachineStorageType::GpI64) {
                    let ops = self.i64_ops();
                    ops.emit_store_slot_i64(self, slot, src)?;
                    return Ok(());
                }
                #[cfg(sf_has_simd)]
                if matches!(ty, MachineStorageType::V128) {
                    let src_reg = self.use_value(src)?;
                    self.emit_store_frame_v128(slot, src_reg)?;
                    self.release_dead_values()?;
                    return Ok(());
                }
                let src_reg = self.use_value(src)?;
                let width = canonical_value_mem_width_for_value(self.program(), src);
                let addr = self.frame_addr(slot)?;
                if !self.try_coalesce_last_store_immediate(src, src_reg, ty, addr, width) {
                    self.emit_machine_inst(MachineInst {
                        kind: MachineInstKind::Store {
                            ty,
                            addr,
                            width,
                            src: MachineValue::Reg(src_reg),
                        },
                    });
                }
                self.release_dead_values()?;
            }
            SsaOp::CALL => {
                return Err(WasmError::internal(
                    "call op must be lowered through its specialized path",
                ));
            }
            _ => {
                // Primitive (Value) op. Copy op/args/result out of the pools
                // before touching &mut self so the borrow of program ends.
                let (op, result, args_vec) = {
                    let program = self.program();
                    let block = self.block();
                    let view = inst.view(program, block);
                    match view {
                        SsaInstView::Value { op, result, args } => {
                            (op.clone(), result, args.to_vec())
                        }
                        _ => unreachable!("SsaOp classified as primitive but view is not Value"),
                    }
                };
                let results_vec: crate::collections::Vec<SsaValue> = if result.is_some() {
                    crate::collections::vec![result]
                } else {
                    crate::collections::Vec::new()
                };
                // Sink pre-mapping is now applied by the caller
                // (lower_module.rs) before dispatching to lower_inst.
                self.lower_leaf(&op, &args_vec, &results_vec)?;
                self.release_dead_values()?;
            }
        }
        Ok(())
    }

    fn ensure_cached_cell_loaded(
        &mut self,
        cell: CellId,
        cached_index: usize,
        ty: MachineStorageType,
    ) -> Result<(), WasmError> {
        if self.is_cache_live(cached_index) && self.cache_has_value(cached_index) {
            return Ok(());
        }
        if let Some((lo, hi, param_ty)) = self.register_param_state(cell) {
            if param_ty != ty {
                return Err(WasmError::internal(
                    "register-passed parameter type does not match cached local type",
                ));
            }
            self.bind_register_param_cached_cell(cached_index, lo, hi)?;
            self.set_cache_live(cached_index, true);
            self.set_cache_has_value(cached_index, true);
            // The frame slot is not authoritative yet. Mark dirty so any
            // boundary/drop that needs a frame view publishes the incoming arg.
            self.set_cache_dirty(cached_index, true);
            self.mark_param_frame_authoritative(cell);
            return Ok(());
        }
        let cached = self.ensure_bound_cached_cell(cached_index)?;
        if cached.ty != ty {
            return Err(WasmError::internal(
                "typed SSA-IR cache load from local slot expects , but cached local is",
            ));
        }
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
                        width: super::lower_regalloc::canonical_cached_cell_mem_width(cached.ty),
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
                    width: super::lower_regalloc::canonical_cached_cell_mem_width(cached.ty),
                    extension: MachineLoadExtension::None,
                },
            });
        }
        self.set_cache_live(cached_index, true);
        self.set_cache_has_value(cached_index, true);
        self.set_cache_dirty(cached_index, false);
        Ok(())
    }

    fn lower_cell_get_slot(&mut self, cell: CellId, dst: SsaValue) -> Result<(), WasmError> {
        let ty = lir_value_storage_type(self.program(), dst);
        let home = self.cell_home(cell)?;
        if let Some((lo, hi, param_ty)) = self.register_param_state(cell) {
            if param_ty == ty {
                self.push_value_location(dst, lo, hi);
                return Ok(());
            }
        }
        if matches!(ty, MachineStorageType::GpI64) && self.gp_reg_width() == 4 {
            let ops = self.i64_ops();
            ops.emit_load_slot_i64(self, home, dst)?;
            return Ok(());
        }
        let dst_reg = self.alloc_slot_load_value(dst)?;
        #[cfg(sf_has_simd)]
        if matches!(ty, MachineStorageType::V128) {
            self.emit_load_frame_v128(home, dst_reg, MachineRegOwner::LinearValue)?;
            return Ok(());
        }
        let width = canonical_value_mem_width_for_value(self.program(), dst);
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty,
                dst: dst_reg,
                addr: self.frame_addr(home)?,
                width,
                extension: MachineLoadExtension::None,
            },
        });
        Ok(())
    }

    fn lower_cell_get_cache(&mut self, cell: CellId, dst: SsaValue) -> Result<(), WasmError> {
        let ty = lir_value_storage_type(self.program(), dst);
        let Some(cached_index) = self.cached_cell_index(cell) else {
            return Err(WasmError::internal("CellGetCache on non-cached local slot"));
        };
        self.ensure_cached_cell_loaded(cell, cached_index, ty)?;
        let cached = self.ensure_bound_cached_cell(cached_index)?;
        if matches!(ty, MachineStorageType::GpI64) && self.gp_reg_width() == 4 {
            let cached_hi = cached.hi_reg.ok_or_else(|| {
                WasmError::internal("cached i64 local is missing a high-half register")
            })?;
            self.push_value_location(dst, cached.reg, Some(cached_hi));
            return Ok(());
        }

        // Source-alias: map value directly to cache register, no emit.
        // materialize_cache_aliases will copy aliased values out before the
        // cache register is overwritten by a later CellSetCache or drop.
        self.push_value_location(dst, cached.reg, None);
        Ok(())
    }

    fn lower_cell_ensure_cache(&mut self, cell: CellId) -> Result<(), WasmError> {
        let Some(cached_index) = self.cached_cell_index(cell) else {
            return Err(WasmError::internal(
                "CellEnsureCache on non-cached local slot",
            ));
        };
        let ty = self
            .bound_cached_cell(cached_index)
            .map(|cached| cached.ty)
            .unwrap_or_else(|| self.cached_cells()[cached_index].ty);
        self.ensure_cached_cell_loaded(cell, cached_index, ty)
    }

    fn lower_cell_reserve_cache(&mut self, cell: CellId) -> Result<(), WasmError> {
        let Some(cached_index) = self.cached_cell_index(cell) else {
            return Err(WasmError::internal(
                "CellReserveCache on non-cached local slot",
            ));
        };
        self.ensure_bound_cached_cell(cached_index)?;
        self.mark_param_frame_authoritative(cell);
        self.set_cache_live(cached_index, true);
        self.set_cache_has_value(cached_index, false);
        self.set_cache_dirty(cached_index, false);
        Ok(())
    }

    fn lower_cell_set_slot(&mut self, cell: CellId, src: SsaValue) -> Result<(), WasmError> {
        let ty = lir_value_storage_type(self.program(), src);
        let home = self.cell_home(cell)?;
        if matches!(ty, MachineStorageType::GpI64) && self.gp_reg_width() == 4 {
            let ops = self.i64_ops();
            ops.emit_store_slot_i64(self, home, src)?;
            self.mark_param_frame_authoritative(cell);
            return Ok(());
        }
        #[cfg(sf_has_simd)]
        if matches!(ty, MachineStorageType::V128) {
            let src_reg = self.use_value(src)?;
            self.emit_store_frame_v128(home, src_reg)?;
            self.release_dead_values()?;
            self.mark_param_frame_authoritative(cell);
            return Ok(());
        }
        let src_reg = self.use_value(src)?;
        let width = canonical_value_mem_width_for_value(self.program(), src);
        let addr = self.frame_addr(home)?;
        if !self.try_coalesce_last_store_immediate(src, src_reg, ty, addr, width) {
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Store {
                    ty,
                    addr,
                    width,
                    src: MachineValue::Reg(src_reg),
                },
            });
        }
        self.release_dead_values()?;
        self.mark_param_frame_authoritative(cell);
        Ok(())
    }

    fn lower_cell_set_cache(&mut self, cell: CellId, src: SsaValue) -> Result<(), WasmError> {
        let ty = lir_value_storage_type(self.program(), src);
        let Some(cached_index) = self.cached_cell_index(cell) else {
            return Err(WasmError::internal("CellSetCache on non-cached local slot"));
        };
        let cached = self
            .try_bind_cached_cell_from_dying_value(cached_index, src, ty)?
            .unwrap_or(
                self.ensure_bound_cached_cell(cached_index)
                    .map_err(|_err| {
                        WasmError::internal(
                            "CellSetCache(slot=, src=, remaining_uses=) in block b failed",
                        )
                    })?,
            );
        if cached.ty != ty {
            return Err(WasmError::internal(
                "typed SSA-IR store to cached local slot uses value , but cached local is",
            ));
        }
        self.set_cache_live(cached_index, true);
        self.set_cache_has_value(cached_index, true);
        self.mark_cache_dirty(cached_index);
        self.mark_param_frame_authoritative(cell);

        if matches!(ty, MachineStorageType::GpI64) && self.gp_reg_width() == 4 {
            let cached_hi = cached.hi_reg.ok_or_else(|| {
                WasmError::internal("cached i64 local is missing a high-half register")
            })?;
            self.materialize_cache_aliases(cached.reg, &[src])?;
            let (src_lo, src_hi) = self.use_i64_value_pair(src)?;
            if src_lo == cached.reg && src_hi == cached_hi {
                self.release_dead_values()?;
                return Ok(());
            }
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Move {
                    owner: MachineRegOwner::CachedCell,
                    ty: MachineStorageType::GpWord,
                    dst: cached.reg,
                    src: MachineValue::Reg(src_lo),
                },
            });
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Move {
                    owner: MachineRegOwner::CachedCell,
                    ty: MachineStorageType::GpWord,
                    dst: cached_hi,
                    src: MachineValue::Reg(src_hi),
                },
            });
            self.release_dead_values()?;
            return Ok(());
        }

        let cache_reg = cached.reg;
        if self.try_value_reg(src) == Some(cache_reg) {
            let _ = self.use_value(src)?;
            self.release_dead_values()?;
            return Ok(());
        }
        self.materialize_cache_aliases(cache_reg, &[])?;
        let src_reg = self.use_value(src)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Move {
                owner: MachineRegOwner::CachedCell,
                ty: cached.ty,
                dst: cache_reg,
                src: MachineValue::Reg(src_reg),
            },
        });
        self.release_dead_values()?;
        Ok(())
    }

    pub(super) fn lower_leaf_special(
        &mut self,
        op: &PrimitiveOpKind,
        args: &[SsaOperand],
        results: &[SsaValue],
        continuation: MachineBlockId,
        trap: MachineBlockId,
    ) -> Result<Option<LeafLowering>, WasmError> {
        use PrimitiveOpKind as P;

        let lowered = match op {
            P::MemorySize { mem_idx } => {
                self.lower_memory_size(*mem_idx, results)?;
                LeafLowering::InPlace
            }
            P::MemoryGrow { mem_idx } => {
                self.lower_memory_grow(*mem_idx, args, results)?;
                LeafLowering::InPlace
            }
            P::MemoryFill { imm0, .. } => {
                self.lower_memory_fill(*imm0, args)?;
                LeafLowering::InPlace
            }
            P::MemoryCopy { imm0, imm1 } => {
                self.lower_memory_copy(*imm0, *imm1, args)?;
                LeafLowering::InPlace
            }
            P::MemoryInit { imm0, imm1 } => {
                self.lower_memory_init(*imm0, *imm1, args)?;
                LeafLowering::InPlace
            }
            P::DataDrop { data_idx } => {
                self.lower_data_drop(*data_idx)?;
                LeafLowering::InPlace
            }
            P::TableGrow { table_idx } => {
                self.lower_table_grow(*table_idx, args, results)?;
                LeafLowering::InPlace
            }
            P::TableFill { imm0, .. } => {
                self.lower_table_fill(*imm0, args)?;
                LeafLowering::InPlace
            }
            P::TableCopy { imm0, imm1 } => {
                self.lower_table_copy(*imm0, *imm1, args)?;
                LeafLowering::InPlace
            }
            P::TableInit { imm0, imm1 } => {
                self.lower_table_init(*imm0, *imm1, args)?;
                LeafLowering::InPlace
            }
            P::ElemDrop { elem_idx } => {
                self.lower_elem_drop(*elem_idx)?;
                LeafLowering::InPlace
            }
            P::EhAllocExnRef { tag_idx } => {
                self.lower_eh_alloc_exn_ref(*tag_idx, results)?;
                LeafLowering::InPlace
            }
            P::GlobalGet { idx } => {
                self.lower_global_get(*idx, results)?;
                LeafLowering::InPlace
            }
            P::GlobalSet { idx } => {
                self.lower_global_set(*idx, args)?;
                LeafLowering::InPlace
            }
            P::TableSize { table_idx } => {
                self.lower_table_size(*table_idx, results)?;
                LeafLowering::InPlace
            }
            P::TableGet { table_idx } => {
                self.lower_table_get(*table_idx, args, results, continuation, trap)?
            }
            P::TableSet { table_idx } => {
                self.lower_table_set(*table_idx, args, continuation, trap)?
            }
            #[cfg(sf_has_simd)]
            P::V128Load { offset, memidx } => {
                self.lower_v128_load(*offset, *memidx, args, results)?
            }
            #[cfg(sf_has_simd)]
            P::SimdMemLoadV128 {
                opcode,
                offset,
                memidx,
            } => self.lower_simd_mem_load(*opcode, *offset, *memidx, args, results)?,
            #[cfg(sf_has_simd)]
            P::SimdMemLoadLaneV128 {
                opcode,
                lane,
                offset,
                memidx,
            } => self.lower_simd_mem_load_lane(*opcode, *lane, *offset, *memidx, args, results)?,
            #[cfg(sf_has_simd)]
            P::V128Store { offset, memidx } => self.lower_v128_store(*offset, *memidx, args)?,
            #[cfg(sf_has_simd)]
            P::SimdMemStoreLaneV128 {
                opcode,
                lane,
                offset,
                memidx,
            } => self.lower_simd_mem_store_lane(*opcode, *lane, *offset, *memidx, args)?,
            primitive => {
                if let Some(spec) = super::lower_leaf_special::machine_load(primitive) {
                    return Ok(Some(self.lower_memory_load(
                        spec,
                        args,
                        results,
                        continuation,
                        trap,
                    )?));
                }
                if let Some(spec) = super::lower_leaf_special::machine_store(primitive) {
                    return Ok(Some(self.lower_memory_store(
                        spec,
                        args,
                        continuation,
                        trap,
                    )?));
                }
                return Ok(None);
            }
        };
        Ok(Some(lowered))
    }

    #[inline]
    fn lower_pair_aware_operand(
        &mut self,
        operand: SsaOperand,
    ) -> Result<(MachineValue, Option<MachineValue>), WasmError> {
        match operand.decode() {
            DecodedOperand::Value(value)
                if self.gp_reg_width() == 4
                    && matches!(
                        lir_value_storage_type(self.program(), value),
                        MachineStorageType::GpI64
                    ) =>
            {
                let (lo, hi) = self.use_value_regs(value)?;
                Ok((
                    MachineValue::Reg(lo),
                    Some(MachineValue::Reg(hi.expect("pair value"))),
                ))
            }
            _ => Ok((self.lower_operand(operand)?, None)),
        }
    }

    #[inline]
    fn lower_single_result_gp_op(
        &mut self,
        results: &[SsaValue],
        build: impl FnOnce(MachineReg) -> MachineInstKind,
    ) -> Result<(), WasmError> {
        let dst = self.alloc_result_value(single_result(results)?)?;
        self.emit_machine_inst(MachineInst { kind: build(dst) });
        Ok(())
    }

    #[cfg(not(sf_has_simd))]
    #[inline]
    fn simd_lowering_disabled(&self) -> WasmError {
        WasmError::internal("SIMD lowering requested without sf_has_simd".into())
    }

    #[cfg(sf_has_simd)]
    fn lower_v128_const(&mut self, results: &[SsaValue], value: [u8; 16]) -> Result<(), WasmError> {
        let dst = self.alloc_result_value(single_result(results)?)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::V128Const { dst, bytes: value },
        });
        Ok(())
    }

    #[cfg(not(sf_has_simd))]
    fn lower_v128_const(
        &mut self,
        _results: &[SsaValue],
        _value: [u8; 16],
    ) -> Result<(), WasmError> {
        Err(self.simd_lowering_disabled())
    }

    #[cfg(sf_has_simd)]
    fn lower_simd_unary(
        &mut self,
        args: &[SsaOperand],
        results: &[SsaValue],
        opcode: u32,
    ) -> Result<(), WasmError> {
        let src = self.lower_operand(single_arg(args)?)?;
        let dead = simd_dead_values(args);
        let result = single_result(results)?;
        let dst = self.alloc_result_value_reusing_dead_inputs(result, &dead)?;
        let dst_ty = lir_value_storage_type(self.program(), result);
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::SimdUnary {
                opcode,
                dst_ty,
                dst,
                src,
            },
        });
        Ok(())
    }

    #[cfg(not(sf_has_simd))]
    fn lower_simd_unary(
        &mut self,
        _args: &[SsaOperand],
        _results: &[SsaValue],
        _opcode: u32,
    ) -> Result<(), WasmError> {
        Err(self.simd_lowering_disabled())
    }

    #[cfg(sf_has_simd)]
    fn lower_simd_binary(
        &mut self,
        args: &[SsaOperand],
        results: &[SsaValue],
        opcode: u32,
    ) -> Result<(), WasmError> {
        let (lhs_arg, rhs_arg) = two_args(args)?;
        let lhs = self.lower_operand(lhs_arg)?;
        let rhs = self.lower_operand(rhs_arg)?;
        let dead = simd_dead_values(args);
        let result = single_result(results)?;
        let dst = self.alloc_result_value_reusing_dead_inputs(result, &dead)?;
        let dst_ty = lir_value_storage_type(self.program(), result);
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::SimdBinary {
                opcode,
                dst_ty,
                dst,
                lhs,
                rhs,
            },
        });
        Ok(())
    }

    #[cfg(not(sf_has_simd))]
    fn lower_simd_binary(
        &mut self,
        _args: &[SsaOperand],
        _results: &[SsaValue],
        _opcode: u32,
    ) -> Result<(), WasmError> {
        Err(self.simd_lowering_disabled())
    }

    #[cfg(sf_has_simd)]
    fn lower_simd_ternary(
        &mut self,
        args: &[SsaOperand],
        results: &[SsaValue],
        opcode: u32,
    ) -> Result<(), WasmError> {
        let (a_arg, b_arg, c_arg) = three_args(args)?;
        let a = self.lower_operand(a_arg)?;
        let b = self.lower_operand(b_arg)?;
        let c = self.lower_operand(c_arg)?;
        let dead = simd_dead_values(args);
        let result = single_result(results)?;
        let dst = self.alloc_result_value_reusing_dead_inputs(result, &dead)?;
        let dst_ty = lir_value_storage_type(self.program(), result);
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::SimdTernary {
                opcode,
                dst_ty,
                dst,
                a,
                b,
                c,
            },
        });
        Ok(())
    }

    #[cfg(not(sf_has_simd))]
    fn lower_simd_ternary(
        &mut self,
        _args: &[SsaOperand],
        _results: &[SsaValue],
        _opcode: u32,
    ) -> Result<(), WasmError> {
        Err(self.simd_lowering_disabled())
    }

    #[cfg(sf_has_simd)]
    fn lower_simd_shift(
        &mut self,
        args: &[SsaOperand],
        results: &[SsaValue],
        opcode: u32,
    ) -> Result<(), WasmError> {
        let (vector_arg, shift_arg) = two_args(args)?;
        let vector = self.lower_operand(vector_arg)?;
        let shift = self.lower_operand(shift_arg)?;
        let dead = simd_dead_values(args);
        let result = single_result(results)?;
        let dst = self.alloc_result_value_reusing_dead_inputs(result, &dead)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::SimdShift {
                opcode,
                dst,
                vector,
                shift,
            },
        });
        Ok(())
    }

    #[cfg(not(sf_has_simd))]
    fn lower_simd_shift(
        &mut self,
        _args: &[SsaOperand],
        _results: &[SsaValue],
        _opcode: u32,
    ) -> Result<(), WasmError> {
        Err(self.simd_lowering_disabled())
    }

    #[cfg(sf_has_simd)]
    fn lower_simd_extract_lane(
        &mut self,
        args: &[SsaOperand],
        results: &[SsaValue],
        opcode: u32,
        lane: u8,
    ) -> Result<(), WasmError> {
        let src = self.lower_operand(single_arg(args)?)?;
        let dead = simd_dead_values(args);
        let result = single_result(results)?;
        let dst = self.alloc_result_value_reusing_dead_inputs(result, &dead)?;
        let dst_ty = lir_value_storage_type(self.program(), result);
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::SimdExtractLane {
                opcode,
                lane,
                dst_ty,
                dst,
                src,
            },
        });
        Ok(())
    }

    #[cfg(not(sf_has_simd))]
    fn lower_simd_extract_lane(
        &mut self,
        _args: &[SsaOperand],
        _results: &[SsaValue],
        _opcode: u32,
        _lane: u8,
    ) -> Result<(), WasmError> {
        Err(self.simd_lowering_disabled())
    }

    #[cfg(sf_has_simd)]
    fn lower_simd_replace_lane(
        &mut self,
        args: &[SsaOperand],
        results: &[SsaValue],
        opcode: u32,
        lane: u8,
    ) -> Result<(), WasmError> {
        let (vector_arg, scalar_arg) = two_args(args)?;
        let vector = self.lower_operand(vector_arg)?;
        let scalar = self.lower_operand(scalar_arg)?;
        let dead = simd_dead_values(args);
        let dst = self.alloc_result_value_reusing_dead_inputs(single_result(results)?, &dead)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::SimdReplaceLane {
                opcode,
                lane,
                dst,
                vector,
                scalar,
            },
        });
        Ok(())
    }

    #[cfg(not(sf_has_simd))]
    fn lower_simd_replace_lane(
        &mut self,
        _args: &[SsaOperand],
        _results: &[SsaValue],
        _opcode: u32,
        _lane: u8,
    ) -> Result<(), WasmError> {
        Err(self.simd_lowering_disabled())
    }

    #[cfg(sf_has_simd)]
    fn lower_simd_shuffle(
        &mut self,
        args: &[SsaOperand],
        results: &[SsaValue],
        lanes: [u8; 16],
    ) -> Result<(), WasmError> {
        let (lhs_arg, rhs_arg) = two_args(args)?;
        let lhs = self.lower_operand(lhs_arg)?;
        let rhs = self.lower_operand(rhs_arg)?;
        let dead = simd_dead_values(args);
        let dst = self.alloc_result_value_reusing_dead_inputs(single_result(results)?, &dead)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::SimdShuffle {
                dst,
                lhs,
                rhs,
                lanes,
            },
        });
        Ok(())
    }

    #[cfg(not(sf_has_simd))]
    fn lower_simd_shuffle(
        &mut self,
        _args: &[SsaOperand],
        _results: &[SsaValue],
        _lanes: [u8; 16],
    ) -> Result<(), WasmError> {
        Err(self.simd_lowering_disabled())
    }

    fn lower_struct_get(
        &mut self,
        type_idx: u32,
        field_idx: u32,
        signed: Option<bool>,
        args: &[SsaOperand],
        results: &[SsaValue],
    ) -> Result<(), WasmError> {
        let src = self.lower_operand(single_arg(args)?)?;
        let result = single_result(results)?;
        let result_ty = lir_value_storage_type(self.program(), result);
        let (dst, dst_hi) = if self.gp_reg_width() == 4 && result_ty == MachineStorageType::GpI64 {
            let (dst_lo, dst_hi) = self.alloc_i64_value_pair(result)?;
            (dst_lo, Some(dst_hi))
        } else {
            (self.alloc_result_value(result)?, None)
        };
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::StructGet {
                type_idx,
                field_idx,
                signed,
                ty: result_ty,
                src,
                dst,
                dst_hi,
            },
        });
        Ok(())
    }

    fn lower_struct_new(
        &mut self,
        type_idx: u32,
        args: &[SsaOperand],
        results: &[SsaValue],
    ) -> Result<(), WasmError> {
        let mut fields = collections::Vec::with_capacity(args.len());
        for &arg in args {
            fields.push(self.lower_pair_aware_operand(arg)?);
        }
        self.lower_single_result_gp_op(results, |dst| MachineInstKind::StructNew {
            type_idx,
            fields,
            dst,
        })
    }

    fn lower_struct_set(
        &mut self,
        type_idx: u32,
        field_idx: u32,
        args: &[SsaOperand],
    ) -> Result<(), WasmError> {
        let (ref_arg, value_arg) = two_args(args)?;
        let ref_src = self.lower_operand(ref_arg)?;
        let (value_lo, value_hi) = self.lower_pair_aware_operand(value_arg)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::StructSet {
                type_idx,
                field_idx,
                ref_src,
                value_lo,
                value_hi,
            },
        });
        Ok(())
    }

    fn lower_array_new(
        &mut self,
        type_idx: u32,
        args: &[SsaOperand],
        results: &[SsaValue],
    ) -> Result<(), WasmError> {
        let (init_arg, length_arg) = two_args(args)?;
        let (init_lo, init_hi) = self.lower_pair_aware_operand(init_arg)?;
        let length = self.lower_operand(length_arg)?;
        self.lower_single_result_gp_op(results, |dst| MachineInstKind::ArrayNew {
            type_idx,
            init_lo,
            init_hi,
            length,
            dst,
        })
    }

    fn lower_array_new_fixed(
        &mut self,
        type_idx: u32,
        args: &[SsaOperand],
        results: &[SsaValue],
    ) -> Result<(), WasmError> {
        let mut elements = collections::Vec::with_capacity(args.len());
        for &arg in args {
            elements.push(self.lower_pair_aware_operand(arg)?);
        }
        self.lower_single_result_gp_op(results, |dst| MachineInstKind::ArrayNewFixed {
            type_idx,
            elements,
            dst,
        })
    }

    fn lower_array_new_data(
        &mut self,
        type_idx: u32,
        data_idx: u32,
        args: &[SsaOperand],
        results: &[SsaValue],
    ) -> Result<(), WasmError> {
        let (src_arg, len_arg) = two_args(args)?;
        let src = self.lower_operand(src_arg)?;
        let len = self.lower_operand(len_arg)?;
        self.lower_single_result_gp_op(results, |dst| MachineInstKind::ArrayNewData {
            type_idx,
            data_idx,
            src,
            len,
            dst,
        })
    }

    fn lower_array_new_elem(
        &mut self,
        type_idx: u32,
        elem_idx: u32,
        args: &[SsaOperand],
        results: &[SsaValue],
    ) -> Result<(), WasmError> {
        let (src_arg, len_arg) = two_args(args)?;
        let src = self.lower_operand(src_arg)?;
        let len = self.lower_operand(len_arg)?;
        self.lower_single_result_gp_op(results, |dst| MachineInstKind::ArrayNewElem {
            type_idx,
            elem_idx,
            src,
            len,
            dst,
        })
    }

    fn lower_array_get(
        &mut self,
        type_idx: u32,
        signed: Option<bool>,
        args: &[SsaOperand],
        results: &[SsaValue],
    ) -> Result<(), WasmError> {
        let (ref_arg, index_arg) = two_args(args)?;
        let ref_src = self.lower_operand(ref_arg)?;
        let index = self.lower_operand(index_arg)?;
        let result = single_result(results)?;
        let result_ty = lir_value_storage_type(self.program(), result);
        let (dst, dst_hi) = if self.gp_reg_width() == 4 && result_ty == MachineStorageType::GpI64 {
            let (dst_lo, dst_hi) = self.alloc_i64_value_pair(result)?;
            (dst_lo, Some(dst_hi))
        } else {
            (self.alloc_result_value(result)?, None)
        };
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::ArrayGet {
                type_idx,
                signed,
                ty: result_ty,
                ref_src,
                index,
                dst,
                dst_hi,
            },
        });
        Ok(())
    }

    fn lower_array_set(&mut self, type_idx: u32, args: &[SsaOperand]) -> Result<(), WasmError> {
        let (ref_arg, index_arg, value_arg) = three_args(args)?;
        let ref_src = self.lower_operand(ref_arg)?;
        let index = self.lower_operand(index_arg)?;
        let (value_lo, value_hi) = self.lower_pair_aware_operand(value_arg)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::ArraySet {
                type_idx,
                ref_src,
                index,
                value_lo,
                value_hi,
            },
        });
        Ok(())
    }

    fn lower_array_fill(&mut self, type_idx: u32, args: &[SsaOperand]) -> Result<(), WasmError> {
        let (ref_arg, index_arg, value_arg, len_arg) = four_args(args)?;
        let ref_src = self.lower_operand(ref_arg)?;
        let index = self.lower_operand(index_arg)?;
        let (value_lo, value_hi) = self.lower_pair_aware_operand(value_arg)?;
        let len = self.lower_operand(len_arg)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::ArrayFill {
                type_idx,
                ref_src,
                index,
                value_lo,
                value_hi,
                len,
            },
        });
        Ok(())
    }

    fn lower_array_copy(
        &mut self,
        dst_type_idx: u32,
        src_type_idx: u32,
        args: &[SsaOperand],
    ) -> Result<(), WasmError> {
        let (dst_ref_arg, dst_index_arg, src_ref_arg, src_index_arg, len_arg) = five_args(args)?;
        let dst_ref = self.lower_operand(dst_ref_arg)?;
        let dst_index = self.lower_operand(dst_index_arg)?;
        let src_ref = self.lower_operand(src_ref_arg)?;
        let src_index = self.lower_operand(src_index_arg)?;
        let len = self.lower_operand(len_arg)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::ArrayCopy {
                dst_type_idx,
                src_type_idx,
                dst_ref,
                dst_index,
                src_ref,
                src_index,
                len,
            },
        });
        Ok(())
    }

    fn lower_array_init_data(
        &mut self,
        type_idx: u32,
        data_idx: u32,
        args: &[SsaOperand],
    ) -> Result<(), WasmError> {
        let (ref_arg, dst_index_arg, src_index_arg, len_arg) = four_args(args)?;
        let ref_src = self.lower_operand(ref_arg)?;
        let dst_index = self.lower_operand(dst_index_arg)?;
        let src_index = self.lower_operand(src_index_arg)?;
        let len = self.lower_operand(len_arg)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::ArrayInitData {
                type_idx,
                data_idx,
                ref_src,
                dst_index,
                src_index,
                len,
            },
        });
        Ok(())
    }

    fn lower_array_init_elem(
        &mut self,
        type_idx: u32,
        elem_idx: u32,
        args: &[SsaOperand],
    ) -> Result<(), WasmError> {
        let (ref_arg, dst_index_arg, src_index_arg, len_arg) = four_args(args)?;
        let ref_src = self.lower_operand(ref_arg)?;
        let dst_index = self.lower_operand(dst_index_arg)?;
        let src_index = self.lower_operand(src_index_arg)?;
        let len = self.lower_operand(len_arg)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::ArrayInitElem {
                type_idx,
                elem_idx,
                ref_src,
                dst_index,
                src_index,
                len,
            },
        });
        Ok(())
    }

    fn lower_array_len(
        &mut self,
        args: &[SsaOperand],
        results: &[SsaValue],
    ) -> Result<(), WasmError> {
        let src = self.lower_operand(single_arg(args)?)?;
        self.lower_single_result_gp_op(results, |dst| MachineInstKind::ArrayLen { src, dst })
    }

    pub(super) fn lower_leaf(
        &mut self,
        op: &PrimitiveOpKind,
        args: &[SsaOperand],
        results: &[SsaValue],
    ) -> Result<(), WasmError> {
        use PrimitiveOpKind as P;
        let primitive = op;

        {
            let ops = self.i64_ops();
            if ops.lower_i64_leaf(self, primitive, args, results)? {
                return Ok(());
            }
        }

        match primitive {
            P::Drop | P::Nop => {
                for arg in args {
                    if let DecodedOperand::Value(v) = arg.decode() {
                        let _ = self.use_value(v)?;
                    }
                }
                Ok(())
            }
            P::I32Const { value } => self.lower_const(results, *value as u64),
            P::I64Const { value } => self.lower_const(results, *value),
            P::F32Const { value } => {
                self.lower_float_const(results, MachineFloatWidth::F32, u64::from(*value))
            }
            P::F64Const { value } => {
                self.lower_float_const(results, MachineFloatWidth::F64, *value)
            }
            P::V128Const { value } => self.lower_v128_const(results, *value),
            P::V128Not => self.lower_simd_unary(args, results, OpcodeFD::V128_NOT as u32),
            P::V128And => self.lower_simd_binary(args, results, OpcodeFD::V128_AND as u32),
            P::V128AndNot => self.lower_simd_binary(args, results, OpcodeFD::V128_ANDNOT as u32),
            P::V128Or => self.lower_simd_binary(args, results, OpcodeFD::V128_OR as u32),
            P::V128Xor => self.lower_simd_binary(args, results, OpcodeFD::V128_XOR as u32),
            P::V128Bitselect => {
                self.lower_simd_ternary(args, results, OpcodeFD::V128_BITSELECT as u32)
            }
            P::V128AnyTrue => self.lower_simd_unary(args, results, OpcodeFD::V128_ANY_TRUE as u32),
            P::I8x16AllTrue => {
                self.lower_simd_unary(args, results, OpcodeFD::I8X16_ALL_TRUE as u32)
            }
            P::I16x8AllTrue => {
                self.lower_simd_unary(args, results, OpcodeFD::I16X8_ALL_TRUE as u32)
            }
            P::I32x4AllTrue => {
                self.lower_simd_unary(args, results, OpcodeFD::I32X4_ALL_TRUE as u32)
            }
            P::I64x2AllTrue => {
                self.lower_simd_unary(args, results, OpcodeFD::I64X2_ALL_TRUE as u32)
            }
            P::I8x16Bitmask => self.lower_simd_unary(args, results, OpcodeFD::I8X16_BITMASK as u32),
            P::I16x8Bitmask => self.lower_simd_unary(args, results, OpcodeFD::I16X8_BITMASK as u32),
            P::I32x4Bitmask => self.lower_simd_unary(args, results, OpcodeFD::I32X4_BITMASK as u32),
            P::I64x2Bitmask => self.lower_simd_unary(args, results, OpcodeFD::I64X2_BITMASK as u32),
            P::I8x16Splat => self.lower_simd_unary(args, results, OpcodeFD::I8X16_SPLAT as u32),
            P::I16x8Splat => self.lower_simd_unary(args, results, OpcodeFD::I16X8_SPLAT as u32),
            P::I32x4Splat => self.lower_simd_unary(args, results, OpcodeFD::I32X4_SPLAT as u32),
            P::I64x2Splat => self.lower_simd_unary(args, results, OpcodeFD::I64X2_SPLAT as u32),
            P::F32x4Splat => self.lower_simd_unary(args, results, OpcodeFD::F32X4_SPLAT as u32),
            P::F64x2Splat => self.lower_simd_unary(args, results, OpcodeFD::F64X2_SPLAT as u32),
            P::I8x16ExtractLaneS { lane } => self.lower_simd_extract_lane(
                args,
                results,
                OpcodeFD::I8X16_EXTRACT_LANE_S as u32,
                *lane,
            ),
            P::I8x16ExtractLaneU { lane } => self.lower_simd_extract_lane(
                args,
                results,
                OpcodeFD::I8X16_EXTRACT_LANE_U as u32,
                *lane,
            ),
            P::I16x8ExtractLaneS { lane } => self.lower_simd_extract_lane(
                args,
                results,
                OpcodeFD::I16X8_EXTRACT_LANE_S as u32,
                *lane,
            ),
            P::I16x8ExtractLaneU { lane } => self.lower_simd_extract_lane(
                args,
                results,
                OpcodeFD::I16X8_EXTRACT_LANE_U as u32,
                *lane,
            ),
            P::I32x4ExtractLane { lane } => self.lower_simd_extract_lane(
                args,
                results,
                OpcodeFD::I32X4_EXTRACT_LANE as u32,
                *lane,
            ),
            P::I64x2ExtractLane { lane } => self.lower_simd_extract_lane(
                args,
                results,
                OpcodeFD::I64X2_EXTRACT_LANE as u32,
                *lane,
            ),
            P::F32x4ExtractLane { lane } => self.lower_simd_extract_lane(
                args,
                results,
                OpcodeFD::F32X4_EXTRACT_LANE as u32,
                *lane,
            ),
            P::F64x2ExtractLane { lane } => self.lower_simd_extract_lane(
                args,
                results,
                OpcodeFD::F64X2_EXTRACT_LANE as u32,
                *lane,
            ),
            P::SimdReplaceLane { opcode, lane } => {
                self.lower_simd_replace_lane(args, results, *opcode, *lane)
            }
            P::I8x16Shuffle { lanes } => self.lower_simd_shuffle(args, results, *lanes),
            P::SimdUnaryV128 { opcode } => self.lower_simd_unary(args, results, *opcode),
            P::SimdBinaryV128 { opcode } => self.lower_simd_binary(args, results, *opcode),
            P::SimdTernaryV128 { opcode } => self.lower_simd_ternary(args, results, *opcode),
            P::SimdShiftV128 { opcode } => self.lower_simd_shift(args, results, *opcode),
            P::I32x4Add => self.lower_simd_binary(args, results, OpcodeFD::I32X4_ADD as u32),
            P::I64x2Add => self.lower_simd_binary(args, results, OpcodeFD::I64X2_ADD as u32),

            P::RefNull => self.lower_const(results, self.gp_word_max_imm()),
            P::RefFunc { func_idx } => {
                self.lower_single_result_gp_op(results, |dst| MachineInstKind::RefFunc {
                    func_idx: *func_idx,
                    dst,
                })
            }
            P::RefIsNull => self.lower_ref_is_null(args, results),
            P::RefAsNonNull => {
                let src = self.lower_operand(single_arg(args)?)?;
                self.lower_single_result_gp_op(results, |dst| MachineInstKind::RefAsNonNull {
                    src,
                    dst,
                })
            }
            P::RefEq => {
                let (lhs, rhs) = two_args(args)?;
                let lhs = self.lower_operand(lhs)?;
                let rhs = self.lower_operand(rhs)?;
                self.lower_single_result_gp_op(results, |dst| MachineInstKind::RefEq {
                    lhs,
                    rhs,
                    dst,
                })
            }
            P::RefI31 => {
                let src = self.lower_operand(single_arg(args)?)?;
                self.lower_single_result_gp_op(results, |dst| MachineInstKind::RefI31 { src, dst })
            }
            P::I31GetS => {
                let src = self.lower_operand(single_arg(args)?)?;
                self.lower_single_result_gp_op(results, |dst| MachineInstKind::I31GetS { src, dst })
            }
            P::I31GetU => {
                let src = self.lower_operand(single_arg(args)?)?;
                self.lower_single_result_gp_op(results, |dst| MachineInstKind::I31GetU { src, dst })
            }
            P::AnyConvertExtern => {
                let src = self.lower_operand(single_arg(args)?)?;
                self.lower_single_result_gp_op(results, |dst| MachineInstKind::AnyConvertExtern {
                    src,
                    dst,
                })
            }
            P::ExternConvertAny => {
                let src = self.lower_operand(single_arg(args)?)?;
                self.lower_single_result_gp_op(results, |dst| MachineInstKind::ExternConvertAny {
                    src,
                    dst,
                })
            }
            P::RefTest { ref_type } => {
                let src_arg = single_arg(args)?;
                let mut reuse_candidates = collections::Vec::new();
                if let DecodedOperand::Value(value) = src_arg.decode() {
                    reuse_candidates.push(value);
                }
                let src = self.lower_operand(src_arg)?;
                let result = single_result(results)?;
                let dst = if reuse_candidates.is_empty() {
                    self.alloc_result_value(result)?
                } else {
                    self.alloc_result_value_reusing_dead_inputs(result, &reuse_candidates)?
                };
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::RefTest {
                        ref_type: *ref_type,
                        src,
                        dst,
                    },
                });
                Ok(())
            }
            P::RefCast { ref_type } => {
                let src = self.lower_operand(single_arg(args)?)?;
                self.lower_single_result_gp_op(results, |dst| MachineInstKind::RefCast {
                    ref_type: *ref_type,
                    src,
                    dst,
                })
            }
            P::StructNew { type_idx, .. } => self.lower_struct_new(*type_idx, args, results),
            P::StructNewDefault { type_idx } => {
                self.lower_single_result_gp_op(results, |dst| MachineInstKind::StructNewDefault {
                    type_idx: *type_idx,
                    dst,
                })
            }
            P::StructGet {
                type_idx,
                field_idx,
            } => self.lower_struct_get(*type_idx, *field_idx, None, args, results),
            P::StructGetS {
                type_idx,
                field_idx,
            } => self.lower_struct_get(*type_idx, *field_idx, Some(true), args, results),
            P::StructGetU {
                type_idx,
                field_idx,
            } => self.lower_struct_get(*type_idx, *field_idx, Some(false), args, results),
            P::StructSet {
                type_idx,
                field_idx,
            } => self.lower_struct_set(*type_idx, *field_idx, args),
            P::ArrayNew { type_idx } => self.lower_array_new(*type_idx, args, results),
            P::ArrayNewDefault { type_idx } => {
                let length = self.lower_operand(single_arg(args)?)?;
                self.lower_single_result_gp_op(results, |dst| MachineInstKind::ArrayNewDefault {
                    type_idx: *type_idx,
                    length,
                    dst,
                })
            }
            P::ArrayNewFixed { type_idx, .. } => {
                self.lower_array_new_fixed(*type_idx, args, results)
            }
            P::ArrayNewData { type_idx, data_idx } => {
                self.lower_array_new_data(*type_idx, *data_idx, args, results)
            }
            P::ArrayNewElem { type_idx, elem_idx } => {
                self.lower_array_new_elem(*type_idx, *elem_idx, args, results)
            }
            P::ArrayGet { type_idx } => self.lower_array_get(*type_idx, None, args, results),
            P::ArrayGetS { type_idx } => self.lower_array_get(*type_idx, Some(true), args, results),
            P::ArrayGetU { type_idx } => {
                self.lower_array_get(*type_idx, Some(false), args, results)
            }
            P::ArraySet { type_idx } => self.lower_array_set(*type_idx, args),
            P::ArrayFill { type_idx } => self.lower_array_fill(*type_idx, args),
            P::ArrayCopy {
                dst_type_idx,
                src_type_idx,
            } => self.lower_array_copy(*dst_type_idx, *src_type_idx, args),
            P::ArrayInitData { type_idx, data_idx } => {
                self.lower_array_init_data(*type_idx, *data_idx, args)
            }
            P::ArrayInitElem { type_idx, elem_idx } => {
                self.lower_array_init_elem(*type_idx, *elem_idx, args)
            }
            P::ArrayLen => self.lower_array_len(args, results),
            P::Select => self.lower_select(args, results),
            // i32.eqz / i64.eqz lower as IntCompare against zero so that the
            // existing fuse_compare_branch peephole can collapse the common
            // `i32.eqz; br_if` pattern into a single conditional branch.
            P::I32Eqz => self.lower_int_eqz(args, results, MachineIntWidth::I32),
            P::I64Eqz => self.lower_int_eqz(args, results, MachineIntWidth::I64),
            primitive => {
                if let Some((width, op)) = super::lower_leaf_arith::machine_int_binary(primitive) {
                    return self.lower_int_binary(args, results, width, op);
                }
                if let Some((width, kind, sign)) =
                    super::lower_leaf_arith::machine_int_compare(primitive)
                {
                    return self.lower_int_compare(args, results, width, kind, sign);
                }
                if let Some((width, op)) = super::lower_leaf_arith::machine_int_unary(primitive) {
                    return self.lower_int_unary(args, results, width, op);
                }
                if let Some((width, op)) = super::lower_leaf_arith::machine_float_binary(primitive)
                {
                    return self.lower_float_binary(args, results, width, op);
                }
                if let Some((width, kind)) =
                    super::lower_leaf_arith::machine_float_compare(primitive)
                {
                    return self.lower_float_compare(args, results, width, kind);
                }
                if let Some((width, op)) = super::lower_leaf_arith::machine_float_unary(primitive) {
                    return self.lower_float_unary(args, results, width, op);
                }
                if let Some(op) = super::lower_leaf_arith::machine_convert(primitive) {
                    return self.lower_convert(args, results, op);
                }

                Err(WasmError::internal(
                    "primitive is not lowered to MachineIR yet",
                ))
            }
        }
    }

    fn lower_edge(&self, edge: &SsaEdge) -> Result<MachineEdge, WasmError> {
        let target_block = MachineBlockId(edge.target.as_u32());
        let target = self
            .program()
            .blocks
            .get(edge.target.as_usize())
            .ok_or_else(|| {
                WasmError::internal("edge target is out of range during native lowering")
            })?;
        // Size for the common case of single-reg values; will grow if pair
        // regs (32-bit targets) push extra hi halves. Counting params+entries
        // avoids the original `params.len()` under-estimate, which ignored
        // cache entries and forced push-growth past them.
        let cache_entries = self.block_entry_cache_params(target.id.0);
        let mut args = collections::Vec::with_capacity(target.params.len() + cache_entries.len());
        for target_param in &target.params {
            let binding = edge
                .bindings
                .iter()
                .find(|binding| binding.param == *target_param)
                .ok_or_else(|| {
                    WasmError::internal(
                        "missing SSA-IR edge binding for target param during native lowering"
                            .into(),
                    )
                })?;
            let regs = self.value_regs_for_edge(binding.value)?;
            args.push(MachineValue::Reg(regs.0));
            if let Some(hi) = regs.1 {
                args.push(MachineValue::Reg(hi));
            }
        }
        for entry in cache_entries.iter().copied() {
            let cached_index = usize::from(entry.cached_index);
            let cached = self.bound_cached_cell(cached_index).ok_or_else(|| {
                let _cell = self.cached_cells()[cached_index].cell;
                WasmError::internal("edge to b expects cached local slot to stay resident, but source block b has no binding")
            })?;
            let _cell = cached.cell;
            if !self.is_cache_live(cached_index) {
                return Err(WasmError::internal("edge to b expects cached local slot to stay resident, but source block b marked it dead"));
            }
            if entry.needs_value && !self.cache_has_value(cached_index) {
                return Err(WasmError::internal("edge to b expects cached local slot to carry a real value, but source block b only reserved the lane"));
            }
            args.push(if entry.needs_value {
                MachineValue::Reg(cached.reg)
            } else {
                MachineValue::ReservedReg(entry.regs.lo)
            });
            if let Some(hi) = cached.hi_reg {
                args.push(if entry.needs_value {
                    MachineValue::Reg(hi)
                } else {
                    MachineValue::ReservedReg(entry.regs.hi.unwrap_or(hi))
                });
            }
        }
        Ok(MachineEdge {
            target: target_block,
            args,
        })
    }

    /// Look up the register pair for a value (used by lower_edge).
    fn value_regs_for_edge(
        &self,
        value: SsaValue,
    ) -> Result<(MachineReg, Option<MachineReg>), WasmError> {
        self.try_value_regs(value).ok_or_else(|| {
            WasmError::internal("no machine register pair assigned for SSA-IR value")
        })
    }
}

#[cfg(sf_has_simd)]
fn simd_dead_values(args: &[SsaOperand]) -> collections::Vec<SsaValue> {
    args.iter()
        .filter_map(|arg| match arg.decode() {
            DecodedOperand::Value(value) => Some(value),
            DecodedOperand::Const(_) | DecodedOperand::None => None,
        })
        .collect()
}
