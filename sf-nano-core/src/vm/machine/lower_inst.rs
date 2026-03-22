//! Instruction dispatch — lower_ops, lower_terminator, lower_inst, lower_leaf,
//! lower_leaf_special, lower_edge.

use alloc::vec::Vec;

use crate::{
    error::WasmError,
    vm::{
        machine::machine_ir::{
            MachineBranchCond, MachineBlockId, MachineEdge, MachineFloatWidth, MachineInst,
            MachineInstKind, MachineLoadExtension, MachineMemWidth, MachineReg,
            MachineStorageType, MachineTerminator, MachineTrapKind, MachineValue,
        },
        middle::ssa_ir::{
            ir::{SsaEdge, SsaInst, SsaInstKind, SsaTerminator, SsaValue},
            leaf::SsaLeafOp,
        },
        wasm::primitive_op::PrimitiveOpKind,
    },
};

use super::{
    lower_context::BlockLowerContext,
    lower_regalloc::{canonical_value_mem_width_for_value, lir_value_storage_type},
};

pub(super) enum LeafLowering {
    InPlace,
    Split {
        continuation: MachineBlockId,
        trap: MachineBlockId,
        trap_kind: MachineTrapKind,
        terminator: MachineTerminator,
        continuation_ops: Vec<MachineInst>,
    },
}

impl<'a> BlockLowerContext<'a> {
    pub(super) fn lower_ops(&mut self) -> Result<(), WasmError> {
        // Copy ops index range to avoid borrow conflict with self.
        let op_count = self.block().ops.len();
        for i in 0..op_count {
            // Re-borrow block for each iteration to satisfy the borrow checker.
            let inst_ptr = &self.block().ops[i] as *const SsaInst;
            // SAFETY: the block reference is stable for the lifetime of self,
            // and lower_inst does not modify block.ops.
            let inst = unsafe { &*inst_ptr };
            self.lower_inst(inst)?;
        }
        Ok(())
    }

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
                let mut lowered = Vec::with_capacity(entries.len());
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
                        "LIR return reached native lowering with live transient SSA values; results must be published before return".into(),
                    ));
                }
                Ok(MachineTerminator::Return)
            }
            SsaTerminator::TrapUnreachable => Ok(MachineTerminator::Trap {
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

    pub(super) fn lower_inst(&mut self, inst: &SsaInst) -> Result<(), WasmError> {
        match &inst.kind {
            SsaInstKind::LoadSlot { slot, dst } => {
                let ty = lir_value_storage_type(self.program(), *dst);
                if self.gp_reg_width() == 4 && matches!(ty, MachineStorageType::GpI64) {
                    let (dst_lo, dst_hi) = self.alloc_i64_value_pair(*dst)?;
                    if let Some(cached_index) = self.cached_local_index(*slot) {
                        let cached = self.cached_locals()[cached_index];
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
                let width = canonical_value_mem_width_for_value(self.program(), *dst);
                if let Some(cached_index) = self.cached_local_index(*slot) {
                    let cached = self.cached_locals()[cached_index];
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
            SsaInstKind::StoreSlot { slot, src } => {
                let ty = lir_value_storage_type(self.program(), *src);
                if self.gp_reg_width() == 4 && matches!(ty, MachineStorageType::GpI64) {
                    let (src_lo, src_hi) = self.use_i64_value_pair(*src)?;
                    if let Some(cached_index) = self.cached_local_index(*slot) {
                        let cached = self.cached_locals()[cached_index];
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
                let width = canonical_value_mem_width_for_value(self.program(), *src);
                if let Some(cached_index) = self.cached_local_index(*slot) {
                    let cached = self.cached_locals()[cached_index];
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
            SsaInstKind::Value { op, args, results } => {
                self.lower_leaf(op, args, results)?;
                self.release_dead_values()?;
            }
            SsaInstKind::Boundary(boundary) => {
                return Err(WasmError::internal(alloc::format!(
                    "boundary op {:?} must be lowered through its specialized native path",
                    boundary
                )));
            }
        }
        Ok(())
    }

    pub(super) fn lower_leaf_special(
        &mut self,
        op: &SsaLeafOp,
        args: &[SsaValue],
        results: &[SsaValue],
        continuation: MachineBlockId,
        trap: MachineBlockId,
    ) -> Result<Option<LeafLowering>, WasmError> {
        use PrimitiveOpKind as P;

        let lowered = match op.primitive() {
            P::MemorySize { mem_idx } => {
                self.lower_memory_size(*mem_idx, results)?;
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

    pub(super) fn lower_leaf(
        &mut self,
        op: &SsaLeafOp,
        args: &[SsaValue],
        results: &[SsaValue],
    ) -> Result<(), WasmError> {
        use PrimitiveOpKind as P;
        let primitive = op.primitive();

        if self.gp_reg_width() == 4 && self.lower_i64_pair_leaf(primitive, args, results)? {
            return Ok(());
        }

        match primitive {
            P::Drop | P::Nop => {
                for arg in args {
                    let _ = self.use_value(*arg)?;
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
            P::RefNull => self.lower_const(results, self.gp_word_max_imm()),
            P::RefFunc { func_idx } => self.lower_const(results, *func_idx as u64),
            P::RefIsNull => self.lower_ref_is_null(args, results),
            P::Select => self.lower_select(args, results),
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
                if let Some((width, op)) = super::lower_leaf_arith::machine_float_unary(primitive)
                {
                    return self.lower_float_unary(args, results, width, op);
                }
                if let Some(op) = super::lower_leaf_arith::machine_convert(primitive) {
                    return self.lower_convert(args, results, op);
                }

                Err(WasmError::internal(alloc::format!(
                    "primitive {:?} is not lowered to MachineIR yet",
                    primitive
                )))
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
            let regs = self.value_regs_for_edge(binding.value)?;
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

    /// Look up the register pair for a value (used by lower_edge).
    fn value_regs_for_edge(
        &self,
        value: SsaValue,
    ) -> Result<(MachineReg, Option<MachineReg>), WasmError> {
        self.try_value_regs(value).ok_or_else(|| {
            WasmError::internal(alloc::format!(
                "no machine register pair assigned for LIR value {:?}",
                value
            ))
        })
    }
}
