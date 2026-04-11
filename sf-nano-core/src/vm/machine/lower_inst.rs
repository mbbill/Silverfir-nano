//! Instruction dispatch — lower_ops, lower_terminator, lower_inst, lower_leaf,
//! lower_leaf_special, lower_edge.

use crate::collections;

use crate::{
    error::WasmError,
    vm::{
        machine::machine_ir::{
            MachineBlockId, MachineBranchCond, MachineEdge, MachineFloatWidth, MachineInst,
            MachineInstKind, MachineIntWidth, MachineLoadExtension, MachineReg, MachineStorageType,
            MachineTerminator, MachineTrapKind, MachineValue,
        },
        middle::ssa_ir::{
            ir::{SsaEdge, SsaInst, SsaInstKind, SsaOperand, SsaTerminator, SsaValue},
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
            SsaTerminator::TrapUnreachable => Ok(MachineTerminator::Trap {
                kind: MachineTrapKind::Unreachable,
            }),
        }
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
        let Some(cached_index) = self.cached_local_index(sink_slot) else {
            return Ok(());
        };
        // Only premap when the cache is already bound. If not yet bound,
        // skip — the later LocalSetCache will handle binding via
        // try_bind_cached_local_from_dying_value without needing an
        // extra register.
        let Some(cached) = self.bound_cached_local(cached_index) else {
            return Ok(());
        };
        let cache_reg = cached.reg;
        let cache_hi_reg = cached.hi_reg;
        let mut arg_vals = [SsaValue(u32::MAX); 4];
        let mut n = 0;
        for a in args.iter() {
            if let SsaOperand::Value(v) = a {
                if n < arg_vals.len() {
                    arg_vals[n] = *v;
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
        match &inst.kind {
            SsaInstKind::LocalGetSlot { slot, dst } => {
                self.lower_local_get_slot(*slot, *dst)?;
            }
            SsaInstKind::LocalGetCache { slot, dst } => {
                self.lower_local_get_cache(*slot, *dst)?;
            }
            SsaInstKind::LocalSetSlot { slot, src } => {
                self.lower_local_set_slot(*slot, *src)?;
            }
            SsaInstKind::LocalSetCache { slot, src } => {
                self.lower_local_set_cache(*slot, *src)?;
            }
            SsaInstKind::LocalEnsureCache { slot } => {
                self.lower_local_ensure_cache(*slot)?;
            }
            SsaInstKind::LocalReserveCache { slot } => {
                self.lower_local_reserve_cache(*slot)?;
            }
            SsaInstKind::LocalDropCache { slot } => {
                if let Some(index) = self.cached_local_index(*slot) {
                    self.emit_drop_cached_local(index)?;
                }
            }
            SsaInstKind::Fill { slot, dst } => {
                let ty = lir_value_storage_type(self.program(), *dst);
                self.apply_sink_premap(&[], &[*dst])?;
                if matches!(ty, MachineStorageType::GpI64) {
                    let ops = self.i64_ops();
                    ops.emit_load_slot_i64(self, *slot, *dst)?;
                    return Ok(());
                }
                let dst_reg = self.alloc_slot_load_value(*dst)?;
                let width = canonical_value_mem_width_for_value(self.program(), *dst);
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Load {
                        owner: crate::vm::machine::machine_ir::MachineRegOwner::LinearValue,
                        ty,
                        dst: dst_reg,
                        addr: self.frame_addr(*slot)?,
                        width,
                        extension: MachineLoadExtension::None,
                    },
                });
            }
            SsaInstKind::Spill { slot, src } => {
                let ty = lir_value_storage_type(self.program(), *src);
                if matches!(ty, MachineStorageType::GpI64) {
                    let ops = self.i64_ops();
                    ops.emit_store_slot_i64(self, *slot, *src)?;
                    return Ok(());
                }
                let src_reg = self.use_value(*src)?;
                let width = canonical_value_mem_width_for_value(self.program(), *src);
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
                self.release_dead_values()?;
            }
            SsaInstKind::Value { op, args, results } => {
                // Sink pre-mapping is now applied by the caller
                // (lower_module.rs) before dispatching to lower_inst.
                self.lower_leaf(op, args, results)?;
                self.release_dead_values()?;
            }
            SsaInstKind::Call(_call) => {
                return Err(WasmError::internal(
                    "call op must be lowered through its specialized path",
                ));
            }
        }
        Ok(())
    }

    fn ensure_cached_local_loaded(
        &mut self,
        _slot: crate::vm::middle::frame::FrameSlot,
        cached_index: usize,
        ty: MachineStorageType,
    ) -> Result<(), WasmError> {
        if self.is_cache_live(cached_index) && self.cache_has_value(cached_index) {
            return Ok(());
        }
        let cached = self.ensure_bound_cached_local(cached_index)?;
        if cached.ty != ty {
            return Err(WasmError::internal(
                "typed SSA-IR cache load from local slot expects , but cached local is",
            ));
        }
        if matches!(cached.ty, MachineStorageType::GpI64) {
            let ops = self.i64_ops();
            ops.emit_reload_cached_i64(self, &cached)?;
        } else {
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Load {
                    owner: crate::vm::machine::machine_ir::MachineRegOwner::CachedLocal,
                    ty: cached.ty,
                    dst: cached.reg,
                    addr: self.frame_addr(cached.slot)?,
                    width: super::lower_regalloc::canonical_cached_local_mem_width(cached.ty),
                    extension: MachineLoadExtension::None,
                },
            });
        }
        self.set_cache_live(cached_index, true);
        self.set_cache_has_value(cached_index, true);
        self.set_cache_dirty(cached_index, false);
        Ok(())
    }

    fn lower_local_get_slot(
        &mut self,
        slot: crate::vm::middle::frame::FrameSlot,
        dst: SsaValue,
    ) -> Result<(), WasmError> {
        let ty = lir_value_storage_type(self.program(), dst);
        if matches!(ty, MachineStorageType::GpI64) && self.gp_reg_width() == 4 {
            let ops = self.i64_ops();
            ops.emit_load_slot_i64(self, slot, dst)?;
            return Ok(());
        }
        let dst_reg = self.alloc_slot_load_value(dst)?;
        let width = canonical_value_mem_width_for_value(self.program(), dst);
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: crate::vm::machine::machine_ir::MachineRegOwner::LinearValue,
                ty,
                dst: dst_reg,
                addr: self.frame_addr(slot)?,
                width,
                extension: MachineLoadExtension::None,
            },
        });
        Ok(())
    }

    fn lower_local_get_cache(
        &mut self,
        slot: crate::vm::middle::frame::FrameSlot,
        dst: SsaValue,
    ) -> Result<(), WasmError> {
        let ty = lir_value_storage_type(self.program(), dst);
        let Some(cached_index) = self.cached_local_index(slot) else {
            return Err(WasmError::internal(
                "LocalGetCache on non-cached local slot",
            ));
        };
        self.ensure_cached_local_loaded(slot, cached_index, ty)
            .map_err(|_err| WasmError::internal("LocalGetCache(slot=, dst=) in block b failed"))?;
        let cached = self.ensure_bound_cached_local(cached_index)?;
        if matches!(ty, MachineStorageType::GpI64) && self.gp_reg_width() == 4 {
            let cached_hi = cached.hi_reg.ok_or_else(|| {
                WasmError::internal("cached i64 local is missing a high-half register")
            })?;
            let (dst_lo, dst_hi) = self.alloc_i64_value_pair(dst)?;
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Move {
                    owner: crate::vm::machine::machine_ir::MachineRegOwner::LinearValue,
                    ty: MachineStorageType::GpWord,
                    dst: dst_lo,
                    src: MachineValue::Reg(cached.reg),
                },
            });
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Move {
                    owner: crate::vm::machine::machine_ir::MachineRegOwner::LinearValue,
                    ty: MachineStorageType::GpWord,
                    dst: dst_hi,
                    src: MachineValue::Reg(cached_hi),
                },
            });
            return Ok(());
        }

        // Source-alias: map value directly to cache register, no emit.
        // materialize_cache_aliases will copy aliased values out before the
        // cache register is overwritten by a later LocalSetCache or drop.
        self.push_value_location(dst, cached.reg, None);
        Ok(())
    }

    fn lower_local_ensure_cache(
        &mut self,
        slot: crate::vm::middle::frame::FrameSlot,
    ) -> Result<(), WasmError> {
        let Some(cached_index) = self.cached_local_index(slot) else {
            return Err(WasmError::internal(
                "LocalEnsureCache on non-cached local slot",
            ));
        };
        let ty = self
            .bound_cached_local(cached_index)
            .map(|cached| cached.ty)
            .unwrap_or_else(|| self.cached_locals()[cached_index].ty);
        self.ensure_cached_local_loaded(slot, cached_index, ty)
            .map_err(|_err| WasmError::internal("LocalEnsureCache(slot=) in block b failed"))
    }

    fn lower_local_reserve_cache(
        &mut self,
        slot: crate::vm::middle::frame::FrameSlot,
    ) -> Result<(), WasmError> {
        let Some(cached_index) = self.cached_local_index(slot) else {
            return Err(WasmError::internal(
                "LocalReserveCache on non-cached local slot",
            ));
        };
        self.ensure_bound_cached_local(cached_index)
            .map_err(|_err| WasmError::internal("LocalReserveCache(slot=) in block b failed"))?;
        self.set_cache_live(cached_index, true);
        self.set_cache_has_value(cached_index, false);
        self.set_cache_dirty(cached_index, false);
        Ok(())
    }

    fn lower_local_set_slot(
        &mut self,
        slot: crate::vm::middle::frame::FrameSlot,
        src: SsaValue,
    ) -> Result<(), WasmError> {
        let ty = lir_value_storage_type(self.program(), src);
        if matches!(ty, MachineStorageType::GpI64) && self.gp_reg_width() == 4 {
            let ops = self.i64_ops();
            ops.emit_store_slot_i64(self, slot, src)?;
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
        Ok(())
    }

    fn lower_local_set_cache(
        &mut self,
        slot: crate::vm::middle::frame::FrameSlot,
        src: SsaValue,
    ) -> Result<(), WasmError> {
        let ty = lir_value_storage_type(self.program(), src);
        let Some(cached_index) = self.cached_local_index(slot) else {
            return Err(WasmError::internal(
                "LocalSetCache on non-cached local slot",
            ));
        };
        let cached = self
            .try_bind_cached_local_from_dying_value(cached_index, src, ty)?
            .unwrap_or(
                self.ensure_bound_cached_local(cached_index)
                    .map_err(|_err| {
                        WasmError::internal(
                            "LocalSetCache(slot=, src=, remaining_uses=) in block b failed",
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

        if matches!(ty, MachineStorageType::GpI64) && self.gp_reg_width() == 4 {
            let cached_hi = cached.hi_reg.ok_or_else(|| {
                WasmError::internal("cached i64 local is missing a high-half register")
            })?;
            let (src_lo, src_hi) = self.use_i64_value_pair(src)?;
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Move {
                    owner: crate::vm::machine::machine_ir::MachineRegOwner::CachedLocal,
                    ty: MachineStorageType::GpWord,
                    dst: cached.reg,
                    src: MachineValue::Reg(src_lo),
                },
            });
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Move {
                    owner: crate::vm::machine::machine_ir::MachineRegOwner::CachedLocal,
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
                owner: crate::vm::machine::machine_ir::MachineRegOwner::CachedLocal,
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
        op: &SsaLeafOp,
        args: &[SsaOperand],
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
        args: &[SsaOperand],
        results: &[SsaValue],
    ) -> Result<(), WasmError> {
        use PrimitiveOpKind as P;
        let primitive = op.primitive();

        {
            let ops = self.i64_ops();
            if ops.lower_i64_leaf(self, primitive, args, results)? {
                return Ok(());
            }
        }

        match primitive {
            P::Drop | P::Nop => {
                for arg in args {
                    if let SsaOperand::Value(v) = arg {
                        let _ = self.use_value(*v)?;
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
            P::RefNull => self.lower_const(results, self.gp_word_max_imm()),
            P::RefFunc { func_idx } => self.lower_const(results, *func_idx as u64),
            P::RefIsNull => self.lower_ref_is_null(args, results),
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
        let mut args = collections::Vec::with_capacity(target.params.len());
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
        for entry in self.block_entry_cache_params(target.id.0).iter().copied() {
            let cached = self.bound_cached_local(entry.cached_index).ok_or_else(|| {
                let _slot = self.cached_locals()[entry.cached_index].slot;
                WasmError::internal("edge to b expects cached local slot to stay resident, but source block b has no binding")
            })?;
            let _slot = cached.slot;
            if !self.is_cache_live(entry.cached_index) {
                return Err(WasmError::internal("edge to b expects cached local slot to stay resident, but source block b marked it dead"));
            }
            if entry.needs_value && !self.cache_has_value(entry.cached_index) {
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
