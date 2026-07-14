use crate::{
    error::WasmError,
    vm::{
        machine::machine_ir::{
            MachineArgSrc, MachineArgSrcPair, MachineBlockId, MachineBlockParam, MachineBranchCond,
            MachineCallArgs, MachineCallLaneArg, MachineCallResults, MachineCallRuntime,
            MachineCallTarget, MachineCompareKind, MachineConstId, MachineFrameRegion,
            MachineFuncId, MachineInst, MachineInstKind, MachineIntBinaryOp, MachineLoadExtension,
            MachineMemWidth, MachineParamLoc, MachineReg, MachineRegOwner, MachineResultDst,
            MachineReturnAbi, MachineSign, MachineStorageType, MachineTerminator, MachineTrapKind,
            MachineValue,
        },
        middle::{
            frame::{FrameSlot, FrameSpan},
            ssa_ir::ir::{SsaCallArgs, SsaCallOperandLoc, SsaValue},
        },
        runtime::runtime_call::{
            RuntimeCallFrameRegion, RuntimeCallMeta, RuntimeCallTargetKind,
            RuntimeCallTypeCheckKind,
        },
    },
};

use crate::collections;

use super::{
    lower_const_pool::ConstPoolBuilder,
    lower_context::{BlockLowerContext, ValueRegs},
    lower_module::slot_offset_bytes,
    lower_regalloc::machine_block_params_for_value,
    lower_regalloc::{canonical_value_mem_width_for_value, lir_value_storage_type},
};

impl<'a> BlockLowerContext<'a> {
    pub(super) fn lower_call_internal(
        &mut self,
        callee: u32,
        args: &SsaCallArgs,
        results: FrameSpan,
        live_result: Option<SsaValue>,
        continuation: MachineBlockId,
    ) -> Result<(MachineTerminator, collections::Vec<MachineBlockParam>), WasmError> {
        self.publish_register_params_to_frame()?;
        let arg_span = args.frame_span();
        let callee_id = MachineFuncId(callee);
        let (
            callee_frame_prefix_slots,
            callee_total_frame_slots,
            callee_results,
            return_abi,
            param_locs,
        ) = {
            let callee_runtime = self.runtime_for_func(callee_id)?;
            let results_count = callee_runtime
                .return_results
                .map(|region| region.slots)
                .unwrap_or(0);
            (
                callee_runtime.frame_prefix_slots,
                callee_runtime.total_frame_slots,
                results_count,
                callee_runtime.return_abi.clone(),
                callee_runtime.param_locs.clone(),
            )
        };
        if args.total_params > callee_frame_prefix_slots {
            return Err(WasmError::internal(
                "direct local call passes more arguments than fit in the callee local prefix"
                    .into(),
            ));
        }
        if callee_results != results.count {
            return Err(WasmError::internal(
                "direct local call result span does not match the callee return-result contract"
                    .into(),
            ));
        }

        let mut load_args_from_frame = false;
        if self.use_explicit_stack_prechecks() {
            let call_regs = match self.borrow_free_gp_dynamic_regs(2) {
                Ok(regs) => regs,
                Err(_) => {
                    self.publish_call_args_to_frame(args)?;
                    self.ensure_no_live_values(
                        "prepared SSA-IR call fallback reached native lowering with live linear SSA values; values must be published before the call",
                    )?;
                    load_args_from_frame = true;
                    self.borrow_free_gp_dynamic_regs(2)?
                }
            };
            let callee_frame_base = call_regs[0];
            let stack_limit = call_regs[1];

            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: self.gp_word_int_width(),
                    op: MachineIntBinaryOp::Add,
                    dst: callee_frame_base,
                    lhs: MachineValue::Reg(self.frame_base_reg()),
                    rhs: MachineValue::Imm64(slot_offset_bytes(arg_span.start)? as u64),
                },
            });

            self.emit_direct_call_stack_precheck(
                callee_frame_base,
                stack_limit,
                callee_total_frame_slots,
            )?;
        }
        let machine_args = if load_args_from_frame {
            self.prepare_internal_call_args_from_frame(args, &param_locs, args.frame_base)?
        } else {
            self.prepare_internal_call_args(args, &param_locs)?
        };
        self.ensure_no_live_values(
            "prepared SSA-IR call reached native lowering with live linear SSA values; values must be published before the call",
        )?;
        let (cache_success_args, cache_continuation_params) =
            self.prepare_cached_cells_for_local_call()?;

        // Note: zero-init of the callee's non-param locals is performed by
        // the callee itself at function entry, only for slots flagged by the
        // SsaProgram's `cell_info.reads_before_write` analysis. Locals
        // that are guaranteed to be written before any read need no init.

        let (results, mut success_args, mut continuation_params) =
            self.call_result_placement(&return_abi, results, live_result)?;
        success_args.extend(cache_success_args);
        continuation_params.extend(cache_continuation_params);

        Ok((
            MachineTerminator::Call {
                target: MachineCallTarget::Direct(callee_id),
                frame_delta: slot_offset_bytes(arg_span.start)? as u32,
                args: machine_args,
                results,
                success: crate::vm::machine::machine_ir::MachineEdge {
                    target: continuation,
                    args: success_args,
                },
            },
            continuation_params,
        ))
    }

    pub(super) fn lower_tail_call_internal(
        &mut self,
        callee: u32,
        args: &SsaCallArgs,
        return_results: Option<FrameSpan>,
    ) -> Result<MachineTerminator, WasmError> {
        self.publish_call_args_to_frame(args)?;
        self.publish_register_params_to_frame()?;
        self.ensure_no_live_values(
            "prepared SSA-IR tail call reached native lowering with live linear SSA values; values must be published before the tail transfer",
        )?;

        let arg_span = args.frame_span();
        let callee_id = MachineFuncId(callee);
        let (callee_frame_prefix_slots, callee_total_frame_slots, callee_results) = {
            let callee_runtime = self.runtime_for_func(callee_id)?;
            let results_count = callee_runtime
                .return_results
                .map(|region| region.slots)
                .unwrap_or(0);
            (
                callee_runtime.frame_prefix_slots,
                callee_runtime.total_frame_slots,
                results_count,
            )
        };
        if args.total_params > callee_frame_prefix_slots {
            return Err(WasmError::internal(
                "direct local tail call passes more arguments than fit in the callee local prefix"
                    .into(),
            ));
        }
        let return_slots = return_results.map(|region| region.count).unwrap_or(0);
        if callee_results != return_slots {
            return Err(WasmError::internal(
                "direct local tail call result span does not match the callee return-result contract"
                    .into(),
            ));
        }

        self.emit_save_dirty_cached_cells()?;

        self.emit_repack_tail_call_args_to_frame_prefix(arg_span)?;
        let param_locs = self.runtime_for_func(callee_id)?.param_locs.clone();
        if self.use_explicit_stack_prechecks() {
            let call_regs = self.borrow_free_gp_dynamic_regs(2)?;
            let callee_frame_base = call_regs[0];
            let stack_limit = call_regs[1];
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Move {
                    owner: MachineRegOwner::LinearValue,
                    ty: MachineStorageType::GpWord,
                    dst: callee_frame_base,
                    src: MachineValue::Reg(self.frame_base_reg()),
                },
            });

            self.emit_direct_call_stack_precheck(
                callee_frame_base,
                stack_limit,
                callee_total_frame_slots,
            )?;
        }
        let machine_args =
            self.prepare_internal_call_args_from_frame(args, &param_locs, FrameSlot(0))?;

        Ok(MachineTerminator::TailCall {
            target: MachineCallTarget::Direct(callee_id),
            args: machine_args,
        })
    }

    pub(super) fn emit_repack_tail_call_args_to_frame_prefix(
        &mut self,
        args: FrameSpan,
    ) -> Result<(), WasmError> {
        if args.start == FrameSlot(0) {
            return Ok(());
        }

        if self.gp_reg_width() == 4 {
            let temps = self.borrow_free_gp_dynamic_regs(2)?;
            let tmp_lo = temps[0];
            let tmp_hi = temps[1];
            for index in 0..args.count {
                let src_slot = args.start.advance(index);
                let dst_slot = FrameSlot(index);
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: tmp_lo,
                        addr: self.frame_addr_offset(src_slot, 0)?,
                        width: MachineMemWidth::U32,
                        extension: MachineLoadExtension::None,
                    },
                });
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: tmp_hi,
                        addr: self.frame_addr_offset(src_slot, 4)?,
                        width: MachineMemWidth::U32,
                        extension: MachineLoadExtension::None,
                    },
                });
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: self.frame_addr_offset(dst_slot, 0)?,
                        width: MachineMemWidth::U32,
                        src: MachineValue::Reg(tmp_lo),
                    },
                });
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: self.frame_addr_offset(dst_slot, 4)?,
                        width: MachineMemWidth::U32,
                        src: MachineValue::Reg(tmp_hi),
                    },
                });
            }
        } else {
            let temps = self.borrow_free_gp_dynamic_regs(1)?;
            let tmp = temps[0];
            for index in 0..args.count {
                let src_slot = args.start.advance(index);
                let dst_slot = FrameSlot(index);
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: tmp,
                        addr: self.frame_addr(src_slot)?,
                        width: MachineMemWidth::U64,
                        extension: MachineLoadExtension::None,
                    },
                });
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: self.frame_addr(dst_slot)?,
                        width: MachineMemWidth::U64,
                        src: MachineValue::Reg(tmp),
                    },
                });
            }
        }
        Ok(())
    }

    pub(super) fn lower_call_runtime(
        &mut self,
        func_idx: u32,
        args: &SsaCallArgs,
        results: FrameSpan,
        live_result: Option<SsaValue>,
        const_pool: &mut ConstPoolBuilder,
    ) -> Result<(), WasmError> {
        self.publish_call_args_to_frame(args)?;
        self.ensure_no_live_values(
            "prepared SSA-IR runtime call reached native lowering with live linear SSA values; values must be published before the call",
        )?;

        // Runtime-dispatch calls are modeled as inline runtime-entry calls
        // rather than MIR terminators. Unlike local calls, they do not transfer
        // control to another compiled MachineIR function or create a MachineIR
        // continuation edge; the runtime-call entry returns inline to this
        // block.
        //
        // The surrounding MIR still makes the call boundary explicit by
        // emitting cached-local save/reload and mem0 cache refresh around the
        // runtime-call instruction sequence.
        let metadata =
            self.build_call_runtime_meta_direct(func_idx, args.frame_span(), results, const_pool);
        self.emit_call_runtime(metadata)?;
        if let Some(value) = live_result {
            self.materialize_runtime_call_result(value, results.start)?;
        }
        Ok(())
    }

    pub(super) fn call_result_placement(
        &mut self,
        return_abi: &MachineReturnAbi,
        caller_results: FrameSpan,
        live_result: Option<SsaValue>,
    ) -> Result<
        (
            MachineCallResults,
            collections::Vec<MachineValue>,
            collections::Vec<MachineBlockParam>,
        ),
        WasmError,
    > {
        let Some(value) = live_result else {
            return Ok((
                call_results(return_abi, caller_results),
                collections::Vec::new(),
                collections::Vec::new(),
            ));
        };
        if self.remaining_use_count(value) == 0 {
            return Ok((
                MachineCallResults::None,
                collections::Vec::new(),
                collections::Vec::new(),
            ));
        }
        self.apply_sink_premap(&[], &[value])?;
        if caller_results.count == 0 {
            return Err(WasmError::internal(
                "live scalar call result requested for zero-result call".into(),
            ));
        }
        if caller_results.count != 1 {
            return Err(WasmError::internal(
                "live scalar call result requested for multi-result call".into(),
            ));
        }

        match return_abi {
            MachineReturnAbi::ScalarGp { ty } => {
                let value_ty = lir_value_storage_type(self.program(), value);
                if value_ty != *ty {
                    return Err(WasmError::internal(
                        "live GP scalar call result type does not match callee return ABI".into(),
                    ));
                }
                let reg = self.alloc_value_in_bank(value, *ty)?;
                Ok((
                    MachineCallResults::ScalarGp {
                        dst: MachineResultDst::Reg(reg),
                        ty: *ty,
                    },
                    collections::vec![MachineValue::Reg(reg)],
                    machine_block_params_for_value(ValueRegs { lo: reg, hi: None }, *ty),
                ))
            }
            MachineReturnAbi::ScalarGpPair => {
                let value_ty = lir_value_storage_type(self.program(), value);
                if value_ty != MachineStorageType::GpI64 {
                    return Err(WasmError::internal(
                        "live GP-pair call result type does not match callee return ABI".into(),
                    ));
                }
                let (lo, hi) = self.alloc_i64_value_pair(value)?;
                Ok((
                    MachineCallResults::ScalarGpPair {
                        lo: MachineResultDst::Reg(lo),
                        hi: MachineResultDst::Reg(hi),
                    },
                    collections::vec![MachineValue::Reg(lo), MachineValue::Reg(hi)],
                    machine_block_params_for_value(
                        ValueRegs { lo, hi: Some(hi) },
                        MachineStorageType::GpI64,
                    ),
                ))
            }
            MachineReturnAbi::ScalarFp { ty } => {
                let value_ty = lir_value_storage_type(self.program(), value);
                if value_ty != *ty {
                    return Err(WasmError::internal(
                        "live FP scalar call result type does not match callee return ABI".into(),
                    ));
                }
                let reg = self.alloc_value_in_bank(value, *ty)?;
                Ok((
                    MachineCallResults::ScalarFp {
                        dst: MachineResultDst::Reg(reg),
                        ty: *ty,
                    },
                    collections::vec![MachineValue::Reg(reg)],
                    machine_block_params_for_value(ValueRegs { lo: reg, hi: None }, *ty),
                ))
            }
            MachineReturnAbi::None | MachineReturnAbi::FrameFallback { .. } => Err(
                WasmError::internal("live scalar call result requires a scalar return ABI".into()),
            ),
        }
    }

    pub(super) fn materialize_runtime_call_result(
        &mut self,
        value: SsaValue,
        slot: FrameSlot,
    ) -> Result<(), WasmError> {
        if self.remaining_use_count(value) == 0 {
            return Ok(());
        }
        self.apply_sink_premap(&[], &[value])?;
        let ty = lir_value_storage_type(self.program(), value);
        if matches!(ty, MachineStorageType::GpI64) {
            let ops = self.i64_ops();
            ops.emit_load_slot_i64(self, slot, value)?;
            return Ok(());
        }

        #[cfg(sf_has_simd)]
        if matches!(ty, MachineStorageType::V128) {
            return Err(WasmError::internal(
                "runtime call scalar result cannot materialize v128 through scalar ABI".into(),
            ));
        }

        let dst = self.alloc_value_in_bank(value, ty)?;
        let width = canonical_value_mem_width_for_value(self.program(), value);
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty,
                dst,
                addr: self.frame_addr(slot)?,
                width,
                extension: MachineLoadExtension::None,
            },
        });
        Ok(())
    }

    pub(super) fn build_call_runtime_meta_direct(
        &self,
        func_idx: u32,
        args: FrameSpan,
        results: FrameSpan,
        const_pool: &mut ConstPoolBuilder,
    ) -> MachineConstId {
        // Direct runtime calls carry the Wasm function index as an immediate in
        // the metadata record. The runtime entry will use it directly.
        const_pool.call_runtime_meta(RuntimeCallMeta {
            func_idx_source: func_idx,
            func_idx_source_kind: RuntimeCallTargetKind::Immediate as u32,
            expected_type_idx: u32::MAX,
            type_check_kind: RuntimeCallTypeCheckKind::None as u32,
            args: runtime_call_region(args),
            results: runtime_call_region(results),
        })
    }

    pub(super) fn build_call_runtime_meta_indirect(
        &self,
        type_idx: u32,
        func_idx_slot: FrameSlot,
        args: FrameSpan,
        results: FrameSpan,
        const_pool: &mut ConstPoolBuilder,
    ) -> MachineConstId {
        // Indirect runtime calls resolve the target at runtime. MachineIR
        // publishes the frame slot that now holds the resolved function index,
        // and the runtime entry reloads it from there.
        const_pool.call_runtime_meta(RuntimeCallMeta {
            func_idx_source: func_idx_slot.0 as u32,
            func_idx_source_kind: RuntimeCallTargetKind::FrameSlot as u32,
            expected_type_idx: type_idx,
            type_check_kind: RuntimeCallTypeCheckKind::IndirectCall as u32,
            args: runtime_call_region(args),
            results: runtime_call_region(results),
        })
    }

    pub(super) fn build_call_runtime_meta_ref(
        &self,
        type_idx: u32,
        ref_slot: FrameSlot,
        args: FrameSpan,
        results: FrameSpan,
        const_pool: &mut ConstPoolBuilder,
    ) -> MachineConstId {
        const_pool.call_runtime_meta(RuntimeCallMeta {
            func_idx_source: ref_slot.0 as u32,
            func_idx_source_kind: RuntimeCallTargetKind::FrameSlot as u32,
            expected_type_idx: type_idx,
            type_check_kind: RuntimeCallTypeCheckKind::CallRef as u32,
            args: runtime_call_region(args),
            results: runtime_call_region(results),
        })
    }
    pub(super) fn build_runtime_call_ops(
        &self,
        metadata: MachineConstId,
    ) -> collections::Vec<MachineInst> {
        // The runtime-call instruction itself is the foreign-call boundary.
        // The reloads that follow repair machine-visible runtime state that may
        // have changed while the host callback executed, most importantly the
        // cached mem0 base/size pair after a possible memory growth.
        collections::vec![
            MachineInst {
                kind: MachineInstKind::CallRuntime(MachineCallRuntime { metadata }),
            },
            MachineInst {
                kind: MachineInstKind::Load {
                    owner: MachineRegOwner::LinearValue,
                    ty: MachineStorageType::GpWord,
                    dst: self.regfile().mem0_base(),
                    addr: self.runtime_addr(self.runtime_abi_layout().context.mem0_base_offset),
                    width: self.gp_word_mem_width(),
                    extension: MachineLoadExtension::None,
                },
            },
            MachineInst {
                kind: MachineInstKind::Load {
                    owner: MachineRegOwner::LinearValue,
                    ty: MachineStorageType::GpWord,
                    dst: self.regfile().mem0_size(),
                    addr: self.runtime_addr(self.runtime_abi_layout().context.mem0_size_offset),
                    width: self.gp_word_mem_width(),
                    extension: MachineLoadExtension::None,
                },
            },
        ]
    }

    pub(super) fn emit_call_runtime(&mut self, metadata: MachineConstId) -> Result<(), WasmError> {
        // Runtime calls are frame-slot based and return inline in the same
        // MIR block. Because the runtime call may clobber every machine
        // register, cached locals must be published to their canonical frame
        // slots before the call. Re-caching after the call is explicit in
        // SSA-IR.
        self.publish_register_params_to_frame()?;
        self.emit_save_dirty_cached_cells()?;
        self.emit_machine_ops(self.build_runtime_call_ops(metadata));
        Ok(())
    }

    pub(super) fn publish_call_args_to_frame(
        &mut self,
        args: &SsaCallArgs,
    ) -> Result<(), WasmError> {
        for arg in &args.live_suffix {
            self.publish_value_to_frame_slot(arg.value, arg.frame_slot)?;
        }
        self.release_dead_values()?;
        Ok(())
    }

    pub(super) fn publish_call_operand_to_frame(
        &mut self,
        loc: &SsaCallOperandLoc,
    ) -> Result<FrameSlot, WasmError> {
        match loc {
            SsaCallOperandLoc::Stack { slot } => Ok(*slot),
            SsaCallOperandLoc::Live { value, slot, .. } => {
                self.publish_value_to_frame_slot(*value, *slot)?;
                self.release_dead_values()?;
                Ok(*slot)
            }
        }
    }

    pub(super) fn prepare_internal_call_args_from_frame(
        &mut self,
        args: &SsaCallArgs,
        param_locs: &[MachineParamLoc],
        frame_base: FrameSlot,
    ) -> Result<MachineCallArgs, WasmError> {
        self.prepare_internal_call_args_from_frame_span(
            FrameSpan::new(frame_base, args.total_params),
            param_locs,
        )
    }

    pub(super) fn prepare_internal_call_args_from_frame_span(
        &mut self,
        args: FrameSpan,
        param_locs: &[MachineParamLoc],
    ) -> Result<MachineCallArgs, WasmError> {
        let mut call_args = self.empty_call_args_from_span(args, param_locs);
        for loc in param_locs {
            match *loc {
                MachineParamLoc::Frame { .. } => {}
                MachineParamLoc::GpArg {
                    param_index,
                    lane,
                    ty,
                } => {
                    let slot = args.start.advance(param_index);
                    call_args.lane_args.push(MachineCallLaneArg::Gp {
                        param_index,
                        lane,
                        src: MachineArgSrc::FrameSlot(slot),
                        ty,
                    });
                }
                MachineParamLoc::GpArgPair {
                    param_index,
                    lo_lane,
                    hi_lane,
                } => {
                    let slot = args.start.advance(param_index);
                    call_args.lane_args.push(MachineCallLaneArg::GpPair {
                        param_index,
                        lo_lane,
                        hi_lane,
                        src: MachineArgSrcPair {
                            lo: MachineArgSrc::FrameSlot(slot),
                            hi: MachineArgSrc::FrameSlotOffset {
                                slot,
                                byte_offset: 4,
                            },
                        },
                    });
                }
                MachineParamLoc::FpArg {
                    param_index,
                    lane,
                    ty,
                } => {
                    let slot = args.start.advance(param_index);
                    call_args.lane_args.push(MachineCallLaneArg::Fp {
                        param_index,
                        lane,
                        src: MachineArgSrc::FrameSlot(slot),
                        ty,
                    });
                }
            }
        }
        Ok(call_args)
    }

    fn prepare_internal_call_args(
        &mut self,
        args: &SsaCallArgs,
        param_locs: &[MachineParamLoc],
    ) -> Result<MachineCallArgs, WasmError> {
        let mut call_args = self.empty_call_args(args, param_locs, args.frame_base);
        for loc in param_locs {
            match *loc {
                MachineParamLoc::Frame { param_index, .. } => {
                    if let Some(arg) = call_live_arg(args, param_index) {
                        self.publish_value_to_frame_slot(arg.value, arg.frame_slot)?;
                    }
                }
                MachineParamLoc::GpArg {
                    param_index,
                    lane,
                    ty,
                } => {
                    let slot = args.frame_base.advance(param_index);
                    let src = if let Some(arg) = call_live_arg(args, param_index) {
                        let src = self.try_value_reg(arg.value).ok_or_else(|| {
                            WasmError::internal("live call argument is missing a machine register")
                        })?;
                        let _ = self.use_value(arg.value)?;
                        MachineArgSrc::Reg(src)
                    } else {
                        MachineArgSrc::FrameSlot(slot)
                    };
                    call_args.lane_args.push(MachineCallLaneArg::Gp {
                        param_index,
                        lane,
                        src,
                        ty,
                    });
                }
                MachineParamLoc::GpArgPair {
                    param_index,
                    lo_lane,
                    hi_lane,
                } => {
                    let slot = args.frame_base.advance(param_index);
                    let src = if let Some(arg) = call_live_arg(args, param_index) {
                        let (src_lo, src_hi) = self
                            .try_value_regs(arg.value)
                            .and_then(|(lo, hi)| hi.map(|hi| (lo, hi)))
                            .ok_or_else(|| {
                                WasmError::internal(
                                    "live i64 call argument is missing a machine register pair",
                                )
                            })?;
                        let _ = self.use_i64_value_pair(arg.value)?;
                        MachineArgSrcPair {
                            lo: MachineArgSrc::Reg(src_lo),
                            hi: MachineArgSrc::Reg(src_hi),
                        }
                    } else {
                        MachineArgSrcPair {
                            lo: MachineArgSrc::FrameSlot(slot),
                            hi: MachineArgSrc::FrameSlotOffset {
                                slot,
                                byte_offset: 4,
                            },
                        }
                    };
                    call_args.lane_args.push(MachineCallLaneArg::GpPair {
                        param_index,
                        lo_lane,
                        hi_lane,
                        src,
                    });
                }
                MachineParamLoc::FpArg {
                    param_index,
                    lane,
                    ty,
                } => {
                    let slot = args.frame_base.advance(param_index);
                    let src = if let Some(arg) = call_live_arg(args, param_index) {
                        let src = self.try_value_reg(arg.value).ok_or_else(|| {
                            WasmError::internal(
                                "live FP call argument is missing a machine register",
                            )
                        })?;
                        let _ = self.use_value(arg.value)?;
                        MachineArgSrc::Reg(src)
                    } else {
                        MachineArgSrc::FrameSlot(slot)
                    };
                    call_args.lane_args.push(MachineCallLaneArg::Fp {
                        param_index,
                        lane,
                        src,
                        ty,
                    });
                }
            }
        }
        self.release_dead_values()?;
        Ok(call_args)
    }

    fn empty_call_args(
        &self,
        args: &SsaCallArgs,
        param_locs: &[MachineParamLoc],
        frame_base: FrameSlot,
    ) -> MachineCallArgs {
        self.empty_call_args_from_span(FrameSpan::new(frame_base, args.total_params), param_locs)
    }

    fn empty_call_args_from_span(
        &self,
        args: FrameSpan,
        param_locs: &[MachineParamLoc],
    ) -> MachineCallArgs {
        let mut frame_count = 0u16;
        while frame_count < args.count {
            match param_locs.get(frame_count as usize) {
                Some(MachineParamLoc::Frame { .. }) | None => {
                    frame_count = frame_count.saturating_add(1);
                }
                Some(_) => break,
            }
        }
        MachineCallArgs {
            frame_params: frame_region(FrameSpan::new(args.start, frame_count)),
            lane_args: collections::Vec::new(),
        }
    }

    pub(super) fn publish_value_to_frame_slot(
        &mut self,
        value: SsaValue,
        slot: FrameSlot,
    ) -> Result<(), WasmError> {
        let ty = lir_value_storage_type(self.program(), value);
        if matches!(ty, MachineStorageType::GpI64) {
            let ops = self.i64_ops();
            ops.emit_store_slot_i64(self, slot, value)?;
            return Ok(());
        }
        #[cfg(sf_has_simd)]
        if matches!(ty, MachineStorageType::V128) {
            let src_reg = self.use_value(value)?;
            self.emit_store_frame_v128(slot, src_reg)?;
            return Ok(());
        }
        let src_reg = self.use_value(value)?;
        let width = canonical_value_mem_width_for_value(self.program(), value);
        let addr = self.frame_addr(slot)?;
        if !self.try_coalesce_last_store_immediate(value, src_reg, ty, addr, width) {
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Store {
                    ty,
                    addr,
                    width,
                    src: MachineValue::Reg(src_reg),
                },
            });
        }
        Ok(())
    }

    fn emit_direct_call_stack_precheck(
        &mut self,
        callee_frame_base: MachineReg,
        stack_limit: MachineReg,
        callee_total_frame_slots: u16,
    ) -> Result<(), WasmError> {
        if !self.use_explicit_stack_prechecks() {
            return Ok(());
        }

        // Stack growth is downward. Load the stack-end guard, subtract the
        // callee frame footprint, and trap if the proposed callee frame base
        // would cross below that limit.
        let callee_total_bytes = slot_offset_bytes(FrameSlot(callee_total_frame_slots))? as u64;
        let stack_end_offset = self.runtime_abi_layout().context.stack_end_offset;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: stack_limit,
                addr: self.runtime_addr(stack_end_offset),
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::IntBinary {
                width: self.gp_word_int_width(),
                op: MachineIntBinaryOp::Sub,
                dst: stack_limit,
                lhs: MachineValue::Reg(stack_limit),
                rhs: MachineValue::Imm64(callee_total_bytes),
            },
        });
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::TrapIf {
                kind: MachineTrapKind::StackOverflow,
                cond: MachineBranchCond::IntCompare {
                    width: self.gp_word_int_width(),
                    kind: MachineCompareKind::Gt,
                    sign: MachineSign::Unsigned,
                    lhs: MachineValue::Reg(callee_frame_base),
                    rhs: MachineValue::Reg(stack_limit),
                },
            },
        });
        Ok(())
    }

    pub(super) fn emit_dynamic_call_stack_precheck(
        &mut self,
        callee_frame_base: MachineReg,
        stack_limit: MachineReg,
        callee_total_frame_bytes: MachineReg,
    ) -> Result<(), WasmError> {
        if !self.use_explicit_stack_prechecks() {
            return Ok(());
        }

        // Same check as the direct path, except the callee frame size has been
        // loaded dynamically from the runtime-published local-call-info table.
        let stack_end_offset = self.runtime_abi_layout().context.stack_end_offset;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: stack_limit,
                addr: self.runtime_addr(stack_end_offset),
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::IntBinary {
                width: self.gp_word_int_width(),
                op: MachineIntBinaryOp::Sub,
                dst: stack_limit,
                lhs: MachineValue::Reg(stack_limit),
                rhs: MachineValue::Reg(callee_total_frame_bytes),
            },
        });
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::TrapIf {
                kind: MachineTrapKind::StackOverflow,
                cond: MachineBranchCond::IntCompare {
                    width: self.gp_word_int_width(),
                    kind: MachineCompareKind::Gt,
                    sign: MachineSign::Unsigned,
                    lhs: MachineValue::Reg(callee_frame_base),
                    rhs: MachineValue::Reg(stack_limit),
                },
            },
        });
        Ok(())
    }
}

fn call_live_arg(
    args: &SsaCallArgs,
    param_index: u16,
) -> Option<&crate::vm::middle::ssa_ir::ir::SsaCallLiveArg> {
    args.live_suffix
        .iter()
        .find(|arg| arg.param_index == param_index)
}

#[inline]
fn runtime_call_region(span: FrameSpan) -> RuntimeCallFrameRegion {
    // Runtime calls use frame-relative regions all the way down to the runtime
    // entry, so the Wasm frame span lowers directly to the serialized ABI
    // record without any extra scratch slot or repacking.
    RuntimeCallFrameRegion {
        base_slot: span.start.0,
        slots: span.count,
    }
}

#[inline]
fn frame_region(span: FrameSpan) -> MachineFrameRegion {
    MachineFrameRegion {
        base_slot: span.start.0,
        slots: span.count,
    }
}

#[inline]
fn call_results(return_abi: &MachineReturnAbi, caller_results: FrameSpan) -> MachineCallResults {
    if caller_results.count == 0 {
        return MachineCallResults::None;
    }
    match return_abi {
        MachineReturnAbi::None => MachineCallResults::None,
        MachineReturnAbi::ScalarGp { ty } => MachineCallResults::ScalarGp {
            dst: MachineResultDst::FrameSlot(caller_results.start),
            ty: *ty,
        },
        MachineReturnAbi::ScalarGpPair => MachineCallResults::ScalarGpPair {
            lo: MachineResultDst::FrameSlot(caller_results.start),
            hi: MachineResultDst::FrameSlot(caller_results.start),
        },
        MachineReturnAbi::ScalarFp { ty } => MachineCallResults::ScalarFp {
            dst: MachineResultDst::FrameSlot(caller_results.start),
            ty: *ty,
        },
        MachineReturnAbi::FrameFallback { results } => MachineCallResults::FrameFallback {
            callee_results: *results,
            caller_results: frame_region(caller_results),
        },
    }
}
