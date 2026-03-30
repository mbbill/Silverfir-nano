use crate::{
    error::WasmError,
    vm::{
        machine::machine_ir::{
            MachineBlockId, MachineBranchCond, MachineCompareKind, MachineConstId, MachineExternId,
            MachineFuncId, MachineFrameRegion, MachineHelperCall, MachineHelperSymbol,
            MachineInst, MachineInstKind, MachineIntBinaryOp, MachineLoadExtension, MachineMemWidth,
            MachineReg, MachineSign, MachineStorageType, MachineTerminator, MachineTrapKind,
            MachineValue,
        },
        middle::{
            frame::{FrameSlot, FrameSpan},
            ssa_ir::ir::SsaBoundaryOp,
        },
        runtime::helper_meta::{
            CallExternalMeta, CallIndirectExternalMeta, DataDropMeta, ElemDropMeta,
            HelperFrameRegion, MemoryCopyMeta, MemoryFillMeta, MemoryGrowMeta, MemoryInitMeta,
            TableCopyMeta, TableFillMeta, TableGrowMeta, TableInitMeta,
        },
    },
};

use super::{lower_context::BlockLowerContext, lower_module::slot_offset_bytes, lower_sidecar::SidecarBuilder};

impl<'a> BlockLowerContext<'a> {
    pub(super) fn lower_call_internal(
        &mut self,
        callee: u32,
        args: FrameSpan,
        results: FrameSpan,
        continuation: MachineBlockId,
    ) -> Result<MachineTerminator, WasmError> {
        self.ensure_no_live_values(
            "prepared SSA-IR call reached native lowering with live transient SSA values; values must be published before the call",
        )?;

        let callee_id = MachineFuncId(callee);
        let callee_runtime = self.runtime_for_func(callee_id)?;
        let call_scratch = callee_runtime.call_scratch.ok_or_else(|| {
            WasmError::internal("direct local call requires callee call scratch".into())
        })?;
        if call_scratch.slots < self.call_link_layout().slot_count {
            return Err(WasmError::internal(
                "callee call scratch is smaller than the machine call-link layout".into(),
            ));
        }
        if args.count > callee_runtime.frame_prefix_slots {
            return Err(WasmError::internal(
                "direct local call passes more arguments than fit in the callee local prefix"
                    .into(),
            ));
        }
        let callee_results = callee_runtime
            .return_results
            .map(|region| region.slots)
            .unwrap_or(0);
        if callee_results != results.count {
            return Err(WasmError::internal(
                "direct local call result span does not match the callee return-result contract"
                    .into(),
            ));
        }

        self.emit_save_dirty_cached_locals()?;

        let call_regs = self.borrow_free_transients(2)?;
        let callee_frame_base = call_regs[0];
        let stack_limit = call_regs[1];

        // Native local calls reuse the caller operand span as the callee frame
        // prefix, so arguments are already in place when control transfers.
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::IntBinary {
                width: self.gp_word_int_width(),
                op: MachineIntBinaryOp::Add,
                dst: callee_frame_base,
                lhs: MachineValue::Reg(self.frame_base_reg()),
                rhs: MachineValue::Imm64(slot_offset_bytes(args.start)? as u64),
            },
        });

        self.emit_direct_call_stack_precheck(
            callee_frame_base,
            stack_limit,
            callee_runtime.total_frame_slots,
        )?;

        for slot in args.count..callee_runtime.frame_prefix_slots {
            if self.gp_reg_width() == 4 {
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: self.frame_region_addr(
                            callee_frame_base,
                            MachineFrameRegion {
                                base_slot: slot,
                                slots: 1,
                            },
                            0,
                        )?,
                        width: MachineMemWidth::U32,
                        src: MachineValue::Imm64(0),
                    },
                });
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: self.frame_region_addr(
                            callee_frame_base,
                            MachineFrameRegion {
                                base_slot: slot,
                                slots: 1,
                            },
                            4,
                        )?,
                        width: MachineMemWidth::U32,
                        src: MachineValue::Imm64(0),
                    },
                });
            } else {
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpI64,
                        addr: self.frame_addr_from(callee_frame_base, FrameSlot(slot))?,
                        width: MachineMemWidth::U64,
                        src: MachineValue::Imm64(0),
                    },
                });
            }
        }

        self.store_call_link(callee_frame_base, call_scratch, continuation, results)?;

        Ok(MachineTerminator::CallDirect {
            callee: callee_id,
            callee_frame_base,
            continuation,
        })
    }

    pub(super) fn lower_call_external(
        &mut self,
        func_idx: u32,
        args: FrameSpan,
        results: FrameSpan,
        sidecar: &mut SidecarBuilder,
    ) -> Result<(), WasmError> {
        self.ensure_no_live_values(
            "prepared SSA-IR external call reached native lowering with live transient SSA values; values must be published before the call",
        )?;

        let target = sidecar.extern_target(MachineHelperSymbol::CallExternal);
        let metadata = sidecar.call_external_meta(CallExternalMeta {
            func_idx,
            args: args.into(),
            results: results.into(),
        });
        self.emit_helper_backed_boundary(target, metadata)
    }

    pub(super) fn call_indirect_external_site(
        &self,
        func_idx_slot: FrameSlot,
        args: FrameSpan,
        results: FrameSpan,
        sidecar: &mut SidecarBuilder,
    ) -> MachineHelperCall {
        let target = sidecar.extern_target(MachineHelperSymbol::CallIndirectExternal);
        let metadata = sidecar.call_indirect_external_meta(CallIndirectExternalMeta {
            func_idx_slot: func_idx_slot.0 as u32,
            args: args.into(),
            results: results.into(),
        });
        MachineHelperCall { target, metadata }
    }

    pub(super) fn lower_runtime(
        &mut self,
        boundary: &SsaBoundaryOp,
        sidecar: &mut SidecarBuilder,
    ) -> Result<(), WasmError> {
        self.ensure_no_live_values(
            "prepared SSA-IR runtime boundary reached native lowering with live transient SSA values; values must be published before the boundary",
        )?;

        let (target, metadata) = self.runtime_call_site(boundary, sidecar)?;
        self.emit_helper_backed_boundary(target, metadata)
    }

    fn emit_helper_backed_boundary(
        &mut self,
        target: MachineExternId,
        metadata: MachineConstId,
    ) -> Result<(), WasmError> {
        // Helper-backed boundaries are slot-based and may clobber cache regs,
        // so cached locals must be synchronized through their canonical frame
        // slots across the helper call.
        self.emit_save_dirty_cached_locals()?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::CallHelper(MachineHelperCall { target, metadata }),
        });
        self.emit_reload_mem0_cache_regs();
        self.emit_reload_cached_locals()?;
        Ok(())
    }

    fn runtime_call_site(
        &self,
        boundary: &SsaBoundaryOp,
        sidecar: &mut SidecarBuilder,
    ) -> Result<
        (
            MachineExternId,
            MachineConstId,
        ),
        WasmError,
    > {
        use MachineHelperSymbol as H;

        let pair = match boundary {
            SsaBoundaryOp::MemoryGrow { mem_idx, io } => (
                sidecar.extern_target(H::MemoryGrow),
                sidecar.memory_grow_meta(MemoryGrowMeta {
                    mem_idx: *mem_idx,
                    io: (*io).into(),
                }),
            ),
            SsaBoundaryOp::MemoryFill { mem_idx, args } => (
                sidecar.extern_target(H::MemoryFill),
                sidecar.memory_fill_meta(MemoryFillMeta {
                    mem_idx: *mem_idx,
                    args: span_region_with_slots(*args, 3, "memory.fill args")?,
                }),
            ),
            SsaBoundaryOp::MemoryCopy {
                dst_mem_idx,
                src_mem_idx,
                args,
            } => (
                sidecar.extern_target(H::MemoryCopy),
                sidecar.memory_copy_meta(MemoryCopyMeta {
                    dst_mem_idx: *dst_mem_idx,
                    src_mem_idx: *src_mem_idx,
                    args: span_region_with_slots(*args, 3, "memory.copy args")?,
                }),
            ),
            SsaBoundaryOp::TableGrow {
                table_idx,
                args,
                results,
            } => (
                sidecar.extern_target(H::TableGrow),
                sidecar.table_grow_meta(TableGrowMeta {
                    table_idx: *table_idx,
                    args: span_region_with_slots(*args, 2, "table.grow args")?,
                    results: span_region_with_slots(*results, 1, "table.grow results")?,
                }),
            ),
            SsaBoundaryOp::TableFill { table_idx, args } => (
                sidecar.extern_target(H::TableFill),
                sidecar.table_fill_meta(TableFillMeta {
                    table_idx: *table_idx,
                    args: span_region_with_slots(*args, 3, "table.fill args")?,
                }),
            ),
            SsaBoundaryOp::TableCopy {
                dst_table_idx,
                src_table_idx,
                args,
            } => (
                sidecar.extern_target(H::TableCopy),
                sidecar.table_copy_meta(TableCopyMeta {
                    dst_table_idx: *dst_table_idx,
                    src_table_idx: *src_table_idx,
                    args: span_region_with_slots(*args, 3, "table.copy args")?,
                }),
            ),
            SsaBoundaryOp::MemoryInit {
                data_idx,
                mem_idx,
                args,
            } => (
                sidecar.extern_target(H::MemoryInit),
                sidecar.memory_init_meta(MemoryInitMeta {
                    data_idx: *data_idx,
                    mem_idx: *mem_idx,
                    args: span_region_with_slots(*args, 3, "memory.init args")?,
                }),
            ),
            SsaBoundaryOp::DataDrop { data_idx } => (
                sidecar.extern_target(H::DataDrop),
                sidecar.data_drop_meta(DataDropMeta {
                    data_idx: *data_idx,
                }),
            ),
            SsaBoundaryOp::TableInit {
                elem_idx,
                table_idx,
                args,
            } => (
                sidecar.extern_target(H::TableInit),
                sidecar.table_init_meta(TableInitMeta {
                    elem_idx: *elem_idx,
                    table_idx: *table_idx,
                    args: span_region_with_slots(*args, 3, "table.init args")?,
                }),
            ),
            SsaBoundaryOp::ElemDrop { elem_idx } => (
                sidecar.extern_target(H::ElemDrop),
                sidecar.elem_drop_meta(ElemDropMeta {
                    elem_idx: *elem_idx,
                }),
            ),
            _ => {
                return Err(WasmError::internal(
                    "non-runtime boundary reached runtime helper lowering".into(),
                ))
            }
        };
        Ok(pair)
    }

    fn emit_direct_call_stack_precheck(
        &mut self,
        callee_frame_base: MachineReg,
        stack_limit: MachineReg,
        callee_total_frame_slots: u16,
    ) -> Result<(), WasmError> {
        let callee_total_bytes = slot_offset_bytes(FrameSlot(callee_total_frame_slots))? as u64;
        let stack_end_offset = self.runtime_abi_layout().context.stack_end_offset;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
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

    fn store_call_link(
        &mut self,
        callee_frame_base: MachineReg,
        call_scratch: MachineFrameRegion,
        continuation: MachineBlockId,
        results: FrameSpan,
    ) -> Result<(), WasmError> {
        let caller_result_base = slot_offset_bytes(results.start)? as u64;
        let call_link = self.call_link_layout();
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Store {
                ty: MachineStorageType::GpWord,
                addr: self.frame_region_addr(
                    callee_frame_base,
                    call_scratch,
                    call_link.continuation_offset,
                )?,
                width: self.gp_word_mem_width(),
                src: MachineValue::Imm64(continuation.0 as u64),
            },
        });
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Store {
                ty: MachineStorageType::GpWord,
                addr: self.frame_region_addr(
                    callee_frame_base,
                    call_scratch,
                    call_link.caller_frame_offset,
                )?,
                width: self.gp_word_mem_width(),
                src: MachineValue::Reg(self.frame_base_reg()),
            },
        });
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Store {
                ty: MachineStorageType::GpWord,
                addr: self.frame_region_addr(
                    callee_frame_base,
                    call_scratch,
                    call_link.caller_result_base_offset,
                )?,
                width: self.gp_word_mem_width(),
                src: MachineValue::Imm64(caller_result_base),
            },
        });
        Ok(())
    }
}

fn span_region_with_slots(
    span: FrameSpan,
    slots: u16,
    context: &'static str,
) -> Result<HelperFrameRegion, WasmError> {
    if span.count != slots {
        return Err(WasmError::internal(alloc::format!(
            "{context} expected {} slots but got {}",
            slots,
            span.count,
        )));
    }
    Ok(span.into())
}
