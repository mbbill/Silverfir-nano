use crate::{
    error::WasmError,
    vm::{
        machine::machine_ir::{
            MachineAddr, MachineBlockId, MachineBranchCond, MachineCallExternal,
            MachineCompareKind, MachineConstId, MachineFuncId, MachineInst, MachineInstKind,
            MachineIntBinaryOp, MachineLoadExtension, MachineMemWidth, MachineReg, MachineRegOwner,
            MachineSign, MachineStorageType, MachineTerminator, MachineTrapKind, MachineValue,
        },
        middle::frame::{FrameSlot, FrameSpan},
        runtime::external::{ExternalCallFrameRegion, ExternalCallMeta, ExternalCallTargetKind},
    },
};

use crate::collections;

use super::{
    lower_const_pool::ConstPoolBuilder, lower_context::BlockLowerContext,
    lower_module::slot_offset_bytes,
};

impl<'a> BlockLowerContext<'a> {
    pub(super) fn lower_call_internal(
        &mut self,
        callee: u32,
        args: FrameSpan,
        results: FrameSpan,
        continuation: MachineBlockId,
    ) -> Result<MachineTerminator, WasmError> {
        self.ensure_no_live_values(
            "prepared SSA-IR call reached native lowering with live linear SSA values; values must be published before the call",
        )?;

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
        if args.count > callee_frame_prefix_slots {
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

        self.emit_save_dirty_cached_locals()?;

        // Two scratch registers:
        // - `callee_frame_base` is the absolute address of the callee's frame
        //   base. It is alive at the terminator: the backend reads it to set
        //   `MACHINE_FP_REG` before the native call.
        // - `stack_limit` is used by the per-call stack-overflow precheck and
        //   then reused after the precheck as `caller_result_base` (the
        //   absolute address of the caller's result-receive region). The
        //   backend reads `caller_result_base` at the terminator to push it
        //   into the backend-private call record so the callee's `Return`
        //   can copy results into it.
        //
        // (1D will hoist the stack precheck out of the call site, at which
        // point a single temp will suffice.)
        let call_regs = self.borrow_free_gp_dynamic_regs(2)?;
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
            callee_total_frame_slots,
        )?;

        // After the precheck, reuse the second borrowed temp as
        // `caller_result_base = caller_fp + results.start_offset`. The
        // backend will hand this off to the callee through the
        // backend-private call record; the callee's `Return` copies the
        // function's `return_results` region into `*caller_result_base`.
        let caller_result_base = stack_limit;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::IntBinary {
                width: self.gp_word_int_width(),
                op: MachineIntBinaryOp::Add,
                dst: caller_result_base,
                lhs: MachineValue::Reg(self.frame_base_reg()),
                rhs: MachineValue::Imm64(slot_offset_bytes(results.start)? as u64),
            },
        });

        // Note: zero-init of the callee's non-param locals is performed by
        // the callee itself at function entry, only for slots flagged by the
        // SsaProgram's `local_slot_info.reads_before_write` analysis. Locals
        // that are guaranteed to be written before any read need no init.

        Ok(MachineTerminator::CallDirect {
            callee: callee_id,
            callee_frame_base,
            caller_result_base,
            continuation,
        })
    }

    pub(super) fn lower_call_external(
        &mut self,
        func_idx: u32,
        args: FrameSpan,
        results: FrameSpan,
        const_pool: &mut ConstPoolBuilder,
    ) -> Result<(), WasmError> {
        self.ensure_no_live_values(
            "prepared SSA-IR external call reached native lowering with live linear SSA values; values must be published before the call",
        )?;

        // External calls are modeled as inline runtime-entry calls rather than MIR
        // terminators. Unlike local calls, they do not transfer control to
        // another compiled MachineIR function or create a MachineIR
        // continuation edge; the runtime external-call entry returns inline to
        // this block.
        //
        // The surrounding MIR still makes the call boundary explicit by
        // emitting cached-local save/reload and mem0 cache refresh around the
        // external-call instruction sequence.
        let metadata = self.build_call_external_meta_direct(func_idx, args, results, const_pool);
        self.emit_call_external(metadata)
    }

    pub(super) fn build_call_external_meta_direct(
        &self,
        func_idx: u32,
        args: FrameSpan,
        results: FrameSpan,
        const_pool: &mut ConstPoolBuilder,
    ) -> MachineConstId {
        // Direct external calls carry the Wasm function index as an immediate
        // in the metadata record. The runtime entry will use it directly.
        const_pool.call_external_meta(ExternalCallMeta {
            func_idx_source: func_idx,
            func_idx_source_kind: ExternalCallTargetKind::Immediate as u32,
            args: external_call_region(args),
            results: external_call_region(results),
        })
    }

    pub(super) fn build_call_external_meta_indirect(
        &self,
        func_idx_slot: FrameSlot,
        args: FrameSpan,
        results: FrameSpan,
        const_pool: &mut ConstPoolBuilder,
    ) -> MachineConstId {
        // Indirect external calls resolve the target at runtime. MachineIR
        // publishes the frame slot that now holds the resolved function index,
        // and the runtime entry reloads it from there.
        const_pool.call_external_meta(ExternalCallMeta {
            func_idx_source: func_idx_slot.0 as u32,
            func_idx_source_kind: ExternalCallTargetKind::FrameSlot as u32,
            args: external_call_region(args),
            results: external_call_region(results),
        })
    }

    pub(super) fn build_external_call_ops(
        &self,
        metadata: MachineConstId,
    ) -> collections::Vec<MachineInst> {
        // The external-call instruction itself is the foreign-call boundary.
        // The reloads that follow repair machine-visible runtime state that may
        // have changed while the host callback executed, most importantly the
        // cached mem0 base/size pair after a possible memory growth.
        collections::vec![
            MachineInst {
                kind: MachineInstKind::CallExternal(MachineCallExternal { metadata }),
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

    fn emit_call_external(&mut self, metadata: MachineConstId) -> Result<(), WasmError> {
        // External calls are frame-slot based and return inline in the same
        // MIR block. Because the runtime call may clobber every machine
        // register, cached locals must be published to their canonical frame
        // slots before the call. Re-caching after the call is explicit in
        // SSA-IR.
        self.emit_save_dirty_cached_locals()?;
        self.emit_machine_ops(self.build_external_call_ops(metadata));
        Ok(())
    }

    fn emit_direct_call_stack_precheck(
        &mut self,
        callee_frame_base: MachineReg,
        stack_limit: MachineReg,
        callee_total_frame_slots: u16,
    ) -> Result<(), WasmError> {
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

    pub(super) fn emit_zero_canonical_slot_at_addr(
        &mut self,
        base: MachineReg,
    ) -> Result<(), WasmError> {
        if self.gp_reg_width() == 4 {
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Store {
                    ty: MachineStorageType::GpWord,
                    addr: MachineAddr { base, offset: 0 },
                    width: MachineMemWidth::U32,
                    src: MachineValue::Imm64(0),
                },
            });
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Store {
                    ty: MachineStorageType::GpWord,
                    addr: MachineAddr { base, offset: 4 },
                    width: MachineMemWidth::U32,
                    src: MachineValue::Imm64(0),
                },
            });
        } else {
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Store {
                    ty: MachineStorageType::GpI64,
                    addr: MachineAddr { base, offset: 0 },
                    width: MachineMemWidth::U64,
                    src: MachineValue::Imm64(0),
                },
            });
        }
        Ok(())
    }
}

#[inline]
fn external_call_region(span: FrameSpan) -> ExternalCallFrameRegion {
    // External calls use frame-relative regions all the way down to the runtime
    // entry, so the Wasm frame span lowers directly to the serialized ABI
    // record without any extra scratch slot or repacking.
    ExternalCallFrameRegion {
        base_slot: span.start.0,
        slots: span.count,
    }
}
