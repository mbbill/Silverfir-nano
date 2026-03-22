//! x86_64 backend: control flow emission methods for FunctionCompiler.

use alloc::vec::Vec;

use crate::{
    error::WasmError,
    vm::machine::machine_ir::{
        MachineBlockId, MachineBlockParam, MachineCompareKind, MachineFloatWidth, MachineBranchCond,
        MachineReg, MachineValue, MACHINE_CTX_REG, MACHINE_FP_REG,
    },
};

use super::{
    abi::{map_fixed_reg, FP_SCRATCH0, FP_SCRATCH1, FP_SCRATCH2, SCRATCH0, SCRATCH1},
    compile::{
        DirectCallPatch, FunctionCompiler, LabelKind, LocalPtrPatch, PendingLocalPtrPatch,
    },
    compile_helpers::{is_fallthrough_edge, map_float_cond, map_int_cond, ParallelSource},
    enc::{self, Cc},
    reg::X86Reg,
};

use crate::vm::runtime::context::ctx_offset;

impl<'a> FunctionCompiler<'a> {
    // ── Branch / branch_if ─────────────────────────────────────────────────

    pub(super) fn emit_branch(
        &mut self,
        cond: &MachineBranchCond,
        then_edge: &crate::vm::machine::machine_ir::MachineEdge,
        else_edge: &crate::vm::machine::machine_ir::MachineEdge,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        let then_fallthrough =
            is_fallthrough_edge(self, then_edge.target, &then_edge.args, fallthrough);
        let else_fallthrough =
            is_fallthrough_edge(self, else_edge.target, &else_edge.args, fallthrough);
        let then_label = (!then_fallthrough)
            .then(|| self.emit_edge(then_edge.target, &then_edge.args))
            .transpose()?;
        let else_label = (!else_fallthrough)
            .then(|| self.emit_edge(else_edge.target, &else_edge.args))
            .transpose()?;
        match *cond {
            MachineBranchCond::Value(value) => match value {
                MachineValue::Imm64(0) => {
                    if let Some(label) = else_label {
                        self.emit_jmp(label);
                    }
                }
                MachineValue::Imm64(_) => {
                    if let Some(label) = then_label {
                        self.emit_jmp(label);
                    }
                }
                MachineValue::Reg(reg) => {
                    let reg = self.map_gp_reg(reg)?;
                    enc::test_rr_64(&mut self.text, reg, reg);
                    if else_fallthrough {
                        if let Some(label) = then_label {
                            self.emit_jcc(Cc::NE, label);
                        }
                    } else if then_fallthrough {
                        if let Some(label) = else_label {
                            self.emit_jcc(Cc::E, label);
                        }
                    } else if let (Some(then_label), Some(else_label)) = (then_label, else_label) {
                        self.emit_jcc(Cc::NE, then_label);
                        self.emit_jmp(else_label);
                    }
                }
            },
            MachineBranchCond::IntCompare {
                width,
                kind,
                sign,
                lhs,
                rhs,
            } => {
                self.emit_cmp_values(width, lhs, rhs)?;
                let cc = map_int_cond(kind, sign);
                if else_fallthrough {
                    if let Some(label) = then_label {
                        self.emit_jcc(cc, label);
                    }
                } else if then_fallthrough {
                    if let Some(label) = else_label {
                        self.emit_jcc(cc.invert(), label);
                    }
                } else if let (Some(then_label), Some(else_label)) = (then_label, else_label) {
                    self.emit_jcc(cc, then_label);
                    self.emit_jmp(else_label);
                }
            }
            MachineBranchCond::FloatCompare {
                width,
                kind,
                lhs,
                rhs,
            } => {
                return self.emit_float_branch(
                    width,
                    kind,
                    lhs,
                    rhs,
                    then_label,
                    else_label,
                    then_fallthrough,
                    else_fallthrough,
                );
            }
        }
        Ok(())
    }

    pub(super) fn emit_branch_if(
        &mut self,
        cond: &MachineBranchCond,
        trap_label: usize,
    ) -> Result<(), WasmError> {
        match *cond {
            MachineBranchCond::Value(value) => match value {
                MachineValue::Imm64(0) => {}
                MachineValue::Imm64(_) => self.emit_jmp(trap_label),
                MachineValue::Reg(reg) => {
                    let reg = self.map_gp_reg(reg)?;
                    enc::test_rr_64(&mut self.text, reg, reg);
                    self.emit_jcc(Cc::NE, trap_label);
                }
            },
            MachineBranchCond::IntCompare {
                width,
                kind,
                sign,
                lhs,
                rhs,
            } => {
                self.emit_cmp_values(width, lhs, rhs)?;
                self.emit_jcc(map_int_cond(kind, sign), trap_label);
            }
            MachineBranchCond::FloatCompare {
                width,
                kind,
                lhs,
                rhs,
            } => {
                let lhs_fp = self.prepare_float_operand(width, lhs, SCRATCH0, FP_SCRATCH0)?;
                let rhs_fp = self.prepare_float_operand(width, rhs, SCRATCH1, FP_SCRATCH1)?;
                match width {
                    MachineFloatWidth::F32 => {
                        enc::ucomiss(&mut self.text, lhs_fp as u8, rhs_fp as u8)
                    }
                    MachineFloatWidth::F64 => {
                        enc::ucomisd(&mut self.text, lhs_fp as u8, rhs_fp as u8)
                    }
                };
                self.emit_jcc(map_float_cond(kind), trap_label);
            }
        }
        Ok(())
    }

    fn emit_float_branch(
        &mut self,
        width: MachineFloatWidth,
        kind: MachineCompareKind,
        lhs: MachineValue,
        rhs: MachineValue,
        then_label: Option<usize>,
        else_label: Option<usize>,
        then_fallthrough: bool,
        else_fallthrough: bool,
    ) -> Result<(), WasmError> {
        let lhs_fp = self.prepare_float_operand(width, lhs, SCRATCH0, FP_SCRATCH0)?;
        // Same scratch selection as emit_float_compare: avoid clobbering live
        // FP transients in FP_SCRATCH1.
        let rhs_fp_scratch = if lhs_fp != FP_SCRATCH0 as u32 {
            FP_SCRATCH0
        } else {
            FP_SCRATCH2
        };
        if matches!(rhs, MachineValue::Imm64(0)) {
            enc::xorpd(&mut self.text, rhs_fp_scratch as u8, rhs_fp_scratch as u8);
            match width {
                MachineFloatWidth::F32 => {
                    enc::ucomiss(&mut self.text, lhs_fp as u8, rhs_fp_scratch as u8)
                }
                MachineFloatWidth::F64 => {
                    enc::ucomisd(&mut self.text, lhs_fp as u8, rhs_fp_scratch as u8)
                }
            };
        } else {
            let rhs_fp = self.prepare_float_operand(width, rhs, SCRATCH1, rhs_fp_scratch)?;
            match width {
                MachineFloatWidth::F32 => enc::ucomiss(&mut self.text, lhs_fp as u8, rhs_fp as u8),
                MachineFloatWidth::F64 => enc::ucomisd(&mut self.text, lhs_fp as u8, rhs_fp as u8),
            };
        }
        let cc = map_float_cond(kind);
        if else_fallthrough {
            if let Some(label) = then_label {
                self.emit_jcc(cc, label);
            }
        } else if then_fallthrough {
            if let Some(label) = else_label {
                self.emit_jcc(cc.invert(), label);
            }
        } else if let (Some(then_label), Some(else_label)) = (then_label, else_label) {
            self.emit_jcc(cc, then_label);
            self.emit_jmp(else_label);
        }
        Ok(())
    }

    // ── Return sequence ──────────────────────────────────────────────────────

    pub(super) fn emit_return_sequence(&mut self) -> Result<(), WasmError> {
        let runtime = *self.runtime_for(self.function.id)?;
        let call_scratch = runtime.call_scratch.ok_or_else(|| {
            WasmError::internal("x86_64 local return requires call scratch".into())
        })?;
        let call_link = self.compiled.runtime().call_link;
        let continuation_offset =
            (call_scratch.base_slot as i32) * 8 + call_link.continuation_offset as i32;
        let caller_frame_offset =
            (call_scratch.base_slot as i32) * 8 + call_link.caller_frame_offset as i32;
        let caller_result_base_offset =
            (call_scratch.base_slot as i32) * 8 + call_link.caller_result_base_offset as i32;

        let fp = map_fixed_reg(MACHINE_FP_REG);
        // Load continuation address into RDI (caller-saved, not SCRATCH0=RAX)
        enc::load_64(&mut self.text, X86Reg::RDI, fp, continuation_offset);
        // Load caller frame pointer
        enc::load_64(&mut self.text, SCRATCH1, fp, caller_frame_offset);
        // Load caller result base offset
        enc::load_64(&mut self.text, SCRATCH0, fp, caller_result_base_offset);
        // result_ptr = caller_fp + result_base
        enc::add_rr_64(&mut self.text, SCRATCH0, SCRATCH1);

        // Copy results
        if let Some(results) = runtime.return_results {
            for index in 0..results.slots as i32 {
                enc::load_64(
                    &mut self.text,
                    X86Reg::RCX,
                    fp,
                    (results.base_slot as i32 + index) * 8,
                );
                enc::store_64(&mut self.text, SCRATCH0, index * 8, X86Reg::RCX);
            }
        }

        // Restore caller FP
        enc::mov_rr_64(&mut self.text, fp, SCRATCH1);
        // Jump to continuation
        enc::jmp_reg(&mut self.text, X86Reg::RDI);
        Ok(())
    }

    // ── Call direct ──────────────────────────────────────────────────────────

    pub(super) fn emit_call_direct(
        &mut self,
        callee: crate::vm::machine::machine_ir::MachineFuncId,
        callee_frame_base: MachineReg,
        continuation: MachineBlockId,
    ) -> Result<(), WasmError> {
        let callee_runtime = self.runtime_for(callee)?;
        let call_scratch = callee_runtime.call_scratch.ok_or_else(|| {
            WasmError::internal("x86_64 direct local call requires callee call scratch".into())
        })?;
        let continuation_slot_offset = (call_scratch.base_slot as i32) * 8
            + self.compiled.runtime().call_link.continuation_offset as i32;
        let callee_fp = self.map_gp_reg(callee_frame_base)?;

        // Load continuation address via movabs (patched after label resolution).
        // mov_ri_64 emits REX.W B8+rd imm64 (2 + 8 = 10 bytes).
        enc::movabs_ri_64(&mut self.text, SCRATCH0, 0); // placeholder (patched later)
        let cont_imm_offset = self.text.len() - 8;
        let continuation_label = self.block_label(continuation)?;
        self.local_ptr_patches.push(PendingLocalPtrPatch {
            literal_offset: cont_imm_offset,
            target_label: continuation_label,
        });

        // Store continuation address into callee's call-link slot
        // (caller_frame and caller_result_base are stored by the lowered MachineIR
        // instructions before this CallDirect terminator)
        enc::store_64(
            &mut self.text,
            callee_fp,
            continuation_slot_offset,
            SCRATCH0,
        );

        // Load callee entry address via movabs (patched after compilation)
        enc::movabs_ri_64(&mut self.text, SCRATCH0, 0); // placeholder (patched later)
        let callee_imm_offset = self.text.len() - 8;
        self.direct_call_patches.push(DirectCallPatch {
            literal_offset: callee_imm_offset,
            callee,
        });

        // Set FP to callee frame
        enc::mov_rr_64(&mut self.text, map_fixed_reg(MACHINE_FP_REG), callee_fp);
        // Jump to callee
        enc::jmp_reg(&mut self.text, SCRATCH0);
        Ok(())
    }

    // ── Call indirect ────────────────────────────────────────────────────────

    pub(super) fn emit_call_indirect(
        &mut self,
        callee_target: MachineValue,
        callee_frame_base: MachineReg,
        arg_slots: u16,
        caller_result_base: u16,
        continuation: MachineBlockId,
    ) -> Result<(), WasmError> {
        // Load function table base address
        enc::movabs_ri_64(&mut self.text, SCRATCH0, 0); // placeholder (patched later)
        let table_base_offset = self.text.len() - 8;
        self.function_table_patches.push(table_base_offset);

        // Compute table entry: table_base + callee_id * 32 (sizeof X86_64FunctionInfo)
        let callee_id_reg = self.materialize_value(SCRATCH1, callee_target)?;
        if callee_id_reg != SCRATCH1 {
            enc::mov_rr_64(&mut self.text, SCRATCH1, callee_id_reg);
        }
        // callee_id * 32 = callee_id << 5
        enc::shl_imm_64(&mut self.text, SCRATCH1, 5);
        enc::add_rr_64(&mut self.text, SCRATCH0, SCRATCH1);

        // Resolve callee_fp BEFORE loading function info (which clobbers RDI/RSI/RDX/RCX).
        // Save callee_fp to R8 (caller-saved transient, not used by info loads).
        let callee_fp_orig = self.map_gp_reg(callee_frame_base)?;
        let callee_fp = X86Reg::R8;
        if callee_fp_orig != callee_fp {
            enc::mov_rr_64(&mut self.text, callee_fp, callee_fp_orig);
        }

        // Load function info fields: entry(0), total_frame_bytes(8), frame_prefix_slots(16), call_scratch_base(24)
        enc::load_64(&mut self.text, X86Reg::RDI, SCRATCH0, 0); // entry
        enc::load_64(&mut self.text, X86Reg::RSI, SCRATCH0, 8); // total_frame_bytes
        enc::load_64(&mut self.text, X86Reg::RDX, SCRATCH0, 16); // frame_prefix_slots
        enc::load_64(&mut self.text, X86Reg::RCX, SCRATCH0, 24); // call_scratch_base_slot
                                                                 // Stack overflow check: callee_fp + total_frame_bytes > stack_end?
        enc::lea_64(&mut self.text, SCRATCH0, callee_fp, 0);
        enc::add_rr_64(&mut self.text, SCRATCH0, X86Reg::RSI);
        enc::load_64(
            &mut self.text,
            SCRATCH1,
            map_fixed_reg(MACHINE_CTX_REG),
            ctx_offset::STACK_END as i32,
        );
        enc::cmp_rr_64(&mut self.text, SCRATCH0, SCRATCH1);
        self.emit_jcc(Cc::A, self.stack_overflow_label);

        // Zero callee prefix: from callee_fp + arg_slots*8 to callee_fp + frame_prefix_slots*8
        self.emit_zero_dynamic_callee_prefix(callee_fp, arg_slots)?;

        // Load continuation address
        enc::movabs_ri_64(&mut self.text, SCRATCH0, 0); // placeholder (patched later)
        let cont_imm_offset = self.text.len() - 8;
        let continuation_label = self.block_label(continuation)?;
        self.local_ptr_patches.push(PendingLocalPtrPatch {
            literal_offset: cont_imm_offset,
            target_label: continuation_label,
        });

        // call_scratch_base_slot is in RCX (in units of slots)
        // Compute call-link base: callee_fp + call_scratch_base_slot * 8
        enc::shl_imm_64(&mut self.text, X86Reg::RCX, 3);
        enc::add_rr_64(&mut self.text, X86Reg::RCX, callee_fp);

        let call_link = self.compiled.runtime().call_link;
        // Store continuation
        enc::store_64(
            &mut self.text,
            X86Reg::RCX,
            call_link.continuation_offset as i32,
            SCRATCH0,
        );
        // Store caller frame pointer
        enc::store_64(
            &mut self.text,
            X86Reg::RCX,
            call_link.caller_frame_offset as i32,
            map_fixed_reg(MACHINE_FP_REG),
        );
        // Store caller result base
        self.materialize_u64(SCRATCH1, u64::from(caller_result_base) * 8);
        enc::store_64(
            &mut self.text,
            X86Reg::RCX,
            call_link.caller_result_base_offset as i32,
            SCRATCH1,
        );

        // Set FP to callee and jump to entry
        enc::mov_rr_64(&mut self.text, map_fixed_reg(MACHINE_FP_REG), callee_fp);
        enc::jmp_reg(&mut self.text, X86Reg::RDI);
        Ok(())
    }

    fn emit_zero_dynamic_callee_prefix(
        &mut self,
        callee_fp: X86Reg,
        arg_slots: u16,
    ) -> Result<(), WasmError> {
        // Zero from callee_fp + arg_slots*8 up to callee_fp + frame_prefix_slots*8
        // frame_prefix_slots is in RDX (from function table load)
        self.materialize_u64(SCRATCH0, u64::from(arg_slots) * 8);
        enc::add_rr_64(&mut self.text, SCRATCH0, callee_fp);
        // end = callee_fp + frame_prefix_slots * 8
        enc::shl_imm_64(&mut self.text, X86Reg::RDX, 3);
        enc::add_rr_64(&mut self.text, X86Reg::RDX, callee_fp);
        // If start >= end, skip
        enc::cmp_rr_64(&mut self.text, SCRATCH0, X86Reg::RDX);
        let done = self.new_label(LabelKind::Edge);
        self.emit_jcc(Cc::AE, done);
        let loop_label = self.new_label(LabelKind::Edge);
        self.bind_label(loop_label);
        enc::store_imm32_64(&mut self.text, SCRATCH0, 0, 0);
        enc::add_ri_64(&mut self.text, SCRATCH0, 8);
        enc::cmp_rr_64(&mut self.text, SCRATCH0, X86Reg::RDX);
        self.emit_jcc(Cc::B, loop_label);
        self.bind_label(done);
        Ok(())
    }

    // ── Jump table ───────────────────────────────────────────────────────────

    pub(super) fn emit_jump_table(
        &mut self,
        index: MachineValue,
        entries: &[crate::vm::machine::machine_ir::MachineEdge],
    ) -> Result<(), WasmError> {
        if entries.is_empty() {
            return Err(WasmError::internal(
                "x86_64 MachineIR jump table requires at least one entry".into(),
            ));
        }
        if entries.len() == 1 {
            let label = self.emit_edge(entries[0].target, &entries[0].args)?;
            self.emit_jmp(label);
            return Ok(());
        }

        let index_reg = self.materialize_value(SCRATCH1, index)?;
        // Clamp index to (entries.len() - 1)
        self.materialize_u64(X86Reg::RAX, (entries.len() - 1) as u64);
        enc::cmp_rr_64(&mut self.text, index_reg, X86Reg::RAX);
        enc::cmovcc_rr_64(&mut self.text, Cc::A, SCRATCH1, X86Reg::RAX);
        if index_reg != SCRATCH1 {
            enc::mov_rr_64(&mut self.text, SCRATCH1, index_reg);
            enc::cmovcc_rr_64(&mut self.text, Cc::A, SCRATCH1, X86Reg::RAX);
        }

        // Load table base address (absolute, patched later)
        enc::movabs_ri_64(&mut self.text, SCRATCH0, 0); // placeholder (patched later)
        let table_base_imm_offset = self.text.len() - 8;

        // index * 8 for table entry
        enc::shl_imm_64(&mut self.text, SCRATCH1, 3);
        enc::add_rr_64(&mut self.text, SCRATCH0, SCRATCH1);
        // Load target address from table
        enc::load_64(&mut self.text, SCRATCH0, SCRATCH0, 0);
        enc::jmp_reg(&mut self.text, SCRATCH0);

        // Emit jump table entries (each is a u64 absolute address, patched later)
        let table_offset = self.text.len();
        self.resolved_ptr_patches.push(LocalPtrPatch {
            literal_offset: table_base_imm_offset,
            target_offset: table_offset,
        });

        for entry in entries {
            let label = self.emit_edge(entry.target, &entry.args)?;
            let literal_offset = self.text.emit_u64(0);
            self.local_ptr_patches.push(PendingLocalPtrPatch {
                literal_offset,
                target_label: label,
            });
        }
        Ok(())
    }

    // ── Parallel moves ───────────────────────────────────────────────────────

    pub(super) fn emit_parallel_moves(
        &mut self,
        params: &[MachineBlockParam],
        args: &[MachineValue],
        arg_float_widths: &[Option<MachineFloatWidth>],
    ) -> Result<(), WasmError> {
        let mut pending = Vec::new();
        for ((&dst, &arg), &float_width) in
            params.iter().zip(args.iter()).zip(arg_float_widths.iter())
        {
            let src = match arg {
                MachineValue::Reg(reg) => ParallelSource::Reg { reg, float_width },
                MachineValue::Imm64(value) => ParallelSource::Imm(value),
            };
            if matches!(src, ParallelSource::Reg { reg, .. } if reg == dst.reg) {
                continue;
            }
            pending.push((dst, src));
        }

        while !pending.is_empty() {
            let mut ready = None;
            for index in 0..pending.len() {
                let dst = pending[index].0.reg;
                let blocked = pending.iter().enumerate().any(|(other_index, (_, src))| {
                    other_index != index
                        && matches!(src, ParallelSource::Reg { reg, .. } if *reg == dst)
                });
                if !blocked {
                    ready = Some(index);
                    break;
                }
            }
            if let Some(index) = ready {
                let (dst, src) = pending.remove(index);
                self.emit_source_move(dst, src)?;
                continue;
            }

            // Cycle detected — break it with a temporary.
            let (dst, src) = pending.remove(0);
            let ParallelSource::Reg {
                reg: src_reg,
                float_width,
            } = src
            else {
                self.emit_source_move(dst, src)?;
                continue;
            };
            if dst.ty.is_fp() {
                let dst_fp = self.map_fp_reg(dst.reg)? as u8;
                let width = dst.ty.float_width().expect("FP param width");
                match width {
                    MachineFloatWidth::F32 => {
                        enc::movss_rr(&mut self.text, FP_SCRATCH2 as u8, dst_fp)
                    }
                    MachineFloatWidth::F64 => {
                        enc::movsd_rr(&mut self.text, FP_SCRATCH2 as u8, dst_fp)
                    }
                };
                self.emit_source_move(
                    dst,
                    ParallelSource::Reg {
                        reg: src_reg,
                        float_width,
                    },
                )?;
            } else {
                let dst_gp = self.map_gp_reg(dst.reg)?;
                let src_gp = self.map_gp_reg(src_reg)?;
                enc::mov_rr_64(&mut self.text, SCRATCH1, dst_gp);
                enc::mov_rr_64(&mut self.text, dst_gp, src_gp);
            }
            for (_, source) in pending.iter_mut() {
                if matches!(*source, ParallelSource::Reg { reg, .. } if reg == dst.reg) {
                    *source = if dst.ty.is_fp() {
                        ParallelSource::FpTemp(dst.ty.float_width().expect("FP temp width"))
                    } else {
                        ParallelSource::GpTemp
                    };
                }
            }
        }
        Ok(())
    }

    fn emit_source_move(
        &mut self,
        dst: MachineBlockParam,
        src: ParallelSource,
    ) -> Result<(), WasmError> {
        match src {
            ParallelSource::Reg {
                reg: src_reg,
                float_width: src_float_width,
            } => {
                if let Some(width) = dst.ty.float_width() {
                    let dst_fp = self.map_fp_reg(dst.reg)? as u8;
                    if self.is_fp_reg(src_reg) {
                        let src_fp = self.map_fp_reg(src_reg)? as u8;
                        match width {
                            MachineFloatWidth::F32 => enc::movss_rr(&mut self.text, dst_fp, src_fp),
                            MachineFloatWidth::F64 => enc::movsd_rr(&mut self.text, dst_fp, src_fp),
                        };
                    } else {
                        let src_gp = self.map_gp_reg(src_reg)?;
                        match width {
                            MachineFloatWidth::F32 => {
                                enc::movd_xmm_r32(&mut self.text, dst_fp, src_gp)
                            }
                            MachineFloatWidth::F64 => {
                                enc::movq_xmm_r64(&mut self.text, dst_fp, src_gp)
                            }
                        };
                    }
                    self.set_fp_reg_width(dst.reg, width)?;
                } else {
                    let dst_gp = self.map_gp_reg(dst.reg)?;
                    if self.is_fp_reg(src_reg) {
                        let src_fp = self.map_fp_reg(src_reg)? as u8;
                        match src_float_width.ok_or_else(|| {
                            WasmError::invalid(alloc::format!(
                                "x86_64 edge move is missing float-width metadata for machine reg {}",
                                src_reg.0
                            ))
                        })? {
                            MachineFloatWidth::F32 => {
                                enc::movd_r32_xmm(&mut self.text, dst_gp, src_fp)
                            }
                            MachineFloatWidth::F64 => {
                                enc::movq_r64_xmm(&mut self.text, dst_gp, src_fp)
                            }
                        };
                    } else {
                        let src_gp = self.map_gp_reg(src_reg)?;
                        enc::mov_rr_64(&mut self.text, dst_gp, src_gp);
                    }
                }
            }
            ParallelSource::Imm(value) => {
                if let Some(width) = dst.ty.float_width() {
                    let dst_fp = self.map_fp_reg(dst.reg)? as u8;
                    self.materialize_u64(SCRATCH0, value);
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movd_xmm_r32(&mut self.text, dst_fp, SCRATCH0)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movq_xmm_r64(&mut self.text, dst_fp, SCRATCH0)
                        }
                    };
                    self.set_fp_reg_width(dst.reg, width)?;
                } else {
                    self.materialize_u64(self.map_gp_reg(dst.reg)?, value);
                }
            }
            ParallelSource::GpTemp => {
                let dst_gp = self.map_gp_reg(dst.reg)?;
                enc::mov_rr_64(&mut self.text, dst_gp, SCRATCH1);
            }
            ParallelSource::FpTemp(width) => {
                let dst_fp = self.map_fp_reg(dst.reg)? as u8;
                match width {
                    MachineFloatWidth::F32 => {
                        enc::movss_rr(&mut self.text, dst_fp, FP_SCRATCH2 as u8)
                    }
                    MachineFloatWidth::F64 => {
                        enc::movsd_rr(&mut self.text, dst_fp, FP_SCRATCH2 as u8)
                    }
                };
                self.set_fp_reg_width(dst.reg, width)?;
            }
        }
        Ok(())
    }
}
