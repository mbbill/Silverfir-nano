//! x86_64 backend: control flow lowering methods for X86_64Backend.

use crate::{
    error::WasmError,
    vm::jit::machine::machine_ir::{
        MachineBlockId, MachineBranchCond, MachineCallArgs, MachineCallResults, MachineCallTarget,
        MachineCompareKind, MachineConstId, MachineEdge, MachineTerminator, MachineTrapKind,
        MachineValue, MACHINE_CTX_REG, MACHINE_FP_REG,
    },
};

use super::{
    abi::{map_fixed_reg, C_ARG0, C_ARG1, C_ARG2},
    backend::X86_64Backend,
    enc::{self, Cc},
    fusion::map_int_cond,
    reg::X86Reg,
};

use crate::vm::jit::arch::common::helpers::is_fallthrough_edge;
use crate::vm::jit::arch::common::helpers::trap_code;
use crate::vm::jit::arch::common::pipeline::emit_call_arg_lanes;
use crate::vm::jit::arch::common::template::{
    decode_template_chain_next, encode_template_chain_next, TemplateBranchSense,
};
use crate::vm::jit::arch::common::types::{DirectCallPatch, LocalPtrPatch, PendingLocalPtrPatch};
use crate::vm::runtime::{runtime_call::call_runtime_entry_ptr, trap::raise_trap};

fn caller_results_base_delta(results: &MachineCallResults) -> u32 {
    match results {
        MachineCallResults::FrameFallback { caller_results, .. } => {
            u32::from(caller_results.base_slot) * 8
        }
        MachineCallResults::None
        | MachineCallResults::ScalarGp { .. }
        | MachineCallResults::ScalarGpPair { .. }
        | MachineCallResults::ScalarFp { .. } => 0,
    }
}

impl<'a> X86_64Backend<'a> {
    // ── Main terminator dispatch ─────────────────────────────────────────────

    pub(super) fn lower_terminator_dispatch(
        &mut self,
        term: &MachineTerminator,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        match term {
            MachineTerminator::Jump(edge) => {
                if is_fallthrough_edge(
                    edge.target,
                    &edge.args,
                    fallthrough,
                    self.core.mir_blocks()?,
                ) {
                    return Ok(());
                }
                let label = self.core.emit_edge(edge.target, &edge.args)?;
                self.emit_jmp(label);
                Ok(())
            }
            MachineTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            } => self.lower_branch(cond, then_edge, else_edge, fallthrough),
            MachineTerminator::Return | MachineTerminator::ReturnScalar { .. } => {
                self.lower_return_sequence()
            }
            MachineTerminator::Trap { kind } => {
                let trap_label = self.core.ensure_trap_label(*kind);
                self.emit_jmp(trap_label);
                Ok(())
            }
            MachineTerminator::JumpTable { index, entries } => {
                self.lower_jump_table(*index, entries)
            }
            MachineTerminator::Call {
                target,
                frame_delta,
                args,
                results,
                success,
            } => self.lower_call(target, *frame_delta, args, results, success, fallthrough),
            MachineTerminator::TailCall { target, args } => self.lower_tail_call(target, args),
        }
    }

    // ── Branch / branch_if ─────────────────────────────────────────────────

    fn lower_branch(
        &mut self,
        cond: &MachineBranchCond,
        then_edge: &MachineEdge,
        else_edge: &MachineEdge,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        let blocks = self.core.mir_blocks()?;
        let then_fallthrough =
            is_fallthrough_edge(then_edge.target, &then_edge.args, fallthrough, blocks);
        let else_fallthrough =
            is_fallthrough_edge(else_edge.target, &else_edge.args, fallthrough, blocks);
        let then_label = (!then_fallthrough)
            .then(|| self.core.emit_edge(then_edge.target, &then_edge.args))
            .transpose()?;
        let else_label = (!else_fallthrough)
            .then(|| self.core.emit_edge(else_edge.target, &else_edge.args))
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
                    // Wasm branch conditions are i32 values; ignore any stale
                    // upper half that may remain in a GpWord carrier.
                    enc::test_rr_32(&mut self.core.text, reg, reg);
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
                MachineValue::ReservedReg(_reg) => {
                    return Err(WasmError::internal(
                        "x86_64 branch condition cannot read reserved cache register",
                    ));
                }
            },
            MachineBranchCond::IntCompare {
                width,
                kind,
                sign,
                lhs,
                rhs,
            } => {
                self.lower_cmp_values(width, lhs, rhs)?;
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
            MachineBranchCond::TestBits {
                width,
                kind,
                src,
                mask,
            } => {
                self.lower_tst_values(width, src, mask)?;
                let cc = match kind {
                    MachineCompareKind::Eq => Cc::E,
                    MachineCompareKind::Ne => Cc::NE,
                    _ => {
                        return Err(WasmError::internal(
                            "TestBits branch: unsupported compare kind",
                        ))
                    }
                };
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
        }
        Ok(())
    }

    pub(super) fn lower_branch_if(
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
                    // Wasm branch conditions are i32 values; ignore any stale
                    // upper half that may remain in a GpWord carrier.
                    enc::test_rr_32(&mut self.core.text, reg, reg);
                    self.emit_jcc(Cc::NE, trap_label);
                }
                MachineValue::ReservedReg(_reg) => {
                    return Err(WasmError::internal(
                        "x86_64 trap-if cannot read reserved cache register",
                    ));
                }
            },
            MachineBranchCond::IntCompare {
                width,
                kind,
                sign,
                lhs,
                rhs,
            } => {
                self.lower_cmp_values(width, lhs, rhs)?;
                self.emit_jcc(map_int_cond(kind, sign), trap_label);
            }
            MachineBranchCond::TestBits {
                width,
                kind,
                src,
                mask,
            } => {
                self.lower_tst_values(width, src, mask)?;
                let cc = match kind {
                    MachineCompareKind::Eq => Cc::E,
                    MachineCompareKind::Ne => Cc::NE,
                    _ => {
                        return Err(WasmError::internal(
                            "TestBits branch_if: unsupported compare kind",
                        ))
                    }
                };
                self.emit_jcc(cc, trap_label);
            }
        }
        Ok(())
    }

    pub(crate) fn emit_template_skip_unless(
        &mut self,
        cond: &MachineBranchCond,
        jump_when: TemplateBranchSense,
        skip_bytes: usize,
    ) -> Result<(), WasmError> {
        match *cond {
            MachineBranchCond::Value(value) => match value {
                MachineValue::Imm64(value) => {
                    let cond_true = value != 0;
                    let jump_on_true = jump_when == TemplateBranchSense::IfTrue;
                    if cond_true != jump_on_true {
                        self.emit_template_skip_jump(skip_bytes)?;
                    }
                }
                MachineValue::Reg(reg) => {
                    let reg = self.map_gp_reg(reg)?;
                    enc::test_rr_32(&mut self.core.text, reg, reg);
                    let cc = match jump_when {
                        TemplateBranchSense::IfTrue => Cc::E,
                        TemplateBranchSense::IfFalse => Cc::NE,
                    };
                    self.emit_template_skip_jcc(cc, skip_bytes)?;
                }
                MachineValue::ReservedReg(_) => {
                    return Err(WasmError::internal(
                        "x86_64 template branch cannot read reserved cache register",
                    ));
                }
            },
            MachineBranchCond::IntCompare {
                width,
                kind,
                sign,
                lhs,
                rhs,
            } => {
                self.lower_cmp_values(width, lhs, rhs)?;
                let cc = match jump_when {
                    TemplateBranchSense::IfTrue => map_int_cond(kind, sign).invert(),
                    TemplateBranchSense::IfFalse => map_int_cond(kind, sign),
                };
                self.emit_template_skip_jcc(cc, skip_bytes)?;
            }
            MachineBranchCond::TestBits {
                width,
                kind,
                src,
                mask,
            } => {
                self.lower_tst_values(width, src, mask)?;
                let cc = match kind {
                    MachineCompareKind::Eq => Cc::E,
                    MachineCompareKind::Ne => Cc::NE,
                    _ => {
                        return Err(WasmError::internal(
                            "x86_64 template TestBits branch uses unsupported compare kind",
                        ))
                    }
                };
                let cc = match jump_when {
                    TemplateBranchSense::IfTrue => cc.invert(),
                    TemplateBranchSense::IfFalse => cc,
                };
                self.emit_template_skip_jcc(cc, skip_bytes)?;
            }
        }
        Ok(())
    }

    fn emit_template_skip_jump(&mut self, skip_bytes: usize) -> Result<(), WasmError> {
        let rel32_offset = enc::jmp_rel32(&mut self.core.text);
        let target = self
            .core
            .text
            .len()
            .checked_add(skip_bytes)
            .ok_or_else(|| WasmError::internal("x86_64 template skip offset overflow"))?;
        enc::patch_rel32(&mut self.core.text, rel32_offset, target);
        Ok(())
    }

    fn emit_template_skip_jcc(&mut self, cc: Cc, skip_bytes: usize) -> Result<(), WasmError> {
        let rel32_offset = enc::jcc_rel32(&mut self.core.text, cc);
        let target = self
            .core
            .text
            .len()
            .checked_add(skip_bytes)
            .ok_or_else(|| WasmError::internal("x86_64 template skip offset overflow"))?;
        enc::patch_rel32(&mut self.core.text, rel32_offset, target);
        Ok(())
    }

    pub(crate) fn emit_template_jump_placeholder(
        &mut self,
        next: usize,
    ) -> Result<usize, WasmError> {
        let site = self.core.text.emit_u8(0xE9);
        self.core.text.emit_u32(encode_template_chain_next(next)?);
        Ok(site)
    }

    pub(crate) fn read_template_jump_next(&self, site: usize) -> Result<usize, WasmError> {
        Ok(decode_template_chain_next(
            self.core.text.read_u32(site + 1),
        ))
    }

    pub(crate) fn patch_template_jump(
        &mut self,
        site: usize,
        target: usize,
    ) -> Result<(), WasmError> {
        enc::patch_rel32(&mut self.core.text, site + 1, target);
        Ok(())
    }

    pub(crate) fn emit_template_jump_to_offset(&mut self, target: usize) -> Result<(), WasmError> {
        let rel32_offset = enc::jmp_rel32(&mut self.core.text);
        enc::patch_rel32(&mut self.core.text, rel32_offset, target);
        Ok(())
    }

    // ── Trap stub ────────────────────────────────────────────────────────────

    /// Emit a trap stub: set up `raise_trap(ctx, trap_code)`, call it, then
    /// branch to `body_local_error_label`. After `raise_trap` returns,
    /// `C_RET0` (RAX) already holds `NativeCallStatus::Error`, so the
    /// propagation status flows unchanged through the unified Return tail.
    pub(super) fn lower_trap_dispatch(&mut self, kind: MachineTrapKind) {
        enc::mov_rr_64(&mut self.core.text, C_ARG0, map_fixed_reg(MACHINE_CTX_REG));
        self.materialize_u64(C_ARG1, trap_code(kind));
        let call_scratch = self.gp_scratch.claim_rax().detach();
        self.materialize_u64(*call_scratch, raise_trap as *const () as usize as u64);
        enc::call_reg(&mut self.core.text, *call_scratch);
        // JMP body_local_error_label — preserves RAX (the trap kind).
        let body_local_error_label = self.core.body_local_error_label;
        self.emit_jmp(body_local_error_label);
    }

    // ── Return sequence ──────────────────────────────────────────────────────

    /// Unified Return lowering. Undoes the body prelude alignment shim,
    /// pops the caller's call record from the host stack, copies the
    /// function's `return_results` region into `*caller_result_base`,
    /// restores `MACHINE_FP_REG` to the caller's frame pointer, sets
    /// `C_RET0 = 0` (the success status), and executes a native `ret`.
    ///
    /// Stack layout at entry to this sequence (after the body prelude's
    /// `sub rsp, 8`):
    ///
    /// ```text
    ///   [rsp +  0] = alignment shim (body prelude)
    ///   [rsp +  8] = return address (pushed by `call internal_entry`)
    ///   [rsp + 16] = caller_result_base  (low slot of call record)
    ///   [rsp + 24] = caller_fp           (high slot of call record)
    /// ```
    ///
    /// The matching error path is `body_local_error_label` (see
    /// `lower_body_local_error_tail` in `x86_64/backend.rs`).
    fn lower_return_sequence(&mut self) -> Result<(), WasmError> {
        let runtime = self.core.runtime_for(self.core.func_id)?.clone();
        let fp = map_fixed_reg(MACHINE_FP_REG);
        let result_base = self.gp_scratch.scoped_alloc().detach();
        let temp = self.gp_scratch.scoped_alloc().detach();
        // 1. Undo the body prelude alignment shim. After this, [rsp+0] =
        //    return address, [rsp+8] = caller_result_base, [rsp+16] = caller_fp.
        enc::add_rsp_imm8(&mut self.core.text, 8);

        // 2. Load caller_result_base into a scratch register held across
        //    the copy loop.
        enc::load_64(&mut self.core.text, *result_base, X86Reg::RSP, 8);

        // 3. Copy each return slot from the callee frame to *result_base.
        if let Some(results) = runtime.return_results {
            for index in 0..results.slots as i32 {
                enc::load_64(
                    &mut self.core.text,
                    *temp,
                    fp,
                    (results.base_slot as i32 + index) * 8,
                );
                enc::store_64(&mut self.core.text, *result_base, index * 8, *temp);
            }
        }

        // 4. Load caller_fp into fp_reg (MACHINE_FP_REG).
        enc::load_64(&mut self.core.text, fp, X86Reg::RSP, 16);

        // 5. Success status: C_RET0 = 0. (xor eax, eax)
        enc::xor_rr_32(&mut self.core.text, super::abi::C_RET0, super::abi::C_RET0);

        // 6. Native return. `ret 16` pops the return address and then
        //    releases the 16-byte call record sitting just above it.
        enc::ret_imm16(&mut self.core.text, 16);
        Ok(())
    }

    // ── Compiled call ────────────────────────────────────────────────────────

    /// Lower a compiled SF->SF call. The only target-specific part is how the
    /// callee entry address is materialized; the call-record protocol,
    /// frame-pointer switch, post-call status check, and continuation handling
    /// are shared.
    fn lower_call(
        &mut self,
        target: &MachineCallTarget,
        frame_delta: u32,
        args: &MachineCallArgs,
        results: &MachineCallResults,
        success: &MachineEdge,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        let fp_reg = map_fixed_reg(MACHINE_FP_REG);
        let body_local_error_label = self.core.body_local_error_label;
        let continuation_label = self.core.block_label(success.target)?;
        let continuation_is_fallthrough = fallthrough == Some(success.target);

        emit_call_arg_lanes::<Self>(self, args)?;

        let callee_fp_idx = self.gp_scratch.alloc();
        let callee_fp = self.gp_scratch.reg(callee_fp_idx);
        self.materialize_frame_addr(callee_fp, fp_reg, frame_delta);

        let caller_result_base_idx = self.gp_scratch.alloc();
        let caller_result_base = self.gp_scratch.reg(caller_result_base_idx);
        self.materialize_frame_addr(
            caller_result_base,
            fp_reg,
            caller_results_base_delta(results),
        );

        // Push the call record: caller_fp at higher slot, caller_result_base
        // at lower slot. Matches the layout consumed by the body's unified
        // Return and body_local_error_tail.
        enc::push(&mut self.core.text, fp_reg);
        enc::push(&mut self.core.text, caller_result_base);
        self.gp_scratch.free_index(caller_result_base_idx);

        // Switch fp_reg to the callee's frame base.
        enc::mov_rr_64(&mut self.core.text, fp_reg, callee_fp);
        self.gp_scratch.free_index(callee_fp_idx);

        match target {
            MachineCallTarget::Direct(callee) => {
                let call_scratch = self.gp_scratch.claim_rax().detach();
                // Load the callee's internal entry address via patchable movabs.
                enc::movabs_ri_64(&mut self.core.text, *call_scratch, 0);
                let callee_imm_offset = self.core.text.len() - 8;
                self.core
                    .direct_call_patches
                    .push(DirectCallPatch::address_literal(callee_imm_offset, *callee));
                // CALL scratch — pushes the return address (= this call site's
                // fall-through) onto the host stack.
                enc::call_reg(&mut self.core.text, *call_scratch);
            }
            MachineCallTarget::Indirect { callee_entry, .. } => {
                let callee_entry = self.map_gp_reg(*callee_entry)?;
                enc::call_reg(&mut self.core.text, callee_entry);
            }
        }

        // --- callee returns here. C_RET0 holds 0 (success) or trap kind
        //     (error). Status check propagates to body_local_error_label.
        enc::test_rr_64(&mut self.core.text, super::abi::C_RET0, super::abi::C_RET0);
        self.emit_jcc(Cc::NE, body_local_error_label);

        // Continuation: branch only if the next emitted block is not the
        // continuation block.
        if !continuation_is_fallthrough {
            self.emit_jmp(continuation_label);
        }
        Ok(())
    }

    fn lower_tail_call(
        &mut self,
        target: &MachineCallTarget,
        args: &MachineCallArgs,
    ) -> Result<(), WasmError> {
        let fp_reg = map_fixed_reg(MACHINE_FP_REG);

        emit_call_arg_lanes::<Self>(self, args)?;

        let scratch_idx = match target {
            MachineCallTarget::Direct(callee) => {
                let scratch_idx = self.gp_scratch.alloc();
                let scratch = self.gp_scratch.reg(scratch_idx);
                enc::movabs_ri_64(&mut self.core.text, scratch, 0);
                let callee_imm_offset = self.core.text.len() - 8;
                self.core
                    .direct_call_patches
                    .push(DirectCallPatch::address_literal(callee_imm_offset, *callee));
                Some(scratch_idx)
            }
            MachineCallTarget::Indirect { .. } => None,
        };

        enc::add_rsp_imm8(&mut self.core.text, 8);
        // Tail-call arguments have already been repacked to the current frame
        // prefix, so the callee reuses the current MACHINE_FP_REG value.
        let _ = fp_reg;

        match target {
            MachineCallTarget::Direct(_) => {
                let scratch = self
                    .gp_scratch
                    .reg(scratch_idx.expect("direct tail-call scratch"));
                enc::jmp_reg(&mut self.core.text, scratch);
            }
            MachineCallTarget::Indirect { callee_entry, .. } => {
                let callee_entry = self.map_gp_reg(*callee_entry)?;
                enc::jmp_reg(&mut self.core.text, callee_entry);
            }
        }

        if let Some(scratch_idx) = scratch_idx {
            self.gp_scratch.free_index(scratch_idx);
        }
        Ok(())
    }

    fn materialize_frame_addr(&mut self, dst: X86Reg, base: X86Reg, delta: u32) {
        if dst != base {
            enc::mov_rr_64(&mut self.core.text, dst, base);
        }
        if delta != 0 {
            enc::add_ri_64(&mut self.core.text, dst, delta as i32);
        }
    }

    // ── Call runtime ─────────────────────────────────────────────────────────

    pub(super) fn lower_call_runtime_term(&mut self, const_idx: usize) -> Result<(), WasmError> {
        let metadata = self
            .core
            .compiled
            .const_ptr(MachineConstId(const_idx as u32))
            .ok_or_else(|| WasmError::internal("x86_64 runtime-call metadata is out of range"))?;
        // Runtime calls are inline runtime calls. Pass the current context,
        // the active Wasm frame pointer, and the const-pool metadata record.
        enc::mov_rr_64(&mut self.core.text, C_ARG0, map_fixed_reg(MACHINE_CTX_REG));
        enc::mov_rr_64(&mut self.core.text, C_ARG1, map_fixed_reg(MACHINE_FP_REG));
        self.materialize_u64(C_ARG2, metadata as u64);
        let call_scratch = self.gp_scratch.claim_rax().detach();
        self.materialize_u64(*call_scratch, call_runtime_entry_ptr() as usize as u64);
        enc::call_reg(&mut self.core.text, *call_scratch);

        // Nonzero helper status → propagate via body_local_error_label.
        // C_RET0 is already set by raise_trap to NativeCallStatus::Error.
        enc::test_rr_64(&mut self.core.text, super::abi::C_RET0, super::abi::C_RET0);
        let body_local_error_label = self.core.body_local_error_label;
        self.emit_jcc(Cc::NE, body_local_error_label);
        Ok(())
    }

    // ── Jump table ───────────────────────────────────────────────────────────

    fn lower_jump_table(
        &mut self,
        index: MachineValue,
        entries: &[MachineEdge],
    ) -> Result<(), WasmError> {
        if entries.is_empty() {
            return Err(WasmError::internal(
                "x86_64 MachineIR jump table requires at least one entry".into(),
            ));
        }
        if entries.len() == 1 {
            let label = self.core.emit_edge(entries[0].target, &entries[0].args)?;
            self.emit_jmp(label);
            return Ok(());
        }
        let index_scratch = self.gp_scratch.scoped_alloc().detach();
        let table_scratch = self.gp_scratch.scoped_alloc().detach();

        // `br_table` indices are Wasm i32 values. Zero-extend into a scratch
        // first so stale upper halves in a GpWord carrier cannot skew the
        // unsigned clamp and dispatch.
        match index {
            MachineValue::Imm64(value) => {
                self.materialize_u64(*index_scratch, value as u32 as u64);
            }
            MachineValue::Reg(reg) => {
                let index_reg = self.map_gp_reg(reg)?;
                enc::mov_rr_32(&mut self.core.text, *index_scratch, index_reg);
            }
            MachineValue::ReservedReg(_reg) => {
                return Err(WasmError::internal(
                    "x86_64 jump table cannot read reserved cache register",
                ));
            }
        }

        // Clamp index to (entries.len() - 1) using 32-bit unsigned compare.
        self.materialize_u64(*table_scratch, (entries.len() - 1) as u64);
        enc::cmp_rr_32(&mut self.core.text, *index_scratch, *table_scratch);
        enc::cmovcc_rr_32(&mut self.core.text, Cc::A, *index_scratch, *table_scratch);

        // Load table base address (absolute, patched later)
        enc::movabs_ri_64(&mut self.core.text, *table_scratch, 0);
        let table_base_imm_offset = self.core.text.len() - 8;

        // index * 8 for table entry
        enc::shl_imm_64(&mut self.core.text, *index_scratch, 3);
        enc::add_rr_64(&mut self.core.text, *table_scratch, *index_scratch);
        // Load target address from table
        enc::load_64(&mut self.core.text, *table_scratch, *table_scratch, 0);
        enc::jmp_reg(&mut self.core.text, *table_scratch);

        // Emit jump table entries (each is a u64 absolute address, patched later)
        let table_offset = self.core.text.len();
        self.core.resolved_ptr_patches.push(LocalPtrPatch {
            literal_offset: table_base_imm_offset,
            target_offset: table_offset,
        });

        for entry in entries {
            let label = self.core.emit_edge(entry.target, &entry.args)?;
            let literal_offset = self.core.text.emit_u64(0);
            self.core.local_ptr_patches.push(PendingLocalPtrPatch {
                literal_offset,
                target_label: label,
            });
        }
        Ok(())
    }
}
