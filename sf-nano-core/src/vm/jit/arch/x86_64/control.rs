//! x86_64 backend: control flow lowering methods for X86_64Backend.

use crate::{
    error::WasmError,
    vm::jit::machine::machine_ir::{
        MachineBlockId, MachineBranchCond, MachineCallArgs, MachineCallResults, MachineCallTarget,
        MachineCompareKind, MachineConstId, MachineEdge, MachineFloatWidth, MachineIntWidth,
        MachineResultDst, MachineResultSrc, MachineReturnValue, MachineStorageType,
        MachineTerminator, MachineTrapKind, MachineValue, MACHINE_CTX_REG, MACHINE_FP_REG,
    },
};

use super::{
    abi::{map_fixed_reg, C_ARG0, C_ARG1, C_ARG2, W2W_FP_RET0, W2W_GP_RET0},
    backend::{PendingJumpTable, X86_64Backend},
    enc::{self, Cc},
    fusion::map_int_cond,
    reg::X86Reg,
};

use crate::collections;
use crate::vm::jit::arch::common::helpers::is_fallthrough_edge;
use crate::vm::jit::arch::common::helpers::trap_code;
use crate::vm::jit::arch::common::pipeline::{emit_call_arg_lanes, emit_parallel_moves};
use crate::vm::jit::arch::common::template::{
    decode_template_chain_next, encode_template_chain_next, TemplateBranchSense,
};
use crate::vm::jit::arch::common::types::DirectCallPatch;
use crate::vm::jit::runtime::{runtime_call::call_runtime_entry_ptr, trap::raise_trap};

fn scalar_fp_width(ty: MachineStorageType) -> Result<MachineFloatWidth, WasmError> {
    ty.float_width()
        .ok_or_else(|| WasmError::internal("x86_64 scalar FP return uses non-FP storage"))
}

impl<'a> X86_64Backend<'a> {
    /// Only equality consumes the same ZF as the i32 producer. Signed and
    /// unsigned ordering still need CMP, as does every i64 comparison.
    fn lower_branch_compare(
        &mut self,
        width: MachineIntWidth,
        kind: MachineCompareKind,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        if width == MachineIntWidth::I32
            && matches!(kind, MachineCompareKind::Eq | MachineCompareKind::Ne)
        {
            let reg = match (lhs, rhs) {
                (MachineValue::Reg(reg), MachineValue::Imm64(0))
                | (MachineValue::Imm64(0), MachineValue::Reg(reg)) => Some(reg),
                _ => None,
            };
            if let Some(reg) = reg {
                if self.flags32_current(self.map_gp_reg(reg)?) {
                    return Ok(());
                }
            }
        }
        self.lower_cmp_values(width, lhs, rhs)
    }

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
                // An unconditional edge's moves run in every execution, so
                // emit them inline and jump straight to the target instead
                // of round-tripping through an out-of-line stub — a loop
                // latch through a stub costs a second taken jump per
                // iteration. Conditional edges keep stubs: their moves are
                // taken-path-only.
                self.emit_jump_edge_inline(edge, fallthrough)
            }
            MachineTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            } => self.lower_branch(cond, then_edge, else_edge, fallthrough),
            MachineTerminator::Return => self.lower_return_sequence(None),
            MachineTerminator::ReturnScalar { value } => self.lower_return_sequence(Some(value)),
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

    /// Emit an unconditional jump edge's parallel moves inline, then a
    /// single jump to the target (or nothing when the target is the
    /// fallthrough block). Identity moves emit no code, so this subsumes
    /// the direct-label shortcut of `emit_edge`.
    fn emit_jump_edge_inline(
        &mut self,
        edge: &MachineEdge,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        let blocks = self.core.mir_blocks()?;
        // Identity first, before any width lookup: identity edges emit no
        // moves and their FP args may legitimately carry values (v128)
        // outside the scalar width tracking. This mirrors `emit_edge`.
        if !self.core.is_identity_edge(blocks, edge.target, &edge.args) {
            let block = blocks
                .get(edge.target.as_usize())
                .ok_or_else(|| WasmError::internal("jump edge target block is out of range"))?;
            let arg_float_widths = edge
                .args
                .iter()
                .map(|arg| match arg {
                    MachineValue::Reg(reg) if self.core.is_fp_reg(*reg) => {
                        self.core.fp_reg_width(*reg).map(Some)
                    }
                    MachineValue::ReservedReg(_)
                    | MachineValue::Reg(_)
                    | MachineValue::Imm64(_) => Ok(None),
                })
                .collect::<Result<collections::Vec<_>, _>>()?;
            emit_parallel_moves::<Self>(self, &block.params, &edge.args, &arg_float_widths)?;
        }
        if fallthrough != Some(edge.target) {
            let label = self.core.block_label(edge.target)?;
            self.emit_jmp(label);
        }
        Ok(())
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
                    // upper half that may remain in a GpWord carrier. Skip
                    // the test when EFLAGS already carries this register's
                    // 32-bit result, letting the ALU op and jcc macro-fuse.
                    if !self.flags32_current(reg) {
                        enc::test_rr_32(&mut self.core.text, reg, reg);
                    }
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
                self.lower_branch_compare(width, kind, lhs, rhs)?;
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
                    // upper half that may remain in a GpWord carrier. Skip
                    // the test when EFLAGS already carries this register's
                    // 32-bit result.
                    if !self.flags32_current(reg) {
                        enc::test_rr_32(&mut self.core.text, reg, reg);
                    }
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
                self.lower_branch_compare(width, kind, lhs, rhs)?;
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
                    // Same flags reuse as lower_branch: skip the test when
                    // EFLAGS already carries this register's 32-bit result.
                    if !self.flags32_current(reg) {
                        enc::test_rr_32(&mut self.core.text, reg, reg);
                    }
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
                self.lower_branch_compare(width, kind, lhs, rhs)?;
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

    /// Unified Return lowering, mirroring the arm64 reference sequence.
    /// Scalar returns travel in the wasm-to-wasm return lanes; frame-
    /// fallback results were already canonicalized into the callee's
    /// fixed fallback slots `[0, result_count)` by return canonical-
    /// ization, and the *caller* copies them out after the call. The
    /// callee undoes its body frame, sets `C_RET0 = 0`, and `ret`s with
    /// `MACHINE_FP_REG` still pointing at its own frame — the caller
    /// restores its frame pointer by delta after the call returns.
    ///
    /// The matching error path is `body_local_error_label` (see
    /// `lower_body_local_error_tail` in `x86_64/backend.rs`).
    fn lower_return_sequence(
        &mut self,
        value: Option<&MachineReturnValue>,
    ) -> Result<(), WasmError> {
        if let Some(value) = value {
            self.lower_return_value_to_lanes(value)?;
        }
        // Undo the body frame (shim + preserved saves); [rsp+0] is then
        // the return address.
        self.lower_body_frame_undo();

        // Success status: C_RET0 = 0. (xor eax, eax)
        enc::xor_rr_32(&mut self.core.text, super::abi::C_RET0, super::abi::C_RET0);
        enc::ret(&mut self.core.text);
        Ok(())
    }

    fn lower_return_value_to_lanes(&mut self, value: &MachineReturnValue) -> Result<(), WasmError> {
        match value {
            MachineReturnValue::ScalarGp { src, .. } => {
                self.move_gp_result_src_to_reg(src, W2W_GP_RET0)
            }
            MachineReturnValue::ScalarFp { src, ty } => {
                self.move_fp_result_src_to_reg(src, W2W_FP_RET0, scalar_fp_width(*ty)?)
            }
            MachineReturnValue::ScalarGpPair { .. } => Err(WasmError::internal(
                "x86_64 cannot lower 32-bit GP pair scalar return".into(),
            )),
        }
    }

    fn move_gp_result_src_to_reg(
        &mut self,
        src: &MachineResultSrc,
        dst: X86Reg,
    ) -> Result<(), WasmError> {
        let fp = map_fixed_reg(MACHINE_FP_REG);
        match *src {
            MachineResultSrc::Reg(reg) => {
                let src = self.map_gp_reg(reg)?;
                if src != dst {
                    enc::mov_rr_64(&mut self.core.text, dst, src);
                }
            }
            MachineResultSrc::FrameSlot(slot) => {
                enc::load_64(&mut self.core.text, dst, fp, i32::from(slot.0) * 8);
            }
            MachineResultSrc::FrameSlotOffset { slot, byte_offset } => {
                enc::load_64(
                    &mut self.core.text,
                    dst,
                    fp,
                    i32::from(slot.0) * 8 + i32::from(byte_offset),
                );
            }
        }
        Ok(())
    }

    fn move_fp_result_src_to_reg(
        &mut self,
        src: &MachineResultSrc,
        dst: u32,
        width: MachineFloatWidth,
    ) -> Result<(), WasmError> {
        let fp = map_fixed_reg(MACHINE_FP_REG);
        let dst = dst as enc::Xmm;
        match *src {
            MachineResultSrc::Reg(reg) => {
                let src = self.map_fp_reg(reg)? as enc::Xmm;
                if src != dst {
                    enc::movaps_rr(&mut self.core.text, dst, src);
                }
            }
            MachineResultSrc::FrameSlot(slot) => {
                let offset = i32::from(slot.0) * 8;
                match width {
                    MachineFloatWidth::F32 => enc::movss_load(&mut self.core.text, dst, fp, offset),
                    MachineFloatWidth::F64 => enc::movsd_load(&mut self.core.text, dst, fp, offset),
                }
            }
            MachineResultSrc::FrameSlotOffset { slot, byte_offset } => {
                let offset = i32::from(slot.0) * 8 + i32::from(byte_offset);
                match width {
                    MachineFloatWidth::F32 => enc::movss_load(&mut self.core.text, dst, fp, offset),
                    MachineFloatWidth::F64 => enc::movsd_load(&mut self.core.text, dst, fp, offset),
                }
            }
        }
        Ok(())
    }

    // ── Compiled call ────────────────────────────────────────────────────────

    /// Lower a compiled SF->SF call, mirroring the arm64 reference
    /// sequence: caller-side frame-pointer switch by delta, a native near
    /// call, caller-side frame-pointer restore, status check, and
    /// caller-side result placement. No call record is pushed.
    fn lower_call(
        &mut self,
        target: &MachineCallTarget,
        frame_delta: u32,
        args: &MachineCallArgs,
        results: &MachineCallResults,
        success: &MachineEdge,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        let body_local_error_label = self.core.body_local_error_label;
        let continuation_label = self.core.block_label(success.target)?;
        let continuation_is_fallthrough = fallthrough == Some(success.target);

        emit_call_arg_lanes::<Self>(self, args)?;

        // Switch MACHINE_FP_REG to the callee frame. The callee returns
        // with fp still pointing at its own frame; the caller restores it
        // below before observing status or fallback results.
        self.adjust_frame_pointer_by_delta(frame_delta, true)?;

        match target {
            MachineCallTarget::Direct(callee) => {
                // Near call; the rel32 displacement is patched at link
                // time once every callee address is known.
                let rel32_offset = enc::call_rel32(&mut self.core.text);
                self.core
                    .direct_call_patches
                    .push(DirectCallPatch::x64_rel32(rel32_offset, *callee));
            }
            MachineCallTarget::Indirect { callee_entry, .. } => {
                let callee_entry = self.map_gp_reg(*callee_entry)?;
                enc::call_reg(&mut self.core.text, callee_entry);
            }
        }

        // --- callee returns here. Restore the caller frame before status
        // propagation; the caller's error tail expects its own frame.
        self.adjust_frame_pointer_by_delta(frame_delta, false)?;

        // C_RET0 holds 0 (success) or trap kind (error). Status check
        // propagates to body_local_error_label.
        enc::test_rr_64(&mut self.core.text, super::abi::C_RET0, super::abi::C_RET0);
        self.emit_jcc(Cc::NE, body_local_error_label);

        self.lower_call_result_placement(frame_delta, results)?;

        // Continuation: branch only if the next emitted block is not the
        // continuation block.
        if !continuation_is_fallthrough {
            self.emit_jmp(continuation_label);
        }
        Ok(())
    }

    fn adjust_frame_pointer_by_delta(&mut self, delta: u32, add: bool) -> Result<(), WasmError> {
        if delta == 0 {
            return Ok(());
        }
        let delta = i32::try_from(delta)
            .map_err(|_| WasmError::internal("x86_64 call frame delta exceeds i32".into()))?;
        let fp_reg = map_fixed_reg(MACHINE_FP_REG);
        let delta = if add { delta } else { -delta };
        enc::add_ri_64(&mut self.core.text, fp_reg, delta);
        Ok(())
    }

    fn lower_call_result_placement(
        &mut self,
        frame_delta: u32,
        results: &MachineCallResults,
    ) -> Result<(), WasmError> {
        match results {
            MachineCallResults::None => Ok(()),
            MachineCallResults::ScalarGp { dst, .. } => {
                self.place_gp_call_result(*dst, W2W_GP_RET0)
            }
            MachineCallResults::ScalarFp { dst, ty } => {
                self.place_fp_call_result(*dst, W2W_FP_RET0, scalar_fp_width(*ty)?)
            }
            MachineCallResults::ScalarGpPair { .. } => Err(WasmError::internal(
                "x86_64 cannot lower 32-bit GP pair scalar call result".into(),
            )),
            MachineCallResults::FrameFallback {
                callee_results,
                caller_results,
            } => {
                if callee_results.slots != caller_results.slots {
                    return Err(WasmError::internal(
                        "x86_64 frame-fallback result slot count mismatch",
                    ));
                }
                let fp_reg = map_fixed_reg(MACHINE_FP_REG);
                let temp = self.gp_scratch.scoped_alloc().detach();
                for index in 0..i32::from(callee_results.slots) {
                    let src_offset = i32::try_from(frame_delta)
                        .ok()
                        .and_then(|delta| {
                            delta.checked_add((i32::from(callee_results.base_slot) + index) * 8)
                        })
                        .ok_or_else(|| {
                            WasmError::internal("x86_64 result source offset overflow")
                        })?;
                    let dst_offset = (i32::from(caller_results.base_slot) + index) * 8;
                    if src_offset == dst_offset {
                        continue;
                    }
                    enc::load_64(&mut self.core.text, *temp, fp_reg, src_offset);
                    enc::store_64(&mut self.core.text, fp_reg, dst_offset, *temp);
                }
                Ok(())
            }
        }
    }

    fn place_gp_call_result(
        &mut self,
        dst: MachineResultDst,
        src: X86Reg,
    ) -> Result<(), WasmError> {
        match dst {
            MachineResultDst::Reg(reg) => {
                let dst = self.map_gp_reg(reg)?;
                if dst != src {
                    enc::mov_rr_64(&mut self.core.text, dst, src);
                }
            }
            MachineResultDst::FrameSlot(slot) => {
                enc::store_64(
                    &mut self.core.text,
                    map_fixed_reg(MACHINE_FP_REG),
                    i32::from(slot.0) * 8,
                    src,
                );
            }
        }
        Ok(())
    }

    fn place_fp_call_result(
        &mut self,
        dst: MachineResultDst,
        src: u32,
        width: MachineFloatWidth,
    ) -> Result<(), WasmError> {
        let src = src as enc::Xmm;
        match dst {
            MachineResultDst::Reg(reg) => {
                let dst = self.map_fp_reg(reg)? as enc::Xmm;
                if dst != src {
                    enc::movaps_rr(&mut self.core.text, dst, src);
                }
            }
            MachineResultDst::FrameSlot(slot) => {
                let fp_reg = map_fixed_reg(MACHINE_FP_REG);
                let offset = i32::from(slot.0) * 8;
                match width {
                    MachineFloatWidth::F32 => {
                        enc::movss_store(&mut self.core.text, fp_reg, offset, src)
                    }
                    MachineFloatWidth::F64 => {
                        enc::movsd_store(&mut self.core.text, fp_reg, offset, src)
                    }
                }
            }
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

        match target {
            MachineCallTarget::Direct(callee) => {
                // Undo the body frame so the callee sees the entry-state
                // stack (its own prelude re-establishes shim and saves),
                // then enter via a patched near jump.
                self.lower_body_frame_undo();
                let rel32_offset = enc::jmp_rel32(&mut self.core.text);
                self.core
                    .direct_call_patches
                    .push(DirectCallPatch::x64_rel32(rel32_offset, *callee));
            }
            MachineCallTarget::Indirect { callee_entry, .. } => {
                // Materialize the callee entry into a backend-owned scratch
                // BEFORE undoing the body frame: the undo pops preserved
                // lanes, and an indirect entry value could live in one of
                // them. Pool scratches are never popped.
                let scratch_idx = self.gp_scratch.alloc();
                let scratch = self.gp_scratch.reg(scratch_idx);
                let callee_entry = self.map_gp_reg(*callee_entry)?;
                if callee_entry != scratch {
                    enc::mov_rr_64(&mut self.core.text, scratch, callee_entry);
                }
                self.lower_body_frame_undo();
                enc::jmp_reg(&mut self.core.text, scratch);
                self.gp_scratch.free_index(scratch_idx);
            }
        }

        // Tail-call arguments have already been repacked to the current frame
        // prefix, so the callee reuses the current MACHINE_FP_REG value.
        let _ = fp_reg;
        Ok(())
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

        // Load table base address (absolute, patched later), then dispatch
        // through one scaled memory-indirect jump: the SIB scale replaces
        // the shift/add/load chain on the indirect-branch critical path.
        enc::movabs_ri_64(&mut self.core.text, *table_scratch, 0);
        let table_base_imm_offset = self.core.text.len() - 8;
        enc::jmp_mem_index_scale8(&mut self.core.text, *table_scratch, *index_scratch);

        // The entry words flush after the function body (next to the FP
        // literal pool) so ~8 bytes per target of data never sit in the
        // instruction stream between this dispatch and its handlers.
        let entry_labels = entries
            .iter()
            .map(|entry| self.core.emit_edge(entry.target, &entry.args))
            .collect::<Result<collections::Vec<_>, _>>()?;
        self.pending_jump_tables.push(PendingJumpTable {
            base_imm_offset: table_base_imm_offset,
            entry_labels,
        });
        Ok(())
    }
}
