//! ARM64 terminator emission: branches, calls, traps, jump tables.

use crate::error::WasmError;
use crate::vm::machine::machine_ir::{
    MachineBlockId, MachineBranchCond, MachineCallTarget, MachineCompareKind, MachineConstId,
    MachineEdge, MachineReg, MachineTerminator, MachineTrapKind, MachineValue, MACHINE_CTX_REG,
    MACHINE_FP_REG,
};

use super::abi::map_fixed_reg;
use super::fusion::map_int_cond;
use super::inst::{materialize_u64_into, prepare_gp};
use super::{abi, enc};
use crate::vm::arch::common::helpers::is_fallthrough_edge;
use crate::vm::arch::common::helpers::trap_code;
use crate::vm::arch::common::types::{LocalPtrPatch, PendingLocalPtrPatch};
use crate::vm::runtime::{runtime_call::call_runtime_entry_ptr, trap::raise_trap};

impl<'a> super::backend::Arm64Backend<'a> {
    // ── Main terminator dispatch ─────────────────────────────────────────────────

    /// Main terminator dispatch -- called by `ArchBackend::emit_terminator`.
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
                    &self.core.function.program.blocks,
                ) {
                    return Ok(());
                }
                let label = self.core.emit_edge(edge.target, &edge.args)?;
                self.lower_b(label);
                Ok(())
            }
            MachineTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            } => self.lower_branch(cond, then_edge, else_edge, fallthrough),
            MachineTerminator::Return => self.lower_return_sequence(),
            MachineTerminator::Trap { kind } => {
                self.lower_trap_dispatch(*kind);
                Ok(())
            }
            MachineTerminator::JumpTable { index, entries } => {
                self.lower_jump_table(*index, entries)
            }
            MachineTerminator::Call {
                target,
                callee_frame_base,
                caller_result_base,
                continuation,
            } => self.lower_call(
                target,
                *callee_frame_base,
                *caller_result_base,
                *continuation,
                fallthrough,
            ),
            MachineTerminator::TailCall {
                target,
                callee_frame_base,
            } => self.lower_tail_call(target, *callee_frame_base),
        }
    }

    // ── Branch ───────────────────────────────────────────────────────────────────

    fn lower_branch(
        &mut self,
        cond: &MachineBranchCond,
        then_edge: &MachineEdge,
        else_edge: &MachineEdge,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        let blocks = &self.core.function.program.blocks;
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
                        self.lower_b(label);
                    }
                }
                MachineValue::Imm64(_) => {
                    if let Some(label) = then_label {
                        self.lower_b(label);
                    }
                }
                MachineValue::Reg(reg) => {
                    let reg = self.map_gp_reg(reg)?;
                    if else_fallthrough {
                        if let Some(label) = then_label {
                            self.lower_cbnz(reg, label);
                        }
                    } else if then_fallthrough {
                        if let Some(label) = else_label {
                            self.lower_cbz(reg, label);
                        }
                    } else if let (Some(then_label), Some(else_label)) = (then_label, else_label) {
                        self.lower_cbnz(reg, then_label);
                        self.lower_b(else_label);
                    }
                }
                MachineValue::ReservedReg(_reg) => {
                    return Err(WasmError::internal(
                        "arm64 branch condition cannot read reserved cache register",
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
                // Compare-with-zero patterns (e.g. wasm `i32.eqz; br_if`) become
                // single-instruction `cbz`/`cbnz`. Both lhs == 0 and rhs == 0
                // shapes appear depending on whether the wasm operand was
                // hoisted left or right. The 64-bit cbz form is correct for
                // i32 operands too because AArch64 32-bit ops zero the upper
                // 32 bits of the X register.
                let zero_reg = match (kind, lhs, rhs) {
                    (
                        MachineCompareKind::Eq | MachineCompareKind::Ne,
                        MachineValue::Reg(r),
                        MachineValue::Imm64(0),
                    )
                    | (
                        MachineCompareKind::Eq | MachineCompareKind::Ne,
                        MachineValue::Imm64(0),
                        MachineValue::Reg(r),
                    ) => Some(r),
                    _ => None,
                };
                if let Some(reg) = zero_reg {
                    let reg = self.map_gp_reg(reg)?;
                    let is_eq = matches!(kind, MachineCompareKind::Eq);
                    let emit = |this: &mut Self, take_label: usize, take_eq: bool| {
                        if take_eq {
                            this.lower_cbz(reg, take_label);
                        } else {
                            this.lower_cbnz(reg, take_label);
                        }
                    };
                    if else_fallthrough {
                        if let Some(label) = then_label {
                            emit(self, label, is_eq);
                        }
                    } else if then_fallthrough {
                        if let Some(label) = else_label {
                            emit(self, label, !is_eq);
                        }
                    } else if let (Some(then_label), Some(else_label)) = (then_label, else_label) {
                        emit(self, then_label, is_eq);
                        self.lower_b(else_label);
                    }
                } else {
                    self.lower_cmp_values(width, lhs, rhs)?;
                    if else_fallthrough {
                        if let Some(label) = then_label {
                            self.lower_b_cond(map_int_cond(kind, sign), label);
                        }
                    } else if then_fallthrough {
                        if let Some(label) = else_label {
                            self.lower_b_cond(map_int_cond(kind, sign).invert(), label);
                        }
                    } else if let (Some(then_label), Some(else_label)) = (then_label, else_label) {
                        self.lower_b_cond(map_int_cond(kind, sign), then_label);
                        self.lower_b(else_label);
                    }
                }
            }
            MachineBranchCond::TestBits {
                width,
                kind,
                src,
                mask,
            } => {
                self.lower_tst_values(width, src, mask)?;
                let cond = match kind {
                    MachineCompareKind::Eq => enc::Cond::Eq,
                    MachineCompareKind::Ne => enc::Cond::Ne,
                    _ => {
                        return Err(WasmError::internal(
                            "TestBits branch: unsupported compare kind",
                        ))
                    }
                };
                if else_fallthrough {
                    if let Some(label) = then_label {
                        self.lower_b_cond(cond, label);
                    }
                } else if then_fallthrough {
                    if let Some(label) = else_label {
                        self.lower_b_cond(cond.invert(), label);
                    }
                } else if let (Some(then_label), Some(else_label)) = (then_label, else_label) {
                    self.lower_b_cond(cond, then_label);
                    self.lower_b(else_label);
                }
            }
        }
        Ok(())
    }

    // ── Trap-if ──────────────────────────────────────────────────────────────────

    pub(super) fn lower_trap_if(
        &mut self,
        kind: MachineTrapKind,
        cond: &MachineBranchCond,
    ) -> Result<(), WasmError> {
        let trap_label = self.core.ensure_trap_label(kind);
        self.lower_branch_if(cond, trap_label)
    }

    pub(super) fn lower_branch_if(
        &mut self,
        cond: &MachineBranchCond,
        trap_label: usize,
    ) -> Result<(), WasmError> {
        match *cond {
            MachineBranchCond::Value(value) => match value {
                MachineValue::Imm64(0) => {}
                MachineValue::Imm64(_) => self.lower_b(trap_label),
                MachineValue::Reg(reg) => {
                    let reg = self.map_gp_reg(reg)?;
                    self.lower_cbnz(reg, trap_label);
                }
                MachineValue::ReservedReg(_reg) => {
                    return Err(WasmError::internal(
                        "arm64 trap-if cannot read reserved cache register",
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
                self.lower_b_cond(map_int_cond(kind, sign), trap_label);
            }
            MachineBranchCond::TestBits {
                width,
                kind,
                src,
                mask,
            } => {
                self.lower_tst_values(width, src, mask)?;
                let cond = match kind {
                    MachineCompareKind::Eq => enc::Cond::Eq,
                    MachineCompareKind::Ne => enc::Cond::Ne,
                    _ => {
                        return Err(WasmError::internal(
                            "TestBits branch_if: unsupported compare kind",
                        ))
                    }
                };
                self.lower_b_cond(cond, trap_label);
            }
        }
        Ok(())
    }

    // ── Compiled call ────────────────────────────────────────────────────────────

    fn lower_call(
        &mut self,
        target: &MachineCallTarget,
        callee_frame_base: MachineReg,
        caller_result_base: MachineReg,
        continuation: MachineBlockId,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        let callee_fp = self.map_gp_reg(callee_frame_base)?;
        let caller_result_base = self.map_gp_reg(caller_result_base)?;
        let fp_reg = map_fixed_reg(MACHINE_FP_REG);
        let body_local_error_label = self.core.body_local_error_label;
        let continuation_label = self.core.block_label(continuation)?;
        let continuation_is_fallthrough = fallthrough == Some(continuation);

        // Push the backend-private call record onto the host stack:
        //   stp caller_result_base, fp_reg, [sp, #-16]!
        // — order: low slot = caller_result_base, high slot = caller fp.
        // The body's unified Return / body_local_error_label pop in the
        // matching order: ldp scratch_rb, scratch_fp, [sp], #16.
        self.core.text.emit_u32(enc::stp_64_pre_index(
            caller_result_base,
            fp_reg,
            abi::stack_reg(),
            -16,
        ));

        // Switch fp_reg to the callee's frame base.
        self.core.text.emit_u32(enc::mov_reg_64(fp_reg, callee_fp));

        let scratch_idx = match target {
            MachineCallTarget::Direct(callee) => {
                // Load the callee's internal entry address from a patchable literal
                // and BLR to it. The literal itself is deferred to the per-function
                // literal pool (flushed by `lower_function_literal_pool`); the LDR
                // here is emitted with a placeholder offset and patched at flush
                // time. Deferring the literal lets us elide the trailing
                // `b continuation` when the continuation block is the next emitted
                // block — without that deferral, falling through would land in
                // the inline literal bytes.
                let scratch_idx = self.gp_scratch.alloc();
                let scratch = self.gp_scratch.reg(scratch_idx);
                let callee_load = self.core.text.emit_u32(enc::ldr_lit_64(scratch, 0));
                self.core.text.emit_u32(enc::blr(scratch));
                self.pending_call_literals
                    .push(super::backend::PendingCallLiteral {
                        ldr_offset: callee_load,
                        scratch_reg: scratch,
                        callee: *callee,
                    });
                Some(scratch_idx)
            }
            MachineCallTarget::Indirect { callee_entry, .. } => {
                let callee_entry = self.map_gp_reg(*callee_entry)?;
                self.core.text.emit_u32(enc::blr(callee_entry));
                None
            }
        };

        // --- callee returns here. C_RET0 holds 0 (success) or trap kind
        // (error). Status check propagates to body_local_error_label.
        self.lower_cbnz(abi::C_RET0, body_local_error_label);

        // Continuation: branch only if the next emitted block is not the
        // continuation block. The block layout pass already prefers placing
        // the continuation immediately after the call (see
        // `CompilerCore::extend_block_trace`), so adjacency is the common
        // case and the explicit `b` is dead code we can elide.
        if !continuation_is_fallthrough {
            self.lower_b(continuation_label);
        }

        if let Some(scratch_idx) = scratch_idx {
            self.gp_scratch.free_index(scratch_idx);
        }
        Ok(())
    }

    fn lower_tail_call(
        &mut self,
        target: &MachineCallTarget,
        callee_frame_base: MachineReg,
    ) -> Result<(), WasmError> {
        let callee_fp = self.map_gp_reg(callee_frame_base)?;
        let fp_reg = map_fixed_reg(MACHINE_FP_REG);
        let x29 = abi::host_fp_reg();
        let x30 = abi::host_lr_reg();

        let scratch_idx = match target {
            MachineCallTarget::Direct(callee) => {
                let scratch_idx = self.gp_scratch.alloc();
                let scratch = self.gp_scratch.reg(scratch_idx);
                let callee_load = self.core.text.emit_u32(enc::ldr_lit_64(scratch, 0));
                self.pending_call_literals
                    .push(super::backend::PendingCallLiteral {
                        ldr_offset: callee_load,
                        scratch_reg: scratch,
                        callee: *callee,
                    });
                Some(scratch_idx)
            }
            MachineCallTarget::Indirect { .. } => None,
        };

        self.core
            .text
            .emit_u32(enc::ldp_64_post_index(x29, x30, abi::stack_reg(), 16));
        self.core.text.emit_u32(enc::mov_reg_64(fp_reg, callee_fp));

        match target {
            MachineCallTarget::Direct(_) => {
                let scratch = self
                    .gp_scratch
                    .reg(scratch_idx.expect("direct tail-call scratch"));
                self.core.text.emit_u32(enc::br(scratch));
            }
            MachineCallTarget::Indirect { callee_entry, .. } => {
                let callee_entry = self.map_gp_reg(*callee_entry)?;
                self.core.text.emit_u32(enc::br(callee_entry));
            }
        }

        if let Some(scratch_idx) = scratch_idx {
            self.gp_scratch.free_index(scratch_idx);
        }
        Ok(())
    }

    // ── Jump table (br_table) ────────────────────────────────────────────────────

    fn lower_jump_table(
        &mut self,
        index: MachineValue,
        entries: &[MachineEdge],
    ) -> Result<(), WasmError> {
        if entries.is_empty() {
            return Err(WasmError::internal(
                "arm64 MachineIR jump table requires at least one entry".into(),
            ));
        }
        if entries.len() == 1 {
            let label = self.core.emit_edge(entries[0].target, &entries[0].args)?;
            self.lower_b(label);
            return Ok(());
        }

        let s0 = self.gp_scratch.scoped_alloc().detach();
        let s1 = self.gp_scratch.scoped_alloc().detach();
        let index_reg = prepare_gp(
            self.core.compiled.backend(),
            &self.core.fp_reg_widths,
            &mut self.core.text,
            &self.gp_scratch,
            index,
        )?
        .detach();
        // Keep C-ABI argument registers out of normal control lowering. `s1`
        // holds the clamped jump-table index first, then the scaled byte offset.
        self.materialize_u64(*s1, (entries.len() - 1) as u64);
        self.core.text.emit_u32(enc::cmp_reg_64(*index_reg, *s1));
        self.core
            .text
            .emit_u32(enc::csel_64(*s1, *index_reg, *s1, enc::Cond::Ls));

        let table_base_load = self.core.text.emit_u32(enc::ldr_lit_64(*s0, 0));
        self.core.text.emit_u32(enc::lsl_imm_64(*s1, *s1, 3));
        self.core.text.emit_u32(enc::ldr_reg_64(*s0, *s0, *s1));
        self.core.text.emit_u32(enc::br(*s0));

        let table_base_literal = self.core.text.emit_u64(0);
        let table_offset = self.core.text.len();
        let table_base_delta =
            ((table_base_literal as isize - table_base_load as isize) / 4) as i32;
        self.core
            .text
            .patch_u32(table_base_load, enc::ldr_lit_64(*s0, table_base_delta));
        self.core.resolved_ptr_patches.push(LocalPtrPatch {
            literal_offset: table_base_literal,
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

    // ── Return sequence ──────────────────────────────────────────────────────────

    /// Unified Return lowering. Pops the body prelude link save and the
    /// caller's call record from the host stack, copies the function's
    /// `return_results` region into `*caller_result_base`, restores
    /// `MACHINE_FP_REG` to the caller's frame pointer, sets `C_RET0 = 0`
    /// (the success status), and executes the platform `ret`.
    ///
    /// Uses only the 2-wide arm64 GP scratch pool by reading the call
    /// record fields with `ldr` (instead of popping with `ldp`) so that
    /// `caller_result_base` and the per-iteration data temp can coexist.
    /// The call record is freed with a single `add sp, #16` after the
    /// copy loop.
    ///
    /// The matching error path is `body_local_error_label` (see
    /// `lower_body_local_error_tail` in `arm64/backend.rs`).
    fn lower_return_sequence(&mut self) -> Result<(), WasmError> {
        let runtime = self.runtime_for(self.core.function.id)?.clone();
        let fp_reg = map_fixed_reg(MACHINE_FP_REG);
        let x29 = abi::host_fp_reg();
        let x30 = abi::host_lr_reg();

        // 1. Pop the body prelude link save (matches the prelude's
        //    `stp x29, x30, [sp, #-16]!`). After this, sp points at the
        //    caller's call record:
        //      [sp + 0] = caller_result_base
        //      [sp + 8] = caller_fp
        self.core
            .text
            .emit_u32(enc::ldp_64_post_index(x29, x30, abi::stack_reg(), 16));

        // 2. Allocate two scratches.
        let scratch_a_idx = self.gp_scratch.alloc();
        let scratch_b_idx = self.gp_scratch.alloc();
        let scratch_a = self.gp_scratch.reg(scratch_a_idx);
        let scratch_b = self.gp_scratch.reg(scratch_b_idx);

        // 3. Load caller_result_base into scratch_a. We hold it across the
        //    copy loop and release scratch_b for use as the per-iteration
        //    data temp.
        self.core
            .text
            .emit_u32(enc::ldr_64(scratch_a, abi::stack_reg(), 0));

        // 4. Copy each return slot from the callee frame to *scratch_a.
        if let Some(results) = runtime.return_results {
            for index in 0..results.slots as u32 {
                self.core.text.emit_u32(enc::ldr_64(
                    scratch_b,
                    fp_reg,
                    results.base_slot as u32 + index,
                ));
                self.core
                    .text
                    .emit_u32(enc::str_64(scratch_b, scratch_a, index));
            }
        }

        // 5. Load caller_fp into scratch_b.
        self.core
            .text
            .emit_u32(enc::ldr_64(scratch_b, abi::stack_reg(), 1));

        // 6. Pop the call record from the host stack.
        self.core
            .text
            .emit_u32(enc::add_imm_64(abi::stack_reg(), abi::stack_reg(), 16));

        // 7. Restore MACHINE_FP_REG to the caller's frame pointer.
        self.core.text.emit_u32(enc::mov_reg_64(fp_reg, scratch_b));

        // 8. Success status: C_RET0 = 0.
        self.core.text.emit_u32(enc::mov_zero_64(abi::C_RET0));

        // 9. Native return (uses LR, which the body prelude's ldp restored
        //    above).
        self.core.text.emit_u32(enc::ret());

        self.gp_scratch.free_index(scratch_b_idx);
        self.gp_scratch.free_index(scratch_a_idx);
        Ok(())
    }

    // ── Call runtime ─────────────────────────────────────────────────────────────

    pub(super) fn lower_call_runtime(&mut self, const_idx: usize) -> Result<(), WasmError> {
        let metadata = self
            .core
            .compiled
            .const_ptr(MachineConstId(const_idx as u32))
            .ok_or_else(|| WasmError::internal("arm64 runtime-call metadata is out of range"))?;

        // Runtime calls are inline runtime calls, not CFG terminators. Pass the
        // current context, the active Wasm frame pointer, and the constant-pool
        // metadata record that describes where args/results live in that frame.
        self.core
            .text
            .emit_u32(enc::mov_reg_64(abi::C_ARG0, map_fixed_reg(MACHINE_CTX_REG)));
        self.core
            .text
            .emit_u32(enc::mov_reg_64(abi::C_ARG1, map_fixed_reg(MACHINE_FP_REG)));
        self.materialize_u64(abi::C_ARG2, metadata as u64);
        let call_scratch_idx = self.gp_scratch.alloc();
        let call_scratch = self.gp_scratch.reg(call_scratch_idx);
        self.materialize_u64(call_scratch, call_runtime_entry_ptr() as usize as u64);
        self.core.text.emit_u32(enc::blr(call_scratch));
        self.gp_scratch.free_index(call_scratch_idx);

        // Nonzero helper status means the runtime stored a WasmError in
        // the NativeContext. Branch to the body-local error tail, which
        // pops the call record and propagates upward via the unified
        // Return mechanism (the caller's post-BL `cbnz w0` does the rest).
        // C_RET0 is already set by raise_trap to NativeCallStatus::Error
        // (= 1), so the propagation status flows through automatically.
        let body_local_error_label = self.core.body_local_error_label;
        self.lower_cbnz(abi::C_RET0, body_local_error_label);
        Ok(())
    }

    // ── Trap stub ────────────────────────────────────────────────────────────────

    /// Emit a trap stub -- called by `ArchBackend::emit_trap`.
    pub(super) fn lower_trap_dispatch(&mut self, kind: MachineTrapKind) {
        // Set up arguments: x0 = ctx, x1 = trap code
        self.core
            .text
            .emit_u32(enc::mov_reg_64(abi::C_ARG0, map_fixed_reg(MACHINE_CTX_REG)));
        materialize_u64_into(&mut self.core.text, abi::C_ARG1, trap_code(kind));
        let call_scratch_idx = self.gp_scratch.alloc();
        let call_scratch = self.gp_scratch.reg(call_scratch_idx);
        materialize_u64_into(&mut self.core.text, call_scratch, raise_trap as u64);
        self.core.text.emit_u32(enc::blr(call_scratch));
        self.gp_scratch.free_index(call_scratch_idx);
        // raise_trap returned with C_RET0 = NativeCallStatus::Error (= 1).
        // Branch to body_local_error_label, which preserves C_RET0 and
        // propagates upward to the caller through the unified Return tail.
        let body_local_error_label = self.core.body_local_error_label;
        self.lower_b(body_local_error_label);
    }
}
