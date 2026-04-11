//! x86_64 backend: control flow lowering methods for X86_64Backend.

use crate::{
    error::WasmError,
    vm::machine::machine_ir::{
        MachineBlockId, MachineBranchCond, MachineCompareKind, MachineConstId, MachineEdge,
        MachineFuncId, MachineReg, MachineTerminator, MachineTrapKind, MachineValue,
        MACHINE_CTX_REG, MACHINE_FP_REG,
    },
};

use super::{
    abi::{map_fixed_reg, C_ARG0, C_ARG1, C_ARG2},
    backend::X86_64Backend,
    enc::{self, Cc},
    fusion::map_int_cond,
    reg::X86Reg,
};

use crate::vm::arch::common::helpers::trap_code;
use crate::vm::arch::common::types::{DirectCallPatch, LocalPtrPatch, PendingLocalPtrPatch};
use crate::vm::arch::shared_64::is_fallthrough_edge;

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
                    &self.core.function.program.blocks,
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
            MachineTerminator::Return => self.lower_return_sequence(),
            MachineTerminator::Trap { kind } => {
                self.lower_trap_dispatch(*kind);
                Ok(())
            }
            MachineTerminator::JumpTable { index, entries } => {
                self.lower_jump_table(*index, entries)
            }
            MachineTerminator::CallDirect {
                callee,
                callee_frame_base,
                caller_result_base,
                continuation,
            } => self.lower_call_direct(
                *callee,
                *callee_frame_base,
                *caller_result_base,
                *continuation,
                fallthrough,
            ),
            MachineTerminator::CallIndirect {
                callee_target,
                callee_entry,
                callee_frame_base,
                caller_result_base,
                continuation,
            } => self.lower_call_indirect(
                *callee_target,
                *callee_entry,
                *callee_frame_base,
                *caller_result_base,
                *continuation,
                fallthrough,
            ),
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

    // ── Trap stub ────────────────────────────────────────────────────────────

    /// Emit a trap stub: set up `raise_trap(ctx, trap_code)`, call it, then
    /// branch to `body_local_error_label`. After `raise_trap` returns,
    /// `C_RET0` (RAX) already holds `NativeCallStatus::Error`, so the
    /// propagation status flows unchanged through the unified Return tail.
    pub(super) fn lower_trap_dispatch(&mut self, kind: MachineTrapKind) {
        enc::mov_rr_64(&mut self.core.text, C_ARG0, map_fixed_reg(MACHINE_CTX_REG));
        self.materialize_u64(C_ARG1, trap_code(kind));
        let call_scratch = self.gp_scratch.claim_rax().detach();
        self.materialize_u64(
            *call_scratch,
            crate::vm::runtime::trap::raise_trap as usize as u64,
        );
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
        let runtime = self.core.runtime_for(self.core.function.id)?.clone();
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

    // ── Call direct ──────────────────────────────────────────────────────────

    /// Lower a local SF→SF direct call. Pushes the call record (caller's
    /// `caller_result_base` + caller's fp_reg), switches fp_reg to the
    /// callee's frame base, emits `mov scratch, imm64` + `call scratch` where
    /// the imm64 is patched to the callee's internal entry at module-link
    /// time, then a post-call status check that branches to
    /// `body_local_error_label` on non-zero RAX.
    fn lower_call_direct(
        &mut self,
        callee: MachineFuncId,
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
        let call_scratch = self.gp_scratch.claim_rax().detach();

        // Push the call record: caller_fp at higher slot, caller_result_base
        // at lower slot. Matches the layout consumed by the body's unified
        // Return and body_local_error_tail.
        enc::push(&mut self.core.text, fp_reg);
        enc::push(&mut self.core.text, caller_result_base);

        // Switch fp_reg to the callee's frame base.
        enc::mov_rr_64(&mut self.core.text, fp_reg, callee_fp);

        // Load the callee's internal entry address via patchable movabs.
        enc::movabs_ri_64(&mut self.core.text, *call_scratch, 0);
        let callee_imm_offset = self.core.text.len() - 8;
        self.core.direct_call_patches.push(DirectCallPatch {
            literal_offset: callee_imm_offset,
            callee,
        });
        // CALL scratch — pushes the return address (= this call site's
        // fall-through) onto the host stack.
        enc::call_reg(&mut self.core.text, *call_scratch);

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

    // ── Call indirect ────────────────────────────────────────────────────────

    fn lower_call_indirect(
        &mut self,
        _callee_target: MachineReg,
        callee_entry: MachineReg,
        callee_frame_base: MachineReg,
        caller_result_base: MachineReg,
        continuation: MachineBlockId,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        let callee_fp = self.map_gp_reg(callee_frame_base)?;
        let caller_result_base = self.map_gp_reg(caller_result_base)?;
        let callee_entry = self.map_gp_reg(callee_entry)?;
        let fp_reg = map_fixed_reg(MACHINE_FP_REG);
        let body_local_error_label = self.core.body_local_error_label;
        let continuation_label = self.core.block_label(continuation)?;
        let continuation_is_fallthrough = fallthrough == Some(continuation);

        // Push the call record (same shape as CallDirect).
        enc::push(&mut self.core.text, fp_reg);
        enc::push(&mut self.core.text, caller_result_base);

        // Switch fp_reg to the callee's frame base.
        enc::mov_rr_64(&mut self.core.text, fp_reg, callee_fp);

        // CALL callee_entry — runtime-resolved target.
        enc::call_reg(&mut self.core.text, callee_entry);

        // Status check + continuation branch.
        enc::test_rr_64(&mut self.core.text, super::abi::C_RET0, super::abi::C_RET0);
        self.emit_jcc(Cc::NE, body_local_error_label);
        if !continuation_is_fallthrough {
            self.emit_jmp(continuation_label);
        }
        Ok(())
    }

    // ── Call external ────────────────────────────────────────────────────────

    pub(super) fn lower_call_external_term(&mut self, const_idx: usize) -> Result<(), WasmError> {
        let metadata = self
            .core
            .compiled
            .const_ptr(MachineConstId(const_idx as u32))
            .ok_or_else(|| WasmError::internal("x86_64 external-call metadata is out of range"))?;
        // External calls are inline runtime calls. Pass the current context,
        // the active Wasm frame pointer, and the const-pool metadata record.
        enc::mov_rr_64(&mut self.core.text, C_ARG0, map_fixed_reg(MACHINE_CTX_REG));
        enc::mov_rr_64(&mut self.core.text, C_ARG1, map_fixed_reg(MACHINE_FP_REG));
        self.materialize_u64(C_ARG2, metadata as u64);
        let call_scratch = self.gp_scratch.claim_rax().detach();
        self.materialize_u64(
            *call_scratch,
            crate::vm::runtime::external::call_external_entry_ptr() as usize as u64,
        );
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
