//! x86_64 backend: control flow lowering methods for X86_64Backend.

use crate::{
    error::WasmError,
    vm::machine::machine_ir::{
        MachineBlockId, MachineBranchCond, MachineCompareKind, MachineEdge, MachineFloatWidth,
        MachineFuncId, MachineIntWidth, MachineReg, MachineTerminator, MachineTrapKind,
        MachineValue, MACHINE_CTX_REG, MACHINE_FP_REG,
    },
};

use super::{
    abi::map_fixed_reg,
    backend::X86_64Backend,
    enc::{self, Cc},
    fusion::{map_float_cond, map_int_cond},
    reg::X86Reg,
};

use crate::vm::arch::common::helpers::{is_fallthrough_edge, trap_code};
use crate::vm::arch::common::types::{DirectCallPatch, LocalPtrPatch, PendingLocalPtrPatch};
use crate::vm::runtime::{context::ctx_offset, layout::local_call_info_abi_layout};

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
                call_link_base,
                continuation,
            } => {
                self.lower_call_direct(*callee, *callee_frame_base, *call_link_base, *continuation)
            }
            MachineTerminator::CallIndirect {
                callee_target,
                callee_entry,
                callee_frame_base,
                call_link_base,
                continuation,
            } => self.lower_call_indirect(
                *callee_target,
                *callee_entry,
                *callee_frame_base,
                *call_link_base,
                *continuation,
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
                    enc::test_rr_64(&mut self.core.text, reg, reg);
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
                        return Err(WasmError::internal(alloc::format!(
                            "TestBits branch: unsupported compare kind {:?}",
                            kind
                        )))
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
                    enc::test_rr_64(&mut self.core.text, reg, reg);
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
                        return Err(WasmError::internal(alloc::format!(
                            "TestBits branch_if: unsupported compare kind {:?}",
                            kind
                        )))
                    }
                };
                self.emit_jcc(cc, trap_label);
            }
        }
        Ok(())
    }

    fn lower_float_branch(
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
        let lhs_fp =
            self.prepare_float_operand(width, lhs, self.gp_scratch.reg(0), self.fp_scratch.reg(0))?;
        let rhs_fp_scratch = if lhs_fp != self.fp_scratch.reg(0) as u32 {
            self.fp_scratch.reg(0)
        } else {
            self.fp_scratch.reg(2)
        };
        if matches!(rhs, MachineValue::Imm64(0)) {
            enc::xorpd(
                &mut self.core.text,
                rhs_fp_scratch as u8,
                rhs_fp_scratch as u8,
            );
            match width {
                MachineFloatWidth::F32 => {
                    enc::ucomiss(&mut self.core.text, lhs_fp as u8, rhs_fp_scratch as u8)
                }
                MachineFloatWidth::F64 => {
                    enc::ucomisd(&mut self.core.text, lhs_fp as u8, rhs_fp_scratch as u8)
                }
            };
        } else {
            let rhs_fp =
                self.prepare_float_operand(width, rhs, self.gp_scratch.reg(1), rhs_fp_scratch)?;
            match width {
                MachineFloatWidth::F32 => {
                    enc::ucomiss(&mut self.core.text, lhs_fp as u8, rhs_fp as u8)
                }
                MachineFloatWidth::F64 => {
                    enc::ucomisd(&mut self.core.text, lhs_fp as u8, rhs_fp as u8)
                }
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

    // ── Trap stub ────────────────────────────────────────────────────────────

    pub(super) fn lower_trap_dispatch(&mut self, kind: MachineTrapKind) {
        use super::abi::{C_ARG0, C_ARG1};
        enc::mov_rr_64(&mut self.core.text, C_ARG0, map_fixed_reg(MACHINE_CTX_REG));
        self.materialize_u64(C_ARG1, trap_code(kind));
        // MOV R11, raise_trap address
        self.materialize_u64(
            self.gp_scratch.reg(1),
            super::helpers::x86_64_raise_trap as u64,
        );
        // CALL R11
        enc::call_reg(&mut self.core.text, self.gp_scratch.reg(1));
        // JMP return_error_label
        let return_error_label = self.core.return_error_label;
        self.emit_jmp(return_error_label);
    }

    // ── Return sequence ──────────────────────────────────────────────────────

    fn lower_return_sequence(&mut self) -> Result<(), WasmError> {
        let runtime = *self.core.runtime_for(self.core.function.id)?;
        let call_scratch = runtime.call_scratch.ok_or_else(|| {
            WasmError::internal("x86_64 local return requires call scratch".into())
        })?;
        let call_link = self.core.compiled.abi().call_link;
        let continuation_offset =
            (call_scratch.base_slot as i32) * 8 + call_link.continuation_offset as i32;
        let caller_frame_offset =
            (call_scratch.base_slot as i32) * 8 + call_link.caller_frame_offset as i32;
        let caller_result_base_offset =
            (call_scratch.base_slot as i32) * 8 + call_link.caller_result_base_offset as i32;

        let fp = map_fixed_reg(MACHINE_FP_REG);
        // Load continuation address into RDI (caller-saved, not self.gp_scratch.reg(0)=RAX)
        enc::load_64(&mut self.core.text, X86Reg::RDI, fp, continuation_offset);
        // Load caller frame pointer
        enc::load_64(
            &mut self.core.text,
            self.gp_scratch.reg(1),
            fp,
            caller_frame_offset,
        );
        // Load caller result base offset
        enc::load_64(
            &mut self.core.text,
            self.gp_scratch.reg(0),
            fp,
            caller_result_base_offset,
        );
        // result_ptr = caller_fp + result_base
        enc::add_rr_64(
            &mut self.core.text,
            self.gp_scratch.reg(0),
            self.gp_scratch.reg(1),
        );

        // Copy results
        if let Some(results) = runtime.return_results {
            for index in 0..results.slots as i32 {
                enc::load_64(
                    &mut self.core.text,
                    X86Reg::RCX,
                    fp,
                    (results.base_slot as i32 + index) * 8,
                );
                enc::store_64(
                    &mut self.core.text,
                    self.gp_scratch.reg(0),
                    index * 8,
                    X86Reg::RCX,
                );
            }
        }

        // Restore caller FP
        enc::mov_rr_64(&mut self.core.text, fp, self.gp_scratch.reg(1));
        // Jump to continuation
        enc::jmp_reg(&mut self.core.text, X86Reg::RDI);
        Ok(())
    }

    // ── Call direct ──────────────────────────────────────────────────────────

    fn lower_call_direct(
        &mut self,
        callee: MachineFuncId,
        callee_frame_base: MachineReg,
        call_link_base: MachineReg,
        continuation: MachineBlockId,
    ) -> Result<(), WasmError> {
        let callee_fp = self.map_gp_reg(callee_frame_base)?;
        let call_link_base = self.map_gp_reg(call_link_base)?;
        let call_link = self.core.compiled.abi().call_link;

        // Load continuation address via movabs (patched after label resolution).
        enc::movabs_ri_64(&mut self.core.text, self.gp_scratch.reg(0), 0);
        let cont_imm_offset = self.core.text.len() - 8;
        let continuation_label = self.core.block_label(continuation)?;
        self.core.local_ptr_patches.push(PendingLocalPtrPatch {
            literal_offset: cont_imm_offset,
            target_label: continuation_label,
        });

        // Store continuation address into callee's call-link slot
        enc::store_64(
            &mut self.core.text,
            call_link_base,
            call_link.continuation_offset as i32,
            self.gp_scratch.reg(0),
        );

        // Load callee entry address via movabs (patched after compilation)
        enc::movabs_ri_64(&mut self.core.text, self.gp_scratch.reg(0), 0);
        let callee_imm_offset = self.core.text.len() - 8;
        self.core.direct_call_patches.push(DirectCallPatch {
            literal_offset: callee_imm_offset,
            callee,
        });

        // Set FP to callee frame
        enc::mov_rr_64(
            &mut self.core.text,
            map_fixed_reg(MACHINE_FP_REG),
            callee_fp,
        );
        // Jump to callee
        enc::jmp_reg(&mut self.core.text, self.gp_scratch.reg(0));
        Ok(())
    }

    // ── Call indirect ────────────────────────────────────────────────────────

    fn lower_call_indirect(
        &mut self,
        _callee_target: MachineReg,
        callee_entry: MachineReg,
        callee_frame_base: MachineReg,
        call_link_base: MachineReg,
        continuation: MachineBlockId,
    ) -> Result<(), WasmError> {
        let callee_fp_orig = self.map_gp_reg(callee_frame_base)?;
        let callee_fp = X86Reg::R8;
        if callee_fp_orig != callee_fp {
            enc::mov_rr_64(&mut self.core.text, callee_fp, callee_fp_orig);
        }
        enc::movabs_ri_64(&mut self.core.text, self.gp_scratch.reg(0), 0);
        let cont_imm_offset = self.core.text.len() - 8;
        let continuation_label = self.core.block_label(continuation)?;
        self.core.local_ptr_patches.push(PendingLocalPtrPatch {
            literal_offset: cont_imm_offset,
            target_label: continuation_label,
        });

        let call_link_base = self.map_gp_reg(call_link_base)?;
        let call_link = self.core.compiled.abi().call_link;
        enc::store_64(
            &mut self.core.text,
            call_link_base,
            call_link.continuation_offset as i32,
            self.gp_scratch.reg(0),
        );

        let callee_entry = self.map_gp_reg(callee_entry)?;
        enc::mov_rr_64(
            &mut self.core.text,
            map_fixed_reg(MACHINE_FP_REG),
            callee_fp,
        );
        enc::jmp_reg(&mut self.core.text, callee_entry);
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

        let index_reg = self.materialize_value(self.gp_scratch.reg(1), index)?;
        // Clamp index to (entries.len() - 1)
        self.materialize_u64(X86Reg::RAX, (entries.len() - 1) as u64);
        enc::cmp_rr_64(&mut self.core.text, index_reg, X86Reg::RAX);
        enc::cmovcc_rr_64(
            &mut self.core.text,
            Cc::A,
            self.gp_scratch.reg(1),
            X86Reg::RAX,
        );
        if index_reg != self.gp_scratch.reg(1) {
            enc::mov_rr_64(&mut self.core.text, self.gp_scratch.reg(1), index_reg);
            enc::cmovcc_rr_64(
                &mut self.core.text,
                Cc::A,
                self.gp_scratch.reg(1),
                X86Reg::RAX,
            );
        }

        // Load table base address (absolute, patched later)
        enc::movabs_ri_64(&mut self.core.text, self.gp_scratch.reg(0), 0);
        let table_base_imm_offset = self.core.text.len() - 8;

        // index * 8 for table entry
        enc::shl_imm_64(&mut self.core.text, self.gp_scratch.reg(1), 3);
        enc::add_rr_64(
            &mut self.core.text,
            self.gp_scratch.reg(0),
            self.gp_scratch.reg(1),
        );
        // Load target address from table
        enc::load_64(
            &mut self.core.text,
            self.gp_scratch.reg(0),
            self.gp_scratch.reg(0),
            0,
        );
        enc::jmp_reg(&mut self.core.text, self.gp_scratch.reg(0));

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
