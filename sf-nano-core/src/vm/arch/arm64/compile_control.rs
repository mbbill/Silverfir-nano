//! Control-flow emission methods for `FunctionCompiler`:
//! Terminators, branches, calls, traps, and related dispatch.

use crate::error::WasmError;
use crate::vm::machine::machine_ir::{
    MachineBranchCond, MachineBlockId, MachineConstId,
    MachineEdge, MachineFloatWidth, MachineFuncId, MachineReg,
    MachineTerminator, MachineTrapKind, MachineValue,
    MACHINE_CTX_REG, MACHINE_FP_REG,
};

use super::abi::{
    map_fixed_reg, FP_SCRATCH0, FP_SCRATCH1, SCRATCH0, SCRATCH1,
};
use super::enc::{self, Cond};
use super::reg::Arm64Reg;
use super::compile::{
    DirectCallPatch, FunctionCompiler, LabelKind, PendingLocalPtrPatch,
};
use super::compile_fusion::is_fallthrough_edge;
use super::compile_helpers::{map_int_cond, trap_code};
use super::arm64_raise_trap;
use crate::vm::runtime::context::ctx_offset;

impl<'a> FunctionCompiler<'a> {
    pub(super) fn emit_terminator(
        &mut self,
        term: &MachineTerminator,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        match term {
            MachineTerminator::Jump(edge) => {
                if is_fallthrough_edge(self, edge.target, &edge.args, fallthrough) {
                    return Ok(());
                }
                let label = self.emit_edge(edge.target, &edge.args)?;
                self.emit_b(label);
                Ok(())
            }
            MachineTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            } => self.emit_branch(cond, then_edge, else_edge, fallthrough),
            MachineTerminator::Return => self.emit_return_sequence(),
            MachineTerminator::Trap { kind } => {
                self.emit_trap(*kind);
                Ok(())
            }
            MachineTerminator::JumpTable { index, entries } => {
                self.emit_jump_table(*index, entries)
            }
            MachineTerminator::CallDirect {
                callee,
                callee_frame_base,
                continuation,
            } => self.emit_call_direct(*callee, *callee_frame_base, *continuation),
            MachineTerminator::CallIndirect {
                callee_target,
                callee_frame_base,
                arg_slots,
                caller_result_base,
                continuation,
            } => self.emit_call_indirect(
                *callee_target,
                *callee_frame_base,
                *arg_slots,
                *caller_result_base,
                *continuation,
            ),
        }
    }

    pub(super) fn emit_branch(
        &mut self,
        cond: &MachineBranchCond,
        then_edge: &MachineEdge,
        else_edge: &MachineEdge,
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
                        self.emit_b(label);
                    }
                }
                MachineValue::Imm64(_) => {
                    if let Some(label) = then_label {
                        self.emit_b(label);
                    }
                }
                MachineValue::Reg(reg) => {
                    let reg = self.map_gp_reg(reg)?;
                    if else_fallthrough {
                        if let Some(label) = then_label {
                            self.emit_cbnz(reg, label);
                        }
                    } else if then_fallthrough {
                        if let Some(label) = else_label {
                            self.emit_cbz(reg, label);
                        }
                    } else if let (Some(then_label), Some(else_label)) = (then_label, else_label) {
                        self.emit_cbnz(reg, then_label);
                        self.emit_b(else_label);
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
                if else_fallthrough {
                    if let Some(label) = then_label {
                        self.emit_b_cond(map_int_cond(kind, sign), label);
                    }
                } else if then_fallthrough {
                    if let Some(label) = else_label {
                        self.emit_b_cond(map_int_cond(kind, sign).invert(), label);
                    }
                } else if let (Some(then_label), Some(else_label)) = (then_label, else_label) {
                    self.emit_b_cond(map_int_cond(kind, sign), then_label);
                    self.emit_b(else_label);
                }
            }
        }
        Ok(())
    }

    /// Emit a conditional branch when the CPU flags have already been set
    /// by a preceding CMP/FCMP.
    pub(super) fn emit_fused_cond_branch(
        &mut self,
        cond: Cond,
        then_edge: &MachineEdge,
        else_edge: &MachineEdge,
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

        if else_fallthrough {
            if let Some(label) = then_label {
                self.emit_b_cond(cond, label);
            }
        } else if then_fallthrough {
            if let Some(label) = else_label {
                self.emit_b_cond(cond.invert(), label);
            }
        } else if let (Some(then_label), Some(else_label)) = (then_label, else_label) {
            self.emit_b_cond(cond, then_label);
            self.emit_b(else_label);
        }
        Ok(())
    }

    /// Emit FCMP without CSET (for float compare-and-branch fusion).
    pub(super) fn emit_fcmp_values(
        &mut self,
        width: MachineFloatWidth,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        let lhs_fp = self.prepare_float_operand(width, lhs, SCRATCH0, FP_SCRATCH0)?;
        if matches!(rhs, MachineValue::Imm64(0)) {
            match width {
                MachineFloatWidth::F32 => self.text.emit_u32(enc::fcmp_s_zero(lhs_fp)),
                MachineFloatWidth::F64 => self.text.emit_u32(enc::fcmp_d_zero(lhs_fp)),
            };
        } else {
            let rhs_fp = self.prepare_float_operand(width, rhs, SCRATCH1, FP_SCRATCH1)?;
            match width {
                MachineFloatWidth::F32 => self.text.emit_u32(enc::fcmp_s(lhs_fp, rhs_fp)),
                MachineFloatWidth::F64 => self.text.emit_u32(enc::fcmp_d(lhs_fp, rhs_fp)),
            };
        }
        Ok(())
    }

    pub(super) fn emit_trap_if(
        &mut self,
        kind: MachineTrapKind,
        cond: &MachineBranchCond,
    ) -> Result<(), WasmError> {
        let trap_label = self.ensure_trap_label(kind);
        self.emit_branch_if(cond, trap_label)
    }

    pub(super) fn emit_branch_if(
        &mut self,
        cond: &MachineBranchCond,
        trap_label: usize,
    ) -> Result<(), WasmError> {
        match *cond {
            MachineBranchCond::Value(value) => match value {
                MachineValue::Imm64(0) => {}
                MachineValue::Imm64(_) => self.emit_b(trap_label),
                MachineValue::Reg(reg) => {
                    let reg = self.map_gp_reg(reg)?;
                    self.emit_cbnz(reg, trap_label);
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
                self.emit_b_cond(map_int_cond(kind, sign), trap_label);
            }
        }
        Ok(())
    }

    pub(super) fn emit_call_direct(
        &mut self,
        callee: MachineFuncId,
        callee_frame_base: MachineReg,
        continuation: MachineBlockId,
    ) -> Result<(), WasmError> {
        let callee_runtime = self.runtime_for(callee)?;
        let call_scratch = callee_runtime.call_scratch.ok_or_else(|| {
            WasmError::internal("arm64 direct local call requires callee call scratch".into())
        })?;
        let continuation_slot = call_scratch.base_slot
            + (self.compiled.runtime().call_link.continuation_offset / 8) as u16;
        let callee_fp = self.map_gp_reg(callee_frame_base)?;

        let continuation_load = self.text.emit_u32(enc::ldr_lit_64(SCRATCH0, 0));
        if continuation_slot < 4096 {
            self.text
                .emit_u32(enc::str_64(SCRATCH0, callee_fp, continuation_slot as u32));
        } else {
            self.materialize_u64(SCRATCH1, u64::from(continuation_slot) * 8);
            self.text
                .emit_u32(enc::add_reg_64(SCRATCH1, callee_fp, SCRATCH1));
            self.text
                .emit_u32(enc::str_reg_64(SCRATCH0, SCRATCH1, Arm64Reg::Xzr));
        }

        let callee_load = self.text.emit_u32(enc::ldr_lit_64(SCRATCH0, 0));
        self.text
            .emit_u32(enc::mov_reg_64(map_fixed_reg(MACHINE_FP_REG), callee_fp));
        self.text.emit_u32(enc::br(SCRATCH0));

        let continuation_literal = self.text.emit_u64(0);
        let callee_literal = self.text.emit_u64(0);

        let continuation_label = self.block_label(continuation)?;
        let continuation_delta =
            ((continuation_literal as isize - continuation_load as isize) / 4) as i32;
        self.text.patch_u32(
            continuation_load,
            enc::ldr_lit_64(SCRATCH0, continuation_delta),
        );
        let callee_delta = ((callee_literal as isize - callee_load as isize) / 4) as i32;
        self.text
            .patch_u32(callee_load, enc::ldr_lit_64(SCRATCH0, callee_delta));

        self.local_ptr_patches.push(PendingLocalPtrPatch {
            literal_offset: continuation_literal,
            target_label: continuation_label,
        });
        self.direct_call_patches.push(DirectCallPatch {
            literal_offset: callee_literal,
            callee,
        });
        Ok(())
    }

    pub(super) fn emit_jump_table(
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
            let label = self.emit_edge(entries[0].target, &entries[0].args)?;
            self.emit_b(label);
            return Ok(());
        }

        let index_reg = self.materialize_value(SCRATCH1, index)?;
        self.materialize_u64(Arm64Reg::X0, (entries.len() - 1) as u64);
        self.text.emit_u32(enc::cmp_reg_64(index_reg, Arm64Reg::X0));
        self.text
            .emit_u32(enc::csel_64(SCRATCH1, index_reg, Arm64Reg::X0, Cond::Ls));

        let table_base_load = self.text.emit_u32(enc::ldr_lit_64(SCRATCH0, 0));
        self.materialize_u64(Arm64Reg::X0, 3);
        self.text
            .emit_u32(enc::lslv_64(SCRATCH1, SCRATCH1, Arm64Reg::X0));
        self.text
            .emit_u32(enc::ldr_reg_64(SCRATCH0, SCRATCH0, SCRATCH1));
        self.text.emit_u32(enc::br(SCRATCH0));

        let table_base_literal = self.text.emit_u64(0);
        let table_offset = self.text.len();
        let table_base_delta =
            ((table_base_literal as isize - table_base_load as isize) / 4) as i32;
        self.text
            .patch_u32(table_base_load, enc::ldr_lit_64(SCRATCH0, table_base_delta));
        self.resolved_ptr_patches.push(super::compile::LocalPtrPatch {
            literal_offset: table_base_literal,
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

    pub(super) fn emit_return_sequence(&mut self) -> Result<(), WasmError> {
        let runtime = *self.runtime_for(self.function.id)?;
        let call_scratch = runtime.call_scratch.ok_or_else(|| {
            WasmError::internal("arm64 local return requires call scratch".into())
        })?;
        let call_link = self.compiled.runtime().call_link;
        let continuation_slot = call_scratch.base_slot + (call_link.continuation_offset / 8) as u16;
        let caller_frame_slot = call_scratch.base_slot + (call_link.caller_frame_offset / 8) as u16;
        let caller_result_base_slot =
            call_scratch.base_slot + (call_link.caller_result_base_offset / 8) as u16;

        self.text.emit_u32(enc::ldr_64(
            SCRATCH0,
            map_fixed_reg(MACHINE_FP_REG),
            continuation_slot as u32,
        ));
        self.text.emit_u32(enc::ldr_64(
            SCRATCH1,
            map_fixed_reg(MACHINE_FP_REG),
            caller_frame_slot as u32,
        ));
        self.text.emit_u32(enc::ldr_64(
            Arm64Reg::X0,
            map_fixed_reg(MACHINE_FP_REG),
            caller_result_base_slot as u32,
        ));
        self.text
            .emit_u32(enc::add_reg_64(Arm64Reg::X0, SCRATCH1, Arm64Reg::X0));

        if let Some(results) = runtime.return_results {
            for index in 0..results.slots as u32 {
                self.text.emit_u32(enc::ldr_64(
                    Arm64Reg::X1,
                    map_fixed_reg(MACHINE_FP_REG),
                    results.base_slot as u32 + index,
                ));
                self.text
                    .emit_u32(enc::str_64(Arm64Reg::X1, Arm64Reg::X0, index));
            }
        }

        self.text
            .emit_u32(enc::mov_reg_64(map_fixed_reg(MACHINE_FP_REG), SCRATCH1));
        self.text.emit_u32(enc::br(SCRATCH0));
        Ok(())
    }

    pub(super) fn emit_call_indirect(
        &mut self,
        callee_target: MachineValue,
        callee_frame_base: MachineReg,
        arg_slots: u16,
        caller_result_base: u16,
        continuation: MachineBlockId,
    ) -> Result<(), WasmError> {
        let callee_id_reg = self.materialize_value(SCRATCH1, callee_target)?;
        let table_base_load = self.text.emit_u32(enc::ldr_lit_64(SCRATCH0, 0));
        let skip_table_literal = self.text.emit_u32(enc::b(0)); // skip over literal
        self.function_table_patches.push(self.text.emit_u64(0));
        let table_base_literal = self
            .function_table_patches
            .last()
            .copied()
            .expect("function table literal recorded");
        let after_table_literal = self.text.len();
        // Patch the skip branch
        let skip_delta = ((after_table_literal as isize - skip_table_literal as isize) / 4) as i32;
        self.text.patch_u32(skip_table_literal, enc::b(skip_delta));
        // Patch the ldr literal offset
        let table_base_delta =
            ((table_base_literal as isize - table_base_load as isize) / 4) as i32;
        self.text
            .patch_u32(table_base_load, enc::ldr_lit_64(SCRATCH0, table_base_delta));

        self.materialize_u64(Arm64Reg::X0, 5);
        self.text
            .emit_u32(enc::lslv_64(SCRATCH1, callee_id_reg, Arm64Reg::X0));
        self.text
            .emit_u32(enc::add_reg_64(SCRATCH0, SCRATCH0, SCRATCH1));
        self.text.emit_u32(enc::ldr_64(Arm64Reg::X0, SCRATCH0, 0));
        self.text.emit_u32(enc::ldr_64(Arm64Reg::X1, SCRATCH0, 1));
        self.text.emit_u32(enc::ldr_64(Arm64Reg::X2, SCRATCH0, 2));
        self.text.emit_u32(enc::ldr_64(Arm64Reg::X3, SCRATCH0, 3));

        let callee_fp = self.map_gp_reg(callee_frame_base)?;
        self.text
            .emit_u32(enc::add_reg_64(SCRATCH0, callee_fp, Arm64Reg::X1));
        self.text.emit_u32(enc::ldr_64(
            SCRATCH1,
            map_fixed_reg(MACHINE_CTX_REG),
            (ctx_offset::STACK_END / 8) as u32,
        ));
        self.text.emit_u32(enc::cmp_reg_64(SCRATCH0, SCRATCH1));
        self.emit_b_cond(Cond::Hi, self.stack_overflow_label);

        self.emit_zero_dynamic_callee_prefix(callee_fp, arg_slots)?;

        let continuation_load = self.text.emit_u32(enc::ldr_lit_64(SCRATCH0, 0));
        let skip_cont_literal = self.text.emit_u32(enc::b(0)); // skip over literal
        let continuation_literal = self.text.emit_u64(0);
        let after_cont_literal = self.text.len();
        let skip_cont_delta =
            ((after_cont_literal as isize - skip_cont_literal as isize) / 4) as i32;
        self.text
            .patch_u32(skip_cont_literal, enc::b(skip_cont_delta));
        let continuation_label = self.block_label(continuation)?;
        let continuation_delta =
            ((continuation_literal as isize - continuation_load as isize) / 4) as i32;
        self.text.patch_u32(
            continuation_load,
            enc::ldr_lit_64(SCRATCH0, continuation_delta),
        );
        self.local_ptr_patches.push(PendingLocalPtrPatch {
            literal_offset: continuation_literal,
            target_label: continuation_label,
        });

        self.materialize_u64(SCRATCH1, 3);
        self.text
            .emit_u32(enc::lslv_64(Arm64Reg::X3, Arm64Reg::X3, SCRATCH1));
        self.text
            .emit_u32(enc::add_reg_64(Arm64Reg::X3, callee_fp, Arm64Reg::X3));
        self.text
            .emit_u32(enc::str_reg_64(SCRATCH0, Arm64Reg::X3, Arm64Reg::Xzr));
        self.text
            .emit_u32(enc::str_64(map_fixed_reg(MACHINE_FP_REG), Arm64Reg::X3, 1));
        self.materialize_u64(SCRATCH1, u64::from(caller_result_base) * 8);
        self.text.emit_u32(enc::str_64(SCRATCH1, Arm64Reg::X3, 2));

        self.text
            .emit_u32(enc::mov_reg_64(map_fixed_reg(MACHINE_FP_REG), callee_fp));
        self.text.emit_u32(enc::br(Arm64Reg::X0));
        Ok(())
    }

    pub(super) fn emit_zero_dynamic_callee_prefix(
        &mut self,
        callee_fp: Arm64Reg,
        arg_slots: u16,
    ) -> Result<(), WasmError> {
        self.materialize_u64(SCRATCH0, u64::from(arg_slots) * 8);
        self.text
            .emit_u32(enc::add_reg_64(SCRATCH0, callee_fp, SCRATCH0));
        self.materialize_u64(SCRATCH1, 3);
        self.text
            .emit_u32(enc::lslv_64(Arm64Reg::X2, Arm64Reg::X2, SCRATCH1));
        self.text
            .emit_u32(enc::add_reg_64(Arm64Reg::X2, callee_fp, Arm64Reg::X2));
        self.text.emit_u32(enc::cmp_reg_64(SCRATCH0, Arm64Reg::X2));
        let done = self.new_label(LabelKind::Edge);
        let loop_label = self.new_label(LabelKind::Edge);
        self.emit_b_cond(Cond::Hs, done);
        self.bind_label(loop_label);
        self.text
            .emit_u32(enc::str_reg_64(Arm64Reg::Xzr, SCRATCH0, Arm64Reg::Xzr));
        self.text.emit_u32(enc::add_imm_64(SCRATCH0, SCRATCH0, 8));
        self.text.emit_u32(enc::cmp_reg_64(SCRATCH0, Arm64Reg::X2));
        self.emit_b_cond(Cond::Lo, loop_label);
        self.bind_label(done);
        Ok(())
    }

    pub(super) fn emit_call_helper(
        &mut self,
        extern_idx: usize,
        const_idx: usize,
    ) -> Result<(), WasmError> {
        let binding = self
            .compiled
            .module()
            .externs
            .get(extern_idx)
            .ok_or_else(|| WasmError::internal("arm64 helper target is out of range".into()))?;
        let metadata = self
            .compiled
            .const_ptr(MachineConstId(
                const_idx as u32,
            ))
            .ok_or_else(|| WasmError::internal("arm64 helper metadata is out of range".into()))?;
        self.text.emit_u32(enc::mov_reg_64(
            Arm64Reg::X0,
            map_fixed_reg(MACHINE_CTX_REG),
        ));
        self.text
            .emit_u32(enc::mov_reg_64(Arm64Reg::X1, map_fixed_reg(MACHINE_FP_REG)));
        self.materialize_u64(Arm64Reg::X2, metadata as u64);
        self.materialize_u64(
            SCRATCH0,
            crate::vm::runtime::helpers::resolve_helper_entry(binding.symbol) as usize as u64,
        );
        self.text.emit_u32(enc::blr(SCRATCH0));
        self.emit_cbnz(Arm64Reg::X0, self.return_error_label);
        Ok(())
    }

    pub(super) fn emit_trap(&mut self, kind: MachineTrapKind) {
        self.text.emit_u32(enc::mov_reg_64(
            Arm64Reg::X0,
            map_fixed_reg(MACHINE_CTX_REG),
        ));
        self.materialize_u64(Arm64Reg::X1, trap_code(kind));
        self.materialize_u64(SCRATCH0, arm64_raise_trap as usize as u64);
        self.text.emit_u32(enc::blr(SCRATCH0));
        self.emit_b(self.return_error_label);
    }

}
