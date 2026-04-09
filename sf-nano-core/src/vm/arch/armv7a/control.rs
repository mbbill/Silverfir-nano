//! Terminator and branch-condition compilation for the ARMv7-A backend.

use crate::{
    error::WasmError,
    vm::machine::machine_ir::{
        MachineBlockId, MachineBranchCond, MachineCompareKind, MachineSign, MachineTerminator,
        MachineValue,
    },
};

use super::{
    abi::map_reg,
    backend::{Arm32Backend, BranchFixupKind},
    enc::{self, Cond},
    inst::{emit_load_u32_into, prepare_gp},
    reg::Arm32Reg,
};

// ─── Terminator compilation ─────────────────────────────────────────────────

impl<'a> Arm32Backend<'a> {
    pub(super) fn lower_terminator_dispatch(
        &mut self,
        terminator: &MachineTerminator,
        _fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        match terminator {
            MachineTerminator::Return => {
                self.emit_return_sequence()?;
            }

            MachineTerminator::Jump(edge) => {
                let label = self.core.emit_edge(edge.target, &edge.args)?;
                self.emit_branch(BranchFixupKind::B, label);
            }

            MachineTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            } => {
                let then_label = self.core.emit_edge(then_edge.target, &then_edge.args)?;
                let else_label = self.core.emit_edge(else_edge.target, &else_edge.args)?;

                let arm_cond = self.compile_branch_condition(cond)?;
                self.emit_branch(BranchFixupKind::BCond(arm_cond), then_label);
                self.emit_branch(BranchFixupKind::B, else_label);
            }

            MachineTerminator::Trap { kind } => {
                let trap_label = self.core.ensure_trap_label(*kind);
                self.emit_branch(BranchFixupKind::B, trap_label);
            }

            MachineTerminator::CallDirect {
                callee,
                callee_frame_base,
                caller_result_base,
                continuation,
            } => {
                self.emit_call_direct(
                    *callee,
                    *callee_frame_base,
                    *caller_result_base,
                    *continuation,
                )?;
            }

            MachineTerminator::CallIndirect {
                callee_target,
                callee_entry,
                callee_frame_base,
                caller_result_base,
                continuation,
            } => {
                self.emit_call_indirect(
                    *callee_target,
                    *callee_entry,
                    *callee_frame_base,
                    *caller_result_base,
                    *continuation,
                )?;
            }

            MachineTerminator::JumpTable { index, entries } => {
                if entries.is_empty() {
                    return Err(WasmError::internal(
                        "armv7a jump table requires at least one entry".into(),
                    ));
                }
                if entries.len() == 1 {
                    let label = self.core.emit_edge(entries[0].target, &entries[0].args)?;
                    self.emit_branch(BranchFixupKind::B, label);
                    return Ok(());
                }

                // Clamp index to entries.len()-1
                let index_hw = prepare_gp(&mut self.core.text, &self.gp_scratch, *index)?.detach();
                let max_idx = (entries.len() - 1) as u32;
                {
                    let clamp = self.gp_scratch.scoped_alloc();
                    emit_load_u32_into(&mut self.core.text, *clamp, max_idx);
                    self.core.text.emit_u32(enc::cmp_reg(*index_hw, *clamp));
                    // If index > max, use max (conditional move)
                    self.core
                        .text
                        .emit_u32(enc::mov_reg_cond(Cond::Hi, *index_hw, *clamp));
                }

                // Emit edge stubs and collect their labels
                let mut edge_label_ids = alloc::vec::Vec::with_capacity(entries.len());
                for entry in entries {
                    let label = self.core.emit_edge(entry.target, &entry.args)?;
                    edge_label_ids.push(label);
                }

                // ARM reads PC as current+8 for data-processing instructions, so
                // keep one 4-byte padding slot between the dispatch ADD and the
                // first branch-table entry.
                self.core.text.emit_u32(enc::add_reg_lsl_imm(
                    Arm32Reg::R15,
                    Arm32Reg::R15,
                    *index_hw,
                    2,
                ));
                self.core.text.emit_u32(enc::nop());

                // Emit branch table entries (will be patched by resolve_fixups)
                for &label_id in &edge_label_ids {
                    self.emit_branch(BranchFixupKind::B, label_id);
                }
            }
        }
        Ok(())
    }

    // ─── Branch condition compilation ───────────────────────────────────────────

    pub(super) fn compile_branch_condition(
        &mut self,
        cond: &MachineBranchCond,
    ) -> Result<Cond, WasmError> {
        match cond {
            MachineBranchCond::Value(value) => {
                // Branch taken if value != 0
                let hw = prepare_gp(&mut self.core.text, &self.gp_scratch, *value)?.detach();
                self.core.text.emit_u32(enc::cmp_imm(*hw, 0, 0));
                Ok(Cond::Ne)
            }

            MachineBranchCond::IntCompare {
                width: _,
                kind,
                sign,
                lhs,
                rhs,
            } => {
                let lhs_gp = prepare_gp(&mut self.core.text, &self.gp_scratch, *lhs)?;
                let lhs_hw = *lhs_gp;

                match rhs {
                    MachineValue::Reg(r) => {
                        self.core.text.emit_u32(enc::cmp_reg(lhs_hw, map_reg(*r)?));
                    }
                    MachineValue::Imm64(v) => {
                        if let Some((imm8, rot)) = enc::encode_arm_imm(*v as u32) {
                            self.core.text.emit_u32(enc::cmp_imm(lhs_hw, imm8, rot));
                        } else {
                            let s = self.gp_scratch.scoped_alloc();
                            emit_load_u32_into(&mut self.core.text, *s, *v as u32);
                            self.core.text.emit_u32(enc::cmp_reg(lhs_hw, *s));
                        }
                    }
                    MachineValue::ReservedReg(reg) => {
                        return Err(WasmError::internal(alloc::format!(
                            "armv7a branch IntCompare cannot read reserved cache register {} as rhs",
                            reg.0
                        )));
                    }
                }
                drop(lhs_gp);

                Ok(match (kind, sign) {
                    (MachineCompareKind::Eq, _) => Cond::Eq,
                    (MachineCompareKind::Ne, _) => Cond::Ne,
                    (MachineCompareKind::Lt, MachineSign::Signed) => Cond::Lt,
                    (MachineCompareKind::Lt, MachineSign::Unsigned) => Cond::Cc,
                    (MachineCompareKind::Gt, MachineSign::Signed) => Cond::Gt,
                    (MachineCompareKind::Gt, MachineSign::Unsigned) => Cond::Hi,
                    (MachineCompareKind::Le, MachineSign::Signed) => Cond::Le,
                    (MachineCompareKind::Le, MachineSign::Unsigned) => Cond::Ls,
                    (MachineCompareKind::Ge, MachineSign::Signed) => Cond::Ge,
                    (MachineCompareKind::Ge, MachineSign::Unsigned) => Cond::Cs,
                })
            }

            MachineBranchCond::TestBits {
                kind, src, mask, ..
            } => {
                let src_hw = match src {
                    MachineValue::Reg(r) => map_reg(*r)?,
                    MachineValue::Imm64(v) => {
                        let s = self.gp_scratch.scoped_alloc();
                        emit_load_u32_into(&mut self.core.text, *s, *v as u32);
                        let hw = *s;
                        drop(s);
                        hw
                    }
                    MachineValue::ReservedReg(reg) => {
                        return Err(WasmError::internal(alloc::format!(
                            "armv7a branch TestBits cannot read reserved cache register {} as src",
                            reg.0
                        )));
                    }
                };
                match mask {
                    MachineValue::Reg(r) => {
                        self.core.text.emit_u32(enc::tst_reg(src_hw, map_reg(*r)?));
                    }
                    MachineValue::Imm64(v) => {
                        if let Some((imm8, rot)) = enc::encode_arm_imm(*v as u32) {
                            self.core.text.emit_u32(enc::tst_imm(src_hw, imm8, rot));
                        } else {
                            let s = self.gp_scratch.scoped_alloc();
                            let tmp = *s;
                            emit_load_u32_into(&mut self.core.text, tmp, *v as u32);
                            self.core.text.emit_u32(enc::tst_reg(src_hw, tmp));
                        }
                    }
                    MachineValue::ReservedReg(reg) => {
                        return Err(WasmError::internal(alloc::format!(
                            "armv7a branch TestBits cannot read reserved cache register {} as mask",
                            reg.0
                        )));
                    }
                }
                Ok(match kind {
                    MachineCompareKind::Eq => Cond::Eq,
                    MachineCompareKind::Ne => Cond::Ne,
                    _ => {
                        return Err(WasmError::internal(alloc::format!(
                            "TestBits branch: unsupported compare kind {:?}",
                            kind
                        )));
                    }
                })
            }
        }
    }
}
