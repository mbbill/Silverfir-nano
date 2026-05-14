//! Terminator and branch-condition compilation for the 32-bit ARM backend.

use crate::collections;

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
};
use crate::vm::arch::common::template::{
    decode_template_chain_next, encode_template_chain_next, template_i32_delta, TemplateBranchSense,
};

// Only the A32 br_table dispatch path references `Arm32Reg` directly (for
// the `ADD PC, PC, Rm, LSL #2` PC-relative jump). On Thumb-2 builds that
// path is cfg-branched out in favor of a `cmp + b.eq` chain, so the import
// would be unused.
#[cfg(not(sf_arm32_isa_thumb))]
use super::reg::Arm32Reg;

fn checked_arm32_template_branch_delta(
    delta: i32,
    context: &'static str,
) -> Result<i32, WasmError> {
    if delta & 0b11 != 0 {
        return Err(WasmError::internal(context));
    }
    #[cfg(sf_arm32_isa_thumb)]
    let in_range = (-16_777_216..=16_777_214).contains(&delta);
    #[cfg(not(sf_arm32_isa_thumb))]
    let in_range = (-16_777_216..=16_777_214).contains(&delta);
    if !in_range {
        return Err(WasmError::internal(context));
    }
    Ok(delta)
}

// ─── Terminator compilation ─────────────────────────────────────────────────

impl<'a> Arm32Backend<'a> {
    pub(super) fn lower_terminator_dispatch(
        &mut self,
        terminator: &MachineTerminator,
        _fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        match terminator {
            MachineTerminator::Return | MachineTerminator::ReturnScalar { .. } => {
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

            MachineTerminator::Call {
                target,
                frame_delta,
                args,
                results,
                success,
            } => {
                self.emit_call(target, *frame_delta, args, results, success)?;
            }

            MachineTerminator::TailCall { target, args } => {
                self.emit_tail_call(target, args)?;
            }

            MachineTerminator::JumpTable { index, entries } => {
                if entries.is_empty() {
                    return Err(WasmError::internal(
                        "arm32 jump table requires at least one entry".into(),
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
                    super::inst::emit_mov_reg_cond_into(
                        &mut self.core.text,
                        Cond::Hi,
                        *index_hw,
                        *clamp,
                    );
                }

                // Emit edge stubs and collect their labels
                let mut edge_label_ids = collections::Vec::with_capacity(entries.len());
                for entry in entries {
                    let label = self.core.emit_edge(entry.target, &entry.args)?;
                    edge_label_ids.push(label);
                }

                // Dispatch. A32 uses a PC-relative branch table:
                // `ADD PC, PC, Rindex, LSL #2` (A32 reads PC as instr+8, so
                // leave one NOP pad between the ADD and the first table slot).
                // Thumb-2 forbids `ADD PC, PC, Rm, LSL #N` (T3 is UNPREDICTABLE
                // when Rd=PC and shift != 0), so we use a compare-and-branch
                // chain instead. O(N) vs A32's O(1), fine for wasm br_tables
                // which are almost always small; the clamp above still bounds
                // `index` to 0..=N-1 so the final branch can be unconditional.
                #[cfg(not(sf_arm32_isa_thumb))]
                {
                    self.core.text.emit_u32(enc::add_reg_lsl_imm(
                        Arm32Reg::R15,
                        Arm32Reg::R15,
                        *index_hw,
                        2,
                    ));
                    self.core.text.emit_u32(enc::nop());
                    for &label_id in &edge_label_ids {
                        self.emit_branch(BranchFixupKind::B, label_id);
                    }
                }
                #[cfg(sf_arm32_isa_thumb)]
                {
                    // Clamp guarantees index < len, so dispatch via
                    // `cmp + beq` for each entry except the last, which
                    // falls through to an unconditional branch. For
                    // indices beyond the modified-immediate range, go
                    // through MOVW/MOVT + cmp_reg.
                    let last = edge_label_ids.len() - 1;
                    for (i, &label_id) in edge_label_ids.iter().enumerate() {
                        if i == last {
                            self.emit_branch(BranchFixupKind::B, label_id);
                        } else {
                            let idx = i as u32;
                            if let Some((imm8, rot)) = enc::encode_arm_imm(idx) {
                                self.core.text.emit_u32(enc::cmp_imm(*index_hw, imm8, rot));
                            } else {
                                let s = self.gp_scratch.scoped_alloc();
                                super::inst::emit_load_u32_into(&mut self.core.text, *s, idx);
                                self.core.text.emit_u32(enc::cmp_reg(*index_hw, *s));
                            }
                            self.emit_branch(BranchFixupKind::BCond(Cond::Eq), label_id);
                        }
                    }
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
                    MachineValue::ReservedReg(_reg) => {
                        return Err(WasmError::internal(
                            "arm32 branch IntCompare cannot read reserved cache register as rhs",
                        ));
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
                    MachineValue::ReservedReg(_reg) => {
                        return Err(WasmError::internal(
                            "arm32 branch TestBits cannot read reserved cache register as src",
                        ));
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
                    MachineValue::ReservedReg(_reg) => {
                        return Err(WasmError::internal(
                            "arm32 branch TestBits cannot read reserved cache register as mask",
                        ));
                    }
                }
                Ok(match kind {
                    MachineCompareKind::Eq => Cond::Eq,
                    MachineCompareKind::Ne => Cond::Ne,
                    _ => {
                        return Err(WasmError::internal(
                            "TestBits branch: unsupported compare kind",
                        ));
                    }
                })
            }
        }
    }

    pub(crate) fn emit_template_skip_unless(
        &mut self,
        cond: &MachineBranchCond,
        jump_when: TemplateBranchSense,
        skip_bytes: usize,
    ) -> Result<(), WasmError> {
        let arm_cond = self.compile_branch_condition(cond)?;
        let skip_cond = match jump_when {
            TemplateBranchSense::IfTrue => arm_cond.invert(),
            TemplateBranchSense::IfFalse => arm_cond,
        };
        let target = self
            .core
            .text
            .len()
            .checked_add(4)
            .and_then(|offset| offset.checked_add(skip_bytes))
            .ok_or_else(|| WasmError::internal("arm32 template skip offset overflow"))?;
        let delta = checked_arm32_template_branch_delta(
            template_i32_delta(self.core.text.len(), target)?,
            "arm32 template skip branch out of range",
        )?;
        self.core.text.emit_u32(enc::b_cond(skip_cond, delta));
        Ok(())
    }

    pub(crate) fn emit_template_jump_placeholder(
        &mut self,
        next: usize,
    ) -> Result<usize, WasmError> {
        Ok(self.core.text.emit_u32(encode_template_chain_next(next)?))
    }

    pub(crate) fn read_template_jump_next(&self, site: usize) -> Result<usize, WasmError> {
        Ok(decode_template_chain_next(self.core.text.read_u32(site)))
    }

    pub(crate) fn patch_template_jump(
        &mut self,
        site: usize,
        target: usize,
    ) -> Result<(), WasmError> {
        let delta = checked_arm32_template_branch_delta(
            template_i32_delta(site, target)?,
            "arm32 template jump branch out of range",
        )?;
        self.core.text.patch_u32(site, enc::b(delta));
        Ok(())
    }

    pub(crate) fn emit_template_jump_to_offset(&mut self, target: usize) -> Result<(), WasmError> {
        let site = self.core.text.len();
        let delta = checked_arm32_template_branch_delta(
            template_i32_delta(site, target)?,
            "arm32 template jump branch out of range",
        )?;
        self.core.text.emit_u32(enc::b(delta));
        Ok(())
    }
}
