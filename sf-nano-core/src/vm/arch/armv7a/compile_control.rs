//! Terminator and branch-condition compilation for the ARMv7-A backend.

use crate::{
    error::WasmError,
    vm::machine::machine_ir::{
        MachineBranchCond, MachineCompareKind, MachineFloatWidth, MachineSign, MachineTerminator,
        MachineValue,
    },
};

use super::{
    abi::{emit_shared_epilogue, map_fixed_reg, map_reg, FP_SCRATCH1, FP_SCRATCH2, SCRATCH0, SCRATCH1},
    armv7a_raise_trap,
    enc::{self, Cond},
    reg::Arm32Reg,
};

use crate::vm::machine::machine_ir::MACHINE_CTX_REG;
use super::compile::{BranchFixupKind, FunctionCompiler, LabelKind};

use super::compile_helpers::{
    emit_host_call, materialize_float_value_dreg, trap_kind_to_u32,
};

// ─── Terminator compilation ─────────────────────────────────────────────────

pub(super) fn compile_terminator(
    fc: &mut FunctionCompiler<'_>,
    terminator: &MachineTerminator,
) -> Result<(), WasmError> {
    match terminator {
        MachineTerminator::Return => {
            fc.emit_return_sequence()?;
        }

        MachineTerminator::Jump(edge) => {
            let label = fc.emit_edge(edge.target, &edge.args)?;
            fc.emit_branch(BranchFixupKind::B, label);
        }

        MachineTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => {
            let then_label = fc.emit_edge(then_edge.target, &then_edge.args)?;
            let else_label = fc.emit_edge(else_edge.target, &else_edge.args)?;

            let arm_cond = compile_branch_condition(fc, cond)?;
            fc.emit_branch(BranchFixupKind::BCond(arm_cond), then_label);
            fc.emit_branch(BranchFixupKind::B, else_label);
        }

        MachineTerminator::Trap { kind } => {
            let return_error_label = fc.return_error_label;
            fc.text
                .emit_u32(enc::mov_reg(Arm32Reg::R0, map_fixed_reg(MACHINE_CTX_REG)));
            let trap_code = trap_kind_to_u32(*kind);
            fc.emit_load_u32(Arm32Reg::R1, trap_code);
            fc.emit_load_u32(Arm32Reg::R2, fc.current_trap_site());
            emit_host_call(fc, armv7a_raise_trap as usize);
            fc.emit_branch(BranchFixupKind::B, return_error_label);
        }

        MachineTerminator::CallDirect {
            callee,
            callee_frame_base,
            continuation,
        } => {
            fc.emit_call_direct(*callee, *callee_frame_base, *continuation)?;
        }

        MachineTerminator::CallIndirect {
            callee_target,
            callee_frame_base,
            arg_slots,
            caller_result_base,
            continuation,
        } => {
            fc.emit_call_indirect(
                *callee_target,
                *callee_frame_base,
                *arg_slots,
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
                let label = fc.emit_edge(entries[0].target, &entries[0].args)?;
                fc.emit_branch(BranchFixupKind::B, label);
                return Ok(());
            }

            // Clamp index to entries.len()-1
            let index_hw = match index {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            let max_idx = (entries.len() - 1) as u32;
            let clamp_hw = if index_hw == SCRATCH0 {
                SCRATCH1
            } else {
                SCRATCH0
            };
            fc.emit_load_u32(clamp_hw, max_idx);
            fc.text.emit_u32(enc::cmp_reg(index_hw, clamp_hw));
            // If index > max, use max (conditional move)
            fc.text
                .emit_u32(enc::mov_reg_cond(Cond::Hi, index_hw, clamp_hw));

            // Emit edge stubs and collect their labels
            let mut edge_label_ids = alloc::vec::Vec::with_capacity(entries.len());
            for entry in entries {
                let label = fc.emit_edge(entry.target, &entry.args)?;
                edge_label_ids.push(label);
            }

            // ARM reads PC as current+8 for data-processing instructions, so
            // keep one 4-byte padding slot between the dispatch ADD and the
            // first branch-table entry.
            fc.text.emit_u32(enc::add_reg_lsl_imm(
                Arm32Reg::R15,
                Arm32Reg::R15,
                index_hw,
                2,
            ));
            fc.text.emit_u32(enc::nop());

            // Emit branch table entries (will be patched by resolve_fixups)
            for &label_id in &edge_label_ids {
                fc.emit_branch(BranchFixupKind::B, label_id);
            }
        }
    }
    Ok(())
}

// ─── Branch condition compilation ───────────────────────────────────────────

pub(super) fn compile_branch_condition(
    fc: &mut FunctionCompiler<'_>,
    cond: &MachineBranchCond,
) -> Result<Cond, WasmError> {
    match cond {
        MachineBranchCond::Value(value) => {
            // Branch taken if value != 0
            let hw = match value {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            fc.text.emit_u32(enc::cmp_imm(hw, 0, 0));
            Ok(Cond::Ne)
        }

        MachineBranchCond::IntCompare {
            width,
            kind,
            sign,
            lhs,
            rhs,
        } => {
            let lhs_hw = match lhs {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };

            match rhs {
                MachineValue::Reg(r) => {
                    fc.text.emit_u32(enc::cmp_reg(lhs_hw, map_reg(*r)?));
                }
                MachineValue::Imm64(v) => {
                    if let Some((imm8, rot)) = enc::encode_arm_imm(*v as u32) {
                        fc.text.emit_u32(enc::cmp_imm(lhs_hw, imm8, rot));
                    } else {
                        let tmp = if lhs_hw == SCRATCH0 {
                            Arm32Reg::R3
                        } else {
                            SCRATCH0
                        };
                        fc.emit_load_u32(tmp, *v as u32);
                        fc.text.emit_u32(enc::cmp_reg(lhs_hw, tmp));
                    }
                }
            }

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

        MachineBranchCond::FloatCompare {
            width,
            kind,
            lhs,
            rhs,
        } => {
            let lhs_d = materialize_float_value_dreg(fc, *width, lhs, FP_SCRATCH1)?;
            let rhs_d = materialize_float_value_dreg(fc, *width, rhs, FP_SCRATCH2)?;

            match width {
                MachineFloatWidth::F64 => {
                    fc.text.emit_u32(enc::vcmp_d(lhs_d, rhs_d));
                }
                MachineFloatWidth::F32 => {
                    fc.text.emit_u32(enc::vcmp_s(lhs_d * 2, rhs_d * 2));
                }
            }
            fc.text.emit_u32(enc::vmrs_apsr());

            Ok(match kind {
                MachineCompareKind::Eq => Cond::Eq,
                MachineCompareKind::Ne => Cond::Ne,
                MachineCompareKind::Lt => Cond::Mi,
                MachineCompareKind::Gt => Cond::Gt,
                MachineCompareKind::Le => Cond::Ls,
                MachineCompareKind::Ge => Cond::Ge,
            })
        }
    }
}
