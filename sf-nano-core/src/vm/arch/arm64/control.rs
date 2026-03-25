//! ARM64 terminator emission: branches, calls, traps, jump tables.

use crate::error::WasmError;
use crate::vm::machine::machine_ir::{
    MachineBranchCond, MachineBlockId, MachineConstId,
    MachineEdge, MachineFloatWidth, MachineFuncId, MachineReg,
    MachineTerminator, MachineTrapKind, MachineValue,
    MACHINE_CTX_REG, MACHINE_FP_REG,
};

use super::{abi, enc, reg::Arm64Reg};
use super::abi::map_fixed_reg;
use super::inst::{materialize_u64_into, prepare_gp};
use crate::vm::arch::common::helpers::{is_fallthrough_edge, trap_code};
use crate::vm::arch::common::types::{DirectCallPatch, PendingLocalPtrPatch, LocalPtrPatch};
use crate::vm::runtime::context::ctx_offset;

use super::fusion::map_int_cond;

impl<'a> super::backend::Arm64Backend<'a> {

// ── Main terminator dispatch ─────────────────────────────────────────────────

/// Main terminator dispatch -- called by `ArchBackend::emit_terminator`.
pub(super) fn lower_terminator_dispatch(&mut self,
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
        MachineTerminator::CallDirect {
            callee,
            callee_frame_base,
            continuation,
        } => self.lower_call_direct(*callee, *callee_frame_base, *continuation),
        MachineTerminator::CallIndirect {
            callee_target,
            callee_frame_base,
            arg_slots,
            caller_result_base,
            continuation,
        } => self.lower_call_indirect(
            *callee_target,
            *callee_frame_base,
            *arg_slots,
            *caller_result_base,
            *continuation,
        ),
    }
}

// ── Branch ───────────────────────────────────────────────────────────────────

fn lower_branch(&mut self,
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
        },
        MachineBranchCond::IntCompare {
            width,
            kind,
            sign,
            lhs,
            rhs,
        } => {
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
    Ok(())
}

// ── Fused conditional branch (float compare-branch fusion) ───────────────────

/// Emit a conditional branch when the CPU flags have already been set
/// by a preceding CMP/FCMP.
pub(super) fn lower_fused_cond_branch(&mut self,
    cond: enc::Cond,
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
    Ok(())
}

// ── FCMP (for float compare-and-branch fusion) ──────────────────────────────

/// Emit FCMP without CSET (for float compare-and-branch fusion).
pub(super) fn lower_fcmp_values(&mut self,
    width: MachineFloatWidth,
    lhs: MachineValue,
    rhs: MachineValue,
) -> Result<(), WasmError> {
    let lhs_fp = super::inst::prepare_fp(
        self.core.compiled.backend(), &self.core.fp_reg_widths,
        &mut self.core.text, &self.gp_scratch, &self.fp_scratch,
        width, lhs,
    )?;
    if matches!(rhs, MachineValue::Imm64(0)) {
        match width {
            MachineFloatWidth::F32 => self.core.text.emit_u32(enc::fcmp_s_zero(lhs_fp.reg())),
            MachineFloatWidth::F64 => self.core.text.emit_u32(enc::fcmp_d_zero(lhs_fp.reg())),
        };
    } else {
        let rhs_fp = super::inst::prepare_fp(
            self.core.compiled.backend(), &self.core.fp_reg_widths,
            &mut self.core.text, &self.gp_scratch, &self.fp_scratch,
            width, rhs,
        )?;
        match width {
            MachineFloatWidth::F32 => self.core.text.emit_u32(enc::fcmp_s(lhs_fp.reg(), rhs_fp.reg())),
            MachineFloatWidth::F64 => self.core.text.emit_u32(enc::fcmp_d(lhs_fp.reg(), rhs_fp.reg())),
        };
    }
    Ok(())
}

// ── Trap-if ──────────────────────────────────────────────────────────────────

pub(super) fn lower_trap_if(&mut self,
    kind: MachineTrapKind,
    cond: &MachineBranchCond,
) -> Result<(), WasmError> {
    let trap_label = self.core.ensure_trap_label(kind);
    self.lower_branch_if(cond, trap_label)
}

pub(super) fn lower_branch_if(&mut self,
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
    }
    Ok(())
}

// ── Direct call ──────────────────────────────────────────────────────────────

fn lower_call_direct(&mut self,
    callee: MachineFuncId,
    callee_frame_base: MachineReg,
    continuation: MachineBlockId,
) -> Result<(), WasmError> {
    let callee_runtime = self.runtime_for(callee)?;
    let call_scratch = callee_runtime.call_scratch.ok_or_else(|| {
        WasmError::internal("arm64 direct local call requires callee call scratch".into())
    })?;
    let continuation_slot = call_scratch.base_slot
        + (self.core.compiled.runtime().call_link.continuation_offset / 8) as u16;
    let callee_fp = self.map_gp_reg(callee_frame_base)?;

    let s0 = self.gp_scratch.scoped_alloc().release();
    let s1 = self.gp_scratch.scoped_alloc().release();

    let continuation_load = self.core.text.emit_u32(enc::ldr_lit_64(s0, 0));
    if continuation_slot < 4096 {
        self.core.text.emit_u32(enc::str_64(s0, callee_fp, continuation_slot as u32));
    } else {
        self.materialize_u64(s1, u64::from(continuation_slot) * 8);
        self.core.text.emit_u32(enc::add_reg_64(s1, callee_fp, s1));
        self.core.text.emit_u32(enc::str_reg_64(s0, s1, Arm64Reg::Xzr));
    }

    let callee_load = self.core.text.emit_u32(enc::ldr_lit_64(s0, 0));
    self.core.text.emit_u32(enc::mov_reg_64(map_fixed_reg(MACHINE_FP_REG), callee_fp));
    self.core.text.emit_u32(enc::br(s0));

    let continuation_literal = self.core.text.emit_u64(0);
    let callee_literal = self.core.text.emit_u64(0);

    let continuation_label = self.core.block_label(continuation)?;
    let continuation_delta =
        ((continuation_literal as isize - continuation_load as isize) / 4) as i32;
    self.core.text.patch_u32(continuation_load, enc::ldr_lit_64(s0, continuation_delta));
    let callee_delta = ((callee_literal as isize - callee_load as isize) / 4) as i32;
    self.core.text.patch_u32(callee_load, enc::ldr_lit_64(s0, callee_delta));

    self.core.local_ptr_patches.push(PendingLocalPtrPatch {
        literal_offset: continuation_literal,
        target_label: continuation_label,
    });
    self.core.direct_call_patches.push(DirectCallPatch {
        literal_offset: callee_literal,
        callee,
    });
    Ok(())
}

// ── Jump table (br_table) ────────────────────────────────────────────────────

fn lower_jump_table(&mut self,
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

    let s0 = self.gp_scratch.scoped_alloc().release();
    let s1 = self.gp_scratch.scoped_alloc().release();
    let index_reg = prepare_gp(
        self.core.compiled.backend(), &self.core.fp_reg_widths,
        &mut self.core.text, &self.gp_scratch, index,
    )?.release();
    self.materialize_u64(abi::C_ARG0, (entries.len() - 1) as u64);
    self.core.text.emit_u32(enc::cmp_reg_64(index_reg, abi::C_ARG0));
    self.core.text.emit_u32(enc::csel_64(s1, index_reg, abi::C_ARG0, enc::Cond::Ls));

    let table_base_load = self.core.text.emit_u32(enc::ldr_lit_64(s0, 0));
    self.materialize_u64(abi::C_ARG0, 3);
    self.core.text.emit_u32(enc::lslv_64(s1, s1, abi::C_ARG0));
    self.core.text.emit_u32(enc::ldr_reg_64(s0, s0, s1));
    self.core.text.emit_u32(enc::br(s0));

    let table_base_literal = self.core.text.emit_u64(0);
    let table_offset = self.core.text.len();
    let table_base_delta =
        ((table_base_literal as isize - table_base_load as isize) / 4) as i32;
    self.core.text.patch_u32(table_base_load, enc::ldr_lit_64(s0, table_base_delta));
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

fn lower_return_sequence(&mut self) -> Result<(), WasmError> {
    let runtime = *self.runtime_for(self.core.function.id)?;
    let call_scratch = runtime.call_scratch.ok_or_else(|| {
        WasmError::internal("arm64 local return requires call scratch".into())
    })?;
    let call_link = self.core.compiled.runtime().call_link;
    let continuation_slot = call_scratch.base_slot + (call_link.continuation_offset / 8) as u16;
    let caller_frame_slot = call_scratch.base_slot + (call_link.caller_frame_offset / 8) as u16;
    let caller_result_base_slot =
        call_scratch.base_slot + (call_link.caller_result_base_offset / 8) as u16;

    let s0 = self.gp_scratch.scoped_alloc().release();
    let s1 = self.gp_scratch.scoped_alloc().release();

    self.core.text.emit_u32(enc::ldr_64(s0, map_fixed_reg(MACHINE_FP_REG), continuation_slot as u32));
    self.core.text.emit_u32(enc::ldr_64(s1, map_fixed_reg(MACHINE_FP_REG), caller_frame_slot as u32));
    self.core.text.emit_u32(enc::ldr_64(abi::C_ARG0, map_fixed_reg(MACHINE_FP_REG), caller_result_base_slot as u32));
    self.core.text.emit_u32(enc::add_reg_64(abi::C_ARG0, s1, abi::C_ARG0));

    if let Some(results) = runtime.return_results {
        for index in 0..results.slots as u32 {
            self.core.text.emit_u32(enc::ldr_64(abi::C_ARG1, map_fixed_reg(MACHINE_FP_REG), results.base_slot as u32 + index));
            self.core.text.emit_u32(enc::str_64(abi::C_ARG1, abi::C_ARG0, index));
        }
    }

    self.core.text.emit_u32(enc::mov_reg_64(map_fixed_reg(MACHINE_FP_REG), s1));
    self.core.text.emit_u32(enc::br(s0));
    Ok(())
}

// ── Indirect call ────────────────────────────────────────────────────────────

fn lower_call_indirect(&mut self,
    callee_target: MachineValue,
    callee_frame_base: MachineReg,
    arg_slots: u16,
    caller_result_base: u16,
    continuation: MachineBlockId,
) -> Result<(), WasmError> {
    let s0 = self.gp_scratch.scoped_alloc().release();
    let s1 = self.gp_scratch.scoped_alloc().release();

    // Load the callee function id into a register
    let callee_id_reg = prepare_gp(
        self.core.compiled.backend(), &self.core.fp_reg_widths,
        &mut self.core.text, &self.gp_scratch, callee_target,
    )?.release();

    // Load function table base from a literal pool entry
    let table_base_load = self.core.text.emit_u32(enc::ldr_lit_64(s0, 0));
    let skip_table_literal = self.core.text.emit_u32(enc::b(0)); // skip over literal
    self.core.function_table_patches.push(self.core.text.emit_u64(0));
    let table_base_literal = self
        .core
        .function_table_patches
        .last()
        .copied()
        .expect("function table literal recorded");
    let after_table_literal = self.core.text.len();

    // Patch the skip branch
    let skip_delta = ((after_table_literal as isize - skip_table_literal as isize) / 4) as i32;
    self.core.text.patch_u32(skip_table_literal, enc::b(skip_delta));
    // Patch the ldr literal offset
    let table_base_delta =
        ((table_base_literal as isize - table_base_load as isize) / 4) as i32;
    self.core
        .text
        .patch_u32(table_base_load, enc::ldr_lit_64(s0, table_base_delta));

    // Index into the function table: each entry is 32 bytes (1 << 5)
    self.materialize_u64(abi::C_ARG0, 5);
    self.core
        .text
        .emit_u32(enc::lslv_64(s1, callee_id_reg, abi::C_ARG0));
    self.core
        .text
        .emit_u32(enc::add_reg_64(s0, s0, s1));

    // Load function info fields:
    // [0] = entry, [1] = total_frame_bytes, [2] = frame_prefix_slots, [3] = call_scratch_base_slot
    self.core.text.emit_u32(enc::ldr_64(abi::C_ARG0, s0, 0));
    self.core.text.emit_u32(enc::ldr_64(abi::C_ARG1, s0, 1));
    self.core.text.emit_u32(enc::ldr_64(abi::C_ARG2, s0, 2));
    self.core.text.emit_u32(enc::ldr_64(abi::C_ARG3, s0, 3));

    // Stack overflow check: callee_fp + total_frame_bytes > stack_end?
    let callee_fp = self.map_gp_reg(callee_frame_base)?;
    self.core
        .text
        .emit_u32(enc::add_reg_64(s0, callee_fp, abi::C_ARG1));
    self.core.text.emit_u32(enc::ldr_64(
        s1,
        map_fixed_reg(MACHINE_CTX_REG),
        (ctx_offset::STACK_END / 8) as u32,
    ));
    self.core.text.emit_u32(enc::cmp_reg_64(s0, s1));
    let stack_overflow_label = self.core.stack_overflow_label;
    self.lower_b_cond(enc::Cond::Hi, stack_overflow_label);

    // Zero-fill the dynamic callee prefix (between arg_slots and frame_prefix_slots)
    self.lower_zero_dynamic_callee_prefix(callee_fp, arg_slots)?;

    // Load continuation address from literal pool
    let continuation_load = self.core.text.emit_u32(enc::ldr_lit_64(s0, 0));
    let skip_cont_literal = self.core.text.emit_u32(enc::b(0)); // skip over literal
    let continuation_literal = self.core.text.emit_u64(0);
    let after_cont_literal = self.core.text.len();
    let skip_cont_delta =
        ((after_cont_literal as isize - skip_cont_literal as isize) / 4) as i32;
    self.core
        .text
        .patch_u32(skip_cont_literal, enc::b(skip_cont_delta));
    let continuation_label = self.core.block_label(continuation)?;
    let continuation_delta =
        ((continuation_literal as isize - continuation_load as isize) / 4) as i32;
    self.core.text.patch_u32(
        continuation_load,
        enc::ldr_lit_64(s0, continuation_delta),
    );
    self.core.local_ptr_patches.push(PendingLocalPtrPatch {
        literal_offset: continuation_literal,
        target_label: continuation_label,
    });

    // Store continuation, caller frame, and caller result base into call scratch area
    // C_ARG3 = call_scratch_base_slot; convert to byte offset and add to callee_fp
    self.materialize_u64(s1, 3);
    self.core
        .text
        .emit_u32(enc::lslv_64(abi::C_ARG3, abi::C_ARG3, s1));
    self.core
        .text
        .emit_u32(enc::add_reg_64(abi::C_ARG3, callee_fp, abi::C_ARG3));
    self.core
        .text
        .emit_u32(enc::str_reg_64(s0, abi::C_ARG3, Arm64Reg::Xzr));
    self.core
        .text
        .emit_u32(enc::str_64(map_fixed_reg(MACHINE_FP_REG), abi::C_ARG3, 1));
    self.materialize_u64(s1, u64::from(caller_result_base) * 8);
    self.core.text.emit_u32(enc::str_64(s1, abi::C_ARG3, 2));

    // Set new frame pointer and jump to callee entry
    self.core
        .text
        .emit_u32(enc::mov_reg_64(map_fixed_reg(MACHINE_FP_REG), callee_fp));
    self.core.text.emit_u32(enc::br(abi::C_ARG0));
    Ok(())
}

// ── Zero dynamic callee prefix ───────────────────────────────────────────────

fn lower_zero_dynamic_callee_prefix(&mut self,
    callee_fp: Arm64Reg,
    arg_slots: u16,
) -> Result<(), WasmError> {
    let s0 = self.gp_scratch.scoped_alloc().release();
    let s1 = self.gp_scratch.scoped_alloc().release();

    // s0 = callee_fp + arg_slots * 8 (start of prefix to zero)
    self.materialize_u64(s0, u64::from(arg_slots) * 8);
    self.core
        .text
        .emit_u32(enc::add_reg_64(s0, callee_fp, s0));
    // C_ARG2 = callee_fp + frame_prefix_slots * 8 (end of prefix)
    self.materialize_u64(s1, 3);
    self.core
        .text
        .emit_u32(enc::lslv_64(abi::C_ARG2, abi::C_ARG2, s1));
    self.core
        .text
        .emit_u32(enc::add_reg_64(abi::C_ARG2, callee_fp, abi::C_ARG2));
    self.core.text.emit_u32(enc::cmp_reg_64(s0, abi::C_ARG2));

    let done = self.core.new_label();
    let loop_label = self.core.new_label();
    self.lower_b_cond(enc::Cond::Hs, done);
    self.core.bind_label(loop_label);
    self.core
        .text
        .emit_u32(enc::str_reg_64(Arm64Reg::Xzr, s0, Arm64Reg::Xzr));
    self.core.text.emit_u32(enc::add_imm_64(s0, s0, 8));
    self.core.text.emit_u32(enc::cmp_reg_64(s0, abi::C_ARG2));
    self.lower_b_cond(enc::Cond::Lo, loop_label);
    self.core.bind_label(done);
    Ok(())
}

// ── Call helper ──────────────────────────────────────────────────────────────

pub(super) fn lower_call_helper(&mut self,
    extern_idx: usize,
    const_idx: usize,
) -> Result<(), WasmError> {
    let binding = self
        .core
        .compiled
        .module()
        .externs
        .get(extern_idx)
        .ok_or_else(|| WasmError::internal("arm64 helper target is out of range".into()))?;
    let metadata = self
        .core
        .compiled
        .const_ptr(MachineConstId(const_idx as u32))
        .ok_or_else(|| WasmError::internal("arm64 helper metadata is out of range".into()))?;

    // Set up arguments: x0 = ctx, x1 = fp, x2 = metadata pointer
    self.core.text.emit_u32(enc::mov_reg_64(
        abi::C_ARG0,
        map_fixed_reg(MACHINE_CTX_REG),
    ));
    self.core
        .text
        .emit_u32(enc::mov_reg_64(abi::C_ARG1, map_fixed_reg(MACHINE_FP_REG)));
    self.materialize_u64(abi::C_ARG2, metadata as u64);
    let call_scratch = self.gp_scratch.scoped_alloc().release();
    self.materialize_u64(
        call_scratch,
        crate::vm::runtime::helpers::resolve_helper_entry(binding.symbol) as usize as u64,
    );
    self.core.text.emit_u32(enc::blr(call_scratch));

    // If helper returned nonzero, branch to the error return path
    let return_error_label = self.core.return_error_label;
    self.lower_cbnz(abi::C_RET0, return_error_label);
    Ok(())
}

// ── Trap stub ────────────────────────────────────────────────────────────────

/// Emit a trap stub -- called by `ArchBackend::emit_trap`.
pub(super) fn lower_trap_dispatch(&mut self, kind: MachineTrapKind) {
    // Set up arguments: x0 = ctx, x1 = trap code
    self.core.text.emit_u32(enc::mov_reg_64(
        abi::C_ARG0,
        map_fixed_reg(MACHINE_CTX_REG),
    ));
    materialize_u64_into(&mut self.core.text, abi::C_ARG1, trap_code(kind));
    let call_scratch = self.gp_scratch.scoped_alloc().release();
    materialize_u64_into(
        &mut self.core.text,
        call_scratch,
        super::helpers::arm64_raise_trap as u64,
    );
    self.core.text.emit_u32(enc::blr(call_scratch));
    // Branch to the shared error-return epilogue
    let return_error_label = self.core.return_error_label;
    self.lower_b(return_error_label);
}
}
