//! ARM64 terminator emission: branches, calls, traps, jump tables.

use crate::error::WasmError;
use crate::vm::machine::machine_ir::{
    MachineBranchCond, MachineBlockId, MachineCompareKind, MachineConstId,
    MachineEdge, MachineFuncId, MachineReg,
    MachineTerminator, MachineTrapKind, MachineValue,
    MACHINE_CTX_REG, MACHINE_FP_REG,
};

use super::{abi, enc};
use super::abi::map_fixed_reg;
use super::inst::{materialize_u64_into, prepare_gp};
use crate::vm::arch::common::helpers::{is_fallthrough_edge, trap_code};
use crate::vm::arch::common::types::{DirectCallPatch, LocalPtrPatch, PendingLocalPtrPatch};
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
            call_link_base,
            continuation,
        } => self.lower_call_direct(*callee, *callee_frame_base, *call_link_base, *continuation),
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
                _ => return Err(WasmError::internal(
                    alloc::format!("TestBits branch: unsupported compare kind {:?}", kind),
                )),
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
                _ => return Err(WasmError::internal(
                    alloc::format!("TestBits branch_if: unsupported compare kind {:?}", kind),
                )),
            };
            self.lower_b_cond(cond, trap_label);
        }
    }
    Ok(())
}

// ── Direct call ──────────────────────────────────────────────────────────────

fn lower_call_direct(&mut self,
    callee: MachineFuncId,
    callee_frame_base: MachineReg,
    call_link_base: MachineReg,
    continuation: MachineBlockId,
) -> Result<(), WasmError> {
    let callee_fp = self.map_gp_reg(callee_frame_base)?;
    let call_link_base = self.map_gp_reg(call_link_base)?;
    let continuation_offset = self.core.compiled.abi().call_link.continuation_offset;
    let continuation_label = self.core.block_label(continuation)?;

    let s0_idx = self.gp_scratch.alloc();
    let s1_idx = self.gp_scratch.alloc();
    let s0 = self.gp_scratch.reg(s0_idx);
    let s1 = self.gp_scratch.reg(s1_idx);

    // Load the native continuation address from a patchable literal and write
    // it into the callee's call-link record. MachineIR already chose the
    // record location; the backend only fills in the native return address.
    let continuation_load = self.core.text.emit_u32(enc::ldr_lit_64(s0, 0));
    if continuation_offset < 4096 {
        self.core
            .text
            .emit_u32(enc::str_64(s0, call_link_base, continuation_offset as u32));
    } else {
        self.materialize_u64(s1, continuation_offset as u64);
        self.core
            .text
            .emit_u32(enc::add_reg_64(s1, call_link_base, s1));
        self.core.text.emit_u32(enc::str_reg_64_base(s0, s1));
    }

    // Direct local calls still use a patchable literal for the callee entry:
    // the final native address is not known until the whole module has been
    // laid out by the common pipeline.
    let callee_load = self.core.text.emit_u32(enc::ldr_lit_64(s0, 0));
    self.core.text.emit_u32(enc::mov_reg_64(map_fixed_reg(MACHINE_FP_REG), callee_fp));
    self.core.text.emit_u32(enc::br(s0));

    // Reserve the literal words now, then record how they should be patched
    // once block labels and final function entry addresses are known.
    let continuation_literal = self.core.text.emit_u64(0);
    let callee_literal = self.core.text.emit_u64(0);

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
    self.gp_scratch.free_index(s1_idx);
    self.gp_scratch.free_index(s0_idx);
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
    // Keep C-ABI argument registers out of normal control lowering. `s1`
    // holds the clamped jump-table index first, then the scaled byte offset.
    self.materialize_u64(s1, (entries.len() - 1) as u64);
    self.core.text.emit_u32(enc::cmp_reg_64(index_reg, s1));
    self.core.text.emit_u32(enc::csel_64(s1, index_reg, s1, enc::Cond::Ls));

    let table_base_load = self.core.text.emit_u32(enc::ldr_lit_64(s0, 0));
    self.core.text.emit_u32(enc::lsl_imm_64(s1, s1, 3));
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
    let call_link = self.core.compiled.abi().call_link;
    let continuation_slot = call_scratch.base_slot + (call_link.continuation_offset / 8) as u16;
    let caller_frame_slot = call_scratch.base_slot + (call_link.caller_frame_offset / 8) as u16;
    let caller_result_base_slot =
        call_scratch.base_slot + (call_link.caller_result_base_offset / 8) as u16;

    let s0_idx = self.gp_scratch.alloc();
    let s1_idx = self.gp_scratch.alloc();
    let s0 = self.gp_scratch.reg(s0_idx);
    let s1 = self.gp_scratch.reg(s1_idx);

    self.core.text.emit_u32(enc::ldr_64(s0, map_fixed_reg(MACHINE_FP_REG), continuation_slot as u32));
    self.core.text.emit_u32(enc::ldr_64(s1, map_fixed_reg(MACHINE_FP_REG), caller_frame_slot as u32));
    self.core.text.emit_u32(enc::ldr_64(abi::C_ARG0, map_fixed_reg(MACHINE_FP_REG), caller_result_base_slot as u32));
    self.core.text.emit_u32(enc::add_reg_64(abi::C_ARG0, s1, abi::C_ARG0));

    if let Some(results) = runtime.return_results {
        // Results live in the callee frame until return. Copy them back into
        // the caller's result window before restoring the caller frame.
        for index in 0..results.slots as u32 {
            self.core.text.emit_u32(enc::ldr_64(abi::C_ARG1, map_fixed_reg(MACHINE_FP_REG), results.base_slot as u32 + index));
            self.core.text.emit_u32(enc::str_64(abi::C_ARG1, abi::C_ARG0, index));
        }
    }

    // Restore caller FP and branch to the continuation pointer saved in the
    // call-link record.
    self.core.text.emit_u32(enc::mov_reg_64(map_fixed_reg(MACHINE_FP_REG), s1));
    self.core.text.emit_u32(enc::br(s0));
    self.gp_scratch.free_index(s1_idx);
    self.gp_scratch.free_index(s0_idx);
    Ok(())
}

// ── Indirect call ────────────────────────────────────────────────────────────

fn lower_call_indirect(&mut self,
    _callee_target: MachineReg,
    callee_entry: MachineReg,
    callee_frame_base: MachineReg,
    call_link_base: MachineReg,
    continuation: MachineBlockId,
) -> Result<(), WasmError> {
    let callee_fp = self.map_gp_reg(callee_frame_base)?;
    let call_link_base = self.map_gp_reg(call_link_base)?;
    let continuation_offset = self.core.compiled.abi().call_link.continuation_offset;
    let continuation_label = self.core.block_label(continuation)?;
    let callee_entry = self.map_gp_reg(callee_entry)?;
    let s0_idx = self.gp_scratch.alloc();
    let s0 = self.gp_scratch.reg(s0_idx);

    // For indirect local calls the callee entry is already a runtime register,
    // but the continuation is still a backend-resolved block label. Materialize
    // that continuation address through a local literal so we can store it in
    // the prepared call-link record.
    let continuation_load = self.core.text.emit_u32(enc::ldr_lit_64(s0, 0));
    let skip_cont_literal = self.core.text.emit_u32(enc::b(0)); // skip over literal
    let continuation_literal = self.core.text.emit_u64(0);
    let after_cont_literal = self.core.text.len();
    let skip_cont_delta =
        ((after_cont_literal as isize - skip_cont_literal as isize) / 4) as i32;
    self.core
        .text
        .patch_u32(skip_cont_literal, enc::b(skip_cont_delta));
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

    if continuation_offset == 0 {
        self.core
            .text
            .emit_u32(enc::str_reg_64_base(s0, call_link_base));
    } else {
        let s1_idx = self.gp_scratch.alloc();
        let s1 = self.gp_scratch.reg(s1_idx);
        self.materialize_u64(s1, continuation_offset as u64);
        self.core
            .text
            .emit_u32(enc::add_reg_64(s1, call_link_base, s1));
        self.core.text.emit_u32(enc::str_reg_64_base(s0, s1));
        self.gp_scratch.free_index(s1_idx);
    }

    // MachineIR already resolved the dynamic local target to a native entry
    // address. At this point the backend only commits the transfer.
    self.core
        .text
        .emit_u32(enc::mov_reg_64(map_fixed_reg(MACHINE_FP_REG), callee_fp));
    self.core.text.emit_u32(enc::br(callee_entry));
    self.gp_scratch.free_index(s0_idx);
    Ok(())
}

// ── Call external ────────────────────────────────────────────────────────────

pub(super) fn lower_call_external(&mut self,
    const_idx: usize,
) -> Result<(), WasmError> {
    let metadata = self
        .core
        .compiled
        .const_ptr(MachineConstId(const_idx as u32))
        .ok_or_else(|| WasmError::internal("arm64 external-call metadata is out of range".into()))?;

    // External calls are inline runtime calls, not CFG terminators. Pass the
    // current context, the active Wasm frame pointer, and the constant-pool
    // metadata record that describes where args/results live in that frame.
    self.core.text.emit_u32(enc::mov_reg_64(
        abi::C_ARG0,
        map_fixed_reg(MACHINE_CTX_REG),
    ));
    self.core
        .text
        .emit_u32(enc::mov_reg_64(abi::C_ARG1, map_fixed_reg(MACHINE_FP_REG)));
    self.materialize_u64(abi::C_ARG2, metadata as u64);
    let call_scratch_idx = self.gp_scratch.alloc();
    let call_scratch = self.gp_scratch.reg(call_scratch_idx);
    self.materialize_u64(
        call_scratch,
        crate::vm::runtime::external::call_external_entry_ptr() as usize as u64,
    );
    self.core.text.emit_u32(enc::blr(call_scratch));
    self.gp_scratch.free_index(call_scratch_idx);

    // Nonzero helper status means the runtime stored a WasmError in the
    // NativeContext, so branch to the shared error-return path.
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
    let call_scratch_idx = self.gp_scratch.alloc();
    let call_scratch = self.gp_scratch.reg(call_scratch_idx);
    materialize_u64_into(
        &mut self.core.text,
        call_scratch,
        crate::vm::runtime::trap::raise_trap as u64,
    );
    self.core.text.emit_u32(enc::blr(call_scratch));
    self.gp_scratch.free_index(call_scratch_idx);
    // Branch to the shared error-return epilogue
    let return_error_label = self.core.return_error_label;
    self.lower_b(return_error_label);
}
}
