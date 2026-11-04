//! `memory.copy` / `memory.fill` on a 32-bit memory 0 as a direct libc call.
//!
//! Memory 0's base and byte length already live in fixed registers, so once
//! the range is checked these operations are a plain `memmove` / `memset`.
//! The general runtime-helper path instead saves every C-clobbered dynamic
//! lane (seven GP pushes and all fourteen XMM lanes), goes through the
//! opcode dispatcher, and resolves the memory through the store. Here only
//! the lanes holding values still needed after the operation are saved.
//!
//! Sequence:
//!
//! 1. Zero-extend the length into RDX, and for a copy the source offset
//!    into RAX, then the destination offset into RAX. RAX and RDX are
//!    backend-owned and never hold MachineIR values, so staging cannot
//!    overwrite another operand.
//! 2. Check each accessed range `[offset, offset + len)` against the memory
//!    length in the context's `MEM0_SIZE` field and branch to the shared
//!    out-of-bounds trap, the source range before the destination is staged
//!    over it. Both operands are zero-extended 32-bit values, so their
//!    64-bit sum cannot wrap. Nothing has touched the stack yet, so the trap
//!    needs no cleanup, and no byte is written before both ranges of a copy
//!    are proven.
//! 3. Push the live C-clobbered GP lanes, then reserve the XMM save area,
//!    an alignment pad and the ABI's shadow space, and save the live
//!    C-clobbered XMM lanes. The body keeps RSP 16-byte aligned between
//!    instructions, as the preserved-helper frame already relies on.
//! 4. Zero-extend the source offset or fill byte into RCX, an allocatable
//!    lane that is saved or dead by now. Form the C arguments from the
//!    staged values, last argument first, so no staged value is overwritten
//!    before it is read (checked below for both ABIs), and call through RAX.
//! 5. Restore in reverse order.

use core::ffi::c_void;

use crate::{
    collections,
    error::WasmError,
    vm::jit::machine::{
        machine_ir::{
            MachineReg, MachineTrapKind, MachineValue, MACHINE_FIXED_REG_COUNT,
            MACHINE_MEM0_BASE_REG,
        },
        peephole::helpers::reg_live_after,
    },
};

use super::{
    abi::{self, map_fixed_reg, C_ARG0, C_ARG1, C_ARG2},
    backend::X86_64Backend,
    callconv::{CALLEE_SAVED_GP, CALLEE_SAVED_XMM, CALL_SHADOW_BYTES},
    enc::{self, Cc},
    preserved::{movaps_load_rsp, movaps_store_rsp},
    reg::X86Reg,
};

unsafe extern "C" {
    fn memset(dest: *mut c_void, value: i32, len: usize) -> *mut c_void;
    fn memmove(dest: *mut c_void, src: *const c_void, len: usize) -> *mut c_void;
}

/// Staging registers: destination offset, source offset or fill byte,
/// length. RAX and RDX are backend-owned scratch registers; RCX is an
/// allocatable lane, written only after the live lanes are saved.
const STAGE_DEST: X86Reg = X86Reg::RAX;
const STAGE_SECOND: X86Reg = X86Reg::RCX;
const STAGE_LEN: X86Reg = X86Reg::RDX;

// Arguments are formed in the order C_ARG2 (from RDX), C_ARG1 (from RCX),
// C_ARG0 (from RAX). Each write must miss every staged register not yet
// read: C_ARG2 may not be RAX or RCX, and C_ARG1 may not be RAX. SysV
// (RDI, RSI, RDX) and Win64 (RCX, RDX, R8) both satisfy this.
const _: () = assert!(
    C_ARG2 as u8 != STAGE_DEST as u8
        && C_ARG2 as u8 != STAGE_SECOND as u8
        && C_ARG1 as u8 != STAGE_DEST as u8
);

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Mem0Bulk {
    Copy,
    Fill,
}

impl X86_64Backend<'_> {
    /// Lower `memory.copy` (`second` = source offset) or `memory.fill`
    /// (`second` = fill value) on memory 0. The caller checks that memory 0
    /// is a 32-bit memory and, for a copy, that both sides are memory 0.
    pub(super) fn lower_mem0_bulk(
        &mut self,
        op: Mem0Bulk,
        dest: MachineValue,
        second: MachineValue,
        len: MachineValue,
    ) -> Result<(), WasmError> {
        let dest_reg = self.gp_scratch.claim_rax().detach();
        let len_reg = self.gp_scratch.claim_rdx().detach();
        debug_assert!(*dest_reg == STAGE_DEST && *len_reg == STAGE_LEN);

        // Range checks branch to a trap, so they run before anything is
        // pushed, with only the two backend scratches: the source range is
        // checked through RAX before the destination is staged there.
        self.stage_u32(STAGE_LEN, len)?;
        if op == Mem0Bulk::Copy {
            self.stage_u32(STAGE_DEST, second)?;
            self.check_mem0_range(STAGE_DEST, second);
        }
        self.stage_u32(STAGE_DEST, dest)?;
        self.check_mem0_range(STAGE_DEST, dest);

        let (saved_gp, saved_xmm) = self.live_c_clobbered_lanes()?;
        let pad = if saved_gp.len() % 2 == 1 { 8 } else { 0 };
        let xmm_bytes = saved_xmm.len() as u32 * 16;
        let frame_bytes = CALL_SHADOW_BYTES + xmm_bytes + pad;
        for &reg in &saved_gp {
            enc::push(&mut self.core.text, reg);
        }
        if frame_bytes != 0 {
            enc::sub_rsp_imm32(&mut self.core.text, frame_bytes);
        }
        for (slot, &xmm) in saved_xmm.iter().enumerate() {
            let disp = (CALL_SHADOW_BYTES + slot as u32 * 16) as i32;
            movaps_store_rsp(&mut self.core.text, xmm, disp);
        }

        // RCX is an allocatable lane: past the saves it is either preserved
        // on the stack or dead, and its own value, if `second` lives there,
        // is read here before any argument register is written.
        self.stage_u32(STAGE_SECOND, second)?;
        let mem0_base = map_fixed_reg(MACHINE_MEM0_BASE_REG);
        if C_ARG2 != STAGE_LEN {
            enc::mov_rr_64(&mut self.core.text, C_ARG2, STAGE_LEN);
        }
        let target = match op {
            Mem0Bulk::Copy => {
                enc::lea_sum(&mut self.core.text, true, C_ARG1, mem0_base, STAGE_SECOND);
                memmove as *const () as usize
            }
            Mem0Bulk::Fill => {
                if C_ARG1 != STAGE_SECOND {
                    enc::mov_rr_32(&mut self.core.text, C_ARG1, STAGE_SECOND);
                }
                memset as *const () as usize
            }
        };
        enc::lea_sum(&mut self.core.text, true, C_ARG0, mem0_base, STAGE_DEST);
        self.materialize_u64(X86Reg::RAX, target as u64);
        enc::call_reg(&mut self.core.text, X86Reg::RAX);

        for (slot, &xmm) in saved_xmm.iter().enumerate() {
            let disp = (CALL_SHADOW_BYTES + slot as u32 * 16) as i32;
            movaps_load_rsp(&mut self.core.text, xmm, disp);
        }
        if frame_bytes != 0 {
            enc::add_rsp_imm32(&mut self.core.text, frame_bytes);
        }
        for &reg in saved_gp.iter().rev() {
            enc::pop(&mut self.core.text, reg);
        }
        Ok(())
    }

    /// Zero-extend a 32-bit operand into a staging register.
    fn stage_u32(&mut self, target: X86Reg, value: MachineValue) -> Result<(), WasmError> {
        if let MachineValue::Imm64(imm) = value {
            self.materialize_u64(target, imm & u64::from(u32::MAX));
            return Ok(());
        }
        let source = self.materialize_value(target, value)?;
        if source != target {
            enc::mov_rr_32(&mut self.core.text, target, source);
        }
        Ok(())
    }

    /// Trap unless `offset + len <= memory length`; `offset` is staged in
    /// `staged` and the length in `STAGE_LEN`. The staged offset is restored.
    fn check_mem0_range(&mut self, staged: X86Reg, offset: MachineValue) {
        let mem0_size = Self::mem0_size_mem();
        let trap = self
            .core
            .ensure_trap_label(MachineTrapKind::MemoryOutOfBounds);
        if offset == MachineValue::Imm64(0) {
            enc::cmp_reg_mem(&mut self.core.text, true, STAGE_LEN, mem0_size);
            self.emit_jcc(Cc::A, trap);
            return;
        }
        enc::add_rr_64(&mut self.core.text, staged, STAGE_LEN);
        enc::cmp_reg_mem(&mut self.core.text, true, staged, mem0_size);
        self.emit_jcc(Cc::A, trap);
        enc::sub_rr_64(&mut self.core.text, staged, STAGE_LEN);
    }

    /// Dynamic lanes a C callee may clobber that hold a MachineIR value
    /// read after the current instruction. Without the MachineIR position,
    /// every C-clobbered dynamic lane counts as live.
    fn live_c_clobbered_lanes(
        &self,
    ) -> Result<(collections::Vec<X86Reg>, collections::Vec<u32>), WasmError> {
        let config = abi::compile_backend_config();
        let position = self
            .core
            .current_block
            .zip(self.core.current_op_index)
            .and_then(|(block_id, op_index)| {
                let blocks = self.core.mir_blocks().ok()?;
                let block = blocks.iter().find(|block| block.id == block_id)?;
                Some((block.ops.get(op_index + 1..)?, &block.terminator))
            });
        let live = |reg: MachineReg| {
            position
                .is_none_or(|(remaining, terminator)| reg_live_after(remaining, terminator, reg))
        };

        let mut gp = collections::Vec::new();
        for index in 0..u16::from(config.gp_dynamic_budget) {
            let reg = MachineReg(MACHINE_FIXED_REG_COUNT + index);
            let mapped = abi::map_reg(reg)?;
            if !CALLEE_SAVED_GP.contains(&mapped) && live(reg) {
                gp.push(mapped);
            }
        }
        let mut xmm = collections::Vec::new();
        for index in 0..abi::FP_MACHINE_REG_COUNT {
            let reg = MachineReg(config.first_fp_reg() + index as u16);
            let mapped = abi::fp_machine_reg(index)
                .ok_or_else(|| WasmError::internal("x86_64 FP dynamic lane has no XMM"))?;
            if !CALLEE_SAVED_XMM.contains(&mapped) && live(reg) {
                xmm.push(mapped);
            }
        }
        Ok((gp, xmm))
    }
}
