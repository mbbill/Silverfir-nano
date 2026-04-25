//! Preserved-helper frame support for RV64.
//!
//! Runtime helper calls cross the C ABI, so caller-clobbered JIT dynamic
//! registers must be saved around the call. Callee-saved dynamic registers live
//! in `s*` registers and are protected by the platform ABI.

use crate::error::WasmError;
use crate::vm::machine::machine_ir::{MachineValue, MACHINE_CTX_REG};

use super::backend::Riscv64Backend;
use super::{abi, enc};

impl<'a> Riscv64Backend<'a> {
    pub(super) fn emit_adjust_stack_down(&mut self, mut bytes: u32) {
        while bytes > 2047 {
            self.emit_addi(abi::stack_reg(), abi::stack_reg(), -2047);
            bytes -= 2047;
        }
        if bytes != 0 {
            self.emit_addi(abi::stack_reg(), abi::stack_reg(), -(bytes as i32));
        }
    }

    pub(super) fn emit_adjust_stack_up(&mut self, mut bytes: u32) {
        while bytes > 2047 {
            self.emit_addi(abi::stack_reg(), abi::stack_reg(), 2047);
            bytes -= 2047;
        }
        if bytes != 0 {
            self.emit_addi(abi::stack_reg(), abi::stack_reg(), bytes as i32);
        }
    }

    pub(super) fn emit_preserved_frame_open(&mut self) {
        self.emit_preserved_frame_open_with_prefix(0);
    }

    pub(super) fn emit_preserved_frame_open_with_prefix(&mut self, prefix_bytes: u32) {
        self.emit_adjust_stack_down(abi::PRESERVED_HELPER_FRAME_SIZE + prefix_bytes);
        self.emit_save_preserved_gp(prefix_bytes);
        self.emit_save_preserved_fp(prefix_bytes);
    }

    pub(super) fn emit_io_store_imm(&mut self, slot: usize, value: u32) {
        self.emit_io_store_imm_at(0, slot, value);
    }

    pub(super) fn emit_io_store_imm_at(&mut self, base_slots: usize, slot: usize, value: u32) {
        let scratch = self.gp_scratch.scoped_alloc().detach();
        self.materialize_u64(*scratch, value as u64);
        self.emit_store_raw(
            0b011,
            *scratch,
            abi::stack_reg(),
            ((base_slots + slot) * 8) as i32,
        );
    }

    pub(super) fn emit_io_store_value(
        &mut self,
        slot: usize,
        value: MachineValue,
    ) -> Result<(), WasmError> {
        self.emit_io_store_value_at(0, slot, value)
    }

    pub(super) fn emit_io_store_value_at(
        &mut self,
        base_slots: usize,
        slot: usize,
        value: MachineValue,
    ) -> Result<(), WasmError> {
        let scratch = self.gp_scratch.scoped_alloc().detach();
        self.load_value_into(*scratch, value)?;
        self.emit_store_raw(
            0b011,
            *scratch,
            abi::stack_reg(),
            ((base_slots + slot) * 8) as i32,
        );
        Ok(())
    }

    pub(super) fn emit_preserved_call_and_close(
        &mut self,
        op_code: u32,
        result_scratch_idx: Option<u8>,
    ) {
        self.emit_preserved_call_and_close_with_prefix(op_code, result_scratch_idx, 0);
    }

    pub(super) fn emit_preserved_call_and_close_with_prefix(
        &mut self,
        op_code: u32,
        result_scratch_idx: Option<u8>,
        prefix_bytes: u32,
    ) {
        use crate::vm::runtime::preserved::{io as preserved_io, preserved_entry};

        let call_scratch_idx = result_scratch_idx.unwrap_or_else(|| self.gp_scratch.alloc());
        let call_scratch = self.gp_scratch.reg(call_scratch_idx);

        self.emit_mv(abi::C_ARG0, abi::map_fixed_reg(MACHINE_CTX_REG));
        self.materialize_u64(abi::C_ARG1, op_code as u64);
        if prefix_bytes == 0 {
            self.emit_mv(abi::C_ARG2, abi::stack_reg());
        } else if Riscv64Backend::fits_i12(prefix_bytes as i32) {
            self.emit_addi(abi::C_ARG2, abi::stack_reg(), prefix_bytes as i32);
        } else {
            self.materialize_u64(abi::C_ARG2, prefix_bytes as u64);
            self.core
                .text
                .emit_u32(enc::add(abi::C_ARG2, abi::stack_reg(), abi::C_ARG2));
        }
        self.materialize_u64(call_scratch, preserved_entry as *const () as usize as u64);
        self.core
            .text
            .emit_u32(enc::jalr(abi::link_reg(), call_scratch, 0));

        let error_path = self.core.new_label();
        self.emit_branch_to(enc::Cond::Ne, abi::C_RET0, abi::zero_reg(), error_path);

        if result_scratch_idx.is_some() {
            self.emit_load_raw(
                0b011,
                call_scratch,
                abi::stack_reg(),
                prefix_bytes as i32 + (preserved_io::RET0 as i32 * 8),
            );
        }

        self.emit_restore_preserved_gp(prefix_bytes);
        self.emit_restore_preserved_fp(prefix_bytes);
        self.emit_adjust_stack_up(abi::PRESERVED_HELPER_FRAME_SIZE + prefix_bytes);

        let done = self.core.new_label();
        self.emit_jal(abi::zero_reg(), done);

        self.core.bind_label(error_path);
        self.emit_adjust_stack_up(abi::PRESERVED_HELPER_FRAME_SIZE + prefix_bytes);
        let body_local_error_label = self.core.body_local_error_label;
        self.emit_jal(abi::zero_reg(), body_local_error_label);

        self.core.bind_label(done);
        if result_scratch_idx.is_none() {
            self.gp_scratch.free_index(call_scratch_idx);
        }
    }

    fn emit_save_preserved_gp(&mut self, prefix_bytes: u32) {
        let base_off = abi::PRESERVED_HELPER_GP_OFFSET + prefix_bytes;
        for (slot, reg) in abi::gp_dynamic_caller_saved_regs()
            .iter()
            .copied()
            .enumerate()
        {
            self.emit_store_raw(
                0b011,
                reg,
                abi::stack_reg(),
                (base_off + slot as u32 * 8) as i32,
            );
        }
    }

    pub(super) fn emit_restore_preserved_gp(&mut self, prefix_bytes: u32) {
        let base_off = abi::PRESERVED_HELPER_GP_OFFSET + prefix_bytes;
        for (slot, reg) in abi::gp_dynamic_caller_saved_regs()
            .iter()
            .copied()
            .enumerate()
        {
            self.emit_load_raw(
                0b011,
                reg,
                abi::stack_reg(),
                (base_off + slot as u32 * 8) as i32,
            );
        }
    }

    fn emit_save_preserved_fp(&mut self, prefix_bytes: u32) {
        let base_off = abi::PRESERVED_HELPER_FP_OFFSET + prefix_bytes;
        for (slot, reg) in abi::fp_dynamic_caller_saved_regs()
            .iter()
            .copied()
            .enumerate()
        {
            self.emit_fp_store_raw(
                0b011,
                reg,
                abi::stack_reg(),
                (base_off + slot as u32 * 8) as i32,
            );
        }
    }

    pub(super) fn emit_restore_preserved_fp(&mut self, prefix_bytes: u32) {
        let base_off = abi::PRESERVED_HELPER_FP_OFFSET + prefix_bytes;
        for (slot, reg) in abi::fp_dynamic_caller_saved_regs()
            .iter()
            .copied()
            .enumerate()
        {
            self.emit_fp_load_raw(
                0b011,
                reg,
                abi::stack_reg(),
                (base_off + slot as u32 * 8) as i32,
            );
        }
    }
}
