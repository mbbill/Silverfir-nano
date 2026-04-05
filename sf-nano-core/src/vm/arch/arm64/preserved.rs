//! Preserved-helper frame: save/restore all caller-clobbered JIT registers
//! around a C runtime helper call.  Instruction lowering code in `inst.rs`
//! calls these helpers when it needs to make a preserved call.

use crate::error::WasmError;
use crate::vm::machine::machine_ir::{MachineValue, MACHINE_CTX_REG};

use super::backend::Arm64Backend;
use super::inst::{materialize_u64_into, prepare_gp};
use super::{abi, enc};

impl<'a> Arm64Backend<'a> {
    // ── Frame open / close ──────────────────────────────────────────────────

    /// Open the preserved-helper frame: allocate stack space, save all
    /// caller-clobbered JIT registers.  After this, the I/O area is at SP+0
    /// and all JIT register values are preserved on the native stack.
    pub(super) fn emit_preserved_frame_open(&mut self) {
        self.core.text.emit_u32(enc::sub_imm_64(
            abi::stack_reg(),
            abi::stack_reg(),
            abi::PRESERVED_HELPER_FRAME_SIZE,
        ));
        self.emit_save_preserved_gp();
        self.emit_save_preserved_fp();
    }

    /// Store a u32 immediate into an I/O slot.
    pub(super) fn emit_io_store_imm(&mut self, slot: usize, value: u32) {
        let scratch = *self.gp_scratch.scoped_alloc();
        materialize_u64_into(&mut self.core.text, scratch, value as u64);
        self.core
            .text
            .emit_u32(enc::str_64(scratch, abi::stack_reg(), slot as u32));
    }

    /// Store a MachineValue into an I/O slot.
    pub(super) fn emit_io_store_value(
        &mut self,
        slot: usize,
        value: MachineValue,
    ) -> Result<(), WasmError> {
        let gp = prepare_gp(
            self.core.compiled.backend(),
            &self.core.fp_reg_widths,
            &mut self.core.text,
            &self.gp_scratch,
            value,
        )?;
        self.core
            .text
            .emit_u32(enc::str_64(*gp, abi::stack_reg(), slot as u32));
        Ok(())
    }

    /// Emit the BLR call to the preserved runtime entry, stash status in a scratch register
    /// and optionally keep the helper result in a caller-owned scratch register,
    /// restore all JIT registers, deallocate the frame, and branch to the error
    /// path on nonzero status.
    ///
    /// When `result_scratch_idx` is `Some`, the caller must have reserved that GP
    /// scratch slot already and remains responsible for freeing it after consuming
    /// the helper result.
    pub(super) fn emit_preserved_call_and_close(
        &mut self,
        op_code: u32,
        result_scratch_idx: Option<u8>,
    ) {
        use crate::vm::runtime::preserved::{io as preserved_io, preserved_entry};

        let call_scratch_idx = result_scratch_idx.unwrap_or_else(|| self.gp_scratch.alloc());
        let status_scratch_idx = self.gp_scratch.alloc();
        let call_scratch = self.gp_scratch.reg(call_scratch_idx);
        let status_scratch = self.gp_scratch.reg(status_scratch_idx);

        // Set up C calling convention: x0=ctx, x1=op_code, x2=io (=SP).
        self.core.text.emit_u32(enc::mov_reg_64(
            abi::C_ARG0,
            abi::map_fixed_reg(MACHINE_CTX_REG),
        ));
        materialize_u64_into(&mut self.core.text, abi::C_ARG1, op_code as u64);
        self.core
            .text
            .emit_u32(enc::add_imm_64(abi::C_ARG2, abi::stack_reg(), 0));
        materialize_u64_into(
            &mut self.core.text,
            call_scratch,
            preserved_entry as usize as u64,
        );
        self.core.text.emit_u32(enc::blr(call_scratch));

        // Stash status and result in scratch regs before restoring.
        self.core
            .text
            .emit_u32(enc::mov_reg_64(status_scratch, abi::C_RET0));
        if result_scratch_idx.is_some() {
            self.core.text.emit_u32(enc::ldr_64(
                call_scratch,
                abi::stack_reg(),
                preserved_io::RET0 as u32,
            ));
        }

        // Restore all caller-clobbered JIT registers.
        self.emit_restore_preserved_fp();
        self.emit_restore_preserved_gp();

        // Deallocate frame.
        self.core.text.emit_u32(enc::add_imm_64(
            abi::stack_reg(),
            abi::stack_reg(),
            abi::PRESERVED_HELPER_FRAME_SIZE,
        ));

        // Check status.
        let return_error_label = self.core.return_error_label;
        self.lower_cbnz(status_scratch, return_error_label);

        self.gp_scratch.free_index(status_scratch_idx);
        if result_scratch_idx.is_none() {
            self.gp_scratch.free_index(call_scratch_idx);
        }
    }

    // ── Register save/restore ───────────────────────────────────────────────

    fn emit_save_preserved_gp(&mut self) {
        let base_off = abi::PRESERVED_HELPER_GP_OFFSET;
        let mut slot = 0u32;
        let regs = abi::gp_dynamic_caller_saved_regs();
        let mut i = 0;
        while i + 1 < regs.len() {
            self.core.text.emit_u32(enc::stp_64(
                regs[i],
                regs[i + 1],
                abi::stack_reg(),
                ((base_off + slot * 8) / 8) as i32,
            ));
            slot += 2;
            i += 2;
        }
        if i < regs.len() {
            self.core.text.emit_u32(enc::str_64(
                regs[i],
                abi::stack_reg(),
                (base_off + slot * 8) / 8,
            ));
        }
    }

    fn emit_restore_preserved_gp(&mut self) {
        let base_off = abi::PRESERVED_HELPER_GP_OFFSET;
        let mut slot = 0u32;
        let regs = abi::gp_dynamic_caller_saved_regs();
        let mut i = 0;
        while i + 1 < regs.len() {
            self.core.text.emit_u32(enc::ldp_64(
                regs[i],
                regs[i + 1],
                abi::stack_reg(),
                ((base_off + slot * 8) / 8) as i32,
            ));
            slot += 2;
            i += 2;
        }
        if i < regs.len() {
            self.core.text.emit_u32(enc::ldr_64(
                regs[i],
                abi::stack_reg(),
                (base_off + slot * 8) / 8,
            ));
        }
    }

    fn emit_save_preserved_fp(&mut self) {
        let base_off = abi::PRESERVED_HELPER_FP_OFFSET;
        let mut slot = 0u32;
        for reg in abi::fp_dynamic_caller_saved_regs().iter().copied() {
            self.core.text.emit_u32(enc::str_d(
                reg,
                abi::stack_reg(),
                (base_off + slot * 8) / 8,
            ));
            slot += 1;
        }
    }

    fn emit_restore_preserved_fp(&mut self) {
        let base_off = abi::PRESERVED_HELPER_FP_OFFSET;
        let mut slot = 0u32;
        for reg in abi::fp_dynamic_caller_saved_regs().iter().copied() {
            self.core.text.emit_u32(enc::ldr_d(
                reg,
                abi::stack_reg(),
                (base_off + slot * 8) / 8,
            ));
            slot += 1;
        }
    }
} // impl Arm64Backend (preserved.rs)
