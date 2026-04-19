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

    pub(super) fn emit_adjust_stack_down(&mut self, mut bytes: u32) {
        while bytes > 4080 {
            self.core
                .text
                .emit_u32(enc::sub_imm_64(abi::stack_reg(), abi::stack_reg(), 4080));
            bytes -= 4080;
        }
        if bytes != 0 {
            self.core
                .text
                .emit_u32(enc::sub_imm_64(abi::stack_reg(), abi::stack_reg(), bytes));
        }
    }

    pub(super) fn emit_adjust_stack_up(&mut self, mut bytes: u32) {
        while bytes > 4080 {
            self.core
                .text
                .emit_u32(enc::add_imm_64(abi::stack_reg(), abi::stack_reg(), 4080));
            bytes -= 4080;
        }
        if bytes != 0 {
            self.core
                .text
                .emit_u32(enc::add_imm_64(abi::stack_reg(), abi::stack_reg(), bytes));
        }
    }

    /// Open the preserved-helper frame: allocate stack space, save all
    /// caller-clobbered JIT registers.  After this, the I/O area is at SP+0
    /// and all JIT register values are preserved on the native stack.
    pub(super) fn emit_preserved_frame_open(&mut self) {
        self.emit_preserved_frame_open_with_prefix(0);
    }

    pub(super) fn emit_preserved_frame_open_with_prefix(&mut self, prefix_bytes: u32) {
        self.emit_adjust_stack_down(abi::PRESERVED_HELPER_FRAME_SIZE + prefix_bytes);
        self.emit_save_preserved_gp(prefix_bytes);
        self.emit_save_preserved_fp(prefix_bytes);
    }

    /// Store a u32 immediate into an I/O slot.
    pub(super) fn emit_io_store_imm(&mut self, slot: usize, value: u32) {
        self.emit_io_store_imm_at(0, slot, value);
    }

    pub(super) fn emit_io_store_imm_at(&mut self, base_slots: usize, slot: usize, value: u32) {
        let scratch = *self.gp_scratch.scoped_alloc();
        materialize_u64_into(&mut self.core.text, scratch, value as u64);
        self.core.text.emit_u32(enc::str_64(
            scratch,
            abi::stack_reg(),
            (base_slots + slot) as u32,
        ));
    }

    /// Store a MachineValue into an I/O slot.
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
        let gp = prepare_gp(
            self.core.compiled.backend(),
            &self.core.fp_reg_widths,
            &mut self.core.text,
            &self.gp_scratch,
            value,
        )?;
        self.core.text.emit_u32(enc::str_64(
            *gp,
            abi::stack_reg(),
            (base_slots + slot) as u32,
        ));
        Ok(())
    }

    /// Emit the BLR call to the preserved runtime entry, branch immediately on
    /// `C_RET0` if the helper trapped, and otherwise keep the helper result in
    /// a caller-owned scratch register while restoring the preserved JIT state.
    ///
    /// When `result_scratch_idx` is `Some`, the caller must have reserved that GP
    /// scratch slot already and remains responsible for freeing it after consuming
    /// the helper result.
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

        // Set up C calling convention: x0=ctx, x1=op_code, x2=io (=SP).
        self.core.text.emit_u32(enc::mov_reg_64(
            abi::C_ARG0,
            abi::map_fixed_reg(MACHINE_CTX_REG),
        ));
        materialize_u64_into(&mut self.core.text, abi::C_ARG1, op_code as u64);
        if prefix_bytes == 0 {
            self.core
                .text
                .emit_u32(enc::add_imm_64(abi::C_ARG2, abi::stack_reg(), 0));
        } else if prefix_bytes <= 4095 {
            self.core
                .text
                .emit_u32(enc::add_imm_64(abi::C_ARG2, abi::stack_reg(), prefix_bytes));
        } else {
            materialize_u64_into(&mut self.core.text, abi::C_ARG2, prefix_bytes as u64);
            self.core
                .text
                .emit_u32(enc::add_reg_64(abi::C_ARG2, abi::stack_reg(), abi::C_ARG2));
        }
        materialize_u64_into(
            &mut self.core.text,
            call_scratch,
            preserved_entry as usize as u64,
        );
        self.core.text.emit_u32(enc::blr(call_scratch));

        // Errors exit the current function via `body_local_error_label`, so
        // they only need to discard the preserved frame and keep `C_RET0`
        // intact. Prior to this branch-first shape, restoring the GP save set
        // could clobber x0/x1 and silently drop post-call helper traps.
        // Success keeps the helper result live across the restore.
        let error_path = self.core.new_label();
        self.lower_cbnz(abi::C_RET0, error_path);

        if result_scratch_idx.is_some() {
            self.core.text.emit_u32(enc::ldr_64(
                call_scratch,
                abi::stack_reg(),
                prefix_bytes / 8 + preserved_io::RET0 as u32,
            ));
        }

        // Restore all caller-clobbered JIT registers.
        self.emit_restore_preserved_fp(prefix_bytes);
        self.emit_restore_preserved_gp(prefix_bytes);

        // Deallocate frame.
        self.emit_adjust_stack_up(abi::PRESERVED_HELPER_FRAME_SIZE + prefix_bytes);

        let done = self.core.new_label();
        self.lower_b(done);

        self.core.bind_label(error_path);
        self.emit_adjust_stack_up(abi::PRESERVED_HELPER_FRAME_SIZE + prefix_bytes);
        let body_local_error_label = self.core.body_local_error_label;
        self.lower_b(body_local_error_label);

        self.core.bind_label(done);
        if result_scratch_idx.is_none() {
            self.gp_scratch.free_index(call_scratch_idx);
        }
    }

    // ── Register save/restore ───────────────────────────────────────────────

    fn emit_save_preserved_gp(&mut self, prefix_bytes: u32) {
        let base_off = abi::PRESERVED_HELPER_GP_OFFSET + prefix_bytes;
        let mut slot = 0u32;
        let regs = abi::gp_dynamic_caller_saved_regs();
        let mut i = 0;
        while i + 1 < regs.len() {
            let offset_units = ((base_off + slot * 8) / 8) as i32;
            if (-64..64).contains(&offset_units) {
                self.core.text.emit_u32(enc::stp_64(
                    regs[i],
                    regs[i + 1],
                    abi::stack_reg(),
                    offset_units,
                ));
            } else {
                self.core.text.emit_u32(enc::str_64(
                    regs[i],
                    abi::stack_reg(),
                    (base_off + slot * 8) / 8,
                ));
                self.core.text.emit_u32(enc::str_64(
                    regs[i + 1],
                    abi::stack_reg(),
                    (base_off + (slot + 1) * 8) / 8,
                ));
            }
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

    pub(super) fn emit_restore_preserved_gp(&mut self, prefix_bytes: u32) {
        let base_off = abi::PRESERVED_HELPER_GP_OFFSET + prefix_bytes;
        let mut slot = 0u32;
        let regs = abi::gp_dynamic_caller_saved_regs();
        let mut i = 0;
        while i + 1 < regs.len() {
            let offset_units = ((base_off + slot * 8) / 8) as i32;
            if (-64..64).contains(&offset_units) {
                self.core.text.emit_u32(enc::ldp_64(
                    regs[i],
                    regs[i + 1],
                    abi::stack_reg(),
                    offset_units,
                ));
            } else {
                self.core.text.emit_u32(enc::ldr_64(
                    regs[i],
                    abi::stack_reg(),
                    (base_off + slot * 8) / 8,
                ));
                self.core.text.emit_u32(enc::ldr_64(
                    regs[i + 1],
                    abi::stack_reg(),
                    (base_off + (slot + 1) * 8) / 8,
                ));
            }
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

    fn emit_save_preserved_fp(&mut self, prefix_bytes: u32) {
        let base_off = abi::PRESERVED_HELPER_FP_OFFSET + prefix_bytes;
        let mut slot = 0u32;
        for reg in abi::fp_dynamic_caller_saved_regs().iter().copied() {
            // SIMD builds let the FP bank carry v128 values, so preserved
            // spills must save the whole Q reg. Scalar-only builds only need
            // the low 64-bit D view.
            #[cfg(sf_has_simd)]
            {
                self.core.text.emit_u32(enc::str_q(
                    reg,
                    abi::stack_reg(),
                    (base_off + slot * 16) / 16,
                ));
            }
            #[cfg(not(sf_has_simd))]
            {
                self.core.text.emit_u32(enc::str_d(
                    reg,
                    abi::stack_reg(),
                    (base_off + slot * 8) / 8,
                ));
            }
            slot += 1;
        }
    }

    pub(super) fn emit_restore_preserved_fp(&mut self, prefix_bytes: u32) {
        let base_off = abi::PRESERVED_HELPER_FP_OFFSET + prefix_bytes;
        let mut slot = 0u32;
        for reg in abi::fp_dynamic_caller_saved_regs().iter().copied() {
            // Match emit_save_preserved_fp: restore the full Q reg when SIMD is
            // enabled, otherwise only the scalar D payload is live.
            #[cfg(sf_has_simd)]
            {
                self.core.text.emit_u32(enc::ldr_q(
                    reg,
                    abi::stack_reg(),
                    (base_off + slot * 16) / 16,
                ));
            }
            #[cfg(not(sf_has_simd))]
            {
                self.core.text.emit_u32(enc::ldr_d(
                    reg,
                    abi::stack_reg(),
                    (base_off + slot * 8) / 8,
                ));
            }
            slot += 1;
        }
    }
} // impl Arm64Backend (preserved.rs)
