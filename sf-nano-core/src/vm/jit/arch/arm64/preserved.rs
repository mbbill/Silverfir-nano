//! Preserved-helper frame: save/restore all caller-clobbered JIT registers
//! around a C runtime helper call.  Instruction lowering code in `inst.rs`
//! calls these helpers when it needs to make a preserved call.

use crate::error::WasmError;
use crate::vm::jit::machine::machine_ir::{MachineValue, MACHINE_CTX_REG};

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
        self.refresh_general_helper_save_sets();
        self.emit_adjust_stack_down(abi::PRESERVED_HELPER_FRAME_SIZE + prefix_bytes);
        self.emit_save_preserved_gp(prefix_bytes);
        self.emit_save_preserved_fp(prefix_bytes);
    }

    /// Open the smallest legal frame for a direct infallible C helper.
    ///
    /// The overwhelmingly common bulk-memory case has exactly two live GP
    /// lanes and no FP lanes. A pre-index STP both allocates the aligned frame
    /// and saves them; the matching post-index LDP closes it in one
    /// instruction. Other save shapes retain the general preserved frame.
    pub(super) fn emit_infallible_host_frame_open(&mut self) -> bool {
        self.refresh_infallible_helper_save_sets();
        if self.helper_saved_gp.len() == 2 && self.helper_saved_fp.is_empty() {
            self.core.text.emit_u32(enc::stp_64_pre_index(
                self.helper_saved_gp[0],
                self.helper_saved_gp[1],
                abi::stack_reg(),
                -16,
            ));
            true
        } else {
            self.emit_adjust_stack_down(abi::PRESERVED_HELPER_FRAME_SIZE);
            self.emit_save_preserved_gp(0);
            self.emit_save_preserved_fp(0);
            false
        }
    }

    /// General runtime helpers may inspect or update VM state whose machine
    /// register dependencies are not represented by ordinary MIR liveness.
    /// Preserve every caller-clobbered lane used anywhere in this body, as the
    /// original helper-call contract requires.
    fn refresh_general_helper_save_sets(&mut self) {
        self.helper_saved_gp.clear();
        self.helper_saved_fp.clear();
        self.helper_saved_gp
            .extend_from_slice(abi::gp_dynamic_caller_saved_regs());
        self.helper_saved_fp
            .extend_from_slice(abi::fp_dynamic_caller_saved_regs());
    }

    /// Raw infallible host calls have no VM-level side effects beyond their
    /// explicit memory range, so normal machine liveness is sufficient here.
    fn refresh_infallible_helper_save_sets(&mut self) {
        use crate::vm::jit::arch::common::core::FunctionBody;
        use crate::vm::jit::machine::peephole::helpers::reg_live_after;

        self.helper_saved_gp.clear();
        self.helper_saved_fp.clear();
        let FunctionBody::Mir(function) = self.core.body else {
            self.helper_saved_gp
                .extend_from_slice(abi::gp_dynamic_caller_saved_regs());
            self.helper_saved_fp
                .extend_from_slice(abi::fp_dynamic_caller_saved_regs());
            return;
        };
        let Some(block_id) = self.core.current_block else {
            return;
        };
        let Some(op_index) = self.core.current_op_index else {
            return;
        };
        let Some(block) = function
            .program
            .blocks
            .iter()
            .find(|block| block.id == block_id)
        else {
            return;
        };
        let remaining = block.ops.get(op_index + 1..).unwrap_or(&[]);
        for index in 0..self.helper_gp_candidates.len() {
            let (machine, mapped) = self.helper_gp_candidates[index];
            if reg_live_after(remaining, &block.terminator, machine) {
                self.helper_saved_gp.push(mapped);
            }
        }
        for index in 0..self.helper_fp_candidates.len() {
            let (machine, mapped) = self.helper_fp_candidates[index];
            if reg_live_after(remaining, &block.terminator, machine) {
                self.helper_saved_fp.push(mapped);
            }
        }
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
    ) -> Result<(), WasmError> {
        self.emit_preserved_call_and_close_with_prefix(op_code, result_scratch_idx, 0)
    }

    pub(super) fn emit_preserved_call_and_close_with_prefix(
        &mut self,
        op_code: u32,
        result_scratch_idx: Option<u8>,
        prefix_bytes: u32,
    ) -> Result<(), WasmError> {
        use crate::vm::runtime::preserved::preserved_entry;

        self.emit_preserved_call_and_close_impl(
            Some(op_code),
            preserved_entry as *const () as usize,
            result_scratch_idx,
            prefix_bytes,
        )
    }

    pub(super) fn emit_preserved_direct_call_and_close(
        &mut self,
        entry: usize,
        result_scratch_idx: Option<u8>,
    ) -> Result<(), WasmError> {
        self.emit_preserved_call_and_close_impl(None, entry, result_scratch_idx, 0)
    }

    /// Close a preserved frame around an infallible C helper such as libc's
    /// `memset`/`memmove`. The caller has already performed Wasm bounds checks
    /// and placed C arguments, so no status dispatch or I/O frame is needed.
    pub(super) fn emit_infallible_host_call_and_close(
        &mut self,
        entry: usize,
        cached_target: Option<super::reg::Arm64Reg>,
        compact_frame: bool,
    ) -> Result<(), WasmError> {
        let call_scratch_idx = cached_target.is_none().then(|| self.gp_scratch.alloc());
        let call_scratch = cached_target.unwrap_or_else(|| {
            self.gp_scratch
                .reg(call_scratch_idx.expect("uncached helper target scratch"))
        });
        if cached_target.is_none() {
            materialize_u64_into(&mut self.core.text, call_scratch, entry as u64);
        }
        self.emit_body_returning_blr(call_scratch)?;
        if compact_frame {
            debug_assert_eq!(self.helper_saved_gp.len(), 2);
            debug_assert!(self.helper_saved_fp.is_empty());
            self.core.text.emit_u32(enc::ldp_64_post_index(
                self.helper_saved_gp[0],
                self.helper_saved_gp[1],
                abi::stack_reg(),
                16,
            ));
        } else {
            self.emit_restore_preserved_fp(0);
            self.emit_restore_preserved_gp(0);
            self.emit_adjust_stack_up(abi::PRESERVED_HELPER_FRAME_SIZE);
        }
        if let Some(index) = call_scratch_idx {
            self.gp_scratch.free_index(index);
        }
        Ok(())
    }

    fn emit_preserved_call_and_close_impl(
        &mut self,
        op_code: Option<u32>,
        entry: usize,
        result_scratch_idx: Option<u8>,
        prefix_bytes: u32,
    ) -> Result<(), WasmError> {
        use crate::vm::runtime::preserved::io as preserved_io;

        let call_scratch_idx = result_scratch_idx.unwrap_or_else(|| self.gp_scratch.alloc());
        let call_scratch = self.gp_scratch.reg(call_scratch_idx);

        // Generic calls use x0=ctx, x1=op_code, x2=io. Direct hot-path
        // helpers use x0=ctx, x1=io and skip the opcode dispatcher.
        self.core.text.emit_u32(enc::mov_reg_64(
            abi::C_ARG0,
            abi::map_fixed_reg(MACHINE_CTX_REG),
        ));
        if let Some(op_code) = op_code {
            materialize_u64_into(&mut self.core.text, abi::C_ARG1, op_code as u64);
        }
        let io_arg = if op_code.is_some() {
            abi::C_ARG2
        } else {
            abi::C_ARG1
        };
        if prefix_bytes == 0 {
            self.core
                .text
                .emit_u32(enc::add_imm_64(io_arg, abi::stack_reg(), 0));
        } else if prefix_bytes <= 4095 {
            self.core
                .text
                .emit_u32(enc::add_imm_64(io_arg, abi::stack_reg(), prefix_bytes));
        } else {
            materialize_u64_into(&mut self.core.text, io_arg, prefix_bytes as u64);
            self.core
                .text
                .emit_u32(enc::add_reg_64(io_arg, abi::stack_reg(), io_arg));
        }
        materialize_u64_into(&mut self.core.text, call_scratch, entry as u64);
        self.emit_body_returning_blr(call_scratch)?;

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
        Ok(())
    }

    // ── Register save/restore ───────────────────────────────────────────────

    fn emit_save_preserved_gp(&mut self, prefix_bytes: u32) {
        let base_off = abi::PRESERVED_HELPER_GP_OFFSET + prefix_bytes;
        let mut slot = 0u32;
        let mut i = 0;
        while i + 1 < self.helper_saved_gp.len() {
            let first = self.helper_saved_gp[i];
            let second = self.helper_saved_gp[i + 1];
            let offset_units = ((base_off + slot * 8) / 8) as i32;
            if (-64..64).contains(&offset_units) {
                self.core
                    .text
                    .emit_u32(enc::stp_64(first, second, abi::stack_reg(), offset_units));
            } else {
                self.core.text.emit_u32(enc::str_64(
                    first,
                    abi::stack_reg(),
                    (base_off + slot * 8) / 8,
                ));
                self.core.text.emit_u32(enc::str_64(
                    second,
                    abi::stack_reg(),
                    (base_off + (slot + 1) * 8) / 8,
                ));
            }
            slot += 2;
            i += 2;
        }
        if i < self.helper_saved_gp.len() {
            self.core.text.emit_u32(enc::str_64(
                self.helper_saved_gp[i],
                abi::stack_reg(),
                (base_off + slot * 8) / 8,
            ));
        }
    }

    pub(super) fn emit_restore_preserved_gp(&mut self, prefix_bytes: u32) {
        let base_off = abi::PRESERVED_HELPER_GP_OFFSET + prefix_bytes;
        let mut slot = 0u32;
        let mut i = 0;
        while i + 1 < self.helper_saved_gp.len() {
            let first = self.helper_saved_gp[i];
            let second = self.helper_saved_gp[i + 1];
            let offset_units = ((base_off + slot * 8) / 8) as i32;
            if (-64..64).contains(&offset_units) {
                self.core
                    .text
                    .emit_u32(enc::ldp_64(first, second, abi::stack_reg(), offset_units));
            } else {
                self.core.text.emit_u32(enc::ldr_64(
                    first,
                    abi::stack_reg(),
                    (base_off + slot * 8) / 8,
                ));
                self.core.text.emit_u32(enc::ldr_64(
                    second,
                    abi::stack_reg(),
                    (base_off + (slot + 1) * 8) / 8,
                ));
            }
            slot += 2;
            i += 2;
        }
        if i < self.helper_saved_gp.len() {
            self.core.text.emit_u32(enc::ldr_64(
                self.helper_saved_gp[i],
                abi::stack_reg(),
                (base_off + slot * 8) / 8,
            ));
        }
    }

    fn emit_save_preserved_fp(&mut self, prefix_bytes: u32) {
        let base_off = abi::PRESERVED_HELPER_FP_OFFSET + prefix_bytes;
        let mut slot = 0u32;
        for index in 0..self.helper_saved_fp.len() {
            let reg = self.helper_saved_fp[index];
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
        for index in 0..self.helper_saved_fp.len() {
            let reg = self.helper_saved_fp[index];
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
