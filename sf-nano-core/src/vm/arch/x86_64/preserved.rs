//! Preserved-helper frame: save/restore caller-clobbered JIT state around a
//! bulk-memory/table runtime helper call.
//!
//! Memory/table ops on x86_64 route through the shared
//! `preserved_entry(ctx, op_code, io_ptr) -> u32` runtime helper. The
//! generated sequence is:
//!
//!   1. Save caller-clobbered GP dynamic regs (see
//!      `save_caller_clobbered_gp_dynamic`). Total 48 bytes keeps RSP
//!      16-aligned.
//!   2. `sub rsp, PRESERVED_IO_BYTES` (`io::SLOT_COUNT * 8` bytes).
//!   3. Write IMM0/IMM1/ARG0..ARG2 slots on the stack.
//!   4. `mov rdi, ctx ; mov rsi, op_code ; lea rdx, [rsp]`
//!   5. `mov r11, preserved_entry ; call r11`
//!   6. Stash status (RAX) into a scratch; optionally read RET0 into another
//!      scratch.
//!   7. `add rsp, PRESERVED_IO_BYTES`
//!   8. Restore caller-clobbered GP dynamic regs.
//!   9. `test status, status ; jne body_local_error_label`
//!
//! FP dynamic registers are not saved around these calls. The MachineIR
//! pipeline is responsible for publishing cached state before helper calls.

use crate::{
    error::WasmError,
    vm::machine::machine_ir::{MachineValue, MACHINE_CTX_REG},
};

use super::{
    abi::{map_fixed_reg, C_ARG0, C_ARG1, C_ARG2, C_RET0},
    backend::X86_64Backend,
    enc::{self, Cc},
    reg::X86Reg,
};

const PRESERVED_IO_BYTES: u8 = (crate::vm::runtime::preserved::io::SLOT_COUNT * 8) as u8;

impl<'a> X86_64Backend<'a> {
    pub(super) fn emit_preserved_io_open(&mut self) {
        self.save_caller_clobbered_gp_dynamic();
        enc::sub_rsp_imm8(&mut self.core.text, PRESERVED_IO_BYTES);
    }

    pub(super) fn emit_io_store_imm(&mut self, slot: usize, value: u32) {
        let scratch = self.gp_scratch.scoped_alloc().detach();
        self.materialize_u64(*scratch, value as u64);
        enc::store_64(
            &mut self.core.text,
            X86Reg::RSP,
            (slot as i32) * 8,
            *scratch,
        );
    }

    pub(super) fn emit_io_store_value(
        &mut self,
        slot: usize,
        value: MachineValue,
    ) -> Result<(), WasmError> {
        let scratch = self.gp_scratch.scoped_alloc().detach();
        let gp = self.materialize_value(*scratch, value)?;
        enc::store_64(&mut self.core.text, X86Reg::RSP, (slot as i32) * 8, gp);
        Ok(())
    }

    pub(super) fn emit_io_store_u32_value(
        &mut self,
        slot: usize,
        value: MachineValue,
    ) -> Result<(), WasmError> {
        let scratch = self.gp_scratch.scoped_alloc().detach();
        match value {
            // Bulk-memory/table helper operands are Wasm i32 values.
            MachineValue::Imm64(imm) => self.materialize_u64(*scratch, imm as u32 as u64),
            MachineValue::Reg(reg) => {
                let src = self.map_gp_reg(reg)?;
                // Force a low-word move so the helper always sees a clean
                // zero-extended u32 even if the producer left stale high bits.
                enc::mov_rr_32(&mut self.core.text, *scratch, src);
            }
            MachineValue::ReservedReg(reg) => {
                return Err(WasmError::internal(alloc::format!(
                    "x86_64 cannot materialize reserved cache register {}",
                    reg.0
                )));
            }
        }
        enc::store_64(
            &mut self.core.text,
            X86Reg::RSP,
            (slot as i32) * 8,
            *scratch,
        );
        Ok(())
    }

    /// Call preserved_entry, handle status/result, tear down frame, check
    /// status. If `result_dst` is `Some`, the RET0 slot is loaded into that
    /// register after the caller-clobbered restore.
    pub(super) fn emit_preserved_call_and_close(
        &mut self,
        op_code: u32,
        result_dst: Option<X86Reg>,
    ) {
        use crate::vm::runtime::preserved::{io as preserved_io, preserved_entry};

        // `R11` is volatile on both SysV and Win64 and is not used for any
        // argument slot here, so it can carry the helper target without
        // clobbering ABI inputs.
        let call_target = X86Reg::R11;
        // Keep the result in a non-restored caller-clobbered register so the
        // dynamic restore below cannot overwrite it.
        let result_scratch = X86Reg::RCX;

        enc::mov_rr_64(&mut self.core.text, C_ARG0, map_fixed_reg(MACHINE_CTX_REG));
        self.materialize_u64(C_ARG1, op_code as u64);
        enc::mov_rr_64(&mut self.core.text, C_ARG2, X86Reg::RSP);
        self.materialize_u64(call_target, preserved_entry as usize as u64);
        enc::call_reg(&mut self.core.text, call_target);

        // Read the result slot (if any) before restoring caller-clobbered
        // regs, because `result_dst` might alias one of the pushed regs.
        // We stash it into `result_scratch` and move it after restoration.
        if result_dst.is_some() {
            enc::load_64(
                &mut self.core.text,
                result_scratch,
                X86Reg::RSP,
                preserved_io::RET0 as i32 * 8,
            );
        }

        // Tear down the I/O area before popping caller-clobbered regs.
        enc::add_rsp_imm8(&mut self.core.text, PRESERVED_IO_BYTES);

        // Status lives in RAX and survives the dynamic restore because RAX is
        // backend-owned on x86_64, not part of the saved dynamic subset.
        self.restore_caller_clobbered_gp_dynamic();

        // If caller wanted the result, move it into the target dst now that
        // the stack is restored.
        if let Some(dst) = result_dst {
            if dst != result_scratch {
                enc::mov_rr_64(&mut self.core.text, dst, result_scratch);
            }
        }

        // Status check: non-zero means the helper trapped — branch to the
        // body-local error tail to propagate via the unified Return.
        enc::test_rr_64(&mut self.core.text, C_RET0, C_RET0);
        let body_local_error_label = self.core.body_local_error_label;
        self.emit_jcc(Cc::NE, body_local_error_label);
    }
}
