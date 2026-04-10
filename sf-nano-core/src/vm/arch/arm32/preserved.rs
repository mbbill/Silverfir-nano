//! Preserved-helper bridge: marshal bulk-memory/table ops through the shared
//! `preserved_entry` runtime helper.
//!
//! The arm64 backend has a dedicated preserved-helper frame that saves all
//! caller-clobbered JIT registers before the call. arm32 piggy-backs on the
//! existing `emit_host_call` path, which already saves D3-D7 via the
//! per-function `helper_scratch` slots; we additionally spill R0-R3 and R9.
//!
//! Layout while the helper runs:
//!
//!   sp +  0  ─┐  io[0..SLOT_COUNT] (`io::SLOT_COUNT × 8` bytes: IMM0, IMM1,
//!              │                   ARG0, ARG1, ARG2, RET0, scratch, scratch)
//!   sp + 64  ─┘
//!   sp + 64  ─┐  R0..R3, R9 spill area (24 bytes)
//!   sp + 88  ─┘

use crate::{
    error::WasmError,
    vm::machine::machine_ir::{MachineReg, MachineValue, MACHINE_CTX_REG},
};

use super::{
    abi::{map_fixed_reg, map_reg},
    backend::{Arm32Backend, BranchFixupKind},
    enc,
    inst::emit_load_u32_into,
    reg::Arm32Reg,
};

const PRESERVED_IO_BYTES: u32 = crate::vm::runtime::preserved::io::SLOT_COUNT as u32 * 8;
const PRESERVED_GP_SPILL_BYTES: u32 = 24;

impl<'a> Arm32Backend<'a> {
    /// Encode a u32 immediate write into io[slot] (low + zero high half).
    fn emit_preserved_io_store_imm(&mut self, slot: usize, value: u32) {
        let s = self.gp_scratch.scoped_alloc();
        emit_load_u32_into(&mut self.core.text, *s, value);
        self.core
            .text
            .emit_u32(enc::str_imm(*s, Arm32Reg::SP, (slot * 8) as i32));
        emit_load_u32_into(&mut self.core.text, *s, 0);
        self.core
            .text
            .emit_u32(enc::str_imm(*s, Arm32Reg::SP, (slot * 8 + 4) as i32));
    }

    /// Encode a `MachineValue` write into io[slot]. Wasm-side values fit in
    /// 32 bits on this target, so the high half of the slot is zero.
    fn emit_preserved_io_store_value(
        &mut self,
        slot: usize,
        value: MachineValue,
    ) -> Result<(), WasmError> {
        match value {
            MachineValue::Reg(r) => {
                let src = map_reg(r)?;
                self.core
                    .text
                    .emit_u32(enc::str_imm(src, Arm32Reg::SP, (slot * 8) as i32));
                let s = self.gp_scratch.scoped_alloc();
                emit_load_u32_into(&mut self.core.text, *s, 0);
                self.core
                    .text
                    .emit_u32(enc::str_imm(*s, Arm32Reg::SP, (slot * 8 + 4) as i32));
            }
            MachineValue::Imm64(v) => {
                let s = self.gp_scratch.scoped_alloc();
                emit_load_u32_into(&mut self.core.text, *s, v as u32);
                self.core
                    .text
                    .emit_u32(enc::str_imm(*s, Arm32Reg::SP, (slot * 8) as i32));
                emit_load_u32_into(&mut self.core.text, *s, (v >> 32) as u32);
                self.core
                    .text
                    .emit_u32(enc::str_imm(*s, Arm32Reg::SP, (slot * 8 + 4) as i32));
            }
            MachineValue::ReservedReg(reg) => {
                return Err(WasmError::internal(alloc::format!(
                    "arm32 preserved-helper cannot consume reserved cache register {} as a real value",
                    reg.0
                )));
            }
        }
        Ok(())
    }

    /// Lower a preserved-helper call. Allocates the I/O area, populates the
    /// IMM/ARG slots, dispatches `preserved_entry(ctx, op_code, io_ptr)`, and
    /// propagates errors via `body_local_error_label`. If `result_dst` is
    /// `Some`, the helper's RET0 slot is moved into that GP register on the
    /// success path.
    pub(super) fn emit_preserved_helper_call(
        &mut self,
        op_code: u32,
        imms: &[(usize, u32)],
        args: &[(usize, MachineValue)],
        result_dst: Option<MachineReg>,
    ) -> Result<(), WasmError> {
        use crate::vm::runtime::preserved::{io as preserved_io, preserved_entry};

        // Step 1: spill R0..R3, R9 onto the host stack so the JIT register
        // file survives the helper call. After this `sp -= 24` and the saved
        // area lives at sp[0..20].
        self.spill_caller_saved_gp_regs();

        // Step 2: allocate the I/O area below the spilled GP region. After
        // this, sp[0..PRESERVED_IO_BYTES] = io area and
        // sp[PRESERVED_IO_BYTES..PRESERVED_IO_BYTES+24] = saved GPs.
        self.core.text.emit_u32(enc::sub_imm(
            Arm32Reg::SP,
            Arm32Reg::SP,
            PRESERVED_IO_BYTES,
            0,
        ));

        // Step 3: populate IMM and ARG slots. R0..R3 / R9 are still live
        // here (the spill only copied them; it did not modify them), so
        // `map_reg` lookups for ARG MachineValues read the correct physical
        // registers.
        for &(slot, imm) in imms {
            self.emit_preserved_io_store_imm(slot, imm);
        }
        for &(slot, value) in args {
            self.emit_preserved_io_store_value(slot, value)?;
        }

        // Step 4: set up the C calling convention. r0 = ctx, r1 = op_code,
        // r2 = io_ptr (= sp). `emit_host_call` below will save/restore D3-D7
        // around the BLX via the per-function helper_scratch.
        self.core
            .text
            .emit_u32(enc::mov_reg(Arm32Reg::R0, map_fixed_reg(MACHINE_CTX_REG)));
        self.emit_load_u32(Arm32Reg::R1, op_code);
        self.core
            .text
            .emit_u32(enc::mov_reg(Arm32Reg::R2, Arm32Reg::SP));

        // Step 5: dispatch the helper.
        self.emit_host_call(preserved_entry as usize);

        // Step 6: stash status (R0) into SCRATCH0 (R12). After this we commit
        // to two diverging tails — success and error — that need to restore the
        // JIT register file in different ways. Note that R0 itself is not yet
        // clobbered: the success-path restore will overwrite it from the spill
        // area, but the error-path restore skips R0 so the helper's status code
        // propagates back to the C caller via the body-local error tail.
        self.core
            .text
            .emit_u32(enc::mov_reg(Arm32Reg::R12, Arm32Reg::R0));

        // Step 7: read the result (if any) into SCRATCH1 (R14 / LR). LR is
        // free at this point — the body prelude saved the body's real LR onto
        // the host stack at function entry, and the success path will never
        // need LR before the next instruction-level scratch alloc overwrites
        // it. (The error tail loads its own LR before `bx`.)
        if result_dst.is_some() {
            self.core.text.emit_u32(enc::ldr_imm(
                Arm32Reg::R14,
                Arm32Reg::SP,
                (preserved_io::RET0 * 8) as i32,
            ));
        }

        // Step 8: free the I/O area. Both tails need this.
        self.core.text.emit_u32(enc::add_imm(
            Arm32Reg::SP,
            Arm32Reg::SP,
            PRESERVED_IO_BYTES,
            0,
        ));

        // Step 9: branch on status. The two tails differ in how they pop the
        // spill area: the success path runs `restore_caller_saved_gp_regs`
        // (which restores R0..R3, R9 from the spill and pops 24 bytes), while
        // the error path uses a bare `add sp, #24` so R0 retains the helper's
        // status code while it propagates to `body_local_error_label`.
        let error_path = self.core.new_label();
        self.core.text.emit_u32(enc::cmp_imm(Arm32Reg::R12, 0, 0));
        self.emit_branch(BranchFixupKind::BCond(enc::Cond::Ne), error_path);

        // Success tail.
        self.restore_caller_saved_gp_regs(&[]);
        if let Some(dst) = result_dst {
            let dst_hw = map_reg(dst)?;
            self.core.text.emit_u32(enc::mov_reg(dst_hw, Arm32Reg::R14));
        }
        let after = self.core.new_label();
        self.emit_branch(BranchFixupKind::B, after);

        // Error tail.
        self.core.bind_label(error_path);
        self.core.text.emit_u32(enc::add_imm(
            Arm32Reg::SP,
            Arm32Reg::SP,
            PRESERVED_GP_SPILL_BYTES,
            0,
        ));
        let body_local_error_label = self.core.body_local_error_label;
        self.emit_branch(BranchFixupKind::B, body_local_error_label);

        // Continuation.
        self.core.bind_label(after);
        Ok(())
    }
}
