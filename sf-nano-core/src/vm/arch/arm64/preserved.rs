//! Preserved-helper frame: save/restore all caller-clobbered JIT registers
//! around a C runtime helper call.  Instruction lowering code in `inst.rs`
//! calls these helpers when it needs to make a preserved call.

use crate::error::WasmError;
use crate::vm::machine::machine_ir::{MachineValue, MACHINE_CTX_REG};

use super::{abi, enc};
use super::backend::Arm64Backend;
use super::inst::{materialize_u64_into, prepare_gp};
use super::reg::Arm64Reg;

impl<'a> Arm64Backend<'a> {

// ── Frame open / close ──────────────────────────────────────────────────

/// Open the preserved-helper frame: allocate stack space, save all
/// caller-clobbered JIT registers.  After this, the I/O area is at SP+0
/// and all JIT register values are preserved on the native stack.
pub(super) fn emit_preserved_frame_open(&mut self) {
    self.core.text.emit_u32(enc::sub_imm_64(
        Arm64Reg::SP, Arm64Reg::SP, abi::PRESERVED_HELPER_FRAME_SIZE,
    ));
    self.emit_save_preserved_gp();
    self.emit_save_preserved_fp();
}

/// Store a u32 immediate into an I/O slot (uses X16 as scratch).
pub(super) fn emit_io_store_imm(&mut self, slot: usize, value: u32) {
    materialize_u64_into(&mut self.core.text, Arm64Reg::X16, value as u64);
    self.core.text.emit_u32(enc::str_64(Arm64Reg::X16, Arm64Reg::SP, slot as u32));
}

/// Store a MachineValue into an I/O slot.
pub(super) fn emit_io_store_value(&mut self, slot: usize, value: MachineValue) -> Result<(), WasmError> {
    let gp = prepare_gp(
        self.core.compiled.backend(), &self.core.fp_reg_widths,
        &mut self.core.text, &self.gp_scratch, value,
    )?.release();
    self.core.text.emit_u32(enc::str_64(gp, Arm64Reg::SP, slot as u32));
    Ok(())
}

/// Emit the BLR call to preserved_helper, stash status in X16 and result
/// in X17, restore all JIT registers, deallocate the frame, and branch
/// to the error path on nonzero status.
pub(super) fn emit_preserved_call_and_close(&mut self, op_code: u32) {
    use crate::vm::runtime::helpers::{preserved_helper, preserved_io};

    // Set up C calling convention: x0=ctx, x1=op_code, x2=io (=SP).
    self.core.text.emit_u32(enc::mov_reg_64(
        abi::C_ARG0, abi::map_fixed_reg(MACHINE_CTX_REG),
    ));
    materialize_u64_into(&mut self.core.text, abi::C_ARG1, op_code as u64);
    self.core.text.emit_u32(enc::add_imm_64(abi::C_ARG2, Arm64Reg::SP, 0));
    let call_scratch = Arm64Reg::X17;
    materialize_u64_into(&mut self.core.text, call_scratch, preserved_helper as usize as u64);
    self.core.text.emit_u32(enc::blr(call_scratch));

    // Stash status and result in scratch regs before restoring.
    self.core.text.emit_u32(enc::mov_reg_64(Arm64Reg::X16, abi::C_RET0));
    self.core.text.emit_u32(enc::ldr_64(
        Arm64Reg::X17, Arm64Reg::SP, preserved_io::RET0 as u32,
    ));

    // Restore all caller-clobbered JIT registers.
    self.emit_restore_preserved_fp();
    self.emit_restore_preserved_gp();

    // Deallocate frame.
    self.core.text.emit_u32(enc::add_imm_64(
        Arm64Reg::SP, Arm64Reg::SP, abi::PRESERVED_HELPER_FRAME_SIZE,
    ));

    // Check status.
    let return_error_label = self.core.return_error_label;
    self.lower_cbnz(Arm64Reg::X16, return_error_label);
}

// ── Register save/restore ───────────────────────────────────────────────

fn emit_save_preserved_gp(&mut self) {
    let base_off = abi::PRESERVED_HELPER_GP_OFFSET;
    let mut slot = 0u32;
    for regs in [abi::gp_transient_regs(), abi::gp_caller_saved_cache()] {
        let mut i = 0;
        while i + 1 < regs.len() {
            self.core.text.emit_u32(enc::stp_64(
                regs[i], regs[i + 1], Arm64Reg::SP,
                ((base_off + slot * 8) / 8) as i32,
            ));
            slot += 2;
            i += 2;
        }
        if i < regs.len() {
            self.core.text.emit_u32(enc::str_64(
                regs[i], Arm64Reg::SP, (base_off + slot * 8) / 8,
            ));
            slot += 1;
        }
    }
}

fn emit_restore_preserved_gp(&mut self) {
    let base_off = abi::PRESERVED_HELPER_GP_OFFSET;
    let mut slot = 0u32;
    for regs in [abi::gp_transient_regs(), abi::gp_caller_saved_cache()] {
        let mut i = 0;
        while i + 1 < regs.len() {
            self.core.text.emit_u32(enc::ldp_64(
                regs[i], regs[i + 1], Arm64Reg::SP,
                ((base_off + slot * 8) / 8) as i32,
            ));
            slot += 2;
            i += 2;
        }
        if i < regs.len() {
            self.core.text.emit_u32(enc::ldr_64(
                regs[i], Arm64Reg::SP, (base_off + slot * 8) / 8,
            ));
            slot += 1;
        }
    }
}

fn emit_save_preserved_fp(&mut self) {
    let base_off = abi::PRESERVED_HELPER_FP_OFFSET;
    let mut slot = 0u32;
    for regs in [abi::fp_transient_regs(), abi::fp_caller_saved_cache()] {
        for reg in regs.iter().copied() {
            self.core.text.emit_u32(enc::str_d(
                reg, Arm64Reg::SP, (base_off + slot * 8) / 8,
            ));
            slot += 1;
        }
    }
}

fn emit_restore_preserved_fp(&mut self) {
    let base_off = abi::PRESERVED_HELPER_FP_OFFSET;
    let mut slot = 0u32;
    for regs in [abi::fp_transient_regs(), abi::fp_caller_saved_cache()] {
        for reg in regs.iter().copied() {
            self.core.text.emit_u32(enc::ldr_d(
                reg, Arm64Reg::SP, (base_off + slot * 8) / 8,
            ));
            slot += 1;
        }
    }
}

} // impl Arm64Backend (preserved.rs)
