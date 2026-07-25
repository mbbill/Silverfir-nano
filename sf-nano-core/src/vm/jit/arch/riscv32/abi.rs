//! RV32 ABI policy over the shared RISC-V register plan.

use crate::{
    error::WasmError,
    vm::{
        backend::BackendConfig, jit::arch::common::scratch_pool::ScratchPool,
        jit::machine::machine_ir::MachineReg,
    },
};

use crate::vm::jit::arch::riscv::{
    abi as common,
    reg::{RiscvFpReg, RiscvReg},
};

pub(super) const GP_UNIT_BYTES: u8 = 4;
pub(super) const GP_SAVE_BYTES: u32 = 4;
pub(super) const FP_SAVE_BYTES: u32 = 8;
const SCALAR_CALL_SCRATCH_SLOTS: u16 = 8;

const fn align_up(value: u32, align: u32) -> u32 {
    value.div_ceil(align) * align
}

pub(super) const C_ARG0: RiscvReg = common::C_ARG0;
pub(super) const C_ARG1: RiscvReg = common::C_ARG1;
pub(super) const C_ARG2: RiscvReg = common::C_ARG2;
pub(super) const C_ARG3: RiscvReg = common::C_ARG3;
pub(super) const C_ARG4: RiscvReg = common::C_ARG4;
pub(super) const C_RET0: RiscvReg = common::C_RET0;
pub(super) const C_RET1: RiscvReg = common::C_RET1;

#[inline]
pub(crate) const fn compile_backend_config() -> BackendConfig {
    common::compile_backend_config(GP_UNIT_BYTES, SCALAR_CALL_SCRATCH_SLOTS)
}

pub(super) fn new_gp_scratch_pool() -> ScratchPool<RiscvReg, 6> {
    common::new_gp_scratch_pool()
}

pub(super) fn new_fp_scratch_pool() -> ScratchPool<RiscvFpReg, 2> {
    common::new_fp_scratch_pool()
}

#[inline]
pub(super) const fn max_fp_machine_regs() -> usize {
    common::max_fp_machine_regs()
}

#[inline]
pub(super) const fn max_total_machine_regs() -> usize {
    common::max_total_machine_regs()
}

#[inline]
pub(super) fn map_fixed_reg(reg: MachineReg) -> RiscvReg {
    common::map_fixed_reg(reg)
}

#[inline]
pub(super) fn map_reg(reg: MachineReg) -> Result<RiscvReg, WasmError> {
    common::map_reg(reg, compile_backend_config())
}

#[inline]
pub(super) fn map_fp_reg(reg: MachineReg) -> Result<RiscvFpReg, WasmError> {
    common::map_fp_reg(reg, compile_backend_config())
}

#[inline]
pub(super) fn callee_saved_regs() -> &'static [RiscvReg] {
    common::callee_saved_regs()
}

#[inline]
pub(super) const fn callee_saved_gp_count() -> usize {
    common::callee_saved_gp_count()
}

#[inline]
pub(super) fn callee_saved_fp_regs() -> &'static [RiscvFpReg] {
    common::callee_saved_fp_regs()
}

#[inline]
pub(super) const fn callee_saved_fp_count() -> usize {
    common::callee_saved_fp_count()
}

#[inline]
pub(super) fn gp_dynamic_caller_saved_regs() -> &'static [RiscvReg] {
    common::gp_dynamic_caller_saved_regs()
}

#[inline]
pub(super) fn fp_dynamic_caller_saved_regs() -> &'static [RiscvFpReg] {
    common::fp_dynamic_caller_saved_regs()
}

pub(super) const PRESERVED_HELPER_GP_OFFSET: u32 = common::preserved_io_size();

pub(super) const PRESERVED_HELPER_GP_SIZE: u32 =
    common::gp_dynamic_caller_saved_count() as u32 * GP_SAVE_BYTES;

pub(super) const PRESERVED_HELPER_FP_OFFSET: u32 = align_up(
    PRESERVED_HELPER_GP_OFFSET + PRESERVED_HELPER_GP_SIZE,
    FP_SAVE_BYTES,
);

pub(super) const PRESERVED_HELPER_FRAME_SIZE: u32 = {
    let raw =
        PRESERVED_HELPER_FP_OFFSET + common::fp_dynamic_caller_saved_count() as u32 * FP_SAVE_BYTES;
    raw.div_ceil(16) * 16
};

const _: () = assert!(PRESERVED_HELPER_FP_OFFSET % FP_SAVE_BYTES == 0);
const _: () = assert!(PRESERVED_HELPER_FRAME_SIZE % 16 == 0);

#[inline]
pub(super) const fn stack_reg() -> RiscvReg {
    common::stack_reg()
}

#[inline]
pub(super) const fn link_reg() -> RiscvReg {
    common::link_reg()
}

#[inline]
pub(super) const fn global_pointer_reg() -> RiscvReg {
    common::global_pointer_reg()
}

#[inline]
pub(super) const fn thread_pointer_reg() -> RiscvReg {
    common::thread_pointer_reg()
}

#[inline]
pub(super) const fn zero_reg() -> RiscvReg {
    common::zero_reg()
}
