//! RV64 register plan.

use crate::{
    error::WasmError,
    vm::{
        backend::BackendConfig,
        machine::machine_ir::{
            fp_reg_index, gp_dynamic_index, MachineReg, MACHINE_CTX_REG, MACHINE_FIXED_REG_COUNT,
            MACHINE_FP_REG, MACHINE_MEM0_BASE_REG, MACHINE_MEM0_SIZE_REG,
        },
        runtime::preserved::io as preserved_io,
    },
};

use super::reg::{RiscvFpReg, RiscvReg};
use crate::vm::arch::common::scratch_pool::ScratchPool;

#[inline]
const fn x(index: u8) -> RiscvReg {
    RiscvReg::from_raw(index)
}

#[inline]
const fn f(index: u8) -> RiscvFpReg {
    RiscvFpReg::from_raw(index)
}

struct RegPlan {
    ctx: RiscvReg,
    fp: RiscvReg,
    mem0_base: RiscvReg,
    mem0_size: RiscvReg,
    gp_unit_bytes: u8,
    gp_dynamic: &'static [RiscvReg],
    gp_dynamic_caller_saved: &'static [RiscvReg],
    gp_scratch: &'static [RiscvReg],
    callee_saved: &'static [RiscvReg],
    fp_dynamic: &'static [RiscvFpReg],
    fp_dynamic_caller_saved: &'static [RiscvFpReg],
    fp_scratch: &'static [RiscvFpReg],
    callee_saved_fp: &'static [RiscvFpReg],
}

const REG_PLAN: RegPlan = RegPlan {
    // Keep fixed MachineIR roles in RISC-V callee-saved registers so host
    // helper calls cannot disturb them.
    ctx: x(18),       // s2
    fp: x(19),        // s3
    mem0_base: x(20), // s4
    mem0_size: x(21), // s5

    gp_unit_bytes: 8,

    // Prefer caller-clobbered registers for short-lived values, matching the
    // ARM64 allocation strategy. `a0-a2` may alias the C ABI boundary because
    // boundary lowering publishes live state before setting call arguments.
    // Keep them after the less-special caller-clobbered regs so small functions
    // do not consume return/argument regs unnecessarily. Then use the
    // callee-saved bank for values that benefit from surviving C calls without
    // helper-side spills.
    gp_dynamic: &[
        // Less-special caller-saved regs first: a3-a7, t2, t3-t6. The
        // current C-boundary lowering stages directly through a0-a2.
        x(13),
        x(14),
        x(15),
        x(16),
        x(17),
        x(7),
        x(28),
        x(29),
        x(30),
        x(31),
        // Caller-saved C argument/result registers are valid dynamic regs, but
        // come after the less-special caller-saved bank.
        x(10),
        x(11),
        x(12),
        // Callee-saved dynamic bank: s0-s1, s6-s11.
        x(8),
        x(9),
        x(22),
        x(23),
        x(24),
        x(25),
        x(26),
        x(27),
    ],
    gp_dynamic_caller_saved: &[
        x(13),
        x(14),
        x(15),
        x(16),
        x(17),
        x(7),
        x(28),
        x(29),
        x(30),
        x(31),
        x(10),
        x(11),
        x(12),
    ],
    gp_scratch: &[x(5), x(6)], // t0-t1

    // Save RA because public-entry calls overwrite it before the C ABI
    // epilogue returns to Rust. Save dynamic s0-s1, the fixed s2-s5 roles,
    // and dynamic s6-s11.
    callee_saved: &[
        x(1),
        x(8),
        x(9),
        x(18),
        x(19),
        x(20),
        x(21),
        x(22),
        x(23),
        x(24),
        x(25),
        x(26),
        x(27),
    ],

    fp_dynamic: &[
        f(10),
        f(11),
        f(12),
        f(13),
        f(14),
        f(15),
        f(16),
        f(17),
        f(2),
        f(3),
        f(4),
        f(5),
        f(6),
        f(7),
        f(28),
        f(29),
        f(30),
        f(31),
        f(8),
        f(9),
        f(18),
        f(19),
        f(20),
        f(21),
        f(22),
        f(23),
        f(24),
        f(25),
        f(26),
        f(27),
    ],
    fp_dynamic_caller_saved: &[
        f(10),
        f(11),
        f(12),
        f(13),
        f(14),
        f(15),
        f(16),
        f(17),
        f(2),
        f(3),
        f(4),
        f(5),
        f(6),
        f(7),
        f(28),
        f(29),
        f(30),
        f(31),
    ],
    fp_scratch: &[f(0), f(1)],
    callee_saved_fp: &[
        f(8),
        f(9),
        f(18),
        f(19),
        f(20),
        f(21),
        f(22),
        f(23),
        f(24),
        f(25),
        f(26),
        f(27),
    ],
};

const _: () = assert!(
    REG_PLAN.gp_dynamic.len() + REG_PLAN.gp_scratch.len() + 4 + 5 == 32,
    "RV64 GP register plan must account for all 32 x-registers"
);

const _: () = assert!(
    REG_PLAN.fp_dynamic.len() + REG_PLAN.fp_scratch.len() == 32,
    "RV64 FP register plan must account for all 32 f-registers"
);

pub(super) const C_ARG0: RiscvReg = x(10); // a0
pub(super) const C_ARG1: RiscvReg = x(11); // a1
pub(super) const C_ARG2: RiscvReg = x(12); // a2
pub(super) const C_RET0: RiscvReg = x(10); // a0

const SCALAR_CALL_SCRATCH_SLOTS: u16 = 3;

#[inline]
pub(crate) const fn compile_backend_config() -> BackendConfig {
    BackendConfig::new(
        REG_PLAN.gp_unit_bytes,
        REG_PLAN.gp_dynamic.len() as u8,
        REG_PLAN.fp_dynamic.len() as u8,
        SCALAR_CALL_SCRATCH_SLOTS,
    )
}

pub(super) fn new_gp_scratch_pool() -> ScratchPool<RiscvReg, 2> {
    ScratchPool::new([REG_PLAN.gp_scratch[0], REG_PLAN.gp_scratch[1]])
}

pub(super) fn new_fp_scratch_pool() -> ScratchPool<RiscvFpReg, 2> {
    ScratchPool::new([REG_PLAN.fp_scratch[0], REG_PLAN.fp_scratch[1]])
}

pub(super) const FP_MACHINE_REG_COUNT: usize = REG_PLAN.fp_dynamic.len();

#[inline]
pub(super) const fn max_gp_mapped_regs() -> usize {
    MACHINE_FIXED_REG_COUNT as usize + REG_PLAN.gp_dynamic.len()
}

#[inline]
pub(super) const fn max_fp_machine_regs() -> usize {
    FP_MACHINE_REG_COUNT
}

#[inline]
pub(super) const fn max_total_machine_regs() -> usize {
    max_gp_mapped_regs() + max_fp_machine_regs()
}

#[inline]
pub(super) fn map_fixed_reg(reg: MachineReg) -> RiscvReg {
    match reg {
        MACHINE_CTX_REG => REG_PLAN.ctx,
        MACHINE_FP_REG => REG_PLAN.fp,
        MACHINE_MEM0_BASE_REG => REG_PLAN.mem0_base,
        MACHINE_MEM0_SIZE_REG => REG_PLAN.mem0_size,
        _ => unreachable!("not a fixed machine reg"),
    }
}

#[inline]
pub(super) fn map_reg(reg: MachineReg) -> Result<RiscvReg, WasmError> {
    if reg.0 < MACHINE_FIXED_REG_COUNT {
        return Ok(map_fixed_reg(reg));
    }
    let config = compile_backend_config();
    gp_dynamic_index(reg, config)
        .and_then(|index| REG_PLAN.gp_dynamic.get(index).copied())
        .ok_or_else(|| WasmError::invalid("riscv64 has no GP mapping for machine reg"))
}

#[inline]
pub(super) fn fp_machine_reg(index: usize) -> Option<RiscvFpReg> {
    REG_PLAN.fp_dynamic.get(index).copied()
}

#[inline]
pub(super) fn map_fp_reg(reg: MachineReg) -> Result<RiscvFpReg, WasmError> {
    let config = compile_backend_config();
    let index = fp_reg_index(reg, config)
        .ok_or_else(|| WasmError::invalid("expected FP register, got machine reg"))?;
    fp_machine_reg(index)
        .ok_or_else(|| WasmError::invalid("riscv64 has no FP mapping for machine reg"))
}

#[inline]
pub(super) fn callee_saved_regs() -> &'static [RiscvReg] {
    REG_PLAN.callee_saved
}

#[inline]
pub(super) const fn callee_saved_gp_count() -> usize {
    REG_PLAN.callee_saved.len()
}

#[inline]
pub(super) fn callee_saved_fp_regs() -> &'static [RiscvFpReg] {
    REG_PLAN.callee_saved_fp
}

#[inline]
pub(super) const fn callee_saved_fp_count() -> usize {
    REG_PLAN.callee_saved_fp.len()
}

#[inline]
pub(super) fn gp_dynamic_caller_saved_regs() -> &'static [RiscvReg] {
    REG_PLAN.gp_dynamic_caller_saved
}

#[inline]
pub(super) fn fp_dynamic_caller_saved_regs() -> &'static [RiscvFpReg] {
    REG_PLAN.fp_dynamic_caller_saved
}

const fn preserved_io_size() -> u32 {
    preserved_io::SLOT_COUNT as u32 * 8
}

pub(super) const PRESERVED_HELPER_GP_OFFSET: u32 = preserved_io_size();

pub(super) const PRESERVED_HELPER_FP_OFFSET: u32 =
    PRESERVED_HELPER_GP_OFFSET + REG_PLAN.gp_dynamic_caller_saved.len() as u32 * 8;

pub(super) const PRESERVED_HELPER_FRAME_SIZE: u32 = {
    let raw = PRESERVED_HELPER_FP_OFFSET + REG_PLAN.fp_dynamic_caller_saved.len() as u32 * 8;
    raw.div_ceil(16) * 16
};

#[inline]
pub(super) const fn stack_reg() -> RiscvReg {
    RiscvReg::SP
}

#[inline]
pub(super) const fn link_reg() -> RiscvReg {
    RiscvReg::RA
}

#[inline]
pub(super) const fn zero_reg() -> RiscvReg {
    RiscvReg::ZERO
}
