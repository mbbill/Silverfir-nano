//! Shared RISC-V register plan.
//!
//! RV64 and RV32 use the same architectural register roles. Backend-specific
//! wrappers choose XLEN-sized frame math and MachineIR GP unit width.

use crate::{
    error::WasmError,
    vm::{
        backend::BackendConfig,
        jit::machine::machine_ir::{
            fp_reg_index, gp_dynamic_index, MachineReg, MACHINE_CTX_REG, MACHINE_FIXED_REG_COUNT,
            MACHINE_FP_REG, MACHINE_MEM0_BASE_REG, MACHINE_MEM0_SIZE_REG,
        },
        runtime::preserved::io as preserved_io,
    },
};

use super::reg::{RiscvFpReg, RiscvReg};
use crate::vm::jit::arch::common::scratch_pool::ScratchPool;

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
    gp_dynamic: &'static [RiscvReg],
    gp_dynamic_caller_saved: &'static [RiscvReg],
    gp_scratch: &'static [RiscvReg],
    callee_saved: &'static [RiscvReg],
    fp_dynamic: &'static [RiscvFpReg],
    fp_dynamic_caller_saved: &'static [RiscvFpReg],
    fp_scratch: &'static [RiscvFpReg],
    callee_saved_fp: &'static [RiscvFpReg],
}

#[cfg(sf_backend_riscv32)]
const GP_DYNAMIC: &[RiscvReg] = &[
    // Less-special caller-saved regs first: a3-a7, t3-t6. RV32 keeps t2,
    // gp, and tp in the scratch pool to support register-only i64 lowering.
    x(13),
    x(14),
    x(15),
    x(16),
    x(17),
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
];

#[cfg(sf_backend_riscv64)]
const GP_DYNAMIC: &[RiscvReg] = &[
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
];

#[cfg(sf_backend_riscv32)]
const GP_DYNAMIC_CALLER_SAVED: &[RiscvReg] = &[
    x(13),
    x(14),
    x(15),
    x(16),
    x(17),
    x(28),
    x(29),
    x(30),
    x(31),
    x(10),
    x(11),
    x(12),
];

#[cfg(sf_backend_riscv64)]
const GP_DYNAMIC_CALLER_SAVED: &[RiscvReg] = &[
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
];

#[cfg(sf_backend_riscv32)]
const GP_SCRATCH: &[RiscvReg] = &[
    x(5), // t0
    x(6), // t1
    x(1), // ra
    x(3), // gp
    x(4), // tp
    x(7), // t2
];

#[cfg(sf_backend_riscv64)]
const GP_SCRATCH: &[RiscvReg] = &[
    x(5), // t0
    x(6), // t1
    x(1), // ra
];

const REG_PLAN: RegPlan = RegPlan {
    // Keep fixed MachineIR roles in RISC-V callee-saved registers so host
    // helper calls cannot disturb them.
    ctx: x(18),       // s2
    fp: x(19),        // s3
    mem0_base: x(20), // s4
    mem0_size: x(21), // s5

    // Prefer caller-clobbered registers for short-lived values, matching the
    // ARM64 allocation strategy. `a0-a2` may alias the C ABI boundary because
    // boundary lowering publishes live state before setting call arguments.
    // Keep them after the less-special caller-clobbered regs so small functions
    // do not consume return/argument regs unnecessarily. Then use the
    // callee-saved bank for values that benefit from surviving C calls without
    // helper-side spills.
    gp_dynamic: GP_DYNAMIC,
    gp_dynamic_caller_saved: GP_DYNAMIC_CALLER_SAVED,
    gp_scratch: GP_SCRATCH,

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

    // FP register banks vanish when `sf_fp_dp` is off (no F/D extension on
    // the target — e.g. RV32IMAC for Pico 2 RV mode). Keep `fp_scratch` as
    // a 2-slot pool of placeholder regs so the backend's `ScratchPool<…, 2>`
    // type stays the same; the FP code paths are unreachable and stripped
    // by the linker. Mirrors the arm32/thumbm posture.
    #[cfg(sf_fp_dp)]
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
    #[cfg(not(sf_fp_dp))]
    fp_dynamic: &[],

    #[cfg(sf_fp_dp)]
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
    #[cfg(not(sf_fp_dp))]
    fp_dynamic_caller_saved: &[],

    fp_scratch: &[f(0), f(1)],

    #[cfg(sf_fp_dp)]
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
    #[cfg(not(sf_fp_dp))]
    callee_saved_fp: &[],
};

#[cfg(sf_backend_riscv32)]
const _: () = assert!(
    REG_PLAN.gp_dynamic.len() + REG_PLAN.gp_scratch.len() + 4 + 2 == 32,
    "RV32 GP register plan must account for all 32 x-registers"
);

#[cfg(sf_backend_riscv64)]
const _: () = assert!(
    REG_PLAN.gp_dynamic.len() + REG_PLAN.gp_scratch.len() + 4 + 4 == 32,
    "RISC-V GP register plan must account for all 32 x-registers"
);

// Compile-time check applies only to the F/D-enabled register plan. In the
// no-fp_dp build, `fp_dynamic` is intentionally empty.
#[cfg(sf_fp_dp)]
const _: () = assert!(
    REG_PLAN.fp_dynamic.len() + REG_PLAN.fp_scratch.len() == 32,
    "RISC-V FP register plan must account for all 32 f-registers"
);

pub(crate) const C_ARG0: RiscvReg = x(10); // a0
pub(crate) const C_ARG1: RiscvReg = x(11); // a1
pub(crate) const C_ARG2: RiscvReg = x(12); // a2
#[cfg(sf_backend_riscv32)]
pub(crate) const C_ARG3: RiscvReg = x(13); // a3
#[cfg(sf_backend_riscv32)]
pub(crate) const C_ARG4: RiscvReg = x(14); // a4
pub(crate) const C_RET0: RiscvReg = x(10); // a0
#[cfg(sf_backend_riscv32)]
pub(crate) const C_RET1: RiscvReg = x(11); // a1

#[inline]
pub(crate) const fn compile_backend_config(
    gp_unit_bytes: u8,
    call_scratch_slots: u16,
) -> BackendConfig {
    BackendConfig::new(
        gp_unit_bytes,
        REG_PLAN.gp_dynamic.len() as u8,
        REG_PLAN.fp_dynamic.len() as u8,
        call_scratch_slots,
    )
}

#[cfg(sf_backend_riscv32)]
pub(crate) fn new_gp_scratch_pool() -> ScratchPool<RiscvReg, 6> {
    ScratchPool::new([
        REG_PLAN.gp_scratch[0],
        REG_PLAN.gp_scratch[1],
        REG_PLAN.gp_scratch[2],
        REG_PLAN.gp_scratch[3],
        REG_PLAN.gp_scratch[4],
        REG_PLAN.gp_scratch[5],
    ])
}

#[cfg(sf_backend_riscv64)]
pub(crate) fn new_gp_scratch_pool() -> ScratchPool<RiscvReg, 3> {
    ScratchPool::new([
        REG_PLAN.gp_scratch[0],
        REG_PLAN.gp_scratch[1],
        REG_PLAN.gp_scratch[2],
    ])
}

pub(crate) fn new_fp_scratch_pool() -> ScratchPool<RiscvFpReg, 2> {
    ScratchPool::new([REG_PLAN.fp_scratch[0], REG_PLAN.fp_scratch[1]])
}

pub(crate) const FP_MACHINE_REG_COUNT: usize = REG_PLAN.fp_dynamic.len();

#[inline]
pub(crate) const fn max_gp_mapped_regs() -> usize {
    MACHINE_FIXED_REG_COUNT as usize + REG_PLAN.gp_dynamic.len()
}

#[inline]
pub(crate) const fn max_fp_machine_regs() -> usize {
    FP_MACHINE_REG_COUNT
}

#[inline]
pub(crate) const fn max_total_machine_regs() -> usize {
    max_gp_mapped_regs() + max_fp_machine_regs()
}

#[inline]
pub(crate) fn map_fixed_reg(reg: MachineReg) -> RiscvReg {
    match reg {
        MACHINE_CTX_REG => REG_PLAN.ctx,
        MACHINE_FP_REG => REG_PLAN.fp,
        MACHINE_MEM0_BASE_REG => REG_PLAN.mem0_base,
        MACHINE_MEM0_SIZE_REG => REG_PLAN.mem0_size,
        _ => unreachable!("not a fixed machine reg"),
    }
}

#[inline]
pub(crate) fn map_reg(reg: MachineReg, config: BackendConfig) -> Result<RiscvReg, WasmError> {
    if reg.0 < MACHINE_FIXED_REG_COUNT {
        return Ok(map_fixed_reg(reg));
    }
    gp_dynamic_index(reg, config)
        .and_then(|index| REG_PLAN.gp_dynamic.get(index).copied())
        .ok_or_else(|| WasmError::invalid("riscv has no GP mapping for machine reg"))
}

#[inline]
pub(crate) fn fp_machine_reg(index: usize) -> Option<RiscvFpReg> {
    REG_PLAN.fp_dynamic.get(index).copied()
}

#[inline]
pub(crate) fn map_fp_reg(reg: MachineReg, config: BackendConfig) -> Result<RiscvFpReg, WasmError> {
    let index = fp_reg_index(reg, config)
        .ok_or_else(|| WasmError::invalid("expected FP register, got machine reg"))?;
    fp_machine_reg(index)
        .ok_or_else(|| WasmError::invalid("riscv has no FP mapping for machine reg"))
}

#[inline]
pub(crate) fn callee_saved_regs() -> &'static [RiscvReg] {
    REG_PLAN.callee_saved
}

#[inline]
pub(crate) const fn callee_saved_gp_count() -> usize {
    REG_PLAN.callee_saved.len()
}

#[inline]
pub(crate) fn callee_saved_fp_regs() -> &'static [RiscvFpReg] {
    REG_PLAN.callee_saved_fp
}

#[inline]
pub(crate) const fn callee_saved_fp_count() -> usize {
    REG_PLAN.callee_saved_fp.len()
}

#[inline]
pub(crate) fn gp_dynamic_caller_saved_regs() -> &'static [RiscvReg] {
    REG_PLAN.gp_dynamic_caller_saved
}

#[inline]
pub(crate) const fn gp_dynamic_caller_saved_count() -> usize {
    REG_PLAN.gp_dynamic_caller_saved.len()
}

#[inline]
pub(crate) fn fp_dynamic_caller_saved_regs() -> &'static [RiscvFpReg] {
    REG_PLAN.fp_dynamic_caller_saved
}

#[inline]
pub(crate) const fn fp_dynamic_caller_saved_count() -> usize {
    REG_PLAN.fp_dynamic_caller_saved.len()
}

#[inline]
pub(crate) const fn preserved_io_size() -> u32 {
    preserved_io::SLOT_COUNT as u32 * 8
}

#[inline]
pub(crate) const fn stack_reg() -> RiscvReg {
    RiscvReg::SP
}

#[inline]
pub(crate) const fn link_reg() -> RiscvReg {
    RiscvReg::RA
}

#[inline]
#[cfg(sf_backend_riscv32)]
pub(crate) const fn global_pointer_reg() -> RiscvReg {
    x(3)
}

#[inline]
#[cfg(sf_backend_riscv32)]
pub(crate) const fn thread_pointer_reg() -> RiscvReg {
    x(4)
}

#[inline]
pub(crate) const fn zero_reg() -> RiscvReg {
    RiscvReg::ZERO
}
