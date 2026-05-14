//! ARM64 register plan — single source of truth for register allocation.
//!
//! `REG_PLAN` declares every register role in one place. `BackendConfig` is
//! derived from the array lengths, so budgets can never drift out of sync with
//! the physical register tables. The raw plan stays private to this module;
//! backend code must come through the accessors below.

use crate::{
    error::WasmError,
    vm::{
        backend::BackendConfig,
        machine::machine_ir::{
            gp_dynamic_index, MachineReg, MACHINE_CTX_REG, MACHINE_FIXED_REG_COUNT, MACHINE_FP_REG,
            MACHINE_MEM0_BASE_REG, MACHINE_MEM0_SIZE_REG,
        },
        runtime::preserved::io as preserved_io,
    },
};

use super::reg::{Arm64FpReg, Arm64Reg};
use crate::vm::arch::common::scratch_pool::ScratchPool;

#[inline]
const fn gp(index: u8) -> Arm64Reg {
    Arm64Reg::from_raw(index)
}

#[inline]
const fn fp(index: u8) -> Arm64FpReg {
    Arm64FpReg::from_raw(index)
}

// ── Register plan ────────────────────────────────────────────────────────────

// AAPCS64 only preserves the low 64 bits of V8-V15 across C calls. Scalar
// builds therefore omit them from the preserved-helper caller-saved set and
// rely on the ABI. SIMD builds let those regs carry full v128 values, so the
// preserved-helper wrapper must save/restore V8-V15 explicitly to protect the
// upper 64 bits across helper calls.
#[cfg(sf_has_simd)]
const FP_DYNAMIC_CALLER_SAVED: &[Arm64FpReg] = &[
    fp(3),
    fp(4),
    fp(5),
    fp(6),
    fp(7),
    fp(2),
    fp(16),
    fp(17),
    fp(18),
    fp(19),
    fp(20),
    fp(8),
    fp(9),
    fp(10),
    fp(11),
    fp(12),
    fp(13),
    fp(14),
    fp(15),
    fp(21),
    fp(22),
    fp(23),
    fp(24),
    fp(25),
    fp(26),
    fp(27),
    fp(28),
    fp(29),
    fp(30),
    fp(31),
];

#[cfg(not(sf_has_simd))]
const FP_DYNAMIC_CALLER_SAVED: &[Arm64FpReg] = &[
    fp(3),
    fp(4),
    fp(5),
    fp(6),
    fp(7),
    fp(2),
    fp(16),
    fp(17),
    fp(18),
    fp(19),
    fp(20),
    fp(21),
    fp(22),
    fp(23),
    fp(24),
    fp(25),
    fp(26),
    fp(27),
    fp(28),
    fp(29),
    fp(30),
    fp(31),
];

struct RegPlan {
    // Fixed MachineIR roles
    ctx: Arm64Reg,
    fp: Arm64Reg,
    mem0_base: Arm64Reg,
    mem0_size: Arm64Reg,
    // GP budget unit size
    gp_unit_bytes: u8,
    // Preferred GP dynamic order. This is an allocation-order preference only;
    // ownership stays in lowering state, not the machine-register number.
    gp_dynamic: &'static [Arm64Reg],
    // Caller-clobbered subset of the GP dynamic bank. Preserved-helper
    // wrappers save exactly these because callee-saved dynamic regs already
    // survive the C ABI boundary.
    gp_dynamic_caller_saved: &'static [Arm64Reg],
    gp_scratch: &'static [Arm64Reg],
    // Preferred FP dynamic order. As with GP, this is only allocation order.
    fp_dynamic: &'static [Arm64FpReg],
    // Caller-clobbered subset of the FP dynamic bank.
    fp_dynamic_caller_saved: &'static [Arm64FpReg],
    fp_scratch: &'static [Arm64FpReg],
    // Callee-saved sets
    callee_saved_gp_pairs: &'static [(Arm64Reg, Arm64Reg)],
    callee_saved_fp: &'static [Arm64FpReg],
    // Stack
    stack_alignment_bytes: u32,
}

const REG_PLAN: RegPlan = RegPlan {
    ctx: gp(19),
    fp: gp(20),
    mem0_base: gp(21),
    mem0_size: gp(22),

    gp_unit_bytes: 8,

    gp_dynamic: &[
        gp(3),
        gp(4),
        gp(5),
        gp(6),
        gp(7),
        gp(8),
        gp(0),
        gp(1),
        gp(2),
        gp(9),
        gp(10),
        gp(11),
        gp(12),
        gp(13),
        gp(14),
        gp(23),
        gp(24),
        gp(25),
        gp(26),
        gp(27),
        gp(28),
        gp(15),
    ],
    gp_dynamic_caller_saved: &[
        gp(3),
        gp(4),
        gp(5),
        gp(6),
        gp(7),
        gp(8),
        gp(0),
        gp(1),
        gp(2),
        gp(9),
        gp(10),
        gp(11),
        gp(12),
        gp(13),
        gp(14),
        gp(15),
    ],
    gp_scratch: &[gp(16), gp(17)],

    fp_dynamic: &[
        fp(3),
        fp(4),
        fp(5),
        fp(6),
        fp(7),
        fp(2),
        fp(16),
        fp(17),
        fp(18),
        fp(19),
        fp(20),
        fp(21),
        fp(22),
        fp(23),
        fp(24),
        fp(25),
        fp(26),
        fp(27),
        fp(28),
        fp(29),
        fp(30),
        fp(31),
        fp(8),
        fp(9),
        fp(10),
        fp(11),
        fp(12),
        fp(13),
        fp(14),
        fp(15),
    ],
    fp_dynamic_caller_saved: FP_DYNAMIC_CALLER_SAVED,
    fp_scratch: &[fp(0), fp(1)],

    callee_saved_gp_pairs: &[
        (gp(19), gp(20)),
        (gp(21), gp(22)),
        (gp(23), gp(24)),
        (gp(25), gp(26)),
        (gp(27), gp(28)),
        (gp(29), gp(30)),
    ],
    callee_saved_fp: &[fp(8), fp(9), fp(10), fp(11), fp(12), fp(13), fp(14), fp(15)],

    stack_alignment_bytes: 16,
};

// Compile-time checks
const _: () = assert!(
    REG_PLAN.fp_dynamic.len() + REG_PLAN.fp_scratch.len() == 32,
    "FP register plan must account for all 32 D-registers"
);
const _: () = assert!(
    GP_VOLATILE_DYNAMIC as usize + GP_PRESERVED_DYNAMIC as usize + GP_INTERNAL_SCRATCH as usize
        == REG_PLAN.gp_dynamic.len(),
    "arm64 GP volatility counts must match gp_dynamic"
);
const _: () = assert!(
    FP_VOLATILE_DYNAMIC as usize + FP_PRESERVED_DYNAMIC as usize == REG_PLAN.fp_dynamic.len(),
    "arm64 FP volatility counts must match fp_dynamic"
);

#[inline]
pub(super) const fn callee_saved_gp_pair_count() -> usize {
    REG_PLAN.callee_saved_gp_pairs.len()
}

#[inline]
pub(super) const fn callee_saved_fp_count() -> usize {
    REG_PLAN.callee_saved_fp.len()
}

#[inline]
pub(super) const fn stack_alignment_bytes() -> u32 {
    REG_PLAN.stack_alignment_bytes
}

#[inline]
pub(super) fn callee_saved_gp_pairs() -> &'static [(Arm64Reg, Arm64Reg)] {
    REG_PLAN.callee_saved_gp_pairs
}

#[inline]
pub(super) fn callee_saved_fp_regs() -> &'static [Arm64FpReg] {
    REG_PLAN.callee_saved_fp
}

// ── C ABI boundary registers ─────────────────────────────────────────────────
// Platform calling-convention facts, used only at the C↔JIT boundary.
//
// These are foreign ABI registers, not extra MachineIR roles. They may alias
// caller-clobbered dynamic or scratch regs, but must not alias the fixed
// MachineIR roles. Boundary lowering is what makes that safe: SSA values are
// dead at the boundary and local state has already been published.

pub(super) const C_ARG0: Arm64Reg = gp(0);
pub(super) const C_ARG1: Arm64Reg = gp(1);
pub(super) const C_ARG2: Arm64Reg = gp(2);
pub(super) const C_RET0: Arm64Reg = gp(0);
pub(super) const W2W_GP_RET0: Arm64Reg = gp(1);

#[inline]
pub(super) const fn stack_reg() -> Arm64Reg {
    gp(31)
}

#[inline]
pub(super) const fn zero_reg() -> Arm64Reg {
    gp(31)
}

#[inline]
pub(super) const fn link_reg() -> Arm64Reg {
    gp(30)
}

/// AArch64 host frame-pointer register (`x29`). Used by the body prelude
/// pairing along with `link_reg()` (`x30`) to back up the link state on the
/// host stack across nested local calls.
#[inline]
pub(super) const fn host_fp_reg() -> Arm64Reg {
    gp(29)
}

/// AArch64 host link register (`x30`). Same physical register as
/// `link_reg()`; the alias clarifies intent at body-prelude / Return sites.
#[inline]
pub(super) const fn host_lr_reg() -> Arm64Reg {
    gp(30)
}

#[inline]
pub(super) const fn fp_zero_reg() -> Arm64FpReg {
    fp(0)
}

// ── Derived config ───────────────────────────────────────────────────────────

const SCALAR_CALL_SCRATCH_SLOTS: u16 = 3;
const GP_VOLATILE_DYNAMIC: u8 = 15;
const GP_PRESERVED_DYNAMIC: u8 = 6;
const GP_INTERNAL_SCRATCH: u8 = 1;
const FP_VOLATILE_DYNAMIC: u8 = 22;
const FP_PRESERVED_DYNAMIC: u8 = 8;
const GP_ARG_LANES: u8 = 4;
const FP_ARG_LANES: u8 = 4;

#[inline]
pub(crate) const fn compile_backend_config() -> BackendConfig {
    BackendConfig::with_volatility(
        REG_PLAN.gp_unit_bytes,
        GP_VOLATILE_DYNAMIC,
        GP_PRESERVED_DYNAMIC,
        GP_INTERNAL_SCRATCH,
        FP_VOLATILE_DYNAMIC,
        FP_PRESERVED_DYNAMIC,
        GP_ARG_LANES,
        FP_ARG_LANES,
        true,
        SCALAR_CALL_SCRATCH_SLOTS,
    )
}

// ── Scratch pool construction ────────────────────────────────────────────────

pub(super) fn new_gp_scratch_pool() -> ScratchPool<Arm64Reg, 2> {
    ScratchPool::new([REG_PLAN.gp_scratch[0], REG_PLAN.gp_scratch[1]])
}

pub(super) fn new_fp_scratch_pool() -> ScratchPool<Arm64FpReg, 2> {
    ScratchPool::new([REG_PLAN.fp_scratch[0], REG_PLAN.fp_scratch[1]])
}

// ── Capacity queries ─────────────────────────────────────────────────────────

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

// ── Mapping: MachineReg → physical register ──────────────────────────────────

#[inline]
pub(super) fn map_fixed_reg(reg: MachineReg) -> Arm64Reg {
    match reg {
        MACHINE_CTX_REG => REG_PLAN.ctx,
        MACHINE_FP_REG => REG_PLAN.fp,
        MACHINE_MEM0_BASE_REG => REG_PLAN.mem0_base,
        MACHINE_MEM0_SIZE_REG => REG_PLAN.mem0_size,
        _ => unreachable!("not a fixed machine reg"),
    }
}

#[inline]
pub(super) fn map_reg(reg: MachineReg) -> Result<Arm64Reg, WasmError> {
    if reg.0 < MACHINE_FIXED_REG_COUNT {
        return Ok(map_fixed_reg(reg));
    }
    let config = compile_backend_config();
    gp_dynamic_index(reg, config)
        .and_then(|index| REG_PLAN.gp_dynamic.get(index).copied())
        .ok_or_else(|| WasmError::invalid("arm64 has no GP mapping for machine reg"))
}

#[inline]
pub(super) fn fp_machine_reg(index: usize) -> Option<Arm64FpReg> {
    REG_PLAN.fp_dynamic.get(index).copied()
}

// ── Preserved-helper save sets ──────────────────────────────────────────────
//
// Derived from REG_PLAN — not duplicated. The preserved-helper wrapper saves
// the caller-clobbered dynamic subset because any of those regs may carry live
// JIT state when control crosses into a C helper.

pub(super) fn gp_dynamic_caller_saved_regs() -> &'static [Arm64Reg] {
    REG_PLAN.gp_dynamic_caller_saved
}

pub(super) fn fp_dynamic_caller_saved_regs() -> &'static [Arm64FpReg] {
    REG_PLAN.fp_dynamic_caller_saved
}

/// Total number of GP registers saved by the preserved-helper.
const PRESERVED_GP_COUNT: usize = REG_PLAN.gp_dynamic_caller_saved.len();

/// Total number of FP registers saved by the preserved-helper.
const PRESERVED_FP_COUNT: usize = REG_PLAN.fp_dynamic_caller_saved.len();

// Scalar-only builds keep FP values in the low 64-bit D view. SIMD builds use
// the same bank for full v128 values, so preserved spills must save whole Q
// regs instead of only the low half.
#[cfg(sf_has_simd)]
const PRESERVED_FP_REG_BYTES: u32 = 16;
#[cfg(not(sf_has_simd))]
const PRESERVED_FP_REG_BYTES: u32 = 8;

const fn preserved_io_size() -> u32 {
    preserved_io::SLOT_COUNT as u32 * 8
}

/// Byte offset of the GP save area within the preserved-helper frame.
pub(super) const PRESERVED_HELPER_GP_OFFSET: u32 = preserved_io_size();

/// Byte offset of the FP save area within the preserved-helper frame.
pub(super) const PRESERVED_HELPER_FP_OFFSET: u32 =
    PRESERVED_HELPER_GP_OFFSET + PRESERVED_GP_COUNT as u32 * 8;

/// Stack frame size for the preserved-helper save area + I/O region,
/// rounded up to 16-byte alignment.
pub(super) const PRESERVED_HELPER_FRAME_SIZE: u32 = {
    let raw = PRESERVED_HELPER_FP_OFFSET + PRESERVED_FP_COUNT as u32 * PRESERVED_FP_REG_BYTES;
    raw.div_ceil(16) * 16
};
