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
            classify_fp_reg, classify_gp_reg, MachineReg, MACHINE_CTX_REG, MACHINE_FIXED_REG_COUNT,
            MACHINE_FP_REG, MACHINE_MEM0_BASE_REG, MACHINE_MEM0_SIZE_REG,
        },
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

struct RegPlan {
    // Fixed MachineIR roles
    ctx: Arm64Reg,
    fp: Arm64Reg,
    mem0_base: Arm64Reg,
    mem0_size: Arm64Reg,
    // GP budget unit size
    gp_unit_bytes: u8,
    // GP dynamic partition (ordered: cache first, then transient)
    //   gp_local_cache: first `gp_local_cache_callee_saved` entries are
    //   ABI callee-saved; the remainder are caller-saved.
    gp_local_cache: &'static [Arm64Reg],
    gp_local_cache_callee_saved: usize,
    gp_transient: &'static [Arm64Reg],
    gp_scratch: &'static [Arm64Reg],
    // FP dynamic partition (ordered: transient first, then cache)
    //   fp_local_cache: first `fp_local_cache_callee_saved` entries are
    //   ABI callee-saved; the remainder are caller-saved.
    fp_transient: &'static [Arm64FpReg],
    fp_local_cache: &'static [Arm64FpReg],
    fp_local_cache_callee_saved: usize,
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

    gp_local_cache: &[
        // callee-saved (first 6)
        gp(23),
        gp(24),
        gp(25),
        gp(26),
        gp(27),
        gp(28),
        // caller-saved (remaining)
        gp(9),
        gp(10),
        gp(11),
        gp(12),
        gp(13),
        gp(14),
        gp(15),
    ],
    gp_local_cache_callee_saved: 6,
    gp_transient: &[
        gp(3),
        gp(4),
        gp(5),
        gp(6),
        gp(7),
        gp(8),
        gp(0),
        gp(1),
        gp(2),
    ],
    gp_scratch: &[gp(16), gp(17)],

    fp_transient: &[
        fp(3),
        fp(4),
        fp(5),
        fp(6),
        fp(7),
        fp(16),
        fp(17),
        fp(18),
        fp(19),
        fp(20),
    ],
    fp_local_cache: &[
        // callee-saved (first 8)
        fp(8),
        fp(9),
        fp(10),
        fp(11),
        fp(12),
        fp(13),
        fp(14),
        fp(15),
        // caller-saved (remaining)
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
    ],
    fp_local_cache_callee_saved: 8,
    fp_scratch: &[fp(0), fp(1), fp(2)],

    callee_saved_gp_pairs: &[
        (gp(19), gp(20)),
        (gp(21), gp(22)),
        (gp(23), gp(24)),
        (gp(25), gp(26)),
        (gp(27), gp(28)),
        (gp(29), gp(30)),
    ],
    callee_saved_fp: &[
        fp(8),
        fp(9),
        fp(10),
        fp(11),
        fp(12),
        fp(13),
        fp(14),
        fp(15),
    ],

    stack_alignment_bytes: 16,
};

// Compile-time checks
const _: () = assert!(
    REG_PLAN.fp_transient.len() + REG_PLAN.fp_local_cache.len() + REG_PLAN.fp_scratch.len() == 32,
    "FP register plan must account for all 32 D-registers"
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
// caller-saved transient or scratch regs, but must not alias the fixed
// MachineIR roles. Boundary lowering is what makes that safe: transients are
// dead at the boundary and cached locals have already been published.

pub(super) const C_ARG0: Arm64Reg = gp(0);
pub(super) const C_ARG1: Arm64Reg = gp(1);
pub(super) const C_ARG2: Arm64Reg = gp(2);
pub(super) const C_RET0: Arm64Reg = gp(0);

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

#[inline]
pub(super) const fn fp_zero_reg() -> Arm64FpReg {
    fp(0)
}

// ── Derived config ───────────────────────────────────────────────────────────

#[inline]
pub(crate) const fn compile_backend_config() -> BackendConfig {
    BackendConfig::new(
        REG_PLAN.gp_local_cache.len() as u8,
        REG_PLAN.gp_transient.len() as u8,
        REG_PLAN.fp_local_cache.len() as u8,
        REG_PLAN.fp_transient.len() as u8,
        REG_PLAN.gp_unit_bytes,
        3,
    )
}

// ── Scratch pool construction ────────────────────────────────────────────────

pub(super) fn new_gp_scratch_pool() -> ScratchPool<Arm64Reg, 2> {
    ScratchPool::new([REG_PLAN.gp_scratch[0], REG_PLAN.gp_scratch[1]])
}

pub(super) fn new_fp_scratch_pool() -> ScratchPool<Arm64FpReg, 3> {
    ScratchPool::new([
        REG_PLAN.fp_scratch[0],
        REG_PLAN.fp_scratch[1],
        REG_PLAN.fp_scratch[2],
    ])
}

// ── Capacity queries ─────────────────────────────────────────────────────────

pub(super) const FP_MACHINE_REG_COUNT: usize =
    REG_PLAN.fp_transient.len() + REG_PLAN.fp_local_cache.len();

#[inline]
pub(super) const fn max_gp_mapped_regs() -> usize {
    MACHINE_FIXED_REG_COUNT as usize + REG_PLAN.gp_local_cache.len() + REG_PLAN.gp_transient.len()
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
    match classify_gp_reg(reg, config) {
        Some((index, true)) => REG_PLAN.gp_local_cache.get(index).copied(),
        Some((index, false)) => REG_PLAN.gp_transient.get(index).copied(),
        None => return Ok(map_fixed_reg(reg)),
    }
    .ok_or_else(|| {
        WasmError::invalid(alloc::format!(
            "arm64 has no GP mapping for machine reg {}",
            reg.0
        ))
    })
}

#[inline]
pub(super) fn fp_machine_reg(index: usize) -> Option<Arm64FpReg> {
    let config = compile_backend_config();
    let (i, is_cache) = classify_fp_reg(index, config);
    if is_cache {
        REG_PLAN.fp_local_cache.get(i).copied()
    } else {
        REG_PLAN.fp_transient.get(i).copied()
    }
}

// ── Preserved-helper save sets ──────────────────────────────────────────────
//
// Derived from REG_PLAN — not duplicated.  The preserved-helper wrapper
// saves all caller-clobbered registers that may hold live JIT state:
//   GP: all transients + caller-saved local-cache
//   FP: all transients + caller-saved local-cache

/// All GP transient registers (all caller-saved).
pub(super) fn gp_transient_regs() -> &'static [Arm64Reg] { REG_PLAN.gp_transient }

/// Caller-saved portion of the GP local cache (after the callee-saved prefix).
pub(super) fn gp_caller_saved_cache() -> &'static [Arm64Reg] {
    &REG_PLAN.gp_local_cache[REG_PLAN.gp_local_cache_callee_saved..]
}

/// All FP transient registers (all caller-saved).
pub(super) fn fp_transient_regs() -> &'static [Arm64FpReg] { REG_PLAN.fp_transient }

/// Caller-saved portion of the FP local cache (after the callee-saved prefix).
pub(super) fn fp_caller_saved_cache() -> &'static [Arm64FpReg] {
    &REG_PLAN.fp_local_cache[REG_PLAN.fp_local_cache_callee_saved..]
}

/// Total number of GP registers saved by the preserved-helper.
const PRESERVED_GP_COUNT: usize =
    REG_PLAN.gp_transient.len()
    + REG_PLAN.gp_local_cache.len() - REG_PLAN.gp_local_cache_callee_saved;

/// Total number of FP registers saved by the preserved-helper.
const PRESERVED_FP_COUNT: usize =
    REG_PLAN.fp_transient.len()
    + REG_PLAN.fp_local_cache.len() - REG_PLAN.fp_local_cache_callee_saved;

const fn preserved_io_size() -> u32 {
    crate::vm::runtime::helpers::preserved_io::SLOT_COUNT as u32 * 8
}

/// Byte offset of the GP save area within the preserved-helper frame.
pub(super) const PRESERVED_HELPER_GP_OFFSET: u32 = preserved_io_size();

/// Byte offset of the FP save area within the preserved-helper frame.
pub(super) const PRESERVED_HELPER_FP_OFFSET: u32 =
    PRESERVED_HELPER_GP_OFFSET + PRESERVED_GP_COUNT as u32 * 8;

/// Stack frame size for the preserved-helper save area + I/O region,
/// rounded up to 16-byte alignment.
pub(super) const PRESERVED_HELPER_FRAME_SIZE: u32 = {
    let raw = PRESERVED_HELPER_FP_OFFSET + PRESERVED_FP_COUNT as u32 * 8;
    raw.div_ceil(16) * 16
};
