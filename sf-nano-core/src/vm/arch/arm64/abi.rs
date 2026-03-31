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
    ctx: Arm64Reg::X19,
    fp: Arm64Reg::X20,
    mem0_base: Arm64Reg::X21,
    mem0_size: Arm64Reg::X22,

    gp_unit_bytes: 8,

    gp_local_cache: &[
        // callee-saved (first 6)
        Arm64Reg::X23,
        Arm64Reg::X24,
        Arm64Reg::X25,
        Arm64Reg::X26,
        Arm64Reg::X27,
        Arm64Reg::X28,
        // caller-saved (remaining)
        Arm64Reg::X9,
        Arm64Reg::X10,
        Arm64Reg::X11,
        Arm64Reg::X12,
        Arm64Reg::X13,
        Arm64Reg::X14,
        Arm64Reg::X15,
    ],
    gp_local_cache_callee_saved: 6,
    gp_transient: &[
        Arm64Reg::X3,
        Arm64Reg::X4,
        Arm64Reg::X5,
        Arm64Reg::X6,
        Arm64Reg::X7,
        Arm64Reg::X8,
        Arm64Reg::X0,
        Arm64Reg::X1,
        Arm64Reg::X2,
    ],
    gp_scratch: &[Arm64Reg::X16, Arm64Reg::X17],

    fp_transient: &[
        Arm64FpReg::new(3),
        Arm64FpReg::new(4),
        Arm64FpReg::new(5),
        Arm64FpReg::new(6),
        Arm64FpReg::new(7),
        Arm64FpReg::new(16),
        Arm64FpReg::new(17),
        Arm64FpReg::new(18),
        Arm64FpReg::new(19),
        Arm64FpReg::new(20),
    ],
    fp_local_cache: &[
        // callee-saved (first 8)
        Arm64FpReg::new(8),
        Arm64FpReg::new(9),
        Arm64FpReg::new(10),
        Arm64FpReg::new(11),
        Arm64FpReg::new(12),
        Arm64FpReg::new(13),
        Arm64FpReg::new(14),
        Arm64FpReg::new(15),
        // caller-saved (remaining)
        Arm64FpReg::new(21),
        Arm64FpReg::new(22),
        Arm64FpReg::new(23),
        Arm64FpReg::new(24),
        Arm64FpReg::new(25),
        Arm64FpReg::new(26),
        Arm64FpReg::new(27),
        Arm64FpReg::new(28),
        Arm64FpReg::new(29),
        Arm64FpReg::new(30),
        Arm64FpReg::new(31),
    ],
    fp_local_cache_callee_saved: 8,
    fp_scratch: &[Arm64FpReg::new(0), Arm64FpReg::new(1), Arm64FpReg::new(2)],

    callee_saved_gp_pairs: &[
        (Arm64Reg::X19, Arm64Reg::X20),
        (Arm64Reg::X21, Arm64Reg::X22),
        (Arm64Reg::X23, Arm64Reg::X24),
        (Arm64Reg::X25, Arm64Reg::X26),
        (Arm64Reg::X27, Arm64Reg::X28),
        (Arm64Reg::X29, Arm64Reg::X30),
    ],
    callee_saved_fp: &[
        Arm64FpReg::new(8),
        Arm64FpReg::new(9),
        Arm64FpReg::new(10),
        Arm64FpReg::new(11),
        Arm64FpReg::new(12),
        Arm64FpReg::new(13),
        Arm64FpReg::new(14),
        Arm64FpReg::new(15),
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

pub(super) const C_ARG0: Arm64Reg = Arm64Reg::X0;
pub(super) const C_ARG1: Arm64Reg = Arm64Reg::X1;
pub(super) const C_ARG2: Arm64Reg = Arm64Reg::X2;
pub(super) const C_ARG3: Arm64Reg = Arm64Reg::X3;
pub(super) const C_RET0: Arm64Reg = Arm64Reg::X0;

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
