//! x86_64 register plan — single source of truth for register allocation.
//!
//! `REG_PLAN` declares every register role in one place.  `BackendConfig` is
//! derived from the array lengths, so budgets can never drift out of sync with
//! the physical register tables.

use crate::{
    error::WasmError,
    vm::{
        backend::BackendConfig,
        machine::machine_ir::{
            MachineReg, MACHINE_CTX_REG, MACHINE_FIXED_REG_COUNT, MACHINE_FP_REG,
            MACHINE_MEM0_BASE_REG, MACHINE_MEM0_SIZE_REG,
            classify_gp_reg, classify_fp_reg,
        },
    },
};

use super::reg::X86Reg;
use crate::vm::arch::common::scratch_pool::ScratchPool;

// ── Register plan ────────────────────────────────────────────────────────────

pub(super) struct RegPlan {
    pub ctx: X86Reg,
    pub fp: X86Reg,
    pub mem0_base: X86Reg,
    pub mem0_size: X86Reg,
    pub gp_unit_bytes: u8,
    pub gp_local_cache: &'static [X86Reg],
    pub gp_transient: &'static [X86Reg],
    pub gp_scratch: &'static [X86Reg],
    pub fp_transient: &'static [u32],
    pub fp_local_cache: &'static [u32],
    pub fp_scratch: &'static [u32],
    pub callee_saved_gp: &'static [X86Reg],
    pub stack_alignment_bytes: u32,
}

pub(super) const REG_PLAN: RegPlan = RegPlan {
    ctx: X86Reg::RBX,
    fp: X86Reg::RBP,
    mem0_base: X86Reg::R12,
    mem0_size: X86Reg::R13,

    gp_unit_bytes: 8,

    // GP local cache: callee-saved — free at helper calls
    gp_local_cache: &[X86Reg::R14, X86Reg::R15],
    // GP transient: caller-saved — dead at call boundaries
    gp_transient: &[
        X86Reg::RCX, X86Reg::RDX, X86Reg::RSI, X86Reg::RDI,
        X86Reg::R8, X86Reg::R9, X86Reg::R10,
    ],
    gp_scratch: &[X86Reg::RAX, X86Reg::R11],

    // FP transient XMM registers: XMM0-XMM5
    fp_transient: &[0, 1, 2, 3, 4, 5],
    // FP cached local XMM registers: XMM6-XMM15
    // All caller-saved on System V — must be spilled around helper calls.
    fp_local_cache: &[6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    fp_scratch: &[0, 1, 2], // XMM0, XMM1, XMM2

    // Callee-saved GP registers (fixed + cached locals).
    // No callee-saved FP regs on System V AMD64.
    #[cfg(not(target_os = "windows"))]
    callee_saved_gp: &[X86Reg::RBX, X86Reg::RBP, X86Reg::R12, X86Reg::R13, X86Reg::R14, X86Reg::R15],
    #[cfg(target_os = "windows")]
    callee_saved_gp: &[X86Reg::RBX, X86Reg::RBP, X86Reg::R12, X86Reg::R13, X86Reg::R14, X86Reg::R15, X86Reg::RDI, X86Reg::RSI],

    stack_alignment_bytes: 16,
};

// Compile-time check: transients + cache = 16 XMM regs.
const _: () = assert!(
    REG_PLAN.fp_transient.len() + REG_PLAN.fp_local_cache.len() == 16,
    "FP register plan must account for all 16 XMM registers"
);

// ── C ABI boundary registers ─────────────────────────────────────────────────

#[cfg(not(target_os = "windows"))]
pub(super) const C_ARG0: X86Reg = X86Reg::RDI;
#[cfg(not(target_os = "windows"))]
pub(super) const C_ARG1: X86Reg = X86Reg::RSI;
#[cfg(not(target_os = "windows"))]
pub(super) const C_ARG2: X86Reg = X86Reg::RDX;
#[cfg(target_os = "windows")]
pub(super) const C_ARG0: X86Reg = X86Reg::RCX;
#[cfg(target_os = "windows")]
pub(super) const C_ARG1: X86Reg = X86Reg::RDX;
#[cfg(target_os = "windows")]
pub(super) const C_ARG2: X86Reg = X86Reg::R8;
#[cfg(target_os = "windows")]
pub(super) const C_ARG3: X86Reg = X86Reg::R9;
pub(super) const C_RET0: X86Reg = X86Reg::RAX;

// ── Derived config ───────────────────────────────────────────────────────────

#[inline]
pub(crate) const fn compile_backend_config() -> BackendConfig {
    BackendConfig::new_with_gp_unit_bytes(
        REG_PLAN.gp_local_cache.len() as u8,
        REG_PLAN.gp_transient.len() as u8,
        REG_PLAN.fp_local_cache.len() as u8,
        REG_PLAN.fp_transient.len() as u8,
        REG_PLAN.gp_unit_bytes,
    )
}

// ── Scratch pool construction ────────────────────────────────────────────────

pub(super) fn new_gp_scratch_pool() -> ScratchPool<X86Reg, 2> {
    ScratchPool::new([REG_PLAN.gp_scratch[0], REG_PLAN.gp_scratch[1]])
}

pub(super) fn new_fp_scratch_pool() -> ScratchPool<u32, 3> {
    ScratchPool::new([REG_PLAN.fp_scratch[0], REG_PLAN.fp_scratch[1], REG_PLAN.fp_scratch[2]])
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
pub(super) fn map_fixed_reg(reg: MachineReg) -> X86Reg {
    match reg {
        MACHINE_CTX_REG => REG_PLAN.ctx,
        MACHINE_FP_REG => REG_PLAN.fp,
        MACHINE_MEM0_BASE_REG => REG_PLAN.mem0_base,
        MACHINE_MEM0_SIZE_REG => REG_PLAN.mem0_size,
        _ => unreachable!("not a fixed machine reg"),
    }
}

#[inline]
pub(super) fn map_reg(reg: MachineReg) -> Result<X86Reg, WasmError> {
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
            "x86_64 has no GP mapping for machine reg {}",
            reg.0
        ))
    })
}

#[inline]
pub(super) fn fp_machine_reg(index: usize) -> Option<u32> {
    let config = compile_backend_config();
    let (i, is_cache) = classify_fp_reg(index, config);
    if is_cache {
        REG_PLAN.fp_local_cache.get(i).copied()
    } else {
        REG_PLAN.fp_transient.get(i).copied()
    }
}
