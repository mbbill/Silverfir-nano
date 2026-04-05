//! x86_64 register plan — single source of truth for register allocation.
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
    },
};

use super::reg::X86Reg;
use crate::vm::arch::common::scratch_pool::ScratchPool;

// ── Register plan ────────────────────────────────────────────────────────────

struct RegPlan {
    ctx: X86Reg,
    fp: X86Reg,
    mem0_base: X86Reg,
    mem0_size: X86Reg,
    gp_unit_bytes: u8,
    /// Ordered GP dynamic bank. Earlier entries are preferred for allocation.
    gp_dynamic: &'static [X86Reg],
    gp_scratch: &'static [X86Reg],
    /// Ordered FP dynamic bank. Earlier entries are preferred for allocation.
    fp_dynamic: &'static [u32],
    fp_scratch: &'static [u32],
    callee_saved_gp: &'static [X86Reg],
    stack_alignment_bytes: u32,
}

const REG_PLAN: RegPlan = RegPlan {
    ctx: X86Reg::RBX,
    fp: X86Reg::RBP,
    mem0_base: X86Reg::R12,
    mem0_size: X86Reg::R13,

    gp_unit_bytes: 8,

    // Prefer caller-saved GP regs first for short-lived SSA traffic, then
    // callee-saved dynamic regs for longer-lived residency.
    gp_dynamic: &[
        X86Reg::RCX,
        X86Reg::RDX,
        X86Reg::RSI,
        X86Reg::RDI,
        X86Reg::R8,
        X86Reg::R9,
        X86Reg::R10,
        X86Reg::R14,
        X86Reg::R15,
    ],
    gp_scratch: &[X86Reg::RAX, X86Reg::R11],

    // Prefer the low caller-saved XMM lanes first, then the remaining dynamic
    // bank entries. This is an allocation-order preference only.
    fp_dynamic: &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    fp_scratch: &[0, 1, 2], // XMM0, XMM1, XMM2

    // Callee-saved GP registers (fixed roles plus the callee-saved tail of the
    // dynamic bank). No callee-saved FP regs on System V AMD64.
    #[cfg(not(target_os = "windows"))]
    callee_saved_gp: &[
        X86Reg::RBX,
        X86Reg::RBP,
        X86Reg::R12,
        X86Reg::R13,
        X86Reg::R14,
        X86Reg::R15,
    ],
    #[cfg(target_os = "windows")]
    callee_saved_gp: &[
        X86Reg::RBX,
        X86Reg::RBP,
        X86Reg::R12,
        X86Reg::R13,
        X86Reg::R14,
        X86Reg::R15,
        X86Reg::RDI,
        X86Reg::RSI,
    ],

    stack_alignment_bytes: 16,
};

// Compile-time check: the unified FP dynamic bank covers all 16 XMM regs.
const _: () = assert!(
    REG_PLAN.fp_dynamic.len() == 16,
    "FP register plan must account for all 16 XMM registers"
);

#[inline]
pub(super) const fn callee_saved_gp_count() -> usize {
    REG_PLAN.callee_saved_gp.len()
}

#[inline]
pub(super) const fn stack_alignment_bytes() -> u32 {
    REG_PLAN.stack_alignment_bytes
}

#[inline]
pub(super) fn callee_saved_gp_regs() -> &'static [X86Reg] {
    REG_PLAN.callee_saved_gp
}

// ── C ABI boundary registers ─────────────────────────────────────────────────
//
// These are foreign ABI registers, not extra MachineIR roles. They may alias
// caller-clobbered dynamic or scratch regs, but must not alias the fixed
// MachineIR roles. Boundary lowering is what makes that safe: SSA values are
// dead at the boundary and local state has already been published.

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
    BackendConfig::new(
        REG_PLAN.gp_dynamic.len() as u8,
        REG_PLAN.fp_dynamic.len() as u8,
        REG_PLAN.gp_unit_bytes,
        3,
    )
}

// ── Scratch pool construction ────────────────────────────────────────────────

pub(super) fn new_gp_scratch_pool() -> ScratchPool<X86Reg, 2> {
    ScratchPool::new([REG_PLAN.gp_scratch[0], REG_PLAN.gp_scratch[1]])
}

pub(super) fn new_fp_scratch_pool() -> ScratchPool<u32, 3> {
    ScratchPool::new([
        REG_PLAN.fp_scratch[0],
        REG_PLAN.fp_scratch[1],
        REG_PLAN.fp_scratch[2],
    ])
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
    gp_dynamic_index(reg, config)
        .and_then(|index| REG_PLAN.gp_dynamic.get(index).copied())
        .ok_or_else(|| {
            WasmError::invalid(alloc::format!(
                "x86_64 has no GP mapping for machine reg {}",
                reg.0
            ))
        })
}

#[inline]
pub(super) fn fp_machine_reg(index: usize) -> Option<u32> {
    REG_PLAN.fp_dynamic.get(index).copied()
}
