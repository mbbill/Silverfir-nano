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
        jit::machine::machine_ir::{
            gp_dynamic_index, MachineReg, MACHINE_CTX_REG, MACHINE_FIXED_REG_COUNT, MACHINE_FP_REG,
            MACHINE_MEM0_BASE_REG, MACHINE_MEM0_SIZE_REG,
        },
    },
};

use super::callconv;
use super::reg::X86Reg;
use crate::vm::jit::arch::common::scratch_pool::ScratchPool;

// ── Register plan ────────────────────────────────────────────────────────────

struct RegPlan {
    ctx: X86Reg,
    fp: X86Reg,
    mem0_base: X86Reg,
    mem0_size: X86Reg,
    gp_unit_bytes: u8,
    /// Ordered GP dynamic bank. Earlier entries are preferred for allocation.
    gp_dynamic: &'static [X86Reg],
    /// Backend-owned GP registers required by x86_64 instruction forms.
    ///
    /// These are not part of the dynamic bank: ordinary lowering may need the
    /// exact register (`RCX` for variable shifts, `RAX:RDX` for div/rem), so
    /// x86_64 tracks them with a backend-local GP scratch owner.
    gp_backend_owned: &'static [X86Reg],
    /// Ordered FP dynamic bank. Earlier entries are preferred for allocation.
    fp_dynamic: &'static [u32],
    fp_scratch: &'static [u32],
}

// Callee-saved set comes from `callconv`, not `REG_PLAN`, because the ABI
// decides which GP regs must survive a C call. See `callconv::sysv` /
// `callconv::win64`.

const REG_PLAN: RegPlan = RegPlan {
    ctx: X86Reg::RBX,
    fp: X86Reg::RBP,
    mem0_base: X86Reg::R12,
    mem0_size: X86Reg::R13,

    gp_unit_bytes: 8,

    // Prefer caller-saved GP regs first for short-lived SSA traffic, then
    // callee-saved dynamic regs for longer-lived residency.
    gp_dynamic: &[
        X86Reg::RSI,
        X86Reg::RDI,
        X86Reg::R8,
        X86Reg::R9,
        X86Reg::R10,
        X86Reg::R11,
        X86Reg::R14,
        X86Reg::R15,
    ],
    // x86_64 ordinary lowering sometimes requires these exact registers.
    // They are backend-owned and tracked locally, not handed out by regalloc.
    gp_backend_owned: &[X86Reg::RAX, X86Reg::RCX, X86Reg::RDX],

    // XMM0..XMM1 are reserved for the scratch pool; the dynamic bank
    // starts at XMM2 so the allocator never hands out a lane that the
    // inline sequences (Neg / Min / Copysign / etc.) scribble into as a
    // mask temp. Must stay disjoint from `fp_scratch` below.
    fp_dynamic: &[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    fp_scratch: &[0, 1], // XMM0, XMM1
};

// Compile-time check: the FP dynamic bank and scratch pool together
// must account for all 16 XMM registers, with no overlap.
const _: () = assert!(
    REG_PLAN.fp_dynamic.len() + REG_PLAN.fp_scratch.len() == 16,
    "FP register plan must account for all 16 XMM registers"
);

#[inline]
pub(super) fn callee_saved_gp_regs() -> &'static [X86Reg] {
    callconv::CALLEE_SAVED_GP
}

// ── C ABI boundary registers ─────────────────────────────────────────────────
//
// These are foreign ABI registers, not extra MachineIR roles. They may alias
// caller-clobbered dynamic or scratch regs, but must not alias the fixed
// MachineIR roles. Boundary lowering is what makes that safe: SSA values are
// dead at the boundary and local state has already been published.
//
// The actual values live in `callconv::sysv` / `callconv::win64`; re-exported
// here so existing `use super::abi::{C_ARG0, ...}` imports keep working.

#[allow(unused_imports)]
pub(super) use super::callconv::C_ARG3;
pub(super) use super::callconv::{C_ARG0, C_ARG1, C_ARG2, C_RET0};

// ── Derived config ───────────────────────────────────────────────────────────

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

// ── Scratch pool construction ────────────────────────────────────────────────

pub(super) fn gp_backend_owned_regs() -> [X86Reg; 3] {
    [
        REG_PLAN.gp_backend_owned[0],
        REG_PLAN.gp_backend_owned[1],
        REG_PLAN.gp_backend_owned[2],
    ]
}

pub(super) fn new_fp_scratch_pool() -> ScratchPool<u32, 2> {
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
        .ok_or_else(|| WasmError::invalid("x86_64 has no GP mapping for machine reg"))
}

#[inline]
pub(super) fn fp_machine_reg(index: usize) -> Option<u32> {
    REG_PLAN.fp_dynamic.get(index).copied()
}
