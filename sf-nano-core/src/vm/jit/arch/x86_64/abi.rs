//! x86_64 register plan — single source of truth for register allocation.
//!
//! `REG_PLAN` declares every register role in one place. `BackendConfig` is
//! derived from the array lengths, so budgets can never drift out of sync with
//! the physical register tables. The raw plan stays private to this module;
//! backend code must come through the accessors below.

use crate::{
    error::WasmError,
    vm::{
        jit::backend::BackendConfig,
        jit::machine::machine_ir::{
            gp_dynamic_index, MachineReg, MACHINE_CTX_REG, MACHINE_FIXED_REG_COUNT, MACHINE_FP_REG,
            MACHINE_MEM0_BASE_REG, MACHINE_MEM0_SIZE_REG,
        },
    },
};

use super::callconv;
use super::reg::X86Reg;
use crate::vm::jit::arch::common::scratch_pool::ScratchPool;
use crate::vm::jit::runtime::context::ctx_offset;

// ── Register plan ────────────────────────────────────────────────────────────

struct RegPlan {
    ctx: X86Reg,
    fp: X86Reg,
    mem0_base: X86Reg,
    gp_unit_bytes: u8,
    /// Ordered GP dynamic bank. Earlier entries are preferred for allocation.
    gp_dynamic: &'static [X86Reg],
    /// Backend-owned GP registers required by x86_64 instruction forms.
    ///
    /// These are not part of the dynamic bank: ordinary lowering may need the
    /// exact register (`RAX:RDX` for div/rem, `RAX` for helper status), so
    /// x86_64 tracks them with a backend-local GP scratch owner. `RCX`, the
    /// variable shift count register, is a dynamic lane instead; shift
    /// lowering saves it when the count lives elsewhere.
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

    gp_unit_bytes: 8,

    // Positional lane classes: [volatile | preserved | internal scratch].
    // Caller-saved GP regs first for short-lived SSA traffic; R14, R15 and
    // R13 are callee-saved in both SysV and Win64 and form the preserved
    // lanes — values in them survive C helper calls, and SF->SF bodies
    // lazy-save them (`body_frame_plan`).
    gp_dynamic: &[
        X86Reg::RSI,
        X86Reg::RDI,
        X86Reg::R8,
        X86Reg::R9,
        X86Reg::R10,
        X86Reg::RCX,
        X86Reg::R14,
        X86Reg::R15,
        X86Reg::R13,
        X86Reg::R11,
    ],
    // x86_64 ordinary lowering sometimes requires these exact registers.
    // They are backend-owned and tracked locally, not handed out by regalloc.
    gp_backend_owned: &[X86Reg::RAX, X86Reg::RDX],

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

pub(super) use super::callconv::{C_ARG0, C_ARG1, C_ARG2, C_RET0};

// ── Wasm-to-wasm return lanes ────────────────────────────────────────────────
//
// SF->SF scalar returns travel in registers, not through the frame. Status
// stays in `C_RET0` (RAX); the scalar payload uses lanes that are dead at
// every SF->SF boundary: RDX is backend-owned (never allocated), and XMM0 is
// the first FP scratch (never in the dynamic bank).

pub(super) const W2W_GP_RET0: X86Reg = X86Reg::RDX;
pub(super) const W2W_FP_RET0: u32 = REG_PLAN.fp_scratch[0];

// ── Derived config ───────────────────────────────────────────────────────────

const SCALAR_CALL_SCRATCH_SLOTS: u16 = 3;

// Lane-class widths over `REG_PLAN.gp_dynamic`, in positional order.
const GP_VOLATILE_DYNAMIC: u8 = 6;
const GP_PRESERVED_DYNAMIC: u8 = 3;
const GP_INTERNAL_SCRATCH: u8 = 1;
const GP_ARG_LANES: u8 = 4;
const FP_ARG_LANES: u8 = 4;

const _: () = assert!(
    GP_VOLATILE_DYNAMIC as usize + GP_PRESERVED_DYNAMIC as usize + GP_INTERNAL_SCRATCH as usize
        == REG_PLAN.gp_dynamic.len(),
    "x86_64 GP volatility counts must match gp_dynamic"
);

#[inline]
pub(crate) const fn compile_backend_config() -> BackendConfig {
    BackendConfig::with_volatility(
        REG_PLAN.gp_unit_bytes,
        GP_VOLATILE_DYNAMIC,
        GP_PRESERVED_DYNAMIC,
        GP_INTERNAL_SCRATCH,
        REG_PLAN.fp_dynamic.len() as u8,
        0,
        GP_ARG_LANES,
        FP_ARG_LANES,
        true,
        SCALAR_CALL_SCRATCH_SLOTS,
    )
    // Lazy per-body preserved save: pushes in the body prelude, pops at
    // each return path, and the alignment-shim toggle. Priced above
    // arm64's stp/ldp pair so the solver declines nomination in tiny
    // call-heavy bodies, where the fixed cost outweighs residency
    // (fibonacci-rec measured -4.4% at 3).
    .with_preserved_lane_save_overhead(5)
    // r32-form instructions clear bits 63:32, so the peephole may drop
    // ZeroExtend32 index obligations whose index was defined by one.
    .with_gp32_zero_extending_defs()
}

// ── Scratch pool construction ────────────────────────────────────────────────

pub(super) fn gp_backend_owned_regs() -> [X86Reg; 2] {
    [REG_PLAN.gp_backend_owned[0], REG_PLAN.gp_backend_owned[1]]
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

/// Fixed roles that occupy a register. `MACHINE_MEM0_SIZE_REG` is not one:
/// see [`mem0_size_addr`].
#[inline]
pub(super) fn map_fixed_reg(reg: MachineReg) -> X86Reg {
    match reg {
        MACHINE_CTX_REG => REG_PLAN.ctx,
        MACHINE_FP_REG => REG_PLAN.fp,
        MACHINE_MEM0_BASE_REG => REG_PLAN.mem0_base,
        _ => unreachable!("not a register-backed fixed machine reg"),
    }
}

/// Where x86_64 keeps `MACHINE_MEM0_SIZE_REG`: the runtime context's memory
/// 0 length field, not a register. With guard pages, 32-bit accesses never
/// compare against it, so a pinned register would sit idle in almost every
/// body; R13 is a preserved dynamic lane instead.
///
/// Reading the field equals reading a register reloaded from it, because the
/// register was only ever written by loads of this field (at body entry and
/// after memory growth or a runtime call). Its readers take it as a memory
/// operand; those reloads lower to nothing.
#[inline]
pub(super) fn mem0_size_addr() -> (X86Reg, i32) {
    (REG_PLAN.ctx, ctx_offset::MEM0_SIZE as i32)
}

#[inline]
pub(super) fn map_reg(reg: MachineReg) -> Result<X86Reg, WasmError> {
    if reg == MACHINE_MEM0_SIZE_REG {
        return Err(WasmError::internal(
            "x86_64 keeps memory 0's size in the runtime context, not a register",
        ));
    }
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
