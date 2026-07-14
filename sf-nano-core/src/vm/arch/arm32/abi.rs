//! 32-bit ARM physical register mapping and EABI-derived layout.
//!
//! # GP register plan (R0-R15)
//!
//! ```text
//! Reg    EABI             Role                     Count
//! ─────────────────────────────────────────────────────
//! R0-R2  caller-saved     GP dynamic                  3
//! R3     caller-saved     GP dynamic                  1
//! R4     callee-saved     fixed: MEM0_SIZE            1
//! R5-R7  callee-saved     GP dynamic                  3
//! R8     callee-saved     fixed: CTX                  1
//! R9     platform         GP dynamic                  1
//! R10    callee-saved     fixed: FP                   1
//! R11    callee-saved     fixed: MEM0_BASE            1
//! R12    caller-saved     scratch: SCRATCH0 (IP)      1
//! R13    —                SP (reserved)               1
//! R14    caller-saved     scratch: SCRATCH1 (LR)      1
//! R15    —                PC (reserved)               1
//! ─────────────────────────────────────────────────────
//!                         GP dynamic                  8
//!                         GP scratch                  2
//! ```
//!
//! # FP register plan (D0-D15, VFPv3-D16) — requires `sf_fp_dp`
//!
//! ```text
//! Reg    EABI             Role                    Count
//! ─────────────────────────────────────────────────────
//! D0-D2  caller-saved     FP scratch                 3
//! D3-D15 mixed            FP dynamic                13
//! ─────────────────────────────────────────────────────
//!                         FP dynamic                13
//! ```

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

use super::{enc, reg::Arm32Reg};
use crate::vm::arch::common::{scratch_pool::ScratchPool, text_emitter::TextEmitter};

// ── Register plan ────────────────────────────────────────────────────────────

struct RegPlan {
    // Fixed MachineIR roles
    ctx: Arm32Reg,
    fp: Arm32Reg,
    mem0_base: Arm32Reg,
    mem0_size: Arm32Reg,
    // GP budget unit size
    gp_unit_bytes: u8,
    // Preferred GP dynamic order. Allocation-order preference only.
    gp_dynamic: &'static [Arm32Reg],
    gp_scratch: &'static [Arm32Reg],
    // Preferred FP dynamic order (D-register indices).
    fp_dynamic: &'static [u32],
    fp_scratch: &'static [u32],
    // Callee-saved sets
    callee_saved_gp: &'static [Arm32Reg],
    // Stack
    stack_alignment_bytes: u32,
}

const REG_PLAN: RegPlan = RegPlan {
    ctx: Arm32Reg::R8,
    fp: Arm32Reg::R10,
    mem0_base: Arm32Reg::R11,
    mem0_size: Arm32Reg::R4,

    gp_unit_bytes: 4,

    // Lane order groups by JIT-ABI class: volatile lanes first (caller-saved
    // physicals), then the preserved-dynamic lanes (EABI callee-saved R9, R5
    // — survive body-to-body calls via the per-body lazy save), then the
    // internal-scratch tail (R6, R7).
    gp_dynamic: &[
        Arm32Reg::R3,
        Arm32Reg::R0,
        Arm32Reg::R1,
        Arm32Reg::R2,
        Arm32Reg::R9,
        Arm32Reg::R5,
        Arm32Reg::R6,
        Arm32Reg::R7,
    ],
    gp_scratch: &[Arm32Reg::R12, Arm32Reg::R14],

    #[cfg(sf_fp_dp)]
    fp_dynamic: &[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    #[cfg(not(sf_fp_dp))]
    fp_dynamic: &[],

    fp_scratch: &[0, 1, 2], // D0, D1, D2

    callee_saved_gp: &[
        Arm32Reg::R4,
        Arm32Reg::R5,
        Arm32Reg::R6,
        Arm32Reg::R7,
        Arm32Reg::R8,
        Arm32Reg::R9,
        Arm32Reg::R10,
        Arm32Reg::R11,
        Arm32Reg::R14, // LR
    ],

    stack_alignment_bytes: 8,
};

// Compile-time check: dynamic + scratch must cover all 16 D-regs.
#[cfg(sf_fp_dp)]
const _: () = assert!(
    REG_PLAN.fp_dynamic.len() + REG_PLAN.fp_scratch.len() == 16,
    "FP register plan must account for all 16 D-registers (VFPv3-D16)"
);

// ── C ABI boundary registers ─────────────────────────────────────────────────
// Platform calling-convention facts, used only at the C↔JIT boundary.
//
// These are foreign ABI registers, not extra MachineIR roles. They may alias
// caller-clobbered dynamic or scratch regs, but must not alias the fixed
// MachineIR roles. Boundary lowering is what makes that safe: SSA values are
// dead at the boundary and local state has already been published.

pub(super) const C_ARG0: Arm32Reg = Arm32Reg::R0;
pub(super) const C_ARG1: Arm32Reg = Arm32Reg::R1;
pub(super) const C_ARG2: Arm32Reg = Arm32Reg::R2;
pub(super) const C_RET0: Arm32Reg = Arm32Reg::R0;

/// FP C-ABI boundary register. AAPCS passes/returns f64 in D0, f32 in S0
/// (= D0 low half). Used only at foreign helper call boundaries.
pub(super) const C_FP_RET0: u32 = 0; // D0

// ── Derived constants from REG_PLAN ─────────────────────────────────────────

pub(super) const GP_UNIT_BYTES: u8 = REG_PLAN.gp_unit_bytes;

/// Total FP machine-register capacity.
pub(super) const FP_MACHINE_REG_COUNT: usize = REG_PLAN.fp_dynamic.len();

// ── Scratch pool construction ────────────────────────────────────────────────

pub(super) fn new_gp_scratch_pool() -> ScratchPool<Arm32Reg, 2> {
    ScratchPool::new([REG_PLAN.gp_scratch[0], REG_PLAN.gp_scratch[1]])
}

pub(super) fn new_fp_scratch_pool() -> ScratchPool<u32, 3> {
    ScratchPool::new([
        REG_PLAN.fp_scratch[0],
        REG_PLAN.fp_scratch[1],
        REG_PLAN.fp_scratch[2],
    ])
}

// ── Derived config ───────────────────────────────────────────────────────────

/// Dynamic-bank volatility split: 4 volatile + 2 preserved + 2 internal
/// scratch = the 8 dynamic lanes. The preserved lanes (R9, R5) are EABI
/// callee-saved; bodies that clobber them save/restore lazily in the body
/// prelude (`preserved_lane_save_overhead` = 1 store + ~2 restores).
const GP_VOLATILE_DYNAMIC: u8 = 4;
const GP_PRESERVED_DYNAMIC: u8 = 2;
const GP_INTERNAL_SCRATCH: u8 = 2;

#[inline]
pub(crate) const fn compile_backend_config() -> BackendConfig {
    BackendConfig::with_volatility(
        GP_UNIT_BYTES,
        GP_VOLATILE_DYNAMIC,
        GP_PRESERVED_DYNAMIC,
        GP_INTERNAL_SCRATCH,
        REG_PLAN.fp_dynamic.len() as u8,
        0,
        4,
        4,
        false,
        8,
    )
    .with_preserved_lane_save_overhead(3)
}

// ── Callee-saved sets (ARMv7-specific encoding) ─────────────────────────────

/// EABI callee-saved FP regs (D8-D15). Count is 0 when FP is off.
const CALLEE_SAVED_FP_FIRST: u32 = 8;
#[cfg(sf_fp_dp)]
const CALLEE_SAVED_FP_COUNT: u32 = 8;
#[cfg(not(sf_fp_dp))]
const CALLEE_SAVED_FP_COUNT: u32 = 0;

/// Pad the GP save set to maintain `stack_alignment_bytes` alignment across
/// the PUSH. Each GP register is 4 bytes, so an odd count needs one extra
/// 4-byte slot when the required alignment is 8.
const SHARED_PROLOGUE_ALIGN_PAD_BYTES: u32 = {
    let word_bytes = 4u32;
    let pushed = REG_PLAN.callee_saved_gp.len() as u32 * word_bytes;
    let rem = pushed % REG_PLAN.stack_alignment_bytes;
    if rem != 0 {
        REG_PLAN.stack_alignment_bytes - rem
    } else {
        0
    }
};

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
pub(super) fn fp_machine_reg(index: usize) -> Option<u32> {
    REG_PLAN.fp_dynamic.get(index).copied()
}

#[inline]
pub(super) fn map_fixed_reg(reg: MachineReg) -> Arm32Reg {
    match reg {
        MACHINE_CTX_REG => REG_PLAN.ctx,
        MACHINE_FP_REG => REG_PLAN.fp,
        MACHINE_MEM0_BASE_REG => REG_PLAN.mem0_base,
        MACHINE_MEM0_SIZE_REG => REG_PLAN.mem0_size,
        _ => unreachable!("not a fixed machine reg"),
    }
}

#[inline]
pub(super) fn map_reg(reg: MachineReg) -> Result<Arm32Reg, WasmError> {
    if reg.0 < MACHINE_FIXED_REG_COUNT {
        return Ok(map_fixed_reg(reg));
    }
    let config = compile_backend_config();
    gp_dynamic_index(reg, config)
        .and_then(|index| REG_PLAN.gp_dynamic.get(index).copied())
        .ok_or_else(|| {
            WasmError::invalid("arm32 MachineIR backend has no physical mapping for machine reg")
        })
}

/// Build the register mask for PUSH/POP from the callee-saved list.
fn callee_saved_gp_mask() -> u16 {
    let mut mask = 0u16;
    let mut i = 0;
    while i < REG_PLAN.callee_saved_gp.len() {
        mask |= 1 << REG_PLAN.callee_saved_gp[i].idx();
        i += 1;
    }
    mask
}

pub(super) fn emit_shared_prologue(text: &mut TextEmitter) {
    // Preserve 8-byte stack alignment across all helper and runtime calls.
    // The GP save set has an odd number of words, so reserve one extra word
    // before the push/vpush sequence.
    if SHARED_PROLOGUE_ALIGN_PAD_BYTES != 0 {
        text.emit_u32(enc::sub_imm(
            Arm32Reg::SP,
            Arm32Reg::SP,
            SHARED_PROLOGUE_ALIGN_PAD_BYTES,
            0,
        ));
    }
    // PUSH {R4-R11, LR}
    text.emit_u32(enc::push(callee_saved_gp_mask()));
    if CALLEE_SAVED_FP_COUNT > 0 {
        text.emit_u32(enc::vpush_d(CALLEE_SAVED_FP_FIRST, CALLEE_SAVED_FP_COUNT));
    }
}

pub(super) fn emit_shared_epilogue(text: &mut TextEmitter) {
    if CALLEE_SAVED_FP_COUNT > 0 {
        text.emit_u32(enc::vpop_d(CALLEE_SAVED_FP_FIRST, CALLEE_SAVED_FP_COUNT));
    }
    // Restore the GP save set, then drop the alignment pad and return via LR.
    let pop_mask = callee_saved_gp_mask();
    text.emit_u32(enc::pop(pop_mask));
    if SHARED_PROLOGUE_ALIGN_PAD_BYTES != 0 {
        text.emit_u32(enc::add_imm(
            Arm32Reg::SP,
            Arm32Reg::SP,
            SHARED_PROLOGUE_ALIGN_PAD_BYTES,
            0,
        ));
    }
    text.emit_u32(enc::bx(Arm32Reg::R14));
}
