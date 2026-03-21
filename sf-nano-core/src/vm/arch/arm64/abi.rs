//! ARM64 physical register mapping and ABI-derived layout.
//!
//! This module is the source of truth for facts imposed by the target ABI or
//! by our chosen physical register mapping:
//! - which hardware registers back fixed machine roles
//! - which GP/FP registers are available to the machine-reg file
//! - which FP registers are scratch vs machine-resident
//! - which physical registers must be preserved by the shared prologue/epilogue
//! - stack-frame layout derived from those saved-register sets
//!
//! It intentionally does *not* choose how much of that capacity the compiler
//! spends by default. That policy lives in `config.rs` as a backend budget
//! preset and must fit within the capacities described here.
//!
//! # GP register plan (X0-X30)
//!
//! ```text
//! Reg       AAPCS64          Role                    Count
//! ──────────────────────────────────────────────────────────
//! X0-X2     caller-saved     reserved (helper args)      3
//! X3-X8     caller-saved     GP transient                6
//! X9-X15    caller-saved     GP local cache (tier 2)     7
//! X16-X17   —                scratch                     2
//! X18       —                platform (reserved)         1
//! X19-X22   callee-saved     fixed (ctx/fp/mem0)         4
//! X23-X28   callee-saved     GP local cache (tier 1)     6
//! X29-X30   —                FP/LR (reserved)            2
//! ──────────────────────────────────────────────────────────
//!                            GP transient                6
//!                            GP local cache total       13
//! ```
//!
//! # FP register plan (D0-D31)
//!
//! ```text
//! Reg       AAPCS64          Role                    Count
//! ──────────────────────────────────────────────────────────
//! D0-D2     caller-saved     scratch                    3
//! D3-D7     caller-saved     FP transient (TOS)         5
//! D8-D15    callee-saved     FP local cache (tier 1)    8
//! D16-D20   caller-saved     FP transient (TOS)         5
//! D21-D31   caller-saved     FP local cache (tier 2)   11
//! ──────────────────────────────────────────────────────────
//!                            scratch                    3
//!                            FP transient total        10
//!                            FP local cache total      19
//! ```
//!
//! Helper call cost by tier (same for GP and FP):
//! - scratch/transient: zero (dead at call boundaries)
//! - tier 1 cache (callee-saved): zero (AAPCS64, Rust preserves them)
//! - tier 2 cache (caller-saved): save/restore only in-use locals at call sites

use crate::{
    error::WasmError,
    vm::machine::mir::{
        MachineReg, MACHINE_CTX_REG, MACHINE_FIXED_REG_COUNT, MACHINE_FP_REG,
        MACHINE_MEM0_BASE_REG, MACHINE_MEM0_SIZE_REG,
    },
};

use super::{emit::Arm64TextEmitter, enc, reg::Arm64Reg};

pub(super) const SCRATCH0: Arm64Reg = Arm64Reg::X16;
pub(super) const SCRATCH1: Arm64Reg = Arm64Reg::X17;

/// FP scratch registers (caller-saved, not used for parameter passing in our ABI).
pub(super) const FP_SCRATCH0: u32 = 0; // D0/S0
pub(super) const FP_SCRATCH1: u32 = 1; // D1/S1
pub(super) const FP_SCRATCH2: u32 = 2; // D2/S2

/// Caller-saved FP registers reserved for transient SSA values (TOS lanes).
///
/// Transients are dead at helper and local-call boundaries
/// (`ensure_no_live_values`), so they can freely use caller-saved regs.
///
/// The physical D-register numbers are non-contiguous (D3-D7 then D16-D20)
/// because AAPCS64 makes D8-D15 callee-saved. Those go to the local cache
/// below so that hot locals survive Rust helper calls for free. The remaining
/// caller-saved regs are split between transients and extra cache capacity.
const FP_TRANSIENT_REGS: [u32; 10] = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20];

/// FP registers reserved for cached locals.
///
/// Cached locals persist across block boundaries. The array is ordered so that
/// the first 8 entries are AAPCS64 callee-saved (D8-D15) and survive Rust
/// helper calls for free. The remaining entries are caller-saved (D21-D31)
/// and are spilled/reloaded around helper calls by the portable lowering
/// (`emit_save_all_cached_locals`), which saves every in-use cached local
/// regardless of its physical register.
///
/// `analyze_local_cache_prefs` sorts locals by usage weight, so the hottest
/// locals naturally land in the first (callee-saved) slots.
const FP_LOCAL_CACHE_REGS: [u32; 19] = [
    8, 9, 10, 11, 12, 13, 14, 15, // callee-saved (D8-D15)
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, // caller-saved (D21-D31)
];

/// Total FP machine-register capacity: transients first, then cached locals.
pub(super) const FP_MACHINE_REG_COUNT: usize = FP_TRANSIENT_REGS.len() + FP_LOCAL_CACHE_REGS.len();

// Compile-time check: transients + cache + scratch must cover all 32 D-regs.
const _: () = assert!(
    FP_TRANSIENT_REGS.len() + FP_LOCAL_CACHE_REGS.len() + 3 == 32,
    "FP register plan must account for all 32 D-registers"
);

/// GP registers available to the machine-reg file, ordered for the regfile
/// layout: local cache first (callee-saved tier 1, then caller-saved tier 2),
/// then transients.
const DYNAMIC_REGS: [Arm64Reg; 19] = [
    // GP local cache tier 1: callee-saved — free at helper calls
    Arm64Reg::X23,
    Arm64Reg::X24,
    Arm64Reg::X25,
    Arm64Reg::X26,
    Arm64Reg::X27,
    Arm64Reg::X28,
    // GP local cache tier 2: caller-saved — save/restore at helper calls
    Arm64Reg::X9,
    Arm64Reg::X10,
    Arm64Reg::X11,
    Arm64Reg::X12,
    Arm64Reg::X13,
    Arm64Reg::X14,
    Arm64Reg::X15,
    // GP transient: caller-saved — dead at call boundaries
    Arm64Reg::X3,
    Arm64Reg::X4,
    Arm64Reg::X5,
    Arm64Reg::X6,
    Arm64Reg::X7,
    Arm64Reg::X8,
];

const CALLEE_SAVED_GP_PAIRS: [(Arm64Reg, Arm64Reg); 6] = [
    (Arm64Reg::X19, Arm64Reg::X20),
    (Arm64Reg::X21, Arm64Reg::X22),
    (Arm64Reg::X23, Arm64Reg::X24),
    (Arm64Reg::X25, Arm64Reg::X26),
    (Arm64Reg::X27, Arm64Reg::X28),
    (Arm64Reg::X29, Arm64Reg::X30),
];

/// AAPCS64 callee-saved FP regs that the shared prologue/epilogue must
/// save/restore. These are D8-D15 — the only FP regs the Rust entry-point
/// caller expects us to preserve.
const CALLEE_SAVED_FP_REGS: [u32; 8] = [8, 9, 10, 11, 12, 13, 14, 15];
const STACK_SLOT_BYTES: u32 = core::mem::size_of::<u64>() as u32;
const STACK_ALIGNMENT_BYTES: u32 = 16;
const CALLEE_SAVED_GP_FRAME_SIZE: u32 = CALLEE_SAVED_GP_PAIRS.len() as u32 * (2 * STACK_SLOT_BYTES);
const CALLEE_SAVED_FP_FRAME_OFFSET: u32 = CALLEE_SAVED_GP_FRAME_SIZE;
const CALLEE_SAVED_FP_FRAME_SIZE: u32 = CALLEE_SAVED_FP_REGS.len() as u32 * STACK_SLOT_BYTES;
const CALLEE_SAVED_FRAME_SIZE: u32 = align_up_u32(
    CALLEE_SAVED_FP_FRAME_OFFSET + CALLEE_SAVED_FP_FRAME_SIZE,
    STACK_ALIGNMENT_BYTES,
);

#[inline]
pub(super) const fn max_gp_mapped_regs() -> usize {
    MACHINE_FIXED_REG_COUNT as usize + DYNAMIC_REGS.len()
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
pub(super) const fn fp_machine_reg(index: usize) -> Option<u32> {
    if index < FP_TRANSIENT_REGS.len() {
        return Some(FP_TRANSIENT_REGS[index]);
    }
    let local_index = index - FP_TRANSIENT_REGS.len();
    if local_index < FP_LOCAL_CACHE_REGS.len() {
        return Some(FP_LOCAL_CACHE_REGS[local_index]);
    }
    None
}

#[inline]
pub(super) fn map_fixed_reg(reg: MachineReg) -> Arm64Reg {
    match reg {
        MACHINE_CTX_REG => Arm64Reg::X19,
        MACHINE_FP_REG => Arm64Reg::X20,
        MACHINE_MEM0_BASE_REG => Arm64Reg::X21,
        MACHINE_MEM0_SIZE_REG => Arm64Reg::X22,
        _ => unreachable!("not a fixed machine reg"),
    }
}

#[inline]
pub(super) fn map_reg(reg: MachineReg) -> Result<Arm64Reg, WasmError> {
    if reg.0 < MACHINE_FIXED_REG_COUNT {
        return Ok(map_fixed_reg(reg));
    }
    DYNAMIC_REGS
        .get((reg.0 - MACHINE_FIXED_REG_COUNT) as usize)
        .copied()
        .ok_or_else(|| {
            WasmError::invalid(alloc::format!(
                "arm64 MachineIR backend has no physical mapping for machine reg {}",
                reg.0
            ))
        })
}

#[inline]
pub(super) fn inv_map_reg(reg: Arm64Reg) -> MachineReg {
    match reg {
        Arm64Reg::X19 => MACHINE_CTX_REG,
        Arm64Reg::X20 => MACHINE_FP_REG,
        Arm64Reg::X21 => MACHINE_MEM0_BASE_REG,
        Arm64Reg::X22 => MACHINE_MEM0_SIZE_REG,
        other => {
            let index = DYNAMIC_REGS
                .iter()
                .position(|candidate| *candidate == other)
                .expect("mapped reg must come from dynamic table");
            MachineReg(MACHINE_FIXED_REG_COUNT + index as u16)
        }
    }
}

pub(super) fn emit_shared_prologue(text: &mut Arm64TextEmitter) {
    text.emit_u32(enc::sub_imm_64(
        Arm64Reg::SP,
        Arm64Reg::SP,
        CALLEE_SAVED_FRAME_SIZE,
    ));
    for (index, (lhs, rhs)) in CALLEE_SAVED_GP_PAIRS.iter().copied().enumerate() {
        text.emit_u32(enc::stp_64(
            lhs,
            rhs,
            Arm64Reg::SP,
            stack_pair_imm((index as u32) * 2 * STACK_SLOT_BYTES),
        ));
    }
    for (index, reg) in CALLEE_SAVED_FP_REGS.iter().copied().enumerate() {
        text.emit_u32(enc::str_d(
            reg,
            Arm64Reg::SP,
            stack_u64_slot(CALLEE_SAVED_FP_FRAME_OFFSET + index as u32 * STACK_SLOT_BYTES),
        ));
    }
}

pub(super) fn emit_shared_epilogue(text: &mut Arm64TextEmitter) {
    for (index, reg) in CALLEE_SAVED_FP_REGS.iter().copied().enumerate() {
        text.emit_u32(enc::ldr_d(
            reg,
            Arm64Reg::SP,
            stack_u64_slot(CALLEE_SAVED_FP_FRAME_OFFSET + index as u32 * STACK_SLOT_BYTES),
        ));
    }
    for (index, (lhs, rhs)) in CALLEE_SAVED_GP_PAIRS.iter().copied().enumerate() {
        text.emit_u32(enc::ldp_64(
            lhs,
            rhs,
            Arm64Reg::SP,
            stack_pair_imm((index as u32) * 2 * STACK_SLOT_BYTES),
        ));
    }
    text.emit_u32(enc::add_imm_64(
        Arm64Reg::SP,
        Arm64Reg::SP,
        CALLEE_SAVED_FRAME_SIZE,
    ));
    text.emit_u32(enc::ret());
}

const fn align_up_u32(value: u32, align: u32) -> u32 {
    value.div_ceil(align) * align
}

const fn stack_u64_slot(offset_bytes: u32) -> u32 {
    offset_bytes / STACK_SLOT_BYTES
}

const fn stack_pair_imm(offset_bytes: u32) -> i32 {
    (offset_bytes / STACK_SLOT_BYTES) as i32
}
