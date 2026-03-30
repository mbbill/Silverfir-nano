//! ARMv7-A physical register mapping and EABI-derived layout.
//!
//! # GP register plan (R0-R15)
//!
//! ```text
//! Reg    EABI             Role                    Count
//! ─────────────────────────────────────────────────────
//! R0-R2  caller-saved     GP transient                3
//! R3     caller-saved     GP transient                1
//! R4     callee-saved     fixed: MEM0_SIZE            1
//! R5-R7  callee-saved     GP local cache              3
//! R8     callee-saved     fixed: CTX                  1
//! R9     platform         GP transient                1
//! R10    callee-saved     fixed: FP                   1
//! R11    callee-saved     fixed: MEM0_BASE            1
//! R12    caller-saved     scratch: SCRATCH0 (IP)      1
//! R13    —                SP (reserved)               1
//! R14    caller-saved     scratch: SCRATCH1 (LR)      1
//! R15    —                PC (reserved)               1
//! ─────────────────────────────────────────────────────
//!                         GP transient                5
//!                         GP local cache              3
//!                         GP scratch                  2
//! ```
//!
//! # FP register plan (D0-D15, VFPv3-D16)
//!
//! ```text
//! Reg    EABI             Role                   Count
//! ─────────────────────────────────────────────────────
//! D0-D2  caller-saved     FP scratch                 3
//! D3-D7  caller-saved     FP transient (TOS)         5
//! D8-D15 callee-saved     FP local cache             8
//! ─────────────────────────────────────────────────────
//!                         FP transient               5
//!                         FP local cache             8
//! ```

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

use super::{enc, reg::Arm32Reg};
use crate::vm::arch::common::{scratch_pool::ScratchPool, text_emitter::TextEmitter};

pub(super) const SCRATCH0: Arm32Reg = Arm32Reg::R12;
/// Call-local scratch. LR is saved in the shared prologue and only used as a
/// transient scratch within straight-line sequences that do not issue calls.
pub(super) const SCRATCH1: Arm32Reg = Arm32Reg::R14;

/// FP scratch registers (caller-saved, not used for values or parameters).
pub(super) const FP_SCRATCH0: u32 = 0; // D0
pub(super) const FP_SCRATCH1: u32 = 1; // D1
pub(super) const FP_SCRATCH2: u32 = 2; // D2

// ── Scratch pool construction ────────────────────────────────────────────────

const GP_SCRATCHES: [Arm32Reg; 2] = [SCRATCH0, SCRATCH1];
const FP_SCRATCHES: [u32; 3] = [FP_SCRATCH0, FP_SCRATCH1, FP_SCRATCH2];

pub(super) fn new_gp_scratch_pool() -> ScratchPool<Arm32Reg, 2> {
    ScratchPool::new(GP_SCRATCHES)
}

pub(super) fn new_fp_scratch_pool() -> ScratchPool<u32, 3> {
    ScratchPool::new(FP_SCRATCHES)
}

// ── Dynamic register arrays (MachineIR allocation) ───────────────────────────
//
// These arrays are the single source of truth for register budgets.
// `config.rs` derives BackendConfig from their lengths.
// Ordering: local-cache first, then transient — must match MachineRegFile layout.

pub(super) const GP_UNIT_BYTES: u8 = 4;

/// GP local cache: preferred for cached locals. Shared lowering
/// synchronizes cached locals through frame slots across helper/call
/// boundaries, so this bank does not rely on every register being
/// callee-saved in the C ABI.
///
/// R9 is never used for fixed MachineIR state. It stays in the dynamic bank
/// only, so the fixed roles remain pinned to unquestionably preserved
/// registers.
pub(super) const GP_LOCAL_CACHE: [Arm32Reg; 3] = [Arm32Reg::R5, Arm32Reg::R6, Arm32Reg::R7];

/// GP transient: dead at call boundaries.
pub(super) const GP_TRANSIENT: [Arm32Reg; 5] = [
    Arm32Reg::R3,
    Arm32Reg::R9,
    Arm32Reg::R0,
    Arm32Reg::R1,
    Arm32Reg::R2,
];

/// Caller-saved FP registers reserved for transient SSA values (TOS lanes).
pub(super) const FP_TRANSIENT: [u32; 5] = [3, 4, 5, 6, 7];

/// FP registers reserved for cached locals (D8-D15, callee-saved).
pub(super) const FP_LOCAL_CACHE: [u32; 8] = [8, 9, 10, 11, 12, 13, 14, 15];

/// Total FP machine-register capacity: transients first, then cached locals.
pub(super) const FP_MACHINE_REG_COUNT: usize = FP_TRANSIENT.len() + FP_LOCAL_CACHE.len();

// Compile-time check: transients + cache + scratch must cover all 16 D-regs.
const _: () = assert!(
    FP_TRANSIENT.len() + FP_LOCAL_CACHE.len() + 3 == 16,
    "FP register plan must account for all 16 D-registers (VFPv3-D16)"
);

// ── Derived config ───────────────────────────────────────────────────────────

#[inline]
pub(crate) const fn compile_backend_config() -> BackendConfig {
    BackendConfig::new(
        GP_LOCAL_CACHE.len() as u8,
        GP_TRANSIENT.len() as u8,
        FP_LOCAL_CACHE.len() as u8,
        FP_TRANSIENT.len() as u8,
        GP_UNIT_BYTES,
        8,
    )
}

// ── Callee-saved sets (ARMv7-specific encoding) ─────────────────────────────

/// Callee-saved GP registers to save/restore in prologue/epilogue.
/// R4-R11 are callee-saved in EABI; R14(LR) must also be saved since we call helpers.
const CALLEE_SAVED_GP_REGS: [Arm32Reg; 9] = [
    Arm32Reg::R4,
    Arm32Reg::R5,
    Arm32Reg::R6,
    Arm32Reg::R7,
    Arm32Reg::R8,
    Arm32Reg::R9,
    Arm32Reg::R10,
    Arm32Reg::R11,
    Arm32Reg::R14, // LR
];

/// EABI callee-saved FP regs (D8-D15).
const CALLEE_SAVED_FP_FIRST: u32 = 8;
const CALLEE_SAVED_FP_COUNT: u32 = 8;

const SHARED_PROLOGUE_ALIGN_PAD_BYTES: u32 = if CALLEE_SAVED_GP_REGS.len() % 2 == 1 {
    4
} else {
    0
};

#[inline]
pub(super) const fn max_gp_mapped_regs() -> usize {
    MACHINE_FIXED_REG_COUNT as usize + GP_LOCAL_CACHE.len() + GP_TRANSIENT.len()
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
    let config = compile_backend_config();
    let (i, is_cache) = classify_fp_reg(index, config);
    if is_cache {
        FP_LOCAL_CACHE.get(i).copied()
    } else {
        FP_TRANSIENT.get(i).copied()
    }
}

#[inline]
pub(super) fn map_fixed_reg(reg: MachineReg) -> Arm32Reg {
    match reg {
        MACHINE_CTX_REG => Arm32Reg::R8,
        MACHINE_FP_REG => Arm32Reg::R10,
        MACHINE_MEM0_BASE_REG => Arm32Reg::R11,
        MACHINE_MEM0_SIZE_REG => Arm32Reg::R4,
        _ => unreachable!("not a fixed machine reg"),
    }
}

#[inline]
pub(super) fn map_reg(reg: MachineReg) -> Result<Arm32Reg, WasmError> {
    if reg.0 < MACHINE_FIXED_REG_COUNT {
        return Ok(map_fixed_reg(reg));
    }
    let config = compile_backend_config();
    match classify_gp_reg(reg, config) {
        Some((index, true)) => GP_LOCAL_CACHE.get(index).copied(),
        Some((index, false)) => GP_TRANSIENT.get(index).copied(),
        None => return Ok(map_fixed_reg(reg)),
    }
    .ok_or_else(|| {
        WasmError::invalid(alloc::format!(
            "armv7a MachineIR backend has no physical mapping for machine reg {}",
            reg.0
        ))
    })
}

#[inline]
pub(super) fn inv_map_reg(reg: Arm32Reg) -> MachineReg {
    match reg {
        Arm32Reg::R8 => MACHINE_CTX_REG,
        Arm32Reg::R10 => MACHINE_FP_REG,
        Arm32Reg::R11 => MACHINE_MEM0_BASE_REG,
        Arm32Reg::R4 => MACHINE_MEM0_SIZE_REG,
        other => {
            if let Some(i) = GP_LOCAL_CACHE.iter().position(|c| *c == other) {
                return MachineReg(MACHINE_FIXED_REG_COUNT + i as u16);
            }
            let i = GP_TRANSIENT
                .iter()
                .position(|c| *c == other)
                .expect("mapped reg must come from dynamic table");
            MachineReg(MACHINE_FIXED_REG_COUNT + GP_LOCAL_CACHE.len() as u16 + i as u16)
        }
    }
}

/// Build the register mask for PUSH/POP from the callee-saved list.
fn callee_saved_gp_mask() -> u16 {
    let mut mask = 0u16;
    let mut i = 0;
    while i < CALLEE_SAVED_GP_REGS.len() {
        mask |= 1 << CALLEE_SAVED_GP_REGS[i].idx();
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
    // VPUSH {D8-D15}
    text.emit_u32(enc::vpush_d(CALLEE_SAVED_FP_FIRST, CALLEE_SAVED_FP_COUNT));
}

pub(super) fn emit_shared_epilogue(text: &mut TextEmitter) {
    // VPOP {D8-D15}
    text.emit_u32(enc::vpop_d(CALLEE_SAVED_FP_FIRST, CALLEE_SAVED_FP_COUNT));
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
