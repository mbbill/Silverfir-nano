//! x86_64 physical register mapping and ABI-derived layout.
//!
//! Source of truth for System V AMD64 ABI facts and our physical register
//! mapping. Policy (how many of these to use) lives in `config.rs`.
//!
//! # GP register plan
//!
//! ```text
//! Reg       ABI              Role                      Count
//! ────────────────────────────────────────────────────────────
//! RAX       caller-saved     scratch / return value        1
//! RCX       caller-saved     GP transient                  1
//! RDX       caller-saved     GP transient                  1
//! RSI       caller-saved     GP transient (entry: fp arg)  1
//! RDI       caller-saved     GP transient (entry: ctx arg) 1
//! R8        caller-saved     GP transient                  1
//! R9        caller-saved     GP transient                  1
//! R10       caller-saved     GP transient                  1
//! R11       caller-saved     scratch (helper call addr)    1
//! RBX       callee-saved     MACHINE_CTX_REG (fixed)       1
//! RBP       callee-saved     MACHINE_FP_REG (fixed)        1
//! R12       callee-saved     MACHINE_MEM0_BASE_REG (fixed) 1
//! R13       callee-saved     MACHINE_MEM0_SIZE_REG (fixed) 1
//! R14       callee-saved     GP local cache (tier 1)       1
//! R15       callee-saved     GP local cache (tier 1)       1
//! RSP       —                stack pointer                  1
//! ────────────────────────────────────────────────────────────
//!                            GP transient                   7
//!                            GP local cache total           2
//! ```
//!
//! # FP register plan (XMM0-XMM15)
//!
//! All XMM regs are caller-saved under System V AMD64. We designate:
//! - XMM0-XMM5: FP transients (6)
//! - XMM6-XMM15: FP local cache (10) — save/restore at helper calls
//!
//! There are no callee-saved XMM regs on System V, so all cached locals
//! must be spilled around helper calls (the lowering already handles this).

use crate::{
    error::WasmError,
    vm::native::ir::machine::{
        MachineReg, MACHINE_CTX_REG, MACHINE_FIXED_REG_COUNT, MACHINE_FP_REG,
        MACHINE_MEM0_BASE_REG, MACHINE_MEM0_SIZE_REG,
    },
};

use super::{
    emit::X86_64TextEmitter,
    reg::X86Reg,
};

pub(super) const SCRATCH0: X86Reg = X86Reg::RAX;
pub(super) const SCRATCH1: X86Reg = X86Reg::R11;

/// FP scratch registers (overlap with transients — lowering ensures they're
/// dead at points where scratch is needed).
pub(super) const FP_SCRATCH0: u32 = 0; // XMM0
pub(super) const FP_SCRATCH1: u32 = 1; // XMM1
pub(super) const FP_SCRATCH2: u32 = 2; // XMM2

/// FP transient XMM registers: XMM0-XMM5.
const FP_TRANSIENT_REGS: [u32; 6] = [0, 1, 2, 3, 4, 5];

/// FP cached local XMM registers: XMM6-XMM15.
/// All caller-saved on System V — must be spilled around helper calls.
const FP_LOCAL_CACHE_REGS: [u32; 10] = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// Total FP machine-register capacity.
pub(super) const FP_MACHINE_REG_COUNT: usize =
    FP_TRANSIENT_REGS.len() + FP_LOCAL_CACHE_REGS.len();

// Compile-time check: transients + cache = 16 XMM regs.
const _: () = assert!(
    FP_TRANSIENT_REGS.len() + FP_LOCAL_CACHE_REGS.len() == 16,
    "FP register plan must account for all 16 XMM registers"
);

/// GP registers available to the machine-reg file, ordered: local cache first,
/// then transients. This matches the ARM64 convention where the regfile layout
/// puts cached locals before transients.
const DYNAMIC_REGS: [X86Reg; 9] = [
    // GP local cache tier 1: callee-saved — free at helper calls
    X86Reg::R14,
    X86Reg::R15,
    // GP transient: caller-saved — dead at call boundaries
    X86Reg::RCX,
    X86Reg::RDX,
    X86Reg::RSI,
    X86Reg::RDI,
    X86Reg::R8,
    X86Reg::R9,
    X86Reg::R10,
];

/// Callee-saved GP registers we must push/pop in prologue/epilogue.
/// These are all the callee-saved regs we use: fixed + cached locals.
const CALLEE_SAVED_GP_REGS: [X86Reg; 6] = [
    X86Reg::RBX,  // MACHINE_CTX_REG
    X86Reg::RBP,  // MACHINE_FP_REG
    X86Reg::R12,  // MACHINE_MEM0_BASE_REG
    X86Reg::R13,  // MACHINE_MEM0_SIZE_REG
    X86Reg::R14,  // GP local cache
    X86Reg::R15,  // GP local cache
];

// No callee-saved FP regs on System V AMD64.

const STACK_SLOT_BYTES: u32 = core::mem::size_of::<u64>() as u32;

/// After pushing CALLEE_SAVED_GP_REGS (6 regs = 48 bytes) + return address
/// (8 bytes) = 56 bytes. To maintain 16-byte alignment we need 8 more bytes
/// of padding (56 + 8 = 64 = 16*4).
pub(super) const STACK_PADDING: u32 = 8;

/// Total frame created by shared prologue = pushes + padding.
pub(super) const CALLEE_SAVED_FRAME_SIZE: u32 =
    CALLEE_SAVED_GP_REGS.len() as u32 * STACK_SLOT_BYTES + STACK_PADDING;

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
pub(super) fn map_fixed_reg(reg: MachineReg) -> X86Reg {
    match reg {
        MACHINE_CTX_REG => X86Reg::RBX,
        MACHINE_FP_REG => X86Reg::RBP,
        MACHINE_MEM0_BASE_REG => X86Reg::R12,
        MACHINE_MEM0_SIZE_REG => X86Reg::R13,
        _ => unreachable!("not a fixed machine reg"),
    }
}

#[inline]
pub(super) fn map_reg(reg: MachineReg) -> Result<X86Reg, WasmError> {
    if reg.0 < MACHINE_FIXED_REG_COUNT {
        return Ok(map_fixed_reg(reg));
    }
    DYNAMIC_REGS
        .get((reg.0 - MACHINE_FIXED_REG_COUNT) as usize)
        .copied()
        .ok_or_else(|| {
            WasmError::invalid(alloc::format!(
                "x86_64 MachineIR backend has no physical mapping for machine reg {}",
                reg.0
            ))
        })
}

#[inline]
pub(super) fn inv_map_reg(reg: X86Reg) -> MachineReg {
    match reg {
        X86Reg::RBX => MACHINE_CTX_REG,
        X86Reg::RBP => MACHINE_FP_REG,
        X86Reg::R12 => MACHINE_MEM0_BASE_REG,
        X86Reg::R13 => MACHINE_MEM0_SIZE_REG,
        other => {
            let index = DYNAMIC_REGS
                .iter()
                .position(|candidate| *candidate == other)
                .expect("mapped reg must come from dynamic table");
            MachineReg(MACHINE_FIXED_REG_COUNT + index as u16)
        }
    }
}

/// Emit the shared prologue: push callee-saved GP regs, align stack.
///
/// On entry (System V AMD64):
/// - RDI = ctx pointer
/// - RSI = frame base pointer
/// - RSP is 8-byte aligned (return address already pushed by CALL)
///
/// Prologue:
/// 1. Push callee-saved GP regs
/// 2. Subtract padding for 16-byte alignment
/// 3. Move args to fixed regs (RDI -> RBX, RSI -> RBP)
pub(super) fn emit_shared_prologue(text: &mut X86_64TextEmitter) {
    // Push callee-saved GP regs
    for &reg in &CALLEE_SAVED_GP_REGS {
        emit_push(text, reg);
    }
    // Align stack: sub rsp, STACK_PADDING
    if STACK_PADDING > 0 {
        // SUB RSP, imm8
        emit_sub_rsp_imm8(text, STACK_PADDING as u8);
    }
}

/// Emit the shared epilogue: restore stack, pop callee-saved, ret.
pub(super) fn emit_shared_epilogue(text: &mut X86_64TextEmitter) {
    // Undo stack alignment
    if STACK_PADDING > 0 {
        emit_add_rsp_imm8(text, STACK_PADDING as u8);
    }
    // Pop callee-saved GP regs in reverse order
    for &reg in CALLEE_SAVED_GP_REGS.iter().rev() {
        emit_pop(text, reg);
    }
    // RET
    text.emit_u8(0xC3);
}

// --- Low-level prologue/epilogue helpers ---

fn emit_push(text: &mut X86_64TextEmitter, reg: X86Reg) {
    if reg.needs_rex_ext() {
        text.emit_u8(0x41); // REX.B
    }
    text.emit_u8(0x50 + reg.idx3());
}

fn emit_pop(text: &mut X86_64TextEmitter, reg: X86Reg) {
    if reg.needs_rex_ext() {
        text.emit_u8(0x41); // REX.B
    }
    text.emit_u8(0x58 + reg.idx3());
}

fn emit_sub_rsp_imm8(text: &mut X86_64TextEmitter, imm8: u8) {
    // REX.W + SUB r/m64, imm8: 48 83 EC imm8
    text.emit_bytes(&[0x48, 0x83, 0xEC, imm8]);
}

fn emit_add_rsp_imm8(text: &mut X86_64TextEmitter, imm8: u8) {
    // REX.W + ADD r/m64, imm8: 48 83 C4 imm8
    text.emit_bytes(&[0x48, 0x83, 0xC4, imm8]);
}
