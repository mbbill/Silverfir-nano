//! Microsoft x64 (Win64) calling convention.
//!
//! - First 4 integer args in RCX, RDX, R8, R9.
//! - Integer return in RAX. Sub-64-bit returns (e.g. `u32`) are zero-
//!   extended in practice by both MSVC and MinGW but the ABI does not
//!   strictly mandate it.
//! - Callee-saved GP set adds RDI and RSI on top of the SysV set
//!   (RBX, RBP, R12–R15, RDI, RSI).
//! - Callee-saved XMM set: XMM6–XMM15. We spill those in the prologue.
//! - Shadow space: the caller must reserve 32 bytes of stack for the
//!   callee's register spill area, even if the callee does not use it.

use crate::vm::jit::arch::common::text_emitter::TextEmitter;

use super::super::backend::X86_64Backend;
use super::super::reg::X86Reg;

// ── C ABI boundary registers ────────────────────────────────────────────────

pub(in crate::vm::jit::arch::x86_64) const C_ARG0: X86Reg = X86Reg::RCX;
pub(in crate::vm::jit::arch::x86_64) const C_ARG1: X86Reg = X86Reg::RDX;
pub(in crate::vm::jit::arch::x86_64) const C_ARG2: X86Reg = X86Reg::R8;
pub(in crate::vm::jit::arch::x86_64) const C_RET0: X86Reg = X86Reg::RAX;

// ── Callee-saved set ────────────────────────────────────────────────────────

pub(in crate::vm::jit::arch::x86_64) const CALLEE_SAVED_GP: &[X86Reg] = &[
    X86Reg::RBX,
    X86Reg::RBP,
    X86Reg::R12,
    X86Reg::R13,
    X86Reg::R14,
    X86Reg::R15,
    X86Reg::RDI,
    X86Reg::RSI,
];

// ── Stack frame shape ───────────────────────────────────────────────────────
//
// Layout after entry (from higher addresses to lower):
//   return address                   8
//   callee-saved GP                  8 * 8 = 64
//   XMM6..XMM15 spill area (10 * 16) 160
//   shadow space for our own calls   32
//   alignment padding                0 or 8
//
// `STACK_PADDING` is the total size of "everything below the GP saves",
// so backend.rs's common prologue path subtracts it from RSP uniformly
// and the XMM stores land in the expected slots.

pub(in crate::vm::jit::arch::x86_64) const STACK_PADDING: u32 = {
    let gp = CALLEE_SAVED_GP.len() as u32 * 8;
    let xmm = 10 * 16u32;
    let shadow = 32u32;
    let extra = xmm + shadow;
    let pre = 8 + gp + extra;
    extra + if pre % 16 == 0 { 0 } else { 8 }
};

// Byte offset of XMM6 within the stack padding (just past the shadow space).
const XMM_SPILL_OFFSET: i32 = 32;

// ── Prologue / epilogue extras ──────────────────────────────────────────────
//
// Win64 callers are responsible for spilling XMM6..XMM15 across calls
// because those registers are callee-saved. We spill them immediately
// after subtracting RSP by `STACK_PADDING`.

pub(in crate::vm::jit::arch::x86_64) fn emit_prologue_extra(backend: &mut X86_64Backend) {
    for i in 0..10u32 {
        movaps_store_rsp(
            &mut backend.core.text,
            6 + i,
            XMM_SPILL_OFFSET + (i * 16) as i32,
        );
    }
}

pub(in crate::vm::jit::arch::x86_64) fn emit_epilogue_extra(backend: &mut X86_64Backend) {
    for i in 0..10u32 {
        movaps_load_rsp(
            &mut backend.core.text,
            6 + i,
            XMM_SPILL_OFFSET + (i * 16) as i32,
        );
    }
}

// ── Win64-only instruction encoders ─────────────────────────────────────────
//
// MOVAPS is otherwise unused across the x86_64 backend — only the Win64
// prologue/epilogue need to spill and restore the XMM6..XMM15 callee-saved
// set. Keeping these here (rather than in `arch/x86_64/enc.rs`) means
// `enc.rs` stays ABI-agnostic with zero `sf_os_windows` cfgs.

/// MOVAPS [RSP+disp], XMMn
fn movaps_store_rsp(e: &mut TextEmitter, xmm: u32, disp: i32) {
    let reg = (xmm & 7) as u8;
    if xmm >= 8 {
        e.emit_u8(0x44);
    }
    e.emit_bytes(&[0x0F, 0x29]);
    if disp == 0 {
        e.emit_bytes(&[reg << 3 | 0x04, 0x24]);
    } else if (-128..=127).contains(&disp) {
        e.emit_bytes(&[0x40 | reg << 3 | 0x04, 0x24, disp as u8]);
    } else {
        e.emit_bytes(&[0x80 | reg << 3 | 0x04, 0x24]);
        e.emit_bytes(&(disp as i32).to_le_bytes());
    }
}

/// MOVAPS XMMn, [RSP+disp]
fn movaps_load_rsp(e: &mut TextEmitter, xmm: u32, disp: i32) {
    let reg = (xmm & 7) as u8;
    if xmm >= 8 {
        e.emit_u8(0x44);
    }
    e.emit_bytes(&[0x0F, 0x28]);
    if disp == 0 {
        e.emit_bytes(&[reg << 3 | 0x04, 0x24]);
    } else if (-128..=127).contains(&disp) {
        e.emit_bytes(&[0x40 | reg << 3 | 0x04, 0x24, disp as u8]);
    } else {
        e.emit_bytes(&[0x80 | reg << 3 | 0x04, 0x24]);
        e.emit_bytes(&(disp as i32).to_le_bytes());
    }
}
