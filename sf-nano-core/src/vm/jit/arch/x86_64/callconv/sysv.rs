//! System V AMD64 calling convention (Linux, macOS, FreeBSD, …).
//!
//! - First 6 integer args in RDI, RSI, RDX, RCX, R8, R9.
//! - Integer return in RAX; a `repr(C)` struct of two 64-bit fields is
//!   returned in RAX/RDX.
//! - Callee-saved GP set: RBX, RBP, R12, R13, R14, R15 (6 regs).
//! - No callee-saved XMM registers.

use super::super::backend::X86_64Backend;
use super::super::reg::X86Reg;

// ── C ABI boundary registers ────────────────────────────────────────────────

pub(in crate::vm::jit::arch::x86_64) const C_ARG0: X86Reg = X86Reg::RDI;
pub(in crate::vm::jit::arch::x86_64) const C_ARG1: X86Reg = X86Reg::RSI;
pub(in crate::vm::jit::arch::x86_64) const C_ARG2: X86Reg = X86Reg::RDX;
pub(in crate::vm::jit::arch::x86_64) const C_RET0: X86Reg = X86Reg::RAX;

// ── Callee-saved set ────────────────────────────────────────────────────────

pub(in crate::vm::jit::arch::x86_64) const CALLEE_SAVED_GP: &[X86Reg] = &[
    X86Reg::RBX,
    X86Reg::RBP,
    X86Reg::R12,
    X86Reg::R13,
    X86Reg::R14,
    X86Reg::R15,
];

// ── Raw C calls ─────────────────────────────────────────────────────────────

/// XMM registers a C callee preserves.
pub(in crate::vm::jit::arch::x86_64) const CALLEE_SAVED_XMM: &[u32] = &[];
/// Stack a caller reserves at RSP for the callee's register home area.
pub(in crate::vm::jit::arch::x86_64) const CALL_SHADOW_BYTES: u32 = 0;

// ── Stack frame shape ───────────────────────────────────────────────────────
//
// After pushing the callee-saved GP set (6 regs = 48 bytes) + the return
// address (8 bytes) = 56 bytes. To maintain 16-byte RSP alignment we need
// 8 more bytes of padding (56 + 8 = 64 = 16*4).

pub(in crate::vm::jit::arch::x86_64) const STACK_PADDING: u32 = {
    let n = CALLEE_SAVED_GP.len() as u32;
    if (8 + n * 8) % 16 == 0 {
        0
    } else {
        8
    }
};

// ── Prologue / epilogue extras ──────────────────────────────────────────────
//
// System V has no callee-saved XMM registers, so there is nothing extra
// to spill after the GP saves.

#[inline]
pub(in crate::vm::jit::arch::x86_64) fn emit_prologue_extra(_backend: &mut X86_64Backend) {}

#[inline]
pub(in crate::vm::jit::arch::x86_64) fn emit_epilogue_extra(_backend: &mut X86_64Backend) {}
