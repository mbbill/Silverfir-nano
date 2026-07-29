//! System V AMD64 calling convention (Linux, macOS, FreeBSD, …).
//!
//! - First 6 integer args in RDI, RSI, RDX, RCX, R8, R9.
//! - Integer return in RAX; a `repr(C)` struct of two 64-bit fields is
//!   returned in RAX/RDX.
//! - Callee-saved GP set: RBX, RBP, R12, R13, R14, R15 (6 regs).
//! - No callee-saved XMM registers.

use crate::vm::jit::machine::machine_ir::MACHINE_CTX_REG;

use super::super::abi::map_fixed_reg;
use super::super::backend::X86_64Backend;
use super::super::enc::{self, Cc};
use super::super::helpers::x86_64_trapping_trunc;
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

// ── Trapping trunc helper call ──────────────────────────────────────────────
//
// The SysV helper has signature:
//   extern "C" fn(ctx: *mut NativeContext, src_bits: u64, op_code: u64) -> TruncResult
// where `TruncResult { status: u64, value: u64 }` is returned in RAX/RDX.

pub(in crate::vm::jit::arch::x86_64) fn emit_trapping_trunc_call(
    backend: &mut X86_64Backend,
    src: X86Reg,
    op_code: u64,
    call_scratch: X86Reg,
    error_label: usize,
) {
    backend.save_caller_clobbered_gp_dynamic();

    // Read `src` before clobbering RDI with ctx. `src` is allowed to live in
    // RDI because C ABI arg regs are also normal dynamic GP regs.
    enc::mov_rr_64(&mut backend.core.text, C_ARG1, src);
    enc::mov_rr_64(
        &mut backend.core.text,
        C_ARG0,
        map_fixed_reg(MACHINE_CTX_REG),
    );
    backend.materialize_u64(C_ARG2, op_code);
    backend.materialize_u64(
        call_scratch,
        x86_64_trapping_trunc as *const () as usize as u64,
    );
    enc::call_reg(&mut backend.core.text, call_scratch);

    backend.restore_caller_clobbered_gp_dynamic();

    // Status is in RAX (first field). Branch to error on nonzero.
    enc::test_rr_64(&mut backend.core.text, X86Reg::RAX, X86Reg::RAX);
    backend.emit_jcc(Cc::NE, error_label);
}
