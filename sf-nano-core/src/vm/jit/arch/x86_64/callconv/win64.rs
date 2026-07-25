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

use crate::vm::jit::machine::machine_ir::MACHINE_CTX_REG;

use crate::error::WasmError;
use crate::vm::jit::arch::common::text_emitter::TextEmitter;
use crate::vm::jit::runtime::context::NativeContext;

use super::super::abi::map_fixed_reg;
use super::super::backend::X86_64Backend;
use super::super::enc::{self, Cc};
use super::super::helpers::{
    trunc_f32_to_i32_s, trunc_f32_to_i32_u, trunc_f32_to_i64_s, trunc_f32_to_i64_u,
    trunc_f64_to_i32_s, trunc_f64_to_i32_u, trunc_f64_to_i64_s, trunc_f64_to_i64_u,
};
use super::super::reg::X86Reg;

// ── C ABI boundary registers ────────────────────────────────────────────────

pub(in crate::vm::jit::arch::x86_64) const C_ARG0: X86Reg = X86Reg::RCX;
pub(in crate::vm::jit::arch::x86_64) const C_ARG1: X86Reg = X86Reg::RDX;
pub(in crate::vm::jit::arch::x86_64) const C_ARG2: X86Reg = X86Reg::R8;
pub(in crate::vm::jit::arch::x86_64) const C_ARG3: X86Reg = X86Reg::R9;
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

// ── Win64-only runtime helper ───────────────────────────────────────────────
//
// Win64 variant of the trapping truncation helper. Win64 does not return a
// two-field `repr(C)` struct in registers, so the caller passes an out-
// pointer as a fourth argument and receives the status as a `u32` return
// value. The SysV variant lives in `arch/x86_64/helpers.rs` (as
// `x86_64_trapping_trunc`) because it is always compiled — Win64's variant
// lives here so that SysV builds don't carry it as dead code, and so that
// the `#[cfg(sf_os_windows)]` gate is inherited from the parent callconv
// module instead of being spelled out on the function itself.

unsafe extern "C" fn x86_64_trapping_trunc_win(
    ctx: *mut NativeContext,
    src_bits: u64,
    op_code: u64,
    out_value: *mut u64,
) -> u32 {
    let result = match op_code {
        0 => trunc_f32_to_i32_s(src_bits as u32),
        1 => trunc_f32_to_i32_u(src_bits as u32),
        2 => trunc_f64_to_i32_s(src_bits),
        3 => trunc_f64_to_i32_u(src_bits),
        4 => trunc_f32_to_i64_s(src_bits as u32),
        5 => trunc_f32_to_i64_u(src_bits as u32),
        6 => trunc_f64_to_i64_s(src_bits),
        7 => trunc_f64_to_i64_u(src_bits),
        _ => Err(WasmError::trap("invalid trunc op")),
    };
    match result {
        Ok(value) => {
            unsafe { *out_value = value };
            0
        }
        Err(err) => {
            if let Some(ctx) = unsafe { ctx.as_mut() } {
                ctx.error = Some(err);
            }
            1
        }
    }
}

// ── Trapping trunc helper call ──────────────────────────────────────────────
//
// The Win64 helper has signature:
//   extern "C" fn(ctx, src_bits, op_code, out_value: *mut u64) -> u32
// because Win64 does not return a 2-field `repr(C)` struct in registers.
// We pass RSP as the out-pointer (the save-helper just pushed 64 bytes,
// so [RSP + 0] is the padding slot and is safe scratch).

pub(in crate::vm::jit::arch::x86_64) fn emit_trapping_trunc_call(
    backend: &mut X86_64Backend,
    src: X86Reg,
    op_code: u64,
    result_scratch: X86Reg,
    error_label: usize,
) {
    backend.save_caller_clobbered_gp_dynamic();

    // Read `src` into C_ARG1 (RDX) before clobbering C_ARG0 (RCX) with ctx.
    // `src_gp` comes from the backend-owned GP scratch bank {RAX, RCX, RDX},
    // so if it lives in RCX, writing ctx into RCX first would destroy the
    // float bits before the `C_ARG1 <- src` move could forward them.
    enc::mov_rr_64(&mut backend.core.text, C_ARG1, src);
    enc::mov_rr_64(
        &mut backend.core.text,
        C_ARG0,
        map_fixed_reg(MACHINE_CTX_REG),
    );
    backend.materialize_u64(C_ARG2, op_code);
    // Out-pointer = current RSP (scratch in the padding slot above).
    enc::mov_rr_64(&mut backend.core.text, C_ARG3, X86Reg::RSP);
    backend.materialize_u64(
        X86Reg::R11,
        x86_64_trapping_trunc_win as *const () as usize as u64,
    );
    enc::call_reg(&mut backend.core.text, X86Reg::R11);
    // Read result from [RSP + 0].
    enc::load_64(&mut backend.core.text, result_scratch, X86Reg::RSP, 0);

    backend.restore_caller_clobbered_gp_dynamic();

    // Status is `u32` in EAX. Upper 32 bits are zeroed in practice but
    // test the low 32 explicitly to match the pre-refactor sequence.
    enc::test_rr_32(&mut backend.core.text, X86Reg::RAX, X86Reg::RAX);
    backend.emit_jcc(Cc::NE, error_label);
}
