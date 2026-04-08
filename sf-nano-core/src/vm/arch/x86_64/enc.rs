//! x86_64 instruction encoding.
//!
//! Provides helpers to emit x86_64 machine code into an `TextEmitter`.
//! All instructions are emitted as variable-length byte sequences.
//!
//! Encoding overview:
//! - Optional REX prefix (0x40-0x4F) for 64-bit operand size, extended regs
//! - Opcode (1-3 bytes)
//! - ModR/M byte: mod(2) | reg(3) | rm(3)
//! - Optional SIB byte: scale(2) | index(3) | base(3)
//! - Optional displacement: disp8 or disp32
//! - Optional immediate: imm8, imm16, imm32, or imm64

use super::reg::X86Reg;
use crate::vm::arch::common::text_emitter::TextEmitter;

/// x86_64 condition codes for Jcc / SETcc / CMOVcc.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Cc {
    O = 0x0,
    NO = 0x1,
    B = 0x2,
    AE = 0x3,
    E = 0x4,
    NE = 0x5,
    BE = 0x6,
    A = 0x7,
    S = 0x8,
    NS = 0x9,
    P = 0xA,
    NP = 0xB,
    L = 0xC,
    GE = 0xD,
    LE = 0xE,
    G = 0xF,
}

impl Cc {
    #[inline]
    pub const fn invert(self) -> Cc {
        unsafe { core::mem::transmute(self as u8 ^ 1) }
    }
}

// ─── REX prefix helpers ──────────────────────────────────────────────────────

/// Build REX byte. Any of w/r/x/b can be true.
#[inline]
const fn rex(w: bool, r: bool, x: bool, b: bool) -> u8 {
    0x40 | ((w as u8) << 3) | ((r as u8) << 2) | ((x as u8) << 1) | (b as u8)
}

/// REX.W prefix (64-bit operand size, no extended regs).
const REX_W: u8 = rex(true, false, false, false);

/// Emit REX prefix if needed for reg-reg operations.
/// w = 64-bit operand size, reg = ModR/M.reg field, rm = ModR/M.rm field.
fn emit_rex(e: &mut TextEmitter, w: bool, reg: X86Reg, rm: X86Reg) {
    let r = reg.needs_rex_ext();
    let b = rm.needs_rex_ext();
    if w || r || b {
        e.emit_u8(rex(w, r, false, b));
    }
}

/// Emit REX prefix for SIB addressing.
fn emit_rex_sib(e: &mut TextEmitter, w: bool, reg: X86Reg, index: X86Reg, base: X86Reg) {
    let r = reg.needs_rex_ext();
    let x = index.needs_rex_ext();
    let b = base.needs_rex_ext();
    if w || r || x || b {
        e.emit_u8(rex(w, r, x, b));
    }
}

/// Emit REX prefix for instructions using only rm field (no reg).
fn emit_rex_b(e: &mut TextEmitter, w: bool, rm: X86Reg) {
    let b = rm.needs_rex_ext();
    if w || b {
        e.emit_u8(rex(w, false, false, b));
    }
}

// ─── ModR/M + SIB helpers ────────────────────────────────────────────────────

/// Build ModR/M byte.
#[inline]
const fn modrm(mode: u8, reg: u8, rm: u8) -> u8 {
    (mode << 6) | (reg << 3) | rm
}

/// Build SIB byte.
#[inline]
const fn sib(scale: u8, index: u8, base: u8) -> u8 {
    (scale << 6) | (index << 3) | base
}

/// Emit ModR/M for reg-reg (mod=11).
fn emit_modrm_rr(e: &mut TextEmitter, reg: X86Reg, rm: X86Reg) {
    e.emit_u8(modrm(0b11, reg.idx3(), rm.idx3()));
}

/// Emit ModR/M for [rm] (mod=00), [rm+disp8] (mod=01), [rm+disp32] (mod=10).
/// Handles RSP (SIB needed) and RBP (disp8 needed) special cases.
fn emit_modrm_mem(e: &mut TextEmitter, reg: X86Reg, base: X86Reg, disp: i32) {
    let reg3 = reg.idx3();
    let base3 = base.idx3();

    // RSP/R12 as base requires SIB byte
    if base3 == 4 {
        if disp == 0 {
            // [RSP] => ModR/M=00 + SIB(0,RSP,RSP)
            e.emit_u8(modrm(0b00, reg3, 0b100));
            e.emit_u8(sib(0b00, 0b100, base3));
        } else if fits_i8(disp) {
            e.emit_u8(modrm(0b01, reg3, 0b100));
            e.emit_u8(sib(0b00, 0b100, base3));
            e.emit_u8(disp as u8);
        } else {
            e.emit_u8(modrm(0b10, reg3, 0b100));
            e.emit_u8(sib(0b00, 0b100, base3));
            emit_i32(e, disp);
        }
        return;
    }

    // RBP/R13 as base with zero displacement needs disp8=0
    if base3 == 5 && disp == 0 {
        e.emit_u8(modrm(0b01, reg3, base3));
        e.emit_u8(0);
        return;
    }

    if disp == 0 {
        e.emit_u8(modrm(0b00, reg3, base3));
    } else if fits_i8(disp) {
        e.emit_u8(modrm(0b01, reg3, base3));
        e.emit_u8(disp as u8);
    } else {
        e.emit_u8(modrm(0b10, reg3, base3));
        emit_i32(e, disp);
    }
}

/// Emit ModR/M + SIB for [base + index*scale + disp].
fn emit_modrm_sib(
    e: &mut TextEmitter,
    reg: X86Reg,
    base: X86Reg,
    index: X86Reg,
    scale: u8,
    disp: i32,
) {
    let reg3 = reg.idx3();
    let base3 = base.idx3();
    let index3 = index.idx3();
    let scale_bits = match scale {
        1 => 0b00,
        2 => 0b01,
        4 => 0b10,
        8 => 0b11,
        _ => panic!("invalid SIB scale"),
    };

    if disp == 0 && base3 != 5 {
        // [base + index*scale]
        e.emit_u8(modrm(0b00, reg3, 0b100));
        e.emit_u8(sib(scale_bits, index3, base3));
    } else if fits_i8(disp) {
        e.emit_u8(modrm(0b01, reg3, 0b100));
        e.emit_u8(sib(scale_bits, index3, base3));
        e.emit_u8(disp as u8);
    } else {
        e.emit_u8(modrm(0b10, reg3, 0b100));
        e.emit_u8(sib(scale_bits, index3, base3));
        emit_i32(e, disp);
    }
}

/// Emit RIP-relative ModR/M: [RIP + disp32].
fn emit_modrm_rip(e: &mut TextEmitter, reg: X86Reg, disp32: i32) {
    e.emit_u8(modrm(0b00, reg.idx3(), 0b101));
    emit_i32(e, disp32);
}

#[inline]
fn fits_i8(v: i32) -> bool {
    v >= -128 && v <= 127
}

fn emit_i32(e: &mut TextEmitter, v: i32) {
    e.emit_bytes(&v.to_le_bytes());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Integer ALU — reg, reg
// ═══════════════════════════════════════════════════════════════════════════════

/// Generic ALU reg, reg. `opcode` is the base opcode for "reg, r/m" form.
fn alu_rr(e: &mut TextEmitter, opcode: u8, w: bool, dst: X86Reg, src: X86Reg) {
    emit_rex(e, w, dst, src);
    e.emit_u8(opcode);
    emit_modrm_rr(e, dst, src);
}

/// Generic ALU r/m, imm8 (opcode group /digit, 0x83 for 64-bit).
fn alu_ri8(e: &mut TextEmitter, digit: u8, w: bool, dst: X86Reg, imm8: i8) {
    emit_rex_b(e, w, dst);
    e.emit_u8(0x83);
    e.emit_u8(modrm(0b11, digit, dst.idx3()));
    e.emit_u8(imm8 as u8);
}

/// Generic ALU r/m, imm32 (opcode group /digit, 0x81 for 64-bit).
fn alu_ri32(e: &mut TextEmitter, digit: u8, w: bool, dst: X86Reg, imm: i32) {
    emit_rex_b(e, w, dst);
    e.emit_u8(0x81);
    e.emit_u8(modrm(0b11, digit, dst.idx3()));
    emit_i32(e, imm);
}

// ADD
pub fn add_rr_64(e: &mut TextEmitter, dst: X86Reg, src: X86Reg) {
    alu_rr(e, 0x03, true, dst, src);
}
pub fn add_rr_32(e: &mut TextEmitter, dst: X86Reg, src: X86Reg) {
    alu_rr(e, 0x03, false, dst, src);
}
pub fn add_ri_64(e: &mut TextEmitter, dst: X86Reg, imm: i32) {
    if fits_i8(imm) {
        alu_ri8(e, 0, true, dst, imm as i8);
    } else {
        alu_ri32(e, 0, true, dst, imm);
    }
}
pub fn add_ri_32(e: &mut TextEmitter, dst: X86Reg, imm: i32) {
    if fits_i8(imm) {
        alu_ri8(e, 0, false, dst, imm as i8);
    } else {
        alu_ri32(e, 0, false, dst, imm);
    }
}

// SUB
pub fn sub_rr_64(e: &mut TextEmitter, dst: X86Reg, src: X86Reg) {
    alu_rr(e, 0x2B, true, dst, src);
}
pub fn sub_rr_32(e: &mut TextEmitter, dst: X86Reg, src: X86Reg) {
    alu_rr(e, 0x2B, false, dst, src);
}
pub fn sub_ri_64(e: &mut TextEmitter, dst: X86Reg, imm: i32) {
    if fits_i8(imm) {
        alu_ri8(e, 5, true, dst, imm as i8);
    } else {
        alu_ri32(e, 5, true, dst, imm);
    }
}
pub fn sub_ri_32(e: &mut TextEmitter, dst: X86Reg, imm: i32) {
    if fits_i8(imm) {
        alu_ri8(e, 5, false, dst, imm as i8);
    } else {
        alu_ri32(e, 5, false, dst, imm);
    }
}

// AND
pub fn and_rr_64(e: &mut TextEmitter, dst: X86Reg, src: X86Reg) {
    alu_rr(e, 0x23, true, dst, src);
}
pub fn and_rr_32(e: &mut TextEmitter, dst: X86Reg, src: X86Reg) {
    alu_rr(e, 0x23, false, dst, src);
}
pub fn and_ri_64(e: &mut TextEmitter, dst: X86Reg, imm: i32) {
    if fits_i8(imm) {
        alu_ri8(e, 4, true, dst, imm as i8);
    } else {
        alu_ri32(e, 4, true, dst, imm);
    }
}
pub fn and_ri_32(e: &mut TextEmitter, dst: X86Reg, imm: i32) {
    if fits_i8(imm) {
        alu_ri8(e, 4, false, dst, imm as i8);
    } else {
        alu_ri32(e, 4, false, dst, imm);
    }
}

// OR
pub fn or_rr_64(e: &mut TextEmitter, dst: X86Reg, src: X86Reg) {
    alu_rr(e, 0x0B, true, dst, src);
}
pub fn or_rr_32(e: &mut TextEmitter, dst: X86Reg, src: X86Reg) {
    alu_rr(e, 0x0B, false, dst, src);
}
pub fn or_ri_64(e: &mut TextEmitter, dst: X86Reg, imm: i32) {
    if fits_i8(imm) {
        alu_ri8(e, 1, true, dst, imm as i8);
    } else {
        alu_ri32(e, 1, true, dst, imm);
    }
}
pub fn or_ri_32(e: &mut TextEmitter, dst: X86Reg, imm: i32) {
    if fits_i8(imm) {
        alu_ri8(e, 1, false, dst, imm as i8);
    } else {
        alu_ri32(e, 1, false, dst, imm);
    }
}

// XOR
pub fn xor_rr_64(e: &mut TextEmitter, dst: X86Reg, src: X86Reg) {
    alu_rr(e, 0x33, true, dst, src);
}
pub fn xor_rr_32(e: &mut TextEmitter, dst: X86Reg, src: X86Reg) {
    alu_rr(e, 0x33, false, dst, src);
}
pub fn xor_ri_64(e: &mut TextEmitter, dst: X86Reg, imm: i32) {
    if fits_i8(imm) {
        alu_ri8(e, 6, true, dst, imm as i8);
    } else {
        alu_ri32(e, 6, true, dst, imm);
    }
}
pub fn xor_ri_32(e: &mut TextEmitter, dst: X86Reg, imm: i32) {
    if fits_i8(imm) {
        alu_ri8(e, 6, false, dst, imm as i8);
    } else {
        alu_ri32(e, 6, false, dst, imm);
    }
}

// CMP
pub fn cmp_rr_64(e: &mut TextEmitter, lhs: X86Reg, rhs: X86Reg) {
    alu_rr(e, 0x3B, true, lhs, rhs);
}
pub fn cmp_rr_32(e: &mut TextEmitter, lhs: X86Reg, rhs: X86Reg) {
    alu_rr(e, 0x3B, false, lhs, rhs);
}
pub fn cmp_ri_64(e: &mut TextEmitter, lhs: X86Reg, imm: i32) {
    if fits_i8(imm) {
        alu_ri8(e, 7, true, lhs, imm as i8);
    } else {
        alu_ri32(e, 7, true, lhs, imm);
    }
}
pub fn cmp_ri_32(e: &mut TextEmitter, lhs: X86Reg, imm: i32) {
    if fits_i8(imm) {
        alu_ri8(e, 7, false, lhs, imm as i8);
    } else {
        alu_ri32(e, 7, false, lhs, imm);
    }
}

// TEST
pub fn test_rr_64(e: &mut TextEmitter, lhs: X86Reg, rhs: X86Reg) {
    emit_rex(e, true, rhs, lhs);
    e.emit_u8(0x85);
    emit_modrm_rr(e, rhs, lhs);
}
pub fn test_rr_32(e: &mut TextEmitter, lhs: X86Reg, rhs: X86Reg) {
    emit_rex(e, false, rhs, lhs);
    e.emit_u8(0x85);
    emit_modrm_rr(e, rhs, lhs);
}
pub fn test_ri_64(e: &mut TextEmitter, dst: X86Reg, imm: i32) {
    emit_rex_b(e, true, dst);
    e.emit_u8(0xF7);
    e.emit_u8(modrm(0b11, 0, dst.idx3()));
    emit_i32(e, imm);
}
pub fn test_ri_32(e: &mut TextEmitter, dst: X86Reg, imm: i32) {
    emit_rex_b(e, false, dst);
    e.emit_u8(0xF7);
    e.emit_u8(modrm(0b11, 0, dst.idx3()));
    emit_i32(e, imm);
}

// ═══════════════════════════════════════════════════════════════════════════════
// MOV
// ═══════════════════════════════════════════════════════════════════════════════

/// MOV r64, r64
pub fn mov_rr_64(e: &mut TextEmitter, dst: X86Reg, src: X86Reg) {
    emit_rex(e, true, src, dst);
    e.emit_u8(0x89);
    emit_modrm_rr(e, src, dst);
}

/// MOV r32, r32 (implicitly zero-extends to 64-bit)
pub fn mov_rr_32(e: &mut TextEmitter, dst: X86Reg, src: X86Reg) {
    emit_rex(e, false, src, dst);
    e.emit_u8(0x89);
    emit_modrm_rr(e, src, dst);
}

/// MOV r64, imm64 (REX.W + B8+rd io)
pub fn mov_ri_64(e: &mut TextEmitter, dst: X86Reg, imm: u64) {
    // Optimize: if value fits in 32-bit unsigned, use 32-bit MOV (zero extends)
    if imm <= u32::MAX as u64 {
        mov_ri_32(e, dst, imm as u32);
        return;
    }
    // Optimize: if value fits in sign-extended 32-bit, use MOV r/m64, imm32
    let simm = imm as i64;
    if simm >= i32::MIN as i64 && simm <= i32::MAX as i64 {
        emit_rex_b(e, true, dst);
        e.emit_u8(0xC7);
        e.emit_u8(modrm(0b11, 0, dst.idx3()));
        emit_i32(e, imm as i32);
        return;
    }
    // Full 64-bit immediate
    e.emit_u8(rex(true, false, false, dst.needs_rex_ext()));
    e.emit_u8(0xB8 + dst.idx3());
    e.emit_u64(imm);
}

/// MOVABS r64, imm64 — always emits the full 10-byte form (REX.W + B8+rd + imm64).
/// Use this when the immediate will be patched later and the offset must be predictable.
/// The imm64 starts at `e.len() - 8` after this call.
pub fn movabs_ri_64(e: &mut TextEmitter, dst: X86Reg, imm: u64) {
    e.emit_u8(rex(true, false, false, dst.needs_rex_ext()));
    e.emit_u8(0xB8 + dst.idx3());
    e.emit_u64(imm);
}

/// MOV r32, imm32 (B8+rd id) — zero-extends to 64-bit
pub fn mov_ri_32(e: &mut TextEmitter, dst: X86Reg, imm: u32) {
    if imm == 0 {
        // XOR r32, r32 is smaller and faster
        xor_rr_32(e, dst, dst);
        return;
    }
    if dst.needs_rex_ext() {
        e.emit_u8(0x41); // REX.B
    }
    e.emit_u8(0xB8 + dst.idx3());
    e.emit_bytes(&imm.to_le_bytes());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Shift/rotate
// ═══════════════════════════════════════════════════════════════════════════════

/// Shift r/m by CL. `digit`: 4=SHL, 5=SHR, 7=SAR, 1=ROR/ROL.
fn shift_cl(e: &mut TextEmitter, digit: u8, w: bool, dst: X86Reg) {
    emit_rex_b(e, w, dst);
    e.emit_u8(0xD3);
    e.emit_u8(modrm(0b11, digit, dst.idx3()));
}

/// Shift r/m by imm8.
fn shift_imm(e: &mut TextEmitter, digit: u8, w: bool, dst: X86Reg, imm: u8) {
    emit_rex_b(e, w, dst);
    if imm == 1 {
        e.emit_u8(0xD1);
        e.emit_u8(modrm(0b11, digit, dst.idx3()));
    } else {
        e.emit_u8(0xC1);
        e.emit_u8(modrm(0b11, digit, dst.idx3()));
        e.emit_u8(imm);
    }
}

// SHL
pub fn shl_cl_64(e: &mut TextEmitter, dst: X86Reg) {
    shift_cl(e, 4, true, dst);
}
pub fn shl_cl_32(e: &mut TextEmitter, dst: X86Reg) {
    shift_cl(e, 4, false, dst);
}
pub fn shl_imm_64(e: &mut TextEmitter, dst: X86Reg, imm: u8) {
    shift_imm(e, 4, true, dst, imm);
}
pub fn shl_imm_32(e: &mut TextEmitter, dst: X86Reg, imm: u8) {
    shift_imm(e, 4, false, dst, imm);
}

// SHR
pub fn shr_cl_64(e: &mut TextEmitter, dst: X86Reg) {
    shift_cl(e, 5, true, dst);
}
pub fn shr_cl_32(e: &mut TextEmitter, dst: X86Reg) {
    shift_cl(e, 5, false, dst);
}
pub fn shr_imm_64(e: &mut TextEmitter, dst: X86Reg, imm: u8) {
    shift_imm(e, 5, true, dst, imm);
}
pub fn shr_imm_32(e: &mut TextEmitter, dst: X86Reg, imm: u8) {
    shift_imm(e, 5, false, dst, imm);
}

// SAR
pub fn sar_cl_64(e: &mut TextEmitter, dst: X86Reg) {
    shift_cl(e, 7, true, dst);
}
pub fn sar_cl_32(e: &mut TextEmitter, dst: X86Reg) {
    shift_cl(e, 7, false, dst);
}
pub fn sar_imm_64(e: &mut TextEmitter, dst: X86Reg, imm: u8) {
    shift_imm(e, 7, true, dst, imm);
}
pub fn sar_imm_32(e: &mut TextEmitter, dst: X86Reg, imm: u8) {
    shift_imm(e, 7, false, dst, imm);
}

// ROL
pub fn rol_cl_64(e: &mut TextEmitter, dst: X86Reg) {
    shift_cl(e, 0, true, dst);
}
pub fn rol_cl_32(e: &mut TextEmitter, dst: X86Reg) {
    shift_cl(e, 0, false, dst);
}
pub fn rol_imm_64(e: &mut TextEmitter, dst: X86Reg, imm: u8) {
    shift_imm(e, 0, true, dst, imm);
}
pub fn rol_imm_32(e: &mut TextEmitter, dst: X86Reg, imm: u8) {
    shift_imm(e, 0, false, dst, imm);
}

// ROR
pub fn ror_cl_64(e: &mut TextEmitter, dst: X86Reg) {
    shift_cl(e, 1, true, dst);
}
pub fn ror_cl_32(e: &mut TextEmitter, dst: X86Reg) {
    shift_cl(e, 1, false, dst);
}
pub fn ror_imm_64(e: &mut TextEmitter, dst: X86Reg, imm: u8) {
    shift_imm(e, 1, true, dst, imm);
}
pub fn ror_imm_32(e: &mut TextEmitter, dst: X86Reg, imm: u8) {
    shift_imm(e, 1, false, dst, imm);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Multiply / Divide
// ═══════════════════════════════════════════════════════════════════════════════

/// IMUL r64, r/m64 (two-operand form: dst *= src)
pub fn imul_rr_64(e: &mut TextEmitter, dst: X86Reg, src: X86Reg) {
    emit_rex(e, true, dst, src);
    e.emit_u8(0x0F);
    e.emit_u8(0xAF);
    emit_modrm_rr(e, dst, src);
}

/// IMUL r32, r/m32
pub fn imul_rr_32(e: &mut TextEmitter, dst: X86Reg, src: X86Reg) {
    emit_rex(e, false, dst, src);
    e.emit_u8(0x0F);
    e.emit_u8(0xAF);
    emit_modrm_rr(e, dst, src);
}

/// IDIV r/m64 (signed: RDX:RAX / src -> RAX=quot, RDX=rem)
pub fn idiv_rm_64(e: &mut TextEmitter, src: X86Reg) {
    emit_rex_b(e, true, src);
    e.emit_u8(0xF7);
    e.emit_u8(modrm(0b11, 7, src.idx3()));
}

/// IDIV r/m32
pub fn idiv_rm_32(e: &mut TextEmitter, src: X86Reg) {
    emit_rex_b(e, false, src);
    e.emit_u8(0xF7);
    e.emit_u8(modrm(0b11, 7, src.idx3()));
}

/// DIV r/m64 (unsigned: RDX:RAX / src)
pub fn div_rm_64(e: &mut TextEmitter, src: X86Reg) {
    emit_rex_b(e, true, src);
    e.emit_u8(0xF7);
    e.emit_u8(modrm(0b11, 6, src.idx3()));
}

/// DIV r/m32
pub fn div_rm_32(e: &mut TextEmitter, src: X86Reg) {
    emit_rex_b(e, false, src);
    e.emit_u8(0xF7);
    e.emit_u8(modrm(0b11, 6, src.idx3()));
}

/// CQO: sign-extend RAX into RDX:RAX (64-bit)
pub fn cqo(e: &mut TextEmitter) {
    e.emit_u8(REX_W);
    e.emit_u8(0x99);
}

/// CDQ: sign-extend EAX into EDX:EAX (32-bit)
pub fn cdq(e: &mut TextEmitter) {
    e.emit_u8(0x99);
}

/// NEG r/m64
pub fn neg_64(e: &mut TextEmitter, dst: X86Reg) {
    emit_rex_b(e, true, dst);
    e.emit_u8(0xF7);
    e.emit_u8(modrm(0b11, 3, dst.idx3()));
}

/// NEG r/m32
pub fn neg_32(e: &mut TextEmitter, dst: X86Reg) {
    emit_rex_b(e, false, dst);
    e.emit_u8(0xF7);
    e.emit_u8(modrm(0b11, 3, dst.idx3()));
}

/// NOT r/m64
pub fn not_64(e: &mut TextEmitter, dst: X86Reg) {
    emit_rex_b(e, true, dst);
    e.emit_u8(0xF7);
    e.emit_u8(modrm(0b11, 2, dst.idx3()));
}

/// NOT r/m32
pub fn not_32(e: &mut TextEmitter, dst: X86Reg) {
    emit_rex_b(e, false, dst);
    e.emit_u8(0xF7);
    e.emit_u8(modrm(0b11, 2, dst.idx3()));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Bit manipulation
// ═══════════════════════════════════════════════════════════════════════════════

/// BSR r64, r/m64 (bit scan reverse — find highest set bit)
pub fn bsr_rr_64(e: &mut TextEmitter, dst: X86Reg, src: X86Reg) {
    emit_rex(e, true, dst, src);
    e.emit_u8(0x0F);
    e.emit_u8(0xBD);
    emit_modrm_rr(e, dst, src);
}

/// BSR r32, r/m32
pub fn bsr_rr_32(e: &mut TextEmitter, dst: X86Reg, src: X86Reg) {
    emit_rex(e, false, dst, src);
    e.emit_u8(0x0F);
    e.emit_u8(0xBD);
    emit_modrm_rr(e, dst, src);
}

/// BSF r64, r/m64 (bit scan forward — find lowest set bit)
pub fn bsf_rr_64(e: &mut TextEmitter, dst: X86Reg, src: X86Reg) {
    emit_rex(e, true, dst, src);
    e.emit_u8(0x0F);
    e.emit_u8(0xBC);
    emit_modrm_rr(e, dst, src);
}

/// BSF r32, r/m32
pub fn bsf_rr_32(e: &mut TextEmitter, dst: X86Reg, src: X86Reg) {
    emit_rex(e, false, dst, src);
    e.emit_u8(0x0F);
    e.emit_u8(0xBC);
    emit_modrm_rr(e, dst, src);
}

/// LZCNT r64, r/m64 (leading zero count — needs LZCNT support, F3 0F BD)
pub fn lzcnt_rr_64(e: &mut TextEmitter, dst: X86Reg, src: X86Reg) {
    e.emit_u8(0xF3);
    emit_rex(e, true, dst, src);
    e.emit_u8(0x0F);
    e.emit_u8(0xBD);
    emit_modrm_rr(e, dst, src);
}

/// LZCNT r32, r/m32
pub fn lzcnt_rr_32(e: &mut TextEmitter, dst: X86Reg, src: X86Reg) {
    e.emit_u8(0xF3);
    emit_rex(e, false, dst, src);
    e.emit_u8(0x0F);
    e.emit_u8(0xBD);
    emit_modrm_rr(e, dst, src);
}

/// TZCNT r64, r/m64 (trailing zero count — F3 0F BC)
pub fn tzcnt_rr_64(e: &mut TextEmitter, dst: X86Reg, src: X86Reg) {
    e.emit_u8(0xF3);
    emit_rex(e, true, dst, src);
    e.emit_u8(0x0F);
    e.emit_u8(0xBC);
    emit_modrm_rr(e, dst, src);
}

/// TZCNT r32, r/m32
pub fn tzcnt_rr_32(e: &mut TextEmitter, dst: X86Reg, src: X86Reg) {
    e.emit_u8(0xF3);
    emit_rex(e, false, dst, src);
    e.emit_u8(0x0F);
    e.emit_u8(0xBC);
    emit_modrm_rr(e, dst, src);
}

/// POPCNT r64, r/m64 (F3 0F B8)
pub fn popcnt_rr_64(e: &mut TextEmitter, dst: X86Reg, src: X86Reg) {
    e.emit_u8(0xF3);
    emit_rex(e, true, dst, src);
    e.emit_u8(0x0F);
    e.emit_u8(0xB8);
    emit_modrm_rr(e, dst, src);
}

/// POPCNT r32, r/m32
pub fn popcnt_rr_32(e: &mut TextEmitter, dst: X86Reg, src: X86Reg) {
    e.emit_u8(0xF3);
    emit_rex(e, false, dst, src);
    e.emit_u8(0x0F);
    e.emit_u8(0xB8);
    emit_modrm_rr(e, dst, src);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Sign/zero extension
// ═══════════════════════════════════════════════════════════════════════════════

/// MOVSX r64, r/m8 (sign-extend byte to 64-bit): REX.W 0F BE
pub fn movsx_r64_r8(e: &mut TextEmitter, dst: X86Reg, src: X86Reg) {
    emit_rex(e, true, dst, src);
    e.emit_u8(0x0F);
    e.emit_u8(0xBE);
    emit_modrm_rr(e, dst, src);
}

/// MOVSX r64, r/m16 (sign-extend word to 64-bit): REX.W 0F BF
pub fn movsx_r64_r16(e: &mut TextEmitter, dst: X86Reg, src: X86Reg) {
    emit_rex(e, true, dst, src);
    e.emit_u8(0x0F);
    e.emit_u8(0xBF);
    emit_modrm_rr(e, dst, src);
}

/// MOVSXD r64, r/m32 (sign-extend dword to 64-bit): REX.W 63
pub fn movsxd_r64_r32(e: &mut TextEmitter, dst: X86Reg, src: X86Reg) {
    emit_rex(e, true, dst, src);
    e.emit_u8(0x63);
    emit_modrm_rr(e, dst, src);
}

/// MOVSX r32, r/m8 (sign-extend byte to 32-bit): 0F BE
pub fn movsx_r32_r8(e: &mut TextEmitter, dst: X86Reg, src: X86Reg) {
    // Must emit REX for src idx 4-7 to access low byte (SPL/BPL/SIL/DIL).
    let need_rex = dst.needs_rex_ext() || src.needs_rex_ext() || (src.idx() >= 4 && src.idx() <= 7);
    if need_rex {
        e.emit_u8(rex(false, dst.needs_rex_ext(), false, src.needs_rex_ext()));
    }
    e.emit_u8(0x0F);
    e.emit_u8(0xBE);
    emit_modrm_rr(e, dst, src);
}

/// MOVSX r32, r/m16: 0F BF
pub fn movsx_r32_r16(e: &mut TextEmitter, dst: X86Reg, src: X86Reg) {
    emit_rex(e, false, dst, src);
    e.emit_u8(0x0F);
    e.emit_u8(0xBF);
    emit_modrm_rr(e, dst, src);
}

/// MOVZX r64, r/m8 (zero-extend byte to 64-bit): REX.W 0F B6
pub fn movzx_r64_r8(e: &mut TextEmitter, dst: X86Reg, src: X86Reg) {
    emit_rex(e, true, dst, src);
    e.emit_u8(0x0F);
    e.emit_u8(0xB6);
    emit_modrm_rr(e, dst, src);
}

/// MOVZX r32, r/m8 (zero-extend byte to 32-bit, implicitly to 64): 0F B6
pub fn movzx_r32_r8(e: &mut TextEmitter, dst: X86Reg, src: X86Reg) {
    // Must emit REX when src idx is 4-7 (RSP/RBP/RSI/RDI) to access the
    // low byte (SPL/BPL/SIL/DIL) instead of the legacy high byte (AH/CH/DH/BH).
    let need_rex = dst.needs_rex_ext() || src.needs_rex_ext() || (src.idx() >= 4 && src.idx() <= 7);
    if need_rex {
        e.emit_u8(rex(false, dst.needs_rex_ext(), false, src.needs_rex_ext()));
    }
    e.emit_u8(0x0F);
    e.emit_u8(0xB6);
    emit_modrm_rr(e, dst, src);
}

/// MOVZX r32, r/m16: 0F B7
pub fn movzx_r32_r16(e: &mut TextEmitter, dst: X86Reg, src: X86Reg) {
    emit_rex(e, false, dst, src);
    e.emit_u8(0x0F);
    e.emit_u8(0xB7);
    emit_modrm_rr(e, dst, src);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SETcc / CMOVcc
// ═══════════════════════════════════════════════════════════════════════════════

/// SETcc r/m8: 0F 90+cc
pub fn setcc(e: &mut TextEmitter, cc: Cc, dst: X86Reg) {
    // Need REX prefix for R8-R15 and also to ensure we access the low byte
    // (without REX, regs 4-7 map to AH/CH/DH/BH, not SPL/BPL/SIL/DIL)
    let need_rex = dst.needs_rex_ext() || (dst.idx() >= 4 && dst.idx() <= 7);
    if need_rex {
        e.emit_u8(rex(false, false, false, dst.needs_rex_ext()));
    }
    e.emit_u8(0x0F);
    e.emit_u8(0x90 + cc as u8);
    e.emit_u8(modrm(0b11, 0, dst.idx3()));
}

/// CMOVcc r64, r/m64: REX.W 0F 40+cc
pub fn cmovcc_rr_64(e: &mut TextEmitter, cc: Cc, dst: X86Reg, src: X86Reg) {
    emit_rex(e, true, dst, src);
    e.emit_u8(0x0F);
    e.emit_u8(0x40 + cc as u8);
    emit_modrm_rr(e, dst, src);
}

/// CMOVcc r32, r/m32
pub fn cmovcc_rr_32(e: &mut TextEmitter, cc: Cc, dst: X86Reg, src: X86Reg) {
    emit_rex(e, false, dst, src);
    e.emit_u8(0x0F);
    e.emit_u8(0x40 + cc as u8);
    emit_modrm_rr(e, dst, src);
}

// ═══════════════════════════════════════════════════════════════════════════════
// LEA
// ═══════════════════════════════════════════════════════════════════════════════

/// LEA r64, [base + disp]
pub fn lea_64(e: &mut TextEmitter, dst: X86Reg, base: X86Reg, disp: i32) {
    emit_rex(e, true, dst, base);
    e.emit_u8(0x8D);
    emit_modrm_mem(e, dst, base, disp);
}

/// LEA r64, [base + index*scale + disp]
pub fn lea_sib_64(
    e: &mut TextEmitter,
    dst: X86Reg,
    base: X86Reg,
    index: X86Reg,
    scale: u8,
    disp: i32,
) {
    emit_rex_sib(e, true, dst, index, base);
    e.emit_u8(0x8D);
    emit_modrm_sib(e, dst, base, index, scale, disp);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Memory loads/stores (GP)
// ═══════════════════════════════════════════════════════════════════════════════

/// MOV r64, [base + disp]
pub fn load_64(e: &mut TextEmitter, dst: X86Reg, base: X86Reg, disp: i32) {
    emit_rex(e, true, dst, base);
    e.emit_u8(0x8B);
    emit_modrm_mem(e, dst, base, disp);
}

/// MOV r32, [base + disp] (zero extends)
pub fn load_32(e: &mut TextEmitter, dst: X86Reg, base: X86Reg, disp: i32) {
    emit_rex(e, false, dst, base);
    e.emit_u8(0x8B);
    emit_modrm_mem(e, dst, base, disp);
}

/// MOVZX r32, byte [base + disp]
pub fn load_u8(e: &mut TextEmitter, dst: X86Reg, base: X86Reg, disp: i32) {
    emit_rex(e, false, dst, base);
    e.emit_u8(0x0F);
    e.emit_u8(0xB6);
    emit_modrm_mem(e, dst, base, disp);
}

/// MOVZX r32, word [base + disp]
pub fn load_u16(e: &mut TextEmitter, dst: X86Reg, base: X86Reg, disp: i32) {
    emit_rex(e, false, dst, base);
    e.emit_u8(0x0F);
    e.emit_u8(0xB7);
    emit_modrm_mem(e, dst, base, disp);
}

/// MOVSX r64, byte [base + disp]
pub fn load_s8_64(e: &mut TextEmitter, dst: X86Reg, base: X86Reg, disp: i32) {
    emit_rex(e, true, dst, base);
    e.emit_u8(0x0F);
    e.emit_u8(0xBE);
    emit_modrm_mem(e, dst, base, disp);
}

/// MOVSX r64, word [base + disp]
pub fn load_s16_64(e: &mut TextEmitter, dst: X86Reg, base: X86Reg, disp: i32) {
    emit_rex(e, true, dst, base);
    e.emit_u8(0x0F);
    e.emit_u8(0xBF);
    emit_modrm_mem(e, dst, base, disp);
}

/// MOVSXD r64, dword [base + disp]
pub fn load_s32_64(e: &mut TextEmitter, dst: X86Reg, base: X86Reg, disp: i32) {
    emit_rex(e, true, dst, base);
    e.emit_u8(0x63);
    emit_modrm_mem(e, dst, base, disp);
}

/// MOV [base + disp], r64
pub fn store_64(e: &mut TextEmitter, base: X86Reg, disp: i32, src: X86Reg) {
    emit_rex(e, true, src, base);
    e.emit_u8(0x89);
    emit_modrm_mem(e, src, base, disp);
}

/// MOV [base + disp], r32
pub fn store_32(e: &mut TextEmitter, base: X86Reg, disp: i32, src: X86Reg) {
    emit_rex(e, false, src, base);
    e.emit_u8(0x89);
    emit_modrm_mem(e, src, base, disp);
}

/// MOV byte [base + disp], r8
pub fn store_8(e: &mut TextEmitter, base: X86Reg, disp: i32, src: X86Reg) {
    // Need REX for extended regs or to ensure low-byte access for RSP/RBP/RSI/RDI
    let need_rex =
        src.needs_rex_ext() || base.needs_rex_ext() || (src.idx() >= 4 && src.idx() <= 7);
    if need_rex {
        e.emit_u8(rex(false, src.needs_rex_ext(), false, base.needs_rex_ext()));
    }
    e.emit_u8(0x88);
    emit_modrm_mem(e, src, base, disp);
}

/// MOV word [base + disp], r16 (requires 0x66 prefix)
pub fn store_16(e: &mut TextEmitter, base: X86Reg, disp: i32, src: X86Reg) {
    e.emit_u8(0x66); // operand-size override
    emit_rex(e, false, src, base);
    e.emit_u8(0x89);
    emit_modrm_mem(e, src, base, disp);
}

/// MOV qword [base + disp], imm32 (sign-extended to 64-bit)
pub fn store_imm32_64(e: &mut TextEmitter, base: X86Reg, disp: i32, imm: i32) {
    emit_rex_b(e, true, base);
    e.emit_u8(0xC7);
    emit_modrm_mem(e, X86Reg::RAX, base, disp); // /0
    emit_i32(e, imm);
}

// --- Indexed memory (base + index*scale + disp) ---

/// MOV r64, [base + index*scale + disp]
pub fn load_sib_64(
    e: &mut TextEmitter,
    dst: X86Reg,
    base: X86Reg,
    index: X86Reg,
    scale: u8,
    disp: i32,
) {
    emit_rex_sib(e, true, dst, index, base);
    e.emit_u8(0x8B);
    emit_modrm_sib(e, dst, base, index, scale, disp);
}

/// MOV r32, [base + index*scale + disp]
pub fn load_sib_32(
    e: &mut TextEmitter,
    dst: X86Reg,
    base: X86Reg,
    index: X86Reg,
    scale: u8,
    disp: i32,
) {
    emit_rex_sib(e, false, dst, index, base);
    e.emit_u8(0x8B);
    emit_modrm_sib(e, dst, base, index, scale, disp);
}

/// MOVZX r32, byte [base + index*scale + disp]
pub fn load_sib_u8(
    e: &mut TextEmitter,
    dst: X86Reg,
    base: X86Reg,
    index: X86Reg,
    scale: u8,
    disp: i32,
) {
    emit_rex_sib(e, false, dst, index, base);
    e.emit_u8(0x0F);
    e.emit_u8(0xB6);
    emit_modrm_sib(e, dst, base, index, scale, disp);
}

/// MOVZX r32, word [base + index*scale + disp]
pub fn load_sib_u16(
    e: &mut TextEmitter,
    dst: X86Reg,
    base: X86Reg,
    index: X86Reg,
    scale: u8,
    disp: i32,
) {
    emit_rex_sib(e, false, dst, index, base);
    e.emit_u8(0x0F);
    e.emit_u8(0xB7);
    emit_modrm_sib(e, dst, base, index, scale, disp);
}

/// MOVSX r64, byte [base + index*scale + disp]
pub fn load_sib_s8_64(
    e: &mut TextEmitter,
    dst: X86Reg,
    base: X86Reg,
    index: X86Reg,
    scale: u8,
    disp: i32,
) {
    emit_rex_sib(e, true, dst, index, base);
    e.emit_u8(0x0F);
    e.emit_u8(0xBE);
    emit_modrm_sib(e, dst, base, index, scale, disp);
}

/// MOVSX r64, word [base + index*scale + disp]
pub fn load_sib_s16_64(
    e: &mut TextEmitter,
    dst: X86Reg,
    base: X86Reg,
    index: X86Reg,
    scale: u8,
    disp: i32,
) {
    emit_rex_sib(e, true, dst, index, base);
    e.emit_u8(0x0F);
    e.emit_u8(0xBF);
    emit_modrm_sib(e, dst, base, index, scale, disp);
}

/// MOVSXD r64, dword [base + index*scale + disp]
pub fn load_sib_s32_64(
    e: &mut TextEmitter,
    dst: X86Reg,
    base: X86Reg,
    index: X86Reg,
    scale: u8,
    disp: i32,
) {
    emit_rex_sib(e, true, dst, index, base);
    e.emit_u8(0x63);
    emit_modrm_sib(e, dst, base, index, scale, disp);
}

/// MOV [base + index*scale + disp], r64
pub fn store_sib_64(
    e: &mut TextEmitter,
    base: X86Reg,
    index: X86Reg,
    scale: u8,
    disp: i32,
    src: X86Reg,
) {
    emit_rex_sib(e, true, src, index, base);
    e.emit_u8(0x89);
    emit_modrm_sib(e, src, base, index, scale, disp);
}

/// MOV [base + index*scale + disp], r32
pub fn store_sib_32(
    e: &mut TextEmitter,
    base: X86Reg,
    index: X86Reg,
    scale: u8,
    disp: i32,
    src: X86Reg,
) {
    emit_rex_sib(e, false, src, index, base);
    e.emit_u8(0x89);
    emit_modrm_sib(e, src, base, index, scale, disp);
}

/// MOV byte [base + index*scale + disp], r8
pub fn store_sib_8(
    e: &mut TextEmitter,
    base: X86Reg,
    index: X86Reg,
    scale: u8,
    disp: i32,
    src: X86Reg,
) {
    let need_rex = src.needs_rex_ext()
        || index.needs_rex_ext()
        || base.needs_rex_ext()
        || (src.idx() >= 4 && src.idx() <= 7);
    if need_rex {
        e.emit_u8(rex(
            false,
            src.needs_rex_ext(),
            index.needs_rex_ext(),
            base.needs_rex_ext(),
        ));
    }
    e.emit_u8(0x88);
    emit_modrm_sib(e, src, base, index, scale, disp);
}

/// MOV word [base + index*scale + disp], r16
pub fn store_sib_16(
    e: &mut TextEmitter,
    base: X86Reg,
    index: X86Reg,
    scale: u8,
    disp: i32,
    src: X86Reg,
) {
    e.emit_u8(0x66);
    emit_rex_sib(e, false, src, index, base);
    e.emit_u8(0x89);
    emit_modrm_sib(e, src, base, index, scale, disp);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Push / Pop
// ═══════════════════════════════════════════════════════════════════════════════

pub fn push(e: &mut TextEmitter, reg: X86Reg) {
    if reg.needs_rex_ext() {
        e.emit_u8(0x41);
    }
    e.emit_u8(0x50 + reg.idx3());
}

pub fn pop(e: &mut TextEmitter, reg: X86Reg) {
    if reg.needs_rex_ext() {
        e.emit_u8(0x41);
    }
    e.emit_u8(0x58 + reg.idx3());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Branches / Calls / Return
// ═══════════════════════════════════════════════════════════════════════════════

/// JMP rel32 (E9 cd). Returns offset of the rel32 field for patching.
pub fn jmp_rel32(e: &mut TextEmitter) -> usize {
    e.emit_u8(0xE9);
    let patch = e.len();
    emit_i32(e, 0); // placeholder
    patch
}

/// Jcc rel32 (0F 80+cc cd). Returns offset of the rel32 field.
pub fn jcc_rel32(e: &mut TextEmitter, cc: Cc) -> usize {
    e.emit_u8(0x0F);
    e.emit_u8(0x80 + cc as u8);
    let patch = e.len();
    emit_i32(e, 0); // placeholder
    patch
}

/// CALL rel32 (E8 cd). Returns offset of the rel32 field.
pub fn call_rel32(e: &mut TextEmitter) -> usize {
    e.emit_u8(0xE8);
    let patch = e.len();
    emit_i32(e, 0);
    patch
}

/// CALL r/m64 (FF /2) — indirect call through register.
pub fn call_reg(e: &mut TextEmitter, target: X86Reg) {
    if target.needs_rex_ext() {
        e.emit_u8(0x41);
    }
    e.emit_u8(0xFF);
    e.emit_u8(modrm(0b11, 2, target.idx3()));
}

/// JMP r/m64 (FF /4) — indirect jump through register.
pub fn jmp_reg(e: &mut TextEmitter, target: X86Reg) {
    if target.needs_rex_ext() {
        e.emit_u8(0x41);
    }
    e.emit_u8(0xFF);
    e.emit_u8(modrm(0b11, 4, target.idx3()));
}

/// RET (C3)
pub fn ret(e: &mut TextEmitter) {
    e.emit_u8(0xC3);
}

/// Patch a rel32 fixup. `fixup_offset` is where the 4-byte displacement lives.
/// `target_offset` is the absolute byte offset of the target in the text.
pub fn patch_rel32(e: &mut TextEmitter, fixup_offset: usize, target_offset: usize) {
    // rel32 is relative to the end of the instruction (fixup_offset + 4)
    let rel = target_offset as i64 - (fixup_offset as i64 + 4);
    debug_assert!(
        rel >= i32::MIN as i64 && rel <= i32::MAX as i64,
        "rel32 overflow: offset={}, target={}, rel={}",
        fixup_offset,
        target_offset,
        rel,
    );
    e.patch_i32(fixup_offset, rel as i32);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SSE/SSE2 floating-point
// ═══════════════════════════════════════════════════════════════════════════════

/// XMM register number (0-15).
pub type Xmm = u8;

/// Helper: emit REX for XMM reg-reg (using XMM indices as if they were GP reg nums).
fn emit_xmm_rex(e: &mut TextEmitter, w: bool, reg: Xmm, rm: Xmm) {
    let r = reg >= 8;
    let b = rm >= 8;
    if w || r || b {
        e.emit_u8(rex(w, r, false, b));
    }
}

fn emit_xmm_modrm_rr(e: &mut TextEmitter, reg: Xmm, rm: Xmm) {
    e.emit_u8(modrm(0b11, reg & 7, rm & 7));
}

/// Emit SSE/SSE2 prefix + REX + opcode + ModR/M for xmm,xmm.
fn sse_rr(e: &mut TextEmitter, prefix: u8, op1: u8, op2: u8, dst: Xmm, src: Xmm) {
    if prefix != 0 {
        e.emit_u8(prefix);
    }
    emit_xmm_rex(e, false, dst, src);
    e.emit_u8(op1);
    e.emit_u8(op2);
    emit_xmm_modrm_rr(e, dst, src);
}

/// Helper for XMM memory loads/stores with [base + disp].
fn emit_xmm_modrm_mem(
    e: &mut TextEmitter,
    prefix: u8,
    op1: u8,
    op2: u8,
    xmm: Xmm,
    base: X86Reg,
    disp: i32,
) {
    if prefix != 0 {
        e.emit_u8(prefix);
    }
    let r = xmm >= 8;
    let b = base.needs_rex_ext();
    if r || b {
        e.emit_u8(rex(false, r, false, b));
    }
    e.emit_u8(op1);
    e.emit_u8(op2);
    // Reuse GP modrm_mem logic but with xmm as "reg" field
    let reg3 = xmm & 7;
    let base3 = base.idx3();
    if base3 == 4 {
        if disp == 0 {
            e.emit_u8(modrm(0b00, reg3, 0b100));
            e.emit_u8(sib(0b00, 0b100, base3));
        } else if fits_i8(disp) {
            e.emit_u8(modrm(0b01, reg3, 0b100));
            e.emit_u8(sib(0b00, 0b100, base3));
            e.emit_u8(disp as u8);
        } else {
            e.emit_u8(modrm(0b10, reg3, 0b100));
            e.emit_u8(sib(0b00, 0b100, base3));
            emit_i32(e, disp);
        }
    } else if base3 == 5 && disp == 0 {
        e.emit_u8(modrm(0b01, reg3, base3));
        e.emit_u8(0);
    } else if disp == 0 {
        e.emit_u8(modrm(0b00, reg3, base3));
    } else if fits_i8(disp) {
        e.emit_u8(modrm(0b01, reg3, base3));
        e.emit_u8(disp as u8);
    } else {
        e.emit_u8(modrm(0b10, reg3, base3));
        emit_i32(e, disp);
    }
}

/// Helper for XMM SIB memory access.
fn emit_xmm_modrm_sib(
    e: &mut TextEmitter,
    prefix: u8,
    op1: u8,
    op2: u8,
    xmm: Xmm,
    base: X86Reg,
    index: X86Reg,
    scale: u8,
    disp: i32,
) {
    if prefix != 0 {
        e.emit_u8(prefix);
    }
    let r = xmm >= 8;
    let x = index.needs_rex_ext();
    let b = base.needs_rex_ext();
    if r || x || b {
        e.emit_u8(rex(false, r, x, b));
    }
    e.emit_u8(op1);
    e.emit_u8(op2);
    let reg3 = xmm & 7;
    let base3 = base.idx3();
    let index3 = index.idx3();
    let scale_bits = match scale {
        1 => 0b00,
        2 => 0b01,
        4 => 0b10,
        8 => 0b11,
        _ => panic!("invalid SIB scale"),
    };
    if disp == 0 && base3 != 5 {
        e.emit_u8(modrm(0b00, reg3, 0b100));
        e.emit_u8(sib(scale_bits, index3, base3));
    } else if fits_i8(disp) {
        e.emit_u8(modrm(0b01, reg3, 0b100));
        e.emit_u8(sib(scale_bits, index3, base3));
        e.emit_u8(disp as u8);
    } else {
        e.emit_u8(modrm(0b10, reg3, 0b100));
        e.emit_u8(sib(scale_bits, index3, base3));
        emit_i32(e, disp);
    }
}

/// Helper for XMM RIP-relative addressing: prefix + REX + opcode + ModR/M(RIP).
/// Returns the offset of the disp32 field for patching.
fn emit_xmm_modrm_rip(
    e: &mut TextEmitter,
    prefix: u8,
    op1: u8,
    op2: u8,
    xmm: Xmm,
    disp32: i32,
) -> usize {
    if prefix != 0 {
        e.emit_u8(prefix);
    }
    let r = xmm >= 8;
    if r {
        e.emit_u8(rex(false, r, false, false));
    }
    e.emit_u8(op1);
    e.emit_u8(op2);
    e.emit_u8(modrm(0b00, xmm & 7, 0b101));
    let patch = e.len();
    emit_i32(e, disp32);
    patch
}

// --- F64 (double) arithmetic: prefix 66, 0F xx ---

pub fn movsd_rr(e: &mut TextEmitter, dst: Xmm, src: Xmm) {
    // F2 0F 10 (MOVSD xmm1, xmm2)
    sse_rr(e, 0xF2, 0x0F, 0x10, dst, src);
}

pub fn addsd(e: &mut TextEmitter, dst: Xmm, src: Xmm) {
    sse_rr(e, 0xF2, 0x0F, 0x58, dst, src);
}
pub fn subsd(e: &mut TextEmitter, dst: Xmm, src: Xmm) {
    sse_rr(e, 0xF2, 0x0F, 0x5C, dst, src);
}
pub fn mulsd(e: &mut TextEmitter, dst: Xmm, src: Xmm) {
    sse_rr(e, 0xF2, 0x0F, 0x59, dst, src);
}
pub fn divsd(e: &mut TextEmitter, dst: Xmm, src: Xmm) {
    sse_rr(e, 0xF2, 0x0F, 0x5E, dst, src);
}
pub fn sqrtsd(e: &mut TextEmitter, dst: Xmm, src: Xmm) {
    sse_rr(e, 0xF2, 0x0F, 0x51, dst, src);
}
pub fn minsd(e: &mut TextEmitter, dst: Xmm, src: Xmm) {
    sse_rr(e, 0xF2, 0x0F, 0x5D, dst, src);
}
pub fn maxsd(e: &mut TextEmitter, dst: Xmm, src: Xmm) {
    sse_rr(e, 0xF2, 0x0F, 0x5F, dst, src);
}
pub fn ucomisd(e: &mut TextEmitter, lhs: Xmm, rhs: Xmm) {
    sse_rr(e, 0x66, 0x0F, 0x2E, lhs, rhs);
}

/// XORPD xmm, xmm (zero a register): 66 0F 57
pub fn xorpd(e: &mut TextEmitter, dst: Xmm, src: Xmm) {
    sse_rr(e, 0x66, 0x0F, 0x57, dst, src);
}

/// ANDPD xmm, xmm: 66 0F 54
pub fn andpd(e: &mut TextEmitter, dst: Xmm, src: Xmm) {
    sse_rr(e, 0x66, 0x0F, 0x54, dst, src);
}

/// ORPD xmm, xmm: 66 0F 56
pub fn orpd(e: &mut TextEmitter, dst: Xmm, src: Xmm) {
    sse_rr(e, 0x66, 0x0F, 0x56, dst, src);
}

// --- F32 (single) arithmetic: prefix F3, 0F xx ---

pub fn movss_rr(e: &mut TextEmitter, dst: Xmm, src: Xmm) {
    sse_rr(e, 0xF3, 0x0F, 0x10, dst, src);
}

pub fn addss(e: &mut TextEmitter, dst: Xmm, src: Xmm) {
    sse_rr(e, 0xF3, 0x0F, 0x58, dst, src);
}
pub fn subss(e: &mut TextEmitter, dst: Xmm, src: Xmm) {
    sse_rr(e, 0xF3, 0x0F, 0x5C, dst, src);
}
pub fn mulss(e: &mut TextEmitter, dst: Xmm, src: Xmm) {
    sse_rr(e, 0xF3, 0x0F, 0x59, dst, src);
}
pub fn divss(e: &mut TextEmitter, dst: Xmm, src: Xmm) {
    sse_rr(e, 0xF3, 0x0F, 0x5E, dst, src);
}
pub fn sqrtss(e: &mut TextEmitter, dst: Xmm, src: Xmm) {
    sse_rr(e, 0xF3, 0x0F, 0x51, dst, src);
}
pub fn minss(e: &mut TextEmitter, dst: Xmm, src: Xmm) {
    sse_rr(e, 0xF3, 0x0F, 0x5D, dst, src);
}
pub fn maxss(e: &mut TextEmitter, dst: Xmm, src: Xmm) {
    sse_rr(e, 0xF3, 0x0F, 0x5F, dst, src);
}
pub fn ucomiss(e: &mut TextEmitter, lhs: Xmm, rhs: Xmm) {
    sse_rr(e, 0, 0x0F, 0x2E, lhs, rhs);
}

/// XORPS xmm, xmm (zero a register): 0F 57
pub fn xorps(e: &mut TextEmitter, dst: Xmm, src: Xmm) {
    sse_rr(e, 0, 0x0F, 0x57, dst, src);
}

/// ANDPS xmm, xmm: 0F 54
pub fn andps(e: &mut TextEmitter, dst: Xmm, src: Xmm) {
    sse_rr(e, 0, 0x0F, 0x54, dst, src);
}

/// ORPS xmm, xmm: 0F 56
pub fn orps(e: &mut TextEmitter, dst: Xmm, src: Xmm) {
    sse_rr(e, 0, 0x0F, 0x56, dst, src);
}

// --- FP conversions ---

/// CVTSD2SS xmm, xmm (F64 -> F32): F2 0F 5A
pub fn cvtsd2ss(e: &mut TextEmitter, dst: Xmm, src: Xmm) {
    sse_rr(e, 0xF2, 0x0F, 0x5A, dst, src);
}

/// CVTSS2SD xmm, xmm (F32 -> F64): F3 0F 5A
pub fn cvtss2sd(e: &mut TextEmitter, dst: Xmm, src: Xmm) {
    sse_rr(e, 0xF3, 0x0F, 0x5A, dst, src);
}

/// CVTSI2SD xmm, r32 (i32 -> f64): F2 0F 2A
pub fn cvtsi2sd_r32(e: &mut TextEmitter, dst: Xmm, src: X86Reg) {
    e.emit_u8(0xF2);
    emit_xmm_rex(e, false, dst, src.idx());
    e.emit_u8(0x0F);
    e.emit_u8(0x2A);
    emit_xmm_modrm_rr(e, dst, src.idx());
}

/// CVTSI2SD xmm, r64 (i64 -> f64): F2 REX.W 0F 2A
pub fn cvtsi2sd_r64(e: &mut TextEmitter, dst: Xmm, src: X86Reg) {
    e.emit_u8(0xF2);
    let r = dst >= 8;
    let b = src.needs_rex_ext();
    e.emit_u8(rex(true, r, false, b));
    e.emit_u8(0x0F);
    e.emit_u8(0x2A);
    emit_xmm_modrm_rr(e, dst, src.idx());
}

/// CVTSI2SS xmm, r32 (i32 -> f32): F3 0F 2A
pub fn cvtsi2ss_r32(e: &mut TextEmitter, dst: Xmm, src: X86Reg) {
    e.emit_u8(0xF3);
    emit_xmm_rex(e, false, dst, src.idx());
    e.emit_u8(0x0F);
    e.emit_u8(0x2A);
    emit_xmm_modrm_rr(e, dst, src.idx());
}

/// CVTSI2SS xmm, r64 (i64 -> f32): F3 REX.W 0F 2A
pub fn cvtsi2ss_r64(e: &mut TextEmitter, dst: Xmm, src: X86Reg) {
    e.emit_u8(0xF3);
    let r = dst >= 8;
    let b = src.needs_rex_ext();
    e.emit_u8(rex(true, r, false, b));
    e.emit_u8(0x0F);
    e.emit_u8(0x2A);
    emit_xmm_modrm_rr(e, dst, src.idx());
}

/// CVTTSD2SI r32, xmm (f64 -> i32, truncate): F2 0F 2C
pub fn cvttsd2si_r32(e: &mut TextEmitter, dst: X86Reg, src: Xmm) {
    e.emit_u8(0xF2);
    emit_xmm_rex(e, false, dst.idx(), src);
    e.emit_u8(0x0F);
    e.emit_u8(0x2C);
    emit_xmm_modrm_rr(e, dst.idx(), src);
}

/// CVTTSD2SI r64, xmm (f64 -> i64, truncate): F2 REX.W 0F 2C
pub fn cvttsd2si_r64(e: &mut TextEmitter, dst: X86Reg, src: Xmm) {
    e.emit_u8(0xF2);
    let r = dst.needs_rex_ext();
    let b = src >= 8;
    e.emit_u8(rex(true, r, false, b));
    e.emit_u8(0x0F);
    e.emit_u8(0x2C);
    emit_xmm_modrm_rr(e, dst.idx(), src);
}

/// CVTTSS2SI r32, xmm (f32 -> i32, truncate): F3 0F 2C
pub fn cvttss2si_r32(e: &mut TextEmitter, dst: X86Reg, src: Xmm) {
    e.emit_u8(0xF3);
    emit_xmm_rex(e, false, dst.idx(), src);
    e.emit_u8(0x0F);
    e.emit_u8(0x2C);
    emit_xmm_modrm_rr(e, dst.idx(), src);
}

/// CVTTSS2SI r64, xmm (f32 -> i64, truncate): F3 REX.W 0F 2C
pub fn cvttss2si_r64(e: &mut TextEmitter, dst: X86Reg, src: Xmm) {
    e.emit_u8(0xF3);
    let r = dst.needs_rex_ext();
    let b = src >= 8;
    e.emit_u8(rex(true, r, false, b));
    e.emit_u8(0x0F);
    e.emit_u8(0x2C);
    emit_xmm_modrm_rr(e, dst.idx(), src);
}

// --- GP <-> XMM moves ---

/// MOVD xmm, r32: 66 0F 6E
pub fn movd_xmm_r32(e: &mut TextEmitter, dst: Xmm, src: X86Reg) {
    e.emit_u8(0x66);
    emit_xmm_rex(e, false, dst, src.idx());
    e.emit_u8(0x0F);
    e.emit_u8(0x6E);
    emit_xmm_modrm_rr(e, dst, src.idx());
}

/// MOVD r32, xmm: 66 0F 7E
pub fn movd_r32_xmm(e: &mut TextEmitter, dst: X86Reg, src: Xmm) {
    e.emit_u8(0x66);
    emit_xmm_rex(e, false, src, dst.idx());
    e.emit_u8(0x0F);
    e.emit_u8(0x7E);
    emit_xmm_modrm_rr(e, src, dst.idx());
}

/// MOVQ xmm, r64: 66 REX.W 0F 6E
pub fn movq_xmm_r64(e: &mut TextEmitter, dst: Xmm, src: X86Reg) {
    e.emit_u8(0x66);
    let r = dst >= 8;
    let b = src.needs_rex_ext();
    e.emit_u8(rex(true, r, false, b));
    e.emit_u8(0x0F);
    e.emit_u8(0x6E);
    emit_xmm_modrm_rr(e, dst, src.idx());
}

/// MOVQ r64, xmm: 66 REX.W 0F 7E
pub fn movq_r64_xmm(e: &mut TextEmitter, dst: X86Reg, src: Xmm) {
    e.emit_u8(0x66);
    let r = src >= 8;
    let b = dst.needs_rex_ext();
    e.emit_u8(rex(true, r, false, b));
    e.emit_u8(0x0F);
    e.emit_u8(0x7E);
    emit_xmm_modrm_rr(e, src, dst.idx());
}

// --- FP memory loads/stores ---

/// MOVSD xmm, [base + disp]: F2 0F 10
pub fn movsd_load(e: &mut TextEmitter, dst: Xmm, base: X86Reg, disp: i32) {
    emit_xmm_modrm_mem(e, 0xF2, 0x0F, 0x10, dst, base, disp);
}

/// MOVSD [base + disp], xmm: F2 0F 11
pub fn movsd_store(e: &mut TextEmitter, base: X86Reg, disp: i32, src: Xmm) {
    emit_xmm_modrm_mem(e, 0xF2, 0x0F, 0x11, src, base, disp);
}

/// MOVSS xmm, [base + disp]: F3 0F 10
pub fn movss_load(e: &mut TextEmitter, dst: Xmm, base: X86Reg, disp: i32) {
    emit_xmm_modrm_mem(e, 0xF3, 0x0F, 0x10, dst, base, disp);
}

/// MOVSS [base + disp], xmm: F3 0F 11
pub fn movss_store(e: &mut TextEmitter, base: X86Reg, disp: i32, src: Xmm) {
    emit_xmm_modrm_mem(e, 0xF3, 0x0F, 0x11, src, base, disp);
}

/// MOVSD xmm, [base + index*scale + disp]: F2 0F 10
pub fn movsd_load_sib(
    e: &mut TextEmitter,
    dst: Xmm,
    base: X86Reg,
    index: X86Reg,
    scale: u8,
    disp: i32,
) {
    emit_xmm_modrm_sib(e, 0xF2, 0x0F, 0x10, dst, base, index, scale, disp);
}

/// MOVSD [base + index*scale + disp], xmm: F2 0F 11
pub fn movsd_store_sib(
    e: &mut TextEmitter,
    base: X86Reg,
    index: X86Reg,
    scale: u8,
    disp: i32,
    src: Xmm,
) {
    emit_xmm_modrm_sib(e, 0xF2, 0x0F, 0x11, src, base, index, scale, disp);
}

/// MOVSS xmm, [base + index*scale + disp]: F3 0F 10
pub fn movss_load_sib(
    e: &mut TextEmitter,
    dst: Xmm,
    base: X86Reg,
    index: X86Reg,
    scale: u8,
    disp: i32,
) {
    emit_xmm_modrm_sib(e, 0xF3, 0x0F, 0x10, dst, base, index, scale, disp);
}

/// MOVSS [base + index*scale + disp], xmm: F3 0F 11
pub fn movss_store_sib(
    e: &mut TextEmitter,
    base: X86Reg,
    index: X86Reg,
    scale: u8,
    disp: i32,
    src: Xmm,
) {
    emit_xmm_modrm_sib(e, 0xF3, 0x0F, 0x11, src, base, index, scale, disp);
}

/// MOVSD xmm, [RIP + disp32]: F2 0F 10 /r (RIP-relative).
/// Returns offset of the disp32 for patching.
pub fn movsd_load_rip(e: &mut TextEmitter, dst: Xmm, disp32: i32) -> usize {
    emit_xmm_modrm_rip(e, 0xF2, 0x0F, 0x10, dst, disp32)
}

/// MOVSS xmm, [RIP + disp32]: F3 0F 10 /r (RIP-relative).
pub fn movss_load_rip(e: &mut TextEmitter, dst: Xmm, disp32: i32) -> usize {
    emit_xmm_modrm_rip(e, 0xF3, 0x0F, 0x10, dst, disp32)
}

/// MOVQ xmm, [RIP + disp32]: F3 0F 7E (MOVQ form for loading 64-bit).
/// Returns offset of the disp32 for patching.
pub fn movq_load_rip(e: &mut TextEmitter, dst: Xmm, disp32: i32) -> usize {
    emit_xmm_modrm_rip(e, 0xF3, 0x0F, 0x7E, dst, disp32)
}

// --- SSE4.1 rounding (used for floor, ceil, trunc, nearest) ---

/// ROUNDSD xmm, xmm, imm8: 66 0F 3A 0B
/// imm8: 0=nearest, 1=floor, 2=ceil, 3=truncate (+ 8 for "inexact" suppression)
pub fn roundsd(e: &mut TextEmitter, dst: Xmm, src: Xmm, mode: u8) {
    e.emit_u8(0x66);
    emit_xmm_rex(e, false, dst, src);
    e.emit_u8(0x0F);
    e.emit_u8(0x3A);
    e.emit_u8(0x0B);
    emit_xmm_modrm_rr(e, dst, src);
    e.emit_u8(mode);
}

/// ROUNDSS xmm, xmm, imm8: 66 0F 3A 0A
pub fn roundss(e: &mut TextEmitter, dst: Xmm, src: Xmm, mode: u8) {
    e.emit_u8(0x66);
    emit_xmm_rex(e, false, dst, src);
    e.emit_u8(0x0F);
    e.emit_u8(0x3A);
    e.emit_u8(0x0A);
    emit_xmm_modrm_rr(e, dst, src);
    e.emit_u8(mode);
}

/// ROUNDSD/SS mode constants.
pub const ROUND_NEAREST: u8 = 0x08; // 0 + inexact suppression
pub const ROUND_FLOOR: u8 = 0x09; // 1 + inexact suppression
pub const ROUND_CEIL: u8 = 0x0A; // 2 + inexact suppression
pub const ROUND_TRUNC: u8 = 0x0B; // 3 + inexact suppression

// --- FP negate/abs via bit manipulation with RIP-relative constants ---

/// ANDPD/ANDPS with RIP-relative for absolute value (clear sign bit).
pub fn andpd_rip(e: &mut TextEmitter, dst: Xmm, disp32: i32) -> usize {
    emit_xmm_modrm_rip(e, 0x66, 0x0F, 0x54, dst, disp32)
}

pub fn andps_rip(e: &mut TextEmitter, dst: Xmm, disp32: i32) -> usize {
    emit_xmm_modrm_rip(e, 0, 0x0F, 0x54, dst, disp32)
}

/// XORPD/XORPS with RIP-relative for negation (flip sign bit).
pub fn xorpd_rip(e: &mut TextEmitter, dst: Xmm, disp32: i32) -> usize {
    emit_xmm_modrm_rip(e, 0x66, 0x0F, 0x57, dst, disp32)
}

pub fn xorps_rip(e: &mut TextEmitter, dst: Xmm, disp32: i32) -> usize {
    emit_xmm_modrm_rip(e, 0, 0x0F, 0x57, dst, disp32)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Miscellaneous
// ═══════════════════════════════════════════════════════════════════════════════

/// INT3 (breakpoint / debug trap)
pub fn int3(e: &mut TextEmitter) {
    e.emit_u8(0xCC);
}

/// NOP (single-byte)
pub fn nop(e: &mut TextEmitter) {
    e.emit_u8(0x90);
}

/// UD2 (undefined instruction, guaranteed to trap)
pub fn ud2(e: &mut TextEmitter) {
    e.emit_u8(0x0F);
    e.emit_u8(0x0B);
}

/// SUB RSP, imm8
pub fn sub_rsp_imm8(e: &mut TextEmitter, imm8: u8) {
    e.emit_bytes(&[0x48, 0x83, 0xEC, imm8]);
}

/// ADD RSP, imm8
pub fn add_rsp_imm8(e: &mut TextEmitter, imm8: u8) {
    e.emit_bytes(&[0x48, 0x83, 0xC4, imm8]);
}

/// SUB RSP, imm32
pub fn sub_rsp_imm32(e: &mut TextEmitter, imm32: u32) {
    e.emit_bytes(&[0x48, 0x81, 0xEC]);
    e.emit_bytes(&imm32.to_le_bytes());
}

/// ADD RSP, imm32
pub fn add_rsp_imm32(e: &mut TextEmitter, imm32: u32) {
    e.emit_bytes(&[0x48, 0x81, 0xC4]);
    e.emit_bytes(&imm32.to_le_bytes());
}
