//! ARM32 (ARMv7-A, ARM mode) instruction encoding.
//!
//! All instructions are 32-bit, little-endian. The condition field occupies
//! bits [31:28]. We default to AL (always, 0xE) unless otherwise specified.

use super::reg::Arm32Reg;

/// ARM condition codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Cond {
    Eq = 0b0000,
    Ne = 0b0001,
    Cs = 0b0010, // unsigned >=
    Cc = 0b0011, // unsigned <
    Mi = 0b0100,
    Pl = 0b0101,
    Vs = 0b0110,
    Vc = 0b0111,
    Hi = 0b1000, // unsigned >
    Ls = 0b1001, // unsigned <=
    Ge = 0b1010, // signed >=
    Lt = 0b1011, // signed <
    Gt = 0b1100, // signed >
    Le = 0b1101, // signed <=
    Al = 0b1110, // always
}

impl Cond {
    #[inline]
    pub const fn invert(self) -> Cond {
        match self {
            Cond::Eq => Cond::Ne,
            Cond::Ne => Cond::Eq,
            Cond::Cs => Cond::Cc,
            Cond::Cc => Cond::Cs,
            Cond::Mi => Cond::Pl,
            Cond::Pl => Cond::Mi,
            Cond::Vs => Cond::Vc,
            Cond::Vc => Cond::Vs,
            Cond::Hi => Cond::Ls,
            Cond::Ls => Cond::Hi,
            Cond::Ge => Cond::Lt,
            Cond::Lt => Cond::Ge,
            Cond::Gt => Cond::Le,
            Cond::Le => Cond::Gt,
            Cond::Al => Cond::Al,
        }
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Try to encode an immediate as an ARM 8-bit rotated immediate (imm8 rot4).
/// Returns `Some((imm8, rotate))` if encodable, `None` otherwise.
pub const fn encode_arm_imm(value: u32) -> Option<(u32, u32)> {
    let mut rot = 0u32;
    while rot < 16 {
        let shift = rot * 2;
        let rotated = value.rotate_left(shift);
        if rotated <= 0xFF {
            return Some((rotated, rot));
        }
        rot += 1;
    }
    None
}

/// Return true if the value can be represented as an ARM rotated immediate.
#[inline]
pub const fn is_arm_imm(value: u32) -> bool {
    encode_arm_imm(value).is_some()
}

pub const fn cond_bits(c: Cond) -> u32 {
    (c as u32) << 28
}

const fn rd(r: Arm32Reg) -> u32 {
    (r.idx()) << 12
}

const fn rn(r: Arm32Reg) -> u32 {
    (r.idx()) << 16
}

const fn rm(r: Arm32Reg) -> u32 {
    r.idx()
}

const fn rs(r: Arm32Reg) -> u32 {
    (r.idx()) << 8
}

/// Encode imm12 from (imm8, rotate) pair.
const fn imm12(imm8: u32, rotate: u32) -> u32 {
    (rotate << 8) | imm8
}

// ─── Data processing (register) ─────────────────────────────────────────────

/// Generic data-processing instruction: <op>{S} Rd, Rn, Rm
fn dp_reg(cond: Cond, opcode: u32, s: bool, dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    cond_bits(cond) | (opcode << 21) | if s { 1 << 20 } else { 0 } | rn(lhs) | rd(dst) | rm(rhs)
}

/// Generic data-processing instruction with immediate: <op>{S} Rd, Rn, #imm
fn dp_imm(cond: Cond, opcode: u32, s: bool, dst: Arm32Reg, lhs: Arm32Reg, imm8: u32, rotate: u32) -> u32 {
    cond_bits(cond) | (1 << 25) | (opcode << 21) | if s { 1 << 20 } else { 0 } | rn(lhs) | rd(dst) | imm12(imm8, rotate)
}

/// Data-processing with register shift: <op> Rd, Rn, Rm, <shift> Rs
fn dp_reg_shift(cond: Cond, opcode: u32, s: bool, dst: Arm32Reg, lhs: Arm32Reg, shift_rm: Arm32Reg, shift_type: u32, shift_rs: Arm32Reg) -> u32 {
    cond_bits(cond) | (opcode << 21) | if s { 1 << 20 } else { 0 } | rn(lhs) | rd(dst) | rs(shift_rs) | (shift_type << 5) | (1 << 4) | rm(shift_rm)
}

/// Data-processing with immediate shift: <op> Rd, Rn, Rm, <shift> #imm5
fn dp_imm_shift(cond: Cond, opcode: u32, s: bool, dst: Arm32Reg, lhs: Arm32Reg, shift_rm: Arm32Reg, shift_type: u32, shift_imm: u32) -> u32 {
    cond_bits(cond) | (opcode << 21) | if s { 1 << 20 } else { 0 } | rn(lhs) | rd(dst) | ((shift_imm & 0x1F) << 7) | (shift_type << 5) | rm(shift_rm)
}

// ─── Arithmetic ─────────────────────────────────────────────────────────────

/// ADD Rd, Rn, Rm
#[inline]
pub fn add_reg(dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    dp_reg(Cond::Al, 0b0100, false, dst, lhs, rhs)
}

/// ADDS Rd, Rn, Rm (sets flags)
#[inline]
pub fn adds_reg(dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    dp_reg(Cond::Al, 0b0100, true, dst, lhs, rhs)
}

/// ADC Rd, Rn, Rm (add with carry)
#[inline]
pub fn adc_reg(dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    dp_reg(Cond::Al, 0b0101, false, dst, lhs, rhs)
}

/// ADD Rd, Rn, #imm
#[inline]
pub fn add_imm(dst: Arm32Reg, src: Arm32Reg, imm8: u32, rotate: u32) -> u32 {
    dp_imm(Cond::Al, 0b0100, false, dst, src, imm8, rotate)
}

/// SUB Rd, Rn, Rm
#[inline]
pub fn sub_reg(dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    dp_reg(Cond::Al, 0b0010, false, dst, lhs, rhs)
}

/// SUBS Rd, Rn, Rm (sets flags)
#[inline]
pub fn subs_reg(dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    dp_reg(Cond::Al, 0b0010, true, dst, lhs, rhs)
}

/// SBC Rd, Rn, Rm (sub with carry)
#[inline]
pub fn sbc_reg(dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    dp_reg(Cond::Al, 0b0110, false, dst, lhs, rhs)
}

/// SUB Rd, Rn, #imm
#[inline]
pub fn sub_imm(dst: Arm32Reg, src: Arm32Reg, imm8: u32, rotate: u32) -> u32 {
    dp_imm(Cond::Al, 0b0010, false, dst, src, imm8, rotate)
}

/// RSB Rd, Rn, #imm (reverse subtract: Rd = imm - Rn)
#[inline]
pub fn rsb_imm(dst: Arm32Reg, src: Arm32Reg, imm8: u32, rotate: u32) -> u32 {
    dp_imm(Cond::Al, 0b0011, false, dst, src, imm8, rotate)
}

// ─── Logic ──────────────────────────────────────────────────────────────────

/// AND Rd, Rn, Rm
#[inline]
pub fn and_reg(dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    dp_reg(Cond::Al, 0b0000, false, dst, lhs, rhs)
}

/// ANDS Rd, Rn, Rm
#[inline]
pub fn ands_reg(dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    dp_reg(Cond::Al, 0b0000, true, dst, lhs, rhs)
}

/// AND Rd, Rn, #imm
#[inline]
pub fn and_imm(dst: Arm32Reg, src: Arm32Reg, imm8: u32, rotate: u32) -> u32 {
    dp_imm(Cond::Al, 0b0000, false, dst, src, imm8, rotate)
}

/// ORR Rd, Rn, Rm
#[inline]
pub fn orr_reg(dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    dp_reg(Cond::Al, 0b1100, false, dst, lhs, rhs)
}

/// ORR Rd, Rn, #imm
#[inline]
pub fn orr_imm(dst: Arm32Reg, src: Arm32Reg, imm8: u32, rotate: u32) -> u32 {
    dp_imm(Cond::Al, 0b1100, false, dst, src, imm8, rotate)
}

/// EOR Rd, Rn, Rm
#[inline]
pub fn eor_reg(dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    dp_reg(Cond::Al, 0b0001, false, dst, lhs, rhs)
}

/// EOR Rd, Rn, #imm
#[inline]
pub fn eor_imm(dst: Arm32Reg, src: Arm32Reg, imm8: u32, rotate: u32) -> u32 {
    dp_imm(Cond::Al, 0b0001, false, dst, src, imm8, rotate)
}

/// BIC Rd, Rn, Rm (bit clear: Rd = Rn & ~Rm)
#[inline]
pub fn bic_reg(dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    dp_reg(Cond::Al, 0b1110, false, dst, lhs, rhs)
}

/// BIC Rd, Rn, #imm (bitwise clear with immediate)
#[inline]
pub fn bic_imm(dst: Arm32Reg, lhs: Arm32Reg, imm8: u32, rotate: u32) -> u32 {
    dp_imm(Cond::Al, 0b1110, false, dst, lhs, imm8, rotate)
}

/// MVN Rd, Rm (bitwise NOT)
#[inline]
pub fn mvn_reg(dst: Arm32Reg, src: Arm32Reg) -> u32 {
    // MVN is dp opcode 0b1111, Rn is ignored (set to R0)
    dp_reg(Cond::Al, 0b1111, false, dst, Arm32Reg::R0, src)
}

/// MVN Rd, #imm
#[inline]
pub fn mvn_imm(dst: Arm32Reg, imm8: u32, rotate: u32) -> u32 {
    dp_imm(Cond::Al, 0b1111, false, dst, Arm32Reg::R0, imm8, rotate)
}

// ─── Move ───────────────────────────────────────────────────────────────────

/// MOV Rd, Rm
#[inline]
pub fn mov_reg(dst: Arm32Reg, src: Arm32Reg) -> u32 {
    // MOV is dp opcode 0b1101, Rn is ignored
    dp_reg(Cond::Al, 0b1101, false, dst, Arm32Reg::R0, src)
}

/// MOV{cond} Rd, Rm
#[inline]
pub fn mov_reg_cond(cond: Cond, dst: Arm32Reg, src: Arm32Reg) -> u32 {
    dp_reg(cond, 0b1101, false, dst, Arm32Reg::R0, src)
}

/// MOV Rd, #imm (rotated)
#[inline]
pub fn mov_imm(dst: Arm32Reg, imm8: u32, rotate: u32) -> u32 {
    dp_imm(Cond::Al, 0b1101, false, dst, Arm32Reg::R0, imm8, rotate)
}

/// MOVW Rd, #imm16 (ARMv6T2+: load bottom 16 bits, clear top 16 bits)
#[inline]
pub fn movw(dst: Arm32Reg, imm16: u16) -> u32 {
    let imm = imm16 as u32;
    let imm12 = imm & 0xFFF;
    let imm4 = (imm >> 12) & 0xF;
    cond_bits(Cond::Al) | (0b0011_0000 << 20) | (imm4 << 16) | rd(dst) | imm12
}

/// MOVT Rd, #imm16 (ARMv6T2+: load top 16 bits, preserve bottom 16)
#[inline]
pub fn movt(dst: Arm32Reg, imm16: u16) -> u32 {
    let imm = imm16 as u32;
    let imm12 = imm & 0xFFF;
    let imm4 = (imm >> 12) & 0xF;
    cond_bits(Cond::Al) | (0b0011_0100 << 20) | (imm4 << 16) | rd(dst) | imm12
}

// ─── Shifts ─────────────────────────────────────────────────────────────────

/// LSL Rd, Rm, Rs (logical shift left by register)
#[inline]
pub fn lsl_reg(dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    dp_reg_shift(Cond::Al, 0b1101, false, dst, Arm32Reg::R0, lhs, 0b00, rhs)
}

/// LSL Rd, Rm, #imm5 (logical shift left by immediate)
#[inline]
pub fn lsl_imm(dst: Arm32Reg, src: Arm32Reg, shift: u32) -> u32 {
    if shift == 0 {
        return mov_reg(dst, src);
    }
    dp_imm_shift(Cond::Al, 0b1101, false, dst, Arm32Reg::R0, src, 0b00, shift)
}

/// LSR Rd, Rm, Rs (logical shift right by register)
#[inline]
pub fn lsr_reg(dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    dp_reg_shift(Cond::Al, 0b1101, false, dst, Arm32Reg::R0, lhs, 0b01, rhs)
}

/// LSR Rd, Rm, #imm5 (logical shift right by immediate)
#[inline]
pub fn lsr_imm(dst: Arm32Reg, src: Arm32Reg, shift: u32) -> u32 {
    dp_imm_shift(Cond::Al, 0b1101, false, dst, Arm32Reg::R0, src, 0b01, shift)
}

/// ASR Rd, Rm, Rs (arithmetic shift right by register)
#[inline]
pub fn asr_reg(dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    dp_reg_shift(Cond::Al, 0b1101, false, dst, Arm32Reg::R0, lhs, 0b10, rhs)
}

/// ASR Rd, Rm, #imm5 (arithmetic shift right by immediate)
#[inline]
pub fn asr_imm(dst: Arm32Reg, src: Arm32Reg, shift: u32) -> u32 {
    dp_imm_shift(Cond::Al, 0b1101, false, dst, Arm32Reg::R0, src, 0b10, shift)
}

/// ROR Rd, Rm, Rs (rotate right by register)
#[inline]
pub fn ror_reg(dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    dp_reg_shift(Cond::Al, 0b1101, false, dst, Arm32Reg::R0, lhs, 0b11, rhs)
}

/// ROR Rd, Rm, #imm5 (rotate right by immediate)
#[inline]
pub fn ror_imm(dst: Arm32Reg, src: Arm32Reg, shift: u32) -> u32 {
    dp_imm_shift(Cond::Al, 0b1101, false, dst, Arm32Reg::R0, src, 0b11, shift)
}

// ─── Compare / Test ─────────────────────────────────────────────────────────

/// CMP Rn, Rm (sets flags, no destination)
#[inline]
pub fn cmp_reg(lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    // CMP = SUBS with Rd=R0, S=1; opcode = 0b1010
    dp_reg(Cond::Al, 0b1010, true, Arm32Reg::R0, lhs, rhs)
}

/// CMP Rn, #imm
#[inline]
pub fn cmp_imm(src: Arm32Reg, imm8: u32, rotate: u32) -> u32 {
    dp_imm(Cond::Al, 0b1010, true, Arm32Reg::R0, src, imm8, rotate)
}

/// CMN Rn, Rm (compare negative: flags of Rn + Rm)
#[inline]
pub fn cmn_reg(lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    dp_reg(Cond::Al, 0b1011, true, Arm32Reg::R0, lhs, rhs)
}

/// CMN Rn, #imm (compare negative: flags of Rn + imm)
#[inline]
pub fn cmn_imm(lhs: Arm32Reg, imm8: u32, rotate: u32) -> u32 {
    dp_imm(Cond::Al, 0b1011, true, Arm32Reg::R0, lhs, imm8, rotate)
}

/// TST Rn, Rm (test: flags of Rn AND Rm)
#[inline]
pub fn tst_reg(lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    dp_reg(Cond::Al, 0b1000, true, Arm32Reg::R0, lhs, rhs)
}

/// TST Rn, #imm
#[inline]
pub fn tst_imm(src: Arm32Reg, imm8: u32, rotate: u32) -> u32 {
    dp_imm(Cond::Al, 0b1000, true, Arm32Reg::R0, src, imm8, rotate)
}

// ─── Multiply ───────────────────────────────────────────────────────────────

/// MUL Rd, Rm, Rs (Rd = Rm * Rs, low 32 bits)
#[inline]
pub fn mul(dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    // MUL encoding: cond 0000 000S Rd 0000 Rs 1001 Rm
    cond_bits(Cond::Al) | (rn(dst) >> 4) | rs(rhs) | (0b1001 << 4) | rm(lhs)
}

/// UMULL RdLo, RdHi, Rm, Rs (unsigned 32x32→64)
#[inline]
pub fn umull(dst_lo: Arm32Reg, dst_hi: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    // UMULL: cond 0000 100S RdHi RdLo Rs 1001 Rm
    cond_bits(Cond::Al) | (0b0000100 << 21) | (dst_hi.idx() << 16) | rd(dst_lo) | rs(rhs) | (0b1001 << 4) | rm(lhs)
}

/// SMULL RdLo, RdHi, Rm, Rs (signed 32x32→64)
#[inline]
pub fn smull(dst_lo: Arm32Reg, dst_hi: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    // SMULL: cond 0000 110S RdHi RdLo Rs 1001 Rm
    cond_bits(Cond::Al) | (0b0000110 << 21) | (dst_hi.idx() << 16) | rd(dst_lo) | rs(rhs) | (0b1001 << 4) | rm(lhs)
}

// ─── CLZ ────────────────────────────────────────────────────────────────────

/// CLZ Rd, Rm (count leading zeros)
#[inline]
pub fn clz(dst: Arm32Reg, src: Arm32Reg) -> u32 {
    // CLZ: cond 0001 0110 1111 Rd 1111 0001 Rm
    cond_bits(Cond::Al) | (0b000101101111 << 16) | rd(dst) | (0b11110001 << 0) | rm(src)
}

// ─── Load / Store (immediate offset) ────────────────────────────────────────

/// LDR Rd, [Rn, #offset] (word load, +/- 12-bit immediate)
#[inline]
pub fn ldr_imm(dst: Arm32Reg, base: Arm32Reg, offset: i32) -> u32 {
    let u = if offset >= 0 { 1u32 } else { 0 };
    let abs_offset = offset.unsigned_abs() & 0xFFF;
    cond_bits(Cond::Al) | (0b01 << 26) | (1 << 24) | (u << 23) | (1 << 20) | rn(base) | rd(dst) | abs_offset
}

/// STR Rd, [Rn, #offset] (word store)
#[inline]
pub fn str_imm(src: Arm32Reg, base: Arm32Reg, offset: i32) -> u32 {
    let u = if offset >= 0 { 1u32 } else { 0 };
    let abs_offset = offset.unsigned_abs() & 0xFFF;
    cond_bits(Cond::Al) | (0b01 << 26) | (1 << 24) | (u << 23) | rn(base) | rd(src) | abs_offset
}

/// LDRB Rd, [Rn, #offset] (byte load, zero-extend)
#[inline]
pub fn ldrb_imm(dst: Arm32Reg, base: Arm32Reg, offset: i32) -> u32 {
    let u = if offset >= 0 { 1u32 } else { 0 };
    let abs_offset = offset.unsigned_abs() & 0xFFF;
    cond_bits(Cond::Al) | (0b01 << 26) | (1 << 24) | (u << 23) | (1 << 22) | (1 << 20) | rn(base) | rd(dst) | abs_offset
}

/// STRB Rd, [Rn, #offset] (byte store)
#[inline]
pub fn strb_imm(src: Arm32Reg, base: Arm32Reg, offset: i32) -> u32 {
    let u = if offset >= 0 { 1u32 } else { 0 };
    let abs_offset = offset.unsigned_abs() & 0xFFF;
    cond_bits(Cond::Al) | (0b01 << 26) | (1 << 24) | (u << 23) | (1 << 22) | rn(base) | rd(src) | abs_offset
}

/// LDRH Rd, [Rn, #offset] (halfword load, zero-extend; immediate 8-bit offset)
#[inline]
pub fn ldrh_imm(dst: Arm32Reg, base: Arm32Reg, offset: i32) -> u32 {
    let u = if offset >= 0 { 1u32 } else { 0 };
    let abs_off = offset.unsigned_abs() & 0xFF;
    let imm_hi = (abs_off >> 4) & 0xF;
    let imm_lo = abs_off & 0xF;
    // Encoding: cond 000P U1W0 Rn Rd imm4H 1011 imm4L
    cond_bits(Cond::Al) | (1 << 24) | (u << 23) | (1 << 22) | (1 << 20) | rn(base) | rd(dst) | (imm_hi << 8) | (0b1011 << 4) | imm_lo
}

/// STRH Rd, [Rn, #offset] (halfword store; immediate 8-bit offset)
#[inline]
pub fn strh_imm(src: Arm32Reg, base: Arm32Reg, offset: i32) -> u32 {
    let u = if offset >= 0 { 1u32 } else { 0 };
    let abs_off = offset.unsigned_abs() & 0xFF;
    let imm_hi = (abs_off >> 4) & 0xF;
    let imm_lo = abs_off & 0xF;
    cond_bits(Cond::Al) | (1 << 24) | (u << 23) | (1 << 22) | rn(base) | rd(src) | (imm_hi << 8) | (0b1011 << 4) | imm_lo
}

/// LDRSB Rd, [Rn, #offset] (signed byte load; imm 8-bit offset)
#[inline]
pub fn ldrsb_imm(dst: Arm32Reg, base: Arm32Reg, offset: i32) -> u32 {
    let u = if offset >= 0 { 1u32 } else { 0 };
    let abs_off = offset.unsigned_abs() & 0xFF;
    let imm_hi = (abs_off >> 4) & 0xF;
    let imm_lo = abs_off & 0xF;
    cond_bits(Cond::Al) | (1 << 24) | (u << 23) | (1 << 22) | (1 << 20) | rn(base) | rd(dst) | (imm_hi << 8) | (0b1101 << 4) | imm_lo
}

/// LDRSH Rd, [Rn, #offset] (signed halfword load; imm 8-bit offset)
#[inline]
pub fn ldrsh_imm(dst: Arm32Reg, base: Arm32Reg, offset: i32) -> u32 {
    let u = if offset >= 0 { 1u32 } else { 0 };
    let abs_off = offset.unsigned_abs() & 0xFF;
    let imm_hi = (abs_off >> 4) & 0xF;
    let imm_lo = abs_off & 0xF;
    cond_bits(Cond::Al) | (1 << 24) | (u << 23) | (1 << 22) | (1 << 20) | rn(base) | rd(dst) | (imm_hi << 8) | (0b1111 << 4) | imm_lo
}

// ─── Load / Store (register offset) ────────────────────────────────────────

/// LDR Rd, [Rn, Rm] (word load, register offset)
#[inline]
pub fn ldr_reg(dst: Arm32Reg, base: Arm32Reg, index: Arm32Reg) -> u32 {
    cond_bits(Cond::Al) | (0b011 << 25) | (1 << 24) | (1 << 23) | (1 << 20) | rn(base) | rd(dst) | rm(index)
}

/// STR Rd, [Rn, Rm] (word store, register offset)
#[inline]
pub fn str_reg(src: Arm32Reg, base: Arm32Reg, index: Arm32Reg) -> u32 {
    cond_bits(Cond::Al) | (0b011 << 25) | (1 << 24) | (1 << 23) | rn(base) | rd(src) | rm(index)
}

/// LDR Rd, [Rn, Rm, LSL #shift]
#[inline]
pub fn ldr_reg_lsl(dst: Arm32Reg, base: Arm32Reg, index: Arm32Reg, shift: u32) -> u32 {
    cond_bits(Cond::Al) | (0b011 << 25) | (1 << 24) | (1 << 23) | (1 << 20) | rn(base) | rd(dst) | ((shift & 0x1F) << 7) | rm(index)
}

// ─── 64-bit Load / Store (LDRD/STRD) ───────────────────────────────────────

/// LDRD Rt, Rt2, [Rn, #offset] (load doubleword; Rt must be even, Rt2 = Rt+1)
#[inline]
pub fn ldrd_imm(dst_even: Arm32Reg, base: Arm32Reg, offset: i32) -> u32 {
    let u = if offset >= 0 { 1u32 } else { 0 };
    let abs_off = offset.unsigned_abs() & 0xFF;
    let imm_hi = (abs_off >> 4) & 0xF;
    let imm_lo = abs_off & 0xF;
    // LDRD: cond 000P U1W0 Rn Rt imm4H 1101 imm4L
    cond_bits(Cond::Al) | (1 << 24) | (u << 23) | (1 << 22) | rn(base) | rd(dst_even) | (imm_hi << 8) | (0b1101 << 4) | imm_lo
}

/// STRD Rt, Rt2, [Rn, #offset] (store doubleword; Rt must be even, Rt2 = Rt+1)
#[inline]
pub fn strd_imm(src_even: Arm32Reg, base: Arm32Reg, offset: i32) -> u32 {
    let u = if offset >= 0 { 1u32 } else { 0 };
    let abs_off = offset.unsigned_abs() & 0xFF;
    let imm_hi = (abs_off >> 4) & 0xF;
    let imm_lo = abs_off & 0xF;
    // STRD: cond 000P U1W0 Rn Rt imm4H 1111 imm4L
    cond_bits(Cond::Al) | (1 << 24) | (u << 23) | (1 << 22) | rn(base) | rd(src_even) | (imm_hi << 8) | (0b1111 << 4) | imm_lo
}

// ─── Push / Pop (STMDB/LDMIA with SP) ──────────────────────────────────────

/// PUSH {regs} (STMDB SP!, {regs})
#[inline]
pub fn push(reg_mask: u16) -> u32 {
    // STMDB SP!: cond 1001 0010 1101 reg_list
    cond_bits(Cond::Al) | (0b100100101101 << 16) | (reg_mask as u32)
}

/// POP {regs} (LDMIA SP!, {regs})
#[inline]
pub fn pop(reg_mask: u16) -> u32 {
    // LDMIA SP!: cond 1000 1011 1101 reg_list
    cond_bits(Cond::Al) | (0b100010111101 << 16) | (reg_mask as u32)
}

// ─── Branch ─────────────────────────────────────────────────────────────────

/// B <offset> (unconditional branch, PC-relative; offset in bytes, must be aligned)
#[inline]
pub fn b(byte_offset: i32) -> u32 {
    // The encoded offset is (target - PC) / 4, where PC = current + 8 on ARM.
    // Caller provides the raw byte delta from the branch instruction to the target.
    // So encoded imm24 = (byte_offset - 8) / 4 = (byte_offset >> 2) - 2
    let adjusted = (byte_offset >> 2) - 2;
    cond_bits(Cond::Al) | (0b1010 << 24) | ((adjusted as u32) & 0x00FF_FFFF)
}

/// B{cond} <offset> (conditional branch)
#[inline]
pub fn b_cond(cond: Cond, byte_offset: i32) -> u32 {
    let adjusted = (byte_offset >> 2) - 2;
    cond_bits(cond) | (0b1010 << 24) | ((adjusted as u32) & 0x00FF_FFFF)
}

/// BL <offset> (branch with link)
#[inline]
pub fn bl(byte_offset: i32) -> u32 {
    let adjusted = (byte_offset >> 2) - 2;
    cond_bits(Cond::Al) | (0b1011 << 24) | ((adjusted as u32) & 0x00FF_FFFF)
}

/// BX Rm (branch and exchange — return via LR or indirect call)
#[inline]
pub fn bx(reg: Arm32Reg) -> u32 {
    cond_bits(Cond::Al) | (0b0001_0010_1111_1111_1111_0001 << 0) | rm(reg)
}

/// BLX Rm (branch with link and exchange — indirect call)
#[inline]
pub fn blx_reg(reg: Arm32Reg) -> u32 {
    cond_bits(Cond::Al) | (0b0001_0010_1111_1111_1111_0011 << 0) | rm(reg)
}

// ─── Sign/Zero extend (ARMv6+) ─────────────────────────────────────────────

/// SXTB Rd, Rm (sign-extend byte to 32 bits)
#[inline]
pub fn sxtb(dst: Arm32Reg, src: Arm32Reg) -> u32 {
    // SXTB: cond 0110 1010 1111 Rd 0000 0111 Rm
    cond_bits(Cond::Al) | (0b01101010 << 20) | (0b1111 << 16) | rd(dst) | (0b00000111 << 4) | rm(src)
}

/// SXTH Rd, Rm (sign-extend halfword to 32 bits)
#[inline]
pub fn sxth(dst: Arm32Reg, src: Arm32Reg) -> u32 {
    // SXTH: cond 0110 1011 1111 Rd 0000 0111 Rm
    cond_bits(Cond::Al) | (0b01101011 << 20) | (0b1111 << 16) | rd(dst) | (0b00000111 << 4) | rm(src)
}

/// UXTB Rd, Rm (zero-extend byte)
#[inline]
pub fn uxtb(dst: Arm32Reg, src: Arm32Reg) -> u32 {
    cond_bits(Cond::Al) | (0b01101110 << 20) | (0b1111 << 16) | rd(dst) | (0b00000111 << 4) | rm(src)
}

/// UXTH Rd, Rm (zero-extend halfword)
#[inline]
pub fn uxth(dst: Arm32Reg, src: Arm32Reg) -> u32 {
    cond_bits(Cond::Al) | (0b01101111 << 20) | (0b1111 << 16) | rd(dst) | (0b00000111 << 4) | rm(src)
}

// ─── VFP (floating point) ───────────────────────────────────────────────────

/// VLDR.64 Dd, [Rn, #offset] (load double; offset is in 4-byte units, ±1020)
#[inline]
pub fn vldr_d(dd: u32, base: Arm32Reg, offset: i32) -> u32 {
    let u = if offset >= 0 { 1u32 } else { 0 };
    let abs_off = (offset.unsigned_abs() / 4) & 0xFF;
    let d = (dd >> 4) & 1;
    let vd = dd & 0xF;
    cond_bits(Cond::Al) | (0b1101 << 24) | (u << 23) | (d << 22) | (1 << 20) | rn(base) | (vd << 12) | (0b1011 << 8) | abs_off
}

/// VSTR.64 Dd, [Rn, #offset]
#[inline]
pub fn vstr_d(dd: u32, base: Arm32Reg, offset: i32) -> u32 {
    let u = if offset >= 0 { 1u32 } else { 0 };
    let abs_off = (offset.unsigned_abs() / 4) & 0xFF;
    let d = (dd >> 4) & 1;
    let vd = dd & 0xF;
    cond_bits(Cond::Al) | (0b1101 << 24) | (u << 23) | (d << 22) | rn(base) | (vd << 12) | (0b1011 << 8) | abs_off
}

/// VLDR.32 Sd, [Rn, #offset] (load single; offset in 4-byte units, ±1020)
#[inline]
pub fn vldr_s(sd: u32, base: Arm32Reg, offset: i32) -> u32 {
    let u = if offset >= 0 { 1u32 } else { 0 };
    let abs_off = (offset.unsigned_abs() / 4) & 0xFF;
    let d = sd & 1;
    let vd = (sd >> 1) & 0xF;
    cond_bits(Cond::Al) | (0b1101 << 24) | (u << 23) | (d << 22) | (1 << 20) | rn(base) | (vd << 12) | (0b1010 << 8) | abs_off
}

/// VSTR.32 Sd, [Rn, #offset]
#[inline]
pub fn vstr_s(sd: u32, base: Arm32Reg, offset: i32) -> u32 {
    let u = if offset >= 0 { 1u32 } else { 0 };
    let abs_off = (offset.unsigned_abs() / 4) & 0xFF;
    let d = sd & 1;
    let vd = (sd >> 1) & 0xF;
    cond_bits(Cond::Al) | (0b1101 << 24) | (u << 23) | (d << 22) | rn(base) | (vd << 12) | (0b1010 << 8) | abs_off
}

/// VMOV Dd, Dm (double register copy)
#[inline]
pub fn vmov_d(dst: u32, src: u32) -> u32 {
    let d_dst = (dst >> 4) & 1;
    let vd = dst & 0xF;
    let m_src = (src >> 4) & 1;
    let vm = src & 0xF;
    // VMOV.F64: cond 1110 1D11 0000 Vd 1011 01M0 Vm
    cond_bits(Cond::Al) | (0b11101 << 23) | (d_dst << 22) | (0b110000 << 16) | (vd << 12) | (0b10110100 << 4) | (m_src << 5) | vm
}

/// VMOV Sd, Sm (single register copy)
#[inline]
pub fn vmov_s(dst: u32, src: u32) -> u32 {
    let d_dst = dst & 1;
    let vd = (dst >> 1) & 0xF;
    let m_src = src & 1;
    let vm = (src >> 1) & 0xF;
    // VMOV.F32: cond 1110 1D11 0000 Vd 1010 01M0 Vm
    cond_bits(Cond::Al) | (0b11101 << 23) | (d_dst << 22) | (0b110000 << 16) | (vd << 12) | (0b10100100 << 4) | (m_src << 5) | vm
}

/// VMOV Rd, Sn (move single VFP to GP reg)
#[inline]
pub fn vmov_r_s(dst: Arm32Reg, sn: u32) -> u32 {
    let n = sn & 1;
    let vn = (sn >> 1) & 0xF;
    // VMOV Rt, Sn: cond 1110 000 1 Vn Rt 1010 N001 0000
    cond_bits(Cond::Al) | (0b1110 << 24) | (1 << 20) | (vn << 16) | rd(dst) | (0b1010 << 8) | (n << 7) | (1 << 4)
}

/// VMOV Sn, Rd (move GP reg to single VFP)
#[inline]
pub fn vmov_s_r(sn: u32, src: Arm32Reg) -> u32 {
    let n = sn & 1;
    let vn = (sn >> 1) & 0xF;
    cond_bits(Cond::Al) | (0b1110 << 24) | (vn << 16) | rd(src) | (0b1010 << 8) | (n << 7) | (1 << 4)
}

/// VMOV Rd, Rd2, Dm (move double VFP to two GP regs)
#[inline]
pub fn vmov_rr_d(dst_lo: Arm32Reg, dst_hi: Arm32Reg, dm: u32) -> u32 {
    let m = (dm >> 4) & 1;
    let vm = dm & 0xF;
    // VMOV Rt, Rt2, Dm: cond 1100 0101 Rt2 Rt 1011 00M1 Vm
    cond_bits(Cond::Al) | (0b11000101 << 20) | (dst_hi.idx() << 16) | rd(dst_lo) | (0b1011 << 8) | (m << 5) | (1 << 4) | vm
}

/// VMOV Dm, Rd, Rd2 (move two GP regs to double VFP)
#[inline]
pub fn vmov_d_rr(dm: u32, src_lo: Arm32Reg, src_hi: Arm32Reg) -> u32 {
    let m = (dm >> 4) & 1;
    let vm = dm & 0xF;
    // VMOV Dm, Rt, Rt2: cond 1100 0100 Rt2 Rt 1011 00M1 Vm
    cond_bits(Cond::Al) | (0b11000100 << 20) | (src_hi.idx() << 16) | rd(src_lo) | (0b1011 << 8) | (m << 5) | (1 << 4) | vm
}

/// VFP double-precision binary op helper.
fn vfp_d_binary(opcode_hi: u32, opcode_lo: u32, dd: u32, dn: u32, dm: u32) -> u32 {
    let d = (dd >> 4) & 1;
    let vd = dd & 0xF;
    let n = (dn >> 4) & 1;
    let vn = dn & 0xF;
    let m = (dm >> 4) & 1;
    let vm = dm & 0xF;
    cond_bits(Cond::Al) | (0b11100 << 23) | (d << 22) | (opcode_hi << 20) | (vn << 16) | (vd << 12) | (0b1011 << 8) | (n << 7) | (opcode_lo << 6) | (m << 5) | vm
}

/// VFP single-precision binary op helper.
fn vfp_s_binary(opcode_hi: u32, opcode_lo: u32, sd: u32, sn: u32, sm: u32) -> u32 {
    let d = sd & 1;
    let vd = (sd >> 1) & 0xF;
    let n = sn & 1;
    let vn = (sn >> 1) & 0xF;
    let m = sm & 1;
    let vm = (sm >> 1) & 0xF;
    cond_bits(Cond::Al) | (0b11100 << 23) | (d << 22) | (opcode_hi << 20) | (vn << 16) | (vd << 12) | (0b1010 << 8) | (n << 7) | (opcode_lo << 6) | (m << 5) | vm
}

/// VADD.F64 Dd, Dn, Dm
#[inline]
pub fn vadd_d(dd: u32, dn: u32, dm: u32) -> u32 { vfp_d_binary(0b11, 0b0, dd, dn, dm) }

/// VSUB.F64 Dd, Dn, Dm
#[inline]
pub fn vsub_d(dd: u32, dn: u32, dm: u32) -> u32 { vfp_d_binary(0b11, 0b1, dd, dn, dm) }

/// VMUL.F64 Dd, Dn, Dm
#[inline]
pub fn vmul_d(dd: u32, dn: u32, dm: u32) -> u32 { vfp_d_binary(0b10, 0b0, dd, dn, dm) }

/// VDIV.F64 Dd, Dn, Dm
#[inline]
pub fn vdiv_d(dd: u32, dn: u32, dm: u32) -> u32 {
    let d = (dd >> 4) & 1;
    let vd = dd & 0xF;
    let n = (dn >> 4) & 1;
    let vn = dn & 0xF;
    let m = (dm >> 4) & 1;
    let vm = dm & 0xF;
    cond_bits(Cond::Al) | (0b11101 << 23) | (d << 22) | (0b00 << 20) | (vn << 16) | (vd << 12) | (0b1011 << 8) | (n << 7) | (m << 5) | vm
}

/// VADD.F32 Sd, Sn, Sm
#[inline]
pub fn vadd_s(sd: u32, sn: u32, sm: u32) -> u32 { vfp_s_binary(0b11, 0b0, sd, sn, sm) }

/// VSUB.F32 Sd, Sn, Sm
#[inline]
pub fn vsub_s(sd: u32, sn: u32, sm: u32) -> u32 { vfp_s_binary(0b11, 0b1, sd, sn, sm) }

/// VMUL.F32 Sd, Sn, Sm
#[inline]
pub fn vmul_s(sd: u32, sn: u32, sm: u32) -> u32 { vfp_s_binary(0b10, 0b0, sd, sn, sm) }

/// VDIV.F32 Sd, Sn, Sm
#[inline]
pub fn vdiv_s(dd: u32, dn: u32, dm: u32) -> u32 {
    let d = dd & 1;
    let vd = (dd >> 1) & 0xF;
    let n = dn & 1;
    let vn = (dn >> 1) & 0xF;
    let m = dm & 1;
    let vm = (dm >> 1) & 0xF;
    cond_bits(Cond::Al) | (0b11101 << 23) | (d << 22) | (0b00 << 20) | (vn << 16) | (vd << 12) | (0b1010 << 8) | (n << 7) | (m << 5) | vm
}

/// VSQRT.F64 Dd, Dm
#[inline]
pub fn vsqrt_d(dd: u32, dm: u32) -> u32 {
    let d = (dd >> 4) & 1;
    let vd = dd & 0xF;
    let m = (dm >> 4) & 1;
    let vm = dm & 0xF;
    cond_bits(Cond::Al) | (0b11101 << 23) | (d << 22) | (0b110001 << 16) | (vd << 12) | (0b1011 << 8) | (0b11 << 6) | (m << 5) | vm
}

/// VSQRT.F32 Sd, Sm
#[inline]
pub fn vsqrt_s(sd: u32, sm: u32) -> u32 {
    let d = sd & 1;
    let vd = (sd >> 1) & 0xF;
    let m = sm & 1;
    let vm = (sm >> 1) & 0xF;
    cond_bits(Cond::Al) | (0b11101 << 23) | (d << 22) | (0b110001 << 16) | (vd << 12) | (0b1010 << 8) | (0b11 << 6) | (m << 5) | vm
}

/// VNEG.F64 Dd, Dm
#[inline]
pub fn vneg_d(dd: u32, dm: u32) -> u32 {
    let d = (dd >> 4) & 1;
    let vd = dd & 0xF;
    let m = (dm >> 4) & 1;
    let vm = dm & 0xF;
    cond_bits(Cond::Al) | (0b11101 << 23) | (d << 22) | (0b110001 << 16) | (vd << 12) | (0b1011 << 8) | (0b01 << 6) | (m << 5) | vm
}

/// VNEG.F32 Sd, Sm
#[inline]
pub fn vneg_s(sd: u32, sm: u32) -> u32 {
    let d = sd & 1;
    let vd = (sd >> 1) & 0xF;
    let m = sm & 1;
    let vm = (sm >> 1) & 0xF;
    cond_bits(Cond::Al) | (0b11101 << 23) | (d << 22) | (0b110001 << 16) | (vd << 12) | (0b1010 << 8) | (0b01 << 6) | (m << 5) | vm
}

/// VABS.F64 Dd, Dm
#[inline]
pub fn vabs_d(dd: u32, dm: u32) -> u32 {
    let d = (dd >> 4) & 1;
    let vd = dd & 0xF;
    let m = (dm >> 4) & 1;
    let vm = dm & 0xF;
    cond_bits(Cond::Al) | (0b11101 << 23) | (d << 22) | (0b110000 << 16) | (vd << 12) | (0b1011 << 8) | (0b11 << 6) | (m << 5) | vm
}

/// VABS.F32 Sd, Sm
#[inline]
pub fn vabs_s(sd: u32, sm: u32) -> u32 {
    let d = sd & 1;
    let vd = (sd >> 1) & 0xF;
    let m = sm & 1;
    let vm = (sm >> 1) & 0xF;
    cond_bits(Cond::Al) | (0b11101 << 23) | (d << 22) | (0b110000 << 16) | (vd << 12) | (0b1010 << 8) | (0b11 << 6) | (m << 5) | vm
}

/// VCMP.F64 Dd, Dm
#[inline]
pub fn vcmp_d(dd: u32, dm: u32) -> u32 {
    let d = (dd >> 4) & 1;
    let vd = dd & 0xF;
    let m = (dm >> 4) & 1;
    let vm = dm & 0xF;
    cond_bits(Cond::Al) | (0b11101 << 23) | (d << 22) | (0b110100 << 16) | (vd << 12) | (0b1011 << 8) | (0b01 << 6) | (m << 5) | vm
}

/// VCMP.F32 Sd, Sm
#[inline]
pub fn vcmp_s(sd: u32, sm: u32) -> u32 {
    let d = sd & 1;
    let vd = (sd >> 1) & 0xF;
    let m = sm & 1;
    let vm = (sm >> 1) & 0xF;
    cond_bits(Cond::Al) | (0b11101 << 23) | (d << 22) | (0b110100 << 16) | (vd << 12) | (0b1010 << 8) | (0b01 << 6) | (m << 5) | vm
}

/// VMRS APSR_nzcv, FPSCR (move VFP flags to ARM condition flags)
#[inline]
pub fn vmrs_apsr() -> u32 {
    // VMRS APSR_nzcv, FPSCR: cond 1110 1111 0001 1111 1010 0001 0000
    cond_bits(Cond::Al) | (0b11101111 << 20) | (0b0001 << 16) | (0b1111 << 12) | (0b1010 << 8) | (0b00010000)
}

/// VCVT.F64.F32 Dd, Sm (single to double)
#[inline]
pub fn vcvt_d_s(dd: u32, sm: u32) -> u32 {
    let d = (dd >> 4) & 1;
    let vd = dd & 0xF;
    let m = sm & 1;
    let vm = (sm >> 1) & 0xF;
    cond_bits(Cond::Al) | (0b11101 << 23) | (d << 22) | (0b110111 << 16) | (vd << 12) | (0b1010 << 8) | (0b11 << 6) | (m << 5) | vm
}

/// VCVT.F32.F64 Sd, Dm (double to single)
#[inline]
pub fn vcvt_s_d(sd: u32, dm: u32) -> u32 {
    let d = sd & 1;
    let vd = (sd >> 1) & 0xF;
    let m = (dm >> 4) & 1;
    let vm = dm & 0xF;
    cond_bits(Cond::Al) | (0b11101 << 23) | (d << 22) | (0b110111 << 16) | (vd << 12) | (0b1011 << 8) | (0b11 << 6) | (m << 5) | vm
}

/// VCVT.S32.F64 Sd, Dm (double to signed int, round toward zero)
#[inline]
pub fn vcvt_s32_d(sd: u32, dm: u32) -> u32 {
    let d = sd & 1;
    let vd = (sd >> 1) & 0xF;
    let m = (dm >> 4) & 1;
    let vm = dm & 0xF;
    cond_bits(Cond::Al) | (0b11101 << 23) | (d << 22) | (0b111101 << 16) | (vd << 12) | (0b1011 << 8) | (0b11 << 6) | (m << 5) | vm
}

/// VCVT.U32.F64 Sd, Dm (double to unsigned int, round toward zero)
#[inline]
pub fn vcvt_u32_d(sd: u32, dm: u32) -> u32 {
    let d = sd & 1;
    let vd = (sd >> 1) & 0xF;
    let m = (dm >> 4) & 1;
    let vm = dm & 0xF;
    cond_bits(Cond::Al) | (0b11101 << 23) | (d << 22) | (0b111100 << 16) | (vd << 12) | (0b1011 << 8) | (0b11 << 6) | (m << 5) | vm
}

/// VCVT.S32.F32 Sd, Sm (single to signed int, round toward zero)
#[inline]
pub fn vcvt_s32_s(sd: u32, sm: u32) -> u32 {
    let d = sd & 1;
    let vd = (sd >> 1) & 0xF;
    let m = sm & 1;
    let vm = (sm >> 1) & 0xF;
    cond_bits(Cond::Al) | (0b11101 << 23) | (d << 22) | (0b111101 << 16) | (vd << 12) | (0b1010 << 8) | (0b11 << 6) | (m << 5) | vm
}

/// VCVT.U32.F32 Sd, Sm (single to unsigned int, round toward zero)
#[inline]
pub fn vcvt_u32_s(sd: u32, sm: u32) -> u32 {
    let d = sd & 1;
    let vd = (sd >> 1) & 0xF;
    let m = sm & 1;
    let vm = (sm >> 1) & 0xF;
    cond_bits(Cond::Al) | (0b11101 << 23) | (d << 22) | (0b111100 << 16) | (vd << 12) | (0b1010 << 8) | (0b11 << 6) | (m << 5) | vm
}

/// VCVT.F64.S32 Dd, Sm (signed int to double)
#[inline]
pub fn vcvt_d_s32(dd: u32, sm: u32) -> u32 {
    let d = (dd >> 4) & 1;
    let vd = dd & 0xF;
    let m = sm & 1;
    let vm = (sm >> 1) & 0xF;
    cond_bits(Cond::Al) | (0b11101 << 23) | (d << 22) | (0b111000 << 16) | (vd << 12) | (0b1011 << 8) | (0b01 << 6) | (m << 5) | vm
}

/// VCVT.F64.U32 Dd, Sm (unsigned int to double)
#[inline]
pub fn vcvt_d_u32(dd: u32, sm: u32) -> u32 {
    let d = (dd >> 4) & 1;
    let vd = dd & 0xF;
    let m = sm & 1;
    let vm = (sm >> 1) & 0xF;
    cond_bits(Cond::Al) | (0b11101 << 23) | (d << 22) | (0b111000 << 16) | (vd << 12) | (0b1011 << 8) | (0b11 << 6) | (m << 5) | vm
}

/// VCVT.F32.S32 Sd, Sm (signed int to single)
#[inline]
pub fn vcvt_s_s32(sd: u32, sm: u32) -> u32 {
    let d = sd & 1;
    let vd = (sd >> 1) & 0xF;
    let m = sm & 1;
    let vm = (sm >> 1) & 0xF;
    cond_bits(Cond::Al) | (0b11101 << 23) | (d << 22) | (0b111000 << 16) | (vd << 12) | (0b1010 << 8) | (0b01 << 6) | (m << 5) | vm
}

/// VCVT.F32.U32 Sd, Sm (unsigned int to single)
#[inline]
pub fn vcvt_s_u32(sd: u32, sm: u32) -> u32 {
    let d = sd & 1;
    let vd = (sd >> 1) & 0xF;
    let m = sm & 1;
    let vm = (sm >> 1) & 0xF;
    cond_bits(Cond::Al) | (0b11101 << 23) | (d << 22) | (0b111000 << 16) | (vd << 12) | (0b1010 << 8) | (0b11 << 6) | (m << 5) | vm
}

/// Conditional data-processing with immediate.
#[inline]
pub fn dp_imm_cond(cond: Cond, opcode: u32, s: bool, dst: Arm32Reg, lhs: Arm32Reg, imm8: u32, rotate: u32) -> u32 {
    dp_imm(cond, opcode, s, dst, lhs, imm8, rotate)
}

/// NOP (ARMv6K+: hint encoding)
#[inline]
pub fn nop() -> u32 {
    cond_bits(Cond::Al) | (0b0011_0010_0000_1111_0000_0000_0000_0000)
}

/// BKPT #imm (software breakpoint)
#[inline]
pub fn bkpt(imm: u16) -> u32 {
    let hi = ((imm as u32) >> 4) & 0xFFF;
    let lo = (imm as u32) & 0xF;
    cond_bits(Cond::Al) | (0b00010010 << 20) | (hi << 8) | (0b0111 << 4) | lo
}

/// VPUSH {Dd-D(d+n-1)} (push consecutive D registers)
#[inline]
pub fn vpush_d(first_d: u32, count: u32) -> u32 {
    let d = (first_d >> 4) & 1;
    let vd = first_d & 0xF;
    cond_bits(Cond::Al) | (0b11010 << 23) | (d << 22) | (0b101101 << 16) | (vd << 12) | (0b1011 << 8) | (count * 2)
}

/// VPOP {Dd-D(d+n-1)} (pop consecutive D registers)
#[inline]
pub fn vpop_d(first_d: u32, count: u32) -> u32 {
    let d = (first_d >> 4) & 1;
    let vd = first_d & 0xF;
    cond_bits(Cond::Al) | (0b11001 << 23) | (d << 22) | (0b111101 << 16) | (vd << 12) | (0b1011 << 8) | (count * 2)
}
