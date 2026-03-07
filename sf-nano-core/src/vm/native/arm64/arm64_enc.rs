//! ARM64 instruction encoder for the micro-JIT.
//!
//! Pure functions: each takes register operands + immediates, returns a `u32`.
//! No mmap, no execution — pure data transformation.

use super::reg::Reg;

// ---------------------------------------------------------------------------
// Condition codes
// ---------------------------------------------------------------------------

/// ARM64 condition code for conditional branches and conditional select.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Cond {
    EQ = 0x0,
    NE = 0x1,
    HS = 0x2,
    LO = 0x3,
    MI = 0x4,
    PL = 0x5,
    VS = 0x6,
    VC = 0x7,
    HI = 0x8,
    LS = 0x9,
    GE = 0xA,
    LT = 0xB,
    GT = 0xC,
    LE = 0xD,
    AL = 0xE,
}

impl Cond {
    /// Invert condition (flip lowest bit). Needed for cset encoding.
    pub const fn invert(self) -> Cond {
        // SAFETY: all valid Cond values XOR 1 produce valid Cond values
        unsafe { core::mem::transmute(self as u8 ^ 1) }
    }
}

// ---------------------------------------------------------------------------
// Data Processing — Register (shifted register, no shift)
// ---------------------------------------------------------------------------

// Add/Sub shifted register:
// sf(1) op(1) S(1) 01011 shift(2) 0 Rm(5) imm6(6) Rn(5) Rd(5)
// shift=00, imm6=000000

fn add_sub_shifted_reg(sf: u32, op: u32, s: u32, rd: Reg, rn: Reg, rm: Reg) -> u32 {
    (sf << 31)
        | (op << 30)
        | (s << 29)
        | (0b01011 << 24)
        // shift=00, 0
        | (rm.idx() << 16)
        // imm6=000000
        | (rn.idx() << 5)
        | rd.idx()
}

/// ADD Wd, Wn, Wm
pub fn add_reg_32(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    add_sub_shifted_reg(0, 0, 0, rd, rn, rm)
}

/// ADD Xd, Xn, Xm
pub fn add_reg_64(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    add_sub_shifted_reg(1, 0, 0, rd, rn, rm)
}

/// SUB Wd, Wn, Wm
pub fn sub_reg_32(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    add_sub_shifted_reg(0, 1, 0, rd, rn, rm)
}

/// SUB Xd, Xn, Xm
pub fn sub_reg_64(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    add_sub_shifted_reg(1, 1, 0, rd, rn, rm)
}

/// SUBS Wd, Wn, Wm (sets flags)
pub fn subs_reg_32(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    add_sub_shifted_reg(0, 1, 1, rd, rn, rm)
}

/// SUBS Xd, Xn, Xm (sets flags)
pub fn subs_reg_64(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    add_sub_shifted_reg(1, 1, 1, rd, rn, rm)
}

// Logical shifted register:
// sf(1) opc(2) 01010 shift(2) N(1) Rm(5) imm6(6) Rn(5) Rd(5)
// shift=00, imm6=000000

fn logical_shifted_reg(sf: u32, opc: u32, n: u32, rd: Reg, rn: Reg, rm: Reg) -> u32 {
    (sf << 31)
        | (opc << 29)
        | (0b01010 << 24)
        // shift=00
        | (n << 21)
        | (rm.idx() << 16)
        // imm6=000000
        | (rn.idx() << 5)
        | rd.idx()
}

/// AND Wd, Wn, Wm
pub fn and_reg_32(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    logical_shifted_reg(0, 0b00, 0, rd, rn, rm)
}

/// AND Xd, Xn, Xm
pub fn and_reg_64(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    logical_shifted_reg(1, 0b00, 0, rd, rn, rm)
}

/// ORR Wd, Wn, Wm
pub fn orr_reg_32(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    logical_shifted_reg(0, 0b01, 0, rd, rn, rm)
}

/// ORR Xd, Xn, Xm
pub fn orr_reg_64(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    logical_shifted_reg(1, 0b01, 0, rd, rn, rm)
}

/// EOR Wd, Wn, Wm
pub fn eor_reg_32(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    logical_shifted_reg(0, 0b10, 0, rd, rn, rm)
}

/// EOR Xd, Xn, Xm
pub fn eor_reg_64(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    logical_shifted_reg(1, 0b10, 0, rd, rn, rm)
}

/// ORN Wd, Wn, Wm
pub fn orn_reg_32(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    logical_shifted_reg(0, 0b01, 1, rd, rn, rm)
}

/// ORN Xd, Xn, Xm
pub fn orn_reg_64(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    logical_shifted_reg(1, 0b01, 1, rd, rn, rm)
}

// ---------------------------------------------------------------------------
// Multiply / Divide
// ---------------------------------------------------------------------------

// MUL = MADD rd, rn, rm, xzr: sf 00 11011 000 Rm 0 Ra(=11111) Rn Rd
fn madd(sf: u32, rd: Reg, rn: Reg, rm: Reg, ra: Reg) -> u32 {
    (sf << 31)
        | (0b00_11011_000 << 21)
        | (rm.idx() << 16)
        | (0 << 15) // o0=0 for MADD
        | (ra.idx() << 10)
        | (rn.idx() << 5)
        | rd.idx()
}

fn msub_enc(sf: u32, rd: Reg, rn: Reg, rm: Reg, ra: Reg) -> u32 {
    (sf << 31)
        | (0b00_11011_000 << 21)
        | (rm.idx() << 16)
        | (1 << 15) // o0=1 for MSUB
        | (ra.idx() << 10)
        | (rn.idx() << 5)
        | rd.idx()
}

/// MUL Wd, Wn, Wm (= MADD Wd, Wn, Wm, WZR)
pub fn mul_32(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    madd(0, rd, rn, rm, Reg::XZR)
}

/// MUL Xd, Xn, Xm (= MADD Xd, Xn, Xm, XZR)
pub fn mul_64(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    madd(1, rd, rn, rm, Reg::XZR)
}

/// MSUB Wd, Wn, Wm, Wa
pub fn msub_32(rd: Reg, rn: Reg, rm: Reg, ra: Reg) -> u32 {
    msub_enc(0, rd, rn, rm, ra)
}

// Div: sf 00 11010 110 Rm 00001 o1 Rn Rd
// SDIV: o1=1, UDIV: o1=0

fn div_enc(sf: u32, o1: u32, rd: Reg, rn: Reg, rm: Reg) -> u32 {
    (sf << 31)
        | (0b00_11010_110 << 21)
        | (rm.idx() << 16)
        | (0b00001 << 11)
        | (o1 << 10)
        | (rn.idx() << 5)
        | rd.idx()
}

/// SDIV Wd, Wn, Wm
pub fn sdiv_32(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    div_enc(0, 1, rd, rn, rm)
}

/// UDIV Wd, Wn, Wm
pub fn udiv_32(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    div_enc(0, 0, rd, rn, rm)
}

/// SDIV Xd, Xn, Xm
pub fn sdiv_64(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    div_enc(1, 1, rd, rn, rm)
}

/// UDIV Xd, Xn, Xm
pub fn udiv_64(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    div_enc(1, 0, rd, rn, rm)
}

// ---------------------------------------------------------------------------
// Variable Shifts
// ---------------------------------------------------------------------------

// sf 00 11010 110 Rm 0010 op2(2) Rn Rd
// LSLV: op2=00, LSRV: op2=01, ASRV: op2=10, RORV: op2=11

fn shift_var(sf: u32, op2: u32, rd: Reg, rn: Reg, rm: Reg) -> u32 {
    (sf << 31)
        | (0b00_11010_110 << 21)
        | (rm.idx() << 16)
        | (0b0010 << 12)
        | (op2 << 10)
        | (rn.idx() << 5)
        | rd.idx()
}

/// LSLV Wd, Wn, Wm
pub fn lslv_32(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    shift_var(0, 0b00, rd, rn, rm)
}

/// LSRV Wd, Wn, Wm
pub fn lsrv_32(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    shift_var(0, 0b01, rd, rn, rm)
}

/// ASRV Wd, Wn, Wm
pub fn asrv_32(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    shift_var(0, 0b10, rd, rn, rm)
}

/// RORV Wd, Wn, Wm
pub fn rorv_32(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    shift_var(0, 0b11, rd, rn, rm)
}

/// LSLV Xd, Xn, Xm
pub fn lslv_64(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    shift_var(1, 0b00, rd, rn, rm)
}

/// LSRV Xd, Xn, Xm
pub fn lsrv_64(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    shift_var(1, 0b01, rd, rn, rm)
}

/// ASRV Xd, Xn, Xm
pub fn asrv_64(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    shift_var(1, 0b10, rd, rn, rm)
}

/// RORV Xd, Xn, Xm
pub fn rorv_64(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    shift_var(1, 0b11, rd, rn, rm)
}

// ---------------------------------------------------------------------------
// CLZ / RBIT
// ---------------------------------------------------------------------------

// CLZ: sf 10 11010 110 00000 00010 0 Rn Rd
// RBIT: sf 10 11010 110 00000 00000 0 Rn Rd

/// CLZ Wd, Wn
pub fn clz_32(rd: Reg, rn: Reg) -> u32 {
    (0b0_10_11010_110 << 21)
        | (0b00000 << 16)
        | (0b00010_0 << 10)
        | (rn.idx() << 5)
        | rd.idx()
}

/// CLZ Xd, Xn
pub fn clz_64(rd: Reg, rn: Reg) -> u32 {
    (0b1_10_11010_110 << 21)
        | (0b00000 << 16)
        | (0b00010_0 << 10)
        | (rn.idx() << 5)
        | rd.idx()
}

/// RBIT Wd, Wn
pub fn rbit_32(rd: Reg, rn: Reg) -> u32 {
    (0b0_10_11010_110 << 21)
        | (0b00000 << 16)
        | (0b00000_0 << 10)
        | (rn.idx() << 5)
        | rd.idx()
}

/// RBIT Xd, Xn
pub fn rbit_64(rd: Reg, rn: Reg) -> u32 {
    (0b1_10_11010_110 << 21)
        | (0b00000 << 16)
        | (0b00000_0 << 10)
        | (rn.idx() << 5)
        | rd.idx()
}

// ---------------------------------------------------------------------------
// Data Processing — Immediate (add/sub)
// ---------------------------------------------------------------------------

// sf(1) op(1) S(1) 100010 sh(1) imm12(12) Rn(5) Rd(5)
// sh=0 for unshifted

fn add_sub_imm(sf: u32, op: u32, s: u32, rd: Reg, rn: Reg, imm12: u32) -> u32 {
    debug_assert!(imm12 < 0x1000, "imm12 out of range: {:#x}", imm12);
    (sf << 31)
        | (op << 30)
        | (s << 29)
        | (0b100010 << 23)
        // sh=0
        | (imm12 << 10)
        | (rn.idx() << 5)
        | rd.idx()
}

/// ADD Wd, Wn, #imm12
pub fn add_imm_32(rd: Reg, rn: Reg, imm12: u32) -> u32 {
    add_sub_imm(0, 0, 0, rd, rn, imm12)
}

/// ADD Xd, Xn, #imm12
pub fn add_imm_64(rd: Reg, rn: Reg, imm12: u32) -> u32 {
    add_sub_imm(1, 0, 0, rd, rn, imm12)
}

/// SUB Wd, Wn, #imm12
pub fn sub_imm_32(rd: Reg, rn: Reg, imm12: u32) -> u32 {
    add_sub_imm(0, 1, 0, rd, rn, imm12)
}

/// SUB Xd, Xn, #imm12
pub fn sub_imm_64(rd: Reg, rn: Reg, imm12: u32) -> u32 {
    add_sub_imm(1, 1, 0, rd, rn, imm12)
}

/// SUBS Wd, Wn, #imm12 (sets flags)
pub fn subs_imm_32(rd: Reg, rn: Reg, imm12: u32) -> u32 {
    add_sub_imm(0, 1, 1, rd, rn, imm12)
}

/// SUBS Xd, Xn, #imm12 (sets flags)
pub fn subs_imm_64(rd: Reg, rn: Reg, imm12: u32) -> u32 {
    add_sub_imm(1, 1, 1, rd, rn, imm12)
}

// ---------------------------------------------------------------------------
// Helper aliases (CMP, NEG, MVN)
// ---------------------------------------------------------------------------

/// CMP Wn, Wm (= SUBS WZR, Wn, Wm)
pub fn cmp_reg_32(rn: Reg, rm: Reg) -> u32 {
    subs_reg_32(Reg::XZR, rn, rm)
}

/// CMP Xn, Xm (= SUBS XZR, Xn, Xm)
pub fn cmp_reg_64(rn: Reg, rm: Reg) -> u32 {
    subs_reg_64(Reg::XZR, rn, rm)
}

/// CMP Wn, #imm12 (= SUBS WZR, Wn, #imm12)
pub fn cmp_imm_32(rn: Reg, imm12: u32) -> u32 {
    subs_imm_32(Reg::XZR, rn, imm12)
}

/// CMP Xn, #imm12 (= SUBS XZR, Xn, #imm12)
pub fn cmp_imm_64(rn: Reg, imm12: u32) -> u32 {
    subs_imm_64(Reg::XZR, rn, imm12)
}

/// NEG Wd, Wm (= SUB Wd, WZR, Wm)
pub fn neg_32(rd: Reg, rm: Reg) -> u32 {
    sub_reg_32(rd, Reg::XZR, rm)
}

/// NEG Xd, Xm (= SUB Xd, XZR, Xm)
pub fn neg_64(rd: Reg, rm: Reg) -> u32 {
    sub_reg_64(rd, Reg::XZR, rm)
}

/// MVN Wd, Wm (= ORN Wd, WZR, Wm)
pub fn mvn_32(rd: Reg, rm: Reg) -> u32 {
    orn_reg_32(rd, Reg::XZR, rm)
}

/// MVN Xd, Xm (= ORN Xd, XZR, Xm)
pub fn mvn_64(rd: Reg, rm: Reg) -> u32 {
    orn_reg_64(rd, Reg::XZR, rm)
}

// ---------------------------------------------------------------------------
// Conditional Select
// ---------------------------------------------------------------------------

// sf(1) op(1) S(1) 11010100 Rm(5) cond(4) op2(2) Rn(5) Rd(5)
// CSEL:  op=0, S=0, op2=00
// CSINC: op=0, S=0, op2=01

fn cond_select(sf: u32, op2: u32, rd: Reg, rn: Reg, rm: Reg, cond: Cond) -> u32 {
    (sf << 31)
        | (0b0_0_11010100 << 21)
        | (rm.idx() << 16)
        | ((cond as u32) << 12)
        | (op2 << 10)
        | (rn.idx() << 5)
        | rd.idx()
}

/// CSEL Wd, Wn, Wm, cond
pub fn csel_32(rd: Reg, rn: Reg, rm: Reg, cond: Cond) -> u32 {
    cond_select(0, 0b00, rd, rn, rm, cond)
}

/// CSEL Xd, Xn, Xm, cond
pub fn csel_64(rd: Reg, rn: Reg, rm: Reg, cond: Cond) -> u32 {
    cond_select(1, 0b00, rd, rn, rm, cond)
}

/// CSINC Wd, Wn, Wm, cond
pub fn csinc_32(rd: Reg, rn: Reg, rm: Reg, cond: Cond) -> u32 {
    cond_select(0, 0b01, rd, rn, rm, cond)
}

/// CSINC Xd, Xn, Xm, cond
pub fn csinc_64(rd: Reg, rn: Reg, rm: Reg, cond: Cond) -> u32 {
    cond_select(1, 0b01, rd, rn, rm, cond)
}

/// CSET Wd, cond (= CSINC Wd, WZR, WZR, invert(cond))
pub fn cset_32(rd: Reg, cond: Cond) -> u32 {
    csinc_32(rd, Reg::XZR, Reg::XZR, cond.invert())
}

/// CSET Xd, cond (= CSINC Xd, XZR, XZR, invert(cond))
pub fn cset_64(rd: Reg, cond: Cond) -> u32 {
    csinc_64(rd, Reg::XZR, Reg::XZR, cond.invert())
}

// ---------------------------------------------------------------------------
// Move (register)
// ---------------------------------------------------------------------------

/// MOV Wd, Wm (= ORR Wd, WZR, Wm)
pub fn mov_reg_32(rd: Reg, rm: Reg) -> u32 {
    orr_reg_32(rd, Reg::XZR, rm)
}

/// MOV Xd, Xm (= ORR Xd, XZR, Xm)
pub fn mov_reg_64(rd: Reg, rm: Reg) -> u32 {
    orr_reg_64(rd, Reg::XZR, rm)
}

// ---------------------------------------------------------------------------
// Move wide immediate (MOVZ / MOVK / MOVN)
// ---------------------------------------------------------------------------

// sf(1) opc(2) 100101 hw(2) imm16(16) Rd(5)
// MOVZ: opc=10, MOVK: opc=11, MOVN: opc=00

fn move_wide(sf: u32, opc: u32, rd: Reg, imm16: u16, shift: u32) -> u32 {
    let hw = shift / 16;
    debug_assert!(
        if sf == 0 { hw <= 1 } else { hw <= 3 },
        "shift out of range for move wide: shift={shift}"
    );
    debug_assert!(shift % 16 == 0, "shift must be multiple of 16: {shift}");
    (sf << 31)
        | (opc << 29)
        | (0b100101 << 23)
        | (hw << 21)
        | ((imm16 as u32) << 5)
        | rd.idx()
}

/// MOVZ Wd, #imm16, LSL #shift
pub fn movz_32(rd: Reg, imm16: u16, shift: u32) -> u32 {
    move_wide(0, 0b10, rd, imm16, shift)
}

/// MOVZ Xd, #imm16, LSL #shift
pub fn movz_64(rd: Reg, imm16: u16, shift: u32) -> u32 {
    move_wide(1, 0b10, rd, imm16, shift)
}

/// MOVK Wd, #imm16, LSL #shift
pub fn movk_32(rd: Reg, imm16: u16, shift: u32) -> u32 {
    move_wide(0, 0b11, rd, imm16, shift)
}

/// MOVK Xd, #imm16, LSL #shift
pub fn movk_64(rd: Reg, imm16: u16, shift: u32) -> u32 {
    move_wide(1, 0b11, rd, imm16, shift)
}

/// MOVN Wd, #imm16, LSL #shift
pub fn movn_32(rd: Reg, imm16: u16, shift: u32) -> u32 {
    move_wide(0, 0b00, rd, imm16, shift)
}

/// MOVN Xd, #imm16, LSL #shift
pub fn movn_64(rd: Reg, imm16: u16, shift: u32) -> u32 {
    move_wide(1, 0b00, rd, imm16, shift)
}

// ---------------------------------------------------------------------------
// Load / Store — Unsigned Offset
// ---------------------------------------------------------------------------

// size(2) 111 V(1) 01 opc(2) imm12(12) Rn(5) Rt(5)
// V=0 for GP registers
// imm12 is pre-scaled (caller divides by access size)

fn ldst_unsigned_offset(size: u32, opc: u32, rt: Reg, rn: Reg, imm12: u32) -> u32 {
    debug_assert!(imm12 < 0x1000, "imm12 out of range: {:#x}", imm12);
    (size << 30)
        | (0b111_0_01 << 24)
        | (opc << 22)
        | (imm12 << 10)
        | (rn.idx() << 5)
        | rt.idx()
}

fn fp_ldst_unsigned_offset(size: u32, opc: u32, vt: u32, rn: Reg, imm12: u32) -> u32 {
    debug_assert!(vt < 32, "FP register out of range: {vt}");
    debug_assert!(imm12 < 0x1000, "imm12 out of range: {:#x}", imm12);
    (size << 30)
        | (0b111_1_01 << 24)
        | (opc << 22)
        | (imm12 << 10)
        | (rn.idx() << 5)
        | vt
}

/// LDR Xt, [Xn, #imm12*8] (64-bit load, pre-scaled imm12)
pub fn ldr_64(rt: Reg, rn: Reg, imm12: u32) -> u32 {
    ldst_unsigned_offset(0b11, 0b01, rt, rn, imm12)
}

/// STR Xt, [Xn, #imm12*8] (64-bit store, pre-scaled imm12)
pub fn str_64(rt: Reg, rn: Reg, imm12: u32) -> u32 {
    ldst_unsigned_offset(0b11, 0b00, rt, rn, imm12)
}

/// LDR Wt, [Xn, #imm12*4] (32-bit load, pre-scaled imm12)
pub fn ldr_32(rt: Reg, rn: Reg, imm12: u32) -> u32 {
    ldst_unsigned_offset(0b10, 0b01, rt, rn, imm12)
}

/// STR Wt, [Xn, #imm12*4] (32-bit store, pre-scaled imm12)
pub fn str_32(rt: Reg, rn: Reg, imm12: u32) -> u32 {
    ldst_unsigned_offset(0b10, 0b00, rt, rn, imm12)
}

/// LDR St, [Xn, #imm12*4] (32-bit floating-point load, pre-scaled imm12)
pub fn ldr_f32(vt: u32, rn: Reg, imm12: u32) -> u32 {
    fp_ldst_unsigned_offset(0b10, 0b01, vt, rn, imm12)
}

/// STR St, [Xn, #imm12*4] (32-bit floating-point store, pre-scaled imm12)
pub fn str_f32(vt: u32, rn: Reg, imm12: u32) -> u32 {
    fp_ldst_unsigned_offset(0b10, 0b00, vt, rn, imm12)
}

/// LDR Dt, [Xn, #imm12*8] (64-bit floating-point load, pre-scaled imm12)
pub fn ldr_f64(vt: u32, rn: Reg, imm12: u32) -> u32 {
    fp_ldst_unsigned_offset(0b11, 0b01, vt, rn, imm12)
}

/// STR Dt, [Xn, #imm12*8] (64-bit floating-point store, pre-scaled imm12)
pub fn str_f64(vt: u32, rn: Reg, imm12: u32) -> u32 {
    fp_ldst_unsigned_offset(0b11, 0b00, vt, rn, imm12)
}

/// LDRH Wt, [Xn, #imm12*2] (16-bit load, pre-scaled imm12)
pub fn ldrh(rt: Reg, rn: Reg, imm12: u32) -> u32 {
    ldst_unsigned_offset(0b01, 0b01, rt, rn, imm12)
}

/// STRH Wt, [Xn, #imm12*2] (16-bit store, pre-scaled imm12)
pub fn strh(rt: Reg, rn: Reg, imm12: u32) -> u32 {
    ldst_unsigned_offset(0b01, 0b00, rt, rn, imm12)
}

/// LDRB Wt, [Xn, #imm12] (8-bit load, no scaling)
pub fn ldrb(rt: Reg, rn: Reg, imm12: u32) -> u32 {
    ldst_unsigned_offset(0b00, 0b01, rt, rn, imm12)
}

/// STRB Wt, [Xn, #imm12] (8-bit store, no scaling)
pub fn strb(rt: Reg, rn: Reg, imm12: u32) -> u32 {
    ldst_unsigned_offset(0b00, 0b00, rt, rn, imm12)
}

/// LDRSW Xt, [Xn, #imm12*4] (sign-extend 32→64, pre-scaled imm12)
pub fn ldrsw(rt: Reg, rn: Reg, imm12: u32) -> u32 {
    ldst_unsigned_offset(0b10, 0b10, rt, rn, imm12)
}

/// LDRSH Wt, [Xn, #imm12*2] (sign-extend 16→32, pre-scaled imm12)
pub fn ldrsh_32(rt: Reg, rn: Reg, imm12: u32) -> u32 {
    ldst_unsigned_offset(0b01, 0b11, rt, rn, imm12)
}

/// LDRSB Wt, [Xn, #imm12] (sign-extend 8→32, no scaling)
pub fn ldrsb_32(rt: Reg, rn: Reg, imm12: u32) -> u32 {
    ldst_unsigned_offset(0b00, 0b11, rt, rn, imm12)
}

// ---------------------------------------------------------------------------
// Load / Store — Register Offset
// ---------------------------------------------------------------------------

// size(2) 111 V(1) 00 opc(2) 1 Rm(5) option(3) S(1) 10 Rn(5) Rt(5)
// V=0, option=011 (LSL), S=0 (no shift) → [Xn, Xm]

fn ldst_reg_offset(size: u32, opc: u32, rt: Reg, rn: Reg, rm: Reg) -> u32 {
    (size << 30)
        | (0b111_0_00 << 24)
        | (opc << 22)
        | (1 << 21)
        | (rm.idx() << 16)
        | (0b011 << 13) // option=LSL
        | (0 << 12)     // S=0
        | (0b10 << 10)
        | (rn.idx() << 5)
        | rt.idx()
}

/// LDR Xt, [Xn, Xm] (64-bit load, register offset)
pub fn ldr_64_reg(rt: Reg, rn: Reg, rm: Reg) -> u32 {
    ldst_reg_offset(0b11, 0b01, rt, rn, rm)
}

/// STR Xt, [Xn, Xm] (64-bit store, register offset)
pub fn str_64_reg(rt: Reg, rn: Reg, rm: Reg) -> u32 {
    ldst_reg_offset(0b11, 0b00, rt, rn, rm)
}

/// LDR Wt, [Xn, Xm] (32-bit load, register offset)
pub fn ldr_32_reg(rt: Reg, rn: Reg, rm: Reg) -> u32 {
    ldst_reg_offset(0b10, 0b01, rt, rn, rm)
}

/// STR Wt, [Xn, Xm] (32-bit store, register offset)
pub fn str_32_reg(rt: Reg, rn: Reg, rm: Reg) -> u32 {
    ldst_reg_offset(0b10, 0b00, rt, rn, rm)
}

/// LDRH Wt, [Xn, Xm] (16-bit load, register offset)
pub fn ldrh_reg(rt: Reg, rn: Reg, rm: Reg) -> u32 {
    ldst_reg_offset(0b01, 0b01, rt, rn, rm)
}

/// STRH Wt, [Xn, Xm] (16-bit store, register offset)
pub fn strh_reg(rt: Reg, rn: Reg, rm: Reg) -> u32 {
    ldst_reg_offset(0b01, 0b00, rt, rn, rm)
}

/// LDRB Wt, [Xn, Xm] (8-bit load, register offset)
pub fn ldrb_reg(rt: Reg, rn: Reg, rm: Reg) -> u32 {
    ldst_reg_offset(0b00, 0b01, rt, rn, rm)
}

/// STRB Wt, [Xn, Xm] (8-bit store, register offset)
pub fn strb_reg(rt: Reg, rn: Reg, rm: Reg) -> u32 {
    ldst_reg_offset(0b00, 0b00, rt, rn, rm)
}

fn fp_ldst_reg_offset(size: u32, opc: u32, vt: u32, rn: Reg, rm: Reg) -> u32 {
    debug_assert!(vt < 32, "FP register out of range: {vt}");
    (size << 30)
        | (0b111_1_00 << 24)
        | (opc << 22)
        | (1 << 21)
        | (rm.idx() << 16)
        | (0b011 << 13)
        | (0 << 12)
        | (0b10 << 10)
        | (rn.idx() << 5)
        | vt
}

/// LDR St, [Xn, Xm] (32-bit floating-point load, register offset)
pub fn ldr_f32_reg(vt: u32, rn: Reg, rm: Reg) -> u32 {
    fp_ldst_reg_offset(0b10, 0b01, vt, rn, rm)
}

/// STR St, [Xn, Xm] (32-bit floating-point store, register offset)
pub fn str_f32_reg(vt: u32, rn: Reg, rm: Reg) -> u32 {
    fp_ldst_reg_offset(0b10, 0b00, vt, rn, rm)
}

/// LDR Dt, [Xn, Xm] (64-bit floating-point load, register offset)
pub fn ldr_f64_reg(vt: u32, rn: Reg, rm: Reg) -> u32 {
    fp_ldst_reg_offset(0b11, 0b01, vt, rn, rm)
}

/// STR Dt, [Xn, Xm] (64-bit floating-point store, register offset)
pub fn str_f64_reg(vt: u32, rn: Reg, rm: Reg) -> u32 {
    fp_ldst_reg_offset(0b11, 0b00, vt, rn, rm)
}

// ---------------------------------------------------------------------------
// Sign/Zero Extension (Bitfield operations)
// ---------------------------------------------------------------------------

// SBFM: sf 00 100110 N immr(6) imms(6) Rn(5) Rd(5)
// SXTW: sbfm xd, xn, #0, #31 → sf=1, N=1, immr=000000, imms=011111
// SXTH: sbfm wd, wn, #0, #15 → sf=0, N=0, immr=000000, imms=001111
// SXTB: sbfm wd, wn, #0, #7  → sf=0, N=0, immr=000000, imms=000111

fn sbfm(sf: u32, n: u32, rd: Reg, rn: Reg, immr: u32, imms: u32) -> u32 {
    debug_assert!(immr < 64 && imms < 64);
    (sf << 31)
        | (0b00 << 29)
        | (0b100110 << 23)
        | (n << 22)
        | (immr << 16)
        | (imms << 10)
        | (rn.idx() << 5)
        | rd.idx()
}

// UBFM: sf 10 100110 N immr(6) imms(6) Rn(5) Rd(5)

fn ubfm(sf: u32, n: u32, rd: Reg, rn: Reg, immr: u32, imms: u32) -> u32 {
    debug_assert!(immr < 64 && imms < 64);
    (sf << 31)
        | (0b10 << 29)
        | (0b100110 << 23)
        | (n << 22)
        | (immr << 16)
        | (imms << 10)
        | (rn.idx() << 5)
        | rd.idx()
}

/// SXTW Xd, Wn (sign-extend 32→64)
pub fn sxtw(rd: Reg, rn: Reg) -> u32 {
    sbfm(1, 1, rd, rn, 0, 31)
}

/// SXTH Wd, Wn (sign-extend 16→32)
pub fn sxth_32(rd: Reg, rn: Reg) -> u32 {
    sbfm(0, 0, rd, rn, 0, 15)
}

/// SXTB Wd, Wn (sign-extend 8→32)
pub fn sxtb_32(rd: Reg, rn: Reg) -> u32 {
    sbfm(0, 0, rd, rn, 0, 7)
}

/// SXTB Xd, Xn (sign-extend 8→64)
pub fn sxtb_64(rd: Reg, rn: Reg) -> u32 {
    sbfm(1, 1, rd, rn, 0, 7)
}

/// SXTH Xd, Xn (sign-extend 16→64)
pub fn sxth_64(rd: Reg, rn: Reg) -> u32 {
    sbfm(1, 1, rd, rn, 0, 15)
}

/// UXTH Wd, Wn (zero-extend 16→32, = UBFM Wd, Wn, #0, #15)
pub fn uxth_32(rd: Reg, rn: Reg) -> u32 {
    ubfm(0, 0, rd, rn, 0, 15)
}

/// UXTB Wd, Wn (zero-extend 8→32, = UBFM Wd, Wn, #0, #7)
pub fn uxtb_32(rd: Reg, rn: Reg) -> u32 {
    ubfm(0, 0, rd, rn, 0, 7)
}

/// LSL Wd, Wn, #shift (= UBFM Wd, Wn, #(-shift mod 32), #(31-shift))
pub fn lsl_imm_32(rd: Reg, rn: Reg, shift: u32) -> u32 {
    debug_assert!(shift < 32, "shift out of range: {shift}");
    let immr = (32 - shift) & 0x1F; // -shift mod 32
    let imms = 31 - shift;
    ubfm(0, 0, rd, rn, immr, imms)
}

/// LSR Wd, Wn, #shift (= UBFM Wd, Wn, #shift, #31)
pub fn lsr_imm_32(rd: Reg, rn: Reg, shift: u32) -> u32 {
    debug_assert!(shift < 32, "shift out of range: {shift}");
    ubfm(0, 0, rd, rn, shift, 31)
}

/// ASR Wd, Wn, #shift (= SBFM Wd, Wn, #shift, #31)
pub fn asr_imm_32(rd: Reg, rn: Reg, shift: u32) -> u32 {
    debug_assert!(shift < 32, "shift out of range: {shift}");
    sbfm(0, 0, rd, rn, shift, 31)
}

/// LSL Xd, Xn, #shift (= UBFM Xd, Xn, #(-shift mod 64), #(63-shift))
pub fn lsl_imm_64(rd: Reg, rn: Reg, shift: u32) -> u32 {
    debug_assert!(shift < 64, "shift out of range: {shift}");
    let immr = (64 - shift) & 0x3F;
    let imms = 63 - shift;
    ubfm(1, 1, rd, rn, immr, imms)
}

/// LSR Xd, Xn, #shift (= UBFM Xd, Xn, #shift, #63)
pub fn lsr_imm_64(rd: Reg, rn: Reg, shift: u32) -> u32 {
    debug_assert!(shift < 64, "shift out of range: {shift}");
    ubfm(1, 1, rd, rn, shift, 63)
}

/// ASR Xd, Xn, #shift (= SBFM Xd, Xn, #shift, #63)
pub fn asr_imm_64(rd: Reg, rn: Reg, shift: u32) -> u32 {
    debug_assert!(shift < 64, "shift out of range: {shift}");
    sbfm(1, 1, rd, rn, shift, 63)
}

// ---------------------------------------------------------------------------
// Branch
// ---------------------------------------------------------------------------

/// B imm26 — unconditional branch (offset in instruction units, ±128MB)
pub fn b(imm26: i32) -> u32 {
    debug_assert!(
        imm26 >= -(1 << 25) && imm26 < (1 << 25),
        "imm26 out of range: {imm26}"
    );
    let imm26_bits = (imm26 as u32) & 0x03FF_FFFF;
    (0b000101 << 26) | imm26_bits
}

/// B.cond imm19 — conditional branch (offset in instruction units)
pub fn b_cond(cond: Cond, imm19: i32) -> u32 {
    debug_assert!(
        imm19 >= -(1 << 18) && imm19 < (1 << 18),
        "imm19 out of range: {imm19}"
    );
    let imm19_bits = (imm19 as u32) & 0x0007_FFFF;
    (0b01010100 << 24) | (imm19_bits << 5) | (cond as u32)
}

/// CBZ Xt, imm19 — compare and branch if zero (64-bit)
pub fn cbz_64(rt: Reg, imm19: i32) -> u32 {
    debug_assert!(
        imm19 >= -(1 << 18) && imm19 < (1 << 18),
        "imm19 out of range: {imm19}"
    );
    let imm19_bits = (imm19 as u32) & 0x0007_FFFF;
    (0b1_011010_0 << 24) | (imm19_bits << 5) | rt.idx()
}

/// CBNZ Xt, imm19 — compare and branch if not zero (64-bit)
pub fn cbnz_64(rt: Reg, imm19: i32) -> u32 {
    debug_assert!(
        imm19 >= -(1 << 18) && imm19 < (1 << 18),
        "imm19 out of range: {imm19}"
    );
    let imm19_bits = (imm19 as u32) & 0x0007_FFFF;
    (0b1_011010_1 << 24) | (imm19_bits << 5) | rt.idx()
}

/// CBZ Wt, imm19 — compare and branch if zero (32-bit)
pub fn cbz_32(rt: Reg, imm19: i32) -> u32 {
    debug_assert!(
        imm19 >= -(1 << 18) && imm19 < (1 << 18),
        "imm19 out of range: {imm19}"
    );
    let imm19_bits = (imm19 as u32) & 0x0007_FFFF;
    (0b0_011010_0 << 24) | (imm19_bits << 5) | rt.idx()
}

/// CBNZ Wt, imm19 — compare and branch if not zero (32-bit)
pub fn cbnz_32(rt: Reg, imm19: i32) -> u32 {
    debug_assert!(
        imm19 >= -(1 << 18) && imm19 < (1 << 18),
        "imm19 out of range: {imm19}"
    );
    let imm19_bits = (imm19 as u32) & 0x0007_FFFF;
    (0b0_011010_1 << 24) | (imm19_bits << 5) | rt.idx()
}

/// BR Xn — branch to register
pub fn br(rn: Reg) -> u32 {
    (0b1101011_0000 << 21)
        | (0b11111 << 16)
        | (0b000000 << 10)
        | (rn.idx() << 5)
        | 0b00000
}

/// RET (= RET X30)
pub fn ret() -> u32 {
    (0b1101011_0010 << 21)
        | (0b11111 << 16)
        | (0b000000 << 10)
        | (Reg::LR.idx() << 5)
        | 0b00000
}

/// NOP
pub fn nop() -> u32 {
    0xD503201F
}

// ---------------------------------------------------------------------------
// Floating-Point instructions
// ---------------------------------------------------------------------------

/// FP integer-conversion / FMOV helper.
/// Encoding: sf(1) 00 11110 type(2) 1 rmode(2) opcode(3) 000000 Rn(5) Rd(5)
fn fp_int_conv(sf: u32, type_: u32, rmode: u32, opcode: u32, rd: u32, rn: u32) -> u32 {
    (sf << 31) | (0x1E << 24) | (type_ << 22) | (1 << 21)
        | (rmode << 19) | (opcode << 16) | (rn << 5) | rd
}

/// FMOV Dd, Xn — GPR to FP register (64-bit bit transfer)
pub fn fmov_gpr_to_fp64(fd: u32, rn: Reg) -> u32 {
    fp_int_conv(1, 0b01, 0b00, 0b111, fd, rn.idx())
}

/// FMOV Xd, Dn — FP register to GPR (64-bit bit transfer)
pub fn fmov_fp64_to_gpr(rd: Reg, fn_: u32) -> u32 {
    fp_int_conv(1, 0b01, 0b00, 0b110, rd.idx(), fn_)
}

/// FMOV Sd, Wn — GPR to FP register (32-bit bit transfer)
pub fn fmov_gpr_to_fp32(fd: u32, rn: Reg) -> u32 {
    fp_int_conv(0, 0b00, 0b00, 0b111, fd, rn.idx())
}

/// FMOV Wd, Sn — FP register to GPR (32-bit bit transfer)
pub fn fmov_fp32_to_gpr(rd: Reg, fn_: u32) -> u32 {
    fp_int_conv(0, 0b00, 0b00, 0b110, rd.idx(), fn_)
}

// --- FP arithmetic (2-source) ---
// Encoding: 0 0 0 11110 type(2) 1 Rm(5) opcode(4) 10 Rn(5) Rd(5)

fn fp_binop_enc(type_: u32, opcode: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    (0b00011110 << 24) | (type_ << 22) | (1 << 21)
        | (rm << 16) | (opcode << 12) | (0b10 << 10) | (rn << 5) | rd
}

pub fn fadd_64(rd: u32, rn: u32, rm: u32) -> u32 { fp_binop_enc(0b01, 0b0010, rd, rn, rm) }
pub fn fsub_64(rd: u32, rn: u32, rm: u32) -> u32 { fp_binop_enc(0b01, 0b0011, rd, rn, rm) }
pub fn fmul_64(rd: u32, rn: u32, rm: u32) -> u32 { fp_binop_enc(0b01, 0b0000, rd, rn, rm) }
pub fn fdiv_64(rd: u32, rn: u32, rm: u32) -> u32 { fp_binop_enc(0b01, 0b0001, rd, rn, rm) }
pub fn fmax_64(rd: u32, rn: u32, rm: u32) -> u32 { fp_binop_enc(0b01, 0b0100, rd, rn, rm) }
pub fn fmin_64(rd: u32, rn: u32, rm: u32) -> u32 { fp_binop_enc(0b01, 0b0101, rd, rn, rm) }

pub fn fadd_32(rd: u32, rn: u32, rm: u32) -> u32 { fp_binop_enc(0b00, 0b0010, rd, rn, rm) }
pub fn fsub_32(rd: u32, rn: u32, rm: u32) -> u32 { fp_binop_enc(0b00, 0b0011, rd, rn, rm) }
pub fn fmul_32(rd: u32, rn: u32, rm: u32) -> u32 { fp_binop_enc(0b00, 0b0000, rd, rn, rm) }
pub fn fdiv_32(rd: u32, rn: u32, rm: u32) -> u32 { fp_binop_enc(0b00, 0b0001, rd, rn, rm) }
pub fn fmax_32(rd: u32, rn: u32, rm: u32) -> u32 { fp_binop_enc(0b00, 0b0100, rd, rn, rm) }
pub fn fmin_32(rd: u32, rn: u32, rm: u32) -> u32 { fp_binop_enc(0b00, 0b0101, rd, rn, rm) }

// --- FP unary (1-source) ---
// Encoding: 0 0 0 11110 type(2) 1 opcode(6) 10000 Rn(5) Rd(5)

fn fp_1src_enc(type_: u32, opcode: u32, rd: u32, rn: u32) -> u32 {
    (0b00011110 << 24) | (type_ << 22) | (1 << 21)
        | (opcode << 15) | (0b10000 << 10) | (rn << 5) | rd
}

pub fn fabs_64(rd: u32, rn: u32) -> u32 { fp_1src_enc(0b01, 0b000001, rd, rn) }
pub fn fneg_64(rd: u32, rn: u32) -> u32 { fp_1src_enc(0b01, 0b000010, rd, rn) }
pub fn fsqrt_64(rd: u32, rn: u32) -> u32 { fp_1src_enc(0b01, 0b000011, rd, rn) }
pub fn frintn_64(rd: u32, rn: u32) -> u32 { fp_1src_enc(0b01, 0b001000, rd, rn) }
pub fn frintp_64(rd: u32, rn: u32) -> u32 { fp_1src_enc(0b01, 0b001001, rd, rn) }
pub fn frintm_64(rd: u32, rn: u32) -> u32 { fp_1src_enc(0b01, 0b001010, rd, rn) }
pub fn frintz_64(rd: u32, rn: u32) -> u32 { fp_1src_enc(0b01, 0b001011, rd, rn) }

pub fn fabs_32(rd: u32, rn: u32) -> u32 { fp_1src_enc(0b00, 0b000001, rd, rn) }
pub fn fneg_32(rd: u32, rn: u32) -> u32 { fp_1src_enc(0b00, 0b000010, rd, rn) }
pub fn fsqrt_32(rd: u32, rn: u32) -> u32 { fp_1src_enc(0b00, 0b000011, rd, rn) }
pub fn frintn_32(rd: u32, rn: u32) -> u32 { fp_1src_enc(0b00, 0b001000, rd, rn) }
pub fn frintp_32(rd: u32, rn: u32) -> u32 { fp_1src_enc(0b00, 0b001001, rd, rn) }
pub fn frintm_32(rd: u32, rn: u32) -> u32 { fp_1src_enc(0b00, 0b001010, rd, rn) }
pub fn frintz_32(rd: u32, rn: u32) -> u32 { fp_1src_enc(0b00, 0b001011, rd, rn) }

/// FCVT Dd, Sn — f32 to f64 (promote)
pub fn fcvt_f32_to_f64(dd: u32, sn: u32) -> u32 { fp_1src_enc(0b00, 0b000101, dd, sn) }
/// FCVT Sd, Dn — f64 to f32 (demote)
pub fn fcvt_f64_to_f32(sd: u32, dn: u32) -> u32 { fp_1src_enc(0b01, 0b000100, sd, dn) }

// --- FP compare ---
// Encoding: 0 0 0 11110 type(2) 1 Rm(5) 00 1000 Rn(5) 00000

pub fn fcmp_64(fn_: u32, fm: u32) -> u32 {
    (0b00011110 << 24) | (0b01 << 22) | (1 << 21)
        | (fm << 16) | (0b001000 << 10) | (fn_ << 5)
}

pub fn fcmp_32(fn_: u32, fm: u32) -> u32 {
    (0b00011110 << 24) | (0b00 << 22) | (1 << 21)
        | (fm << 16) | (0b001000 << 10) | (fn_ << 5)
}

/// FCMP Dn, #0.0
pub fn fcmp_64_zero(fn_: u32) -> u32 {
    0x1E60_2008 | (fn_ << 5)
}

/// FCMP Sn, #0.0
pub fn fcmp_32_zero(fn_: u32) -> u32 {
    0x1E20_2008 | (fn_ << 5)
}

// --- Float-to-int conversion (saturating truncate toward zero) ---

pub fn fcvtzs_w_s(wd: Reg, sn: u32) -> u32 { fp_int_conv(0, 0b00, 0b11, 0b000, wd.idx(), sn) }
pub fn fcvtzs_w_d(wd: Reg, dn: u32) -> u32 { fp_int_conv(0, 0b01, 0b11, 0b000, wd.idx(), dn) }
pub fn fcvtzs_x_s(xd: Reg, sn: u32) -> u32 { fp_int_conv(1, 0b00, 0b11, 0b000, xd.idx(), sn) }
pub fn fcvtzs_x_d(xd: Reg, dn: u32) -> u32 { fp_int_conv(1, 0b01, 0b11, 0b000, xd.idx(), dn) }

pub fn fcvtzu_w_s(wd: Reg, sn: u32) -> u32 { fp_int_conv(0, 0b00, 0b11, 0b001, wd.idx(), sn) }
pub fn fcvtzu_w_d(wd: Reg, dn: u32) -> u32 { fp_int_conv(0, 0b01, 0b11, 0b001, wd.idx(), dn) }
pub fn fcvtzu_x_s(xd: Reg, sn: u32) -> u32 { fp_int_conv(1, 0b00, 0b11, 0b001, xd.idx(), sn) }
pub fn fcvtzu_x_d(xd: Reg, dn: u32) -> u32 { fp_int_conv(1, 0b01, 0b11, 0b001, xd.idx(), dn) }

// --- Int-to-float conversion ---

pub fn scvtf_s_w(sd: u32, wn: Reg) -> u32 { fp_int_conv(0, 0b00, 0b00, 0b010, sd, wn.idx()) }
pub fn scvtf_d_w(dd: u32, wn: Reg) -> u32 { fp_int_conv(0, 0b01, 0b00, 0b010, dd, wn.idx()) }
pub fn scvtf_s_x(sd: u32, xn: Reg) -> u32 { fp_int_conv(1, 0b00, 0b00, 0b010, sd, xn.idx()) }
pub fn scvtf_d_x(dd: u32, xn: Reg) -> u32 { fp_int_conv(1, 0b01, 0b00, 0b010, dd, xn.idx()) }

pub fn ucvtf_s_w(sd: u32, wn: Reg) -> u32 { fp_int_conv(0, 0b00, 0b00, 0b011, sd, wn.idx()) }
pub fn ucvtf_d_w(dd: u32, wn: Reg) -> u32 { fp_int_conv(0, 0b01, 0b00, 0b011, dd, wn.idx()) }
pub fn ucvtf_s_x(sd: u32, xn: Reg) -> u32 { fp_int_conv(1, 0b00, 0b00, 0b011, sd, xn.idx()) }
pub fn ucvtf_d_x(dd: u32, xn: Reg) -> u32 { fp_int_conv(1, 0b01, 0b00, 0b011, dd, xn.idx()) }

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::reg::Reg;

    // --- Add/Sub register ---

    #[test]
    fn test_add_reg_32() {
        // add w23, w23, w26
        assert_eq!(add_reg_32(Reg::L0, Reg::L0, Reg::T0), 0x0B1A02F7);
    }

    #[test]
    fn test_add_reg_32_different_regs() {
        // add w27, w26, w0
        // sf=0, op=0, S=0, 01011, shift=00, 0, Rm=00000, imm6=000000, Rn=11010, Rd=11011
        // 0_00_01011_00_0_00000_000000_11010_11011
        // = 0000_1011_0000_0000_0000_0011_0101_1011
        // = 0x0B0003_5B
        assert_eq!(add_reg_32(Reg::T1, Reg::T0, Reg::T3), 0x0B00035B);
    }

    #[test]
    fn test_add_reg_64() {
        // add x21, x21, x20
        // sf=1, op=0, S=0, 01011, 00, 0, Rm=10100, 000000, Rn=10101, Rd=10101
        // 1_00_01011_00_0_10100_000000_10101_10101
        // = 1000_1011_0001_0100_0000_0010_1011_0101
        // = 0x8B1402B5
        assert_eq!(add_reg_64(Reg::PC, Reg::PC, Reg::CTX), 0x8B1402B5);
    }

    #[test]
    fn test_sub_reg_32() {
        // sub w27, w27, w26
        // Rd=T1(27)=11011, Rn=T1(27)=11011, Rm=T0(26)=11010
        // 0_10_01011_00_0_11010_000000_11011_11011 = 0x4B1A037B
        assert_eq!(sub_reg_32(Reg::T1, Reg::T1, Reg::T0), 0x4B1A037B);
    }

    #[test]
    fn test_sub_reg_32_different() {
        // sub w26, w23, w24
        // sf=0, op=1, S=0, 01011, 00, 0, Rm=11000, 000000, Rn=10111, Rd=11010
        // 0_10_01011_00_0_11000_000000_10111_11010
        // = 0100_1011_0001_1000_0000_0010_1111_1010
        // = 0x4B1802FA
        assert_eq!(sub_reg_32(Reg::T0, Reg::L0, Reg::L1), 0x4B1802FA);
    }

    #[test]
    fn test_subs_reg_32() {
        // cmp w23, w26 (= subs wzr, w23, w26)
        // sf=0, op=1, S=1, 01011, 00, 0, Rm=11010, 000000, Rn=10111, Rd=11111
        // 0_11_01011_00_0_11010_000000_10111_11111
        // = 0110_1011_0001_1010_0000_0010_1111_1111
        // = 0x6B1A02FF
        assert_eq!(cmp_reg_32(Reg::L0, Reg::T0), 0x6B1A02FF);
    }

    // --- Logical register ---

    #[test]
    fn test_and_reg_32() {
        // and w26, w27, w26
        // sf=0, opc=00, 01010, 00, N=0, Rm=11010, 000000, Rn=11011, Rd=11010
        // 0_00_01010_00_0_11010_000000_11011_11010
        // = 0000_1010_0001_1010_0000_0011_0111_1010
        // = 0x0A1A037A
        assert_eq!(and_reg_32(Reg::T0, Reg::T1, Reg::T0), 0x0A1A037A);
    }

    #[test]
    fn test_orr_reg_32() {
        // orr w26, w27, w28
        // sf=0, opc=01, 01010, 00, N=0, Rm=11100, 000000, Rn=11011, Rd=11010
        // 0_01_01010_00_0_11100_000000_11011_11010
        // = 0010_1010_0001_1100_0000_0011_0111_1010
        // = 0x2A1C037A
        assert_eq!(orr_reg_32(Reg::T0, Reg::T1, Reg::T2), 0x2A1C037A);
    }

    #[test]
    fn test_eor_reg_32() {
        // eor w26, w27, w0
        // sf=0, opc=10, 01010, 00, N=0, Rm=00000, 000000, Rn=11011, Rd=11010
        // 0_10_01010_00_0_00000_000000_11011_11010
        // = 0100_1010_0000_0000_0000_0011_0111_1010
        // = 0x4A00037A
        assert_eq!(eor_reg_32(Reg::T0, Reg::T1, Reg::T3), 0x4A00037A);
    }

    #[test]
    fn test_mov_reg_32() {
        // mov w26, w23 (= orr w26, wzr, w23)
        // sf=0, opc=01, 01010, 00, N=0, Rm=10111, 000000, Rn=11111, Rd=11010
        // 0_01_01010_00_0_10111_000000_11111_11010
        // = 0010_1010_0001_0111_0000_0011_1111_1010
        // = 0x2A1703FA
        assert_eq!(mov_reg_32(Reg::T0, Reg::L0), 0x2A1703FA);
    }

    #[test]
    fn test_mov_reg_64() {
        // mov x21, x20 (= orr x21, xzr, x20)
        // sf=1, opc=01, 01010, 00, N=0, Rm=10100, 000000, Rn=11111, Rd=10101
        // 1_01_01010_00_0_10100_000000_11111_10101
        // = 1010_1010_0001_0100_0000_0011_1111_0101
        // = 0xAA1403F5
        assert_eq!(mov_reg_64(Reg::PC, Reg::CTX), 0xAA1403F5);
    }

    // --- Multiply / Divide ---

    #[test]
    fn test_mul_32() {
        // mul w26, w27, w28 (= madd w26, w27, w28, wzr)
        // sf=0, 00_11011_000, Rm=11100, 0, Ra=11111, Rn=11011, Rd=11010
        // 0_00_11011_000_11100_0_11111_11011_11010
        // = 0001_1011_0001_1100_0111_1111_0111_1010
        // = 0x1B1C7F7A
        assert_eq!(mul_32(Reg::T0, Reg::T1, Reg::T2), 0x1B1C7F7A);
    }

    #[test]
    fn test_sdiv_32() {
        // sdiv w26, w27, w28
        // sf=0, 00_11010_110, Rm=11100, 00001, o1=1, Rn=11011, Rd=11010
        // 0_00_11010_110_11100_000011_11011_11010
        // = 0001_1010_1101_1100_0000_1111_0111_1010
        // = 0x1ADC0F7A
        assert_eq!(sdiv_32(Reg::T0, Reg::T1, Reg::T2), 0x1ADC0F7A);
    }

    #[test]
    fn test_udiv_32() {
        // udiv w26, w27, w28
        // sf=0, 00_11010_110, Rm=11100, 00001, o1=0, Rn=11011, Rd=11010
        // 0_00_11010_110_11100_000010_11011_11010
        // = 0001_1010_1101_1100_0000_1011_0111_1010
        // = 0x1ADC0B7A
        assert_eq!(udiv_32(Reg::T0, Reg::T1, Reg::T2), 0x1ADC0B7A);
    }

    // --- Variable shifts ---

    #[test]
    fn test_lslv_32() {
        // lslv w26, w27, w28
        // sf=0, 00_11010_110, Rm=11100, 0010, op2=00, Rn=11011, Rd=11010
        // 0_00_11010_110_11100_001000_11011_11010
        // = 0001_1010_1101_1100_0010_0011_0111_1010
        // = 0x1ADC237A
        assert_eq!(lslv_32(Reg::T0, Reg::T1, Reg::T2), 0x1ADC237A);
    }

    #[test]
    fn test_lsrv_32() {
        // lsrv w26, w27, w28
        // sf=0, 00_11010_110, Rm=11100, 0010, op2=01, Rn=11011, Rd=11010
        // 0_00_11010_110_11100_001001_11011_11010
        // = 0001_1010_1101_1100_0010_0111_0111_1010
        // = 0x1ADC277A
        assert_eq!(lsrv_32(Reg::T0, Reg::T1, Reg::T2), 0x1ADC277A);
    }

    #[test]
    fn test_asrv_32() {
        // asrv w26, w27, w28
        // sf=0, 00_11010_110, Rm=11100, 0010, op2=10, Rn=11011, Rd=11010
        // 0_00_11010_110_11100_001010_11011_11010
        // = 0001_1010_1101_1100_0010_1011_0111_1010
        // = 0x1ADC2B7A
        assert_eq!(asrv_32(Reg::T0, Reg::T1, Reg::T2), 0x1ADC2B7A);
    }

    #[test]
    fn test_rorv_32() {
        // rorv w26, w27, w28
        // sf=0, 00_11010_110, Rm=11100, 0010, op2=11, Rn=11011, Rd=11010
        // 0_00_11010_110_11100_001011_11011_11010
        // = 0001_1010_1101_1100_0010_1111_0111_1010
        // = 0x1ADC2F7A
        assert_eq!(rorv_32(Reg::T0, Reg::T1, Reg::T2), 0x1ADC2F7A);
    }

    // --- CLZ / RBIT ---

    #[test]
    fn test_clz_32() {
        // clz w26, w27
        // sf=0, 10_11010_110, 00000, 00010_0, Rn=11011, Rd=11010
        // 0_10_11010_110_00000_000100_11011_11010
        // = 0101_1010_1100_0000_0001_0011_0111_1010
        // = 0x5AC0137A
        assert_eq!(clz_32(Reg::T0, Reg::T1), 0x5AC0137A);
    }

    #[test]
    fn test_rbit_32() {
        // rbit w26, w27
        // sf=0, 10_11010_110, 00000, 00000_0, Rn=11011, Rd=11010
        // 0_10_11010_110_00000_000000_11011_11010
        // = 0101_1010_1100_0000_0000_0011_0111_1010
        // = 0x5AC0037A
        assert_eq!(rbit_32(Reg::T0, Reg::T1), 0x5AC0037A);
    }

    // --- Add/Sub immediate ---

    #[test]
    fn test_add_imm_64() {
        // add x21, x21, #0x20
        assert_eq!(add_imm_64(Reg::PC, Reg::PC, 0x20), 0x910082B5);
    }

    #[test]
    fn test_add_imm_32() {
        // add w23, w23, #1
        // sf=0, op=0, S=0, 100010, sh=0, imm12=000000000001, Rn=10111, Rd=10111
        // 0_0_0_100010_0_000000000001_10111_10111
        // = 0001_0001_0000_0000_0000_0110_1111_0111
        // = 0x110006F7
        assert_eq!(add_imm_32(Reg::L0, Reg::L0, 1), 0x110006F7);
    }

    #[test]
    fn test_sub_imm_32() {
        // sub w23, w23, #1
        // sf=0, op=1, S=0, 100010, sh=0, imm12=000000000001, Rn=10111, Rd=10111
        // 0_1_0_100010_0_000000000001_10111_10111
        // = 0101_0001_0000_0000_0000_0110_1111_0111
        // = 0x510006F7
        assert_eq!(sub_imm_32(Reg::L0, Reg::L0, 1), 0x510006F7);
    }

    #[test]
    fn test_cmp_imm_32() {
        // cmp w26, #0 (= subs wzr, w26, #0)
        // sf=0, op=1, S=1, 100010, sh=0, imm12=0, Rn=11010, Rd=11111
        // 0_1_1_100010_0_000000000000_11010_11111
        // = 0111_0001_0000_0000_0000_0011_0101_1111
        // = 0x7100035F
        assert_eq!(cmp_imm_32(Reg::T0, 0), 0x7100035F);
    }

    // --- Conditional select ---

    #[test]
    fn test_csel_32() {
        // csel w26, w27, w28, eq
        // sf=0, op=0, S=0, 11010100, Rm=11100, cond=0000, op2=00, Rn=11011, Rd=11010
        // 0_0_0_11010100_11100_0000_00_11011_11010
        // = 0001_1010_1001_1100_0000_0011_0111_1010
        // = 0x1A9C037A
        assert_eq!(csel_32(Reg::T0, Reg::T1, Reg::T2, Cond::EQ), 0x1A9C037A);
    }

    #[test]
    fn test_cset_32() {
        // cset w26, eq (= csinc w26, wzr, wzr, ne)
        // sf=0, op=0, S=0, 11010100, Rm=11111, cond=0001(NE), op2=01, Rn=11111, Rd=11010
        // 0_0_0_11010100_11111_0001_01_11111_11010
        // = 0001_1010_1001_1111_0001_0111_1111_1010
        // = 0x1A9F17FA
        assert_eq!(cset_32(Reg::T0, Cond::EQ), 0x1A9F17FA);
    }

    // --- Move wide ---

    #[test]
    fn test_movz_32() {
        // movz w23, #42
        assert_eq!(movz_32(Reg::L0, 42, 0), 0x52800557);
    }

    #[test]
    fn test_movz_32_large() {
        // movz w26, #0xFFFF
        // sf=0, opc=10, 100101, hw=00, imm16=1111111111111111, Rd=11010
        // 0_10_100101_00_1111111111111111_11010
        // = 0101_0010_1001_1111_1111_1111_1111_1010
        // = 0x529FFFFA
        assert_eq!(movz_32(Reg::T0, 0xFFFF, 0), 0x529FFFFA);
    }

    #[test]
    fn test_movz_64() {
        // movz x2, #0x1234
        // sf=1, opc=10, 100101, hw=00, imm16=0001001000110100, Rd=00010
        // 1_10_100101_00_0001001000110100_00010
        // = 1101_0010_1000_0010_0100_0110_1000_0010
        // = 0xD2824682
        assert_eq!(movz_64(Reg::TMP0, 0x1234, 0), 0xD2824682);
    }

    #[test]
    fn test_movk_64() {
        // movk x2, #0x5678, lsl #16
        // sf=1, opc=11, 100101, hw=01, imm16=0101011001111000, Rd=00010
        // 1_11_100101_01_0101011001111000_00010
        // = 1111_0010_1010_1010_1100_1111_0000_0010
        // = 0xF2AACF02
        assert_eq!(movk_64(Reg::TMP0, 0x5678, 16), 0xF2AACF02);
    }

    #[test]
    fn test_movn_32() {
        // movn w26, #0
        // sf=0, opc=00, 100101, hw=00, imm16=0, Rd=11010
        // 0_00_100101_00_0000000000000000_11010
        // = 0001_0010_1000_0000_0000_0000_0001_1010
        // = 0x1280001A
        assert_eq!(movn_32(Reg::T0, 0, 0), 0x1280001A);
    }

    // --- Load / Store unsigned offset ---

    #[test]
    fn test_ldr_64() {
        // ldr x2, [x21, #0]
        assert_eq!(ldr_64(Reg::TMP0, Reg::PC, 0), 0xF94002A2);
        // ldr x1, [x21, #0x20] (offset=0x20, scale=8, imm12=4)
        assert_eq!(ldr_64(Reg::NH, Reg::PC, 4), 0xF94012A1);
    }

    #[test]
    fn test_str_64() {
        // str x2, [x20, #32] (offset=32, scale=8, imm12=4)
        // size=11, 111_0_01, opc=00, imm12=000000000100, Rn=10100, Rt=00010
        // 11_111_0_01_00_000000000100_10100_00010
        // = 1111_1001_0000_0000_0001_0010_1000_0010
        // = 0xF9001282
        assert_eq!(str_64(Reg::TMP0, Reg::CTX, 4), 0xF9001282);
    }

    #[test]
    fn test_ldr_32() {
        // ldr w26, [x2, #0] (32-bit load)
        // size=10, 111_0_01, opc=01, imm12=0, Rn=00010, Rt=11010
        // 10_111_0_01_01_000000000000_00010_11010
        // = 1011_1001_0100_0000_0000_0000_0101_1010
        // = 0xB940005A
        assert_eq!(ldr_32(Reg::T0, Reg::TMP0, 0), 0xB940005A);
    }

    #[test]
    fn test_str_32() {
        // str w26, [x2, #0]
        // size=10, 111_0_01, opc=00, imm12=0, Rn=00010, Rt=11010
        // 10_111_0_01_00_000000000000_00010_11010
        // = 1011_1001_0000_0000_0000_0000_0101_1010
        // = 0xB900005A
        assert_eq!(str_32(Reg::T0, Reg::TMP0, 0), 0xB900005A);
    }

    #[test]
    fn test_ldrb() {
        // ldrb w26, [x2, #0]
        // size=00, 111_0_01, opc=01, imm12=0, Rn=00010, Rt=11010
        // 00_111_0_01_01_000000000000_00010_11010
        // = 0011_1001_0100_0000_0000_0000_0101_1010
        // = 0x3940005A
        assert_eq!(ldrb(Reg::T0, Reg::TMP0, 0), 0x3940005A);
    }

    #[test]
    fn test_ldrh() {
        // ldrh w26, [x2, #0]
        // size=01, 111_0_01, opc=01, imm12=0, Rn=00010, Rt=11010
        // 01_111_0_01_01_000000000000_00010_11010
        // = 0111_1001_0100_0000_0000_0000_0101_1010
        // = 0x7940005A
        assert_eq!(ldrh(Reg::T0, Reg::TMP0, 0), 0x7940005A);
    }

    #[test]
    fn test_str_f64() {
        // str d0, [x1, #8]
        assert_eq!(str_f64(0, Reg::NH, 1), 0xFD000420);
    }

    #[test]
    fn test_ldr_f64() {
        // ldr d0, [x1, #8]
        assert_eq!(ldr_f64(0, Reg::NH, 1), 0xFD400420);
    }

    #[test]
    fn test_str_f32() {
        // str s0, [x1, #4]
        assert_eq!(str_f32(0, Reg::NH, 1), 0xBD000420);
    }

    #[test]
    fn test_ldr_f32() {
        // ldr s0, [x1, #4]
        assert_eq!(ldr_f32(0, Reg::NH, 1), 0xBD400420);
    }

    // --- Load / Store register offset ---

    #[test]
    fn test_ldr_64_reg() {
        // ldr x26, [x2, x3]
        // size=11, 111_0_00, opc=01, 1, Rm=00011, option=011, S=0, 10, Rn=00010, Rt=11010
        // 11_111_0_00_01_1_00011_011_0_10_00010_11010
        // = 1111_1000_0110_0011_0110_1000_0101_1010
        // = 0xF863685A
        assert_eq!(ldr_64_reg(Reg::T0, Reg::TMP0, Reg::TMP1), 0xF863685A);
    }

    #[test]
    fn test_ldr_32_reg() {
        // ldr w26, [x2, x3]
        // size=10, 111_0_00, opc=01, 1, Rm=00011, option=011, S=0, 10, Rn=00010, Rt=11010
        // 10_111_0_00_01_1_00011_011_0_10_00010_11010
        // = 1011_1000_0110_0011_0110_1000_0101_1010
        // = 0xB863685A
        assert_eq!(ldr_32_reg(Reg::T0, Reg::TMP0, Reg::TMP1), 0xB863685A);
    }

    #[test]
    fn test_ldrb_reg() {
        // ldrb w26, [x2, x3]
        // size=00, 111_0_00, opc=01, 1, Rm=00011, option=011, S=0, 10, Rn=00010, Rt=11010
        // 00_111_0_00_01_1_00011_011_0_10_00010_11010
        // = 0011_1000_0110_0011_0110_1000_0101_1010
        // = 0x3863685A
        assert_eq!(ldrb_reg(Reg::T0, Reg::TMP0, Reg::TMP1), 0x3863685A);
    }

    #[test]
    fn test_strb_reg() {
        // strb w26, [x2, x3]
        // size=00, 111_0_00, opc=00, 1, Rm=00011, option=011, S=0, 10, Rn=00010, Rt=11010
        // 00_111_0_00_00_1_00011_011_0_10_00010_11010
        // = 0011_1000_0010_0011_0110_1000_0101_1010
        // = 0x3823685A
        assert_eq!(strb_reg(Reg::T0, Reg::TMP0, Reg::TMP1), 0x3823685A);
    }

    // --- Sign/Zero extension ---

    #[test]
    fn test_sxtw() {
        // sxtw x26, w27 (= sbfm x26, x27, #0, #31)
        // sf=1, 00, 100110, N=1, immr=000000, imms=011111, Rn=11011, Rd=11010
        // 1_00_100110_1_000000_011111_11011_11010
        // = 1001_0011_0100_0000_0111_1111_0111_1010
        // = 0x93407F7A
        assert_eq!(sxtw(Reg::T0, Reg::T1), 0x93407F7A);
    }

    #[test]
    fn test_sxth_32() {
        // sxth w26, w27 (= sbfm w26, w27, #0, #15)
        // sf=0, 00, 100110, N=0, immr=000000, imms=001111, Rn=11011, Rd=11010
        // 0_00_100110_0_000000_001111_11011_11010
        // = 0001_0011_0000_0000_0011_1111_0111_1010
        // = 0x13003F7A
        assert_eq!(sxth_32(Reg::T0, Reg::T1), 0x13003F7A);
    }

    #[test]
    fn test_sxtb_32() {
        // sxtb w26, w27 (= sbfm w26, w27, #0, #7)
        // sf=0, 00, 100110, N=0, immr=000000, imms=000111, Rn=11011, Rd=11010
        // 0_00_100110_0_000000_000111_11011_11010
        // = 0001_0011_0000_0000_0001_1111_0111_1010
        // = 0x13001F7A
        assert_eq!(sxtb_32(Reg::T0, Reg::T1), 0x13001F7A);
    }

    #[test]
    fn test_uxth_32() {
        // uxth w26, w27 (= ubfm w26, w27, #0, #15)
        // sf=0, 10, 100110, N=0, immr=000000, imms=001111, Rn=11011, Rd=11010
        // 0_10_100110_0_000000_001111_11011_11010
        // = 0101_0011_0000_0000_0011_1111_0111_1010
        // = 0x53003F7A
        assert_eq!(uxth_32(Reg::T0, Reg::T1), 0x53003F7A);
    }

    #[test]
    fn test_uxtb_32() {
        // uxtb w26, w27 (= ubfm w26, w27, #0, #7)
        // sf=0, 10, 100110, N=0, immr=000000, imms=000111, Rn=11011, Rd=11010
        // 0_10_100110_0_000000_000111_11011_11010
        // = 0101_0011_0000_0000_0001_1111_0111_1010
        // = 0x53001F7A
        assert_eq!(uxtb_32(Reg::T0, Reg::T1), 0x53001F7A);
    }

    // --- Immediate shifts ---

    #[test]
    fn test_lsl_imm_32() {
        // lsl w26, w27, #2 (= ubfm w26, w27, #30, #29)
        // sf=0, 10, 100110, N=0, immr=011110, imms=011101, Rn=11011, Rd=11010
        // 0_10_100110_0_011110_011101_11011_11010
        // = 0101_0011_0001_1110_0111_0111_0111_1010
        // = 0x531E777A
        assert_eq!(lsl_imm_32(Reg::T0, Reg::T1, 2), 0x531E777A);
    }

    #[test]
    fn test_lsr_imm_32() {
        // lsr w26, w27, #2 (= ubfm w26, w27, #2, #31)
        // sf=0, 10, 100110, N=0, immr=000010, imms=011111, Rn=11011, Rd=11010
        // 0_10_100110_0_000010_011111_11011_11010
        // = 0101_0011_0000_0010_0111_1111_0111_1010
        // = 0x53027F7A
        assert_eq!(lsr_imm_32(Reg::T0, Reg::T1, 2), 0x53027F7A);
    }

    #[test]
    fn test_asr_imm_32() {
        // asr w26, w27, #2 (= sbfm w26, w27, #2, #31)
        // sf=0, 00, 100110, N=0, immr=000010, imms=011111, Rn=11011, Rd=11010
        // 0_00_100110_0_000010_011111_11011_11010
        // = 0001_0011_0000_0010_0111_1111_0111_1010
        // = 0x13027F7A
        assert_eq!(asr_imm_32(Reg::T0, Reg::T1, 2), 0x13027F7A);
    }

    // --- Branch ---

    #[test]
    fn test_br() {
        // br x2
        assert_eq!(br(Reg::TMP0), 0xD61F0040);
    }

    #[test]
    fn test_ret() {
        // ret (= ret x30)
        assert_eq!(ret(), 0xD65F03C0);
    }

    #[test]
    fn test_b() {
        // b #0 (branch to self)
        assert_eq!(b(0), 0x14000000);
        // b #+4 (1 instruction forward)
        assert_eq!(b(1), 0x14000001);
        // b #-4 (1 instruction backward)
        // imm26 = -1 = 0x03FFFFFF
        assert_eq!(b(-1), 0x17FFFFFF);
    }

    #[test]
    fn test_b_cond() {
        // b.eq #+8 (2 instructions forward)
        // 01010100, imm19=00000000000000010, 0, cond=0000
        // 0101_0100_0000_0000_0000_0100_0000_0000
        // = 0x54000040
        assert_eq!(b_cond(Cond::EQ, 2), 0x54000040);
        // b.ne #-4 (1 instruction backward)
        // imm19 = -1 = 0x7FFFF
        // 01010100_1111111111111111111_0_0001
        // = 0x54FFFFE1
        assert_eq!(b_cond(Cond::NE, -1), 0x54FFFFE1);
    }

    #[test]
    fn test_cbz_64() {
        // cbz x26, #+8 (2 instructions forward)
        // sf=1, 011010_0, imm19=00000000000000010, Rt=11010
        // 1_011010_0_00000000000000010_11010
        // = 1011_0100_0000_0000_0000_0000_0101_1010
        // = 0xB400005A
        assert_eq!(cbz_64(Reg::T0, 2), 0xB400005A);
    }

    #[test]
    fn test_cbnz_64() {
        // cbnz x26, #+8 (2 instructions forward)
        // sf=1, 011010_1, imm19=00000000000000010, Rt=11010
        // 1_011010_1_00000000000000010_11010
        // = 1011_0101_0000_0000_0000_0000_0101_1010
        // = 0xB500005A
        assert_eq!(cbnz_64(Reg::T0, 2), 0xB500005A);
    }

    #[test]
    fn test_nop() {
        assert_eq!(nop(), 0xD503201F);
    }

    // --- Neg / MVN ---

    #[test]
    fn test_neg_32() {
        // neg w26, w27 (= sub w26, wzr, w27)
        // sf=0, op=1, S=0, 01011, 00, 0, Rm=11011, 000000, Rn=11111, Rd=11010
        // 0_10_01011_00_0_11011_000000_11111_11010
        // = 0100_1011_0001_1011_0000_0011_1111_1010
        // = 0x4B1B03FA
        assert_eq!(neg_32(Reg::T0, Reg::T1), 0x4B1B03FA);
    }

    #[test]
    fn test_mvn_32() {
        // mvn w26, w27 (= orn w26, wzr, w27)
        // sf=0, opc=01, 01010, 00, N=1, Rm=11011, 000000, Rn=11111, Rd=11010
        // 0_01_01010_00_1_11011_000000_11111_11010
        // = 0010_1010_0011_1011_0000_0011_1111_1010
        // = 0x2A3B03FA
        assert_eq!(mvn_32(Reg::T0, Reg::T1), 0x2A3B03FA);
    }

    // --- Cond invert ---

    #[test]
    fn test_cond_invert() {
        assert_eq!(Cond::EQ.invert(), Cond::NE);
        assert_eq!(Cond::NE.invert(), Cond::EQ);
        assert_eq!(Cond::LT.invert(), Cond::GE);
        assert_eq!(Cond::GE.invert(), Cond::LT);
        assert_eq!(Cond::GT.invert(), Cond::LE);
        assert_eq!(Cond::LE.invert(), Cond::GT);
        assert_eq!(Cond::HI.invert(), Cond::LS);
        assert_eq!(Cond::LS.invert(), Cond::HI);
    }

    // --- Load/Store sign-extending ---

    #[test]
    fn test_ldrsw() {
        // ldrsw x26, [x2, #0]
        // size=10, 111_0_01, opc=10, imm12=0, Rn=00010, Rt=11010
        // 10_111_0_01_10_000000000000_00010_11010
        // = 1011_1001_1000_0000_0000_0000_0101_1010
        // = 0xB980005A
        assert_eq!(ldrsw(Reg::T0, Reg::TMP0, 0), 0xB980005A);
    }

    #[test]
    fn test_ldrsh_32() {
        // ldrsh w26, [x2, #0]
        // size=01, 111_0_01, opc=11, imm12=0, Rn=00010, Rt=11010
        // 01_111_0_01_11_000000000000_00010_11010
        // = 0111_1001_1100_0000_0000_0000_0101_1010
        // = 0x79C0005A
        assert_eq!(ldrsh_32(Reg::T0, Reg::TMP0, 0), 0x79C0005A);
    }

    #[test]
    fn test_ldrsb_32() {
        // ldrsb w26, [x2, #0]
        // size=00, 111_0_01, opc=11, imm12=0, Rn=00010, Rt=11010
        // 00_111_0_01_11_000000000000_00010_11010
        // = 0011_1001_1100_0000_0000_0000_0101_1010
        // = 0x39C0005A
        assert_eq!(ldrsb_32(Reg::T0, Reg::TMP0, 0), 0x39C0005A);
    }
}
