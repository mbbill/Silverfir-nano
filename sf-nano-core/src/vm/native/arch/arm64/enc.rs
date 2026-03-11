//! Small ARM64 encoder for the native backend.

use super::reg::Arm64Reg;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Cond {
    Eq = 0x0,
    Ne = 0x1,
    Hs = 0x2,
    Lo = 0x3,
    Mi = 0x4,
    Pl = 0x5,
    Vs = 0x6,
    Vc = 0x7,
    Hi = 0x8,
    Ls = 0x9,
    Ge = 0xA,
    Lt = 0xB,
    Gt = 0xC,
    Le = 0xD,
    Al = 0xE,
}

impl Cond {
    #[inline]
    pub const fn invert(self) -> Cond {
        unsafe { core::mem::transmute(self as u8 ^ 1) }
    }
}

fn add_sub_shifted_reg(sf: u32, op: u32, s: u32, rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    (sf << 31)
        | (op << 30)
        | (s << 29)
        | (0b01011 << 24)
        | (rm.idx() << 16)
        | (rn.idx() << 5)
        | rd.idx()
}

fn logical_shifted_reg(sf: u32, opc: u32, n: u32, rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    (sf << 31)
        | (opc << 29)
        | (0b01010 << 24)
        | (n << 21)
        | (rm.idx() << 16)
        | (rn.idx() << 5)
        | rd.idx()
}

fn madd(sf: u32, rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg, ra: Arm64Reg) -> u32 {
    (sf << 31)
        | (0b00_11011_000 << 21)
        | (rm.idx() << 16)
        | (ra.idx() << 10)
        | (rn.idx() << 5)
        | rd.idx()
}

fn msub(sf: u32, rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg, ra: Arm64Reg) -> u32 {
    (sf << 31)
        | (0b00_11011_000 << 21)
        | (rm.idx() << 16)
        | (1 << 15)
        | (ra.idx() << 10)
        | (rn.idx() << 5)
        | rd.idx()
}

fn shift_var(sf: u32, op2: u32, rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    (sf << 31)
        | (0b00_11010_110 << 21)
        | (rm.idx() << 16)
        | (0b0010 << 12)
        | (op2 << 10)
        | (rn.idx() << 5)
        | rd.idx()
}

fn add_sub_imm(sf: u32, op: u32, s: u32, rd: Arm64Reg, rn: Arm64Reg, imm12: u32) -> u32 {
    debug_assert!(imm12 < 0x1000);
    (sf << 31)
        | (op << 30)
        | (s << 29)
        | (0b100010 << 23)
        | (imm12 << 10)
        | (rn.idx() << 5)
        | rd.idx()
}

fn cond_select(sf: u32, op2: u32, rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg, cond: Cond) -> u32 {
    (sf << 31)
        | (0b0_0_11010100 << 21)
        | (rm.idx() << 16)
        | ((cond as u32) << 12)
        | (op2 << 10)
        | (rn.idx() << 5)
        | rd.idx()
}

fn move_wide(sf: u32, opc: u32, rd: Arm64Reg, imm16: u16, shift: u32) -> u32 {
    let hw = shift / 16;
    debug_assert!(shift % 16 == 0);
    (sf << 31) | (opc << 29) | (0b100101 << 23) | (hw << 21) | ((imm16 as u32) << 5) | rd.idx()
}

fn sbfm(sf: u32, n: u32, rd: Arm64Reg, rn: Arm64Reg, immr: u32, imms: u32) -> u32 {
    (sf << 31)
        | (0b00 << 29)
        | (0b100110 << 23)
        | (n << 22)
        | (immr << 16)
        | (imms << 10)
        | (rn.idx() << 5)
        | rd.idx()
}

fn ubfm(sf: u32, n: u32, rd: Arm64Reg, rn: Arm64Reg, immr: u32, imms: u32) -> u32 {
    (sf << 31)
        | (0b10 << 29)
        | (0b100110 << 23)
        | (n << 22)
        | (immr << 16)
        | (imms << 10)
        | (rn.idx() << 5)
        | rd.idx()
}

fn ldst_unsigned_offset(size: u32, opc: u32, rt: Arm64Reg, rn: Arm64Reg, imm12: u32) -> u32 {
    debug_assert!(imm12 < 0x1000);
    (size << 30) | (0b111_0_01 << 24) | (opc << 22) | (imm12 << 10) | (rn.idx() << 5) | rt.idx()
}

pub fn add_reg_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    add_sub_shifted_reg(0, 0, 0, rd, rn, rm)
}

pub fn add_reg_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    add_sub_shifted_reg(1, 0, 0, rd, rn, rm)
}

pub fn sub_reg_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    add_sub_shifted_reg(0, 1, 0, rd, rn, rm)
}

pub fn sub_reg_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    add_sub_shifted_reg(1, 1, 0, rd, rn, rm)
}

pub fn subs_reg_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    add_sub_shifted_reg(0, 1, 1, rd, rn, rm)
}

pub fn subs_reg_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    add_sub_shifted_reg(1, 1, 1, rd, rn, rm)
}

pub fn and_reg_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    logical_shifted_reg(0, 0b00, 0, rd, rn, rm)
}

pub fn and_reg_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    logical_shifted_reg(1, 0b00, 0, rd, rn, rm)
}

pub fn orr_reg_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    logical_shifted_reg(0, 0b01, 0, rd, rn, rm)
}

pub fn orr_reg_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    logical_shifted_reg(1, 0b01, 0, rd, rn, rm)
}

pub fn eor_reg_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    logical_shifted_reg(0, 0b10, 0, rd, rn, rm)
}

pub fn eor_reg_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    logical_shifted_reg(1, 0b10, 0, rd, rn, rm)
}

pub fn mul_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    madd(0, rd, rn, rm, Arm64Reg::Xzr)
}

pub fn mul_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    madd(1, rd, rn, rm, Arm64Reg::Xzr)
}

pub fn msub_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg, ra: Arm64Reg) -> u32 {
    msub(0, rd, rn, rm, ra)
}

pub fn msub_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg, ra: Arm64Reg) -> u32 {
    msub(1, rd, rn, rm, ra)
}

pub fn lslv_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    shift_var(0, 0b00, rd, rn, rm)
}

pub fn lsrv_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    shift_var(0, 0b01, rd, rn, rm)
}

pub fn asrv_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    shift_var(0, 0b10, rd, rn, rm)
}

pub fn rorv_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    shift_var(0, 0b11, rd, rn, rm)
}

pub fn lslv_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    shift_var(1, 0b00, rd, rn, rm)
}

pub fn lsrv_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    shift_var(1, 0b01, rd, rn, rm)
}

pub fn asrv_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    shift_var(1, 0b10, rd, rn, rm)
}

pub fn rorv_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    shift_var(1, 0b11, rd, rn, rm)
}

pub fn add_imm_64(rd: Arm64Reg, rn: Arm64Reg, imm12: u32) -> u32 {
    add_sub_imm(1, 0, 0, rd, rn, imm12)
}

pub fn sub_imm_64(rd: Arm64Reg, rn: Arm64Reg, imm12: u32) -> u32 {
    add_sub_imm(1, 1, 0, rd, rn, imm12)
}

pub fn cmp_reg_32(rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    subs_reg_32(Arm64Reg::Xzr, rn, rm)
}

pub fn cmp_reg_64(rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    subs_reg_64(Arm64Reg::Xzr, rn, rm)
}

pub fn cmp_imm_32(rn: Arm64Reg, imm12: u32) -> u32 {
    add_sub_imm(0, 1, 1, Arm64Reg::Xzr, rn, imm12)
}

pub fn cmp_imm_64(rn: Arm64Reg, imm12: u32) -> u32 {
    add_sub_imm(1, 1, 1, Arm64Reg::Xzr, rn, imm12)
}

pub fn cset_64(rd: Arm64Reg, cond: Cond) -> u32 {
    cond_select(1, 0b01, rd, Arm64Reg::Xzr, Arm64Reg::Xzr, cond.invert())
}

pub fn cset_32(rd: Arm64Reg, cond: Cond) -> u32 {
    cond_select(0, 0b01, rd, Arm64Reg::Xzr, Arm64Reg::Xzr, cond.invert())
}

pub fn mov_reg_32(rd: Arm64Reg, rm: Arm64Reg) -> u32 {
    orr_reg_32(rd, Arm64Reg::Xzr, rm)
}

pub fn mov_reg_64(rd: Arm64Reg, rm: Arm64Reg) -> u32 {
    orr_reg_64(rd, Arm64Reg::Xzr, rm)
}

pub fn movz_64(rd: Arm64Reg, imm16: u16, shift: u32) -> u32 {
    move_wide(1, 0b10, rd, imm16, shift)
}

pub fn movk_64(rd: Arm64Reg, imm16: u16, shift: u32) -> u32 {
    move_wide(1, 0b11, rd, imm16, shift)
}

pub fn clz_32(rd: Arm64Reg, rn: Arm64Reg) -> u32 {
    (0b0_10_11010_110 << 21)
        | (0b00000 << 16)
        | (0b00010_0 << 10)
        | (rn.idx() << 5)
        | rd.idx()
}

pub fn clz_64(rd: Arm64Reg, rn: Arm64Reg) -> u32 {
    (0b1_10_11010_110 << 21)
        | (0b00000 << 16)
        | (0b00010_0 << 10)
        | (rn.idx() << 5)
        | rd.idx()
}

pub fn rbit_32(rd: Arm64Reg, rn: Arm64Reg) -> u32 {
    (0b0_10_11010_110 << 21)
        | (0b00000 << 16)
        | (0b00000_0 << 10)
        | (rn.idx() << 5)
        | rd.idx()
}

pub fn rbit_64(rd: Arm64Reg, rn: Arm64Reg) -> u32 {
    (0b1_10_11010_110 << 21)
        | (0b00000 << 16)
        | (0b00000_0 << 10)
        | (rn.idx() << 5)
        | rd.idx()
}

pub fn sxtw(rd: Arm64Reg, rn: Arm64Reg) -> u32 {
    sbfm(1, 1, rd, rn, 0, 31)
}

pub fn sxtb_32(rd: Arm64Reg, rn: Arm64Reg) -> u32 {
    sbfm(0, 0, rd, rn, 0, 7)
}

pub fn sxtb_64(rd: Arm64Reg, rn: Arm64Reg) -> u32 {
    sbfm(1, 1, rd, rn, 0, 7)
}

pub fn sxth_32(rd: Arm64Reg, rn: Arm64Reg) -> u32 {
    sbfm(0, 0, rd, rn, 0, 15)
}

pub fn sxth_64(rd: Arm64Reg, rn: Arm64Reg) -> u32 {
    sbfm(1, 1, rd, rn, 0, 15)
}

pub fn uxth_32(rd: Arm64Reg, rn: Arm64Reg) -> u32 {
    ubfm(0, 0, rd, rn, 0, 15)
}

pub fn ldr_64(rt: Arm64Reg, rn: Arm64Reg, imm12: u32) -> u32 {
    ldst_unsigned_offset(0b11, 0b01, rt, rn, imm12)
}

pub fn str_64(rt: Arm64Reg, rn: Arm64Reg, imm12: u32) -> u32 {
    ldst_unsigned_offset(0b11, 0b00, rt, rn, imm12)
}

pub fn ldr_lit_64(rt: Arm64Reg, imm19: i32) -> u32 {
    let imm19_bits = (imm19 as u32) & 0x0007_FFFF;
    (0b01011000 << 24) | (imm19_bits << 5) | rt.idx()
}

pub fn b(imm26: i32) -> u32 {
    let imm26_bits = (imm26 as u32) & 0x03FF_FFFF;
    (0b000101 << 26) | imm26_bits
}

pub fn b_cond(cond: Cond, imm19: i32) -> u32 {
    let imm19_bits = (imm19 as u32) & 0x0007_FFFF;
    (0b01010100 << 24) | (imm19_bits << 5) | (cond as u32)
}

pub fn cbz_64(rt: Arm64Reg, imm19: i32) -> u32 {
    let imm19_bits = (imm19 as u32) & 0x0007_FFFF;
    (0b1_011010_0 << 24) | (imm19_bits << 5) | rt.idx()
}

pub fn cbnz_64(rt: Arm64Reg, imm19: i32) -> u32 {
    let imm19_bits = (imm19 as u32) & 0x0007_FFFF;
    (0b1_011010_1 << 24) | (imm19_bits << 5) | rt.idx()
}

pub fn br(rn: Arm64Reg) -> u32 {
    (0b1101011_0000 << 21) | (0b11111 << 16) | (rn.idx() << 5)
}

pub fn blr(rn: Arm64Reg) -> u32 {
    (0b1101011_0001 << 21) | (0b11111 << 16) | (rn.idx() << 5)
}

pub fn ret() -> u32 {
    (0b1101011_0010 << 21) | (0b11111 << 16) | (Arm64Reg::X30.idx() << 5)
}

#[cfg(test)]
mod tests {
    use super::{b, blr, br, cbz_64, ldr_lit_64};
    use crate::vm::native::arch::arm64::reg::Arm64Reg;

    #[test]
    fn encodes_branch_register() {
        assert_eq!(br(Arm64Reg::X16), 0xd61f0200);
    }

    #[test]
    fn encodes_branch_link_register() {
        assert_eq!(blr(Arm64Reg::X16), 0xd63f0200);
    }

    #[test]
    fn encodes_literal_load() {
        assert_eq!(ldr_lit_64(Arm64Reg::X16, 2), 0x58000050);
    }

    #[test]
    fn encodes_unconditional_branch() {
        assert_eq!(b(1), 0x14000001);
    }

    #[test]
    fn encodes_cbz() {
        assert_eq!(cbz_64(Arm64Reg::X9, 1), 0xb4000029);
    }
}
