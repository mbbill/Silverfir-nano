//! Minimal RV64GC encoder used by the first RISC-V backend bring-up.
//!
//! The backend intentionally emits only 32-bit base encodings for now, even on
//! targets with the compressed `C` extension. That keeps patch offsets simple
//! while the port is coming up.

use super::reg::{RiscvFpReg, RiscvReg};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Cond {
    Eq,
    Ne,
    Lt,
    Ge,
    Ltu,
    Geu,
}

impl Cond {
    #[inline]
    pub(super) const fn invert(self) -> Self {
        match self {
            Self::Eq => Self::Ne,
            Self::Ne => Self::Eq,
            Self::Lt => Self::Ge,
            Self::Ge => Self::Lt,
            Self::Ltu => Self::Geu,
            Self::Geu => Self::Ltu,
        }
    }

    #[inline]
    const fn funct3(self) -> u32 {
        match self {
            Self::Eq => 0b000,
            Self::Ne => 0b001,
            Self::Lt => 0b100,
            Self::Ge => 0b101,
            Self::Ltu => 0b110,
            Self::Geu => 0b111,
        }
    }
}

#[inline]
fn r_type(funct7: u32, rs2: RiscvReg, rs1: RiscvReg, funct3: u32, rd: RiscvReg, op: u32) -> u32 {
    (funct7 << 25) | (rs2.idx() << 20) | (rs1.idx() << 15) | (funct3 << 12) | (rd.idx() << 7) | op
}

#[inline]
fn fp_r_type(funct7: u32, rs2: RiscvFpReg, rs1: RiscvFpReg, funct3: u32, rd: RiscvFpReg) -> u32 {
    (funct7 << 25)
        | (rs2.idx() << 20)
        | (rs1.idx() << 15)
        | (funct3 << 12)
        | (rd.idx() << 7)
        | 0b1010011
}

#[inline]
fn fp_to_gp_type(funct7: u32, rs2: u32, rs1: RiscvFpReg, funct3: u32, rd: RiscvReg) -> u32 {
    (funct7 << 25) | (rs2 << 20) | (rs1.idx() << 15) | (funct3 << 12) | (rd.idx() << 7) | 0b1010011
}

#[inline]
fn gp_to_fp_type(funct7: u32, rs2: u32, rs1: RiscvReg, funct3: u32, rd: RiscvFpReg) -> u32 {
    (funct7 << 25) | (rs2 << 20) | (rs1.idx() << 15) | (funct3 << 12) | (rd.idx() << 7) | 0b1010011
}

#[inline]
fn i_type(imm: i32, rs1: RiscvReg, funct3: u32, rd: RiscvReg, op: u32) -> u32 {
    (((imm as u32) & 0x0fff) << 20) | (rs1.idx() << 15) | (funct3 << 12) | (rd.idx() << 7) | op
}

#[inline]
fn s_type(imm: i32, rs2: RiscvReg, rs1: RiscvReg, funct3: u32, op: u32) -> u32 {
    let imm = (imm as u32) & 0x0fff;
    ((imm >> 5) << 25)
        | (rs2.idx() << 20)
        | (rs1.idx() << 15)
        | (funct3 << 12)
        | ((imm & 0x1f) << 7)
        | op
}

#[inline]
pub(super) fn b_type(cond: Cond, rs1: RiscvReg, rs2: RiscvReg, offset: i32) -> u32 {
    debug_assert!(
        offset % 2 == 0,
        "RISC-V branch target must be halfword aligned"
    );
    debug_assert!(
        (-4096..=4094).contains(&offset),
        "RISC-V branch offset out of range"
    );
    let imm = (offset as u32) & 0x1fff;
    ((imm >> 12) << 31)
        | (((imm >> 5) & 0x3f) << 25)
        | (rs2.idx() << 20)
        | (rs1.idx() << 15)
        | (cond.funct3() << 12)
        | (((imm >> 1) & 0x0f) << 8)
        | (((imm >> 11) & 0x01) << 7)
        | 0b1100011
}

#[inline]
fn u_type(imm20: u32, rd: RiscvReg, op: u32) -> u32 {
    (imm20 << 12) | (rd.idx() << 7) | op
}

#[inline]
pub(super) fn jal(rd: RiscvReg, offset: i32) -> u32 {
    debug_assert!(
        offset % 2 == 0,
        "RISC-V jal target must be halfword aligned"
    );
    debug_assert!(
        (-(1 << 20)..=((1 << 20) - 2)).contains(&offset),
        "RISC-V jal offset out of range"
    );
    let imm = (offset as u32) & 0x1f_ffff;
    (((imm >> 20) & 0x01) << 31)
        | (((imm >> 1) & 0x03ff) << 21)
        | (((imm >> 11) & 0x01) << 20)
        | (((imm >> 12) & 0x0ff) << 12)
        | (rd.idx() << 7)
        | 0b1101111
}

#[inline]
pub(super) fn addi(rd: RiscvReg, rs1: RiscvReg, imm: i32) -> u32 {
    i_type(imm, rs1, 0b000, rd, 0b0010011)
}

#[inline]
pub(super) fn sltiu(rd: RiscvReg, rs1: RiscvReg, imm: i32) -> u32 {
    i_type(imm, rs1, 0b011, rd, 0b0010011)
}

#[inline]
pub(super) fn xori(rd: RiscvReg, rs1: RiscvReg, imm: i32) -> u32 {
    i_type(imm, rs1, 0b100, rd, 0b0010011)
}

#[inline]
pub(super) fn ori(rd: RiscvReg, rs1: RiscvReg, imm: i32) -> u32 {
    i_type(imm, rs1, 0b110, rd, 0b0010011)
}

#[inline]
pub(super) fn slli(rd: RiscvReg, rs1: RiscvReg, shamt: u32) -> u32 {
    i_type((shamt & 0x3f) as i32, rs1, 0b001, rd, 0b0010011)
}

#[inline]
pub(super) fn srli(rd: RiscvReg, rs1: RiscvReg, shamt: u32) -> u32 {
    i_type((shamt & 0x3f) as i32, rs1, 0b101, rd, 0b0010011)
}

#[inline]
pub(super) fn srai(rd: RiscvReg, rs1: RiscvReg, shamt: u32) -> u32 {
    i_type((0x400 | (shamt & 0x3f)) as i32, rs1, 0b101, rd, 0b0010011)
}

#[inline]
pub(super) fn addiw(rd: RiscvReg, rs1: RiscvReg, imm: i32) -> u32 {
    i_type(imm, rs1, 0b000, rd, 0b0011011)
}

#[inline]
pub(super) fn add(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0, rs2, rs1, 0b000, rd, 0b0110011)
}

#[inline]
pub(super) fn sub(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0x20, rs2, rs1, 0b000, rd, 0b0110011)
}

#[inline]
pub(super) fn mul(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0x01, rs2, rs1, 0b000, rd, 0b0110011)
}

#[inline]
pub(super) fn div(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0x01, rs2, rs1, 0b100, rd, 0b0110011)
}

#[inline]
pub(super) fn divu(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0x01, rs2, rs1, 0b101, rd, 0b0110011)
}

#[inline]
pub(super) fn rem(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0x01, rs2, rs1, 0b110, rd, 0b0110011)
}

#[inline]
pub(super) fn remu(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0x01, rs2, rs1, 0b111, rd, 0b0110011)
}

#[inline]
pub(super) fn and(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0, rs2, rs1, 0b111, rd, 0b0110011)
}

#[inline]
pub(super) fn or(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0, rs2, rs1, 0b110, rd, 0b0110011)
}

#[inline]
pub(super) fn xor(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0, rs2, rs1, 0b100, rd, 0b0110011)
}

#[inline]
pub(super) fn sll(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0, rs2, rs1, 0b001, rd, 0b0110011)
}

#[inline]
pub(super) fn srl(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0, rs2, rs1, 0b101, rd, 0b0110011)
}

#[inline]
pub(super) fn sra(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0x20, rs2, rs1, 0b101, rd, 0b0110011)
}

#[inline]
pub(super) fn slt(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0, rs2, rs1, 0b010, rd, 0b0110011)
}

#[inline]
pub(super) fn sltu(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0, rs2, rs1, 0b011, rd, 0b0110011)
}

#[inline]
pub(super) fn addw(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0, rs2, rs1, 0b000, rd, 0b0111011)
}

#[inline]
pub(super) fn subw(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0x20, rs2, rs1, 0b000, rd, 0b0111011)
}

#[inline]
pub(super) fn mulw(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0x01, rs2, rs1, 0b000, rd, 0b0111011)
}

#[inline]
pub(super) fn divw(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0x01, rs2, rs1, 0b100, rd, 0b0111011)
}

#[inline]
pub(super) fn divuw(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0x01, rs2, rs1, 0b101, rd, 0b0111011)
}

#[inline]
pub(super) fn remw(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0x01, rs2, rs1, 0b110, rd, 0b0111011)
}

#[inline]
pub(super) fn remuw(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0x01, rs2, rs1, 0b111, rd, 0b0111011)
}

#[inline]
pub(super) fn sllw(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0, rs2, rs1, 0b001, rd, 0b0111011)
}

#[inline]
pub(super) fn srlw(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0, rs2, rs1, 0b101, rd, 0b0111011)
}

#[inline]
pub(super) fn sraw(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0x20, rs2, rs1, 0b101, rd, 0b0111011)
}

#[inline]
pub(super) fn load(funct3: u32, rd: RiscvReg, rs1: RiscvReg, imm: i32) -> u32 {
    i_type(imm, rs1, funct3, rd, 0b0000011)
}

#[inline]
pub(super) fn store(funct3: u32, rs2: RiscvReg, rs1: RiscvReg, imm: i32) -> u32 {
    s_type(imm, rs2, rs1, funct3, 0b0100011)
}

#[inline]
pub(super) fn fp_load(funct3: u32, rd: RiscvFpReg, rs1: RiscvReg, imm: i32) -> u32 {
    (((imm as u32) & 0x0fff) << 20)
        | (rs1.idx() << 15)
        | (funct3 << 12)
        | (rd.idx() << 7)
        | 0b0000111
}

#[inline]
pub(super) fn fp_store(funct3: u32, rs2: RiscvFpReg, rs1: RiscvReg, imm: i32) -> u32 {
    let imm = (imm as u32) & 0x0fff;
    ((imm >> 5) << 25)
        | (rs2.idx() << 20)
        | (rs1.idx() << 15)
        | (funct3 << 12)
        | ((imm & 0x1f) << 7)
        | 0b0100111
}

#[inline]
pub(super) fn jalr(rd: RiscvReg, rs1: RiscvReg, imm: i32) -> u32 {
    i_type(imm, rs1, 0b000, rd, 0b1100111)
}

#[inline]
pub(super) fn auipc(rd: RiscvReg, imm20: u32) -> u32 {
    u_type(imm20, rd, 0b0010111)
}

#[inline]
pub(super) fn nop() -> u32 {
    addi(RiscvReg::ZERO, RiscvReg::ZERO, 0)
}

#[inline]
pub(super) fn ret() -> u32 {
    jalr(RiscvReg::ZERO, RiscvReg::RA, 0)
}

const RNE: u32 = 0b000;
const RTZ: u32 = 0b001;
const RDN: u32 = 0b010;
const RUP: u32 = 0b011;

#[inline]
pub(super) fn fmv_s(rd: RiscvFpReg, rs: RiscvFpReg) -> u32 {
    fp_r_type(0b0010000, rs, rs, 0b000, rd)
}

#[inline]
pub(super) fn fmv_d(rd: RiscvFpReg, rs: RiscvFpReg) -> u32 {
    fp_r_type(0b0010001, rs, rs, 0b000, rd)
}

#[inline]
pub(super) fn fabs_s(rd: RiscvFpReg, rs: RiscvFpReg) -> u32 {
    fp_r_type(0b0010000, rs, rs, 0b010, rd)
}

#[inline]
pub(super) fn fabs_d(rd: RiscvFpReg, rs: RiscvFpReg) -> u32 {
    fp_r_type(0b0010001, rs, rs, 0b010, rd)
}

#[inline]
pub(super) fn fneg_s(rd: RiscvFpReg, rs: RiscvFpReg) -> u32 {
    fp_r_type(0b0010000, rs, rs, 0b001, rd)
}

#[inline]
pub(super) fn fneg_d(rd: RiscvFpReg, rs: RiscvFpReg) -> u32 {
    fp_r_type(0b0010001, rs, rs, 0b001, rd)
}

#[inline]
pub(super) fn fsgnj_s(rd: RiscvFpReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_r_type(0b0010000, rhs, lhs, 0b000, rd)
}

#[inline]
pub(super) fn fsgnj_d(rd: RiscvFpReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_r_type(0b0010001, rhs, lhs, 0b000, rd)
}

#[inline]
pub(super) fn fadd_s(rd: RiscvFpReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_r_type(0b0000000, rhs, lhs, RNE, rd)
}

#[inline]
pub(super) fn fadd_d(rd: RiscvFpReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_r_type(0b0000001, rhs, lhs, RNE, rd)
}

#[inline]
pub(super) fn fsub_s(rd: RiscvFpReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_r_type(0b0000100, rhs, lhs, RNE, rd)
}

#[inline]
pub(super) fn fsub_d(rd: RiscvFpReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_r_type(0b0000101, rhs, lhs, RNE, rd)
}

#[inline]
pub(super) fn fmul_s(rd: RiscvFpReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_r_type(0b0001000, rhs, lhs, RNE, rd)
}

#[inline]
pub(super) fn fmul_d(rd: RiscvFpReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_r_type(0b0001001, rhs, lhs, RNE, rd)
}

#[inline]
pub(super) fn fdiv_s(rd: RiscvFpReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_r_type(0b0001100, rhs, lhs, RNE, rd)
}

#[inline]
pub(super) fn fdiv_d(rd: RiscvFpReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_r_type(0b0001101, rhs, lhs, RNE, rd)
}

#[inline]
pub(super) fn fsqrt_s(rd: RiscvFpReg, rs: RiscvFpReg) -> u32 {
    fp_r_type(0b0101100, RiscvFpReg::from_raw(0), rs, RNE, rd)
}

#[inline]
pub(super) fn fsqrt_d(rd: RiscvFpReg, rs: RiscvFpReg) -> u32 {
    fp_r_type(0b0101101, RiscvFpReg::from_raw(0), rs, RNE, rd)
}

#[inline]
pub(super) fn fmin_s(rd: RiscvFpReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_r_type(0b0010100, rhs, lhs, 0b000, rd)
}

#[inline]
pub(super) fn fmin_d(rd: RiscvFpReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_r_type(0b0010101, rhs, lhs, 0b000, rd)
}

#[inline]
pub(super) fn fmax_s(rd: RiscvFpReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_r_type(0b0010100, rhs, lhs, 0b001, rd)
}

#[inline]
pub(super) fn fmax_d(rd: RiscvFpReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_r_type(0b0010101, rhs, lhs, 0b001, rd)
}

#[inline]
pub(super) fn feq_s(rd: RiscvReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_to_gp_type(0b1010000, rhs.idx(), lhs, 0b010, rd)
}

#[inline]
pub(super) fn feq_d(rd: RiscvReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_to_gp_type(0b1010001, rhs.idx(), lhs, 0b010, rd)
}

#[inline]
pub(super) fn flt_s(rd: RiscvReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_to_gp_type(0b1010000, rhs.idx(), lhs, 0b001, rd)
}

#[inline]
pub(super) fn flt_d(rd: RiscvReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_to_gp_type(0b1010001, rhs.idx(), lhs, 0b001, rd)
}

#[inline]
pub(super) fn fle_s(rd: RiscvReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_to_gp_type(0b1010000, rhs.idx(), lhs, 0b000, rd)
}

#[inline]
pub(super) fn fle_d(rd: RiscvReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_to_gp_type(0b1010001, rhs.idx(), lhs, 0b000, rd)
}

#[inline]
pub(super) fn fmv_x_w(rd: RiscvReg, rs: RiscvFpReg) -> u32 {
    fp_to_gp_type(0b1110000, 0, rs, 0b000, rd)
}

#[inline]
pub(super) fn fmv_x_d(rd: RiscvReg, rs: RiscvFpReg) -> u32 {
    fp_to_gp_type(0b1110001, 0, rs, 0b000, rd)
}

#[inline]
pub(super) fn fmv_w_x(rd: RiscvFpReg, rs: RiscvReg) -> u32 {
    gp_to_fp_type(0b1111000, 0, rs, 0b000, rd)
}

#[inline]
pub(super) fn fmv_d_x(rd: RiscvFpReg, rs: RiscvReg) -> u32 {
    gp_to_fp_type(0b1111001, 0, rs, 0b000, rd)
}

#[inline]
pub(super) fn fcvt_s_d(rd: RiscvFpReg, rs: RiscvFpReg) -> u32 {
    fp_r_type(0b0100000, RiscvFpReg::from_raw(1), rs, RNE, rd)
}

#[inline]
pub(super) fn fcvt_d_s(rd: RiscvFpReg, rs: RiscvFpReg) -> u32 {
    fp_r_type(0b0100001, RiscvFpReg::from_raw(0), rs, RNE, rd)
}

#[inline]
pub(super) fn fcvt_s_w(rd: RiscvFpReg, rs: RiscvReg) -> u32 {
    gp_to_fp_type(0b1101000, 0, rs, RNE, rd)
}

#[inline]
pub(super) fn fcvt_s_wu(rd: RiscvFpReg, rs: RiscvReg) -> u32 {
    gp_to_fp_type(0b1101000, 1, rs, RNE, rd)
}

#[inline]
pub(super) fn fcvt_s_l(rd: RiscvFpReg, rs: RiscvReg) -> u32 {
    gp_to_fp_type(0b1101000, 2, rs, RNE, rd)
}

#[inline]
pub(super) fn fcvt_s_lu(rd: RiscvFpReg, rs: RiscvReg) -> u32 {
    gp_to_fp_type(0b1101000, 3, rs, RNE, rd)
}

#[inline]
pub(super) fn fcvt_d_w(rd: RiscvFpReg, rs: RiscvReg) -> u32 {
    gp_to_fp_type(0b1101001, 0, rs, RNE, rd)
}

#[inline]
pub(super) fn fcvt_d_wu(rd: RiscvFpReg, rs: RiscvReg) -> u32 {
    gp_to_fp_type(0b1101001, 1, rs, RNE, rd)
}

#[inline]
pub(super) fn fcvt_d_l(rd: RiscvFpReg, rs: RiscvReg) -> u32 {
    gp_to_fp_type(0b1101001, 2, rs, RNE, rd)
}

#[inline]
pub(super) fn fcvt_d_lu(rd: RiscvFpReg, rs: RiscvReg) -> u32 {
    gp_to_fp_type(0b1101001, 3, rs, RNE, rd)
}

#[inline]
pub(super) fn fcvt_w_s_rtz(rd: RiscvReg, rs: RiscvFpReg) -> u32 {
    fp_to_gp_type(0b1100000, 0, rs, RTZ, rd)
}

#[inline]
pub(super) fn fcvt_wu_s_rtz(rd: RiscvReg, rs: RiscvFpReg) -> u32 {
    fp_to_gp_type(0b1100000, 1, rs, RTZ, rd)
}

#[inline]
pub(super) fn fcvt_l_s_rtz(rd: RiscvReg, rs: RiscvFpReg) -> u32 {
    fp_to_gp_type(0b1100000, 2, rs, RTZ, rd)
}

#[inline]
pub(super) fn fcvt_lu_s_rtz(rd: RiscvReg, rs: RiscvFpReg) -> u32 {
    fp_to_gp_type(0b1100000, 3, rs, RTZ, rd)
}

#[inline]
pub(super) fn fcvt_w_d_rtz(rd: RiscvReg, rs: RiscvFpReg) -> u32 {
    fp_to_gp_type(0b1100001, 0, rs, RTZ, rd)
}

#[inline]
pub(super) fn fcvt_wu_d_rtz(rd: RiscvReg, rs: RiscvFpReg) -> u32 {
    fp_to_gp_type(0b1100001, 1, rs, RTZ, rd)
}

#[inline]
pub(super) fn fcvt_l_d_rtz(rd: RiscvReg, rs: RiscvFpReg) -> u32 {
    fp_to_gp_type(0b1100001, 2, rs, RTZ, rd)
}

#[inline]
pub(super) fn fcvt_lu_d_rtz(rd: RiscvReg, rs: RiscvFpReg) -> u32 {
    fp_to_gp_type(0b1100001, 3, rs, RTZ, rd)
}

#[inline]
pub(super) fn fcvt_l_s_round(rd: RiscvReg, rs: RiscvFpReg, rm: u32) -> u32 {
    fp_to_gp_type(0b1100000, 2, rs, rm, rd)
}

#[inline]
pub(super) fn fcvt_l_d_round(rd: RiscvReg, rs: RiscvFpReg, rm: u32) -> u32 {
    fp_to_gp_type(0b1100001, 2, rs, rm, rd)
}

pub(super) const ROUND_RNE: u32 = RNE;
pub(super) const ROUND_RTZ: u32 = RTZ;
pub(super) const ROUND_RDN: u32 = RDN;
pub(super) const ROUND_RUP: u32 = RUP;
