//! Minimal RISC-V encoder shared by RV64GC and RV32GC backends.
//!
//! The backend intentionally emits only 32-bit base encodings for now, even on
//! targets with the compressed `C` extension. That keeps patch offsets simple
//! while the port is coming up.

use super::reg::{RiscvFpReg, RiscvReg};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Cond {
    Eq,
    Ne,
    Lt,
    Ge,
    Ltu,
    Geu,
}

impl Cond {
    #[inline]
    pub(crate) const fn invert(self) -> Self {
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
pub(crate) fn b_type(cond: Cond, rs1: RiscvReg, rs2: RiscvReg, offset: i32) -> u32 {
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
pub(crate) fn jal(rd: RiscvReg, offset: i32) -> u32 {
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
#[cfg(sf_backend_riscv32)]
pub(crate) fn lui(rd: RiscvReg, imm20: u32) -> u32 {
    u_type(imm20 & 0x000f_ffff, rd, 0b0110111)
}

#[inline]
pub(crate) fn addi(rd: RiscvReg, rs1: RiscvReg, imm: i32) -> u32 {
    i_type(imm, rs1, 0b000, rd, 0b0010011)
}

#[inline]
pub(crate) fn sltiu(rd: RiscvReg, rs1: RiscvReg, imm: i32) -> u32 {
    i_type(imm, rs1, 0b011, rd, 0b0010011)
}

#[inline]
#[cfg(sf_backend_riscv32)]
pub(crate) fn andi(rd: RiscvReg, rs1: RiscvReg, imm: i32) -> u32 {
    i_type(imm, rs1, 0b111, rd, 0b0010011)
}

#[inline]
pub(crate) fn xori(rd: RiscvReg, rs1: RiscvReg, imm: i32) -> u32 {
    i_type(imm, rs1, 0b100, rd, 0b0010011)
}

#[inline]
pub(crate) fn ori(rd: RiscvReg, rs1: RiscvReg, imm: i32) -> u32 {
    i_type(imm, rs1, 0b110, rd, 0b0010011)
}

#[inline]
pub(crate) fn slli(rd: RiscvReg, rs1: RiscvReg, shamt: u32) -> u32 {
    i_type((shamt & 0x3f) as i32, rs1, 0b001, rd, 0b0010011)
}

#[inline]
pub(crate) fn srli(rd: RiscvReg, rs1: RiscvReg, shamt: u32) -> u32 {
    i_type((shamt & 0x3f) as i32, rs1, 0b101, rd, 0b0010011)
}

#[inline]
pub(crate) fn srai(rd: RiscvReg, rs1: RiscvReg, shamt: u32) -> u32 {
    i_type((0x400 | (shamt & 0x3f)) as i32, rs1, 0b101, rd, 0b0010011)
}

#[inline]
#[cfg(sf_backend_riscv64)]
pub(crate) fn addiw(rd: RiscvReg, rs1: RiscvReg, imm: i32) -> u32 {
    i_type(imm, rs1, 0b000, rd, 0b0011011)
}

#[inline]
pub(crate) fn add(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0, rs2, rs1, 0b000, rd, 0b0110011)
}

#[inline]
pub(crate) fn sub(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0x20, rs2, rs1, 0b000, rd, 0b0110011)
}

#[inline]
pub(crate) fn mul(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0x01, rs2, rs1, 0b000, rd, 0b0110011)
}

#[inline]
#[cfg(sf_backend_riscv32)]
pub(crate) fn mulh(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0x01, rs2, rs1, 0b001, rd, 0b0110011)
}

#[inline]
#[cfg(sf_backend_riscv32)]
pub(crate) fn mulhu(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0x01, rs2, rs1, 0b011, rd, 0b0110011)
}

#[inline]
pub(crate) fn div(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0x01, rs2, rs1, 0b100, rd, 0b0110011)
}

#[inline]
pub(crate) fn divu(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0x01, rs2, rs1, 0b101, rd, 0b0110011)
}

#[inline]
pub(crate) fn rem(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0x01, rs2, rs1, 0b110, rd, 0b0110011)
}

#[inline]
pub(crate) fn remu(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0x01, rs2, rs1, 0b111, rd, 0b0110011)
}

#[inline]
pub(crate) fn and(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0, rs2, rs1, 0b111, rd, 0b0110011)
}

#[inline]
pub(crate) fn or(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0, rs2, rs1, 0b110, rd, 0b0110011)
}

#[inline]
pub(crate) fn xor(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0, rs2, rs1, 0b100, rd, 0b0110011)
}

#[inline]
pub(crate) fn sll(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0, rs2, rs1, 0b001, rd, 0b0110011)
}

#[inline]
pub(crate) fn srl(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0, rs2, rs1, 0b101, rd, 0b0110011)
}

#[inline]
pub(crate) fn sra(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0x20, rs2, rs1, 0b101, rd, 0b0110011)
}

#[inline]
pub(crate) fn slt(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0, rs2, rs1, 0b010, rd, 0b0110011)
}

#[inline]
pub(crate) fn sltu(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0, rs2, rs1, 0b011, rd, 0b0110011)
}

#[inline]
#[cfg(sf_backend_riscv64)]
pub(crate) fn addw(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0, rs2, rs1, 0b000, rd, 0b0111011)
}

#[inline]
#[cfg(sf_backend_riscv64)]
pub(crate) fn subw(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0x20, rs2, rs1, 0b000, rd, 0b0111011)
}

#[inline]
#[cfg(sf_backend_riscv64)]
pub(crate) fn mulw(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0x01, rs2, rs1, 0b000, rd, 0b0111011)
}

#[inline]
#[cfg(sf_backend_riscv64)]
pub(crate) fn divw(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0x01, rs2, rs1, 0b100, rd, 0b0111011)
}

#[inline]
#[cfg(sf_backend_riscv64)]
pub(crate) fn divuw(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0x01, rs2, rs1, 0b101, rd, 0b0111011)
}

#[inline]
#[cfg(sf_backend_riscv64)]
pub(crate) fn remw(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0x01, rs2, rs1, 0b110, rd, 0b0111011)
}

#[inline]
#[cfg(sf_backend_riscv64)]
pub(crate) fn remuw(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0x01, rs2, rs1, 0b111, rd, 0b0111011)
}

#[inline]
#[cfg(sf_backend_riscv64)]
pub(crate) fn sllw(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0, rs2, rs1, 0b001, rd, 0b0111011)
}

#[inline]
#[cfg(sf_backend_riscv64)]
pub(crate) fn srlw(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0, rs2, rs1, 0b101, rd, 0b0111011)
}

#[inline]
#[cfg(sf_backend_riscv64)]
pub(crate) fn sraw(rd: RiscvReg, rs1: RiscvReg, rs2: RiscvReg) -> u32 {
    r_type(0x20, rs2, rs1, 0b101, rd, 0b0111011)
}

#[inline]
pub(crate) fn load(funct3: u32, rd: RiscvReg, rs1: RiscvReg, imm: i32) -> u32 {
    i_type(imm, rs1, funct3, rd, 0b0000011)
}

#[inline]
pub(crate) fn store(funct3: u32, rs2: RiscvReg, rs1: RiscvReg, imm: i32) -> u32 {
    s_type(imm, rs2, rs1, funct3, 0b0100011)
}

#[inline]
pub(crate) fn fp_load(funct3: u32, rd: RiscvFpReg, rs1: RiscvReg, imm: i32) -> u32 {
    (((imm as u32) & 0x0fff) << 20)
        | (rs1.idx() << 15)
        | (funct3 << 12)
        | (rd.idx() << 7)
        | 0b0000111
}

#[inline]
pub(crate) fn fp_store(funct3: u32, rs2: RiscvFpReg, rs1: RiscvReg, imm: i32) -> u32 {
    let imm = (imm as u32) & 0x0fff;
    ((imm >> 5) << 25)
        | (rs2.idx() << 20)
        | (rs1.idx() << 15)
        | (funct3 << 12)
        | ((imm & 0x1f) << 7)
        | 0b0100111
}

#[inline]
pub(crate) fn jalr(rd: RiscvReg, rs1: RiscvReg, imm: i32) -> u32 {
    i_type(imm, rs1, 0b000, rd, 0b1100111)
}

#[inline]
pub(crate) fn auipc(rd: RiscvReg, imm20: u32) -> u32 {
    u_type(imm20, rd, 0b0010111)
}

#[inline]
pub(crate) fn nop() -> u32 {
    addi(RiscvReg::ZERO, RiscvReg::ZERO, 0)
}

#[inline]
pub(crate) fn ret() -> u32 {
    jalr(RiscvReg::ZERO, RiscvReg::RA, 0)
}

const RNE: u32 = 0b000;
const RTZ: u32 = 0b001;
#[cfg(sf_backend_riscv64)]
const RDN: u32 = 0b010;
#[cfg(sf_backend_riscv64)]
const RUP: u32 = 0b011;

#[inline]
pub(crate) fn fmv_s(rd: RiscvFpReg, rs: RiscvFpReg) -> u32 {
    fp_r_type(0b0010000, rs, rs, 0b000, rd)
}

#[inline]
pub(crate) fn fmv_d(rd: RiscvFpReg, rs: RiscvFpReg) -> u32 {
    fp_r_type(0b0010001, rs, rs, 0b000, rd)
}

#[inline]
pub(crate) fn fabs_s(rd: RiscvFpReg, rs: RiscvFpReg) -> u32 {
    fp_r_type(0b0010000, rs, rs, 0b010, rd)
}

#[inline]
pub(crate) fn fabs_d(rd: RiscvFpReg, rs: RiscvFpReg) -> u32 {
    fp_r_type(0b0010001, rs, rs, 0b010, rd)
}

#[inline]
pub(crate) fn fneg_s(rd: RiscvFpReg, rs: RiscvFpReg) -> u32 {
    fp_r_type(0b0010000, rs, rs, 0b001, rd)
}

#[inline]
pub(crate) fn fneg_d(rd: RiscvFpReg, rs: RiscvFpReg) -> u32 {
    fp_r_type(0b0010001, rs, rs, 0b001, rd)
}

#[inline]
pub(crate) fn fsgnj_s(rd: RiscvFpReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_r_type(0b0010000, rhs, lhs, 0b000, rd)
}

#[inline]
pub(crate) fn fsgnj_d(rd: RiscvFpReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_r_type(0b0010001, rhs, lhs, 0b000, rd)
}

#[inline]
pub(crate) fn fadd_s(rd: RiscvFpReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_r_type(0b0000000, rhs, lhs, RNE, rd)
}

#[inline]
pub(crate) fn fadd_d(rd: RiscvFpReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_r_type(0b0000001, rhs, lhs, RNE, rd)
}

#[inline]
pub(crate) fn fsub_s(rd: RiscvFpReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_r_type(0b0000100, rhs, lhs, RNE, rd)
}

#[inline]
pub(crate) fn fsub_d(rd: RiscvFpReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_r_type(0b0000101, rhs, lhs, RNE, rd)
}

#[inline]
pub(crate) fn fmul_s(rd: RiscvFpReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_r_type(0b0001000, rhs, lhs, RNE, rd)
}

#[inline]
pub(crate) fn fmul_d(rd: RiscvFpReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_r_type(0b0001001, rhs, lhs, RNE, rd)
}

#[inline]
pub(crate) fn fdiv_s(rd: RiscvFpReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_r_type(0b0001100, rhs, lhs, RNE, rd)
}

#[inline]
pub(crate) fn fdiv_d(rd: RiscvFpReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_r_type(0b0001101, rhs, lhs, RNE, rd)
}

#[inline]
pub(crate) fn fsqrt_s(rd: RiscvFpReg, rs: RiscvFpReg) -> u32 {
    fp_r_type(0b0101100, RiscvFpReg::from_raw(0), rs, RNE, rd)
}

#[inline]
pub(crate) fn fsqrt_d(rd: RiscvFpReg, rs: RiscvFpReg) -> u32 {
    fp_r_type(0b0101101, RiscvFpReg::from_raw(0), rs, RNE, rd)
}

#[inline]
pub(crate) fn fmin_s(rd: RiscvFpReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_r_type(0b0010100, rhs, lhs, 0b000, rd)
}

#[inline]
pub(crate) fn fmin_d(rd: RiscvFpReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_r_type(0b0010101, rhs, lhs, 0b000, rd)
}

#[inline]
pub(crate) fn fmax_s(rd: RiscvFpReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_r_type(0b0010100, rhs, lhs, 0b001, rd)
}

#[inline]
pub(crate) fn fmax_d(rd: RiscvFpReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_r_type(0b0010101, rhs, lhs, 0b001, rd)
}

#[inline]
pub(crate) fn feq_s(rd: RiscvReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_to_gp_type(0b1010000, rhs.idx(), lhs, 0b010, rd)
}

#[inline]
pub(crate) fn feq_d(rd: RiscvReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_to_gp_type(0b1010001, rhs.idx(), lhs, 0b010, rd)
}

#[inline]
pub(crate) fn flt_s(rd: RiscvReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_to_gp_type(0b1010000, rhs.idx(), lhs, 0b001, rd)
}

#[inline]
pub(crate) fn flt_d(rd: RiscvReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_to_gp_type(0b1010001, rhs.idx(), lhs, 0b001, rd)
}

#[inline]
pub(crate) fn fle_s(rd: RiscvReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_to_gp_type(0b1010000, rhs.idx(), lhs, 0b000, rd)
}

#[inline]
pub(crate) fn fle_d(rd: RiscvReg, lhs: RiscvFpReg, rhs: RiscvFpReg) -> u32 {
    fp_to_gp_type(0b1010001, rhs.idx(), lhs, 0b000, rd)
}

#[inline]
pub(crate) fn fmv_x_w(rd: RiscvReg, rs: RiscvFpReg) -> u32 {
    fp_to_gp_type(0b1110000, 0, rs, 0b000, rd)
}

#[inline]
#[cfg(sf_backend_riscv64)]
pub(crate) fn fmv_x_d(rd: RiscvReg, rs: RiscvFpReg) -> u32 {
    fp_to_gp_type(0b1110001, 0, rs, 0b000, rd)
}

#[inline]
pub(crate) fn fmv_w_x(rd: RiscvFpReg, rs: RiscvReg) -> u32 {
    gp_to_fp_type(0b1111000, 0, rs, 0b000, rd)
}

#[inline]
#[cfg(sf_backend_riscv64)]
pub(crate) fn fmv_d_x(rd: RiscvFpReg, rs: RiscvReg) -> u32 {
    gp_to_fp_type(0b1111001, 0, rs, 0b000, rd)
}

#[inline]
pub(crate) fn fcvt_s_d(rd: RiscvFpReg, rs: RiscvFpReg) -> u32 {
    fp_r_type(0b0100000, RiscvFpReg::from_raw(1), rs, RNE, rd)
}

#[inline]
pub(crate) fn fcvt_d_s(rd: RiscvFpReg, rs: RiscvFpReg) -> u32 {
    fp_r_type(0b0100001, RiscvFpReg::from_raw(0), rs, RNE, rd)
}

#[inline]
pub(crate) fn fcvt_s_w(rd: RiscvFpReg, rs: RiscvReg) -> u32 {
    gp_to_fp_type(0b1101000, 0, rs, RNE, rd)
}

#[inline]
pub(crate) fn fcvt_s_wu(rd: RiscvFpReg, rs: RiscvReg) -> u32 {
    gp_to_fp_type(0b1101000, 1, rs, RNE, rd)
}

#[inline]
#[cfg(sf_backend_riscv64)]
pub(crate) fn fcvt_s_l(rd: RiscvFpReg, rs: RiscvReg) -> u32 {
    gp_to_fp_type(0b1101000, 2, rs, RNE, rd)
}

#[inline]
#[cfg(sf_backend_riscv64)]
pub(crate) fn fcvt_s_lu(rd: RiscvFpReg, rs: RiscvReg) -> u32 {
    gp_to_fp_type(0b1101000, 3, rs, RNE, rd)
}

#[inline]
pub(crate) fn fcvt_d_w(rd: RiscvFpReg, rs: RiscvReg) -> u32 {
    gp_to_fp_type(0b1101001, 0, rs, RNE, rd)
}

#[inline]
pub(crate) fn fcvt_d_wu(rd: RiscvFpReg, rs: RiscvReg) -> u32 {
    gp_to_fp_type(0b1101001, 1, rs, RNE, rd)
}

#[inline]
#[cfg(sf_backend_riscv64)]
pub(crate) fn fcvt_d_l(rd: RiscvFpReg, rs: RiscvReg) -> u32 {
    gp_to_fp_type(0b1101001, 2, rs, RNE, rd)
}

#[inline]
#[cfg(sf_backend_riscv64)]
pub(crate) fn fcvt_d_lu(rd: RiscvFpReg, rs: RiscvReg) -> u32 {
    gp_to_fp_type(0b1101001, 3, rs, RNE, rd)
}

#[inline]
pub(crate) fn fcvt_w_s_rtz(rd: RiscvReg, rs: RiscvFpReg) -> u32 {
    fp_to_gp_type(0b1100000, 0, rs, RTZ, rd)
}

#[inline]
pub(crate) fn fcvt_wu_s_rtz(rd: RiscvReg, rs: RiscvFpReg) -> u32 {
    fp_to_gp_type(0b1100000, 1, rs, RTZ, rd)
}

#[inline]
#[cfg(sf_backend_riscv64)]
pub(crate) fn fcvt_l_s_rtz(rd: RiscvReg, rs: RiscvFpReg) -> u32 {
    fp_to_gp_type(0b1100000, 2, rs, RTZ, rd)
}

#[inline]
#[cfg(sf_backend_riscv64)]
pub(crate) fn fcvt_lu_s_rtz(rd: RiscvReg, rs: RiscvFpReg) -> u32 {
    fp_to_gp_type(0b1100000, 3, rs, RTZ, rd)
}

#[inline]
pub(crate) fn fcvt_w_d_rtz(rd: RiscvReg, rs: RiscvFpReg) -> u32 {
    fp_to_gp_type(0b1100001, 0, rs, RTZ, rd)
}

#[inline]
pub(crate) fn fcvt_wu_d_rtz(rd: RiscvReg, rs: RiscvFpReg) -> u32 {
    fp_to_gp_type(0b1100001, 1, rs, RTZ, rd)
}

#[inline]
#[cfg(sf_backend_riscv64)]
pub(crate) fn fcvt_l_d_rtz(rd: RiscvReg, rs: RiscvFpReg) -> u32 {
    fp_to_gp_type(0b1100001, 2, rs, RTZ, rd)
}

#[inline]
#[cfg(sf_backend_riscv64)]
pub(crate) fn fcvt_lu_d_rtz(rd: RiscvReg, rs: RiscvFpReg) -> u32 {
    fp_to_gp_type(0b1100001, 3, rs, RTZ, rd)
}

#[inline]
#[cfg(sf_backend_riscv64)]
pub(crate) fn fcvt_l_s_round(rd: RiscvReg, rs: RiscvFpReg, rm: u32) -> u32 {
    fp_to_gp_type(0b1100000, 2, rs, rm, rd)
}

#[inline]
#[cfg(sf_backend_riscv64)]
pub(crate) fn fcvt_l_d_round(rd: RiscvReg, rs: RiscvFpReg, rm: u32) -> u32 {
    fp_to_gp_type(0b1100001, 2, rs, rm, rd)
}

#[cfg(sf_backend_riscv64)]
pub(crate) const ROUND_RNE: u32 = RNE;
#[cfg(sf_backend_riscv64)]
pub(crate) const ROUND_RTZ: u32 = RTZ;
#[cfg(sf_backend_riscv64)]
pub(crate) const ROUND_RDN: u32 = RDN;
#[cfg(sf_backend_riscv64)]
pub(crate) const ROUND_RUP: u32 = RUP;
