//! Pure instruction encoder for the example backend.
//!
//! The opcodes are intentionally fake. The important part is that every public
//! function takes typed physical registers and returns bytes/words only.

use super::reg::{FpReg, GpReg};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(super) enum Cond {
    Eq = 0x0,
    Ne = 0x1,
    Hs = 0x2,
    Lo = 0x3,
    Mi = 0x4,
    Hi = 0x8,
    Ls = 0x9,
    Ge = 0xA,
    Lt = 0xB,
    Gt = 0xC,
    Le = 0xD,
}

fn encode_rrr(op: u32, rd: GpReg, rn: GpReg, rm: GpReg) -> u32 {
    (op << 24)
        | ((rd.index() as u32) << 16)
        | ((rn.index() as u32) << 8)
        | (rm.index() as u32)
}

fn encode_rri(op: u32, rd: GpReg, rn: GpReg, imm: u16) -> u32 {
    (op << 24)
        | ((rd.index() as u32) << 16)
        | ((rn.index() as u32) << 8)
        | (imm as u32)
}

fn encode_frr(op: u32, rd: FpReg, rn: FpReg) -> u32 {
    (op << 24) | ((rd.index() as u32) << 8) | (rn.index() as u32)
}

fn encode_frrr(op: u32, rd: FpReg, rn: FpReg, rm: FpReg) -> u32 {
    (op << 24)
        | ((rd.index() as u32) << 16)
        | ((rn.index() as u32) << 8)
        | (rm.index() as u32)
}

pub(super) fn mov_reg_64(rd: GpReg, rm: GpReg) -> u32 {
    encode_rrr(0x01, rd, GpReg::ZR, rm)
}

pub(super) fn add_reg_32(rd: GpReg, rn: GpReg, rm: GpReg) -> u32 {
    encode_rrr(0x42, rd, rn, rm)
}

pub(super) fn add_reg_64(rd: GpReg, rn: GpReg, rm: GpReg) -> u32 {
    encode_rrr(0x02, rd, rn, rm)
}

pub(super) fn sub_reg_32(rd: GpReg, rn: GpReg, rm: GpReg) -> u32 {
    encode_rrr(0x43, rd, rn, rm)
}

pub(super) fn sub_reg_64(rd: GpReg, rn: GpReg, rm: GpReg) -> u32 {
    encode_rrr(0x03, rd, rn, rm)
}

pub(super) fn add_imm_64(rd: GpReg, rn: GpReg, imm12: u16) -> u32 {
    encode_rri(0x04, rd, rn, imm12)
}

pub(super) fn sub_imm_64(rd: GpReg, rn: GpReg, imm12: u16) -> u32 {
    encode_rri(0x05, rd, rn, imm12)
}

pub(super) fn cmp_reg_64(rn: GpReg, rm: GpReg) -> u32 {
    encode_rrr(0x06, GpReg::ZR, rn, rm)
}

pub(super) fn cmp_imm_64(rn: GpReg, imm12: u16) -> u32 {
    encode_rri(0x07, GpReg::ZR, rn, imm12)
}

pub(super) fn csel_64(
    rd: GpReg,
    rn: GpReg,
    rm: GpReg,
    cond: Cond,
) -> u32 {
    encode_rrr(0x08 | ((cond as u32) << 4), rd, rn, rm)
}

pub(super) fn ldr_64(rt: GpReg, rn: GpReg, slot: u16) -> u32 {
    encode_rri(0x09, rt, rn, slot)
}

pub(super) fn str_64(rt: GpReg, rn: GpReg, slot: u16) -> u32 {
    encode_rri(0x0A, rt, rn, slot)
}

pub(super) fn fmov_gp_from_d(rd: GpReg, rn: FpReg) -> u32 {
    (0x10 << 24) | ((rd.index() as u32) << 8) | (rn.index() as u32)
}

pub(super) fn fmov_d_from_gp(rd: FpReg, rn: GpReg) -> u32 {
    (0x11 << 24) | ((rd.index() as u32) << 8) | (rn.index() as u32)
}

pub(super) fn fmov_d(rd: FpReg, rn: FpReg) -> u32 {
    encode_frr(0x12, rd, rn)
}

pub(super) fn fadd_d(rd: FpReg, rn: FpReg, rm: FpReg) -> u32 {
    encode_frrr(0x13, rd, rn, rm)
}

pub(super) fn fcmp_d(rn: FpReg, rm: FpReg) -> u32 {
    encode_frrr(0x14, rn, rn, rm)
}

pub(super) fn b(imm26: i32) -> u32 {
    (0x20 << 24) | ((imm26 as u32) & 0x00ff_ffff)
}

pub(super) fn b_cond(cond: Cond, imm19: i32) -> u32 {
    (0x21 << 24) | ((cond as u32) << 20) | ((imm19 as u32) & 0x000f_ffff)
}

pub(super) fn cbz_64(rt: GpReg, imm19: i32) -> u32 {
    (0x22 << 24) | ((rt.index() as u32) << 16) | ((imm19 as u32) & 0x0000_ffff)
}

pub(super) fn cbnz_64(rt: GpReg, imm19: i32) -> u32 {
    (0x23 << 24) | ((rt.index() as u32) << 16) | ((imm19 as u32) & 0x0000_ffff)
}

pub(super) fn br(rn: GpReg) -> u32 {
    (0x24 << 24) | (rn.index() as u32)
}

pub(super) fn ret() -> u32 {
    (0x25 << 24) | (GpReg::LR.index() as u32)
}

pub(super) fn movz_64(rd: GpReg, imm16: u16, shift: u8) -> u32 {
    (0x26 << 24) | ((rd.index() as u32) << 16) | ((shift as u32) << 8) | (imm16 as u32)
}

pub(super) fn movk_64(rd: GpReg, imm16: u16, shift: u8) -> u32 {
    (0x27 << 24) | ((rd.index() as u32) << 16) | ((shift as u32) << 8) | (imm16 as u32)
}
