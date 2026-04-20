use crate::{
    error::WasmError,
    opcodes::OpcodeFD,
    vm::{
        arch::common::text_emitter::TextEmitter,
        machine::machine_ir::{
            MachineAddr, MachineFloatWidth, MachineInst, MachineInstKind, MachineReg,
            MachineStorageType, MachineValue, MACHINE_CTX_REG,
        },
        runtime::preserved::{io as preserved_io, op as preserved_op, preserved_entry},
    },
};

use super::{
    abi::{self, map_fixed_reg, C_ARG0, C_ARG1, C_ARG2, C_RET0},
    backend::X86_64Backend,
    enc::{self, Cc},
    reg::X86Reg,
};

const V128_PREFIX_BYTES: u32 = 16;

fn unsupported_simd_opcode(kind: &str, opcode: u32) -> WasmError {
    debug_assert!(
        false,
        "x86_64 SIMD native codegen does not support {kind} opcode 0x{opcode:x}"
    );
    WasmError::internal("x86_64 SIMD native codegen does not support this opcode")
}

#[derive(Clone, Copy)]
enum OpMap {
    Map0F,
    Map0F38,
    Map0F3A,
}

#[inline]
const fn rex(w: bool, r: bool, x: bool, b: bool) -> u8 {
    0x40 | ((w as u8) << 3) | ((r as u8) << 2) | ((x as u8) << 1) | (b as u8)
}

#[inline]
const fn modrm(mode: u8, reg: u8, rm: u8) -> u8 {
    (mode << 6) | (reg << 3) | rm
}

#[inline]
const fn sib(scale: u8, index: u8, base: u8) -> u8 {
    (scale << 6) | (index << 3) | base
}

#[inline]
fn fits_i8(v: i32) -> bool {
    (-128..=127).contains(&v)
}

fn emit_i32(e: &mut TextEmitter, v: i32) {
    e.emit_bytes(&v.to_le_bytes());
}

fn emit_simd_opcode_map(e: &mut TextEmitter, map: OpMap) {
    e.emit_u8(0x0F);
    match map {
        OpMap::Map0F => {}
        OpMap::Map0F38 => {
            e.emit_u8(0x38);
        }
        OpMap::Map0F3A => {
            e.emit_u8(0x3A);
        }
    }
}

fn emit_simd_prefix_and_rex(e: &mut TextEmitter, prefix: u8, rex_byte: Option<u8>) {
    if prefix != 0 {
        e.emit_u8(prefix);
    }
    if let Some(rex_byte) = rex_byte {
        e.emit_u8(rex_byte);
    }
}

fn emit_simd_rr(e: &mut TextEmitter, prefix: u8, map: OpMap, opcode: u8, dst: u8, src: u8) {
    let r = dst >= 8;
    let b = src >= 8;
    emit_simd_prefix_and_rex(e, prefix, (r || b).then(|| rex(false, r, false, b)));
    emit_simd_opcode_map(e, map);
    e.emit_u8(opcode);
    e.emit_u8(modrm(0b11, dst & 7, src & 7));
}

fn emit_simd_gp_xmm(e: &mut TextEmitter, prefix: u8, map: OpMap, opcode: u8, dst: X86Reg, src: u8) {
    let r = dst.needs_rex_ext();
    let b = src >= 8;
    emit_simd_prefix_and_rex(e, prefix, (r || b).then(|| rex(false, r, false, b)));
    emit_simd_opcode_map(e, map);
    e.emit_u8(opcode);
    e.emit_u8(modrm(0b11, dst.idx3(), src & 7));
}

fn emit_simd_rr_imm8(
    e: &mut TextEmitter,
    prefix: u8,
    map: OpMap,
    opcode: u8,
    dst: u8,
    src: u8,
    imm8: u8,
) {
    emit_simd_rr(e, prefix, map, opcode, dst, src);
    e.emit_u8(imm8);
}

fn emit_simd_digit_imm8(
    e: &mut TextEmitter,
    prefix: u8,
    map: OpMap,
    opcode: u8,
    digit: u8,
    dst: u8,
    imm8: u8,
) {
    let b = dst >= 8;
    emit_simd_prefix_and_rex(e, prefix, b.then(|| rex(false, false, false, true)));
    emit_simd_opcode_map(e, map);
    e.emit_u8(opcode);
    e.emit_u8(modrm(0b11, digit & 7, dst & 7));
    e.emit_u8(imm8);
}

fn emit_simd_rm(
    e: &mut TextEmitter,
    prefix: u8,
    map: OpMap,
    opcode: u8,
    dst: u8,
    base: X86Reg,
    disp: i32,
) {
    let r = dst >= 8;
    let b = base.needs_rex_ext();
    emit_simd_prefix_and_rex(e, prefix, (r || b).then(|| rex(false, r, false, b)));
    emit_simd_opcode_map(e, map);
    e.emit_u8(opcode);
    let reg3 = dst & 7;
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

fn emit_simd_rm_imm8(
    e: &mut TextEmitter,
    prefix: u8,
    map: OpMap,
    opcode: u8,
    dst: u8,
    base: X86Reg,
    disp: i32,
    imm8: u8,
) {
    emit_simd_rm(e, prefix, map, opcode, dst, base, disp);
    e.emit_u8(imm8);
}

fn emit_simd_rm_imm8_rexw(
    e: &mut TextEmitter,
    prefix: u8,
    map: OpMap,
    opcode: u8,
    dst: u8,
    base: X86Reg,
    disp: i32,
    imm8: u8,
    w: bool,
) {
    let r = dst >= 8;
    let b = base.needs_rex_ext();
    emit_simd_prefix_and_rex(e, prefix, (w || r || b).then(|| rex(w, r, false, b)));
    emit_simd_opcode_map(e, map);
    e.emit_u8(opcode);
    let reg3 = dst & 7;
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
    e.emit_u8(imm8);
}

fn emit_simd_mr(
    e: &mut TextEmitter,
    prefix: u8,
    map: OpMap,
    opcode: u8,
    base: X86Reg,
    disp: i32,
    src: u8,
) {
    let r = src >= 8;
    let b = base.needs_rex_ext();
    emit_simd_prefix_and_rex(e, prefix, (r || b).then(|| rex(false, r, false, b)));
    emit_simd_opcode_map(e, map);
    e.emit_u8(opcode);
    let reg3 = src & 7;
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

pub(super) fn movdqa_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0x6F, dst, src);
}

fn movdqu_load(e: &mut TextEmitter, dst: u8, base: X86Reg, disp: i32) {
    emit_simd_rm(e, 0xF3, OpMap::Map0F, 0x6F, dst, base, disp);
}

fn movdqu_store(e: &mut TextEmitter, base: X86Reg, disp: i32, src: u8) {
    emit_simd_mr(e, 0xF3, OpMap::Map0F, 0x7F, base, disp, src);
}

fn pshufb_rm(e: &mut TextEmitter, dst: u8, base: X86Reg, disp: i32) {
    emit_simd_rm(e, 0x66, OpMap::Map0F38, 0x00, dst, base, disp);
}

fn psrldq_rr_imm8(e: &mut TextEmitter, dst: u8, imm8: u8) {
    emit_simd_digit_imm8(e, 0x66, OpMap::Map0F, 0x73, 3, dst, imm8);
}

fn ptest_rr(e: &mut TextEmitter, lhs: u8, rhs: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F38, 0x17, lhs, rhs);
}

fn pmovsxbw_rm(e: &mut TextEmitter, dst: u8, base: X86Reg, disp: i32) {
    emit_simd_rm(e, 0x66, OpMap::Map0F38, 0x20, dst, base, disp);
}

fn pmovsxwd_rm(e: &mut TextEmitter, dst: u8, base: X86Reg, disp: i32) {
    emit_simd_rm(e, 0x66, OpMap::Map0F38, 0x23, dst, base, disp);
}

fn pmovsxdq_rm(e: &mut TextEmitter, dst: u8, base: X86Reg, disp: i32) {
    emit_simd_rm(e, 0x66, OpMap::Map0F38, 0x25, dst, base, disp);
}

fn pmovzxbw_rm(e: &mut TextEmitter, dst: u8, base: X86Reg, disp: i32) {
    emit_simd_rm(e, 0x66, OpMap::Map0F38, 0x30, dst, base, disp);
}

fn pmovzxwd_rm(e: &mut TextEmitter, dst: u8, base: X86Reg, disp: i32) {
    emit_simd_rm(e, 0x66, OpMap::Map0F38, 0x33, dst, base, disp);
}

fn pmovzxdq_rm(e: &mut TextEmitter, dst: u8, base: X86Reg, disp: i32) {
    emit_simd_rm(e, 0x66, OpMap::Map0F38, 0x35, dst, base, disp);
}

fn pinsrb_rm_imm8(e: &mut TextEmitter, dst: u8, base: X86Reg, disp: i32, lane: u8) {
    emit_simd_rm_imm8(e, 0x66, OpMap::Map0F3A, 0x20, dst, base, disp, lane);
}

fn pinsrw_rm_imm8(e: &mut TextEmitter, dst: u8, base: X86Reg, disp: i32, lane: u8) {
    emit_simd_rm_imm8(e, 0x66, OpMap::Map0F, 0xC4, dst, base, disp, lane);
}

fn pinsrd_rm_imm8(e: &mut TextEmitter, dst: u8, base: X86Reg, disp: i32, lane: u8) {
    emit_simd_rm_imm8(e, 0x66, OpMap::Map0F3A, 0x22, dst, base, disp, lane);
}

fn pinsrq_rm_imm8(e: &mut TextEmitter, dst: u8, base: X86Reg, disp: i32, lane: u8) {
    emit_simd_rm_imm8_rexw(e, 0x66, OpMap::Map0F3A, 0x22, dst, base, disp, lane, true);
}

fn pmovmskb_r32_xmm(e: &mut TextEmitter, dst: X86Reg, src: u8) {
    emit_simd_gp_xmm(e, 0x66, OpMap::Map0F, 0xD7, dst, src);
}

fn pcmpeqb_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0x74, dst, src);
}

fn pcmpeqw_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0x75, dst, src);
}

fn pcmpeqd_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0x76, dst, src);
}

fn pcmpeqq_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F38, 0x29, dst, src);
}

fn pand_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0xDB, dst, src);
}

fn pandn_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0xDF, dst, src);
}

fn por_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0xEB, dst, src);
}

fn pxor_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0xEF, dst, src);
}

fn pshufd_rr_imm8(e: &mut TextEmitter, dst: u8, src: u8, imm8: u8) {
    emit_simd_rr_imm8(e, 0x66, OpMap::Map0F, 0x70, dst, src, imm8);
}

fn punpcklqdq_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0x6C, dst, src);
}

fn paddb_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0xFC, dst, src);
}

fn psubb_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0xF8, dst, src);
}

fn paddsb_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0xEC, dst, src);
}

fn paddusb_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0xDC, dst, src);
}

fn psubsb_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0xE8, dst, src);
}

fn psubusb_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0xD8, dst, src);
}

fn paddw_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0xFD, dst, src);
}

fn packsswb_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0x63, dst, src);
}

fn packuswb_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0x67, dst, src);
}

fn psubw_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0xF9, dst, src);
}

fn pmullw_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0xD5, dst, src);
}

fn paddsw_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0xED, dst, src);
}

fn packssdw_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0x6B, dst, src);
}

fn packusdw_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F38, 0x2B, dst, src);
}

fn paddusw_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0xDD, dst, src);
}

fn psubsw_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0xE9, dst, src);
}

fn psubusw_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0xD9, dst, src);
}

fn pavgb_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0xE0, dst, src);
}

fn pavgw_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0xE3, dst, src);
}

fn psubd_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0xFA, dst, src);
}

fn pmulld_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F38, 0x40, dst, src);
}

fn psubq_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0xFB, dst, src);
}

fn pabsb_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F38, 0x1C, dst, src);
}

fn pabsw_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F38, 0x1D, dst, src);
}

fn pabsd_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F38, 0x1E, dst, src);
}

fn pminsb_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F38, 0x38, dst, src);
}

fn pminub_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0xDA, dst, src);
}

fn pmaxsb_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F38, 0x3C, dst, src);
}

fn pmaxub_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0xDE, dst, src);
}

fn pminsw_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0xEA, dst, src);
}

fn pminuw_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F38, 0x3A, dst, src);
}

fn pmaxsw_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0xEE, dst, src);
}

fn pmaxuw_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F38, 0x3E, dst, src);
}

fn pminsd_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F38, 0x39, dst, src);
}

fn pminud_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F38, 0x3B, dst, src);
}

fn pmaxsd_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F38, 0x3D, dst, src);
}

fn pmaxud_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F38, 0x3F, dst, src);
}

fn addpd_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0x58, dst, src);
}

fn subpd_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0x5C, dst, src);
}

fn mulpd_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0x59, dst, src);
}

fn divpd_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0x5E, dst, src);
}

fn sqrtpd_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0x51, dst, src);
}

fn roundpd_rr_imm8(e: &mut TextEmitter, dst: u8, src: u8, imm8: u8) {
    emit_simd_rr_imm8(e, 0x66, OpMap::Map0F3A, 0x09, dst, src, imm8);
}

fn addps_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0, OpMap::Map0F, 0x58, dst, src);
}

fn subps_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0, OpMap::Map0F, 0x5C, dst, src);
}

fn mulps_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0, OpMap::Map0F, 0x59, dst, src);
}

fn divps_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0, OpMap::Map0F, 0x5E, dst, src);
}

fn sqrtps_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0, OpMap::Map0F, 0x51, dst, src);
}

fn roundps_rr_imm8(e: &mut TextEmitter, dst: u8, src: u8, imm8: u8) {
    emit_simd_rr_imm8(e, 0x66, OpMap::Map0F3A, 0x08, dst, src, imm8);
}

fn cvtdq2ps_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0, OpMap::Map0F, 0x5B, dst, src);
}

fn cvtdq2pd_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0xF3, OpMap::Map0F, 0xE6, dst, src);
}

fn cvtpd2ps_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0x66, OpMap::Map0F, 0x5A, dst, src);
}

fn cvtps2pd_rr(e: &mut TextEmitter, dst: u8, src: u8) {
    emit_simd_rr(e, 0, OpMap::Map0F, 0x5A, dst, src);
}

fn cvttss2si_r32_xmm(e: &mut TextEmitter, dst: X86Reg, src: u8) {
    let r = dst.needs_rex_ext();
    let b = src >= 8;
    emit_simd_prefix_and_rex(e, 0xF3, (r || b).then(|| rex(false, r, false, b)));
    emit_simd_opcode_map(e, OpMap::Map0F);
    e.emit_u8(0x2C);
    e.emit_u8(modrm(0b11, dst.idx3(), src & 7));
}

fn cvttss2si_r64_xmm(e: &mut TextEmitter, dst: X86Reg, src: u8) {
    let r = dst.needs_rex_ext();
    let b = src >= 8;
    emit_simd_prefix_and_rex(e, 0xF3, Some(rex(true, r, false, b)));
    emit_simd_opcode_map(e, OpMap::Map0F);
    e.emit_u8(0x2C);
    e.emit_u8(modrm(0b11, dst.idx3(), src & 7));
}

fn cvttsd2si_r64_xmm(e: &mut TextEmitter, dst: X86Reg, src: u8) {
    let r = dst.needs_rex_ext();
    let b = src >= 8;
    emit_simd_prefix_and_rex(e, 0xF2, Some(rex(true, r, false, b)));
    emit_simd_opcode_map(e, OpMap::Map0F);
    e.emit_u8(0x2C);
    e.emit_u8(modrm(0b11, dst.idx3(), src & 7));
}

fn test_rr_64(e: &mut TextEmitter, lhs: X86Reg, rhs: X86Reg) {
    enc::test_rr_64(e, lhs, rhs);
}

fn expect_v128_reg(
    backend: &X86_64Backend<'_>,
    value: MachineValue,
    label: &'static str,
) -> Result<u8, WasmError> {
    match value {
        MachineValue::Reg(reg) => Ok(backend.map_fp_reg(reg)? as u8),
        MachineValue::Imm64(_) | MachineValue::ReservedReg(_) => Err(WasmError::internal(label)),
    }
}

impl<'a> X86_64Backend<'a> {
    pub(super) fn lower_simd_inst_dispatch(&mut self, inst: &MachineInst) -> Result<(), WasmError> {
        match &inst.kind {
            MachineInstKind::Move {
                ty: MachineStorageType::V128,
                dst,
                src,
                ..
            } => self.lower_move_v128(*dst, *src),
            MachineInstKind::V128Const { dst, bytes } => self.lower_v128_const_native(*dst, *bytes),
            MachineInstKind::V128FromRaw { dst, raw, .. } => self.lower_v128_from_raw(*dst, *raw),
            MachineInstKind::V128ToRaw { dst, src } => self.lower_v128_to_raw(*dst, *src),
            MachineInstKind::SimdUnary {
                opcode,
                dst_ty,
                dst,
                src,
            } => self.lower_simd_unary_native(*opcode, *dst_ty, *dst, *src),
            MachineInstKind::SimdBinary {
                opcode,
                dst,
                lhs,
                rhs,
                ..
            } => self.lower_simd_binary_native(*opcode, *dst, *lhs, *rhs),
            MachineInstKind::SimdTernary {
                opcode,
                dst,
                a,
                b,
                c,
                ..
            } => self.lower_simd_ternary_native(*opcode, *dst, *a, *b, *c),
            MachineInstKind::SimdShift {
                opcode,
                dst,
                vector,
                shift,
                ..
            } => self.lower_simd_shift_native(*opcode, *dst, *vector, *shift),
            MachineInstKind::SimdExtractLane {
                opcode,
                lane,
                dst_ty,
                dst,
                src,
            } => self.lower_simd_extract_lane_native(*opcode, *lane, *dst_ty, *dst, *src),
            MachineInstKind::SimdReplaceLane {
                opcode,
                lane,
                dst,
                vector,
                scalar,
            } => self.lower_simd_replace_lane_native(*opcode, *lane, *dst, *vector, *scalar),
            MachineInstKind::SimdShuffle {
                dst,
                lhs,
                rhs,
                lanes,
            } => self.lower_simd_shuffle_native(*dst, *lhs, *rhs, *lanes),
            MachineInstKind::SimdLoad { opcode, dst, addr } => {
                self.lower_simd_load_native(*opcode, *dst, *addr)
            }
            MachineInstKind::SimdStore { opcode, addr, src } => {
                self.lower_simd_store_native(*opcode, *addr, *src)
            }
            MachineInstKind::SimdLoadLane {
                opcode,
                lane,
                dst,
                addr,
                vector,
            } => self.lower_simd_load_lane_native(*opcode, *lane, *dst, *addr, *vector),
            MachineInstKind::SimdStoreLane {
                opcode,
                lane,
                addr,
                vector,
            } => self.lower_simd_store_lane_native(*opcode, *lane, *addr, *vector),
            _ => Err(WasmError::internal(
                "x86_64 SIMD dispatch received a non-SIMD instruction",
            )),
        }
    }

    pub(super) fn lower_select_v128(
        &mut self,
        dst: MachineReg,
        on_true: MachineValue,
        on_false: MachineValue,
        cond: MachineValue,
    ) -> Result<(), WasmError> {
        let dst_fp = self.map_fp_reg(dst)? as u8;
        match cond {
            MachineValue::Imm64(value) => {
                let selected = if value != 0 { on_true } else { on_false };
                self.lower_move_v128(dst, selected)
            }
            MachineValue::Reg(reg) => {
                let cond_gp = self.map_gp_reg(reg)?;
                let false_label = self.core.new_label();
                let done = self.core.new_label();
                test_rr_64(&mut self.core.text, cond_gp, cond_gp);
                self.emit_jcc(Cc::E, false_label);
                let true_fp = expect_v128_reg(
                    self,
                    on_true,
                    "x86_64 v128 select needs a vector true operand",
                )?;
                if dst_fp != true_fp {
                    movdqa_rr(&mut self.core.text, dst_fp, true_fp);
                }
                self.emit_jmp(done);
                self.core.bind_label(false_label);
                let false_fp = expect_v128_reg(
                    self,
                    on_false,
                    "x86_64 v128 select needs a vector false operand",
                )?;
                if dst_fp != false_fp {
                    movdqa_rr(&mut self.core.text, dst_fp, false_fp);
                }
                self.core.bind_label(done);
                Ok(())
            }
            MachineValue::ReservedReg(_) => Err(WasmError::internal(
                "x86_64 v128 select cannot consume a reserved cache condition register",
            )),
        }
    }

    fn lower_move_v128(&mut self, dst: MachineReg, src: MachineValue) -> Result<(), WasmError> {
        let dst_fp = self.map_fp_reg(dst)? as u8;
        let src_fp = expect_v128_reg(self, src, "x86_64 v128 move needs a vector source")?;
        if dst_fp != src_fp {
            movdqa_rr(&mut self.core.text, dst_fp, src_fp);
        }
        Ok(())
    }

    fn lower_simd_unary_native(
        &mut self,
        opcode: u32,
        dst_ty: MachineStorageType,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        match OpcodeFD::try_from(opcode)? {
            OpcodeFD::V128_NOT => {
                let src_fp =
                    expect_v128_reg(self, src, "x86_64 SIMD unary source must be in an FP reg")?;
                let dst_fp = self.map_fp_reg(dst)? as u8;
                if dst_fp != src_fp {
                    movdqa_rr(&mut self.core.text, dst_fp, src_fp);
                }
                let tmp = self.fp_scratch.scoped_alloc().detach();
                pxor_rr(&mut self.core.text, *tmp as u8, *tmp as u8);
                pcmpeqd_rr(&mut self.core.text, *tmp as u8, *tmp as u8);
                pxor_rr(&mut self.core.text, dst_fp, *tmp as u8);
                Ok(())
            }
            OpcodeFD::V128_ANY_TRUE => {
                let src_fp =
                    expect_v128_reg(self, src, "x86_64 SIMD unary source must be in an FP reg")?;
                debug_assert_eq!(dst_ty, MachineStorageType::GpWord);
                let dst_gp = self.map_gp_reg(dst)?;
                ptest_rr(&mut self.core.text, src_fp, src_fp);
                enc::setcc(&mut self.core.text, Cc::NE, dst_gp);
                enc::movzx_r32_r8(&mut self.core.text, dst_gp, dst_gp);
                Ok(())
            }
            OpcodeFD::I8X16_ALL_TRUE => {
                let src_fp =
                    expect_v128_reg(self, src, "x86_64 SIMD unary source must be in an FP reg")?;
                self.lower_simd_all_true(src_fp, dst, SimdAllTrueWidth::B8)
            }
            OpcodeFD::I16X8_ALL_TRUE => {
                let src_fp =
                    expect_v128_reg(self, src, "x86_64 SIMD unary source must be in an FP reg")?;
                self.lower_simd_all_true(src_fp, dst, SimdAllTrueWidth::H16)
            }
            OpcodeFD::I32X4_ALL_TRUE => {
                let src_fp =
                    expect_v128_reg(self, src, "x86_64 SIMD unary source must be in an FP reg")?;
                self.lower_simd_all_true(src_fp, dst, SimdAllTrueWidth::S32)
            }
            OpcodeFD::I64X2_ALL_TRUE => {
                let src_fp =
                    expect_v128_reg(self, src, "x86_64 SIMD unary source must be in an FP reg")?;
                self.lower_simd_all_true(src_fp, dst, SimdAllTrueWidth::D64)
            }
            OpcodeFD::I8X16_BITMASK => {
                let src_fp =
                    expect_v128_reg(self, src, "x86_64 SIMD unary source must be in an FP reg")?;
                self.lower_simd_bitmask(src_fp, dst, SimdBitmaskWidth::B8)
            }
            OpcodeFD::I16X8_BITMASK => {
                let src_fp =
                    expect_v128_reg(self, src, "x86_64 SIMD unary source must be in an FP reg")?;
                self.lower_simd_bitmask(src_fp, dst, SimdBitmaskWidth::H16)
            }
            OpcodeFD::I32X4_BITMASK => {
                let src_fp =
                    expect_v128_reg(self, src, "x86_64 SIMD unary source must be in an FP reg")?;
                self.lower_simd_bitmask(src_fp, dst, SimdBitmaskWidth::S32)
            }
            OpcodeFD::I64X2_BITMASK => {
                let src_fp =
                    expect_v128_reg(self, src, "x86_64 SIMD unary source must be in an FP reg")?;
                self.lower_simd_bitmask(src_fp, dst, SimdBitmaskWidth::D64)
            }
            OpcodeFD::I8X16_SPLAT => self.lower_simd_splat_integer(dst, src, IntegerSplatWidth::B8),
            OpcodeFD::I16X8_SPLAT => {
                self.lower_simd_splat_integer(dst, src, IntegerSplatWidth::H16)
            }
            OpcodeFD::I32X4_SPLAT => {
                self.lower_simd_splat_integer(dst, src, IntegerSplatWidth::S32)
            }
            OpcodeFD::I64X2_SPLAT => {
                self.lower_simd_splat_integer(dst, src, IntegerSplatWidth::D64)
            }
            OpcodeFD::F32X4_SPLAT => self.lower_simd_splat_float(dst, src, FloatSplatWidth::F32),
            OpcodeFD::F64X2_SPLAT => self.lower_simd_splat_float(dst, src, FloatSplatWidth::F64),
            OpcodeFD::I8X16_POPCNT => {
                let src_fp =
                    expect_v128_reg(self, src, "x86_64 i8x16.popcnt source must be in an FP reg")?;
                let dst_fp = self.map_fp_reg(dst)? as u8;
                self.lower_i8x16_popcnt_scalar(dst_fp, src_fp)
            }
            OpcodeFD::I16X8_EXTADD_PAIRWISE_I8X16_S => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "x86_64 i16x8.extadd_pairwise_i8x16_s source must be in an FP reg",
                )?;
                let dst_fp = self.map_fp_reg(dst)? as u8;
                self.lower_simd_pairwise_extadd_scalar(
                    dst_fp,
                    src_fp,
                    IntCompareLaneWidth::B8,
                    true,
                )
            }
            OpcodeFD::I16X8_EXTADD_PAIRWISE_I8X16_U => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "x86_64 i16x8.extadd_pairwise_i8x16_u source must be in an FP reg",
                )?;
                let dst_fp = self.map_fp_reg(dst)? as u8;
                self.lower_simd_pairwise_extadd_scalar(
                    dst_fp,
                    src_fp,
                    IntCompareLaneWidth::B8,
                    false,
                )
            }
            OpcodeFD::I32X4_EXTADD_PAIRWISE_I16X8_S => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "x86_64 i32x4.extadd_pairwise_i16x8_s source must be in an FP reg",
                )?;
                let dst_fp = self.map_fp_reg(dst)? as u8;
                self.lower_simd_pairwise_extadd_scalar(
                    dst_fp,
                    src_fp,
                    IntCompareLaneWidth::H16,
                    true,
                )
            }
            OpcodeFD::I32X4_EXTADD_PAIRWISE_I16X8_U => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "x86_64 i32x4.extadd_pairwise_i16x8_u source must be in an FP reg",
                )?;
                let dst_fp = self.map_fp_reg(dst)? as u8;
                self.lower_simd_pairwise_extadd_scalar(
                    dst_fp,
                    src_fp,
                    IntCompareLaneWidth::H16,
                    false,
                )
            }
            OpcodeFD::I16X8_EXTEND_LOW_I8X16_S => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "x86_64 i16x8.extend_low_i8x16_s source must be in an FP reg",
                )?;
                let dst_fp = self.map_fp_reg(dst)? as u8;
                self.lower_simd_extend_scalar(dst_fp, src_fp, IntCompareLaneWidth::B8, true, false)
            }
            OpcodeFD::I16X8_EXTEND_HIGH_I8X16_S => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "x86_64 i16x8.extend_high_i8x16_s source must be in an FP reg",
                )?;
                let dst_fp = self.map_fp_reg(dst)? as u8;
                self.lower_simd_extend_scalar(dst_fp, src_fp, IntCompareLaneWidth::B8, true, true)
            }
            OpcodeFD::I16X8_EXTEND_LOW_I8X16_U => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "x86_64 i16x8.extend_low_i8x16_u source must be in an FP reg",
                )?;
                let dst_fp = self.map_fp_reg(dst)? as u8;
                self.lower_simd_extend_scalar(dst_fp, src_fp, IntCompareLaneWidth::B8, false, false)
            }
            OpcodeFD::I16X8_EXTEND_HIGH_I8X16_U => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "x86_64 i16x8.extend_high_i8x16_u source must be in an FP reg",
                )?;
                let dst_fp = self.map_fp_reg(dst)? as u8;
                self.lower_simd_extend_scalar(dst_fp, src_fp, IntCompareLaneWidth::B8, false, true)
            }
            OpcodeFD::I32X4_EXTEND_LOW_I16X8_S => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "x86_64 i32x4.extend_low_i16x8_s source must be in an FP reg",
                )?;
                let dst_fp = self.map_fp_reg(dst)? as u8;
                self.lower_simd_extend_scalar(dst_fp, src_fp, IntCompareLaneWidth::H16, true, false)
            }
            OpcodeFD::I32X4_EXTEND_HIGH_I16X8_S => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "x86_64 i32x4.extend_high_i16x8_s source must be in an FP reg",
                )?;
                let dst_fp = self.map_fp_reg(dst)? as u8;
                self.lower_simd_extend_scalar(dst_fp, src_fp, IntCompareLaneWidth::H16, true, true)
            }
            OpcodeFD::I32X4_EXTEND_LOW_I16X8_U => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "x86_64 i32x4.extend_low_i16x8_u source must be in an FP reg",
                )?;
                let dst_fp = self.map_fp_reg(dst)? as u8;
                self.lower_simd_extend_scalar(
                    dst_fp,
                    src_fp,
                    IntCompareLaneWidth::H16,
                    false,
                    false,
                )
            }
            OpcodeFD::I32X4_EXTEND_HIGH_I16X8_U => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "x86_64 i32x4.extend_high_i16x8_u source must be in an FP reg",
                )?;
                let dst_fp = self.map_fp_reg(dst)? as u8;
                self.lower_simd_extend_scalar(dst_fp, src_fp, IntCompareLaneWidth::H16, false, true)
            }
            OpcodeFD::I64X2_EXTEND_LOW_I32X4_S => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "x86_64 i64x2.extend_low_i32x4_s source must be in an FP reg",
                )?;
                let dst_fp = self.map_fp_reg(dst)? as u8;
                self.lower_simd_extend_scalar(dst_fp, src_fp, IntCompareLaneWidth::S32, true, false)
            }
            OpcodeFD::I64X2_EXTEND_HIGH_I32X4_S => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "x86_64 i64x2.extend_high_i32x4_s source must be in an FP reg",
                )?;
                let dst_fp = self.map_fp_reg(dst)? as u8;
                self.lower_simd_extend_scalar(dst_fp, src_fp, IntCompareLaneWidth::S32, true, true)
            }
            OpcodeFD::I64X2_EXTEND_LOW_I32X4_U => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "x86_64 i64x2.extend_low_i32x4_u source must be in an FP reg",
                )?;
                let dst_fp = self.map_fp_reg(dst)? as u8;
                self.lower_simd_extend_scalar(
                    dst_fp,
                    src_fp,
                    IntCompareLaneWidth::S32,
                    false,
                    false,
                )
            }
            OpcodeFD::I64X2_EXTEND_HIGH_I32X4_U => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "x86_64 i64x2.extend_high_i32x4_u source must be in an FP reg",
                )?;
                let dst_fp = self.map_fp_reg(dst)? as u8;
                self.lower_simd_extend_scalar(dst_fp, src_fp, IntCompareLaneWidth::S32, false, true)
            }
            OpcodeFD::I8X16_ABS => {
                let src_fp =
                    expect_v128_reg(self, src, "x86_64 i8x16.abs source must be in an FP reg")?;
                self.lower_simd_int_unary_vector(dst, src_fp, SimdIntUnaryVectorOp::AbsB8)
            }
            OpcodeFD::I8X16_NEG => {
                let src_fp =
                    expect_v128_reg(self, src, "x86_64 i8x16.neg source must be in an FP reg")?;
                self.lower_simd_int_unary_vector(dst, src_fp, SimdIntUnaryVectorOp::NegB8)
            }
            OpcodeFD::I16X8_ABS => {
                let src_fp =
                    expect_v128_reg(self, src, "x86_64 i16x8.abs source must be in an FP reg")?;
                self.lower_simd_int_unary_vector(dst, src_fp, SimdIntUnaryVectorOp::AbsH16)
            }
            OpcodeFD::I16X8_NEG => {
                let src_fp =
                    expect_v128_reg(self, src, "x86_64 i16x8.neg source must be in an FP reg")?;
                self.lower_simd_int_unary_vector(dst, src_fp, SimdIntUnaryVectorOp::NegH16)
            }
            OpcodeFD::I32X4_ABS => {
                let src_fp =
                    expect_v128_reg(self, src, "x86_64 i32x4.abs source must be in an FP reg")?;
                self.lower_simd_int_unary_vector(dst, src_fp, SimdIntUnaryVectorOp::AbsS32)
            }
            OpcodeFD::I32X4_NEG => {
                let src_fp =
                    expect_v128_reg(self, src, "x86_64 i32x4.neg source must be in an FP reg")?;
                self.lower_simd_int_unary_vector(dst, src_fp, SimdIntUnaryVectorOp::NegS32)
            }
            OpcodeFD::I64X2_ABS => {
                let src_fp =
                    expect_v128_reg(self, src, "x86_64 i64x2.abs source must be in an FP reg")?;
                let dst_fp = self.map_fp_reg(dst)? as u8;
                self.lower_i64x2_abs_scalar(dst_fp, src_fp)
            }
            OpcodeFD::I64X2_NEG => {
                let src_fp =
                    expect_v128_reg(self, src, "x86_64 i64x2.neg source must be in an FP reg")?;
                self.lower_simd_int_unary_vector(dst, src_fp, SimdIntUnaryVectorOp::NegD64)
            }
            OpcodeFD::F32X4_ABS => {
                let src_fp =
                    expect_v128_reg(self, src, "x86_64 f32x4.abs source must be in an FP reg")?;
                self.lower_simd_float_unary_masked(
                    dst,
                    src_fp,
                    [
                        0xff, 0xff, 0xff, 0x7f, 0xff, 0xff, 0xff, 0x7f, 0xff, 0xff, 0xff, 0x7f,
                        0xff, 0xff, 0xff, 0x7f,
                    ],
                    true,
                )
            }
            OpcodeFD::F32X4_NEG => {
                let src_fp =
                    expect_v128_reg(self, src, "x86_64 f32x4.neg source must be in an FP reg")?;
                self.lower_simd_float_unary_masked(
                    dst,
                    src_fp,
                    [
                        0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x80,
                        0x00, 0x00, 0x00, 0x80,
                    ],
                    false,
                )
            }
            OpcodeFD::F32X4_SQRT => {
                let src_fp =
                    expect_v128_reg(self, src, "x86_64 f32x4.sqrt source must be in an FP reg")?;
                self.lower_simd_float_unary_vector(dst, src_fp, SimdFloatUnaryVectorOp::SqrtF32)
            }
            OpcodeFD::F32X4_CEIL => {
                let src_fp =
                    expect_v128_reg(self, src, "x86_64 f32x4.ceil source must be in an FP reg")?;
                self.lower_simd_float_unary_vector(
                    dst,
                    src_fp,
                    SimdFloatUnaryVectorOp::RoundF32(enc::ROUND_CEIL),
                )
            }
            OpcodeFD::F32X4_FLOOR => {
                let src_fp =
                    expect_v128_reg(self, src, "x86_64 f32x4.floor source must be in an FP reg")?;
                self.lower_simd_float_unary_vector(
                    dst,
                    src_fp,
                    SimdFloatUnaryVectorOp::RoundF32(enc::ROUND_FLOOR),
                )
            }
            OpcodeFD::F32X4_TRUNC => {
                let src_fp =
                    expect_v128_reg(self, src, "x86_64 f32x4.trunc source must be in an FP reg")?;
                self.lower_simd_float_unary_vector(
                    dst,
                    src_fp,
                    SimdFloatUnaryVectorOp::RoundF32(enc::ROUND_TRUNC),
                )
            }
            OpcodeFD::F32X4_NEAREST => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "x86_64 f32x4.nearest source must be in an FP reg",
                )?;
                self.lower_simd_float_unary_vector(
                    dst,
                    src_fp,
                    SimdFloatUnaryVectorOp::RoundF32(enc::ROUND_NEAREST),
                )
            }
            OpcodeFD::F64X2_ABS => {
                let src_fp =
                    expect_v128_reg(self, src, "x86_64 f64x2.abs source must be in an FP reg")?;
                self.lower_simd_float_unary_masked(
                    dst,
                    src_fp,
                    [
                        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xff, 0xff, 0xff, 0xff,
                        0xff, 0xff, 0xff, 0x7f,
                    ],
                    true,
                )
            }
            OpcodeFD::F64X2_NEG => {
                let src_fp =
                    expect_v128_reg(self, src, "x86_64 f64x2.neg source must be in an FP reg")?;
                self.lower_simd_float_unary_masked(
                    dst,
                    src_fp,
                    [
                        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00,
                        0x00, 0x00, 0x00, 0x80,
                    ],
                    false,
                )
            }
            OpcodeFD::F64X2_SQRT => {
                let src_fp =
                    expect_v128_reg(self, src, "x86_64 f64x2.sqrt source must be in an FP reg")?;
                self.lower_simd_float_unary_vector(dst, src_fp, SimdFloatUnaryVectorOp::SqrtF64)
            }
            OpcodeFD::F64X2_CEIL => {
                let src_fp =
                    expect_v128_reg(self, src, "x86_64 f64x2.ceil source must be in an FP reg")?;
                self.lower_simd_float_unary_vector(
                    dst,
                    src_fp,
                    SimdFloatUnaryVectorOp::RoundF64(enc::ROUND_CEIL),
                )
            }
            OpcodeFD::F64X2_FLOOR => {
                let src_fp =
                    expect_v128_reg(self, src, "x86_64 f64x2.floor source must be in an FP reg")?;
                self.lower_simd_float_unary_vector(
                    dst,
                    src_fp,
                    SimdFloatUnaryVectorOp::RoundF64(enc::ROUND_FLOOR),
                )
            }
            OpcodeFD::F64X2_TRUNC => {
                let src_fp =
                    expect_v128_reg(self, src, "x86_64 f64x2.trunc source must be in an FP reg")?;
                self.lower_simd_float_unary_vector(
                    dst,
                    src_fp,
                    SimdFloatUnaryVectorOp::RoundF64(enc::ROUND_TRUNC),
                )
            }
            OpcodeFD::F64X2_NEAREST => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "x86_64 f64x2.nearest source must be in an FP reg",
                )?;
                self.lower_simd_float_unary_vector(
                    dst,
                    src_fp,
                    SimdFloatUnaryVectorOp::RoundF64(enc::ROUND_NEAREST),
                )
            }
            OpcodeFD::F32X4_CONVERT_I32X4_S => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "x86_64 f32x4.convert_i32x4_s source must be in an FP reg",
                )?;
                self.lower_simd_float_convert_vector(dst, src_fp, SimdFloatConvertOp::I32x4ToF32x4S)
            }
            OpcodeFD::F32X4_CONVERT_I32X4_U => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "x86_64 f32x4.convert_i32x4_u source must be in an FP reg",
                )?;
                let dst_fp = self.map_fp_reg(dst)? as u8;
                self.lower_f32x4_convert_i32x4_u_scalar(dst_fp, src_fp)
            }
            OpcodeFD::F64X2_CONVERT_LOW_I32X4_S => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "x86_64 f64x2.convert_low_i32x4_s source must be in an FP reg",
                )?;
                self.lower_simd_float_convert_vector(
                    dst,
                    src_fp,
                    SimdFloatConvertOp::I32x4LowToF64x2S,
                )
            }
            OpcodeFD::F64X2_CONVERT_LOW_I32X4_U => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "x86_64 f64x2.convert_low_i32x4_u source must be in an FP reg",
                )?;
                let dst_fp = self.map_fp_reg(dst)? as u8;
                self.lower_f64x2_convert_low_i32x4_u_scalar(dst_fp, src_fp)
            }
            OpcodeFD::F32X4_DEMOTE_F64X2_ZERO => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "x86_64 f32x4.demote_f64x2_zero source must be in an FP reg",
                )?;
                self.lower_simd_float_convert_vector(
                    dst,
                    src_fp,
                    SimdFloatConvertOp::F64x2ToF32x4Zero,
                )
            }
            OpcodeFD::F64X2_PROMOTE_LOW_F32X4 => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "x86_64 f64x2.promote_low_f32x4 source must be in an FP reg",
                )?;
                self.lower_simd_float_convert_vector(
                    dst,
                    src_fp,
                    SimdFloatConvertOp::F32x4LowToF64x2,
                )
            }
            OpcodeFD::I32X4_TRUNC_SAT_F32X4_S => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "x86_64 i32x4.trunc_sat_f32x4_s source must be in an FP reg",
                )?;
                let dst_fp = self.map_fp_reg(dst)? as u8;
                self.lower_i32x4_trunc_sat_f32x4_s_scalar(dst_fp, src_fp)
            }
            OpcodeFD::I32X4_TRUNC_SAT_F32X4_U => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "x86_64 i32x4.trunc_sat_f32x4_u source must be in an FP reg",
                )?;
                let dst_fp = self.map_fp_reg(dst)? as u8;
                self.lower_i32x4_trunc_sat_f32x4_u_scalar(dst_fp, src_fp)
            }
            OpcodeFD::I32X4_TRUNC_SAT_F64X2_S_ZERO => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "x86_64 i32x4.trunc_sat_f64x2_s_zero source must be in an FP reg",
                )?;
                let dst_fp = self.map_fp_reg(dst)? as u8;
                self.lower_i32x4_trunc_sat_f64x2_scalar(dst_fp, src_fp, false)
            }
            OpcodeFD::I32X4_TRUNC_SAT_F64X2_U_ZERO => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "x86_64 i32x4.trunc_sat_f64x2_u_zero source must be in an FP reg",
                )?;
                let dst_fp = self.map_fp_reg(dst)? as u8;
                self.lower_i32x4_trunc_sat_f64x2_scalar(dst_fp, src_fp, true)
            }
            other => Err(unsupported_simd_opcode("unary", other as u32)),
        }
    }

    fn lower_simd_binary_native(
        &mut self,
        opcode: u32,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        let dst_fp = self.map_fp_reg(dst)? as u8;
        let lhs_fp = expect_v128_reg(self, lhs, "x86_64 SIMD binary lhs must be in an FP reg")?;
        let rhs_fp = expect_v128_reg(self, rhs, "x86_64 SIMD binary rhs must be in an FP reg")?;

        let rhs_src = if dst_fp == rhs_fp as u8 && dst_fp != lhs_fp as u8 {
            let tmp = self.fp_scratch.scoped_alloc().detach();
            movdqa_rr(&mut self.core.text, *tmp as u8, rhs_fp as u8);
            *tmp as u8
        } else {
            rhs_fp as u8
        };

        match OpcodeFD::try_from(opcode)? {
            OpcodeFD::I8X16_NARROW_I16X8_S => self.lower_simd_narrow_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::H16,
                true,
            ),
            OpcodeFD::I8X16_NARROW_I16X8_U => self.lower_simd_narrow_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::H16,
                false,
            ),
            OpcodeFD::I16X8_NARROW_I32X4_S => self.lower_simd_narrow_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::S32,
                true,
            ),
            OpcodeFD::I16X8_NARROW_I32X4_U => self.lower_simd_narrow_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::S32,
                false,
            ),
            OpcodeFD::V128_AND => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                pand_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::V128_ANDNOT => {
                if dst_fp == lhs_fp as u8 && lhs_fp != rhs_fp as u8 {
                    let tmp = self.fp_scratch.scoped_alloc().detach();
                    movdqa_rr(&mut self.core.text, *tmp as u8, rhs_src);
                    pandn_rr(&mut self.core.text, *tmp as u8, lhs_fp);
                    movdqa_rr(&mut self.core.text, dst_fp, *tmp as u8);
                } else {
                    if dst_fp != rhs_src {
                        movdqa_rr(&mut self.core.text, dst_fp, rhs_src);
                    }
                    pandn_rr(&mut self.core.text, dst_fp, lhs_fp);
                }
                Ok(())
            }
            OpcodeFD::V128_OR => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                por_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::V128_XOR => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                pxor_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::I8X16_SWIZZLE => self.lower_i8x16_swizzle_scalar(dst_fp, lhs_fp, rhs_fp),
            OpcodeFD::I8X16_ADD => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                paddb_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::I8X16_SUB => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                psubb_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::I8X16_ADD_SAT_S => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                paddsb_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::I8X16_ADD_SAT_U => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                paddusb_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::I8X16_SUB_SAT_S => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                psubsb_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::I8X16_SUB_SAT_U => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                psubusb_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::I8X16_MIN_S => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                pminsb_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::I8X16_MIN_U => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                pminub_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::I8X16_MAX_S => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                pmaxsb_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::I8X16_MAX_U => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                pmaxub_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::I8X16_AVGR_U => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                pavgb_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::I8X16_EQ => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::B8,
                SimdCompareOp::Eq,
                false,
            ),
            OpcodeFD::I8X16_NE => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::B8,
                SimdCompareOp::Ne,
                false,
            ),
            OpcodeFD::I8X16_LT_S => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::B8,
                SimdCompareOp::Lt,
                false,
            ),
            OpcodeFD::I8X16_LT_U => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::B8,
                SimdCompareOp::Lt,
                true,
            ),
            OpcodeFD::I8X16_GT_S => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::B8,
                SimdCompareOp::Gt,
                false,
            ),
            OpcodeFD::I8X16_GT_U => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::B8,
                SimdCompareOp::Gt,
                true,
            ),
            OpcodeFD::I8X16_LE_S => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::B8,
                SimdCompareOp::Le,
                false,
            ),
            OpcodeFD::I8X16_LE_U => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::B8,
                SimdCompareOp::Le,
                true,
            ),
            OpcodeFD::I8X16_GE_S => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::B8,
                SimdCompareOp::Ge,
                false,
            ),
            OpcodeFD::I8X16_GE_U => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::B8,
                SimdCompareOp::Ge,
                true,
            ),
            OpcodeFD::I16X8_ADD => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                paddw_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::I16X8_SUB => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                psubw_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::I16X8_MUL => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                pmullw_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::I16X8_ADD_SAT_S => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                paddsw_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::I16X8_ADD_SAT_U => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                paddusw_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::I16X8_SUB_SAT_S => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                psubsw_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::I16X8_SUB_SAT_U => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                psubusw_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::I16X8_MIN_S => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                pminsw_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::I16X8_MIN_U => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                pminuw_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::I16X8_MAX_S => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                pmaxsw_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::I16X8_MAX_U => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                pmaxuw_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::I16X8_AVGR_U => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                pavgw_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::I16X8_Q15MULR_SAT_S => {
                self.lower_i16x8_q15mulr_sat_s_scalar(dst_fp, lhs_fp, rhs_fp)
            }
            OpcodeFD::I16X8_EXTMUL_LOW_I8X16_S => self.lower_simd_extmul_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::B8,
                true,
                false,
            ),
            OpcodeFD::I16X8_EXTMUL_HIGH_I8X16_S => self.lower_simd_extmul_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::B8,
                true,
                true,
            ),
            OpcodeFD::I16X8_EXTMUL_LOW_I8X16_U => self.lower_simd_extmul_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::B8,
                false,
                false,
            ),
            OpcodeFD::I16X8_EXTMUL_HIGH_I8X16_U => self.lower_simd_extmul_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::B8,
                false,
                true,
            ),
            OpcodeFD::I16X8_EQ => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::H16,
                SimdCompareOp::Eq,
                false,
            ),
            OpcodeFD::I16X8_NE => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::H16,
                SimdCompareOp::Ne,
                false,
            ),
            OpcodeFD::I16X8_LT_S => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::H16,
                SimdCompareOp::Lt,
                false,
            ),
            OpcodeFD::I16X8_LT_U => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::H16,
                SimdCompareOp::Lt,
                true,
            ),
            OpcodeFD::I16X8_GT_S => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::H16,
                SimdCompareOp::Gt,
                false,
            ),
            OpcodeFD::I16X8_GT_U => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::H16,
                SimdCompareOp::Gt,
                true,
            ),
            OpcodeFD::I16X8_LE_S => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::H16,
                SimdCompareOp::Le,
                false,
            ),
            OpcodeFD::I16X8_LE_U => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::H16,
                SimdCompareOp::Le,
                true,
            ),
            OpcodeFD::I16X8_GE_S => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::H16,
                SimdCompareOp::Ge,
                false,
            ),
            OpcodeFD::I16X8_GE_U => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::H16,
                SimdCompareOp::Ge,
                true,
            ),
            OpcodeFD::I32X4_ADD => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                emit_simd_rr(
                    &mut self.core.text,
                    0x66,
                    OpMap::Map0F,
                    0xFE,
                    dst_fp,
                    rhs_src,
                );
                Ok(())
            }
            OpcodeFD::I32X4_SUB => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                psubd_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::I32X4_MUL => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                pmulld_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::I32X4_MIN_S => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                pminsd_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::I32X4_MIN_U => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                pminud_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::I32X4_MAX_S => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                pmaxsd_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::I32X4_MAX_U => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                pmaxud_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::I32X4_DOT_I16X8_S => {
                self.lower_i32x4_dot_i16x8_s_scalar(dst_fp, lhs_fp, rhs_fp)
            }
            OpcodeFD::I32X4_EXTMUL_LOW_I16X8_S => self.lower_simd_extmul_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::H16,
                true,
                false,
            ),
            OpcodeFD::I32X4_EXTMUL_HIGH_I16X8_S => self.lower_simd_extmul_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::H16,
                true,
                true,
            ),
            OpcodeFD::I32X4_EXTMUL_LOW_I16X8_U => self.lower_simd_extmul_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::H16,
                false,
                false,
            ),
            OpcodeFD::I32X4_EXTMUL_HIGH_I16X8_U => self.lower_simd_extmul_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::H16,
                false,
                true,
            ),
            OpcodeFD::I32X4_EQ => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::S32,
                SimdCompareOp::Eq,
                false,
            ),
            OpcodeFD::I32X4_NE => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::S32,
                SimdCompareOp::Ne,
                false,
            ),
            OpcodeFD::I32X4_LT_S => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::S32,
                SimdCompareOp::Lt,
                false,
            ),
            OpcodeFD::I32X4_LT_U => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::S32,
                SimdCompareOp::Lt,
                true,
            ),
            OpcodeFD::I32X4_GT_S => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::S32,
                SimdCompareOp::Gt,
                false,
            ),
            OpcodeFD::I32X4_GT_U => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::S32,
                SimdCompareOp::Gt,
                true,
            ),
            OpcodeFD::I32X4_LE_S => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::S32,
                SimdCompareOp::Le,
                false,
            ),
            OpcodeFD::I32X4_LE_U => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::S32,
                SimdCompareOp::Le,
                true,
            ),
            OpcodeFD::I32X4_GE_S => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::S32,
                SimdCompareOp::Ge,
                false,
            ),
            OpcodeFD::I32X4_GE_U => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::S32,
                SimdCompareOp::Ge,
                true,
            ),
            OpcodeFD::I64X2_ADD => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                emit_simd_rr(
                    &mut self.core.text,
                    0x66,
                    OpMap::Map0F,
                    0xD4,
                    dst_fp,
                    rhs_src,
                );
                Ok(())
            }
            OpcodeFD::I64X2_SUB => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                psubq_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::I64X2_MUL => self.lower_i64x2_mul_scalar(dst_fp, lhs_fp, rhs_fp),
            OpcodeFD::I64X2_EXTMUL_LOW_I32X4_S => self.lower_simd_extmul_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::S32,
                true,
                false,
            ),
            OpcodeFD::I64X2_EXTMUL_HIGH_I32X4_S => self.lower_simd_extmul_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::S32,
                true,
                true,
            ),
            OpcodeFD::I64X2_EXTMUL_LOW_I32X4_U => self.lower_simd_extmul_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::S32,
                false,
                false,
            ),
            OpcodeFD::I64X2_EXTMUL_HIGH_I32X4_U => self.lower_simd_extmul_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::S32,
                false,
                true,
            ),
            OpcodeFD::I64X2_EQ => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::D64,
                SimdCompareOp::Eq,
                false,
            ),
            OpcodeFD::I64X2_NE => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::D64,
                SimdCompareOp::Ne,
                false,
            ),
            OpcodeFD::I64X2_LT_S => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::D64,
                SimdCompareOp::Lt,
                false,
            ),
            OpcodeFD::I64X2_GT_S => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::D64,
                SimdCompareOp::Gt,
                false,
            ),
            OpcodeFD::I64X2_LE_S => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::D64,
                SimdCompareOp::Le,
                false,
            ),
            OpcodeFD::I64X2_GE_S => self.lower_simd_int_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                IntCompareLaneWidth::D64,
                SimdCompareOp::Ge,
                false,
            ),
            OpcodeFD::F32X4_ADD => self.lower_simd_float_binary_vector(
                dst_fp,
                lhs_fp,
                rhs_src,
                SimdFloatBinaryVectorOp::AddF32,
            ),
            OpcodeFD::F32X4_SUB => self.lower_simd_float_binary_vector(
                dst_fp,
                lhs_fp,
                rhs_src,
                SimdFloatBinaryVectorOp::SubF32,
            ),
            OpcodeFD::F32X4_MUL => self.lower_simd_float_binary_vector(
                dst_fp,
                lhs_fp,
                rhs_src,
                SimdFloatBinaryVectorOp::MulF32,
            ),
            OpcodeFD::F32X4_DIV => self.lower_simd_float_binary_vector(
                dst_fp,
                lhs_fp,
                rhs_src,
                SimdFloatBinaryVectorOp::DivF32,
            ),
            OpcodeFD::F32X4_MIN => self.lower_simd_float_minmax_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                FloatCompareLaneWidth::F32,
                false,
                true,
            ),
            OpcodeFD::F32X4_MAX => self.lower_simd_float_minmax_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                FloatCompareLaneWidth::F32,
                true,
                true,
            ),
            OpcodeFD::F32X4_PMIN => self.lower_simd_float_minmax_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                FloatCompareLaneWidth::F32,
                false,
                false,
            ),
            OpcodeFD::F32X4_PMAX => self.lower_simd_float_minmax_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                FloatCompareLaneWidth::F32,
                true,
                false,
            ),
            OpcodeFD::F32X4_EQ => self.lower_simd_float_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                FloatCompareLaneWidth::F32,
                SimdCompareOp::Eq,
            ),
            OpcodeFD::F32X4_NE => self.lower_simd_float_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                FloatCompareLaneWidth::F32,
                SimdCompareOp::Ne,
            ),
            OpcodeFD::F32X4_LT => self.lower_simd_float_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                FloatCompareLaneWidth::F32,
                SimdCompareOp::Lt,
            ),
            OpcodeFD::F32X4_GT => self.lower_simd_float_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                FloatCompareLaneWidth::F32,
                SimdCompareOp::Gt,
            ),
            OpcodeFD::F32X4_LE => self.lower_simd_float_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                FloatCompareLaneWidth::F32,
                SimdCompareOp::Le,
            ),
            OpcodeFD::F32X4_GE => self.lower_simd_float_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                FloatCompareLaneWidth::F32,
                SimdCompareOp::Ge,
            ),
            OpcodeFD::F64X2_ADD => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                addpd_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::F64X2_SUB => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                subpd_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::F64X2_MUL => {
                if dst_fp != lhs_fp as u8 {
                    movdqa_rr(&mut self.core.text, dst_fp, lhs_fp as u8);
                }
                mulpd_rr(&mut self.core.text, dst_fp, rhs_src);
                Ok(())
            }
            OpcodeFD::F64X2_DIV => self.lower_simd_float_binary_vector(
                dst_fp,
                lhs_fp,
                rhs_src,
                SimdFloatBinaryVectorOp::DivF64,
            ),
            OpcodeFD::F64X2_MIN => self.lower_simd_float_minmax_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                FloatCompareLaneWidth::F64,
                false,
                true,
            ),
            OpcodeFD::F64X2_MAX => self.lower_simd_float_minmax_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                FloatCompareLaneWidth::F64,
                true,
                true,
            ),
            OpcodeFD::F64X2_PMIN => self.lower_simd_float_minmax_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                FloatCompareLaneWidth::F64,
                false,
                false,
            ),
            OpcodeFD::F64X2_PMAX => self.lower_simd_float_minmax_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                FloatCompareLaneWidth::F64,
                true,
                false,
            ),
            OpcodeFD::F64X2_EQ => self.lower_simd_float_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                FloatCompareLaneWidth::F64,
                SimdCompareOp::Eq,
            ),
            OpcodeFD::F64X2_NE => self.lower_simd_float_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                FloatCompareLaneWidth::F64,
                SimdCompareOp::Ne,
            ),
            OpcodeFD::F64X2_LT => self.lower_simd_float_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                FloatCompareLaneWidth::F64,
                SimdCompareOp::Lt,
            ),
            OpcodeFD::F64X2_GT => self.lower_simd_float_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                FloatCompareLaneWidth::F64,
                SimdCompareOp::Gt,
            ),
            OpcodeFD::F64X2_LE => self.lower_simd_float_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                FloatCompareLaneWidth::F64,
                SimdCompareOp::Le,
            ),
            OpcodeFD::F64X2_GE => self.lower_simd_float_compare_scalar(
                dst_fp,
                lhs_fp,
                rhs_fp,
                FloatCompareLaneWidth::F64,
                SimdCompareOp::Ge,
            ),
            other => Err(unsupported_simd_opcode("binary", other as u32)),
        }
    }

    fn lower_simd_ternary_native(
        &mut self,
        opcode: u32,
        dst: MachineReg,
        a: MachineValue,
        b: MachineValue,
        c: MachineValue,
    ) -> Result<(), WasmError> {
        let dst_fp = self.map_fp_reg(dst)? as u8;
        let a_fp = expect_v128_reg(self, a, "x86_64 SIMD ternary arg a must be in an FP reg")?;
        let b_fp = expect_v128_reg(self, b, "x86_64 SIMD ternary arg b must be in an FP reg")?;
        let c_fp = expect_v128_reg(self, c, "x86_64 SIMD ternary arg c must be in an FP reg")?;
        match OpcodeFD::try_from(opcode)? {
            OpcodeFD::V128_BITSELECT => {
                let masked_b = self.fp_scratch.scoped_alloc().detach();
                let mask_tmp = self.fp_scratch.scoped_alloc().detach();
                movdqa_rr(&mut self.core.text, *masked_b as u8, c_fp);
                movdqa_rr(&mut self.core.text, *mask_tmp as u8, c_fp);
                pandn_rr(&mut self.core.text, *masked_b as u8, b_fp);
                if dst_fp != a_fp {
                    movdqa_rr(&mut self.core.text, dst_fp, a_fp);
                }
                pand_rr(&mut self.core.text, dst_fp, *mask_tmp as u8);
                por_rr(&mut self.core.text, dst_fp, *masked_b as u8);
                Ok(())
            }
            other => Err(unsupported_simd_opcode("ternary", other as u32)),
        }
    }

    fn lower_simd_shift_native(
        &mut self,
        opcode: u32,
        dst: MachineReg,
        vector: MachineValue,
        shift: MachineValue,
    ) -> Result<(), WasmError> {
        let (lane_width, shift_kind) = match OpcodeFD::try_from(opcode)? {
            OpcodeFD::I8X16_SHL => (ShiftLaneWidth::B8, ShiftKind::Left),
            OpcodeFD::I8X16_SHR_S => (ShiftLaneWidth::B8, ShiftKind::RightSigned),
            OpcodeFD::I8X16_SHR_U => (ShiftLaneWidth::B8, ShiftKind::RightUnsigned),
            OpcodeFD::I16X8_SHL => (ShiftLaneWidth::H16, ShiftKind::Left),
            OpcodeFD::I16X8_SHR_S => (ShiftLaneWidth::H16, ShiftKind::RightSigned),
            OpcodeFD::I16X8_SHR_U => (ShiftLaneWidth::H16, ShiftKind::RightUnsigned),
            OpcodeFD::I32X4_SHL => (ShiftLaneWidth::S32, ShiftKind::Left),
            OpcodeFD::I32X4_SHR_S => (ShiftLaneWidth::S32, ShiftKind::RightSigned),
            OpcodeFD::I32X4_SHR_U => (ShiftLaneWidth::S32, ShiftKind::RightUnsigned),
            OpcodeFD::I64X2_SHL => (ShiftLaneWidth::D64, ShiftKind::Left),
            OpcodeFD::I64X2_SHR_S => (ShiftLaneWidth::D64, ShiftKind::RightSigned),
            OpcodeFD::I64X2_SHR_U => (ShiftLaneWidth::D64, ShiftKind::RightUnsigned),
            other => return Err(unsupported_simd_opcode("shift", other as u32)),
        };
        let src_fp = expect_v128_reg(
            self,
            vector,
            "x86_64 SIMD shift vector must be in an FP reg",
        )?;
        let dst_fp = self.map_fp_reg(dst)? as u8;

        let count_src = self.materialize_value(X86Reg::RAX, shift)?;
        if count_src != X86Reg::RCX {
            enc::mov_rr_32(&mut self.core.text, X86Reg::RCX, count_src);
        }
        enc::and_ri_32(&mut self.core.text, X86Reg::RCX, lane_width.mask());

        enc::sub_rsp_imm8(&mut self.core.text, 16);
        movdqu_store(&mut self.core.text, X86Reg::RSP, 0, src_fp);
        for lane in 0..lane_width.lane_count() {
            let offset = lane as i32 * lane_width.byte_width() as i32;
            self.lower_scalar_shift_lane(offset, lane_width, shift_kind)?;
        }
        movdqu_load(&mut self.core.text, dst_fp, X86Reg::RSP, 0);
        enc::add_rsp_imm8(&mut self.core.text, 16);
        Ok(())
    }

    fn lower_simd_extract_lane_native(
        &mut self,
        opcode: u32,
        lane: u8,
        dst_ty: MachineStorageType,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let src_fp = expect_v128_reg(
            self,
            src,
            "x86_64 SIMD extract-lane source must be in an FP reg",
        )?;
        enc::sub_rsp_imm8(&mut self.core.text, 16);
        movdqu_store(&mut self.core.text, X86Reg::RSP, 0, src_fp);
        match OpcodeFD::try_from(opcode)? {
            OpcodeFD::I8X16_EXTRACT_LANE_S => {
                debug_assert_eq!(dst_ty, MachineStorageType::GpWord);
                let dst_gp = self.map_gp_reg(dst)?;
                enc::load_s8_64(&mut self.core.text, dst_gp, X86Reg::RSP, lane as i32);
            }
            OpcodeFD::I8X16_EXTRACT_LANE_U => {
                debug_assert_eq!(dst_ty, MachineStorageType::GpWord);
                let dst_gp = self.map_gp_reg(dst)?;
                enc::load_u8(&mut self.core.text, dst_gp, X86Reg::RSP, lane as i32);
            }
            OpcodeFD::I16X8_EXTRACT_LANE_S => {
                debug_assert_eq!(dst_ty, MachineStorageType::GpWord);
                let dst_gp = self.map_gp_reg(dst)?;
                enc::load_s16_64(&mut self.core.text, dst_gp, X86Reg::RSP, lane as i32 * 2);
            }
            OpcodeFD::I16X8_EXTRACT_LANE_U => {
                debug_assert_eq!(dst_ty, MachineStorageType::GpWord);
                let dst_gp = self.map_gp_reg(dst)?;
                enc::load_u16(&mut self.core.text, dst_gp, X86Reg::RSP, lane as i32 * 2);
            }
            OpcodeFD::I32X4_EXTRACT_LANE => {
                debug_assert_eq!(dst_ty, MachineStorageType::GpWord);
                let dst_gp = self.map_gp_reg(dst)?;
                enc::load_32(&mut self.core.text, dst_gp, X86Reg::RSP, lane as i32 * 4);
            }
            OpcodeFD::I64X2_EXTRACT_LANE => {
                debug_assert!(matches!(
                    dst_ty,
                    MachineStorageType::GpWord | MachineStorageType::GpI64
                ));
                let dst_gp = self.map_gp_reg(dst)?;
                enc::load_64(&mut self.core.text, dst_gp, X86Reg::RSP, lane as i32 * 8);
            }
            OpcodeFD::F32X4_EXTRACT_LANE => {
                debug_assert_eq!(dst_ty, MachineStorageType::Fp32);
                let dst_fp = self.map_fp_reg(dst)? as u8;
                enc::movss_load(&mut self.core.text, dst_fp, X86Reg::RSP, lane as i32 * 4);
                self.core.set_fp_reg_width(dst, MachineFloatWidth::F32)?;
            }
            OpcodeFD::F64X2_EXTRACT_LANE => {
                debug_assert_eq!(dst_ty, MachineStorageType::Fp64);
                let dst_fp = self.map_fp_reg(dst)? as u8;
                enc::movsd_load(&mut self.core.text, dst_fp, X86Reg::RSP, lane as i32 * 8);
                self.core.set_fp_reg_width(dst, MachineFloatWidth::F64)?;
            }
            other => {
                enc::add_rsp_imm8(&mut self.core.text, 16);
                return Err(unsupported_simd_opcode("extract_lane", other as u32));
            }
        }
        enc::add_rsp_imm8(&mut self.core.text, 16);
        Ok(())
    }

    fn materialize_v128_constant_into_fp(&mut self, dst_fp: u8, bytes: [u8; 16]) {
        let low = u64::from_le_bytes(bytes[..8].try_into().expect("slice length"));
        let high = u64::from_le_bytes(bytes[8..].try_into().expect("slice length"));
        enc::sub_rsp_imm8(&mut self.core.text, 16);
        {
            let gp0 = self.gp_scratch.scoped_alloc().detach();
            let gp1 = self.gp_scratch.scoped_alloc().detach();
            self.materialize_u64(*gp0, low);
            self.materialize_u64(*gp1, high);
            enc::store_64(&mut self.core.text, X86Reg::RSP, 0, *gp0);
            enc::store_64(&mut self.core.text, X86Reg::RSP, 8, *gp1);
        }
        movdqu_load(&mut self.core.text, dst_fp, X86Reg::RSP, 0);
        enc::add_rsp_imm8(&mut self.core.text, 16);
    }

    fn lower_v128_const_native(
        &mut self,
        dst: MachineReg,
        bytes: [u8; 16],
    ) -> Result<(), WasmError> {
        let dst_fp = self.map_fp_reg(dst)? as u8;
        self.materialize_v128_constant_into_fp(dst_fp, bytes);
        Ok(())
    }

    fn emit_preserved_call_for_v128_from_raw(&mut self, dst_fp: u8) {
        let call_target = X86Reg::R11;
        let error_label = self.core.new_label();
        let done_label = self.core.new_label();

        enc::mov_rr_64(&mut self.core.text, C_ARG0, map_fixed_reg(MACHINE_CTX_REG));
        self.materialize_u64(C_ARG1, preserved_op::V128_FROM_RAW as u64);
        enc::mov_rr_64(&mut self.core.text, C_ARG2, X86Reg::RSP);
        enc::add_ri_64(&mut self.core.text, C_ARG2, V128_PREFIX_BYTES as i32);
        self.materialize_u64(call_target, preserved_entry as usize as u64);
        enc::call_reg(&mut self.core.text, call_target);

        test_rr_64(&mut self.core.text, C_RET0, C_RET0);
        self.emit_jcc(Cc::NE, error_label);

        for index in 0..abi::FP_MACHINE_REG_COUNT {
            let xmm = abi::fp_machine_reg(index).expect("x86_64 FP dynamic reg") as u8;
            let disp = V128_PREFIX_BYTES as i32
                + (preserved_io::SLOT_COUNT as i32 * 8)
                + (index as i32) * 16;
            movdqu_load(&mut self.core.text, xmm, X86Reg::RSP, disp);
        }
        movdqu_load(&mut self.core.text, dst_fp, X86Reg::RSP, 0);
        enc::add_rsp_imm32(
            &mut self.core.text,
            V128_PREFIX_BYTES
                + (preserved_io::SLOT_COUNT as u32 * 8)
                + (abi::FP_MACHINE_REG_COUNT as u32) * 16,
        );
        self.restore_caller_clobbered_gp_dynamic();
        self.emit_jmp(done_label);

        self.core.bind_label(error_label);
        for index in 0..abi::FP_MACHINE_REG_COUNT {
            let xmm = abi::fp_machine_reg(index).expect("x86_64 FP dynamic reg") as u8;
            let disp = V128_PREFIX_BYTES as i32
                + (preserved_io::SLOT_COUNT as i32 * 8)
                + (index as i32) * 16;
            movdqu_load(&mut self.core.text, xmm, X86Reg::RSP, disp);
        }
        enc::add_rsp_imm32(
            &mut self.core.text,
            V128_PREFIX_BYTES
                + (preserved_io::SLOT_COUNT as u32 * 8)
                + (abi::FP_MACHINE_REG_COUNT as u32) * 16,
        );
        self.restore_caller_clobbered_gp_dynamic();
        let body_local_error_label = self.core.body_local_error_label;
        self.emit_jmp(body_local_error_label);

        self.core.bind_label(done_label);
    }

    fn lower_v128_from_raw(&mut self, dst: MachineReg, raw: MachineValue) -> Result<(), WasmError> {
        self.emit_preserved_io_open_with_prefix(V128_PREFIX_BYTES);
        self.emit_io_store_value_at(V128_PREFIX_BYTES, preserved_io::ARG0, raw)?;
        let dst_fp = self.map_fp_reg(dst)? as u8;
        self.emit_preserved_call_for_v128_from_raw(dst_fp);
        Ok(())
    }

    fn lower_v128_to_raw(&mut self, dst: MachineReg, src: MachineValue) -> Result<(), WasmError> {
        let src_fp = expect_v128_reg(
            self,
            src,
            "x86_64 v128->raw conversion needs a vector source",
        )?;
        let dst_gp = self.map_gp_reg(dst)?;
        self.emit_preserved_io_open_with_prefix(V128_PREFIX_BYTES);
        movdqu_store(&mut self.core.text, X86Reg::RSP, 0, src_fp);
        self.emit_preserved_call_and_close_with_prefix(
            preserved_op::V128_TO_RAW,
            Some(dst_gp),
            V128_PREFIX_BYTES,
        );
        Ok(())
    }

    fn lower_simd_all_true(
        &mut self,
        src_fp: u8,
        dst: MachineReg,
        width: SimdAllTrueWidth,
    ) -> Result<(), WasmError> {
        let tmp = self.fp_scratch.scoped_alloc().detach();
        pxor_rr(&mut self.core.text, *tmp as u8, *tmp as u8);
        match width {
            SimdAllTrueWidth::B8 => pcmpeqb_rr(&mut self.core.text, *tmp as u8, src_fp),
            SimdAllTrueWidth::H16 => pcmpeqw_rr(&mut self.core.text, *tmp as u8, src_fp),
            SimdAllTrueWidth::S32 => pcmpeqd_rr(&mut self.core.text, *tmp as u8, src_fp),
            SimdAllTrueWidth::D64 => pcmpeqq_rr(&mut self.core.text, *tmp as u8, src_fp),
        }
        ptest_rr(&mut self.core.text, *tmp as u8, *tmp as u8);
        let dst_gp = self.map_gp_reg(dst)?;
        enc::setcc(&mut self.core.text, Cc::E, dst_gp);
        enc::movzx_r32_r8(&mut self.core.text, dst_gp, dst_gp);
        Ok(())
    }

    fn lower_simd_bitmask(
        &mut self,
        src_fp: u8,
        dst: MachineReg,
        width: SimdBitmaskWidth,
    ) -> Result<(), WasmError> {
        let dst_gp = self.map_gp_reg(dst)?;
        let tmp_gp = self.gp_scratch.scoped_alloc().detach();
        pmovmskb_r32_xmm(&mut self.core.text, dst_gp, src_fp);
        match width {
            SimdBitmaskWidth::B8 => {}
            SimdBitmaskWidth::H16 => {
                enc::and_ri_32(&mut self.core.text, dst_gp, 0xAAAA);
                enc::shr_imm_32(&mut self.core.text, dst_gp, 1);
                enc::mov_rr_32(&mut self.core.text, *tmp_gp, dst_gp);
                enc::shr_imm_32(&mut self.core.text, *tmp_gp, 1);
                enc::or_rr_32(&mut self.core.text, dst_gp, *tmp_gp);
                enc::and_ri_32(&mut self.core.text, dst_gp, 0x3333);
                enc::mov_rr_32(&mut self.core.text, *tmp_gp, dst_gp);
                enc::shr_imm_32(&mut self.core.text, *tmp_gp, 2);
                enc::or_rr_32(&mut self.core.text, dst_gp, *tmp_gp);
                enc::and_ri_32(&mut self.core.text, dst_gp, 0x0F0F);
                enc::mov_rr_32(&mut self.core.text, *tmp_gp, dst_gp);
                enc::shr_imm_32(&mut self.core.text, *tmp_gp, 4);
                enc::or_rr_32(&mut self.core.text, dst_gp, *tmp_gp);
                enc::and_ri_32(&mut self.core.text, dst_gp, 0x00FF);
            }
            SimdBitmaskWidth::S32 => {
                enc::and_ri_32(&mut self.core.text, dst_gp, 0x8888);
                enc::shr_imm_32(&mut self.core.text, dst_gp, 3);
                enc::mov_rr_32(&mut self.core.text, *tmp_gp, dst_gp);
                enc::shr_imm_32(&mut self.core.text, *tmp_gp, 3);
                enc::or_rr_32(&mut self.core.text, dst_gp, *tmp_gp);
                enc::and_ri_32(&mut self.core.text, dst_gp, 0x0303);
                enc::mov_rr_32(&mut self.core.text, *tmp_gp, dst_gp);
                enc::shr_imm_32(&mut self.core.text, *tmp_gp, 6);
                enc::or_rr_32(&mut self.core.text, dst_gp, *tmp_gp);
                enc::and_ri_32(&mut self.core.text, dst_gp, 0x0F);
            }
            SimdBitmaskWidth::D64 => {
                enc::and_ri_32(&mut self.core.text, dst_gp, 0x8080);
                enc::shr_imm_32(&mut self.core.text, dst_gp, 7);
                enc::mov_rr_32(&mut self.core.text, *tmp_gp, dst_gp);
                enc::shr_imm_32(&mut self.core.text, *tmp_gp, 7);
                enc::or_rr_32(&mut self.core.text, dst_gp, *tmp_gp);
                enc::and_ri_32(&mut self.core.text, dst_gp, 0x03);
            }
        }
        Ok(())
    }

    fn lower_simd_splat_integer(
        &mut self,
        dst: MachineReg,
        src: MachineValue,
        width: IntegerSplatWidth,
    ) -> Result<(), WasmError> {
        let dst_fp = self.map_fp_reg(dst)? as u8;
        match width {
            IntegerSplatWidth::B8 => {
                let scratch0 = self.gp_scratch.scoped_alloc().detach();
                let scratch1 = self.gp_scratch.scoped_alloc().detach();
                let gp = self.materialize_value(*scratch0, src)?;
                if gp != *scratch0 {
                    enc::mov_rr_32(&mut self.core.text, *scratch0, gp);
                }
                enc::and_ri_32(&mut self.core.text, *scratch0, 0xFF);
                enc::mov_ri_32(&mut self.core.text, *scratch1, 0x01010101);
                enc::imul_rr_32(&mut self.core.text, *scratch0, *scratch1);
                enc::movd_xmm_r32(&mut self.core.text, dst_fp, *scratch0);
                pshufd_rr_imm8(&mut self.core.text, dst_fp, dst_fp, 0);
                Ok(())
            }
            IntegerSplatWidth::H16 => {
                let scratch0 = self.gp_scratch.scoped_alloc().detach();
                let scratch1 = self.gp_scratch.scoped_alloc().detach();
                let gp = self.materialize_value(*scratch0, src)?;
                if gp != *scratch0 {
                    enc::mov_rr_32(&mut self.core.text, *scratch0, gp);
                }
                enc::and_ri_32(&mut self.core.text, *scratch0, 0xFFFF);
                enc::mov_ri_32(&mut self.core.text, *scratch1, 0x0001_0001);
                enc::imul_rr_32(&mut self.core.text, *scratch0, *scratch1);
                enc::movd_xmm_r32(&mut self.core.text, dst_fp, *scratch0);
                pshufd_rr_imm8(&mut self.core.text, dst_fp, dst_fp, 0);
                Ok(())
            }
            IntegerSplatWidth::S32 => {
                let scratch0 = self.gp_scratch.scoped_alloc().detach();
                let gp = self.materialize_value(*scratch0, src)?;
                enc::movd_xmm_r32(&mut self.core.text, dst_fp, gp);
                pshufd_rr_imm8(&mut self.core.text, dst_fp, dst_fp, 0);
                Ok(())
            }
            IntegerSplatWidth::D64 => {
                let scratch0 = self.gp_scratch.scoped_alloc().detach();
                let gp = self.materialize_value(*scratch0, src)?;
                enc::movq_xmm_r64(&mut self.core.text, dst_fp, gp);
                punpcklqdq_rr(&mut self.core.text, dst_fp, dst_fp);
                Ok(())
            }
        }
    }

    fn lower_simd_splat_float(
        &mut self,
        dst: MachineReg,
        src: MachineValue,
        width: FloatSplatWidth,
    ) -> Result<(), WasmError> {
        let dst_fp = self.map_fp_reg(dst)? as u8;
        match width {
            FloatSplatWidth::F32 => {
                match src {
                    MachineValue::Reg(reg) if self.core.is_fp_reg(reg) => {
                        let src_fp = self.map_fp_reg(reg)? as u8;
                        if dst_fp != src_fp {
                            movdqa_rr(&mut self.core.text, dst_fp, src_fp);
                        }
                    }
                    _ => {
                        let scratch0 = self.gp_scratch.scoped_alloc().detach();
                        let gp = self.materialize_value(*scratch0, src)?;
                        enc::movd_xmm_r32(&mut self.core.text, dst_fp, gp);
                    }
                }
                pshufd_rr_imm8(&mut self.core.text, dst_fp, dst_fp, 0);
                Ok(())
            }
            FloatSplatWidth::F64 => {
                match src {
                    MachineValue::Reg(reg) if self.core.is_fp_reg(reg) => {
                        let src_fp = self.map_fp_reg(reg)? as u8;
                        if dst_fp != src_fp {
                            movdqa_rr(&mut self.core.text, dst_fp, src_fp);
                        }
                    }
                    _ => {
                        let scratch0 = self.gp_scratch.scoped_alloc().detach();
                        let gp = self.materialize_value(*scratch0, src)?;
                        enc::movq_xmm_r64(&mut self.core.text, dst_fp, gp);
                    }
                }
                punpcklqdq_rr(&mut self.core.text, dst_fp, dst_fp);
                Ok(())
            }
        }
    }

    fn lower_simd_int_unary_vector(
        &mut self,
        dst: MachineReg,
        src_fp: u8,
        op: SimdIntUnaryVectorOp,
    ) -> Result<(), WasmError> {
        let dst_fp = self.map_fp_reg(dst)? as u8;
        match op {
            SimdIntUnaryVectorOp::AbsB8 => pabsb_rr(&mut self.core.text, dst_fp, src_fp),
            SimdIntUnaryVectorOp::AbsH16 => pabsw_rr(&mut self.core.text, dst_fp, src_fp),
            SimdIntUnaryVectorOp::AbsS32 => pabsd_rr(&mut self.core.text, dst_fp, src_fp),
            SimdIntUnaryVectorOp::NegB8 => self.lower_simd_int_neg_vector(dst_fp, src_fp, psubb_rr),
            SimdIntUnaryVectorOp::NegH16 => {
                self.lower_simd_int_neg_vector(dst_fp, src_fp, psubw_rr)
            }
            SimdIntUnaryVectorOp::NegS32 => {
                self.lower_simd_int_neg_vector(dst_fp, src_fp, psubd_rr)
            }
            SimdIntUnaryVectorOp::NegD64 => {
                self.lower_simd_int_neg_vector(dst_fp, src_fp, psubq_rr)
            }
        }
        Ok(())
    }

    fn lower_simd_int_neg_vector(
        &mut self,
        dst_fp: u8,
        src_fp: u8,
        emit_sub: fn(&mut TextEmitter, u8, u8),
    ) {
        if dst_fp == src_fp {
            let zero_fp = self.fp_scratch.scoped_alloc().detach();
            pxor_rr(&mut self.core.text, *zero_fp as u8, *zero_fp as u8);
            emit_sub(&mut self.core.text, *zero_fp as u8, src_fp);
            movdqa_rr(&mut self.core.text, dst_fp, *zero_fp as u8);
        } else {
            pxor_rr(&mut self.core.text, dst_fp, dst_fp);
            emit_sub(&mut self.core.text, dst_fp, src_fp);
        }
    }

    fn lower_i64x2_abs_scalar(&mut self, dst_fp: u8, src_fp: u8) -> Result<(), WasmError> {
        enc::sub_rsp_imm8(&mut self.core.text, 16);
        movdqu_store(&mut self.core.text, X86Reg::RSP, 0, src_fp);
        for lane in 0..2 {
            let offset = lane * 8;
            enc::load_64(&mut self.core.text, X86Reg::RAX, X86Reg::RSP, offset);
            enc::mov_rr_64(&mut self.core.text, X86Reg::RDX, X86Reg::RAX);
            enc::sar_imm_64(&mut self.core.text, X86Reg::RDX, 63);
            enc::xor_rr_64(&mut self.core.text, X86Reg::RAX, X86Reg::RDX);
            enc::sub_rr_64(&mut self.core.text, X86Reg::RAX, X86Reg::RDX);
            enc::store_64(&mut self.core.text, X86Reg::RSP, offset, X86Reg::RAX);
        }
        movdqu_load(&mut self.core.text, dst_fp, X86Reg::RSP, 0);
        enc::add_rsp_imm8(&mut self.core.text, 16);
        Ok(())
    }

    fn lower_f32x4_convert_i32x4_u_scalar(
        &mut self,
        dst_fp: u8,
        src_fp: u8,
    ) -> Result<(), WasmError> {
        enc::sub_rsp_imm8(&mut self.core.text, 32);
        movdqu_store(&mut self.core.text, X86Reg::RSP, 0, src_fp);
        for lane in 0..4 {
            let src_off = lane * 4;
            let out_off = 16 + src_off;
            enc::load_32(&mut self.core.text, X86Reg::RAX, X86Reg::RSP, src_off);
            enc::cvtsi2ss_r64(&mut self.core.text, 0, X86Reg::RAX);
            enc::movss_store(&mut self.core.text, X86Reg::RSP, out_off, 0);
        }
        movdqu_load(&mut self.core.text, dst_fp, X86Reg::RSP, 16);
        enc::add_rsp_imm8(&mut self.core.text, 32);
        Ok(())
    }

    fn lower_i8x16_popcnt_scalar(&mut self, dst_fp: u8, src_fp: u8) -> Result<(), WasmError> {
        enc::sub_rsp_imm8(&mut self.core.text, 32);
        movdqu_store(&mut self.core.text, X86Reg::RSP, 0, src_fp);
        for lane in 0..16 {
            enc::load_u8(&mut self.core.text, X86Reg::RAX, X86Reg::RSP, lane);
            enc::mov_rr_32(&mut self.core.text, X86Reg::RDX, X86Reg::RAX);
            enc::shr_imm_32(&mut self.core.text, X86Reg::RDX, 1);
            enc::and_ri_32(&mut self.core.text, X86Reg::RDX, 0x55);
            enc::sub_rr_32(&mut self.core.text, X86Reg::RAX, X86Reg::RDX);
            enc::mov_rr_32(&mut self.core.text, X86Reg::RDX, X86Reg::RAX);
            enc::and_ri_32(&mut self.core.text, X86Reg::RAX, 0x33);
            enc::shr_imm_32(&mut self.core.text, X86Reg::RDX, 2);
            enc::and_ri_32(&mut self.core.text, X86Reg::RDX, 0x33);
            enc::add_rr_32(&mut self.core.text, X86Reg::RAX, X86Reg::RDX);
            enc::mov_rr_32(&mut self.core.text, X86Reg::RCX, X86Reg::RAX);
            enc::shr_imm_32(&mut self.core.text, X86Reg::RCX, 4);
            enc::add_rr_32(&mut self.core.text, X86Reg::RAX, X86Reg::RCX);
            enc::and_ri_32(&mut self.core.text, X86Reg::RAX, 0x0f);
            enc::store_8(&mut self.core.text, X86Reg::RSP, 16 + lane, X86Reg::RAX);
        }
        movdqu_load(&mut self.core.text, dst_fp, X86Reg::RSP, 16);
        enc::add_rsp_imm8(&mut self.core.text, 32);
        Ok(())
    }

    fn lower_simd_pairwise_extadd_scalar(
        &mut self,
        dst_fp: u8,
        src_fp: u8,
        src_width: IntCompareLaneWidth,
        signed: bool,
    ) -> Result<(), WasmError> {
        enc::sub_rsp_imm8(&mut self.core.text, 32);
        movdqu_store(&mut self.core.text, X86Reg::RSP, 0, src_fp);
        let dst_width = match src_width {
            IntCompareLaneWidth::B8 => IntCompareLaneWidth::H16,
            IntCompareLaneWidth::H16 => IntCompareLaneWidth::S32,
            _ => return Err(WasmError::internal("invalid pairwise extadd source width")),
        };
        for lane in 0..dst_width.lane_count() {
            let lhs_off = (lane * 2 * src_width.byte_width() as usize) as i32;
            let rhs_off = lhs_off + src_width.byte_width() as i32;
            let out_off = 16 + (lane * dst_width.byte_width() as usize) as i32;
            self.emit_load_int_lane_to(X86Reg::RAX, X86Reg::RSP, lhs_off, src_width, signed);
            self.emit_load_int_lane_to(X86Reg::RDX, X86Reg::RSP, rhs_off, src_width, signed);
            enc::add_rr_64(&mut self.core.text, X86Reg::RAX, X86Reg::RDX);
            self.emit_store_int_lane_from(X86Reg::RSP, out_off, dst_width, X86Reg::RAX);
        }
        movdqu_load(&mut self.core.text, dst_fp, X86Reg::RSP, 16);
        enc::add_rsp_imm8(&mut self.core.text, 32);
        Ok(())
    }

    fn lower_simd_extend_scalar(
        &mut self,
        dst_fp: u8,
        src_fp: u8,
        src_width: IntCompareLaneWidth,
        signed: bool,
        high_half: bool,
    ) -> Result<(), WasmError> {
        enc::sub_rsp_imm8(&mut self.core.text, 32);
        movdqu_store(&mut self.core.text, X86Reg::RSP, 0, src_fp);
        let dst_width = match src_width {
            IntCompareLaneWidth::B8 => IntCompareLaneWidth::H16,
            IntCompareLaneWidth::H16 => IntCompareLaneWidth::S32,
            IntCompareLaneWidth::S32 => IntCompareLaneWidth::D64,
            IntCompareLaneWidth::D64 => {
                return Err(WasmError::internal("invalid extend source width"))
            }
        };
        let src_start = if high_half { dst_width.lane_count() } else { 0 };
        for lane in 0..dst_width.lane_count() {
            let src_lane = src_start + lane;
            let src_off = (src_lane * src_width.byte_width() as usize) as i32;
            let out_off = 16 + (lane * dst_width.byte_width() as usize) as i32;
            self.emit_load_int_lane_to(X86Reg::RAX, X86Reg::RSP, src_off, src_width, signed);
            self.emit_store_int_lane_from(X86Reg::RSP, out_off, dst_width, X86Reg::RAX);
        }
        movdqu_load(&mut self.core.text, dst_fp, X86Reg::RSP, 16);
        enc::add_rsp_imm8(&mut self.core.text, 32);
        Ok(())
    }

    fn lower_simd_narrow_scalar(
        &mut self,
        dst_fp: u8,
        lhs_fp: u8,
        rhs_fp: u8,
        src_width: IntCompareLaneWidth,
        signed: bool,
    ) -> Result<(), WasmError> {
        let work_fp = if dst_fp == rhs_fp && dst_fp != lhs_fp {
            let tmp = self.fp_scratch.scoped_alloc().detach();
            let tmp_fp = *tmp as u8;
            movdqa_rr(&mut self.core.text, tmp_fp, lhs_fp);
            tmp_fp
        } else {
            if dst_fp != lhs_fp {
                movdqa_rr(&mut self.core.text, dst_fp, lhs_fp);
            }
            dst_fp
        };
        match (src_width, signed) {
            (IntCompareLaneWidth::H16, true) => packsswb_rr(&mut self.core.text, work_fp, rhs_fp),
            (IntCompareLaneWidth::H16, false) => packuswb_rr(&mut self.core.text, work_fp, rhs_fp),
            (IntCompareLaneWidth::S32, true) => packssdw_rr(&mut self.core.text, work_fp, rhs_fp),
            (IntCompareLaneWidth::S32, false) => packusdw_rr(&mut self.core.text, work_fp, rhs_fp),
            _ => return Err(WasmError::internal("invalid narrow source width")),
        }
        if work_fp != dst_fp {
            movdqa_rr(&mut self.core.text, dst_fp, work_fp);
        }
        Ok(())
    }

    fn lower_simd_extmul_scalar(
        &mut self,
        dst_fp: u8,
        lhs_fp: u8,
        rhs_fp: u8,
        src_width: IntCompareLaneWidth,
        signed: bool,
        high_half: bool,
    ) -> Result<(), WasmError> {
        enc::sub_rsp_imm8(&mut self.core.text, 48);
        movdqu_store(&mut self.core.text, X86Reg::RSP, 0, lhs_fp);
        movdqu_store(&mut self.core.text, X86Reg::RSP, 16, rhs_fp);
        let dst_width = match src_width {
            IntCompareLaneWidth::B8 => IntCompareLaneWidth::H16,
            IntCompareLaneWidth::H16 => IntCompareLaneWidth::S32,
            IntCompareLaneWidth::S32 => IntCompareLaneWidth::D64,
            IntCompareLaneWidth::D64 => {
                return Err(WasmError::internal("invalid extmul source width"))
            }
        };
        let src_start = if high_half { dst_width.lane_count() } else { 0 };
        for lane in 0..dst_width.lane_count() {
            let src_lane = src_start + lane;
            let lhs_off = (src_lane * src_width.byte_width() as usize) as i32;
            let rhs_off = 16 + lhs_off;
            let out_off = 32 + (lane * dst_width.byte_width() as usize) as i32;
            self.emit_load_int_lane_to(X86Reg::RAX, X86Reg::RSP, lhs_off, src_width, signed);
            self.emit_load_int_lane_to(X86Reg::RDX, X86Reg::RSP, rhs_off, src_width, signed);
            enc::imul_rr_64(&mut self.core.text, X86Reg::RAX, X86Reg::RDX);
            self.emit_store_int_lane_from(X86Reg::RSP, out_off, dst_width, X86Reg::RAX);
        }
        movdqu_load(&mut self.core.text, dst_fp, X86Reg::RSP, 32);
        enc::add_rsp_imm8(&mut self.core.text, 48);
        Ok(())
    }

    fn lower_i16x8_q15mulr_sat_s_scalar(
        &mut self,
        dst_fp: u8,
        lhs_fp: u8,
        rhs_fp: u8,
    ) -> Result<(), WasmError> {
        enc::sub_rsp_imm8(&mut self.core.text, 48);
        movdqu_store(&mut self.core.text, X86Reg::RSP, 0, lhs_fp);
        movdqu_store(&mut self.core.text, X86Reg::RSP, 16, rhs_fp);
        for lane in 0..8 {
            let lhs_off = lane * 2;
            let rhs_off = 16 + lhs_off;
            let out_off = 32 + lhs_off;
            enc::load_s16_64(&mut self.core.text, X86Reg::RAX, X86Reg::RSP, lhs_off);
            enc::load_s16_64(&mut self.core.text, X86Reg::RDX, X86Reg::RSP, rhs_off);
            enc::imul_rr_64(&mut self.core.text, X86Reg::RAX, X86Reg::RDX);
            enc::add_ri_64(&mut self.core.text, X86Reg::RAX, 0x4000);
            enc::sar_imm_64(&mut self.core.text, X86Reg::RAX, 15);
            self.emit_saturate_to_width(X86Reg::RAX, IntCompareLaneWidth::H16, true);
            enc::store_16(&mut self.core.text, X86Reg::RSP, out_off, X86Reg::RAX);
        }
        movdqu_load(&mut self.core.text, dst_fp, X86Reg::RSP, 32);
        enc::add_rsp_imm8(&mut self.core.text, 48);
        Ok(())
    }

    fn lower_i32x4_dot_i16x8_s_scalar(
        &mut self,
        dst_fp: u8,
        lhs_fp: u8,
        rhs_fp: u8,
    ) -> Result<(), WasmError> {
        enc::sub_rsp_imm8(&mut self.core.text, 48);
        movdqu_store(&mut self.core.text, X86Reg::RSP, 0, lhs_fp);
        movdqu_store(&mut self.core.text, X86Reg::RSP, 16, rhs_fp);
        for lane in 0..4 {
            let base = lane * 4;
            let out_off = 32 + lane * 4;
            enc::load_s16_64(&mut self.core.text, X86Reg::RAX, X86Reg::RSP, base);
            enc::load_s16_64(&mut self.core.text, X86Reg::RDX, X86Reg::RSP, 16 + base);
            enc::imul_rr_64(&mut self.core.text, X86Reg::RAX, X86Reg::RDX);
            enc::load_s16_64(&mut self.core.text, X86Reg::RCX, X86Reg::RSP, base + 2);
            enc::load_s16_64(&mut self.core.text, X86Reg::RDX, X86Reg::RSP, 16 + base + 2);
            enc::imul_rr_64(&mut self.core.text, X86Reg::RCX, X86Reg::RDX);
            enc::add_rr_64(&mut self.core.text, X86Reg::RAX, X86Reg::RCX);
            enc::store_32(&mut self.core.text, X86Reg::RSP, out_off, X86Reg::RAX);
        }
        movdqu_load(&mut self.core.text, dst_fp, X86Reg::RSP, 32);
        enc::add_rsp_imm8(&mut self.core.text, 48);
        Ok(())
    }

    fn lower_i32x4_trunc_sat_f32x4_u_scalar(
        &mut self,
        dst_fp: u8,
        src_fp: u8,
    ) -> Result<(), WasmError> {
        enc::sub_rsp_imm8(&mut self.core.text, 48);
        movdqu_store(&mut self.core.text, X86Reg::RSP, 0, src_fp);
        enc::xor_rr_32(&mut self.core.text, X86Reg::RAX, X86Reg::RAX);
        enc::store_32(&mut self.core.text, X86Reg::RSP, 32, X86Reg::RAX);
        enc::mov_ri_32(&mut self.core.text, X86Reg::RAX, 4294967296.0f32.to_bits());
        enc::store_32(&mut self.core.text, X86Reg::RSP, 36, X86Reg::RAX);
        for lane in 0..4 {
            let src_off = lane * 4;
            let out_off = 16 + src_off;
            let negative = self.core.new_label();
            let clamp_max = self.core.new_label();
            let nan = self.core.new_label();
            let done = self.core.new_label();
            enc::movss_load(&mut self.core.text, 0, X86Reg::RSP, src_off);
            enc::ucomiss(&mut self.core.text, 0, 0);
            self.emit_jcc(Cc::P, nan);
            enc::movss_load(&mut self.core.text, 1, X86Reg::RSP, 32);
            enc::ucomiss(&mut self.core.text, 0, 1);
            self.emit_jcc(Cc::B, negative);
            enc::movss_load(&mut self.core.text, 1, X86Reg::RSP, 36);
            enc::ucomiss(&mut self.core.text, 0, 1);
            self.emit_jcc(Cc::AE, clamp_max);
            cvttss2si_r64_xmm(&mut self.core.text, X86Reg::RAX, 0);
            enc::store_32(&mut self.core.text, X86Reg::RSP, out_off, X86Reg::RAX);
            self.emit_jmp(done);
            self.core.bind_label(negative);
            enc::xor_rr_32(&mut self.core.text, X86Reg::RAX, X86Reg::RAX);
            enc::store_32(&mut self.core.text, X86Reg::RSP, out_off, X86Reg::RAX);
            self.emit_jmp(done);
            self.core.bind_label(clamp_max);
            enc::mov_ri_32(&mut self.core.text, X86Reg::RAX, u32::MAX);
            enc::store_32(&mut self.core.text, X86Reg::RSP, out_off, X86Reg::RAX);
            self.emit_jmp(done);
            self.core.bind_label(nan);
            enc::xor_rr_32(&mut self.core.text, X86Reg::RAX, X86Reg::RAX);
            enc::store_32(&mut self.core.text, X86Reg::RSP, out_off, X86Reg::RAX);
            self.core.bind_label(done);
        }
        movdqu_load(&mut self.core.text, dst_fp, X86Reg::RSP, 16);
        enc::add_rsp_imm8(&mut self.core.text, 48);
        Ok(())
    }

    fn lower_i32x4_trunc_sat_f64x2_scalar(
        &mut self,
        dst_fp: u8,
        src_fp: u8,
        unsigned: bool,
    ) -> Result<(), WasmError> {
        enc::sub_rsp_imm8(&mut self.core.text, 80);
        movdqu_store(&mut self.core.text, X86Reg::RSP, 0, src_fp);
        enc::mov_ri_64(&mut self.core.text, X86Reg::RAX, 0);
        enc::store_64(&mut self.core.text, X86Reg::RSP, 32, X86Reg::RAX);
        enc::store_64(&mut self.core.text, X86Reg::RSP, 40, X86Reg::RAX);
        enc::mov_ri_64(
            &mut self.core.text,
            X86Reg::RAX,
            if unsigned {
                0.0f64.to_bits()
            } else {
                (-2147483648.0f64).to_bits()
            },
        );
        enc::store_64(&mut self.core.text, X86Reg::RSP, 64, X86Reg::RAX);
        enc::mov_ri_64(
            &mut self.core.text,
            X86Reg::RAX,
            if unsigned {
                4294967296.0f64.to_bits()
            } else {
                2147483648.0f64.to_bits()
            },
        );
        enc::store_64(&mut self.core.text, X86Reg::RSP, 72, X86Reg::RAX);
        for lane in 0..2 {
            let src_off = lane * 8;
            let out_off = 32 + lane * 4;
            let below = self.core.new_label();
            let clamp_max = self.core.new_label();
            let nan = self.core.new_label();
            let done = self.core.new_label();
            enc::movsd_load(&mut self.core.text, 0, X86Reg::RSP, src_off);
            enc::ucomisd(&mut self.core.text, 0, 0);
            self.emit_jcc(Cc::P, nan);
            enc::movsd_load(&mut self.core.text, 1, X86Reg::RSP, 64);
            enc::ucomisd(&mut self.core.text, 0, 1);
            self.emit_jcc(Cc::B, below);
            enc::movsd_load(&mut self.core.text, 1, X86Reg::RSP, 72);
            enc::ucomisd(&mut self.core.text, 0, 1);
            self.emit_jcc(Cc::AE, clamp_max);
            cvttsd2si_r64_xmm(&mut self.core.text, X86Reg::RAX, 0);
            enc::store_32(&mut self.core.text, X86Reg::RSP, out_off, X86Reg::RAX);
            self.emit_jmp(done);
            self.core.bind_label(below);
            if unsigned {
                enc::xor_rr_32(&mut self.core.text, X86Reg::RAX, X86Reg::RAX);
            } else {
                enc::mov_ri_32(&mut self.core.text, X86Reg::RAX, 0x8000_0000u32);
            }
            enc::store_32(&mut self.core.text, X86Reg::RSP, out_off, X86Reg::RAX);
            self.emit_jmp(done);
            self.core.bind_label(clamp_max);
            enc::mov_ri_32(
                &mut self.core.text,
                X86Reg::RAX,
                if unsigned { u32::MAX } else { 0x7fff_ffff },
            );
            enc::store_32(&mut self.core.text, X86Reg::RSP, out_off, X86Reg::RAX);
            self.emit_jmp(done);
            self.core.bind_label(nan);
            enc::xor_rr_32(&mut self.core.text, X86Reg::RAX, X86Reg::RAX);
            enc::store_32(&mut self.core.text, X86Reg::RSP, out_off, X86Reg::RAX);
            self.core.bind_label(done);
        }
        movdqu_load(&mut self.core.text, dst_fp, X86Reg::RSP, 32);
        enc::add_rsp_imm8(&mut self.core.text, 80);
        Ok(())
    }

    fn lower_f64x2_convert_low_i32x4_u_scalar(
        &mut self,
        dst_fp: u8,
        src_fp: u8,
    ) -> Result<(), WasmError> {
        enc::sub_rsp_imm8(&mut self.core.text, 32);
        movdqu_store(&mut self.core.text, X86Reg::RSP, 0, src_fp);
        for lane in 0..2 {
            let src_off = lane * 4;
            let out_off = 16 + lane * 8;
            enc::load_32(&mut self.core.text, X86Reg::RAX, X86Reg::RSP, src_off);
            enc::cvtsi2sd_r64(&mut self.core.text, 0, X86Reg::RAX);
            enc::movsd_store(&mut self.core.text, X86Reg::RSP, out_off, 0);
        }
        movdqu_load(&mut self.core.text, dst_fp, X86Reg::RSP, 16);
        enc::add_rsp_imm8(&mut self.core.text, 32);
        Ok(())
    }

    fn lower_simd_load_extend_native(
        &mut self,
        dst_fp: u8,
        addr: MachineAddr,
        src_width: IntCompareLaneWidth,
        signed: bool,
    ) -> Result<(), WasmError> {
        let base = self.map_gp_reg(addr.base)?;
        match (src_width, signed) {
            (IntCompareLaneWidth::B8, true) => {
                pmovsxbw_rm(&mut self.core.text, dst_fp, base, addr.offset);
            }
            (IntCompareLaneWidth::B8, false) => {
                pmovzxbw_rm(&mut self.core.text, dst_fp, base, addr.offset);
            }
            (IntCompareLaneWidth::H16, true) => {
                pmovsxwd_rm(&mut self.core.text, dst_fp, base, addr.offset);
            }
            (IntCompareLaneWidth::H16, false) => {
                pmovzxwd_rm(&mut self.core.text, dst_fp, base, addr.offset);
            }
            (IntCompareLaneWidth::S32, true) => {
                pmovsxdq_rm(&mut self.core.text, dst_fp, base, addr.offset);
            }
            (IntCompareLaneWidth::S32, false) => {
                pmovzxdq_rm(&mut self.core.text, dst_fp, base, addr.offset);
            }
            (IntCompareLaneWidth::D64, _) => {
                return Err(WasmError::internal("invalid load-extend width"));
            }
        }
        Ok(())
    }

    fn lower_simd_replace_lane_native(
        &mut self,
        opcode: u32,
        lane: u8,
        dst: MachineReg,
        vector: MachineValue,
        scalar: MachineValue,
    ) -> Result<(), WasmError> {
        let dst_fp = self.map_fp_reg(dst)? as u8;
        let vector_fp = expect_v128_reg(self, vector, "x86_64 replace_lane needs a vector input")?;
        enc::sub_rsp_imm8(&mut self.core.text, 16);
        movdqu_store(&mut self.core.text, X86Reg::RSP, 0, vector_fp);
        let (lane_width, float_scalar) = match OpcodeFD::try_from(opcode)? {
            OpcodeFD::I8X16_REPLACE_LANE => (IntCompareLaneWidth::B8, false),
            OpcodeFD::I16X8_REPLACE_LANE => (IntCompareLaneWidth::H16, false),
            OpcodeFD::I32X4_REPLACE_LANE => (IntCompareLaneWidth::S32, false),
            OpcodeFD::I64X2_REPLACE_LANE => (IntCompareLaneWidth::D64, false),
            OpcodeFD::F32X4_REPLACE_LANE => (IntCompareLaneWidth::S32, true),
            OpcodeFD::F64X2_REPLACE_LANE => (IntCompareLaneWidth::D64, true),
            other => {
                enc::add_rsp_imm8(&mut self.core.text, 16);
                return Err(unsupported_simd_opcode("replace_lane", other as u32));
            }
        };
        let lane_off = lane as i32 * lane_width.byte_width() as i32;
        self.store_scalar_bits_to_stack(scalar, lane_off, lane_width, float_scalar)?;
        movdqu_load(&mut self.core.text, dst_fp, X86Reg::RSP, 0);
        enc::add_rsp_imm8(&mut self.core.text, 16);
        Ok(())
    }

    fn lower_simd_shuffle_native(
        &mut self,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
        lanes: [u8; 16],
    ) -> Result<(), WasmError> {
        let dst_fp = self.map_fp_reg(dst)? as u8;
        let lhs_fp = expect_v128_reg(self, lhs, "x86_64 shuffle lhs must be in an FP reg")?;
        let rhs_fp = expect_v128_reg(self, rhs, "x86_64 shuffle rhs must be in an FP reg")?;

        let mut lhs_mask = [0u8; 16];
        let mut rhs_mask = [0u8; 16];
        for (index, lane) in lanes.into_iter().enumerate() {
            if lane < 16 {
                lhs_mask[index] = lane;
                rhs_mask[index] = 0x80;
            } else if lane < 32 {
                lhs_mask[index] = 0x80;
                rhs_mask[index] = lane - 16;
            } else {
                lhs_mask[index] = 0x80;
                rhs_mask[index] = 0x80;
            }
        }

        let tmp = self.fp_scratch.scoped_alloc().detach();
        let tmp_fp = *tmp as u8;
        let gp = self.gp_scratch.scoped_alloc().detach();
        enc::sub_rsp_imm8(&mut self.core.text, 32);
        for chunk in 0..4 {
            let lhs_word = u32::from_le_bytes([
                lhs_mask[chunk * 4],
                lhs_mask[chunk * 4 + 1],
                lhs_mask[chunk * 4 + 2],
                lhs_mask[chunk * 4 + 3],
            ]);
            enc::mov_ri_32(&mut self.core.text, *gp, lhs_word);
            enc::store_32(&mut self.core.text, X86Reg::RSP, (chunk * 4) as i32, *gp);

            let rhs_word = u32::from_le_bytes([
                rhs_mask[chunk * 4],
                rhs_mask[chunk * 4 + 1],
                rhs_mask[chunk * 4 + 2],
                rhs_mask[chunk * 4 + 3],
            ]);
            enc::mov_ri_32(&mut self.core.text, *gp, rhs_word);
            enc::store_32(
                &mut self.core.text,
                X86Reg::RSP,
                16 + (chunk * 4) as i32,
                *gp,
            );
        }

        movdqa_rr(&mut self.core.text, tmp_fp, rhs_fp);
        if dst_fp != lhs_fp {
            movdqa_rr(&mut self.core.text, dst_fp, lhs_fp);
        }
        pshufb_rm(&mut self.core.text, dst_fp, X86Reg::RSP, 0);
        pshufb_rm(&mut self.core.text, tmp_fp, X86Reg::RSP, 16);
        por_rr(&mut self.core.text, dst_fp, tmp_fp);
        enc::add_rsp_imm8(&mut self.core.text, 32);
        Ok(())
    }

    fn lower_simd_load_lane_native(
        &mut self,
        opcode: u32,
        lane: u8,
        dst: MachineReg,
        addr: MachineAddr,
        vector: MachineValue,
    ) -> Result<(), WasmError> {
        let dst_fp = self.map_fp_reg(dst)? as u8;
        let vector_fp = expect_v128_reg(self, vector, "x86_64 load_lane needs a vector input")?;
        let base = self.map_gp_reg(addr.base)?;
        if dst_fp != vector_fp {
            movdqa_rr(&mut self.core.text, dst_fp, vector_fp);
        }
        match OpcodeFD::try_from(opcode)? {
            OpcodeFD::V128_LOAD8_LANE => {
                pinsrb_rm_imm8(&mut self.core.text, dst_fp, base, addr.offset, lane);
            }
            OpcodeFD::V128_LOAD16_LANE => {
                pinsrw_rm_imm8(&mut self.core.text, dst_fp, base, addr.offset, lane);
            }
            OpcodeFD::V128_LOAD32_LANE => {
                pinsrd_rm_imm8(&mut self.core.text, dst_fp, base, addr.offset, lane);
            }
            OpcodeFD::V128_LOAD64_LANE => {
                pinsrq_rm_imm8(&mut self.core.text, dst_fp, base, addr.offset, lane);
            }
            other => {
                return Err(unsupported_simd_opcode("load_lane", other as u32));
            }
        }
        Ok(())
    }

    fn lower_simd_store_lane_native(
        &mut self,
        opcode: u32,
        lane: u8,
        addr: MachineAddr,
        vector: MachineValue,
    ) -> Result<(), WasmError> {
        let vector_fp = expect_v128_reg(self, vector, "x86_64 store_lane needs a vector input")?;
        let base = self.map_gp_reg(addr.base)?;
        let lane_width = match OpcodeFD::try_from(opcode)? {
            OpcodeFD::V128_STORE8_LANE => IntCompareLaneWidth::B8,
            OpcodeFD::V128_STORE16_LANE => IntCompareLaneWidth::H16,
            OpcodeFD::V128_STORE32_LANE => IntCompareLaneWidth::S32,
            OpcodeFD::V128_STORE64_LANE => IntCompareLaneWidth::D64,
            other => {
                return Err(unsupported_simd_opcode("store_lane", other as u32));
            }
        };
        let lane_bytes = lane as u8 * lane_width.byte_width();
        let gp = self.gp_scratch.scoped_alloc().detach();
        let src_fp = if lane_bytes == 0 {
            vector_fp
        } else {
            let tmp = self.fp_scratch.scoped_alloc().detach();
            let tmp_fp = *tmp as u8;
            movdqa_rr(&mut self.core.text, tmp_fp, vector_fp);
            psrldq_rr_imm8(&mut self.core.text, tmp_fp, lane_bytes);
            tmp_fp
        };
        match lane_width {
            IntCompareLaneWidth::B8 => {
                enc::movd_r32_xmm(&mut self.core.text, *gp, src_fp);
                enc::store_8(&mut self.core.text, base, addr.offset, *gp);
            }
            IntCompareLaneWidth::H16 => {
                enc::movd_r32_xmm(&mut self.core.text, *gp, src_fp);
                enc::store_16(&mut self.core.text, base, addr.offset, *gp);
            }
            IntCompareLaneWidth::S32 => {
                enc::movd_r32_xmm(&mut self.core.text, *gp, src_fp);
                enc::store_32(&mut self.core.text, base, addr.offset, *gp);
            }
            IntCompareLaneWidth::D64 => {
                enc::movq_r64_xmm(&mut self.core.text, *gp, src_fp);
                enc::store_64(&mut self.core.text, base, addr.offset, *gp);
            }
        }
        Ok(())
    }

    fn store_scalar_bits_to_stack(
        &mut self,
        scalar: MachineValue,
        offset: i32,
        width: IntCompareLaneWidth,
        float_scalar: bool,
    ) -> Result<(), WasmError> {
        if float_scalar {
            match scalar {
                MachineValue::Imm64(value) => {
                    let scratch = self.gp_scratch.scoped_alloc().detach();
                    self.materialize_u64(*scratch, value);
                    self.emit_store_int_lane_from(X86Reg::RSP, offset, width, *scratch);
                }
                MachineValue::Reg(reg) => {
                    let fp = self.map_fp_reg(reg)? as u8;
                    match width {
                        IntCompareLaneWidth::S32 => {
                            enc::movss_store(&mut self.core.text, X86Reg::RSP, offset, fp)
                        }
                        IntCompareLaneWidth::D64 => {
                            enc::movsd_store(&mut self.core.text, X86Reg::RSP, offset, fp)
                        }
                        _ => return Err(WasmError::internal("invalid float replace lane width")),
                    }
                }
                MachineValue::ReservedReg(_) => {
                    return Err(WasmError::internal(
                        "x86_64 replace_lane scalar cannot be reserved",
                    ));
                }
            }
        } else {
            let gp = match scalar {
                MachineValue::Imm64(value) => {
                    let scratch = self.gp_scratch.scoped_alloc().detach();
                    self.materialize_u64(*scratch, value);
                    *scratch
                }
                MachineValue::Reg(reg) => self.map_gp_reg(reg)?,
                MachineValue::ReservedReg(_) => {
                    return Err(WasmError::internal(
                        "x86_64 replace_lane scalar cannot be reserved",
                    ));
                }
            };
            self.emit_store_int_lane_from(X86Reg::RSP, offset, width, gp);
        }
        Ok(())
    }

    fn emit_load_int_lane_to(
        &mut self,
        dst: X86Reg,
        base: X86Reg,
        offset: i32,
        width: IntCompareLaneWidth,
        signed: bool,
    ) {
        match (width, signed) {
            (IntCompareLaneWidth::B8, true) => {
                enc::load_s8_64(&mut self.core.text, dst, base, offset)
            }
            (IntCompareLaneWidth::B8, false) => {
                enc::load_u8(&mut self.core.text, dst, base, offset)
            }
            (IntCompareLaneWidth::H16, true) => {
                enc::load_s16_64(&mut self.core.text, dst, base, offset)
            }
            (IntCompareLaneWidth::H16, false) => {
                enc::load_u16(&mut self.core.text, dst, base, offset)
            }
            (IntCompareLaneWidth::S32, true) => {
                enc::load_s32_64(&mut self.core.text, dst, base, offset)
            }
            (IntCompareLaneWidth::S32, false) => {
                enc::load_32(&mut self.core.text, dst, base, offset)
            }
            (IntCompareLaneWidth::D64, _) => enc::load_64(&mut self.core.text, dst, base, offset),
        }
    }

    fn emit_store_int_lane_from(
        &mut self,
        base: X86Reg,
        offset: i32,
        width: IntCompareLaneWidth,
        src: X86Reg,
    ) {
        match width {
            IntCompareLaneWidth::B8 => enc::store_8(&mut self.core.text, base, offset, src),
            IntCompareLaneWidth::H16 => enc::store_16(&mut self.core.text, base, offset, src),
            IntCompareLaneWidth::S32 => enc::store_32(&mut self.core.text, base, offset, src),
            IntCompareLaneWidth::D64 => enc::store_64(&mut self.core.text, base, offset, src),
        }
    }

    fn emit_saturate_to_width(&mut self, reg: X86Reg, width: IntCompareLaneWidth, signed: bool) {
        let (min, max) = match (width, signed) {
            (IntCompareLaneWidth::B8, true) => (-128i64, 127i64),
            (IntCompareLaneWidth::B8, false) => (0, 255),
            (IntCompareLaneWidth::H16, true) => (-32768, 32767),
            (IntCompareLaneWidth::H16, false) => (0, 65535),
            (IntCompareLaneWidth::S32, true) => (i32::MIN as i64, i32::MAX as i64),
            (IntCompareLaneWidth::S32, false) => (0, u32::MAX as i64),
            (IntCompareLaneWidth::D64, _) => return,
        };
        let above_min = self.core.new_label();
        let done = self.core.new_label();
        enc::cmp_ri_64(&mut self.core.text, reg, min as i32);
        self.emit_jcc(Cc::GE, above_min);
        enc::mov_ri_64(&mut self.core.text, reg, min as u64);
        self.emit_jmp(done);
        self.core.bind_label(above_min);
        if max <= i32::MAX as i64 {
            let within = self.core.new_label();
            enc::cmp_ri_64(&mut self.core.text, reg, max as i32);
            self.emit_jcc(Cc::LE, within);
            enc::mov_ri_64(&mut self.core.text, reg, max as u64);
            self.core.bind_label(within);
        } else {
            let within = self.core.new_label();
            enc::mov_ri_64(&mut self.core.text, X86Reg::RDX, max as u64);
            enc::cmp_rr_64(&mut self.core.text, reg, X86Reg::RDX);
            self.emit_jcc(Cc::BE, within);
            enc::mov_ri_64(&mut self.core.text, reg, max as u64);
            self.core.bind_label(within);
        }
        self.core.bind_label(done);
    }

    fn lower_simd_float_unary_masked(
        &mut self,
        dst: MachineReg,
        src_fp: u8,
        mask: [u8; 16],
        is_and: bool,
    ) -> Result<(), WasmError> {
        let dst_fp = self.map_fp_reg(dst)? as u8;
        if dst_fp != src_fp {
            movdqa_rr(&mut self.core.text, dst_fp, src_fp);
        }
        let mask_fp = self.fp_scratch.scoped_alloc().detach();
        self.materialize_v128_constant_into_fp(*mask_fp as u8, mask);
        if is_and {
            pand_rr(&mut self.core.text, dst_fp, *mask_fp as u8);
        } else {
            pxor_rr(&mut self.core.text, dst_fp, *mask_fp as u8);
        }
        Ok(())
    }

    fn lower_simd_float_unary_vector(
        &mut self,
        dst: MachineReg,
        src_fp: u8,
        op: SimdFloatUnaryVectorOp,
    ) -> Result<(), WasmError> {
        let dst_fp = self.map_fp_reg(dst)? as u8;
        match op {
            SimdFloatUnaryVectorOp::SqrtF32 => sqrtps_rr(&mut self.core.text, dst_fp, src_fp),
            SimdFloatUnaryVectorOp::SqrtF64 => sqrtpd_rr(&mut self.core.text, dst_fp, src_fp),
            SimdFloatUnaryVectorOp::RoundF32(mode) => {
                roundps_rr_imm8(&mut self.core.text, dst_fp, src_fp, mode)
            }
            SimdFloatUnaryVectorOp::RoundF64(mode) => {
                roundpd_rr_imm8(&mut self.core.text, dst_fp, src_fp, mode)
            }
        }
        Ok(())
    }

    fn lower_simd_float_binary_vector(
        &mut self,
        dst_fp: u8,
        lhs_fp: u8,
        rhs_fp: u8,
        op: SimdFloatBinaryVectorOp,
    ) -> Result<(), WasmError> {
        if dst_fp != lhs_fp {
            movdqa_rr(&mut self.core.text, dst_fp, lhs_fp);
        }
        match op {
            SimdFloatBinaryVectorOp::AddF32 => addps_rr(&mut self.core.text, dst_fp, rhs_fp),
            SimdFloatBinaryVectorOp::SubF32 => subps_rr(&mut self.core.text, dst_fp, rhs_fp),
            SimdFloatBinaryVectorOp::MulF32 => mulps_rr(&mut self.core.text, dst_fp, rhs_fp),
            SimdFloatBinaryVectorOp::DivF32 => divps_rr(&mut self.core.text, dst_fp, rhs_fp),
            SimdFloatBinaryVectorOp::DivF64 => divpd_rr(&mut self.core.text, dst_fp, rhs_fp),
        }
        Ok(())
    }

    fn lower_simd_float_convert_vector(
        &mut self,
        dst: MachineReg,
        src_fp: u8,
        op: SimdFloatConvertOp,
    ) -> Result<(), WasmError> {
        let dst_fp = self.map_fp_reg(dst)? as u8;
        match op {
            SimdFloatConvertOp::I32x4ToF32x4S => cvtdq2ps_rr(&mut self.core.text, dst_fp, src_fp),
            SimdFloatConvertOp::I32x4LowToF64x2S => {
                cvtdq2pd_rr(&mut self.core.text, dst_fp, src_fp)
            }
            SimdFloatConvertOp::F64x2ToF32x4Zero => {
                cvtpd2ps_rr(&mut self.core.text, dst_fp, src_fp)
            }
            SimdFloatConvertOp::F32x4LowToF64x2 => cvtps2pd_rr(&mut self.core.text, dst_fp, src_fp),
        }
        Ok(())
    }

    fn lower_i32x4_trunc_sat_f32x4_s_scalar(
        &mut self,
        dst_fp: u8,
        src_fp: u8,
    ) -> Result<(), WasmError> {
        enc::sub_rsp_imm8(&mut self.core.text, 48);
        movdqu_store(&mut self.core.text, X86Reg::RSP, 0, src_fp);
        enc::mov_ri_32(
            &mut self.core.text,
            X86Reg::RAX,
            (-2147483648.0f32).to_bits(),
        );
        enc::store_32(&mut self.core.text, X86Reg::RSP, 32, X86Reg::RAX);
        enc::mov_ri_32(&mut self.core.text, X86Reg::RAX, 2147483648.0f32.to_bits());
        enc::store_32(&mut self.core.text, X86Reg::RSP, 36, X86Reg::RAX);

        for lane in 0..4 {
            let src_off = lane * 4;
            let out_off = 16 + src_off;
            enc::movss_load(&mut self.core.text, 0, X86Reg::RSP, src_off);
            enc::ucomiss(&mut self.core.text, 0, 0);
            let nan = self.core.new_label();
            let clamp_min = self.core.new_label();
            let clamp_max = self.core.new_label();
            let done = self.core.new_label();
            self.emit_jcc(Cc::P, nan);

            enc::movss_load(&mut self.core.text, 1, X86Reg::RSP, 32);
            enc::ucomiss(&mut self.core.text, 0, 1);
            self.emit_jcc(Cc::B, clamp_min);

            enc::movss_load(&mut self.core.text, 1, X86Reg::RSP, 36);
            enc::ucomiss(&mut self.core.text, 0, 1);
            self.emit_jcc(Cc::AE, clamp_max);

            cvttss2si_r32_xmm(&mut self.core.text, X86Reg::RAX, 0);
            enc::store_32(&mut self.core.text, X86Reg::RSP, out_off, X86Reg::RAX);
            self.emit_jmp(done);

            self.core.bind_label(nan);
            enc::xor_rr_32(&mut self.core.text, X86Reg::RAX, X86Reg::RAX);
            enc::store_32(&mut self.core.text, X86Reg::RSP, out_off, X86Reg::RAX);
            self.emit_jmp(done);

            self.core.bind_label(clamp_min);
            enc::mov_ri_32(&mut self.core.text, X86Reg::RAX, 0x8000_0000);
            enc::store_32(&mut self.core.text, X86Reg::RSP, out_off, X86Reg::RAX);
            self.emit_jmp(done);

            self.core.bind_label(clamp_max);
            enc::mov_ri_32(&mut self.core.text, X86Reg::RAX, 0x7fff_ffff);
            enc::store_32(&mut self.core.text, X86Reg::RSP, out_off, X86Reg::RAX);

            self.core.bind_label(done);
        }

        movdqu_load(&mut self.core.text, dst_fp, X86Reg::RSP, 16);
        enc::add_rsp_imm8(&mut self.core.text, 48);
        Ok(())
    }

    fn lower_simd_float_minmax_scalar(
        &mut self,
        dst_fp: u8,
        lhs_fp: u8,
        rhs_fp: u8,
        width: FloatCompareLaneWidth,
        is_max: bool,
        propagate_nan: bool,
    ) -> Result<(), WasmError> {
        enc::sub_rsp_imm8(&mut self.core.text, 48);
        movdqu_store(&mut self.core.text, X86Reg::RSP, 0, lhs_fp);
        movdqu_store(&mut self.core.text, X86Reg::RSP, 16, rhs_fp);
        for lane in 0..width.lane_count() {
            let lhs_off = lane as i32 * width.byte_width() as i32;
            let rhs_off = 16 + lhs_off;
            let out_off = 32 + lhs_off;
            let unordered = self.core.new_label();
            let choose_rhs = self.core.new_label();
            let equal = self.core.new_label();
            let done = self.core.new_label();
            match width {
                FloatCompareLaneWidth::F32 => {
                    enc::movss_load(&mut self.core.text, 0, X86Reg::RSP, lhs_off);
                    enc::movss_load(&mut self.core.text, 1, X86Reg::RSP, rhs_off);
                    enc::ucomiss(&mut self.core.text, 0, 1);
                }
                FloatCompareLaneWidth::F64 => {
                    enc::movsd_load(&mut self.core.text, 0, X86Reg::RSP, lhs_off);
                    enc::movsd_load(&mut self.core.text, 1, X86Reg::RSP, rhs_off);
                    enc::ucomisd(&mut self.core.text, 0, 1);
                }
            }
            self.emit_jcc(Cc::P, unordered);
            self.emit_jcc(Cc::E, equal);
            self.emit_jcc(if is_max { Cc::B } else { Cc::A }, choose_rhs);

            self.load_and_store_float_raw_lane(width, lhs_off, out_off);
            self.emit_jmp(done);

            self.core.bind_label(choose_rhs);
            self.load_and_store_float_raw_lane(width, rhs_off, out_off);
            self.emit_jmp(done);

            self.core.bind_label(equal);
            self.lower_store_float_equal_lane(width, lhs_off, rhs_off, out_off, is_max);
            self.emit_jmp(done);

            self.core.bind_label(unordered);
            if propagate_nan {
                match width {
                    FloatCompareLaneWidth::F32 => {
                        enc::addss(&mut self.core.text, 0, 1);
                        enc::movss_store(&mut self.core.text, X86Reg::RSP, out_off, 0);
                    }
                    FloatCompareLaneWidth::F64 => {
                        enc::addsd(&mut self.core.text, 0, 1);
                        enc::movsd_store(&mut self.core.text, X86Reg::RSP, out_off, 0);
                    }
                }
            } else {
                self.load_and_store_float_raw_lane(width, lhs_off, out_off);
            }

            self.core.bind_label(done);
        }
        movdqu_load(&mut self.core.text, dst_fp, X86Reg::RSP, 32);
        enc::add_rsp_imm8(&mut self.core.text, 48);
        Ok(())
    }

    fn load_and_store_float_raw_lane(
        &mut self,
        width: FloatCompareLaneWidth,
        src_off: i32,
        out_off: i32,
    ) {
        match width {
            FloatCompareLaneWidth::F32 => {
                enc::load_32(&mut self.core.text, X86Reg::RAX, X86Reg::RSP, src_off);
                enc::store_32(&mut self.core.text, X86Reg::RSP, out_off, X86Reg::RAX);
            }
            FloatCompareLaneWidth::F64 => {
                enc::load_64(&mut self.core.text, X86Reg::RAX, X86Reg::RSP, src_off);
                enc::store_64(&mut self.core.text, X86Reg::RSP, out_off, X86Reg::RAX);
            }
        }
    }

    fn lower_store_float_equal_lane(
        &mut self,
        width: FloatCompareLaneWidth,
        lhs_off: i32,
        rhs_off: i32,
        out_off: i32,
        is_max: bool,
    ) {
        match width {
            FloatCompareLaneWidth::F32 => {
                enc::load_32(&mut self.core.text, X86Reg::RAX, X86Reg::RSP, lhs_off);
                enc::load_32(&mut self.core.text, X86Reg::RDX, X86Reg::RSP, rhs_off);
                enc::mov_rr_32(&mut self.core.text, X86Reg::RCX, X86Reg::RAX);
                enc::and_ri_32(&mut self.core.text, X86Reg::RCX, 0x7fff_ffff);
                let lhs_non_zero = self.core.new_label();
                self.emit_jcc(Cc::NE, lhs_non_zero);
                enc::mov_rr_32(&mut self.core.text, X86Reg::RCX, X86Reg::RDX);
                enc::and_ri_32(&mut self.core.text, X86Reg::RCX, 0x7fff_ffff);
                self.emit_jcc(Cc::NE, lhs_non_zero);
                enc::and_ri_32(&mut self.core.text, X86Reg::RAX, 0x8000_0000u32 as i32);
                enc::and_ri_32(&mut self.core.text, X86Reg::RDX, 0x8000_0000u32 as i32);
                if is_max {
                    enc::and_rr_32(&mut self.core.text, X86Reg::RAX, X86Reg::RDX);
                } else {
                    enc::or_rr_32(&mut self.core.text, X86Reg::RAX, X86Reg::RDX);
                }
                enc::store_32(&mut self.core.text, X86Reg::RSP, out_off, X86Reg::RAX);
                let done = self.core.new_label();
                self.emit_jmp(done);
                self.core.bind_label(lhs_non_zero);
                enc::store_32(&mut self.core.text, X86Reg::RSP, out_off, X86Reg::RAX);
                self.core.bind_label(done);
            }
            FloatCompareLaneWidth::F64 => {
                enc::load_64(&mut self.core.text, X86Reg::RAX, X86Reg::RSP, lhs_off);
                enc::mov_rr_64(&mut self.core.text, X86Reg::RCX, X86Reg::RAX);
                enc::shl_imm_64(&mut self.core.text, X86Reg::RCX, 1);
                enc::test_rr_64(&mut self.core.text, X86Reg::RCX, X86Reg::RCX);
                let lhs_non_zero = self.core.new_label();
                self.emit_jcc(Cc::NE, lhs_non_zero);

                enc::load_64(&mut self.core.text, X86Reg::RDX, X86Reg::RSP, rhs_off);
                enc::mov_rr_64(&mut self.core.text, X86Reg::RCX, X86Reg::RDX);
                enc::shl_imm_64(&mut self.core.text, X86Reg::RCX, 1);
                enc::test_rr_64(&mut self.core.text, X86Reg::RCX, X86Reg::RCX);
                self.emit_jcc(Cc::NE, lhs_non_zero);

                enc::mov_ri_64(&mut self.core.text, X86Reg::RCX, 0x8000_0000_0000_0000);
                enc::and_rr_64(&mut self.core.text, X86Reg::RAX, X86Reg::RCX);
                enc::and_rr_64(&mut self.core.text, X86Reg::RDX, X86Reg::RCX);
                if is_max {
                    enc::and_rr_64(&mut self.core.text, X86Reg::RAX, X86Reg::RDX);
                } else {
                    enc::or_rr_64(&mut self.core.text, X86Reg::RAX, X86Reg::RDX);
                }
                enc::store_64(&mut self.core.text, X86Reg::RSP, out_off, X86Reg::RAX);
                let done = self.core.new_label();
                self.emit_jmp(done);
                self.core.bind_label(lhs_non_zero);
                enc::load_64(&mut self.core.text, X86Reg::RAX, X86Reg::RSP, lhs_off);
                enc::store_64(&mut self.core.text, X86Reg::RSP, out_off, X86Reg::RAX);
                self.core.bind_label(done);
            }
        }
    }

    fn lower_scalar_shift_lane(
        &mut self,
        offset: i32,
        lane_width: ShiftLaneWidth,
        shift_kind: ShiftKind,
    ) -> Result<(), WasmError> {
        let lane = X86Reg::RDX;
        match (lane_width, shift_kind) {
            (ShiftLaneWidth::B8, ShiftKind::Left) => {
                enc::load_u8(&mut self.core.text, lane, X86Reg::RSP, offset);
                enc::shl_cl_32(&mut self.core.text, lane);
                enc::store_8(&mut self.core.text, X86Reg::RSP, offset, lane);
            }
            (ShiftLaneWidth::B8, ShiftKind::RightUnsigned) => {
                enc::load_u8(&mut self.core.text, lane, X86Reg::RSP, offset);
                enc::shr_cl_32(&mut self.core.text, lane);
                enc::store_8(&mut self.core.text, X86Reg::RSP, offset, lane);
            }
            (ShiftLaneWidth::B8, ShiftKind::RightSigned) => {
                enc::load_s8_64(&mut self.core.text, lane, X86Reg::RSP, offset);
                enc::sar_cl_64(&mut self.core.text, lane);
                enc::store_8(&mut self.core.text, X86Reg::RSP, offset, lane);
            }
            (ShiftLaneWidth::H16, ShiftKind::Left) => {
                enc::load_u16(&mut self.core.text, lane, X86Reg::RSP, offset);
                enc::shl_cl_32(&mut self.core.text, lane);
                enc::store_16(&mut self.core.text, X86Reg::RSP, offset, lane);
            }
            (ShiftLaneWidth::H16, ShiftKind::RightUnsigned) => {
                enc::load_u16(&mut self.core.text, lane, X86Reg::RSP, offset);
                enc::shr_cl_32(&mut self.core.text, lane);
                enc::store_16(&mut self.core.text, X86Reg::RSP, offset, lane);
            }
            (ShiftLaneWidth::H16, ShiftKind::RightSigned) => {
                enc::load_s16_64(&mut self.core.text, lane, X86Reg::RSP, offset);
                enc::sar_cl_64(&mut self.core.text, lane);
                enc::store_16(&mut self.core.text, X86Reg::RSP, offset, lane);
            }
            (ShiftLaneWidth::S32, ShiftKind::Left) => {
                enc::load_32(&mut self.core.text, lane, X86Reg::RSP, offset);
                enc::shl_cl_32(&mut self.core.text, lane);
                enc::store_32(&mut self.core.text, X86Reg::RSP, offset, lane);
            }
            (ShiftLaneWidth::S32, ShiftKind::RightUnsigned) => {
                enc::load_32(&mut self.core.text, lane, X86Reg::RSP, offset);
                enc::shr_cl_32(&mut self.core.text, lane);
                enc::store_32(&mut self.core.text, X86Reg::RSP, offset, lane);
            }
            (ShiftLaneWidth::S32, ShiftKind::RightSigned) => {
                enc::load_32(&mut self.core.text, lane, X86Reg::RSP, offset);
                enc::sar_cl_32(&mut self.core.text, lane);
                enc::store_32(&mut self.core.text, X86Reg::RSP, offset, lane);
            }
            (ShiftLaneWidth::D64, ShiftKind::Left) => {
                enc::load_64(&mut self.core.text, lane, X86Reg::RSP, offset);
                enc::shl_cl_64(&mut self.core.text, lane);
                enc::store_64(&mut self.core.text, X86Reg::RSP, offset, lane);
            }
            (ShiftLaneWidth::D64, ShiftKind::RightUnsigned) => {
                enc::load_64(&mut self.core.text, lane, X86Reg::RSP, offset);
                enc::shr_cl_64(&mut self.core.text, lane);
                enc::store_64(&mut self.core.text, X86Reg::RSP, offset, lane);
            }
            (ShiftLaneWidth::D64, ShiftKind::RightSigned) => {
                enc::load_64(&mut self.core.text, lane, X86Reg::RSP, offset);
                enc::sar_cl_64(&mut self.core.text, lane);
                enc::store_64(&mut self.core.text, X86Reg::RSP, offset, lane);
            }
        }
        Ok(())
    }

    fn lower_i8x16_swizzle_scalar(
        &mut self,
        dst_fp: u8,
        lhs_fp: u8,
        rhs_fp: u8,
    ) -> Result<(), WasmError> {
        enc::sub_rsp_imm8(&mut self.core.text, 48);
        movdqu_store(&mut self.core.text, X86Reg::RSP, 0, lhs_fp);
        movdqu_store(&mut self.core.text, X86Reg::RSP, 16, rhs_fp);
        for lane in 0..16 {
            enc::load_u8(&mut self.core.text, X86Reg::RAX, X86Reg::RSP, 16 + lane);
            enc::cmp_ri_32(&mut self.core.text, X86Reg::RAX, 16);
            let zero_label = self.core.new_label();
            let done_label = self.core.new_label();
            self.emit_jcc(Cc::AE, zero_label);
            enc::mov_rr_64(&mut self.core.text, X86Reg::RDX, X86Reg::RSP);
            enc::add_rr_64(&mut self.core.text, X86Reg::RDX, X86Reg::RAX);
            enc::load_u8(&mut self.core.text, X86Reg::RAX, X86Reg::RDX, 0);
            enc::store_8(&mut self.core.text, X86Reg::RSP, 32 + lane, X86Reg::RAX);
            self.emit_jmp(done_label);
            self.core.bind_label(zero_label);
            enc::xor_rr_32(&mut self.core.text, X86Reg::RAX, X86Reg::RAX);
            enc::store_8(&mut self.core.text, X86Reg::RSP, 32 + lane, X86Reg::RAX);
            self.core.bind_label(done_label);
        }
        movdqu_load(&mut self.core.text, dst_fp, X86Reg::RSP, 32);
        enc::add_rsp_imm8(&mut self.core.text, 48);
        Ok(())
    }

    fn lower_i64x2_mul_scalar(
        &mut self,
        dst_fp: u8,
        lhs_fp: u8,
        rhs_fp: u8,
    ) -> Result<(), WasmError> {
        enc::sub_rsp_imm8(&mut self.core.text, 32);
        movdqu_store(&mut self.core.text, X86Reg::RSP, 0, lhs_fp);
        movdqu_store(&mut self.core.text, X86Reg::RSP, 16, rhs_fp);
        enc::load_64(&mut self.core.text, X86Reg::RAX, X86Reg::RSP, 0);
        enc::load_64(&mut self.core.text, X86Reg::RDX, X86Reg::RSP, 16);
        enc::imul_rr_64(&mut self.core.text, X86Reg::RAX, X86Reg::RDX);
        enc::store_64(&mut self.core.text, X86Reg::RSP, 0, X86Reg::RAX);
        enc::load_64(&mut self.core.text, X86Reg::RAX, X86Reg::RSP, 8);
        enc::load_64(&mut self.core.text, X86Reg::RDX, X86Reg::RSP, 24);
        enc::imul_rr_64(&mut self.core.text, X86Reg::RAX, X86Reg::RDX);
        enc::store_64(&mut self.core.text, X86Reg::RSP, 8, X86Reg::RAX);
        movdqu_load(&mut self.core.text, dst_fp, X86Reg::RSP, 0);
        enc::add_rsp_imm8(&mut self.core.text, 32);
        Ok(())
    }

    fn lower_simd_int_compare_scalar(
        &mut self,
        dst_fp: u8,
        lhs_fp: u8,
        rhs_fp: u8,
        lane_width: IntCompareLaneWidth,
        op: SimdCompareOp,
        unsigned: bool,
    ) -> Result<(), WasmError> {
        enc::sub_rsp_imm8(&mut self.core.text, 48);
        movdqu_store(&mut self.core.text, X86Reg::RSP, 0, lhs_fp);
        movdqu_store(&mut self.core.text, X86Reg::RSP, 16, rhs_fp);
        for lane in 0..lane_width.lane_count() {
            let lhs_off = lane as i32 * lane_width.byte_width() as i32;
            let rhs_off = 16 + lhs_off;
            let out_off = 32 + lhs_off;
            match lane_width {
                IntCompareLaneWidth::B8 => {
                    if unsigned {
                        enc::load_u8(&mut self.core.text, X86Reg::RAX, X86Reg::RSP, lhs_off);
                        enc::load_u8(&mut self.core.text, X86Reg::RDX, X86Reg::RSP, rhs_off);
                        enc::cmp_rr_32(&mut self.core.text, X86Reg::RAX, X86Reg::RDX);
                    } else {
                        enc::load_s8_64(&mut self.core.text, X86Reg::RAX, X86Reg::RSP, lhs_off);
                        enc::load_s8_64(&mut self.core.text, X86Reg::RDX, X86Reg::RSP, rhs_off);
                        enc::cmp_rr_64(&mut self.core.text, X86Reg::RAX, X86Reg::RDX);
                    }
                    self.emit_setcc_mask32(self.int_compare_cc(op, unsigned)?);
                    enc::store_8(&mut self.core.text, X86Reg::RSP, out_off, X86Reg::RDX);
                }
                IntCompareLaneWidth::H16 => {
                    if unsigned {
                        enc::load_u16(&mut self.core.text, X86Reg::RAX, X86Reg::RSP, lhs_off);
                        enc::load_u16(&mut self.core.text, X86Reg::RDX, X86Reg::RSP, rhs_off);
                        enc::cmp_rr_32(&mut self.core.text, X86Reg::RAX, X86Reg::RDX);
                    } else {
                        enc::load_s16_64(&mut self.core.text, X86Reg::RAX, X86Reg::RSP, lhs_off);
                        enc::load_s16_64(&mut self.core.text, X86Reg::RDX, X86Reg::RSP, rhs_off);
                        enc::cmp_rr_64(&mut self.core.text, X86Reg::RAX, X86Reg::RDX);
                    }
                    self.emit_setcc_mask32(self.int_compare_cc(op, unsigned)?);
                    enc::store_16(&mut self.core.text, X86Reg::RSP, out_off, X86Reg::RDX);
                }
                IntCompareLaneWidth::S32 => {
                    enc::load_32(&mut self.core.text, X86Reg::RAX, X86Reg::RSP, lhs_off);
                    enc::load_32(&mut self.core.text, X86Reg::RDX, X86Reg::RSP, rhs_off);
                    enc::cmp_rr_32(&mut self.core.text, X86Reg::RAX, X86Reg::RDX);
                    self.emit_setcc_mask32(self.int_compare_cc(op, unsigned)?);
                    enc::store_32(&mut self.core.text, X86Reg::RSP, out_off, X86Reg::RDX);
                }
                IntCompareLaneWidth::D64 => {
                    debug_assert!(!unsigned);
                    enc::load_64(&mut self.core.text, X86Reg::RAX, X86Reg::RSP, lhs_off);
                    enc::load_64(&mut self.core.text, X86Reg::RDX, X86Reg::RSP, rhs_off);
                    enc::cmp_rr_64(&mut self.core.text, X86Reg::RAX, X86Reg::RDX);
                    self.emit_setcc_mask64(self.int_compare_cc(op, false)?);
                    enc::store_64(&mut self.core.text, X86Reg::RSP, out_off, X86Reg::RDX);
                }
            }
        }
        movdqu_load(&mut self.core.text, dst_fp, X86Reg::RSP, 32);
        enc::add_rsp_imm8(&mut self.core.text, 48);
        Ok(())
    }

    fn lower_simd_float_compare_scalar(
        &mut self,
        dst_fp: u8,
        lhs_fp: u8,
        rhs_fp: u8,
        lane_width: FloatCompareLaneWidth,
        op: SimdCompareOp,
    ) -> Result<(), WasmError> {
        enc::sub_rsp_imm8(&mut self.core.text, 48);
        movdqu_store(&mut self.core.text, X86Reg::RSP, 0, lhs_fp);
        movdqu_store(&mut self.core.text, X86Reg::RSP, 16, rhs_fp);
        for lane in 0..lane_width.lane_count() {
            let lhs_off = lane as i32 * lane_width.byte_width() as i32;
            let rhs_off = 16 + lhs_off;
            let out_off = 32 + lhs_off;
            match lane_width {
                FloatCompareLaneWidth::F32 => {
                    enc::movss_load(&mut self.core.text, 0, X86Reg::RSP, lhs_off);
                    enc::movss_load(&mut self.core.text, 1, X86Reg::RSP, rhs_off);
                    enc::ucomiss(&mut self.core.text, 0, 1);
                    self.emit_float_compare_mask32(op);
                    enc::store_32(&mut self.core.text, X86Reg::RSP, out_off, X86Reg::RDX);
                }
                FloatCompareLaneWidth::F64 => {
                    enc::movsd_load(&mut self.core.text, 0, X86Reg::RSP, lhs_off);
                    enc::movsd_load(&mut self.core.text, 1, X86Reg::RSP, rhs_off);
                    enc::ucomisd(&mut self.core.text, 0, 1);
                    self.emit_float_compare_mask64(op);
                    enc::store_64(&mut self.core.text, X86Reg::RSP, out_off, X86Reg::RDX);
                }
            }
        }
        movdqu_load(&mut self.core.text, dst_fp, X86Reg::RSP, 32);
        enc::add_rsp_imm8(&mut self.core.text, 48);
        Ok(())
    }

    fn int_compare_cc(&self, op: SimdCompareOp, unsigned: bool) -> Result<Cc, WasmError> {
        Ok(match (op, unsigned) {
            (SimdCompareOp::Eq, _) => Cc::E,
            (SimdCompareOp::Ne, _) => Cc::NE,
            (SimdCompareOp::Lt, false) => Cc::L,
            (SimdCompareOp::Lt, true) => Cc::B,
            (SimdCompareOp::Gt, false) => Cc::G,
            (SimdCompareOp::Gt, true) => Cc::A,
            (SimdCompareOp::Le, false) => Cc::LE,
            (SimdCompareOp::Le, true) => Cc::BE,
            (SimdCompareOp::Ge, false) => Cc::GE,
            (SimdCompareOp::Ge, true) => Cc::AE,
        })
    }

    fn emit_setcc_mask32(&mut self, cc: Cc) {
        enc::setcc(&mut self.core.text, cc, X86Reg::RAX);
        enc::movzx_r32_r8(&mut self.core.text, X86Reg::RAX, X86Reg::RAX);
        enc::xor_rr_32(&mut self.core.text, X86Reg::RDX, X86Reg::RDX);
        enc::sub_rr_32(&mut self.core.text, X86Reg::RDX, X86Reg::RAX);
    }

    fn emit_setcc_mask64(&mut self, cc: Cc) {
        enc::setcc(&mut self.core.text, cc, X86Reg::RAX);
        enc::movzx_r32_r8(&mut self.core.text, X86Reg::RAX, X86Reg::RAX);
        enc::xor_rr_64(&mut self.core.text, X86Reg::RDX, X86Reg::RDX);
        enc::sub_rr_64(&mut self.core.text, X86Reg::RDX, X86Reg::RAX);
    }

    fn emit_float_compare_mask32(&mut self, op: SimdCompareOp) {
        match op {
            SimdCompareOp::Eq => {
                enc::setcc(&mut self.core.text, Cc::E, X86Reg::RAX);
                enc::setcc(&mut self.core.text, Cc::NP, X86Reg::RDX);
                enc::movzx_r32_r8(&mut self.core.text, X86Reg::RAX, X86Reg::RAX);
                enc::movzx_r32_r8(&mut self.core.text, X86Reg::RDX, X86Reg::RDX);
                enc::and_rr_32(&mut self.core.text, X86Reg::RAX, X86Reg::RDX);
            }
            SimdCompareOp::Ne => {
                enc::setcc(&mut self.core.text, Cc::NE, X86Reg::RAX);
                enc::setcc(&mut self.core.text, Cc::P, X86Reg::RDX);
                enc::movzx_r32_r8(&mut self.core.text, X86Reg::RAX, X86Reg::RAX);
                enc::movzx_r32_r8(&mut self.core.text, X86Reg::RDX, X86Reg::RDX);
                enc::or_rr_32(&mut self.core.text, X86Reg::RAX, X86Reg::RDX);
            }
            SimdCompareOp::Lt => {
                enc::setcc(&mut self.core.text, Cc::B, X86Reg::RAX);
                enc::setcc(&mut self.core.text, Cc::NP, X86Reg::RDX);
                enc::movzx_r32_r8(&mut self.core.text, X86Reg::RAX, X86Reg::RAX);
                enc::movzx_r32_r8(&mut self.core.text, X86Reg::RDX, X86Reg::RDX);
                enc::and_rr_32(&mut self.core.text, X86Reg::RAX, X86Reg::RDX);
            }
            SimdCompareOp::Gt => {
                enc::setcc(&mut self.core.text, Cc::A, X86Reg::RAX);
                enc::movzx_r32_r8(&mut self.core.text, X86Reg::RAX, X86Reg::RAX);
            }
            SimdCompareOp::Le => {
                enc::setcc(&mut self.core.text, Cc::BE, X86Reg::RAX);
                enc::setcc(&mut self.core.text, Cc::NP, X86Reg::RDX);
                enc::movzx_r32_r8(&mut self.core.text, X86Reg::RAX, X86Reg::RAX);
                enc::movzx_r32_r8(&mut self.core.text, X86Reg::RDX, X86Reg::RDX);
                enc::and_rr_32(&mut self.core.text, X86Reg::RAX, X86Reg::RDX);
            }
            SimdCompareOp::Ge => {
                enc::setcc(&mut self.core.text, Cc::AE, X86Reg::RAX);
                enc::movzx_r32_r8(&mut self.core.text, X86Reg::RAX, X86Reg::RAX);
            }
        }
        enc::xor_rr_32(&mut self.core.text, X86Reg::RDX, X86Reg::RDX);
        enc::sub_rr_32(&mut self.core.text, X86Reg::RDX, X86Reg::RAX);
    }

    fn emit_float_compare_mask64(&mut self, op: SimdCompareOp) {
        self.emit_float_compare_mask32(op);
        enc::movsxd_r64_r32(&mut self.core.text, X86Reg::RDX, X86Reg::RDX);
    }

    fn lower_simd_load_native(
        &mut self,
        opcode: u32,
        dst: MachineReg,
        addr: MachineAddr,
    ) -> Result<(), WasmError> {
        match OpcodeFD::try_from(opcode)? {
            OpcodeFD::V128_LOAD => {
                let dst_fp = self.map_fp_reg(dst)? as u8;
                let base = self.map_gp_reg(addr.base)?;
                movdqu_load(&mut self.core.text, dst_fp, base, addr.offset);
                Ok(())
            }
            OpcodeFD::V128_LOAD8X8_S => {
                let dst_fp = self.map_fp_reg(dst)? as u8;
                self.lower_simd_load_extend_native(dst_fp, addr, IntCompareLaneWidth::B8, true)
            }
            OpcodeFD::V128_LOAD8X8_U => {
                let dst_fp = self.map_fp_reg(dst)? as u8;
                self.lower_simd_load_extend_native(dst_fp, addr, IntCompareLaneWidth::B8, false)
            }
            OpcodeFD::V128_LOAD16X4_S => {
                let dst_fp = self.map_fp_reg(dst)? as u8;
                self.lower_simd_load_extend_native(dst_fp, addr, IntCompareLaneWidth::H16, true)
            }
            OpcodeFD::V128_LOAD16X4_U => {
                let dst_fp = self.map_fp_reg(dst)? as u8;
                self.lower_simd_load_extend_native(dst_fp, addr, IntCompareLaneWidth::H16, false)
            }
            OpcodeFD::V128_LOAD32X2_S => {
                let dst_fp = self.map_fp_reg(dst)? as u8;
                self.lower_simd_load_extend_native(dst_fp, addr, IntCompareLaneWidth::S32, true)
            }
            OpcodeFD::V128_LOAD32X2_U => {
                let dst_fp = self.map_fp_reg(dst)? as u8;
                self.lower_simd_load_extend_native(dst_fp, addr, IntCompareLaneWidth::S32, false)
            }
            OpcodeFD::V128_LOAD8_SPLAT => {
                let dst_fp = self.map_fp_reg(dst)? as u8;
                let base = self.map_gp_reg(addr.base)?;
                let scratch0 = self.gp_scratch.scoped_alloc().detach();
                let scratch1 = self.gp_scratch.scoped_alloc().detach();
                enc::load_u8(&mut self.core.text, *scratch0, base, addr.offset);
                enc::and_ri_32(&mut self.core.text, *scratch0, 0xFF);
                enc::mov_ri_32(&mut self.core.text, *scratch1, 0x0101_0101);
                enc::imul_rr_32(&mut self.core.text, *scratch0, *scratch1);
                enc::movd_xmm_r32(&mut self.core.text, dst_fp, *scratch0);
                pshufd_rr_imm8(&mut self.core.text, dst_fp, dst_fp, 0);
                Ok(())
            }
            OpcodeFD::V128_LOAD16_SPLAT => {
                let dst_fp = self.map_fp_reg(dst)? as u8;
                let base = self.map_gp_reg(addr.base)?;
                let scratch0 = self.gp_scratch.scoped_alloc().detach();
                let scratch1 = self.gp_scratch.scoped_alloc().detach();
                enc::load_u16(&mut self.core.text, *scratch0, base, addr.offset);
                enc::and_ri_32(&mut self.core.text, *scratch0, 0xFFFF);
                enc::mov_ri_32(&mut self.core.text, *scratch1, 0x0001_0001);
                enc::imul_rr_32(&mut self.core.text, *scratch0, *scratch1);
                enc::movd_xmm_r32(&mut self.core.text, dst_fp, *scratch0);
                pshufd_rr_imm8(&mut self.core.text, dst_fp, dst_fp, 0);
                Ok(())
            }
            OpcodeFD::V128_LOAD32_SPLAT => {
                let dst_fp = self.map_fp_reg(dst)? as u8;
                let base = self.map_gp_reg(addr.base)?;
                let scratch0 = self.gp_scratch.scoped_alloc().detach();
                enc::load_32(&mut self.core.text, *scratch0, base, addr.offset);
                enc::movd_xmm_r32(&mut self.core.text, dst_fp, *scratch0);
                pshufd_rr_imm8(&mut self.core.text, dst_fp, dst_fp, 0);
                Ok(())
            }
            OpcodeFD::V128_LOAD64_SPLAT => {
                let dst_fp = self.map_fp_reg(dst)? as u8;
                let base = self.map_gp_reg(addr.base)?;
                let scratch0 = self.gp_scratch.scoped_alloc().detach();
                enc::load_64(&mut self.core.text, *scratch0, base, addr.offset);
                enc::movq_xmm_r64(&mut self.core.text, dst_fp, *scratch0);
                punpcklqdq_rr(&mut self.core.text, dst_fp, dst_fp);
                Ok(())
            }
            OpcodeFD::V128_LOAD32_ZERO => {
                let dst_fp = self.map_fp_reg(dst)? as u8;
                let base = self.map_gp_reg(addr.base)?;
                let scratch0 = self.gp_scratch.scoped_alloc().detach();
                enc::load_32(&mut self.core.text, *scratch0, base, addr.offset);
                enc::movd_xmm_r32(&mut self.core.text, dst_fp, *scratch0);
                Ok(())
            }
            OpcodeFD::V128_LOAD64_ZERO => {
                let dst_fp = self.map_fp_reg(dst)? as u8;
                let base = self.map_gp_reg(addr.base)?;
                let scratch0 = self.gp_scratch.scoped_alloc().detach();
                enc::load_64(&mut self.core.text, *scratch0, base, addr.offset);
                enc::movq_xmm_r64(&mut self.core.text, dst_fp, *scratch0);
                Ok(())
            }
            other => Err(unsupported_simd_opcode("load", other as u32)),
        }
    }

    fn lower_simd_store_native(
        &mut self,
        opcode: u32,
        addr: MachineAddr,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        match OpcodeFD::try_from(opcode)? {
            OpcodeFD::V128_STORE => {
                let src_fp =
                    expect_v128_reg(self, src, "x86_64 SIMD store source must be in an FP reg")?;
                let base = self.map_gp_reg(addr.base)?;
                movdqu_store(&mut self.core.text, base, addr.offset, src_fp);
                Ok(())
            }
            other => Err(unsupported_simd_opcode("store", other as u32)),
        }
    }
}

#[derive(Clone, Copy)]
enum SimdAllTrueWidth {
    B8,
    H16,
    S32,
    D64,
}

#[derive(Clone, Copy)]
enum SimdBitmaskWidth {
    B8,
    H16,
    S32,
    D64,
}

#[derive(Clone, Copy)]
enum IntegerSplatWidth {
    B8,
    H16,
    S32,
    D64,
}

#[derive(Clone, Copy)]
enum FloatSplatWidth {
    F32,
    F64,
}

#[derive(Clone, Copy)]
enum ShiftLaneWidth {
    B8,
    H16,
    S32,
    D64,
}

impl ShiftLaneWidth {
    #[inline]
    const fn mask(self) -> i32 {
        match self {
            Self::B8 => 7,
            Self::H16 => 15,
            Self::S32 => 31,
            Self::D64 => 63,
        }
    }

    #[inline]
    const fn lane_count(self) -> usize {
        match self {
            Self::B8 => 16,
            Self::H16 => 8,
            Self::S32 => 4,
            Self::D64 => 2,
        }
    }

    #[inline]
    const fn byte_width(self) -> u8 {
        match self {
            Self::B8 => 1,
            Self::H16 => 2,
            Self::S32 => 4,
            Self::D64 => 8,
        }
    }
}

#[derive(Clone, Copy)]
enum ShiftKind {
    Left,
    RightSigned,
    RightUnsigned,
}

#[derive(Clone, Copy)]
enum IntCompareLaneWidth {
    B8,
    H16,
    S32,
    D64,
}

impl IntCompareLaneWidth {
    #[inline]
    const fn lane_count(self) -> usize {
        match self {
            Self::B8 => 16,
            Self::H16 => 8,
            Self::S32 => 4,
            Self::D64 => 2,
        }
    }

    #[inline]
    const fn byte_width(self) -> u8 {
        match self {
            Self::B8 => 1,
            Self::H16 => 2,
            Self::S32 => 4,
            Self::D64 => 8,
        }
    }
}

#[derive(Clone, Copy)]
enum FloatCompareLaneWidth {
    F32,
    F64,
}

impl FloatCompareLaneWidth {
    #[inline]
    const fn lane_count(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F64 => 2,
        }
    }

    #[inline]
    const fn byte_width(self) -> u8 {
        match self {
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

#[derive(Clone, Copy)]
enum SimdCompareOp {
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
}

#[derive(Clone, Copy)]
enum SimdIntUnaryVectorOp {
    AbsB8,
    AbsH16,
    AbsS32,
    NegB8,
    NegH16,
    NegS32,
    NegD64,
}

#[derive(Clone, Copy)]
enum SimdFloatUnaryVectorOp {
    SqrtF32,
    SqrtF64,
    RoundF32(u8),
    RoundF64(u8),
}

#[derive(Clone, Copy)]
enum SimdFloatBinaryVectorOp {
    AddF32,
    SubF32,
    MulF32,
    DivF32,
    DivF64,
}

#[derive(Clone, Copy)]
enum SimdFloatConvertOp {
    I32x4ToF32x4S,
    I32x4LowToF64x2S,
    F64x2ToF32x4Zero,
    F32x4LowToF64x2,
}
