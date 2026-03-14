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

fn ldst_unsigned_offset(size: u32, opc: u32, rt: Arm64Reg, rn: Arm64Reg, imm12: u32) -> u32 {
    debug_assert!(imm12 < 0x1000);
    (size << 30) | (0b111_0_01 << 24) | (opc << 22) | (imm12 << 10) | (rn.idx() << 5) | rt.idx()
}

fn ldst_register_offset(base: u32, rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    base | (rm.idx() << 16) | (rn.idx() << 5) | rt.idx()
}

fn load_store_pair(base: u32, rt: Arm64Reg, rt2: Arm64Reg, rn: Arm64Reg, imm7: i32) -> u32 {
    let imm7_bits = (imm7 as u32) & 0x7f;
    base | (imm7_bits << 15) | (rt2.idx() << 10) | (rn.idx() << 5) | rt.idx()
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

pub fn mul_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    madd(0, rd, rn, rm, Arm64Reg::Xzr)
}

pub fn mul_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    madd(1, rd, rn, rm, Arm64Reg::Xzr)
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

pub fn lslv_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    shift_var(0, 0b00, rd, rn, rm)
}

pub fn lsrv_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    shift_var(0, 0b01, rd, rn, rm)
}

pub fn asrv_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    shift_var(0, 0b10, rd, rn, rm)
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

pub fn add_imm_64(rd: Arm64Reg, rn: Arm64Reg, imm12: u32) -> u32 {
    add_sub_imm(1, 0, 0, rd, rn, imm12)
}

pub fn sub_imm_64(rd: Arm64Reg, rn: Arm64Reg, imm12: u32) -> u32 {
    add_sub_imm(1, 1, 0, rd, rn, imm12)
}

pub fn cmp_reg_32(rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    add_sub_shifted_reg(0, 1, 1, Arm64Reg::Xzr, rn, rm)
}

pub fn cmp_reg_64(rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    add_sub_shifted_reg(1, 1, 1, Arm64Reg::Xzr, rn, rm)
}

pub fn cmp_imm_64(rn: Arm64Reg, imm12: u32) -> u32 {
    add_sub_imm(1, 1, 1, Arm64Reg::Xzr, rn, imm12)
}

pub fn cset_32(rd: Arm64Reg, cond: Cond) -> u32 {
    cond_select(0, 0b01, rd, Arm64Reg::Xzr, Arm64Reg::Xzr, cond.invert())
}

pub fn cset_64(rd: Arm64Reg, cond: Cond) -> u32 {
    cond_select(1, 0b01, rd, Arm64Reg::Xzr, Arm64Reg::Xzr, cond.invert())
}

pub fn csel_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg, cond: Cond) -> u32 {
    cond_select(1, 0b00, rd, rn, rm, cond)
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
    (0b0_10_11010_110 << 21) | (0b00000 << 16) | (0b00010_0 << 10) | (rn.idx() << 5) | rd.idx()
}

pub fn clz_64(rd: Arm64Reg, rn: Arm64Reg) -> u32 {
    (0b1_10_11010_110 << 21) | (0b00000 << 16) | (0b00010_0 << 10) | (rn.idx() << 5) | rd.idx()
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

pub fn sxtw(rd: Arm64Reg, rn: Arm64Reg) -> u32 {
    sbfm(1, 1, rd, rn, 0, 31)
}

pub fn ldr_64(rt: Arm64Reg, rn: Arm64Reg, imm12: u32) -> u32 {
    ldst_unsigned_offset(0b11, 0b01, rt, rn, imm12)
}

pub fn ldr_reg_64(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0xf860_6800, rt, rn, rm)
}

pub fn str_64(rt: Arm64Reg, rn: Arm64Reg, imm12: u32) -> u32 {
    ldst_unsigned_offset(0b11, 0b00, rt, rn, imm12)
}

pub fn ldr_reg_32(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0xb860_6800, rt, rn, rm)
}

pub fn ldrb_reg(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0x3860_6800, rt, rn, rm)
}

pub fn ldrh_reg(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0x7860_6800, rt, rn, rm)
}

pub fn ldrsw_reg(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0xb8a0_6800, rt, rn, rm)
}

pub fn ldrsb_reg_64(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0x38a0_6800, rt, rn, rm)
}

pub fn ldrsh_reg_64(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0x78a0_6800, rt, rn, rm)
}

pub fn str_reg_64(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0xf820_6800, rt, rn, rm)
}

pub fn str_reg_32(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0xb820_6800, rt, rn, rm)
}

pub fn strb_reg(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0x3820_6800, rt, rn, rm)
}

pub fn strh_reg(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0x7820_6800, rt, rn, rm)
}

pub fn stp_64(rt: Arm64Reg, rt2: Arm64Reg, rn: Arm64Reg, imm7: i32) -> u32 {
    load_store_pair(0xa900_0000, rt, rt2, rn, imm7)
}

pub fn ldp_64(rt: Arm64Reg, rt2: Arm64Reg, rn: Arm64Reg, imm7: i32) -> u32 {
    load_store_pair(0xa940_0000, rt, rt2, rn, imm7)
}

pub fn b(imm26: i32) -> u32 {
    let imm26_bits = (imm26 as u32) & 0x03FF_FFFF;
    (0b000101 << 26) | imm26_bits
}

pub fn ldr_lit_64(rt: Arm64Reg, imm19: i32) -> u32 {
    let imm19_bits = (imm19 as u32) & 0x0007_FFFF;
    (0b01011000 << 24) | (imm19_bits << 5) | rt.idx()
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

// --- Integer data-processing (2 source) ---

fn data_proc_2src(sf: u32, opcode: u32, rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    (sf << 31)
        | (0b0_11010110 << 21)
        | (rm.idx() << 16)
        | (opcode << 10)
        | (rn.idx() << 5)
        | rd.idx()
}

pub fn udiv_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    data_proc_2src(0, 0b000010, rd, rn, rm)
}

pub fn udiv_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    data_proc_2src(1, 0b000010, rd, rn, rm)
}

pub fn sdiv_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    data_proc_2src(0, 0b000011, rd, rn, rm)
}

pub fn sdiv_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    data_proc_2src(1, 0b000011, rd, rn, rm)
}

pub fn rorv_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    shift_var(0, 0b11, rd, rn, rm)
}

pub fn rorv_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    shift_var(1, 0b11, rd, rn, rm)
}

/// MSUB Rd, Rn, Rm, Ra: Rd = Ra - Rn * Rm
fn msub(sf: u32, rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg, ra: Arm64Reg) -> u32 {
    (sf << 31)
        | (0b00_11011_000 << 21)
        | (rm.idx() << 16)
        | (1 << 15)
        | (ra.idx() << 10)
        | (rn.idx() << 5)
        | rd.idx()
}

pub fn msub_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg, ra: Arm64Reg) -> u32 {
    msub(0, rd, rn, rm, ra)
}

pub fn msub_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg, ra: Arm64Reg) -> u32 {
    msub(1, rd, rn, rm, ra)
}

// --- Bit manipulation ---

fn data_proc_1src(sf: u32, opcode2: u32, opcode: u32, rd: Arm64Reg, rn: Arm64Reg) -> u32 {
    (sf << 31)
        | (0b1_0_11010110 << 21)
        | (opcode2 << 16)
        | (opcode << 10)
        | (rn.idx() << 5)
        | rd.idx()
}

pub fn rbit_32(rd: Arm64Reg, rn: Arm64Reg) -> u32 {
    data_proc_1src(0, 0b00000, 0b000000, rd, rn)
}

pub fn rbit_64(rd: Arm64Reg, rn: Arm64Reg) -> u32 {
    data_proc_1src(1, 0b00000, 0b000000, rd, rn)
}

pub fn cmp_imm_32(rn: Arm64Reg, imm12: u32) -> u32 {
    add_sub_imm(0, 1, 1, Arm64Reg::Xzr, rn, imm12)
}

pub fn neg_reg_32(rd: Arm64Reg, rm: Arm64Reg) -> u32 {
    sub_reg_32(rd, Arm64Reg::Xzr, rm)
}

pub fn neg_reg_64(rd: Arm64Reg, rm: Arm64Reg) -> u32 {
    sub_reg_64(rd, Arm64Reg::Xzr, rm)
}

pub fn csel_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg, cond: Cond) -> u32 {
    cond_select(0, 0b00, rd, rn, rm, cond)
}

pub fn mov_reg_32(rd: Arm64Reg, rm: Arm64Reg) -> u32 {
    orr_reg_32(rd, Arm64Reg::Xzr, rm)
}

pub fn movz_32(rd: Arm64Reg, imm16: u16, shift: u32) -> u32 {
    move_wide(0, 0b10, rd, imm16, shift)
}

pub fn movk_32(rd: Arm64Reg, imm16: u16, shift: u32) -> u32 {
    move_wide(0, 0b11, rd, imm16, shift)
}

// --- Floating-point instructions ---
// FP register index is just 0-31, same encoding slot as GP but in FP instruction formats.

/// Scalar floating-point data-processing (2 source)
fn fp_data_proc_2src(ftype: u32, opcode: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    (0b0001_1110 << 24) | (ftype << 22) | (1 << 21) | (rm << 16) | (opcode << 12) | (0b10 << 10) | (rn << 5) | rd
}

/// Scalar floating-point data-processing (1 source)
fn fp_data_proc_1src(ftype: u32, opcode: u32, rd: u32, rn: u32) -> u32 {
    (0b0001_1110 << 24) | (ftype << 22) | (1 << 21) | (opcode << 15) | (0b10000 << 10) | (rn << 5) | rd
}

/// Floating-point comparison
fn fp_compare(ftype: u32, rn: u32, rm: u32, opc: u32) -> u32 {
    (0b0001_1110 << 24) | (ftype << 22) | (1 << 21) | (rm << 16) | (0b00_1000 << 10) | (rn << 5) | opc
}

/// Floating-point conditional select
fn fp_csel(ftype: u32, rd: u32, rn: u32, rm: u32, cond: Cond) -> u32 {
    (0b0001_1110 << 24)
        | (ftype << 22)
        | (1 << 21)
        | (rm << 16)
        | ((cond as u32) << 12)
        | (0b11 << 10)
        | (rn << 5)
        | rd
}

/// Conversion between FP and integer
fn fp_int_conv(sf: u32, ftype: u32, rmode: u32, opcode: u32, rd: u32, rn: u32) -> u32 {
    (sf << 31)
        | (0b0011110 << 24)
        | (ftype << 22)
        | (1 << 21)
        | (rmode << 19)
        | (opcode << 16)
        | (rn << 5)
        | rd
}

/// FP load (unsigned offset): LDR St/Dt, [Xn, #imm]
pub fn fp_ldr_unsigned(size: u32, rt: u32, rn: Arm64Reg, imm12: u32) -> u32 {
    debug_assert!(imm12 < 0x1000);
    (size << 30) | (0b111_1_01 << 24) | (0b01 << 22) | (imm12 << 10) | (rn.idx() << 5) | rt
}

/// FP store (unsigned offset): STR St/Dt, [Xn, #imm]
pub fn fp_str_unsigned(size: u32, rt: u32, rn: Arm64Reg, imm12: u32) -> u32 {
    debug_assert!(imm12 < 0x1000);
    (size << 30) | (0b111_1_01 << 24) | (0b00 << 22) | (imm12 << 10) | (rn.idx() << 5) | rt
}

/// FP load register offset: LDR St/Dt, [Xn, Xm]
pub fn fp_ldr_reg(base: u32, rt: u32, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    base | (rm.idx() << 16) | (rn.idx() << 5) | rt
}

/// FP store register offset: STR St/Dt, [Xn, Xm]
pub fn fp_str_reg(base: u32, rt: u32, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    base | (rm.idx() << 16) | (rn.idx() << 5) | rt
}

// F32 load/store register offset bases
pub const FP_LDR_S_REG_BASE: u32 = 0xbc60_6800;
pub const FP_STR_S_REG_BASE: u32 = 0xbc20_6800;
pub const FP_LDR_D_REG_BASE: u32 = 0xfc60_6800;
pub const FP_STR_D_REG_BASE: u32 = 0xfc20_6800;

// FMOV between GP and FP registers
/// FMOV Wd, Sn (FP to GP, 32-bit)
pub fn fmov_gp_from_s(rd: Arm64Reg, rn: u32) -> u32 {
    fp_int_conv(0, 0b00, 0b00, 0b110, rd.idx(), rn)
}
/// FMOV Sd, Wn (GP to FP, 32-bit)
pub fn fmov_s_from_gp(rd: u32, rn: Arm64Reg) -> u32 {
    fp_int_conv(0, 0b00, 0b00, 0b111, rd, rn.idx())
}
/// FMOV Xd, Dn (FP to GP, 64-bit)
pub fn fmov_gp_from_d(rd: Arm64Reg, rn: u32) -> u32 {
    fp_int_conv(1, 0b01, 0b00, 0b110, rd.idx(), rn)
}
/// FMOV Dd, Xn (GP to FP, 64-bit)
pub fn fmov_d_from_gp(rd: u32, rn: Arm64Reg) -> u32 {
    fp_int_conv(1, 0b01, 0b00, 0b111, rd, rn.idx())
}
/// FMOV Sd, Sn
pub fn fmov_s(rd: u32, rn: u32) -> u32 {
    fp_data_proc_1src(0b00, 0b000000, rd, rn)
}
/// FMOV Dd, Dn
pub fn fmov_d(rd: u32, rn: u32) -> u32 {
    fp_data_proc_1src(0b01, 0b000000, rd, rn)
}

// F32 arithmetic
pub fn fadd_s(rd: u32, rn: u32, rm: u32) -> u32 { fp_data_proc_2src(0b00, 0b0010, rd, rn, rm) }
pub fn fsub_s(rd: u32, rn: u32, rm: u32) -> u32 { fp_data_proc_2src(0b00, 0b0011, rd, rn, rm) }
pub fn fmul_s(rd: u32, rn: u32, rm: u32) -> u32 { fp_data_proc_2src(0b00, 0b0000, rd, rn, rm) }
pub fn fdiv_s(rd: u32, rn: u32, rm: u32) -> u32 { fp_data_proc_2src(0b00, 0b0001, rd, rn, rm) }
pub fn fmin_s(rd: u32, rn: u32, rm: u32) -> u32 { fp_data_proc_2src(0b00, 0b0101, rd, rn, rm) }
pub fn fmax_s(rd: u32, rn: u32, rm: u32) -> u32 { fp_data_proc_2src(0b00, 0b0100, rd, rn, rm) }

// F64 arithmetic
pub fn fadd_d(rd: u32, rn: u32, rm: u32) -> u32 { fp_data_proc_2src(0b01, 0b0010, rd, rn, rm) }
pub fn fsub_d(rd: u32, rn: u32, rm: u32) -> u32 { fp_data_proc_2src(0b01, 0b0011, rd, rn, rm) }
pub fn fmul_d(rd: u32, rn: u32, rm: u32) -> u32 { fp_data_proc_2src(0b01, 0b0000, rd, rn, rm) }
pub fn fdiv_d(rd: u32, rn: u32, rm: u32) -> u32 { fp_data_proc_2src(0b01, 0b0001, rd, rn, rm) }
pub fn fmin_d(rd: u32, rn: u32, rm: u32) -> u32 { fp_data_proc_2src(0b01, 0b0101, rd, rn, rm) }
pub fn fmax_d(rd: u32, rn: u32, rm: u32) -> u32 { fp_data_proc_2src(0b01, 0b0100, rd, rn, rm) }

// F32 unary
pub fn fabs_s(rd: u32, rn: u32) -> u32 { fp_data_proc_1src(0b00, 0b000001, rd, rn) }
pub fn fneg_s(rd: u32, rn: u32) -> u32 { fp_data_proc_1src(0b00, 0b000010, rd, rn) }
pub fn fsqrt_s(rd: u32, rn: u32) -> u32 { fp_data_proc_1src(0b00, 0b000011, rd, rn) }
pub fn frintn_s(rd: u32, rn: u32) -> u32 { fp_data_proc_1src(0b00, 0b001000, rd, rn) }
pub fn frintp_s(rd: u32, rn: u32) -> u32 { fp_data_proc_1src(0b00, 0b001001, rd, rn) }
pub fn frintm_s(rd: u32, rn: u32) -> u32 { fp_data_proc_1src(0b00, 0b001010, rd, rn) }
pub fn frintz_s(rd: u32, rn: u32) -> u32 { fp_data_proc_1src(0b00, 0b001011, rd, rn) }

// F64 unary
pub fn fabs_d(rd: u32, rn: u32) -> u32 { fp_data_proc_1src(0b01, 0b000001, rd, rn) }
pub fn fneg_d(rd: u32, rn: u32) -> u32 { fp_data_proc_1src(0b01, 0b000010, rd, rn) }
pub fn fsqrt_d(rd: u32, rn: u32) -> u32 { fp_data_proc_1src(0b01, 0b000011, rd, rn) }
pub fn frintn_d(rd: u32, rn: u32) -> u32 { fp_data_proc_1src(0b01, 0b001000, rd, rn) }
pub fn frintp_d(rd: u32, rn: u32) -> u32 { fp_data_proc_1src(0b01, 0b001001, rd, rn) }
pub fn frintm_d(rd: u32, rn: u32) -> u32 { fp_data_proc_1src(0b01, 0b001010, rd, rn) }
pub fn frintz_d(rd: u32, rn: u32) -> u32 { fp_data_proc_1src(0b01, 0b001011, rd, rn) }

// FP compare
pub fn fcmp_s(rn: u32, rm: u32) -> u32 { fp_compare(0b00, rn, rm, 0b00000) }
pub fn fcmp_d(rn: u32, rm: u32) -> u32 { fp_compare(0b01, rn, rm, 0b00000) }

// FP conditional select
pub fn fcsel_s(rd: u32, rn: u32, rm: u32, cond: Cond) -> u32 { fp_csel(0b00, rd, rn, rm, cond) }
pub fn fcsel_d(rd: u32, rn: u32, rm: u32, cond: Cond) -> u32 { fp_csel(0b01, rd, rn, rm, cond) }

// FP conversion between sizes
/// FCVT Dd, Sn (F32 -> F64)
pub fn fcvt_d_from_s(rd: u32, rn: u32) -> u32 { fp_data_proc_1src(0b00, 0b000101, rd, rn) }
/// FCVT Sd, Dn (F64 -> F32)
pub fn fcvt_s_from_d(rd: u32, rn: u32) -> u32 { fp_data_proc_1src(0b01, 0b000100, rd, rn) }

// FP to integer conversions (truncation toward zero)
/// FCVTZS Wd, Sn
pub fn fcvtzs_32_s(rd: Arm64Reg, rn: u32) -> u32 { fp_int_conv(0, 0b00, 0b11, 0b000, rd.idx(), rn) }
/// FCVTZS Xd, Sn
pub fn fcvtzs_64_s(rd: Arm64Reg, rn: u32) -> u32 { fp_int_conv(1, 0b00, 0b11, 0b000, rd.idx(), rn) }
/// FCVTZS Wd, Dn
pub fn fcvtzs_32_d(rd: Arm64Reg, rn: u32) -> u32 { fp_int_conv(0, 0b01, 0b11, 0b000, rd.idx(), rn) }
/// FCVTZS Xd, Dn
pub fn fcvtzs_64_d(rd: Arm64Reg, rn: u32) -> u32 { fp_int_conv(1, 0b01, 0b11, 0b000, rd.idx(), rn) }

/// FCVTZU Wd, Sn
pub fn fcvtzu_32_s(rd: Arm64Reg, rn: u32) -> u32 { fp_int_conv(0, 0b00, 0b11, 0b001, rd.idx(), rn) }
/// FCVTZU Xd, Sn
pub fn fcvtzu_64_s(rd: Arm64Reg, rn: u32) -> u32 { fp_int_conv(1, 0b00, 0b11, 0b001, rd.idx(), rn) }
/// FCVTZU Wd, Dn
pub fn fcvtzu_32_d(rd: Arm64Reg, rn: u32) -> u32 { fp_int_conv(0, 0b01, 0b11, 0b001, rd.idx(), rn) }
/// FCVTZU Xd, Dn
pub fn fcvtzu_64_d(rd: Arm64Reg, rn: u32) -> u32 { fp_int_conv(1, 0b01, 0b11, 0b001, rd.idx(), rn) }

// Integer to FP conversions
/// SCVTF Sd, Wn
pub fn scvtf_s_32(rd: u32, rn: Arm64Reg) -> u32 { fp_int_conv(0, 0b00, 0b00, 0b010, rd, rn.idx()) }
/// SCVTF Sd, Xn
pub fn scvtf_s_64(rd: u32, rn: Arm64Reg) -> u32 { fp_int_conv(1, 0b00, 0b00, 0b010, rd, rn.idx()) }
/// SCVTF Dd, Wn
pub fn scvtf_d_32(rd: u32, rn: Arm64Reg) -> u32 { fp_int_conv(0, 0b01, 0b00, 0b010, rd, rn.idx()) }
/// SCVTF Dd, Xn
pub fn scvtf_d_64(rd: u32, rn: Arm64Reg) -> u32 { fp_int_conv(1, 0b01, 0b00, 0b010, rd, rn.idx()) }

/// UCVTF Sd, Wn
pub fn ucvtf_s_32(rd: u32, rn: Arm64Reg) -> u32 { fp_int_conv(0, 0b00, 0b00, 0b011, rd, rn.idx()) }
/// UCVTF Sd, Xn
pub fn ucvtf_s_64(rd: u32, rn: Arm64Reg) -> u32 { fp_int_conv(1, 0b00, 0b00, 0b011, rd, rn.idx()) }
/// UCVTF Dd, Wn
pub fn ucvtf_d_32(rd: u32, rn: Arm64Reg) -> u32 { fp_int_conv(0, 0b01, 0b00, 0b011, rd, rn.idx()) }
/// UCVTF Dd, Xn
pub fn ucvtf_d_64(rd: u32, rn: Arm64Reg) -> u32 { fp_int_conv(1, 0b01, 0b00, 0b011, rd, rn.idx()) }

// NEON instructions for popcnt
/// FMOV Dd, Xn (same as fmov_d_from_gp but with explicit naming for NEON use)
/// CNT Vd.8B, Vn.8B  (count bits per byte)
pub fn cnt_8b(rd: u32, rn: u32) -> u32 {
    (0b0_0_001110_00_10000_00101_10 << 10) | (rn << 5) | rd
}
/// ADDV Bd, Vn.8B  (horizontal add of bytes)
pub fn addv_8b(rd: u32, rn: u32) -> u32 {
    (0b0_0_001110_00_11000_11011_10 << 10) | (rn << 5) | rd
}
/// UMOV Wd, Vn.B[0]  (extract byte 0 to GP)
pub fn umov_b0(rd: Arm64Reg, rn: u32) -> u32 {
    (0b0_0_001110000 << 21) | (0b00001 << 16) | (0b001_1_1_1 << 10) | (rn << 5) | rd.idx()
}
