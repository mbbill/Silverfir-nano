use super::{
    abi,
    reg::{Arm64FpReg, Arm64Reg},
};

/// ARM64 shift type for shifted-register operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum ShiftType {
    Lsl = 0,
    Lsr = 1,
    Asr = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum Cond {
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
    pub(crate) const fn invert(self) -> Cond {
        match self {
            Cond::Eq => Cond::Ne,
            Cond::Ne => Cond::Eq,
            Cond::Hs => Cond::Lo,
            Cond::Lo => Cond::Hs,
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
        }
    }
}

fn add_sub_shifted_reg(sf: u32, op: u32, s: u32, rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    (sf << 31)
        | (op << 30)
        | (s << 29)
        | (0b01011 << 24)
        | (rm.index() << 16)
        | (rn.index() << 5)
        | rd.index()
}

fn add_sub_shifted_reg_ext(
    sf: u32,
    op: u32,
    s: u32,
    rd: Arm64Reg,
    rn: Arm64Reg,
    rm: Arm64Reg,
    shift: ShiftType,
    amount: u32,
) -> u32 {
    debug_assert!(amount < if sf == 0 { 32 } else { 64 });
    (sf << 31)
        | (op << 30)
        | (s << 29)
        | (0b01011 << 24)
        | ((shift as u32) << 22)
        | (rm.index() << 16)
        | (amount << 10)
        | (rn.index() << 5)
        | rd.index()
}

fn logical_shifted_reg(sf: u32, opc: u32, n: u32, rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    (sf << 31)
        | (opc << 29)
        | (0b01010 << 24)
        | (n << 21)
        | (rm.index() << 16)
        | (rn.index() << 5)
        | rd.index()
}

fn logical_shifted_reg_ext(
    sf: u32,
    opc: u32,
    n: u32,
    rd: Arm64Reg,
    rn: Arm64Reg,
    rm: Arm64Reg,
    shift: ShiftType,
    amount: u32,
) -> u32 {
    debug_assert!(amount < if sf == 0 { 32 } else { 64 });
    (sf << 31)
        | (opc << 29)
        | (0b01010 << 24)
        | ((shift as u32) << 22)
        | (n << 21)
        | (rm.index() << 16)
        | (amount << 10)
        | (rn.index() << 5)
        | rd.index()
}

fn logical_imm(sf: u32, opc: u32, n: u32, rd: Arm64Reg, rn: Arm64Reg, immr: u32, imms: u32) -> u32 {
    (sf << 31)
        | (opc << 29)
        | (0b100100 << 23)
        | (n << 22)
        | (immr << 16)
        | (imms << 10)
        | (rn.index() << 5)
        | rd.index()
}

fn madd(sf: u32, rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg, ra: Arm64Reg) -> u32 {
    (sf << 31)
        | (0b00_11011_000 << 21)
        | (rm.index() << 16)
        | (ra.index() << 10)
        | (rn.index() << 5)
        | rd.index()
}

fn shift_var(sf: u32, op2: u32, rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    (sf << 31)
        | (0b00_11010_110 << 21)
        | (rm.index() << 16)
        | (0b0010 << 12)
        | (op2 << 10)
        | (rn.index() << 5)
        | rd.index()
}

fn add_sub_imm(sf: u32, op: u32, s: u32, rd: Arm64Reg, rn: Arm64Reg, imm12: u32) -> u32 {
    debug_assert!(imm12 < 0x1000);
    (sf << 31)
        | (op << 30)
        | (s << 29)
        | (0b100010 << 23)
        | (imm12 << 10)
        | (rn.index() << 5)
        | rd.index()
}

fn cond_select(sf: u32, op2: u32, rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg, cond: Cond) -> u32 {
    (sf << 31)
        | (0b0_0_11010100 << 21)
        | (rm.index() << 16)
        | ((cond as u32) << 12)
        | (op2 << 10)
        | (rn.index() << 5)
        | rd.index()
}

fn move_wide(sf: u32, opc: u32, rd: Arm64Reg, imm16: u16, shift: u32) -> u32 {
    let hw = shift / 16;
    debug_assert!(shift % 16 == 0);
    (sf << 31) | (opc << 29) | (0b100101 << 23) | (hw << 21) | ((imm16 as u32) << 5) | rd.index()
}

fn sbfm(sf: u32, n: u32, rd: Arm64Reg, rn: Arm64Reg, immr: u32, imms: u32) -> u32 {
    (sf << 31)
        | (0b00 << 29)
        | (0b100110 << 23)
        | (n << 22)
        | (immr << 16)
        | (imms << 10)
        | (rn.index() << 5)
        | rd.index()
}

fn ubfm(sf: u32, n: u32, rd: Arm64Reg, rn: Arm64Reg, immr: u32, imms: u32) -> u32 {
    (sf << 31)
        | (0b10 << 29)
        | (0b100110 << 23)
        | (n << 22)
        | (immr << 16)
        | (imms << 10)
        | (rn.index() << 5)
        | rd.index()
}

fn ldst_unsigned_offset(size: u32, opc: u32, rt: Arm64Reg, rn: Arm64Reg, imm12: u32) -> u32 {
    debug_assert!(imm12 < 0x1000);
    (size << 30) | (0b111_0_01 << 24) | (opc << 22) | (imm12 << 10) | (rn.index() << 5) | rt.index()
}

fn ldst_register_offset(base: u32, rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg, scaled: bool) -> u32 {
    base | ((scaled as u32) << 12) | (rm.index() << 16) | (rn.index() << 5) | rt.index()
}

fn fp_ldst_register_offset(
    base: u32,
    rt: Arm64FpReg,
    rn: Arm64Reg,
    rm: Arm64Reg,
    scaled: bool,
) -> u32 {
    base | ((scaled as u32) << 12) | (rm.index() << 16) | (rn.index() << 5) | rt.index()
}

fn load_store_pair(base: u32, rt: Arm64Reg, rt2: Arm64Reg, rn: Arm64Reg, imm7: i32) -> u32 {
    debug_assert!((-64..64).contains(&imm7));
    let imm7_bits = (imm7 as u32) & 0x7f;
    base | (imm7_bits << 15) | (rt2.index() << 10) | (rn.index() << 5) | rt.index()
}

pub(crate) fn add_reg_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    add_sub_shifted_reg(0, 0, 0, rd, rn, rm)
}

pub(crate) fn add_reg_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    add_sub_shifted_reg(1, 0, 0, rd, rn, rm)
}

/// `add Xd, Xn, Wm, UXTW #0` — 64-bit add of base Xn and zero-extended 32-bit Wm.
/// Encoding: sf=1, op=0, S=0, 01011 001 Rm 010 000 Rn Rd
pub(crate) fn add_reg_64_uxtw(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    // ADD (extended register), 64-bit, option=010 (UXTW), shift=0
    0x8B20_4000 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn sub_reg_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    add_sub_shifted_reg(0, 1, 0, rd, rn, rm)
}

pub(crate) fn sub_reg_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    add_sub_shifted_reg(1, 1, 0, rd, rn, rm)
}

pub(crate) fn mul_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    madd(0, rd, rn, rm, abi::zero_reg())
}

pub(crate) fn mul_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    madd(1, rd, rn, rm, abi::zero_reg())
}

pub(crate) fn and_reg_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    logical_shifted_reg(0, 0b00, 0, rd, rn, rm)
}

pub(crate) fn and_reg_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    logical_shifted_reg(1, 0b00, 0, rd, rn, rm)
}

pub(crate) fn orr_reg_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    logical_shifted_reg(0, 0b01, 0, rd, rn, rm)
}

pub(crate) fn orr_reg_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    logical_shifted_reg(1, 0b01, 0, rd, rn, rm)
}

pub(crate) fn eor_reg_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    logical_shifted_reg(0, 0b10, 0, rd, rn, rm)
}

pub(crate) fn eor_reg_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    logical_shifted_reg(1, 0b10, 0, rd, rn, rm)
}

// --- Shifted-register binary operations ---

pub(crate) fn add_reg_shifted_32(
    rd: Arm64Reg,
    rn: Arm64Reg,
    rm: Arm64Reg,
    shift: ShiftType,
    amount: u32,
) -> u32 {
    add_sub_shifted_reg_ext(0, 0, 0, rd, rn, rm, shift, amount)
}

pub(crate) fn add_reg_shifted_64(
    rd: Arm64Reg,
    rn: Arm64Reg,
    rm: Arm64Reg,
    shift: ShiftType,
    amount: u32,
) -> u32 {
    add_sub_shifted_reg_ext(1, 0, 0, rd, rn, rm, shift, amount)
}

pub(crate) fn sub_reg_shifted_32(
    rd: Arm64Reg,
    rn: Arm64Reg,
    rm: Arm64Reg,
    shift: ShiftType,
    amount: u32,
) -> u32 {
    add_sub_shifted_reg_ext(0, 1, 0, rd, rn, rm, shift, amount)
}

pub(crate) fn sub_reg_shifted_64(
    rd: Arm64Reg,
    rn: Arm64Reg,
    rm: Arm64Reg,
    shift: ShiftType,
    amount: u32,
) -> u32 {
    add_sub_shifted_reg_ext(1, 1, 0, rd, rn, rm, shift, amount)
}

pub(crate) fn and_reg_shifted_32(
    rd: Arm64Reg,
    rn: Arm64Reg,
    rm: Arm64Reg,
    shift: ShiftType,
    amount: u32,
) -> u32 {
    logical_shifted_reg_ext(0, 0b00, 0, rd, rn, rm, shift, amount)
}

pub(crate) fn and_reg_shifted_64(
    rd: Arm64Reg,
    rn: Arm64Reg,
    rm: Arm64Reg,
    shift: ShiftType,
    amount: u32,
) -> u32 {
    logical_shifted_reg_ext(1, 0b00, 0, rd, rn, rm, shift, amount)
}

pub(crate) fn orr_reg_shifted_32(
    rd: Arm64Reg,
    rn: Arm64Reg,
    rm: Arm64Reg,
    shift: ShiftType,
    amount: u32,
) -> u32 {
    logical_shifted_reg_ext(0, 0b01, 0, rd, rn, rm, shift, amount)
}

pub(crate) fn orr_reg_shifted_64(
    rd: Arm64Reg,
    rn: Arm64Reg,
    rm: Arm64Reg,
    shift: ShiftType,
    amount: u32,
) -> u32 {
    logical_shifted_reg_ext(1, 0b01, 0, rd, rn, rm, shift, amount)
}

pub(crate) fn eor_reg_shifted_32(
    rd: Arm64Reg,
    rn: Arm64Reg,
    rm: Arm64Reg,
    shift: ShiftType,
    amount: u32,
) -> u32 {
    logical_shifted_reg_ext(0, 0b10, 0, rd, rn, rm, shift, amount)
}

pub(crate) fn eor_reg_shifted_64(
    rd: Arm64Reg,
    rn: Arm64Reg,
    rm: Arm64Reg,
    shift: ShiftType,
    amount: u32,
) -> u32 {
    logical_shifted_reg_ext(1, 0b10, 0, rd, rn, rm, shift, amount)
}

// --- TST (ANDS with Rd=XZR, sets flags, discards result) ---

pub(crate) fn ands_imm_32(rd: Arm64Reg, rn: Arm64Reg, imm: u32) -> Option<u32> {
    let (n, immr, imms) = encode_logical_immediate(imm as u64, 32)?;
    Some(logical_imm(0, 0b11, n, rd, rn, immr, imms))
}

pub(crate) fn ands_imm_64(rd: Arm64Reg, rn: Arm64Reg, imm: u64) -> Option<u32> {
    let (n, immr, imms) = encode_logical_immediate(imm, 64)?;
    Some(logical_imm(1, 0b11, n, rd, rn, immr, imms))
}

pub(crate) fn ands_reg_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    logical_shifted_reg(0, 0b11, 0, rd, rn, rm)
}

pub(crate) fn ands_reg_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    logical_shifted_reg(1, 0b11, 0, rd, rn, rm)
}

pub(crate) fn tst_imm_32(rn: Arm64Reg, imm: u32) -> Option<u32> {
    let (n, immr, imms) = encode_logical_immediate(imm as u64, 32)?;
    Some(logical_imm(0, 0b11, n, abi::zero_reg(), rn, immr, imms))
}

pub(crate) fn tst_imm_64(rn: Arm64Reg, imm: u64) -> Option<u32> {
    let (n, immr, imms) = encode_logical_immediate(imm, 64)?;
    Some(logical_imm(1, 0b11, n, abi::zero_reg(), rn, immr, imms))
}

pub(crate) fn tst_reg_32(rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    logical_shifted_reg(0, 0b11, 0, abi::zero_reg(), rn, rm)
}

pub(crate) fn tst_reg_64(rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    logical_shifted_reg(1, 0b11, 0, abi::zero_reg(), rn, rm)
}

// --- UBFX (unsigned bitfield extract) ---

pub(crate) fn ubfx_32(rd: Arm64Reg, rn: Arm64Reg, lsb: u32, width: u32) -> u32 {
    debug_assert!(lsb + width <= 32 && width > 0);
    ubfm(0, 0, rd, rn, lsb, lsb + width - 1)
}

pub(crate) fn ubfx_64(rd: Arm64Reg, rn: Arm64Reg, lsb: u32, width: u32) -> u32 {
    debug_assert!(lsb + width <= 64 && width > 0);
    ubfm(1, 1, rd, rn, lsb, lsb + width - 1)
}

pub(crate) fn orn_reg_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    logical_shifted_reg(0, 0b01, 1, rd, rn, rm)
}

pub(crate) fn orn_reg_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    logical_shifted_reg(1, 0b01, 1, rd, rn, rm)
}

pub(crate) fn mvn_32(rd: Arm64Reg, rm: Arm64Reg) -> u32 {
    orn_reg_32(rd, abi::zero_reg(), rm)
}

pub(crate) fn mvn_64(rd: Arm64Reg, rm: Arm64Reg) -> u32 {
    orn_reg_64(rd, abi::zero_reg(), rm)
}

pub(crate) fn lslv_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    shift_var(0, 0b00, rd, rn, rm)
}

pub(crate) fn lsrv_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    shift_var(0, 0b01, rd, rn, rm)
}

pub(crate) fn asrv_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    shift_var(0, 0b10, rd, rn, rm)
}

pub(crate) fn lslv_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    shift_var(1, 0b00, rd, rn, rm)
}

pub(crate) fn lsrv_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    shift_var(1, 0b01, rd, rn, rm)
}

pub(crate) fn asrv_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    shift_var(1, 0b10, rd, rn, rm)
}

pub(crate) fn add_imm_64(rd: Arm64Reg, rn: Arm64Reg, imm12: u32) -> u32 {
    add_sub_imm(1, 0, 0, rd, rn, imm12)
}

pub(crate) fn add_imm_32(rd: Arm64Reg, rn: Arm64Reg, imm12: u32) -> u32 {
    add_sub_imm(0, 0, 0, rd, rn, imm12)
}

pub(crate) fn sub_imm_64(rd: Arm64Reg, rn: Arm64Reg, imm12: u32) -> u32 {
    add_sub_imm(1, 1, 0, rd, rn, imm12)
}

pub(crate) fn sub_imm_32(rd: Arm64Reg, rn: Arm64Reg, imm12: u32) -> u32 {
    add_sub_imm(0, 1, 0, rd, rn, imm12)
}

pub(crate) fn cmp_reg_32(rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    add_sub_shifted_reg(0, 1, 1, abi::zero_reg(), rn, rm)
}

pub(crate) fn cmp_reg_64(rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    add_sub_shifted_reg(1, 1, 1, abi::zero_reg(), rn, rm)
}

pub(crate) fn cmp_imm_64(rn: Arm64Reg, imm12: u32) -> u32 {
    add_sub_imm(1, 1, 1, abi::zero_reg(), rn, imm12)
}

pub(crate) fn cset_32(rd: Arm64Reg, cond: Cond) -> u32 {
    cond_select(0, 0b01, rd, abi::zero_reg(), abi::zero_reg(), cond.invert())
}

pub(crate) fn cset_64(rd: Arm64Reg, cond: Cond) -> u32 {
    cond_select(1, 0b01, rd, abi::zero_reg(), abi::zero_reg(), cond.invert())
}

pub(crate) fn csel_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg, cond: Cond) -> u32 {
    cond_select(1, 0b00, rd, rn, rm, cond)
}

pub(crate) fn mov_reg_64(rd: Arm64Reg, rm: Arm64Reg) -> u32 {
    orr_reg_64(rd, abi::zero_reg(), rm)
}

pub(crate) fn movz_64(rd: Arm64Reg, imm16: u16, shift: u32) -> u32 {
    move_wide(1, 0b10, rd, imm16, shift)
}

pub(crate) fn movk_64(rd: Arm64Reg, imm16: u16, shift: u32) -> u32 {
    move_wide(1, 0b11, rd, imm16, shift)
}

pub(crate) fn movn_64(rd: Arm64Reg, imm16: u16, shift: u32) -> u32 {
    move_wide(1, 0b00, rd, imm16, shift)
}

pub(crate) fn clz_32(rd: Arm64Reg, rn: Arm64Reg) -> u32 {
    (0b0_10_11010_110 << 21) | (0b00000 << 16) | (0b00010_0 << 10) | (rn.index() << 5) | rd.index()
}

pub(crate) fn clz_64(rd: Arm64Reg, rn: Arm64Reg) -> u32 {
    (0b1_10_11010_110 << 21) | (0b00000 << 16) | (0b00010_0 << 10) | (rn.index() << 5) | rd.index()
}

pub(crate) fn sxtb_32(rd: Arm64Reg, rn: Arm64Reg) -> u32 {
    sbfm(0, 0, rd, rn, 0, 7)
}

pub(crate) fn sxtb_64(rd: Arm64Reg, rn: Arm64Reg) -> u32 {
    sbfm(1, 1, rd, rn, 0, 7)
}

pub(crate) fn sxth_32(rd: Arm64Reg, rn: Arm64Reg) -> u32 {
    sbfm(0, 0, rd, rn, 0, 15)
}

pub(crate) fn sxth_64(rd: Arm64Reg, rn: Arm64Reg) -> u32 {
    sbfm(1, 1, rd, rn, 0, 15)
}

pub(crate) fn sxtw(rd: Arm64Reg, rn: Arm64Reg) -> u32 {
    sbfm(1, 1, rd, rn, 0, 31)
}

pub(crate) fn lsl_imm_32(rd: Arm64Reg, rn: Arm64Reg, shift: u32) -> u32 {
    debug_assert!(shift < 32);
    ubfm(0, 0, rd, rn, (32 - shift) & 31, 31 - shift)
}

pub(crate) fn lsl_imm_64(rd: Arm64Reg, rn: Arm64Reg, shift: u32) -> u32 {
    debug_assert!(shift < 64);
    ubfm(1, 1, rd, rn, (64 - shift) & 63, 63 - shift)
}

pub(crate) fn lsr_imm_32(rd: Arm64Reg, rn: Arm64Reg, shift: u32) -> u32 {
    debug_assert!(shift < 32);
    ubfm(0, 0, rd, rn, shift, 31)
}

pub(crate) fn lsr_imm_64(rd: Arm64Reg, rn: Arm64Reg, shift: u32) -> u32 {
    debug_assert!(shift < 64);
    ubfm(1, 1, rd, rn, shift, 63)
}

pub(crate) fn asr_imm_32(rd: Arm64Reg, rn: Arm64Reg, shift: u32) -> u32 {
    debug_assert!(shift < 32);
    sbfm(0, 0, rd, rn, shift, 31)
}

pub(crate) fn asr_imm_64(rd: Arm64Reg, rn: Arm64Reg, shift: u32) -> u32 {
    debug_assert!(shift < 64);
    sbfm(1, 1, rd, rn, shift, 63)
}

pub(crate) fn ldr_64(rt: Arm64Reg, rn: Arm64Reg, imm12: u32) -> u32 {
    ldst_unsigned_offset(0b11, 0b01, rt, rn, imm12)
}

pub(crate) fn ldr_32(rt: Arm64Reg, rn: Arm64Reg, imm12: u32) -> u32 {
    ldst_unsigned_offset(0b10, 0b01, rt, rn, imm12)
}

pub(crate) fn ldrb_imm(rt: Arm64Reg, rn: Arm64Reg, imm12: u32) -> u32 {
    ldst_unsigned_offset(0b00, 0b01, rt, rn, imm12)
}

pub(crate) fn ldrh_imm(rt: Arm64Reg, rn: Arm64Reg, imm12: u32) -> u32 {
    ldst_unsigned_offset(0b01, 0b01, rt, rn, imm12)
}

pub(crate) fn ldrsb_imm_64(rt: Arm64Reg, rn: Arm64Reg, imm12: u32) -> u32 {
    ldst_unsigned_offset(0b00, 0b10, rt, rn, imm12)
}

pub(crate) fn ldrsh_imm_64(rt: Arm64Reg, rn: Arm64Reg, imm12: u32) -> u32 {
    ldst_unsigned_offset(0b01, 0b10, rt, rn, imm12)
}

pub(crate) fn ldrsw_imm(rt: Arm64Reg, rn: Arm64Reg, imm12: u32) -> u32 {
    ldst_unsigned_offset(0b10, 0b10, rt, rn, imm12)
}

pub(crate) fn str_32(rt: Arm64Reg, rn: Arm64Reg, imm12: u32) -> u32 {
    ldst_unsigned_offset(0b10, 0b00, rt, rn, imm12)
}

pub(crate) fn strb_imm(rt: Arm64Reg, rn: Arm64Reg, imm12: u32) -> u32 {
    ldst_unsigned_offset(0b00, 0b00, rt, rn, imm12)
}

pub(crate) fn strh_imm(rt: Arm64Reg, rn: Arm64Reg, imm12: u32) -> u32 {
    ldst_unsigned_offset(0b01, 0b00, rt, rn, imm12)
}

pub(crate) fn ldr_reg_64(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0xf860_6800, rt, rn, rm, false)
}

/// `ldr Xt, [Xn, Wm, UXTW]` — load with 32-bit zero-extending index.
pub(crate) fn ldr_reg_64_uxtw(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0xf860_4800, rt, rn, rm, false)
}

/// `str Xt, [Xn, Wm, UXTW]` — store with 32-bit zero-extending index.
pub(crate) fn str_reg_64_uxtw(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0xf820_4800, rt, rn, rm, false)
}

/// `ldr Wt, [Xn, Wm, UXTW]` — load 32-bit with zero-extending index.
pub(crate) fn ldr_reg_32_uxtw(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0xb860_4800, rt, rn, rm, false)
}

/// `ldrb Wt, [Xn, Wm, UXTW]` — load byte with zero-extending index.
pub(crate) fn ldrb_reg_uxtw(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0x3860_4800, rt, rn, rm, false)
}

/// `ldrh Wt, [Xn, Wm, UXTW]` — load halfword with zero-extending index.
pub(crate) fn ldrh_reg_uxtw(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0x7860_4800, rt, rn, rm, false)
}

/// `ldrsb Xt, [Xn, Wm, UXTW]` — load signed byte with zero-extending index.
pub(crate) fn ldrsb_reg_64_uxtw(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0x38a0_4800, rt, rn, rm, false)
}

/// `ldrsh Xt, [Xn, Wm, UXTW]` — load signed halfword with zero-extending index.
pub(crate) fn ldrsh_reg_64_uxtw(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0x78a0_4800, rt, rn, rm, false)
}

/// `ldrsw Xt, [Xn, Wm, UXTW]` — load signed word with zero-extending index.
pub(crate) fn ldrsw_reg_uxtw(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0xb8a0_4800, rt, rn, rm, false)
}

/// `strb Wt, [Xn, Wm, UXTW]` — store byte with zero-extending index.
pub(crate) fn strb_reg_uxtw(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0x3820_4800, rt, rn, rm, false)
}

/// `strh Wt, [Xn, Wm, UXTW]` — store halfword with zero-extending index.
pub(crate) fn strh_reg_uxtw(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0x7820_4800, rt, rn, rm, false)
}

/// `str Wt, [Xn, Wm, UXTW]` — store 32-bit with zero-extending index.
pub(crate) fn str_reg_32_uxtw(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0xb820_4800, rt, rn, rm, false)
}

pub(crate) fn ldr_reg_64_scaled(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0xf860_6800, rt, rn, rm, true)
}

pub(crate) fn str_64(rt: Arm64Reg, rn: Arm64Reg, imm12: u32) -> u32 {
    ldst_unsigned_offset(0b11, 0b00, rt, rn, imm12)
}

pub(crate) fn ldr_reg_32(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0xb860_6800, rt, rn, rm, false)
}

pub(crate) fn ldr_reg_32_scaled(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0xb860_6800, rt, rn, rm, true)
}

pub(crate) fn ldrb_reg(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0x3860_6800, rt, rn, rm, false)
}

pub(crate) fn ldrh_reg(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0x7860_6800, rt, rn, rm, false)
}

pub(crate) fn ldrh_reg_scaled(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0x7860_6800, rt, rn, rm, true)
}

pub(crate) fn ldrsw_reg(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0xb8a0_6800, rt, rn, rm, false)
}

pub(crate) fn ldrsw_reg_scaled(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0xb8a0_6800, rt, rn, rm, true)
}

pub(crate) fn ldrsb_reg_64(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0x38a0_6800, rt, rn, rm, false)
}

pub(crate) fn ldrsh_reg_64(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0x78a0_6800, rt, rn, rm, false)
}

pub(crate) fn ldrsh_reg_64_scaled(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0x78a0_6800, rt, rn, rm, true)
}

pub(crate) fn str_reg_64(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0xf820_6800, rt, rn, rm, false)
}

pub(crate) fn str_reg_64_scaled(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0xf820_6800, rt, rn, rm, true)
}

pub(crate) fn str_reg_32(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0xb820_6800, rt, rn, rm, false)
}

pub(crate) fn str_reg_32_scaled(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0xb820_6800, rt, rn, rm, true)
}

pub(crate) fn strb_reg(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0x3820_6800, rt, rn, rm, false)
}

pub(crate) fn strh_reg(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0x7820_6800, rt, rn, rm, false)
}

pub(crate) fn strh_reg_scaled(rt: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    ldst_register_offset(0x7820_6800, rt, rn, rm, true)
}

pub(crate) fn ldr_s(rt: Arm64FpReg, rn: Arm64Reg, imm12: u32) -> u32 {
    debug_assert!(imm12 < 0x1000);
    0xbd40_0000 | (imm12 << 10) | (rn.index() << 5) | rt.index()
}

pub(crate) fn ldr_d(rt: Arm64FpReg, rn: Arm64Reg, imm12: u32) -> u32 {
    debug_assert!(imm12 < 0x1000);
    0xfd40_0000 | (imm12 << 10) | (rn.index() << 5) | rt.index()
}

pub(crate) fn ldr_q(rt: Arm64FpReg, rn: Arm64Reg, imm12: u32) -> u32 {
    debug_assert!(imm12 < 0x1000);
    0x3dc0_0000 | (imm12 << 10) | (rn.index() << 5) | rt.index()
}

pub(crate) fn str_s(rt: Arm64FpReg, rn: Arm64Reg, imm12: u32) -> u32 {
    debug_assert!(imm12 < 0x1000);
    0xbd00_0000 | (imm12 << 10) | (rn.index() << 5) | rt.index()
}

pub(crate) fn str_d(rt: Arm64FpReg, rn: Arm64Reg, imm12: u32) -> u32 {
    debug_assert!(imm12 < 0x1000);
    0xfd00_0000 | (imm12 << 10) | (rn.index() << 5) | rt.index()
}

pub(crate) fn str_q(rt: Arm64FpReg, rn: Arm64Reg, imm12: u32) -> u32 {
    debug_assert!(imm12 < 0x1000);
    0x3d80_0000 | (imm12 << 10) | (rn.index() << 5) | rt.index()
}

pub(crate) fn ldr_s_reg(rt: Arm64FpReg, rn: Arm64Reg, rm: Arm64Reg, scaled: bool) -> u32 {
    fp_ldst_register_offset(0xbc60_6800, rt, rn, rm, scaled)
}

pub(crate) fn ldr_d_reg(rt: Arm64FpReg, rn: Arm64Reg, rm: Arm64Reg, scaled: bool) -> u32 {
    fp_ldst_register_offset(0xfc60_6800, rt, rn, rm, scaled)
}

/// `ldr Dd, [Xn, Wm, UXTW]` — load with 32-bit zero-extending index.
pub(crate) fn ldr_d_reg_uxtw(rt: Arm64FpReg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    fp_ldst_register_offset(0xfc60_4800, rt, rn, rm, false)
}

/// `ldr Sd, [Xn, Wm, UXTW]` — load with 32-bit zero-extending index.
pub(crate) fn ldr_s_reg_uxtw(rt: Arm64FpReg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    fp_ldst_register_offset(0xbc60_4800, rt, rn, rm, false)
}

pub(crate) fn str_s_reg(rt: Arm64FpReg, rn: Arm64Reg, rm: Arm64Reg, scaled: bool) -> u32 {
    fp_ldst_register_offset(0xbc20_6800, rt, rn, rm, scaled)
}

pub(crate) fn str_d_reg(rt: Arm64FpReg, rn: Arm64Reg, rm: Arm64Reg, scaled: bool) -> u32 {
    fp_ldst_register_offset(0xfc20_6800, rt, rn, rm, scaled)
}

/// `str Dd, [Xn, Wm, UXTW]` — store with 32-bit zero-extending index.
pub(crate) fn str_d_reg_uxtw(rt: Arm64FpReg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    fp_ldst_register_offset(0xfc20_4800, rt, rn, rm, false)
}

/// `str Sd, [Xn, Wm, UXTW]` — store with 32-bit zero-extending index.
pub(crate) fn str_s_reg_uxtw(rt: Arm64FpReg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    fp_ldst_register_offset(0xbc20_4800, rt, rn, rm, false)
}

pub(crate) fn stp_64(rt: Arm64Reg, rt2: Arm64Reg, rn: Arm64Reg, imm7: i32) -> u32 {
    load_store_pair(0xa900_0000, rt, rt2, rn, imm7)
}

pub(crate) fn ldp_64(rt: Arm64Reg, rt2: Arm64Reg, rn: Arm64Reg, imm7: i32) -> u32 {
    load_store_pair(0xa940_0000, rt, rt2, rn, imm7)
}

/// `stp Dt, Dt2, [Xn, #imm]` — store pair of 64-bit FP registers (signed offset).
/// `imm7` is the byte offset divided by 8 (this fn does not scale).
pub(crate) fn stp_d(rt: Arm64FpReg, rt2: Arm64FpReg, rn: Arm64Reg, imm7: i32) -> u32 {
    let imm7_bits = (imm7 as u32) & 0x7f;
    0x6d00_0000 | (imm7_bits << 15) | (rt2.index() << 10) | (rn.index() << 5) | rt.index()
}

/// `ldp Dt, Dt2, [Xn, #imm]` — load pair of 64-bit FP registers (signed offset).
pub(crate) fn ldp_d(rt: Arm64FpReg, rt2: Arm64FpReg, rn: Arm64Reg, imm7: i32) -> u32 {
    let imm7_bits = (imm7 as u32) & 0x7f;
    0x6d40_0000 | (imm7_bits << 15) | (rt2.index() << 10) | (rn.index() << 5) | rt.index()
}

/// `stp Xt, Xt2, [Xn, #imm]!` — 64-bit pre-index store-pair (atomic 16-byte
/// push when Xn is `sp` and `imm` is the negative slot offset).
///
/// `imm` is the byte offset, scaled internally by 8 to fit the 7-bit signed
/// immediate. For pushing onto the host stack, pass `-16`.
pub(crate) fn stp_64_pre_index(rt: Arm64Reg, rt2: Arm64Reg, rn: Arm64Reg, imm: i32) -> u32 {
    load_store_pair(0xa980_0000, rt, rt2, rn, imm / 8)
}

/// `ldp Xt, Xt2, [Xn], #imm` — 64-bit post-index load-pair (atomic 16-byte
/// pop when Xn is `sp` and `imm` is the positive slot offset).
///
/// `imm` is the byte offset, scaled internally by 8 to fit the 7-bit signed
/// immediate. For popping from the host stack, pass `16`.
pub(crate) fn ldp_64_post_index(rt: Arm64Reg, rt2: Arm64Reg, rn: Arm64Reg, imm: i32) -> u32 {
    load_store_pair(0xa8c0_0000, rt, rt2, rn, imm / 8)
}

pub(crate) fn b(imm26: i32) -> u32 {
    let imm26_bits = (imm26 as u32) & 0x03FF_FFFF;
    (0b000101 << 26) | imm26_bits
}

/// `bl #imm` — branch with link, populates LR with the address of the
/// instruction after the BL. `imm26` is signed and counted in 4-byte
/// instruction units (so the actual reach is ±128MB).
pub(crate) fn bl(imm26: i32) -> u32 {
    let imm26_bits = (imm26 as u32) & 0x03FF_FFFF;
    (0b100101 << 26) | imm26_bits
}

pub(crate) fn ldr_lit_64(rt: Arm64Reg, imm19: i32) -> u32 {
    let imm19_bits = (imm19 as u32) & 0x0007_FFFF;
    (0b01011000 << 24) | (imm19_bits << 5) | rt.index()
}

pub(crate) fn b_cond(cond: Cond, imm19: i32) -> u32 {
    let imm19_bits = (imm19 as u32) & 0x0007_FFFF;
    (0b01010100 << 24) | (imm19_bits << 5) | (cond as u32)
}

pub(crate) fn cbz_64(rt: Arm64Reg, imm19: i32) -> u32 {
    let imm19_bits = (imm19 as u32) & 0x0007_FFFF;
    (0b1_011010_0 << 24) | (imm19_bits << 5) | rt.index()
}

pub(crate) fn cbnz_64(rt: Arm64Reg, imm19: i32) -> u32 {
    let imm19_bits = (imm19 as u32) & 0x0007_FFFF;
    (0b1_011010_1 << 24) | (imm19_bits << 5) | rt.index()
}

pub(crate) fn br(rn: Arm64Reg) -> u32 {
    (0b1101011_0000 << 21) | (0b11111 << 16) | (rn.index() << 5)
}

pub(crate) fn blr(rn: Arm64Reg) -> u32 {
    (0b1101011_0001 << 21) | (0b11111 << 16) | (rn.index() << 5)
}

pub(crate) fn ret() -> u32 {
    (0b1101011_0010 << 21) | (0b11111 << 16) | (abi::link_reg().index() << 5)
}

// --- Integer data-processing (2 source) ---

fn data_proc_2src(sf: u32, opcode: u32, rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    (sf << 31)
        | (0b0_11010110 << 21)
        | (rm.index() << 16)
        | (opcode << 10)
        | (rn.index() << 5)
        | rd.index()
}

pub(crate) fn udiv_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    data_proc_2src(0, 0b000010, rd, rn, rm)
}

pub(crate) fn udiv_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    data_proc_2src(1, 0b000010, rd, rn, rm)
}

pub(crate) fn sdiv_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    data_proc_2src(0, 0b000011, rd, rn, rm)
}

pub(crate) fn sdiv_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    data_proc_2src(1, 0b000011, rd, rn, rm)
}

pub(crate) fn rorv_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    shift_var(0, 0b11, rd, rn, rm)
}

pub(crate) fn rorv_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> u32 {
    shift_var(1, 0b11, rd, rn, rm)
}

/// MSUB Rd, Rn, Rm, Ra: Rd = Ra - Rn * Rm
fn msub(sf: u32, rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg, ra: Arm64Reg) -> u32 {
    (sf << 31)
        | (0b00_11011_000 << 21)
        | (rm.index() << 16)
        | (1 << 15)
        | (ra.index() << 10)
        | (rn.index() << 5)
        | rd.index()
}

pub(crate) fn msub_32(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg, ra: Arm64Reg) -> u32 {
    msub(0, rd, rn, rm, ra)
}

pub(crate) fn msub_64(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg, ra: Arm64Reg) -> u32 {
    msub(1, rd, rn, rm, ra)
}

// --- Bit manipulation ---

fn data_proc_1src(sf: u32, opcode2: u32, opcode: u32, rd: Arm64Reg, rn: Arm64Reg) -> u32 {
    (sf << 31)
        | (0b1_0_11010110 << 21)
        | (opcode2 << 16)
        | (opcode << 10)
        | (rn.index() << 5)
        | rd.index()
}

pub(crate) fn rbit_32(rd: Arm64Reg, rn: Arm64Reg) -> u32 {
    data_proc_1src(0, 0b00000, 0b000000, rd, rn)
}

pub(crate) fn rbit_64(rd: Arm64Reg, rn: Arm64Reg) -> u32 {
    data_proc_1src(1, 0b00000, 0b000000, rd, rn)
}

pub(crate) fn cmp_imm_32(rn: Arm64Reg, imm12: u32) -> u32 {
    add_sub_imm(0, 1, 1, abi::zero_reg(), rn, imm12)
}

pub(crate) fn neg_reg_32(rd: Arm64Reg, rm: Arm64Reg) -> u32 {
    sub_reg_32(rd, abi::zero_reg(), rm)
}

pub(crate) fn neg_reg_64(rd: Arm64Reg, rm: Arm64Reg) -> u32 {
    sub_reg_64(rd, abi::zero_reg(), rm)
}

pub(crate) fn mov_reg_32(rd: Arm64Reg, rm: Arm64Reg) -> u32 {
    orr_reg_32(rd, abi::zero_reg(), rm)
}

pub(crate) fn cmp_zero_32(rn: Arm64Reg) -> u32 {
    cmp_reg_32(rn, abi::zero_reg())
}

pub(crate) fn cmp_zero_64(rn: Arm64Reg) -> u32 {
    cmp_reg_64(rn, abi::zero_reg())
}

pub(crate) fn mov_zero_64(rd: Arm64Reg) -> u32 {
    mov_reg_64(rd, abi::zero_reg())
}

pub(crate) fn str_zero_64(rn: Arm64Reg, imm12: u32) -> u32 {
    str_64(abi::zero_reg(), rn, imm12)
}

pub(crate) fn stp_zero_64(rn: Arm64Reg, imm7: i32) -> u32 {
    stp_64(abi::zero_reg(), abi::zero_reg(), rn, imm7)
}

pub(crate) fn ldr_s_base(rt: Arm64FpReg, rn: Arm64Reg) -> u32 {
    ldr_s_reg(rt, rn, abi::zero_reg(), false)
}

pub(crate) fn ldr_d_base(rt: Arm64FpReg, rn: Arm64Reg) -> u32 {
    ldr_d_reg(rt, rn, abi::zero_reg(), false)
}

pub(crate) fn str_s_base(rt: Arm64FpReg, rn: Arm64Reg) -> u32 {
    str_s_reg(rt, rn, abi::zero_reg(), false)
}

pub(crate) fn str_d_base(rt: Arm64FpReg, rn: Arm64Reg) -> u32 {
    str_d_reg(rt, rn, abi::zero_reg(), false)
}

pub(crate) fn ldrb_base(rt: Arm64Reg, rn: Arm64Reg) -> u32 {
    ldrb_reg(rt, rn, abi::zero_reg())
}

pub(crate) fn ldrsb_64_base(rt: Arm64Reg, rn: Arm64Reg) -> u32 {
    ldrsb_reg_64(rt, rn, abi::zero_reg())
}

pub(crate) fn ldrh_base(rt: Arm64Reg, rn: Arm64Reg) -> u32 {
    ldrh_reg(rt, rn, abi::zero_reg())
}

pub(crate) fn ldrsh_64_base(rt: Arm64Reg, rn: Arm64Reg) -> u32 {
    ldrsh_reg_64(rt, rn, abi::zero_reg())
}

pub(crate) fn ldr_reg_32_base(rt: Arm64Reg, rn: Arm64Reg) -> u32 {
    ldr_reg_32(rt, rn, abi::zero_reg())
}

pub(crate) fn ldrsw_base(rt: Arm64Reg, rn: Arm64Reg) -> u32 {
    ldrsw_reg(rt, rn, abi::zero_reg())
}

pub(crate) fn ldr_reg_64_base(rt: Arm64Reg, rn: Arm64Reg) -> u32 {
    ldr_reg_64(rt, rn, abi::zero_reg())
}

pub(crate) fn strb_base(rt: Arm64Reg, rn: Arm64Reg) -> u32 {
    strb_reg(rt, rn, abi::zero_reg())
}

pub(crate) fn strh_base(rt: Arm64Reg, rn: Arm64Reg) -> u32 {
    strh_reg(rt, rn, abi::zero_reg())
}

pub(crate) fn str_reg_32_base(rt: Arm64Reg, rn: Arm64Reg) -> u32 {
    str_reg_32(rt, rn, abi::zero_reg())
}

pub(crate) fn str_reg_64_base(rt: Arm64Reg, rn: Arm64Reg) -> u32 {
    str_reg_64(rt, rn, abi::zero_reg())
}

pub(crate) fn movz_32(rd: Arm64Reg, imm16: u16, shift: u32) -> u32 {
    move_wide(0, 0b10, rd, imm16, shift)
}

pub(crate) fn movk_32(rd: Arm64Reg, imm16: u16, shift: u32) -> u32 {
    move_wide(0, 0b11, rd, imm16, shift)
}

pub(crate) fn movn_32(rd: Arm64Reg, imm16: u16, shift: u32) -> u32 {
    move_wide(0, 0b00, rd, imm16, shift)
}

fn low_mask(bits: u32) -> u64 {
    if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

fn ror_pattern(value: u64, rot: u32, width: u32) -> u64 {
    let mask = low_mask(width);
    let value = value & mask;
    let rot = rot % width;
    if rot == 0 {
        value
    } else {
        ((value >> rot) | (value << (width - rot))) & mask
    }
}

fn repeat_pattern(pattern: u64, elem_bits: u32, width: u32) -> u64 {
    let pattern = pattern & low_mask(elem_bits);
    let mut repeated = 0u64;
    let mut shift = 0u32;
    while shift < width {
        repeated |= pattern << shift;
        shift += elem_bits;
    }
    repeated & low_mask(width)
}

fn encode_logical_immediate(value: u64, width: u32) -> Option<(u32, u32, u32)> {
    debug_assert!(width == 32 || width == 64);

    let value = value & low_mask(width);
    if value == 0 || value == low_mask(width) {
        return None;
    }

    for elem_bits in [2u32, 4, 8, 16, 32, 64] {
        if elem_bits > width {
            break;
        }
        if width % elem_bits != 0 {
            continue;
        }

        let pattern = value & low_mask(elem_bits);
        if repeat_pattern(pattern, elem_bits, width) != value {
            continue;
        }

        for ones in 1..elem_bits {
            let base = low_mask(ones);
            for rot in 0..elem_bits {
                if ror_pattern(base, rot, elem_bits) != pattern {
                    continue;
                }
                let n = u32::from(width == 64 && elem_bits == 64);
                let immr = rot;
                let imms = (((elem_bits << 1).wrapping_neg()) | (ones - 1)) & 0x3f;
                return Some((n, immr, imms));
            }
        }
    }

    None
}

pub(crate) fn and_imm_32(rd: Arm64Reg, rn: Arm64Reg, imm: u32) -> Option<u32> {
    let (n, immr, imms) = encode_logical_immediate(imm as u64, 32)?;
    Some(logical_imm(0, 0b00, n, rd, rn, immr, imms))
}

pub(crate) fn and_imm_64(rd: Arm64Reg, rn: Arm64Reg, imm: u64) -> Option<u32> {
    let (n, immr, imms) = encode_logical_immediate(imm, 64)?;
    Some(logical_imm(1, 0b00, n, rd, rn, immr, imms))
}

pub(crate) fn orr_imm_32(rd: Arm64Reg, rn: Arm64Reg, imm: u32) -> Option<u32> {
    let (n, immr, imms) = encode_logical_immediate(imm as u64, 32)?;
    Some(logical_imm(0, 0b01, n, rd, rn, immr, imms))
}

pub(crate) fn orr_imm_64(rd: Arm64Reg, rn: Arm64Reg, imm: u64) -> Option<u32> {
    let (n, immr, imms) = encode_logical_immediate(imm, 64)?;
    Some(logical_imm(1, 0b01, n, rd, rn, immr, imms))
}

pub(crate) fn eor_imm_32(rd: Arm64Reg, rn: Arm64Reg, imm: u32) -> Option<u32> {
    let (n, immr, imms) = encode_logical_immediate(imm as u64, 32)?;
    Some(logical_imm(0, 0b10, n, rd, rn, immr, imms))
}

pub(crate) fn eor_imm_64(rd: Arm64Reg, rn: Arm64Reg, imm: u64) -> Option<u32> {
    let (n, immr, imms) = encode_logical_immediate(imm, 64)?;
    Some(logical_imm(1, 0b10, n, rd, rn, immr, imms))
}

// --- Floating-point instructions ---
// FP register index is just 0-31, same encoding slot as GP but in FP instruction formats.

/// Scalar floating-point data-processing (2 source)
fn fp_data_proc_2src(
    ftype: u32,
    opcode: u32,
    rd: Arm64FpReg,
    rn: Arm64FpReg,
    rm: Arm64FpReg,
) -> u32 {
    (0b0001_1110 << 24)
        | (ftype << 22)
        | (1 << 21)
        | (rm.index() << 16)
        | (opcode << 12)
        | (0b10 << 10)
        | (rn.index() << 5)
        | rd.index()
}

/// Scalar floating-point data-processing (1 source)
fn fp_data_proc_1src(ftype: u32, opcode: u32, rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    (0b0001_1110 << 24)
        | (ftype << 22)
        | (1 << 21)
        | (opcode << 15)
        | (0b10000 << 10)
        | (rn.index() << 5)
        | rd.index()
}

/// Floating-point comparison
fn fp_compare(ftype: u32, rn: Arm64FpReg, rm: Arm64FpReg, opc: u32) -> u32 {
    (0b0001_1110 << 24)
        | (ftype << 22)
        | (1 << 21)
        | (rm.index() << 16)
        | (0b00_1000 << 10)
        | (rn.index() << 5)
        | opc
}

/// Floating-point conditional select
fn fp_csel(ftype: u32, rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg, cond: Cond) -> u32 {
    (0b0001_1110 << 24)
        | (ftype << 22)
        | (1 << 21)
        | (rm.index() << 16)
        | ((cond as u32) << 12)
        | (0b11 << 10)
        | (rn.index() << 5)
        | rd.index()
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

// FMOV between GP and FP registers
/// FMOV Wd, Sn (FP to GP, 32-bit)
pub(crate) fn fmov_gp_from_s(rd: Arm64Reg, rn: Arm64FpReg) -> u32 {
    fp_int_conv(0, 0b00, 0b00, 0b110, rd.index(), rn.index())
}
/// FMOV Sd, Wn (GP to FP, 32-bit)
pub(crate) fn fmov_s_from_gp(rd: Arm64FpReg, rn: Arm64Reg) -> u32 {
    fp_int_conv(0, 0b00, 0b00, 0b111, rd.index(), rn.index())
}
/// FMOV Xd, Dn (FP to GP, 64-bit)
pub(crate) fn fmov_gp_from_d(rd: Arm64Reg, rn: Arm64FpReg) -> u32 {
    fp_int_conv(1, 0b01, 0b00, 0b110, rd.index(), rn.index())
}
/// FMOV Dd, Xn (GP to FP, 64-bit)
pub(crate) fn fmov_d_from_gp(rd: Arm64FpReg, rn: Arm64Reg) -> u32 {
    fp_int_conv(1, 0b01, 0b00, 0b111, rd.index(), rn.index())
}
/// FMOV Sd, Sn
pub(crate) fn fmov_s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    fp_data_proc_1src(0b00, 0b000000, rd, rn)
}
/// FMOV Dd, Dn
pub(crate) fn fmov_d(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    fp_data_proc_1src(0b01, 0b000000, rd, rn)
}

// F32 arithmetic
pub(crate) fn fadd_s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    fp_data_proc_2src(0b00, 0b0010, rd, rn, rm)
}
pub(crate) fn fsub_s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    fp_data_proc_2src(0b00, 0b0011, rd, rn, rm)
}
pub(crate) fn fmul_s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    fp_data_proc_2src(0b00, 0b0000, rd, rn, rm)
}
pub(crate) fn fdiv_s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    fp_data_proc_2src(0b00, 0b0001, rd, rn, rm)
}
pub(crate) fn fmin_s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    fp_data_proc_2src(0b00, 0b0101, rd, rn, rm)
}
pub(crate) fn fmax_s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    fp_data_proc_2src(0b00, 0b0100, rd, rn, rm)
}

// F64 arithmetic
pub(crate) fn fadd_d(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    fp_data_proc_2src(0b01, 0b0010, rd, rn, rm)
}
pub(crate) fn fsub_d(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    fp_data_proc_2src(0b01, 0b0011, rd, rn, rm)
}
pub(crate) fn fmul_d(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    fp_data_proc_2src(0b01, 0b0000, rd, rn, rm)
}
pub(crate) fn fdiv_d(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    fp_data_proc_2src(0b01, 0b0001, rd, rn, rm)
}
pub(crate) fn fmin_d(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    fp_data_proc_2src(0b01, 0b0101, rd, rn, rm)
}
pub(crate) fn fmax_d(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    fp_data_proc_2src(0b01, 0b0100, rd, rn, rm)
}

// F32 unary
pub(crate) fn fabs_s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    fp_data_proc_1src(0b00, 0b000001, rd, rn)
}
pub(crate) fn fneg_s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    fp_data_proc_1src(0b00, 0b000010, rd, rn)
}
pub(crate) fn fsqrt_s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    fp_data_proc_1src(0b00, 0b000011, rd, rn)
}
pub(crate) fn frintn_s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    fp_data_proc_1src(0b00, 0b001000, rd, rn)
}
pub(crate) fn frintp_s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    fp_data_proc_1src(0b00, 0b001001, rd, rn)
}
pub(crate) fn frintm_s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    fp_data_proc_1src(0b00, 0b001010, rd, rn)
}
pub(crate) fn frintz_s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    fp_data_proc_1src(0b00, 0b001011, rd, rn)
}

// F64 unary
pub(crate) fn fabs_d(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    fp_data_proc_1src(0b01, 0b000001, rd, rn)
}
pub(crate) fn fneg_d(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    fp_data_proc_1src(0b01, 0b000010, rd, rn)
}
pub(crate) fn fsqrt_d(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    fp_data_proc_1src(0b01, 0b000011, rd, rn)
}
pub(crate) fn frintn_d(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    fp_data_proc_1src(0b01, 0b001000, rd, rn)
}
pub(crate) fn frintp_d(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    fp_data_proc_1src(0b01, 0b001001, rd, rn)
}
pub(crate) fn frintm_d(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    fp_data_proc_1src(0b01, 0b001010, rd, rn)
}
pub(crate) fn frintz_d(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    fp_data_proc_1src(0b01, 0b001011, rd, rn)
}

// FP compare
pub(crate) fn fcmp_s(rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    fp_compare(0b00, rn, rm, 0b00000)
}
pub(crate) fn fcmp_d(rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    fp_compare(0b01, rn, rm, 0b00000)
}
/// FCMP Sn, #0.0
pub(crate) fn fcmp_s_zero(rn: Arm64FpReg) -> u32 {
    fp_compare(0b00, rn, abi::fp_zero_reg(), 0b01000)
}
/// FCMP Dn, #0.0
pub(crate) fn fcmp_d_zero(rn: Arm64FpReg) -> u32 {
    fp_compare(0b01, rn, abi::fp_zero_reg(), 0b01000)
}

// FP conditional select
pub(crate) fn fcsel_s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg, cond: Cond) -> u32 {
    fp_csel(0b00, rd, rn, rm, cond)
}
pub(crate) fn fcsel_d(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg, cond: Cond) -> u32 {
    fp_csel(0b01, rd, rn, rm, cond)
}

// FP conversion between sizes
/// FCVT Dd, Sn (F32 -> F64)
pub(crate) fn fcvt_d_from_s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    fp_data_proc_1src(0b00, 0b000101, rd, rn)
}
/// FCVT Sd, Dn (F64 -> F32)
pub(crate) fn fcvt_s_from_d(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    fp_data_proc_1src(0b01, 0b000100, rd, rn)
}

// FP to integer conversions (truncation toward zero)
/// FCVTZS Wd, Sn
pub(crate) fn fcvtzs_32_s(rd: Arm64Reg, rn: Arm64FpReg) -> u32 {
    fp_int_conv(0, 0b00, 0b11, 0b000, rd.index(), rn.index())
}
/// FCVTZS Xd, Sn
pub(crate) fn fcvtzs_64_s(rd: Arm64Reg, rn: Arm64FpReg) -> u32 {
    fp_int_conv(1, 0b00, 0b11, 0b000, rd.index(), rn.index())
}
/// FCVTZS Wd, Dn
pub(crate) fn fcvtzs_32_d(rd: Arm64Reg, rn: Arm64FpReg) -> u32 {
    fp_int_conv(0, 0b01, 0b11, 0b000, rd.index(), rn.index())
}
/// FCVTZS Xd, Dn
pub(crate) fn fcvtzs_64_d(rd: Arm64Reg, rn: Arm64FpReg) -> u32 {
    fp_int_conv(1, 0b01, 0b11, 0b000, rd.index(), rn.index())
}

/// FCVTZU Wd, Sn
pub(crate) fn fcvtzu_32_s(rd: Arm64Reg, rn: Arm64FpReg) -> u32 {
    fp_int_conv(0, 0b00, 0b11, 0b001, rd.index(), rn.index())
}
/// FCVTZU Xd, Sn
pub(crate) fn fcvtzu_64_s(rd: Arm64Reg, rn: Arm64FpReg) -> u32 {
    fp_int_conv(1, 0b00, 0b11, 0b001, rd.index(), rn.index())
}
/// FCVTZU Wd, Dn
pub(crate) fn fcvtzu_32_d(rd: Arm64Reg, rn: Arm64FpReg) -> u32 {
    fp_int_conv(0, 0b01, 0b11, 0b001, rd.index(), rn.index())
}
/// FCVTZU Xd, Dn
pub(crate) fn fcvtzu_64_d(rd: Arm64Reg, rn: Arm64FpReg) -> u32 {
    fp_int_conv(1, 0b01, 0b11, 0b001, rd.index(), rn.index())
}

// Integer to FP conversions
/// SCVTF Sd, Wn
pub(crate) fn scvtf_s_32(rd: Arm64FpReg, rn: Arm64Reg) -> u32 {
    fp_int_conv(0, 0b00, 0b00, 0b010, rd.index(), rn.index())
}
/// SCVTF Sd, Xn
pub(crate) fn scvtf_s_64(rd: Arm64FpReg, rn: Arm64Reg) -> u32 {
    fp_int_conv(1, 0b00, 0b00, 0b010, rd.index(), rn.index())
}
/// SCVTF Dd, Wn
pub(crate) fn scvtf_d_32(rd: Arm64FpReg, rn: Arm64Reg) -> u32 {
    fp_int_conv(0, 0b01, 0b00, 0b010, rd.index(), rn.index())
}
/// SCVTF Dd, Xn
pub(crate) fn scvtf_d_64(rd: Arm64FpReg, rn: Arm64Reg) -> u32 {
    fp_int_conv(1, 0b01, 0b00, 0b010, rd.index(), rn.index())
}

/// UCVTF Sd, Wn
pub(crate) fn ucvtf_s_32(rd: Arm64FpReg, rn: Arm64Reg) -> u32 {
    fp_int_conv(0, 0b00, 0b00, 0b011, rd.index(), rn.index())
}
/// UCVTF Sd, Xn
pub(crate) fn ucvtf_s_64(rd: Arm64FpReg, rn: Arm64Reg) -> u32 {
    fp_int_conv(1, 0b00, 0b00, 0b011, rd.index(), rn.index())
}
/// UCVTF Dd, Wn
pub(crate) fn ucvtf_d_32(rd: Arm64FpReg, rn: Arm64Reg) -> u32 {
    fp_int_conv(0, 0b01, 0b00, 0b011, rd.index(), rn.index())
}
/// UCVTF Dd, Xn
pub(crate) fn ucvtf_d_64(rd: Arm64FpReg, rn: Arm64Reg) -> u32 {
    fp_int_conv(1, 0b01, 0b00, 0b011, rd.index(), rn.index())
}

// NEON instructions for popcnt
/// FMOV Dd, Xn (same as fmov_d_from_gp but with explicit naming for NEON use)
/// CNT Vd.8B, Vn.8B  (count bits per byte)
pub(crate) fn cnt_8b(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    (0b0_0_001110_00_10000_00101_10 << 10) | (rn.index() << 5) | rd.index()
}
/// ADDV Bd, Vn.8B  (horizontal add of bytes)
pub(crate) fn addv_8b(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    (0b0_0_001110_00_11000_11011_10 << 10) | (rn.index() << 5) | rd.index()
}
/// UMOV Wd, Vn.B[0]  (extract byte 0 to GP)
pub(crate) fn umov_b0(rd: Arm64Reg, rn: Arm64FpReg) -> u32 {
    (0b0_0_001110000 << 21) | (0b00001 << 16) | (0b001_1_1_1 << 10) | (rn.index() << 5) | rd.index()
}

pub(crate) fn and_16b(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4e20_1c00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn add_16b(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4e20_8400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn add_8h(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4e60_8400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn urhadd_16b(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6e20_1400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn urhadd_8h(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6e60_1400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn orr_16b(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4ea0_1c00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn eor_16b(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6e20_1c00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn bic_16b(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4e60_1c00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn mvn_16b(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x6e20_5800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn tbl_16b(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4e00_0000 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn tbl_16b_2(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4e00_2000 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn add_4s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4ea0_8400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn add_2d(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4ee0_8400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn addp_4s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4ea0_bc00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn addp_8h(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4e60_bc00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn sub_16b(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6e20_8400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn sub_8h(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6e60_8400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn sub_4s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6ea0_8400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn sub_2d(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6ee0_8400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn mul_8h(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4e60_9c00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn mul_4s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4ea0_9c00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn neg_16b(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x6e20_b800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn neg_8h(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x6e60_b800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn neg_4s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x6ea0_b800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn neg_2d(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x6ee0_b800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn fadd_4s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4e20_d400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn fadd_2d(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4e60_d400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn fsub_4s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4ea0_d400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn fsub_2d(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4ee0_d400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn fmul_4s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6e20_dc00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn fmul_2d(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6e60_dc00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn fdiv_4s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6e20_fc00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn fdiv_2d(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6e60_fc00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn fneg_4s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x6ea0_f800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn fneg_2d(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x6ee0_f800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn fsqrt_4s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x6ea1_f800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn fsqrt_2d(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x6ee1_f800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn dup_16b_from_gp(rd: Arm64FpReg, rn: Arm64Reg) -> u32 {
    0x4e01_0c00 | (rn.index() << 5) | rd.index()
}

pub(crate) fn dup_8h_from_gp(rd: Arm64FpReg, rn: Arm64Reg) -> u32 {
    0x4e02_0c00 | (rn.index() << 5) | rd.index()
}

pub(crate) fn dup_4s_from_gp(rd: Arm64FpReg, rn: Arm64Reg) -> u32 {
    0x4e04_0c00 | (rn.index() << 5) | rd.index()
}

pub(crate) fn dup_2d_from_gp(rd: Arm64FpReg, rn: Arm64Reg) -> u32 {
    0x4e08_0c00 | (rn.index() << 5) | rd.index()
}

pub(crate) fn dup_4s_from_lane0(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x4e04_0400 | (rn.index() << 5) | rd.index()
}

pub(crate) fn dup_2d_from_lane0(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x4e08_0400 | (rn.index() << 5) | rd.index()
}

pub(crate) fn sshll_8h(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x0f08_a400 | (rn.index() << 5) | rd.index()
}

pub(crate) fn ushll_8h(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x2f08_a400 | (rn.index() << 5) | rd.index()
}

pub(crate) fn sshll_4s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x0f10_a400 | (rn.index() << 5) | rd.index()
}

pub(crate) fn ushll_4s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x2f10_a400 | (rn.index() << 5) | rd.index()
}

pub(crate) fn sshll_2d(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x0f20_a400 | (rn.index() << 5) | rd.index()
}

pub(crate) fn ushll_2d(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x2f20_a400 | (rn.index() << 5) | rd.index()
}

pub(crate) fn sshll2_8h(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x4f08_a400 | (rn.index() << 5) | rd.index()
}

pub(crate) fn ushll2_8h(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x6f08_a400 | (rn.index() << 5) | rd.index()
}

pub(crate) fn sshll2_4s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x4f10_a400 | (rn.index() << 5) | rd.index()
}

pub(crate) fn ushll2_4s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x6f10_a400 | (rn.index() << 5) | rd.index()
}

pub(crate) fn sshll2_2d(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x4f20_a400 | (rn.index() << 5) | rd.index()
}

pub(crate) fn ushll2_2d(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x6f20_a400 | (rn.index() << 5) | rd.index()
}

pub(crate) fn saddlp_8h(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x4e20_2800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn uaddlp_8h(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x6e20_2800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn saddlp_4s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x4e60_2800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn uaddlp_4s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x6e60_2800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn smull_8h(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x0e20_c000 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn smull2_8h(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4e20_c000 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn umull_8h(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x2e20_c000 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn umull2_8h(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6e20_c000 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn smull_4s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x0e60_c000 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn smull2_4s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4e60_c000 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn umull_4s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x2e60_c000 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn umull2_4s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6e60_c000 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn smull_2d(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x0ea0_c000 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn smull2_2d(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4ea0_c000 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn umull_2d(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x2ea0_c000 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn umull2_2d(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6ea0_c000 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn sqrdmulh_8h(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6e60_b400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn sshl_16b(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4e20_4400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn ushl_16b(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6e20_4400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn sshl_8h(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4e60_4400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn ushl_8h(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6e60_4400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn sshl_4s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4ea0_4400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn ushl_4s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6ea0_4400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn sshl_2d(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4ee0_4400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn ushl_2d(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6ee0_4400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn cmeq_zero_16b(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x4e20_9800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn cmeq_zero_8h(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x4e60_9800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn cmeq_zero_4s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x4ea0_9800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn cmeq_16b(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6e20_8c00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn cmeq_8h(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6e60_8c00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn cmeq_4s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6ea0_8c00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn cmeq_2d(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6ee0_8c00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn cmgt_16b(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4e20_3400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn cmgt_8h(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4e60_3400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn cmgt_4s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4ea0_3400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn cmgt_2d(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4ee0_3400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn cmhi_16b(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6e20_3400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn cmhi_8h(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6e60_3400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn cmhi_4s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6ea0_3400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn cmge_16b(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4e20_3c00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn cmge_8h(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4e60_3c00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn cmge_4s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4ea0_3c00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn cmge_2d(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4ee0_3c00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn cmhs_16b(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6e20_3c00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn cmhs_8h(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6e60_3c00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn cmhs_4s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6ea0_3c00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn smin_16b(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4e20_6c00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn smin_8h(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4e60_6c00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn smin_4s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4ea0_6c00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn umin_16b(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6e20_6c00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn umin_8h(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6e60_6c00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn umin_4s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6ea0_6c00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn smax_16b(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4e20_6400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn smax_8h(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4e60_6400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn smax_4s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4ea0_6400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn umax_16b(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6e20_6400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn umax_8h(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6e60_6400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn umax_4s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6ea0_6400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn sqadd_16b(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4e20_0c00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn sqadd_8h(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4e60_0c00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn uqadd_16b(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6e20_0c00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn uqadd_8h(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6e60_0c00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn sqsub_16b(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4e20_2c00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn sqsub_8h(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4e60_2c00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn uqsub_16b(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6e20_2c00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn uqsub_8h(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6e60_2c00 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn abs_16b(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x4e20_b800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn abs_8h(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x4e60_b800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn abs_4s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x4ea0_b800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn abs_2d(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x4ee0_b800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn cnt_16b(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x4e20_5800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn fcmeq_4s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4e20_e400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn fcmeq_2d(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4e60_e400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn fcmgt_4s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6ea0_e400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn fcmgt_2d(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6ee0_e400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn fcmge_4s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6e20_e400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn fcmge_2d(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x6e60_e400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn fmin_4s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4ea0_f400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn fmin_2d(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4ee0_f400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn fmax_4s(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4e20_f400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn fmax_2d(rd: Arm64FpReg, rn: Arm64FpReg, rm: Arm64FpReg) -> u32 {
    0x4e60_f400 | (rm.index() << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn fabs_4s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x4ea0_f800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn fabs_2d(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x4ee0_f800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn frintp_4s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x4ea1_8800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn frintp_2d(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x4ee1_8800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn frintm_4s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x4e21_9800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn frintm_2d(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x4e61_9800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn frintz_4s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x4ea1_9800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn frintz_2d(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x4ee1_9800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn frintn_4s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x4e21_8800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn frintn_2d(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x4e61_8800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn fcvtn_2s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x0e61_6800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn fcvtl_2d(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x0e61_7800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn scvtf_4s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x4e21_d800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn ucvtf_4s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x6e21_d800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn scvtf_2d(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x4e61_d800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn ucvtf_2d(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x6e61_d800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn fcvtzs_4s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x4ea1_b800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn fcvtzu_4s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x6ea1_b800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn fcvtzs_2d(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x4ee1_b800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn fcvtzu_2d(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x6ee1_b800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn sqxtn_2s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x0ea1_4800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn uqxtn_2s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x2ea1_4800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn sqxtn_4h(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x0e61_4800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn sqxtn2_8h(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x4e61_4800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn sqxtn_8b(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x0e21_4800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn sqxtn2_16b(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x4e21_4800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn sqxtun_8b(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x2e21_2800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn sqxtun2_16b(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x6e21_2800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn sqxtun_4h(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x2e61_2800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn sqxtun2_8h(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x6e61_2800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn umaxv_16b(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x6e30_a800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn umaxv_8h(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x6e70_a800 | (rn.index() << 5) | rd.index()
}

pub(crate) fn umaxv_4s(rd: Arm64FpReg, rn: Arm64FpReg) -> u32 {
    0x6eb0_a800 | (rn.index() << 5) | rd.index()
}

#[inline]
const fn simd_lane_imm5(lane: u8, elem_bytes: u8) -> u32 {
    let shift = elem_bytes.trailing_zeros();
    (1u32 << shift) | ((lane as u32) << (shift + 1))
}

pub(crate) fn ins_lane_from_gp(rd: Arm64FpReg, lane: u8, rn: Arm64Reg, elem_bytes: u8) -> u32 {
    0x4e00_1c00 | (simd_lane_imm5(lane, elem_bytes) << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn umov_lane_to_gp(rd: Arm64Reg, rn: Arm64FpReg, lane: u8, elem_bytes: u8) -> u32 {
    let base = if elem_bytes == 8 {
        0x4e08_3c00
    } else {
        0x0e00_3c00
    };
    base | (simd_lane_imm5(lane, elem_bytes) << 16) | (rn.index() << 5) | rd.index()
}

pub(crate) fn smov_lane_to_gp(rd: Arm64Reg, rn: Arm64FpReg, lane: u8, elem_bytes: u8) -> u32 {
    debug_assert!(matches!(elem_bytes, 1 | 2));
    0x0e00_2c00 | (simd_lane_imm5(lane, elem_bytes) << 16) | (rn.index() << 5) | rd.index()
}
