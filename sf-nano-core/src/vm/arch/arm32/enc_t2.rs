//! ARMv7-M / ARMv8-M (Thumb-2) instruction encoding.
//!
//! API-compatible with `enc_a32.rs`: mirrors every public function so the
//! rest of the arm32 backend (backend / control / preserved / select / inst /
//! compile) compiles unchanged when `sf_arm32_isa_thumb` is active. Encodings
//! that haven't been implemented yet `todo!()` — the module builds, but any
//! attempt to emit an unimplemented Thumb-2 instruction panics with a
//! descriptive message identifying which one is missing. Callers fill these
//! in one-at-a-time, driven by the smallest failing test.
//!
//! # Emission convention
//!
//! Each encoder returns a `u32` that, when passed to the existing
//! `TextEmitter::emit_u32` (little-endian write), produces the correct
//! Thumb-2 in-memory byte layout. Details:
//!
//! * **32-bit Thumb-2 instructions** are encoded as two halfwords stored in
//!   ascending-address order: halfword 0 (first in memory) holds logical bits
//!   [31:16] of the instruction, halfword 1 (second in memory) holds logical
//!   bits [15:0]. Each halfword is itself little-endian. The encoder packs
//!   them as `(halfword1 << 16) | halfword0` so that `emit_u32` writes bytes
//!   [h0_lo, h0_hi, h1_lo, h1_hi] — matching memory. Use [`thumb32`] to
//!   construct this.
//!
//! * **16-bit Thumb instructions** are padded to a 4-byte slot by appending a
//!   16-bit Thumb NOP (`0xBF00`) as halfword 1. Execution flow runs through
//!   the real instruction; if it falls through, the NOP is a no-op; if it
//!   branches away (BX, B, BL), the NOP is unreachable pad. Uniform 4-byte
//!   slots simplify patching (MOVW/MOVT at fixed offsets) and offset
//!   arithmetic in the backend's patch/fixup tables. Use [`thumb16_pad`] for
//!   this. `TextEmitter::emit_u16` is available if a future caller wants
//!   sub-word slots, but the default discipline is 4-byte per instruction.
//!
//! * **Conditional execution.** A32 encodes a 4-bit condition in every
//!   instruction (bits [31:28]). Thumb-2 has no such field; conditional
//!   execution is expressed via an `IT` prefix block covering 1-4 trailing
//!   instructions, or via the native conditional branch `B<cc>`. Signatures
//!   that take a `Cond` retain it for cross-encoder API compatibility; the
//!   implementation emits `IT <cond>` + the unconditional form where needed.
//!   On ARMv8-M the `IT` block is restricted to a single trailing instruction
//!   (ARM deprecated multi-instruction IT in v8-M mainline) — encoders must
//!   respect this limit when we get to them.
//!
//! * **MOVW/MOVT patching.** The shared patcher in `compile.rs` reads back
//!   bytes at the patch offset as `u32::from_le_bytes`, extracts the Rd
//!   field, and re-emits with the new immediate. Under this encoding
//!   convention the Rd field of a T3 MOVW/MOVT sits at bits [27:24] of the
//!   returned `u32` (halfword1 bits [11:8] after the halfword swap), *not*
//!   at bits [15:12] as for A32. The patcher currently assumes A32 layout —
//!   when MOVW is filled in here, the patcher needs a cfg branch to read
//!   from bits [27:24] instead.

use super::reg::Arm32Reg;

/// Pack two halfwords into the `u32` form expected by `emit_u32`, placing
/// `h0` at the lower memory address (first halfword) and `h1` at the higher
/// memory address (second halfword). See module doc for why this is inverted
/// relative to the logical [h0 h1] reading.
#[inline]
const fn thumb32(h0: u16, h1: u16) -> u32 {
    ((h1 as u32) << 16) | (h0 as u32)
}

/// Pack a 16-bit Thumb instruction into a 4-byte slot by appending a 16-bit
/// Thumb NOP (`0xBF00`) after it. The real instruction is at the lower
/// address and executes first; the NOP serves as pad to keep emit slots
/// word-aligned.
#[inline]
const fn thumb16_pad(instr: u16) -> u32 {
    thumb32(instr, 0xBF00)
}

/// Convert an A32-style 32-bit instruction encoding into the halfword-swapped
/// form expected by `emit_u32` for Thumb-2. Valid only when the A32 encoding
/// has cond=AL (top nibble = 1110) *and* the corresponding Thumb-2 32-bit
/// encoding has bits [31:28] = 1110 — which holds for all VFP / coprocessor
/// access instructions on ARMv7 and later. For these, the bit pattern is
/// identical between the two ISAs; only the byte layout in memory differs
/// (halfword order), hence the simple swap.
#[inline]
const fn a32_to_thumb32(a32_inst: u32) -> u32 {
    (a32_inst >> 16) | (a32_inst << 16)
}

/// Thumb-2 "Data processing (shifted register)" T2/T3 encoding skeleton,
/// no shift applied (LSL #0). Used by the plain reg-reg flavors of ADD/SUB/
/// AND/ORR/EOR/etc. The 4-bit opcode goes into instruction bits [24:21],
/// which are bits [8:5] of halfword 0. The optional S bit goes at
/// instruction bit 20 = halfword 0 bit 4. Halfword 1 sets imm3=0, imm2=0,
/// type=00 (LSL), leaving `(Rd << 8) | Rm`.
#[inline]
const fn thumb32_dp_reg(opcode4: u16, s: bool, rd: u16, rn: u16, rm: u16) -> u32 {
    let s_bit: u16 = if s { 1 } else { 0 };
    // Halfword 0 base = 1110_1010_0000_0000 = 0xEA00 (opcode=0, S=0, Rn=0).
    let h0 = 0xEA00u16 | (opcode4 << 5) | (s_bit << 4) | rn;
    // Halfword 1 = (Rd << 8) | Rm (no shift).
    let h1 = (rd << 8) | rm;
    thumb32(h0, h1)
}

/// Thumb-2 "Data processing (modified immediate)" T2 encoding skeleton.
/// Bring-up limitation: the immediate is assumed to already fit in 8 bits
/// (0..=255), which is the range the A32 callers use when passing
/// `(imm8, rotate=0)` directly. The full Thumb-2 modified-immediate encoder
/// (rotated 8-bit, or byte-replicated 32-bit patterns) is not yet written;
/// callers needing values outside 0..=255 already fall back to MOVW/MOVT
/// because `encode_arm_imm` returns `None` here.
fn thumb32_dp_imm_raw(opcode4: u16, s: bool, rd: u16, rn: u16, value: u32) -> u32 {
    let imm12 = encode_thumb_modified_imm(value).unwrap_or_else(|| {
        panic!(
            "thumb2 dp_imm value {value:#x} does not fit Thumb-2 modified-immediate; \
            caller should have gone through MOVW/MOVT via encode_arm_imm's `None` return"
        )
    });
    let s_bit: u16 = if s { 1 } else { 0 };
    // Split the 12-bit modified-immediate across the encoding's three fields:
    // i = imm12[11] → halfword0 bit 10; imm3 = imm12[10:8] → halfword1 bits
    // [14:12]; imm8 = imm12[7:0] → halfword1 bits [7:0].
    let i = ((imm12 >> 11) & 0x1) as u16;
    let imm3 = ((imm12 >> 8) & 0x7) as u16;
    let imm8 = (imm12 & 0xFF) as u16;
    let h0 = 0xF000u16 | (i << 10) | (opcode4 << 5) | (s_bit << 4) | rn;
    let h1 = (imm3 << 12) | (rd << 8) | imm8;
    thumb32(h0, h1)
}

/// Encode `value` as a 12-bit Thumb-2 modified immediate (ThumbExpandImm).
/// Returns the imm12 bit pattern, or `None` if the value isn't representable.
///
/// Four zero-padded byte-replicated forms live in imm12[11:10] == 0b00; the
/// rotated 8-bit form (imm12[11:10] != 0b00) gets a value with an implicit
/// MSB of 1 rotated right by `imm12[11:7]` in 8..=31.
fn encode_thumb_modified_imm(value: u32) -> Option<u32> {
    // Zero-padded: imm12[11:8] = 0000, imm12[7:0] = value (if value <= 0xFF).
    if value <= 0xFF {
        return Some(value);
    }
    // 0x00XY00XY: imm12[11:8] = 0001.
    let b0 = value & 0xFF;
    if value == (b0 | (b0 << 16)) && (value >> 8) & 0xFF == 0 {
        return Some((0b0001 << 8) | b0);
    }
    // 0xXY00XY00: imm12[11:8] = 0010.
    let b1 = (value >> 8) & 0xFF;
    if value == ((b1 << 8) | (b1 << 24)) && b1 != 0 {
        return Some((0b0010 << 8) | b1);
    }
    // 0xXYXYXYXY: imm12[11:8] = 0011.
    if b0 != 0 && value == (b0 | (b0 << 8) | (b0 << 16) | (b0 << 24)) {
        return Some((0b0011 << 8) | b0);
    }
    // Rotated 8-bit with implicit MSB=1: find a shift in 8..=31 such that
    // ROR(0x80 | low7, shift) == value. Equivalently, rotate `value` LEFT
    // by `shift` bits and check whether the low byte has bit 7 set and the
    // upper 24 bits are zero.
    for shift in 8..=31u32 {
        let rotated = value.rotate_left(shift);
        // After the left-rotation, the 8-bit pattern occupies the low byte
        // with bit 7 = implicit MSB of the 8-bit seed.
        if (rotated & 0xFFFF_FF00) == 0 && (rotated & 0x80) != 0 {
            let low7 = rotated & 0x7F;
            // imm12[11:7] = shift, imm12[6:0] = low7.
            return Some((shift << 7) | low7);
        }
    }
    None
}

/// Shifted variant of [`thumb32_dp_reg`]. The 5-bit shift amount is split
/// as `imm3:imm2`, with `imm3` in halfword 1 bits [14:12] and `imm2` in
/// bits [7:6]. `shift_type` occupies bits [5:4] (00=LSL, 01=LSR, 10=ASR,
/// 11=ROR) — matching the `shift_type` the arm32 callers already pass.
fn thumb32_dp_reg_shifted(
    opcode4: u16,
    s: bool,
    dst: Arm32Reg,
    lhs: Arm32Reg,
    rhs: Arm32Reg,
    shift_type: u32,
    shift_imm: u32,
) -> u32 {
    debug_assert!(shift_imm < 32, "thumb32 shifted: imm5 out of range");
    debug_assert!(shift_type < 4, "thumb32 shifted: type out of range");
    let rd = dst.idx() as u16;
    let rn = lhs.idx() as u16;
    let rm = rhs.idx() as u16;
    let s_bit: u16 = if s { 1 } else { 0 };
    let imm3 = ((shift_imm >> 2) & 0x7) as u16;
    let imm2 = (shift_imm & 0x3) as u16;
    let ty = (shift_type & 0x3) as u16;
    let h0 = 0xEA00u16 | (opcode4 << 5) | (s_bit << 4) | rn;
    let h1 = (imm3 << 12) | (rd << 8) | (imm2 << 6) | (ty << 4) | rm;
    thumb32(h0, h1)
}

/// Condition codes. Same values as A32 per the ARM ARM; the enum is redeclared
/// here (rather than imported from `enc_a32`) because only one of the two
/// encoder modules compiles at a time under `#[cfg(sf_arm32_isa_thumb)]` /
/// `#[cfg(not(...))]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(super) enum Cond {
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
    Al = 0b1110, // always (Thumb-2 unconditional is the default; Al is a no-op)
}

impl Cond {
    #[inline]
    pub(super) const fn invert(self) -> Cond {
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

/// Find an A32-style `(imm8, rotate)` tuple encoding `value`, or `None`.
///
/// Returns a pair compatible with the A32 modified-immediate convention:
/// `value == imm8.rotate_right(2 * rotate)` with `imm8` in 0..=255 and
/// `rotate` in 0..=15. Every Thumb-2 DP-imm consumer in this module
/// reconstructs the logical value via that same identity before
/// re-encoding into Thumb-2's modified-immediate form.
///
/// The A32 rotated-8-bit scheme is a strict subset of what Thumb-2's
/// modified immediate can represent (Thumb-2 additionally has four byte-
/// replicated patterns like `0xXYXYXYXY`), so any value accepted here
/// is guaranteed encodable on Thumb-2. Values that don't fit the rotated
/// 8-bit form return `None` and callers fall back to MOVW/MOVT. The
/// extra Thumb-2-only patterns aren't a must-have for correctness.
pub(super) fn encode_arm_imm(value: u32) -> Option<(u32, u32)> {
    // Thumb-2's modified-immediate accepts a strict *subset* of A32's
    // rotated-8-bit values: it requires MSB=1 for the non-byte-replicated
    // rotated form, and only rotations 8..=31 (not arbitrary even amounts).
    // A value like 0x0007_F000 is A32-encodable (imm8=0x7F, rot=10) but not
    // Thumb-2-encodable — so reject here and let the caller fall back to
    // MOVW/MOVT. The return tuple is still A32-shaped so every Thumb-2
    // consumer in this module can reconstruct `value` via `imm8.rotate_right
    // (2 * rot)` and re-encode via the modified-immediate path.
    if encode_thumb_modified_imm(value).is_none() {
        return None;
    }
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

// `cond_bits` was part of the A32 API but is meaningless on Thumb-2
// (no per-instruction condition field). The one external caller
// (`select.rs::rbit`) now routes through `enc::rbit` which each ISA
// implements itself, so no Thumb-2 `cond_bits` is needed.

// ─── Arithmetic ─────────────────────────────────────────────────────────────

/// `ADD.W Rd, Rn, Rm` — T3 encoding (no shift, no flag set).
pub(super) fn add_reg(dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    thumb32_dp_reg(
        0b1000,
        false,
        dst.idx() as u16,
        lhs.idx() as u16,
        rhs.idx() as u16,
    )
}

// `add_reg_lsl_imm` only has an A32 caller (the PC-relative br_table
// dispatch in `control.rs`, which is cfg-gated out on Thumb-2 builds in
// favor of a `cmp + b.eq` chain). The Thumb-2 equivalent would live here
// if ever needed; omitted to keep this module warning-clean.

/// `ADDS.W Rd, Rn, Rm` — T3 encoding with S=1 (sets NZCV).
pub(super) fn adds_reg(dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    thumb32_dp_reg(
        0b1000,
        true,
        dst.idx() as u16,
        lhs.idx() as u16,
        rhs.idx() as u16,
    )
}

/// `ADC.W Rd, Rn, Rm` — add with carry. Opcode 1010.
pub(super) fn adc_reg(dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    thumb32_dp_reg(
        0b1010,
        false,
        dst.idx() as u16,
        lhs.idx() as u16,
        rhs.idx() as u16,
    )
}

/// `ADD{W}.W Rd, Rn, #imm` — T4 for values 0..=4095, T3 modified-immediate
/// fallback for larger values. Mirror of [`sub_imm`].
pub(super) fn add_imm(dst: Arm32Reg, src: Arm32Reg, imm8: u32, rotate: u32) -> u32 {
    let value = (imm8 & 0xFF).rotate_right(2 * rotate);
    let rn = src.idx() as u16;
    let rd = dst.idx() as u16;
    debug_assert!(rn != 15, "ADDW forbids PC as Rn");
    debug_assert!(rd != 15, "ADDW forbids PC as Rd");
    if value <= 0xFFF {
        let i = ((value >> 11) & 0x1) as u16;
        let imm3 = ((value >> 8) & 0x7) as u16;
        let imm8_lo = (value & 0xFF) as u16;
        let h0 = 0xF200u16 | (i << 10) | rn;
        let h1 = (imm3 << 12) | (rd << 8) | imm8_lo;
        thumb32(h0, h1)
    } else {
        // T3 ADD.W: modified-immediate (opcode 1000).
        thumb32_dp_imm_raw(0b1000, false, rd, rn, value)
    }
}

/// `SUB.W Rd, Rn, Rm` — T2 encoding. Opcode 1101.
pub(super) fn sub_reg(dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    thumb32_dp_reg(
        0b1101,
        false,
        dst.idx() as u16,
        lhs.idx() as u16,
        rhs.idx() as u16,
    )
}

/// `SUBS.W Rd, Rn, Rm` — T2 encoding, sets flags.
pub(super) fn subs_reg(dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    thumb32_dp_reg(
        0b1101,
        true,
        dst.idx() as u16,
        lhs.idx() as u16,
        rhs.idx() as u16,
    )
}

/// `SBC.W Rd, Rn, Rm` — subtract with carry. Opcode 1011.
pub(super) fn sbc_reg(dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    thumb32_dp_reg(
        0b1011,
        false,
        dst.idx() as u16,
        lhs.idx() as u16,
        rhs.idx() as u16,
    )
}

/// `SUB{W}.W Rd, Rn, #imm` — T4 for values 0..=4095, T3 (modified-imm) for
/// larger values that fit Thumb-2's ThumbExpandImm scheme.
///
/// Accepts the A32-style `(imm8, rotate)` pair for API compatibility.
/// Callers that need values outside the combined T3/T4 ranges already go
/// through `encode_arm_imm` (which returns `None` and triggers MOVW/MOVT
/// fallback), so this fn always has a valid encoding to reach for.
pub(super) fn sub_imm(dst: Arm32Reg, src: Arm32Reg, imm8: u32, rotate: u32) -> u32 {
    let value = (imm8 & 0xFF).rotate_right(2 * rotate);
    let rn = src.idx() as u16;
    let rd = dst.idx() as u16;
    debug_assert!(rn != 15, "SUBW forbids PC as Rn");
    debug_assert!(rd != 15, "SUBW forbids PC as Rd");
    if value <= 0xFFF {
        // T4 SUBW: flat 12-bit unsigned immediate, no rotation games.
        let i = ((value >> 11) & 0x1) as u16;
        let imm3 = ((value >> 8) & 0x7) as u16;
        let imm8_lo = (value & 0xFF) as u16;
        let h0 = 0xF2A0u16 | (i << 10) | rn;
        let h1 = (imm3 << 12) | (rd << 8) | imm8_lo;
        thumb32(h0, h1)
    } else {
        // T3 SUB.W: modified-immediate (opcode 1101).
        thumb32_dp_imm_raw(0b1101, false, rd, rn, value)
    }
}

/// `RSB Rd, Rn, #imm` — T2 encoding. Thumb-2 DP-imm opcode 0b1110.
pub(super) fn rsb_imm(dst: Arm32Reg, src: Arm32Reg, imm8: u32, rotate: u32) -> u32 {
    let value = (imm8 & 0xFF).rotate_right(2 * rotate);
    thumb32_dp_imm_raw(0b1110, false, dst.idx() as u16, src.idx() as u16, value)
}

// ─── Logic ──────────────────────────────────────────────────────────────────

/// `AND.W Rd, Rn, Rm` — T2 encoding. Opcode 0000.
pub(super) fn and_reg(dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    thumb32_dp_reg(
        0b0000,
        false,
        dst.idx() as u16,
        lhs.idx() as u16,
        rhs.idx() as u16,
    )
}

/// `AND.W Rd, Rn, #imm` — T1 encoding. Opcode 0000.
pub(super) fn and_imm(dst: Arm32Reg, src: Arm32Reg, imm8: u32, rotate: u32) -> u32 {
    let value = (imm8 & 0xFF).rotate_right(2 * rotate);
    thumb32_dp_imm_raw(0b0000, false, dst.idx() as u16, src.idx() as u16, value)
}

/// `ORR.W Rd, Rn, Rm` — T2 encoding. Opcode 0010.
pub(super) fn orr_reg(dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    thumb32_dp_reg(
        0b0010,
        false,
        dst.idx() as u16,
        lhs.idx() as u16,
        rhs.idx() as u16,
    )
}

/// `ORR.W Rd, Rn, #imm` — T1 encoding. Opcode 0010.
pub(super) fn orr_imm(dst: Arm32Reg, src: Arm32Reg, imm8: u32, rotate: u32) -> u32 {
    let value = (imm8 & 0xFF).rotate_right(2 * rotate);
    thumb32_dp_imm_raw(0b0010, false, dst.idx() as u16, src.idx() as u16, value)
}

/// `EOR.W Rd, Rn, Rm` — T2 encoding. Opcode 0100.
pub(super) fn eor_reg(dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    thumb32_dp_reg(
        0b0100,
        false,
        dst.idx() as u16,
        lhs.idx() as u16,
        rhs.idx() as u16,
    )
}

/// `EOR.W Rd, Rn, #imm` — T1 encoding. Opcode 0100.
pub(super) fn eor_imm(dst: Arm32Reg, src: Arm32Reg, imm8: u32, rotate: u32) -> u32 {
    let value = (imm8 & 0xFF).rotate_right(2 * rotate);
    thumb32_dp_imm_raw(0b0100, false, dst.idx() as u16, src.idx() as u16, value)
}

/// `BIC.W Rd, Rn, #imm` — T1 encoding. Opcode 0001.
pub(super) fn bic_imm(dst: Arm32Reg, lhs: Arm32Reg, imm8: u32, rotate: u32) -> u32 {
    let value = (imm8 & 0xFF).rotate_right(2 * rotate);
    thumb32_dp_imm_raw(0b0001, false, dst.idx() as u16, lhs.idx() as u16, value)
}

// ─── Move ───────────────────────────────────────────────────────────────────

/// `MOV Rd, Rm` — T1 encoding (16-bit "high register" move).
///
/// Covers all of R0..R15 as both source and destination and does *not* set
/// flags. Preferred over T3 MOV.W for a plain register copy because it's
/// half the size and expresses intent cleanly. Padded to a 4-byte slot with
/// a trailing Thumb NOP per the module-level convention.
pub(super) fn mov_reg(dst: Arm32Reg, src: Arm32Reg) -> u32 {
    let rd = dst.idx() as u16;
    let rm = src.idx() as u16;
    // Halfword: 01000110 | D | Rm[3:0] | Rd[2:0]
    //   D = Rd[3], with Rd[2:0] in the low 3 bits.
    let d = (rd >> 3) & 1;
    let rd_lo = rd & 0x7;
    let instr: u16 = 0x4600 | (d << 7) | ((rm & 0xF) << 3) | rd_lo;
    thumb16_pad(instr)
}

// Conditional MOV reg→reg is emitted via `inst::emit_mov_reg_cond_into`
// (an `IT <cond>` + 16-bit unconditional MOV pair) rather than a single
// encoder, since Thumb-2 has no per-instruction condition field. No
// standalone `mov_reg_cond` encoder is needed on this path.

/// `MOV Rd, #imm` — T2 encoding (modified-immediate). Opcode 0010, S=0,
/// Rn=0xF (the DP-imm "Rn" slot is reserved for MOV.W).
pub(super) fn mov_imm(dst: Arm32Reg, imm8: u32, rotate: u32) -> u32 {
    let value = (imm8 & 0xFF).rotate_right(2 * rotate);
    thumb32_dp_imm_raw(0b0010, false, dst.idx() as u16, 0xF, value)
}

/// `MOVW Rd, #imm16` — T3 encoding.
///
/// The 16-bit immediate is split as `imm4 : i : imm3 : imm8` (MSB first),
/// going into halfword0 bits [3:0], halfword0 bit 10, halfword1 bits [14:12],
/// and halfword1 bits [7:0] respectively. Rd sits in halfword1 bits [11:8].
/// Rd cannot be PC (R15) or SP (R13).
pub(super) fn movw(dst: Arm32Reg, imm16: u16) -> u32 {
    let rd = dst.idx() as u16;
    debug_assert!(rd != 15, "MOVW T3 forbids PC");
    debug_assert!(rd != 13, "MOVW T3 forbids SP");
    let imm4 = (imm16 >> 12) & 0xF;
    let i = (imm16 >> 11) & 0x1;
    let imm3 = (imm16 >> 8) & 0x7;
    let imm8 = imm16 & 0xFF;
    // Halfword 0: 1111_0 | i | 10_0100 | imm4   (MOVW opcode = 100100)
    let h0 = 0xF240u16 | (i << 10) | imm4;
    // Halfword 1: 0 | imm3 | Rd | imm8
    let h1 = (imm3 << 12) | (rd << 8) | imm8;
    thumb32(h0, h1)
}

/// `MOVT Rd, #imm16` — T1 encoding. Same shape as [`movw`]; differs only in
/// bit 23 (halfword0 bit 7) which flips from 0 to 1 — so halfword0's fixed
/// pattern becomes 0xF2C0 instead of 0xF240.
pub(super) fn movt(dst: Arm32Reg, imm16: u16) -> u32 {
    let rd = dst.idx() as u16;
    debug_assert!(rd != 15, "MOVT T1 forbids PC");
    debug_assert!(rd != 13, "MOVT T1 forbids SP");
    let imm4 = (imm16 >> 12) & 0xF;
    let i = (imm16 >> 11) & 0x1;
    let imm3 = (imm16 >> 8) & 0x7;
    let imm8 = imm16 & 0xFF;
    let h0 = 0xF2C0u16 | (i << 10) | imm4;
    let h1 = (imm3 << 12) | (rd << 8) | imm8;
    thumb32(h0, h1)
}

// ─── Shifts ─────────────────────────────────────────────────────────────────

/// Shift-by-register (LSL/LSR/ASR/ROR). Thumb-2 "Shift (register)" family.
#[inline]
fn encode_t2_shift_reg(type_bits: u16, dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    let rd = dst.idx() as u16;
    let rn = lhs.idx() as u16;
    let rm = rhs.idx() as u16;
    // h0: 1111_1010_0TT_S_Rn  (TT = type, S = 0 here)
    let h0 = 0xFA00u16 | (type_bits << 5) | rn;
    // h1: 1111 | Rd | 0000 | Rm
    let h1 = 0xF000u16 | (rd << 8) | rm;
    thumb32(h0, h1)
}

pub(super) fn lsl_reg(dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    encode_t2_shift_reg(0b00, dst, lhs, rhs)
}
pub(super) fn lsr_reg(dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    encode_t2_shift_reg(0b01, dst, lhs, rhs)
}
pub(super) fn asr_reg(dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    encode_t2_shift_reg(0b10, dst, lhs, rhs)
}
pub(super) fn ror_reg(dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    encode_t2_shift_reg(0b11, dst, lhs, rhs)
}

/// Shift-by-immediate. These are `MOV.W Rd, Rm, <shift> #imm5` — the plain
/// MOV-register form with a shift, which Thumb-2 encodes via the DP-reg
/// class with opcode=ORR (0b0010) and Rn=0xF.
pub(super) fn lsl_imm(dst: Arm32Reg, src: Arm32Reg, shift: u32) -> u32 {
    if shift == 0 {
        return mov_reg(dst, src);
    }
    thumb32_dp_reg_shifted(0b0010, false, dst, Arm32Reg::R15, src, 0b00, shift)
}
pub(super) fn lsr_imm(dst: Arm32Reg, src: Arm32Reg, shift: u32) -> u32 {
    thumb32_dp_reg_shifted(0b0010, false, dst, Arm32Reg::R15, src, 0b01, shift)
}
pub(super) fn asr_imm(dst: Arm32Reg, src: Arm32Reg, shift: u32) -> u32 {
    thumb32_dp_reg_shifted(0b0010, false, dst, Arm32Reg::R15, src, 0b10, shift)
}
pub(super) fn ror_imm(dst: Arm32Reg, src: Arm32Reg, shift: u32) -> u32 {
    thumb32_dp_reg_shifted(0b0010, false, dst, Arm32Reg::R15, src, 0b11, shift)
}

// ─── Compare / Test ─────────────────────────────────────────────────────────

/// `CMP Rn, Rm` — T2 encoding (16-bit, any registers R0-R15).
///
/// Layout: `01000101 | N | Rm[3:0] | Rn[2:0]`, where N is Rn's high bit.
/// Sets flags; no destination.
pub(super) fn cmp_reg(lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    let rn = lhs.idx() as u16;
    let rm = rhs.idx() as u16;
    let n = (rn >> 3) & 1;
    let rn_lo = rn & 0x7;
    let instr: u16 = 0x4500 | (n << 7) | ((rm & 0xF) << 3) | rn_lo;
    thumb16_pad(instr)
}

/// `CMP Rn, #imm` — T2 encoding (modified-immediate). Opcode 1101, S=1, Rd=0xF.
pub(super) fn cmp_imm(src: Arm32Reg, imm8: u32, rotate: u32) -> u32 {
    let value = (imm8 & 0xFF).rotate_right(2 * rotate);
    thumb32_dp_imm_raw(0b1101, true, 0xF, src.idx() as u16, value)
}

/// `CMN Rn, #imm` — T2 encoding. Opcode 1000, S=1, Rd=0xF.
pub(super) fn cmn_imm(lhs: Arm32Reg, imm8: u32, rotate: u32) -> u32 {
    let value = (imm8 & 0xFF).rotate_right(2 * rotate);
    thumb32_dp_imm_raw(0b1000, true, 0xF, lhs.idx() as u16, value)
}

/// `TST.W Rn, Rm` — T2. Same shape as AND-register with S=1 and Rd=0xF.
pub(super) fn tst_reg(lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    thumb32_dp_reg(0b0000, true, 0xF, lhs.idx() as u16, rhs.idx() as u16)
}

/// `TST Rn, #imm` — T1 encoding. Opcode 0000 (AND), S=1, Rd=0xF (no result).
pub(super) fn tst_imm(src: Arm32Reg, imm8: u32, rotate: u32) -> u32 {
    let value = (imm8 & 0xFF).rotate_right(2 * rotate);
    thumb32_dp_imm_raw(0b0000, true, 0xF, src.idx() as u16, value)
}

// ─── Bitfield extract ───────────────────────────────────────────────────────

/// `UBFX Rd, Rn, #lsb, #width` — T1 encoding. 5-bit lsb (0..=31), width 1..=32.
pub(super) fn ubfx(dst: Arm32Reg, src: Arm32Reg, lsb: u32, width: u32) -> u32 {
    debug_assert!(width > 0 && lsb + width <= 32);
    let rd = dst.idx() as u16;
    let rn = src.idx() as u16;
    let imm3 = ((lsb >> 2) & 0x7) as u16;
    let imm2 = (lsb & 0x3) as u16;
    let widthm1 = ((width - 1) & 0x1F) as u16;
    let h0 = 0xF3C0u16 | rn;
    let h1 = (imm3 << 12) | (rd << 8) | (imm2 << 6) | widthm1;
    thumb32(h0, h1)
}

// ─── Shifted-register data processing ───────────────────────────────────────

/// `ADD.W Rd, Rn, Rm, <shift> #imm5` — generic shifted-register ADD.
pub(super) fn add_reg_shifted(
    dst: Arm32Reg,
    lhs: Arm32Reg,
    rhs: Arm32Reg,
    shift_type: u32,
    shift_imm: u32,
) -> u32 {
    thumb32_dp_reg_shifted(0b1000, false, dst, lhs, rhs, shift_type, shift_imm)
}

pub(super) fn sub_reg_shifted(
    dst: Arm32Reg,
    lhs: Arm32Reg,
    rhs: Arm32Reg,
    shift_type: u32,
    shift_imm: u32,
) -> u32 {
    thumb32_dp_reg_shifted(0b1101, false, dst, lhs, rhs, shift_type, shift_imm)
}

pub(super) fn and_reg_shifted(
    dst: Arm32Reg,
    lhs: Arm32Reg,
    rhs: Arm32Reg,
    shift_type: u32,
    shift_imm: u32,
) -> u32 {
    thumb32_dp_reg_shifted(0b0000, false, dst, lhs, rhs, shift_type, shift_imm)
}

pub(super) fn orr_reg_shifted(
    dst: Arm32Reg,
    lhs: Arm32Reg,
    rhs: Arm32Reg,
    shift_type: u32,
    shift_imm: u32,
) -> u32 {
    thumb32_dp_reg_shifted(0b0010, false, dst, lhs, rhs, shift_type, shift_imm)
}

pub(super) fn eor_reg_shifted(
    dst: Arm32Reg,
    lhs: Arm32Reg,
    rhs: Arm32Reg,
    shift_type: u32,
    shift_imm: u32,
) -> u32 {
    thumb32_dp_reg_shifted(0b0100, false, dst, lhs, rhs, shift_type, shift_imm)
}

// ─── Multiply ───────────────────────────────────────────────────────────────

/// `MUL Rd, Rn, Rm` — T2 encoding. Low 32 bits of Rn*Rm. No flag set.
pub(super) fn mul(dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    let rd = dst.idx() as u16;
    let rn = lhs.idx() as u16;
    let rm = rhs.idx() as u16;
    let h0 = 0xFB00u16 | rn;
    let h1 = 0xF000u16 | (rd << 8) | rm;
    thumb32(h0, h1)
}

/// `MLS Rd, Rn, Rm, Ra` — T1. Rd = Ra - (Rn * Rm).
pub(super) fn mls(dst: Arm32Reg, mul_lhs: Arm32Reg, mul_rhs: Arm32Reg, acc: Arm32Reg) -> u32 {
    let rd = dst.idx() as u16;
    let rn = mul_lhs.idx() as u16;
    let rm = mul_rhs.idx() as u16;
    let ra = acc.idx() as u16;
    let h0 = 0xFB00u16 | rn;
    // h1: Ra | Rd | 0001 | Rm
    let h1 = (ra << 12) | (rd << 8) | (0x1u16 << 4) | rm;
    thumb32(h0, h1)
}

// ─── Divide ─────────────────────────────────────────────────────────────────

/// `SDIV Rd, Rn, Rm` — T1.
pub(super) fn sdiv(dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    let rd = dst.idx() as u16;
    let rn = lhs.idx() as u16;
    let rm = rhs.idx() as u16;
    let h0 = 0xFB90u16 | rn;
    let h1 = 0xF0F0u16 | (rd << 8) | rm;
    thumb32(h0, h1)
}

/// `UDIV Rd, Rn, Rm` — T1.
pub(super) fn udiv(dst: Arm32Reg, lhs: Arm32Reg, rhs: Arm32Reg) -> u32 {
    let rd = dst.idx() as u16;
    let rn = lhs.idx() as u16;
    let rm = rhs.idx() as u16;
    let h0 = 0xFBB0u16 | rn;
    let h1 = 0xF0F0u16 | (rd << 8) | rm;
    thumb32(h0, h1)
}

// ─── CLZ ────────────────────────────────────────────────────────────────────

/// `CLZ Rd, Rm` — T1. Count leading zeros in Rm, result into Rd.
/// Encoding embeds Rm in both the halfword-0 low 4 bits *and* halfword-1
/// low 4 bits (this is how the ARM encoding works — not a typo).
pub(super) fn clz(dst: Arm32Reg, src: Arm32Reg) -> u32 {
    let rd = dst.idx() as u16;
    let rm = src.idx() as u16;
    let h0 = 0xFAB0u16 | rm;
    let h1 = 0xF080u16 | (rd << 8) | rm;
    thumb32(h0, h1)
}

/// `RBIT Rd, Rm` — T1. Reverse bit order.
/// Encoding: `1111_1010_1001_Rm | 1111_Rd_1010_Rm`.
pub(super) fn rbit(dst: Arm32Reg, src: Arm32Reg) -> u32 {
    let rd = dst.idx() as u16;
    let rm = src.idx() as u16;
    let h0 = 0xFA90u16 | rm;
    let h1 = 0xF0A0u16 | (rd << 8) | rm;
    thumb32(h0, h1)
}

// ─── Load / Store (immediate offset) ────────────────────────────────────────

/// `LDR Rt, [Rn, #offset]` — T3 for positive offset (low 12 bits), T4 for
/// -255..=0. Matches A32's silent-truncate-to-12-bits behavior for large
/// positive offsets (see [`encode_t2_load_store_imm`]).
pub(super) fn ldr_imm(dst: Arm32Reg, base: Arm32Reg, offset: i32) -> u32 {
    encode_t2_load_store_imm(dst, base, offset, 0xF8D0, 0xF850, "ldr_imm")
}

/// `STR Rt, [Rn, #offset]` — mirrors [`ldr_imm`], L=0 (store). Halfword0
/// differs by one bit: STR T3 = 0xF8C0 vs LDR T3 = 0xF8D0; STR T4 = 0xF840
/// vs LDR T4 = 0xF850.
pub(super) fn str_imm(src: Arm32Reg, base: Arm32Reg, offset: i32) -> u32 {
    encode_t2_load_store_imm(src, base, offset, 0xF8C0, 0xF840, "str_imm")
}

/// Shared shape for byte/halfword/word LDR/STR immediate encoders.
/// `opcode_t3_base` is the halfword-0 value with Rn=0 for the T3/T2 form
/// (positive imm12 offset). `opcode_t4_base` is the same for T4 form
/// (signed imm8 offset). Offset in 0..=4095 uses T3; -255..=0 uses T4.
///
/// Out-of-range positive offsets silently wrap to their low 12 bits,
/// matching A32's `abs_offset & 0xFFF` behavior — callers that can hit
/// offsets > 4095 should lower via a base+scratch ADD first (callers are
/// shared with the A32 path, which has the same latent constraint). This
/// preserves parity with A32 rather than masking a real codegen bug.
#[inline]
fn encode_t2_load_store_imm(
    rt: Arm32Reg,
    rn: Arm32Reg,
    offset: i32,
    opcode_t3_base: u16,
    opcode_t4_base: u16,
    caller: &'static str,
) -> u32 {
    let rt = rt.idx() as u16;
    let rn = rn.idx() as u16;
    if offset >= 0 {
        let imm12 = (offset as u32 & 0xFFF) as u16;
        thumb32(opcode_t3_base | rn, (rt << 12) | imm12)
    } else if (-255..=0).contains(&offset) {
        let imm8 = offset.unsigned_abs() as u16;
        // T4 halfword 1: Rt | 1 | P=1 | U=0 | W=0 | imm8  → 0x0C00 | imm8
        thumb32(opcode_t4_base | rn, (rt << 12) | 0x0C00 | imm8)
    } else {
        unreachable!(
            "thumb2 {}: offset {offset} is negative beyond T4 range (-255..=0)",
            caller
        )
    }
}

/// `LDRB Rt, [Rn, #offset]` — T2 for 0..=4095, T3 for -255..=0.
pub(super) fn ldrb_imm(dst: Arm32Reg, base: Arm32Reg, offset: i32) -> u32 {
    encode_t2_load_store_imm(dst, base, offset, 0xF890, 0xF810, "ldrb_imm")
}

/// `STRB Rt, [Rn, #offset]`.
pub(super) fn strb_imm(src: Arm32Reg, base: Arm32Reg, offset: i32) -> u32 {
    encode_t2_load_store_imm(src, base, offset, 0xF880, 0xF800, "strb_imm")
}

/// `LDRH Rt, [Rn, #offset]`.
pub(super) fn ldrh_imm(dst: Arm32Reg, base: Arm32Reg, offset: i32) -> u32 {
    encode_t2_load_store_imm(dst, base, offset, 0xF8B0, 0xF830, "ldrh_imm")
}

/// `STRH Rt, [Rn, #offset]`.
pub(super) fn strh_imm(src: Arm32Reg, base: Arm32Reg, offset: i32) -> u32 {
    encode_t2_load_store_imm(src, base, offset, 0xF8A0, 0xF820, "strh_imm")
}

/// `LDRSB Rt, [Rn, #offset]` — sign-extended byte load.
pub(super) fn ldrsb_imm(dst: Arm32Reg, base: Arm32Reg, offset: i32) -> u32 {
    encode_t2_load_store_imm(dst, base, offset, 0xF990, 0xF910, "ldrsb_imm")
}

/// `LDRSH Rt, [Rn, #offset]` — sign-extended halfword load.
pub(super) fn ldrsh_imm(dst: Arm32Reg, base: Arm32Reg, offset: i32) -> u32 {
    encode_t2_load_store_imm(dst, base, offset, 0xF9B0, 0xF930, "ldrsh_imm")
}

// ─── Stack ──────────────────────────────────────────────────────────────────

/// `PUSH {reg_list}` — T2 encoding (STMDB SP!, {reg_list}).
///
/// Supports R0..R12 and LR (R14). SP (R13) and PC (R15) are forbidden by the
/// architecture. The A32-style `reg_mask` (bit N = 1 means Rn is pushed)
/// maps straight to T2's halfword1 layout — bits 0..=12 carry R0..R12, bit
/// 14 is the M (LR) bit, bits 13 and 15 must be zero.
pub(super) fn push(reg_mask: u16) -> u32 {
    debug_assert!(reg_mask & (1 << 13) == 0, "PUSH T2: SP (R13) forbidden");
    debug_assert!(reg_mask & (1 << 15) == 0, "PUSH T2: PC (R15) forbidden");
    debug_assert!(reg_mask != 0, "PUSH T2: empty register list");
    // Halfword 0: STMDB SP! (bits [31:16] of the instruction)
    let h0: u16 = 0xE92D;
    // Halfword 1: 0 | M | 0 | reg_list[12:0] — A32 mask already matches.
    let h1: u16 = reg_mask;
    thumb32(h0, h1)
}

/// `POP {reg_list}` — T2 encoding (LDMIA SP!, {reg_list}).
///
/// Mirrors [`push`]. Supports R0..R12, LR (bit 14 = M) and PC (bit 15 = P,
/// pops directly into PC for a return). SP (R13) is forbidden.
pub(super) fn pop(reg_mask: u16) -> u32 {
    debug_assert!(reg_mask & (1 << 13) == 0, "POP T2: SP (R13) forbidden");
    debug_assert!(reg_mask != 0, "POP T2: empty register list");
    // Halfword 0: LDMIA SP!
    let h0: u16 = 0xE8BD;
    // Halfword 1: P | M | 0 | reg_list[12:0] — A32 mask already matches.
    let h1: u16 = reg_mask;
    thumb32(h0, h1)
}

// ─── Branches ───────────────────────────────────────────────────────────────

/// Shared encoder for the BL / B.W offset layout (`S:imm10 | J1:J2:imm11`).
/// Returns the halfword pair as `(h0, h1)`. `byte_offset` is the label
/// distance from the branch instruction's own address (ARMv7-M PC at
/// execution = `branch_addr + 4`, so we subtract 4 before encoding).
#[inline]
fn encode_t2_long_branch_offset(byte_offset: i32) -> (u32, u32) {
    let pc_relative = byte_offset - 4;
    debug_assert!(
        pc_relative & 1 == 0,
        "thumb-2 branch target must be halfword-aligned"
    );
    assert!(
        (-16_777_216..=16_777_214).contains(&pc_relative),
        "thumb-2 long branch out of ±16MB range: {pc_relative}"
    );
    let off = pc_relative as u32;
    let s = (off >> 24) & 1;
    let i1 = (off >> 23) & 1;
    let i2 = (off >> 22) & 1;
    let imm10 = (off >> 12) & 0x3FF;
    let imm11 = (off >> 1) & 0x7FF;
    let j1 = (!(i1 ^ s)) & 1;
    let j2 = (!(i2 ^ s)) & 1;
    (s << 10 | imm10, (j1 << 13) | (j2 << 11) | imm11)
}

/// `B <label>` — T4 encoding (32-bit, ±16MB unconditional branch).
pub(super) fn b(byte_offset: i32) -> u32 {
    let (s_imm10, j1_j2_imm11) = encode_t2_long_branch_offset(byte_offset);
    let h0: u16 = (0xF000 | s_imm10) as u16;
    // h1: 1 0 J1 1 J2 imm11  →  base 0x9000 = 1001_0000_0000_0000.
    let h1: u16 = (0x9000 | j1_j2_imm11) as u16;
    thumb32(h0, h1)
}

/// `B<cc> <label>` — T3 encoding (32-bit, ±1MB conditional branch).
///
/// Different from T4: uses a 20-bit signed offset (S:J2:J1:imm6:imm11:'0')
/// and an explicit cond field in halfword 0. Range is ±1MB, plenty for
/// within-function control flow.
pub(super) fn b_cond(cond: Cond, byte_offset: i32) -> u32 {
    let pc_relative = byte_offset - 4;
    debug_assert!(
        pc_relative & 1 == 0,
        "thumb-2 B<cc> target must be halfword-aligned"
    );
    assert!(
        (-1_048_576..=1_048_574).contains(&pc_relative),
        "thumb-2 B<cc> out of ±1MB range: {pc_relative}"
    );
    let off = pc_relative as u32;
    let s = (off >> 20) & 1;
    let j2 = (off >> 19) & 1;
    let j1 = (off >> 18) & 1;
    let imm6 = (off >> 12) & 0x3F;
    let imm11 = (off >> 1) & 0x7FF;
    let cond_bits = (cond as u32) & 0xF;
    // Halfword 0: 11110 | S | cond[3:0] | imm6
    let h0: u16 = (0xF000 | (s << 10) | (cond_bits << 6) | imm6) as u16;
    // Halfword 1: 10 | J1 | 0 | J2 | imm11  →  base 0x8000 = 1000_0000_0000_0000.
    let h1: u16 = (0x8000 | (j1 << 13) | (j2 << 11) | imm11) as u16;
    thumb32(h0, h1)
}

/// `BL <label>` — T1 encoding (32-bit, ±16MB, link = LR set on return).
pub(super) fn bl(byte_offset: i32) -> u32 {
    let (s_imm10, j1_j2_imm11) = encode_t2_long_branch_offset(byte_offset);
    let h0: u16 = (0xF000 | s_imm10) as u16;
    // h1: 1 1 J1 1 J2 imm11  →  base 0xD000 = 1101_0000_0000_0000.
    let h1: u16 = (0xD000 | j1_j2_imm11) as u16;
    thumb32(h0, h1)
}

/// `BX Rm` — T1 encoding (16-bit branch-and-exchange).
///
/// Canonical function return is `BX LR` = 0x4770. Consumes the LSB of Rm as
/// the T-bit for mode switching, which is how Thumb/A32 interworking works:
/// when the JIT is in Thumb mode and calls an A32 helper, `BLX <reg>` with
/// the helper's raw (LSB=0) address switches back to A32; when a Thumb
/// function returns, `BX LR` consumes the LR's LSB (set=1 by the caller's
/// `blx` on a Thumb target) and stays in Thumb.
pub(super) fn bx(reg: Arm32Reg) -> u32 {
    // Halfword: 01000111_0_Rm[3:0]_000
    let rm = reg.idx() as u16;
    let instr: u16 = 0x4700 | (rm << 3);
    thumb16_pad(instr)
}

/// `BLX Rm` — T1 encoding (16-bit branch-with-link and exchange).
///
/// Same shape as [`bx`] with the L (link) bit set: halfword bit 7 flips from
/// 0 to 1, so the base pattern changes from 0x4700 to 0x4780. The CPU
/// switches modes according to `Rm`'s LSB — so an A32 helper (LSB=0 Rust
/// function pointer) switches to A32, a JITted Thumb function (LSB=1 from
/// `thumb_interworking_bit`) stays in Thumb.
pub(super) fn blx_reg(reg: Arm32Reg) -> u32 {
    let rm = reg.idx() as u16;
    let instr: u16 = 0x4780 | (rm << 3);
    thumb16_pad(instr)
}

// ─── Sign-extend ────────────────────────────────────────────────────────────

/// `SXTB Rd, Rm` — T2. Sign-extend low 8 bits of Rm into Rd.
pub(super) fn sxtb(dst: Arm32Reg, src: Arm32Reg) -> u32 {
    let rd = dst.idx() as u16;
    let rm = src.idx() as u16;
    thumb32(0xFA4Fu16, 0xF080u16 | (rd << 8) | rm)
}

/// `SXTH Rd, Rm` — T2. Sign-extend low 16 bits of Rm into Rd.
pub(super) fn sxth(dst: Arm32Reg, src: Arm32Reg) -> u32 {
    let rd = dst.idx() as u16;
    let rm = src.idx() as u16;
    thumb32(0xFA0Fu16, 0xF080u16 | (rd << 8) | rm)
}

// ─── VFP: register moves ────────────────────────────────────────────────────

// ── VFP register moves ──────────────────────────────────────────────────────
// Every VFP encoding below is bit-identical to its A32 form (with cond=AL),
// so we compute the A32 bit pattern in-line and halfword-swap via
// [`a32_to_thumb32`] for emit_u32's little-endian write.

const COND_AL_BITS: u32 = 0xE << 28;

/// `VMOV Dd, Dm` (double register copy).
pub(super) fn vmov_d(dst: u32, src: u32) -> u32 {
    let d = (dst >> 4) & 1;
    let vd = dst & 0xF;
    let m = (src >> 4) & 1;
    let vm = src & 0xF;
    a32_to_thumb32(
        COND_AL_BITS
            | (0b11101011 << 20)
            | (d << 22)
            | (vd << 12)
            | (0b1011 << 8)
            | (0b01 << 6)
            | (m << 5)
            | vm,
    )
}

/// `VMOV.F32 Sd, Sm` (single register copy).
pub(super) fn vmov_s(dst: u32, src: u32) -> u32 {
    let d = dst & 1;
    let vd = dst >> 1;
    let m = src & 1;
    let vm = src >> 1;
    a32_to_thumb32(
        COND_AL_BITS
            | (0b11101011 << 20)
            | (d << 22)
            | (vd << 12)
            | (0b1010 << 8)
            | (0b01 << 6)
            | (m << 5)
            | vm,
    )
}

/// `VMOV Rd, Sn` (move from single VFP to core reg).
pub(super) fn vmov_r_s(dst: Arm32Reg, sn: u32) -> u32 {
    let n = sn & 1;
    let vn = sn >> 1;
    a32_to_thumb32(
        COND_AL_BITS
            | (0b1110000_1 << 20)
            | (vn << 16)
            | ((dst.idx() & 0xF) << 12)
            | (0b1010 << 8)
            | (n << 7)
            | (0b001 << 4),
    )
}

/// `VMOV Sn, Rd` (move from core reg to single VFP).
pub(super) fn vmov_s_r(sn: u32, src: Arm32Reg) -> u32 {
    let n = sn & 1;
    let vn = sn >> 1;
    a32_to_thumb32(
        COND_AL_BITS
            | (0b1110000_0 << 20)
            | (vn << 16)
            | ((src.idx() & 0xF) << 12)
            | (0b1010 << 8)
            | (n << 7)
            | (0b001 << 4),
    )
}

/// `VMOV Rt, Rt2, Dm` (move Dm to two core regs; op=1 direction).
pub(super) fn vmov_rr_d(dst_lo: Arm32Reg, dst_hi: Arm32Reg, dm: u32) -> u32 {
    let m = (dm >> 4) & 1;
    let vm = dm & 0xF;
    a32_to_thumb32(
        COND_AL_BITS
            | (0b11000101 << 20)
            | ((dst_hi.idx() & 0xF) << 16)
            | ((dst_lo.idx() & 0xF) << 12)
            | (0b1011 << 8)
            | (0b00 << 6)
            | (m << 5)
            | (0b1 << 4)
            | vm,
    )
}

/// `VMOV Dm, Rt, Rt2` (move two core regs to Dm; op=0 direction).
pub(super) fn vmov_d_rr(dm: u32, src_lo: Arm32Reg, src_hi: Arm32Reg) -> u32 {
    let m = (dm >> 4) & 1;
    let vm = dm & 0xF;
    a32_to_thumb32(
        COND_AL_BITS
            | (0b11000100 << 20)
            | ((src_hi.idx() & 0xF) << 16)
            | ((src_lo.idx() & 0xF) << 12)
            | (0b1011 << 8)
            | (0b00 << 6)
            | (m << 5)
            | (0b1 << 4)
            | vm,
    )
}

// ─── VFP: arithmetic ────────────────────────────────────────────────────────

// ── VFP arithmetic (double / single precision) ──────────────────────────────
// Each encoding is bit-identical to its A32 form with cond=AL; composed
// inline and halfword-swapped via `a32_to_thumb32`.

/// `VADD.F64 Dd, Dn, Dm` — T2 encoding.
pub(super) fn vadd_d(dd: u32, dn: u32, dm: u32) -> u32 {
    let d = (dd >> 4) & 1;
    let vd = dd & 0xF;
    let n = (dn >> 4) & 1;
    let vn = dn & 0xF;
    let m = (dm >> 4) & 1;
    let vm = dm & 0xF;
    a32_to_thumb32(
        COND_AL_BITS
            | (0b11100011 << 20)
            | (d << 22)
            | (vn << 16)
            | (vd << 12)
            | (0b1011 << 8)
            | (n << 7)
            | (m << 5)
            | vm,
    )
}

/// `VSUB.F64 Dd, Dn, Dm` — T2 encoding. Bit 6 = 1 (vs. VADD's bit 6 = 0).
pub(super) fn vsub_d(dd: u32, dn: u32, dm: u32) -> u32 {
    let d = (dd >> 4) & 1;
    let vd = dd & 0xF;
    let n = (dn >> 4) & 1;
    let vn = dn & 0xF;
    let m = (dm >> 4) & 1;
    let vm = dm & 0xF;
    a32_to_thumb32(
        COND_AL_BITS
            | (0b11100011 << 20)
            | (d << 22)
            | (vn << 16)
            | (vd << 12)
            | (0b1011 << 8)
            | (n << 7)
            | (1 << 6)
            | (m << 5)
            | vm,
    )
}

/// `VMUL.F64 Dd, Dn, Dm` — T2 encoding. Opcode 0010 in bits [23:20].
pub(super) fn vmul_d(dd: u32, dn: u32, dm: u32) -> u32 {
    let d = (dd >> 4) & 1;
    let vd = dd & 0xF;
    let n = (dn >> 4) & 1;
    let vn = dn & 0xF;
    let m = (dm >> 4) & 1;
    let vm = dm & 0xF;
    a32_to_thumb32(
        COND_AL_BITS
            | (0b11100010 << 20)
            | (d << 22)
            | (vn << 16)
            | (vd << 12)
            | (0b1011 << 8)
            | (n << 7)
            | (m << 5)
            | vm,
    )
}

/// `VDIV.F64 Dd, Dn, Dm` — T1 encoding. Opcode 1000 in bits [23:20].
pub(super) fn vdiv_d(dd: u32, dn: u32, dm: u32) -> u32 {
    let d = (dd >> 4) & 1;
    let vd = dd & 0xF;
    let n = (dn >> 4) & 1;
    let vn = dn & 0xF;
    let m = (dm >> 4) & 1;
    let vm = dm & 0xF;
    a32_to_thumb32(
        COND_AL_BITS
            | (0b11101000 << 20)
            | (d << 22)
            | (vn << 16)
            | (vd << 12)
            | (0b1011 << 8)
            | (n << 7)
            | (m << 5)
            | vm,
    )
}

/// `VADD.F32 Sd, Sn, Sm`.
pub(super) fn vadd_s(sd: u32, sn: u32, sm: u32) -> u32 {
    let d = sd & 1;
    let vd = sd >> 1;
    let n = sn & 1;
    let vn = sn >> 1;
    let m = sm & 1;
    let vm = sm >> 1;
    a32_to_thumb32(
        COND_AL_BITS
            | (0b11100011 << 20)
            | (d << 22)
            | (vn << 16)
            | (vd << 12)
            | (0b1010 << 8)
            | (n << 7)
            | (m << 5)
            | vm,
    )
}

/// `VSUB.F32 Sd, Sn, Sm`.
pub(super) fn vsub_s(sd: u32, sn: u32, sm: u32) -> u32 {
    let d = sd & 1;
    let vd = sd >> 1;
    let n = sn & 1;
    let vn = sn >> 1;
    let m = sm & 1;
    let vm = sm >> 1;
    a32_to_thumb32(
        COND_AL_BITS
            | (0b11100011 << 20)
            | (d << 22)
            | (vn << 16)
            | (vd << 12)
            | (0b1010 << 8)
            | (n << 7)
            | (1 << 6)
            | (m << 5)
            | vm,
    )
}

/// `VMUL.F32 Sd, Sn, Sm`.
pub(super) fn vmul_s(sd: u32, sn: u32, sm: u32) -> u32 {
    let d = sd & 1;
    let vd = sd >> 1;
    let n = sn & 1;
    let vn = sn >> 1;
    let m = sm & 1;
    let vm = sm >> 1;
    a32_to_thumb32(
        COND_AL_BITS
            | (0b11100010 << 20)
            | (d << 22)
            | (vn << 16)
            | (vd << 12)
            | (0b1010 << 8)
            | (n << 7)
            | (m << 5)
            | vm,
    )
}

/// `VDIV.F32 Sd, Sn, Sm`. The A32 encoder names the params (dd, dn, dm)
/// inherited from a shared signature; they're actually single-precision.
pub(super) fn vdiv_s(dd: u32, dn: u32, dm: u32) -> u32 {
    let d = dd & 1;
    let vd = dd >> 1;
    let n = dn & 1;
    let vn = dn >> 1;
    let m = dm & 1;
    let vm = dm >> 1;
    a32_to_thumb32(
        COND_AL_BITS
            | (0b11101000 << 20)
            | (d << 22)
            | (vn << 16)
            | (vd << 12)
            | (0b1010 << 8)
            | (n << 7)
            | (m << 5)
            | vm,
    )
}

/// `VSQRT.F64 Dd, Dm`.
pub(super) fn vsqrt_d(dd: u32, dm: u32) -> u32 {
    let d = (dd >> 4) & 1;
    let vd = dd & 0xF;
    let m = (dm >> 4) & 1;
    let vm = dm & 0xF;
    a32_to_thumb32(
        COND_AL_BITS
            | (0b11101011 << 20)
            | (d << 22)
            | (0b0001 << 16)
            | (vd << 12)
            | (0b1011 << 8)
            | (0b11 << 6)
            | (m << 5)
            | vm,
    )
}

/// `VSQRT.F32 Sd, Sm`.
pub(super) fn vsqrt_s(sd: u32, sm: u32) -> u32 {
    let d = sd & 1;
    let vd = sd >> 1;
    let m = sm & 1;
    let vm = sm >> 1;
    a32_to_thumb32(
        COND_AL_BITS
            | (0b11101011 << 20)
            | (d << 22)
            | (0b0001 << 16)
            | (vd << 12)
            | (0b1010 << 8)
            | (0b11 << 6)
            | (m << 5)
            | vm,
    )
}

/// `VNEG.F64 Dd, Dm`.
pub(super) fn vneg_d(dd: u32, dm: u32) -> u32 {
    let d = (dd >> 4) & 1;
    let vd = dd & 0xF;
    let m = (dm >> 4) & 1;
    let vm = dm & 0xF;
    a32_to_thumb32(
        COND_AL_BITS
            | (0b11101011 << 20)
            | (d << 22)
            | (0b0001 << 16)
            | (vd << 12)
            | (0b1011 << 8)
            | (0b01 << 6)
            | (m << 5)
            | vm,
    )
}

/// `VNEG.F32 Sd, Sm`.
pub(super) fn vneg_s(sd: u32, sm: u32) -> u32 {
    let d = sd & 1;
    let vd = sd >> 1;
    let m = sm & 1;
    let vm = sm >> 1;
    a32_to_thumb32(
        COND_AL_BITS
            | (0b11101011 << 20)
            | (d << 22)
            | (0b0001 << 16)
            | (vd << 12)
            | (0b1010 << 8)
            | (0b01 << 6)
            | (m << 5)
            | vm,
    )
}

/// `VABS.F64 Dd, Dm`.
pub(super) fn vabs_d(dd: u32, dm: u32) -> u32 {
    let d = (dd >> 4) & 1;
    let vd = dd & 0xF;
    let m = (dm >> 4) & 1;
    let vm = dm & 0xF;
    a32_to_thumb32(
        COND_AL_BITS
            | (0b11101011 << 20)
            | (d << 22)
            | (0b0000 << 16)
            | (vd << 12)
            | (0b1011 << 8)
            | (0b11 << 6)
            | (m << 5)
            | vm,
    )
}

/// `VABS.F32 Sd, Sm`.
pub(super) fn vabs_s(sd: u32, sm: u32) -> u32 {
    let d = sd & 1;
    let vd = sd >> 1;
    let m = sm & 1;
    let vm = sm >> 1;
    a32_to_thumb32(
        COND_AL_BITS
            | (0b11101011 << 20)
            | (d << 22)
            | (0b0000 << 16)
            | (vd << 12)
            | (0b1010 << 8)
            | (0b11 << 6)
            | (m << 5)
            | vm,
    )
}

// ─── VFP: compare ───────────────────────────────────────────────────────────

/// `VCMP.F64 Dd, Dm`.
pub(super) fn vcmp_d(dd: u32, dm: u32) -> u32 {
    let d = (dd >> 4) & 1;
    let vd = dd & 0xF;
    let m = (dm >> 4) & 1;
    let vm = dm & 0xF;
    a32_to_thumb32(
        COND_AL_BITS
            | (0b11101011 << 20)
            | (d << 22)
            | (0b0100 << 16)
            | (vd << 12)
            | (0b1011 << 8)
            | (0b01 << 6)
            | (m << 5)
            | vm,
    )
}

/// `VCMP.F32 Sd, Sm`.
pub(super) fn vcmp_s(sd: u32, sm: u32) -> u32 {
    let d = sd & 1;
    let vd = sd >> 1;
    let m = sm & 1;
    let vm = sm >> 1;
    a32_to_thumb32(
        COND_AL_BITS
            | (0b11101011 << 20)
            | (d << 22)
            | (0b0100 << 16)
            | (vd << 12)
            | (0b1010 << 8)
            | (0b01 << 6)
            | (m << 5)
            | vm,
    )
}

/// `VMRS APSR_nzcv, FPSCR` — copy VFP flags into APSR. The Rt field at
/// instruction bits [15:12] is `0b1111` (APSR_nzcv destination marker), so
/// the constant is `0x0EF1_FA10`, not `0x0EF1_0A10`.
pub(super) fn vmrs_apsr() -> u32 {
    a32_to_thumb32(COND_AL_BITS | 0x0EF1FA10)
}

// ─── VFP: conversions ───────────────────────────────────────────────────────

/// `VCVT.F64.F32 Dd, Sm` — single → double.
pub(super) fn vcvt_d_s(dd: u32, sm: u32) -> u32 {
    let d = (dd >> 4) & 1;
    let vd = dd & 0xF;
    let m = sm & 1;
    let vm = sm >> 1;
    a32_to_thumb32(
        COND_AL_BITS
            | (0b11101011 << 20)
            | (d << 22)
            | (0b0111 << 16)
            | (vd << 12)
            | (0b1010 << 8)
            | (0b11 << 6)
            | (m << 5)
            | vm,
    )
}

/// `VCVT.F32.F64 Sd, Dm` — double → single.
pub(super) fn vcvt_s_d(sd: u32, dm: u32) -> u32 {
    let d = sd & 1;
    let vd = sd >> 1;
    let m = (dm >> 4) & 1;
    let vm = dm & 0xF;
    a32_to_thumb32(
        COND_AL_BITS
            | (0b11101011 << 20)
            | (d << 22)
            | (0b0111 << 16)
            | (vd << 12)
            | (0b1011 << 8)
            | (0b11 << 6)
            | (m << 5)
            | vm,
    )
}

/// `VCVT.F64.S32 Dd, Sm` — signed i32 → double.
pub(super) fn vcvt_d_s32(dd: u32, sm: u32) -> u32 {
    let d = (dd >> 4) & 1;
    let vd = dd & 0xF;
    let m = sm & 1;
    let vm = sm >> 1;
    a32_to_thumb32(
        COND_AL_BITS
            | (0b11101011 << 20)
            | (d << 22)
            | (0b1000 << 16)
            | (vd << 12)
            | (0b1011 << 8)
            | (0b11 << 6)
            | (m << 5)
            | vm,
    )
}

/// `VCVT.F64.U32 Dd, Sm` — unsigned i32 → double.
pub(super) fn vcvt_d_u32(dd: u32, sm: u32) -> u32 {
    let d = (dd >> 4) & 1;
    let vd = dd & 0xF;
    let m = sm & 1;
    let vm = sm >> 1;
    a32_to_thumb32(
        COND_AL_BITS
            | (0b11101011 << 20)
            | (d << 22)
            | (0b1000 << 16)
            | (vd << 12)
            | (0b1011 << 8)
            | (0b01 << 6)
            | (m << 5)
            | vm,
    )
}

/// `VCVT.F32.S32 Sd, Sm` — signed i32 → single.
pub(super) fn vcvt_s_s32(sd: u32, sm: u32) -> u32 {
    let d = sd & 1;
    let vd = sd >> 1;
    let m = sm & 1;
    let vm = sm >> 1;
    a32_to_thumb32(
        COND_AL_BITS
            | (0b11101011 << 20)
            | (d << 22)
            | (0b1000 << 16)
            | (vd << 12)
            | (0b1010 << 8)
            | (0b11 << 6)
            | (m << 5)
            | vm,
    )
}

/// `VCVT.F32.U32 Sd, Sm` — unsigned i32 → single.
pub(super) fn vcvt_s_u32(sd: u32, sm: u32) -> u32 {
    let d = sd & 1;
    let vd = sd >> 1;
    let m = sm & 1;
    let vm = sm >> 1;
    a32_to_thumb32(
        COND_AL_BITS
            | (0b11101011 << 20)
            | (d << 22)
            | (0b1000 << 16)
            | (vd << 12)
            | (0b1010 << 8)
            | (0b01 << 6)
            | (m << 5)
            | vm,
    )
}

// ─── Misc ───────────────────────────────────────────────────────────────────

/// Unconditional Thumb-2 DP-imm encoder with an explicit A32-style opcode.
/// The `cond` parameter is retained for API symmetry with
/// `enc_a32::dp_imm_cond` but must be `Cond::Al` here — Thumb-2 has no
/// per-instruction condition field. Conditional execution is realized by
/// emitting an [`it`] prefix *before* calling this function, via the
/// `emit_dp_imm_cond_into` helper in `inst.rs`.
///
/// **A32/Thumb-2 opcode translation.** A32 packs DP-imm opcodes into bits
/// [24:21] with a different assignment than Thumb-2 does, so callers that
/// pass A32 opcode numbers (0b1101 = MOV, 0b1010 = CMP, etc.) would
/// silently get the wrong Thumb-2 instruction. The `match` below maps
/// A32 → Thumb-2. MOV/MVN additionally pin `Rn = 0xF` (the ORR/ORN-with-
/// no-source encoding that Thumb-2 uses for register-less moves).
pub(super) fn dp_imm_cond(
    cond: Cond,
    opcode: u32,
    s: bool,
    dst: Arm32Reg,
    lhs: Arm32Reg,
    imm8: u32,
    rotate: u32,
) -> u32 {
    debug_assert!(
        matches!(cond, Cond::Al),
        "thumb2 dp_imm_cond: expected Cond::Al (callers prefix with `it(cond)` via the inst.rs helper); got {cond:?}"
    );
    let value = (imm8 & 0xFF).rotate_right(2 * rotate);
    let (t2_op, t2_rn): (u16, u16) = match opcode & 0xF {
        0b0000 => (0b0000, lhs.idx() as u16), // AND
        0b0001 => (0b0100, lhs.idx() as u16), // EOR
        0b0010 => (0b1101, lhs.idx() as u16), // SUB
        0b0011 => (0b1110, lhs.idx() as u16), // RSB
        0b0100 => (0b1000, lhs.idx() as u16), // ADD
        0b0101 => (0b1010, lhs.idx() as u16), // ADC
        0b0110 => (0b1011, lhs.idx() as u16), // SBC
        0b1000 => (0b0000, lhs.idx() as u16), // TST (AND with Rd=0xF at caller)
        0b1001 => (0b0100, lhs.idx() as u16), // TEQ (EOR with Rd=0xF at caller)
        0b1010 => (0b1101, lhs.idx() as u16), // CMP (SUB with Rd=0xF at caller)
        0b1011 => (0b1000, lhs.idx() as u16), // CMN (ADD with Rd=0xF at caller)
        0b1100 => (0b0010, lhs.idx() as u16), // ORR
        0b1101 => (0b0010, 0xF),              // MOV (ORR with Rn=0xF)
        0b1110 => (0b0001, lhs.idx() as u16), // BIC
        0b1111 => (0b0011, 0xF),              // MVN (ORN with Rn=0xF)
        other => panic!("thumb2 dp_imm_cond: unknown A32 opcode {other:#x}"),
    };
    thumb32_dp_imm_raw(t2_op, s, dst.idx() as u16, t2_rn, value)
}

/// `IT <firstcond>` — T1 encoding, single-instruction block (raw 16-bit).
///
/// Unlike most other 16-bit encoders in this module, [`it`] does NOT pad
/// itself to a 4-byte slot: the whole point of `IT` is that the *next*
/// instruction in memory executes conditionally under `firstcond`, and a
/// pad NOP would occupy that conditional-execution slot instead of the
/// intended instruction. Callers must emit `IT` via `emit_u16` immediately
/// followed by the conditional instruction via `emit_u32` (or `emit_u16`
/// for 16-bit conditional forms). See `emit_dp_imm_cond_into` in `inst.rs`
/// for the canonical pattern. Mask `0b1000` = single-instruction block,
/// no T/E alternation — which is the ARMv8-M mainline limit anyway.
pub(super) fn it(cond: Cond) -> u16 {
    let cond_bits = (cond as u16) & 0xF;
    0xBF00 | (cond_bits << 4) | 0b1000
}

/// Thumb 16-bit NOP (`0xBF00`), padded with a trailing 16-bit NOP for the
/// uniform 4-byte slot.
pub(super) fn nop() -> u32 {
    thumb16_pad(0xBF00)
}

/// `VPUSH {Dd..D(d+count-1)}` — T1 encoding (STMDB SP!, VFP list).
///
/// The A32 and T1 encodings are bit-identical; the only difference is byte
/// layout in memory (halfword order). `first_d` is the D-register number in
/// 0..=15 (VFPv3-D16). `count` is the number of consecutive D registers to
/// push.
pub(super) fn vpush_d(first_d: u32, count: u32) -> u32 {
    // Halfword 0 bit pattern (all bits of the full instruction [31:16] with
    // D=0, Rn=13=SP): 1110_1101_0010_1101 = 0xED2D. D bit lives at
    // instruction bit 22, which is halfword0 bit 6 (= 22 - 16).
    let d = ((first_d >> 4) & 1) as u16;
    let vd = (first_d & 0xF) as u16;
    let imm8 = ((count * 2) & 0xFF) as u16;
    let h0 = 0xED2Du16 | (d << 6);
    // Halfword 1: Vd[3:0] | 1011 | imm8 (double-precision VFP list size)
    let h1 = (vd << 12) | (0x0Bu16 << 8) | imm8;
    thumb32(h0, h1)
}

/// `VPOP {Dd..D(d+count-1)}` — T1 encoding (LDMIA SP!, VFP list).
///
/// Mirrors [`vpush_d`]. Halfword 0 bit pattern with D=0, Rn=13: 0xECBD.
pub(super) fn vpop_d(first_d: u32, count: u32) -> u32 {
    let d = ((first_d >> 4) & 1) as u16;
    let vd = (first_d & 0xF) as u16;
    let imm8 = ((count * 2) & 0xFF) as u16;
    let h0 = 0xECBDu16 | (d << 6);
    let h1 = (vd << 12) | (0x0Bu16 << 8) | imm8;
    thumb32(h0, h1)
}
