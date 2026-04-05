//! Instruction selection fusion pass.
//!
//! Fuses adjacent instruction pairs into higher-level semantic operations
//! that backends can lower to single hardware instructions. This is a shared
//! (ISA-neutral) pass — every backend must handle the fused instructions,
//! though the benefit varies per target.
//!
//! # Patterns
//!
//! 1. **Bitfield extract** (`try_fuse_ubfx`):
//!    `ShrU(src, #lsb) + And(shifted, #mask)` → `BitfieldExtractU { src, lsb, bits }`
//!    where mask = `(1 << bits) - 1` (contiguous low-bit mask).
//!    - ARM64/armv7a: `UBFX` (1 insn). x86_64: `SHR + AND` (no net gain, but correct).
//!
//! 2. **Shifted operand** (`try_fuse_shifted_binop`):
//!    `shift(rhs, #amount) + binop(lhs, shifted)` → `IntBinaryShifted { lhs, rhs, shift, amount }`
//!    for Add/Sub/And/Or/Xor with LSL/LSR/ASR.
//!    - ARM64/armv7a: barrel-shifter form (1 insn). x86_64: decomposed (no gain).
//!
//! 3. **Test bits** (`try_fuse_test_bits`):
//!    `And(src, mask) + IntCompare(result, 0, Eq|Ne)` → `TestBits { src, mask, kind }`
//!    when the AND result is only used by the compare.
//!    - ARM64/armv7a: `TST`. x86_64: `TEST`. All backends benefit (1 insn vs 2).
//!
//! # Ordering
//!
//! Must run **after** `copy_propagate`. Copy propagation rewrites register
//! indirections that expose the adjacent ShrU+And / shift+binop pairs.
//! Running before copy propagation finds almost no matches.

use alloc::vec::Vec;

use crate::vm::backend::BackendConfig;
use crate::vm::machine::machine_ir::{
    MachineBlock, MachineCompareKind, MachineInst, MachineInstKind, MachineIntBinaryOp,
    MachineIntWidth, MachineReg, MachineRegOwner, MachineShiftOp, MachineSign, MachineTerminator,
    MachineValue,
};

use super::helpers::reg_live_after;

/// Check that an intermediate register can be safely eliminated.
/// It must hold one linear value (not a cached local snapshot) and be dead
/// after the fused instruction.
fn can_eliminate_reg(
    owner: Option<MachineRegOwner>,
    reg: MachineReg,
    dst: MachineReg,
    later: &[MachineInst],
    term: &MachineTerminator,
) -> bool {
    if owner != Some(MachineRegOwner::LinearValue) {
        return false;
    }
    // If the fused instruction overwrites the same register, it's dead by
    // definition. Otherwise check remaining instructions + terminator.
    reg == dst || !reg_live_after(later, term, reg)
}

pub(super) fn fuse_isel(block: &mut MachineBlock, config: BackendConfig) {
    let ops = &block.ops;
    let term = &block.terminator;
    let mut out: Vec<MachineInst> = Vec::with_capacity(ops.len());
    let mut i = 0;

    while i < ops.len() {
        if i + 1 < ops.len() {
            // --- Pattern 1: UBFX (ShrU + And -> BitfieldExtractU) ---
            if let Some(fused) = try_fuse_ubfx(&ops[i], &ops[i + 1], &ops[i + 2..], term, config) {
                out.push(fused);
                i += 2;
                continue;
            }

            // --- Pattern 2: Shifted operand (shift + binop -> IntBinaryShifted) ---
            if let Some(fused) =
                try_fuse_shifted_binop(&ops[i], &ops[i + 1], &ops[i + 2..], term, config)
            {
                out.push(fused);
                i += 2;
                continue;
            }

            // --- Pattern 3: TST (And + IntCompare(0) -> TestBits) ---
            if let Some(fused) =
                try_fuse_test_bits(&ops[i], &ops[i + 1], &ops[i + 2..], term, config)
            {
                out.push(fused);
                i += 2;
                continue;
            }
        }

        out.push(ops[i].clone());
        i += 1;
    }

    block.ops = out;
}

/// Check if `mask` is a contiguous low-bit mask: `(1 << width) - 1`.
/// Returns the width if so.
fn contiguous_mask_width(mask: u64, int_width: MachineIntWidth) -> Option<u8> {
    if mask == 0 {
        return None;
    }
    let max_bits: u32 = match int_width {
        MachineIntWidth::I32 => 32,
        MachineIntWidth::I64 => 64,
    };
    // mask must be (1 << width) - 1, i.e. all ones in low bits.
    let width = mask.count_ones();
    if width >= max_bits {
        return None;
    }
    if mask == (1u64 << width) - 1 {
        Some(width as u8)
    } else {
        None
    }
}

/// Map MachineIntBinaryOp shift to MachineShiftOp.
fn shift_op(op: MachineIntBinaryOp) -> Option<MachineShiftOp> {
    match op {
        MachineIntBinaryOp::Shl => Some(MachineShiftOp::Lsl),
        MachineIntBinaryOp::ShrU => Some(MachineShiftOp::Lsr),
        MachineIntBinaryOp::ShrS => Some(MachineShiftOp::Asr),
        _ => None,
    }
}

/// Can this binary op use a shifted register operand?
fn supports_shifted_reg(op: MachineIntBinaryOp) -> bool {
    matches!(
        op,
        MachineIntBinaryOp::Add
            | MachineIntBinaryOp::Sub
            | MachineIntBinaryOp::And
            | MachineIntBinaryOp::Or
            | MachineIntBinaryOp::Xor
    )
}

fn is_commutative(op: MachineIntBinaryOp) -> bool {
    matches!(
        op,
        MachineIntBinaryOp::Add
            | MachineIntBinaryOp::And
            | MachineIntBinaryOp::Or
            | MachineIntBinaryOp::Xor
    )
}

fn max_shift(width: MachineIntWidth) -> u8 {
    match width {
        MachineIntWidth::I32 => 31,
        MachineIntWidth::I64 => 63,
    }
}

// ── Pattern 1: UBFX ─────────────────────────────────────────────────────────

/// Match `ShrU(src, #shift) + And(shifted, #mask)` where mask is contiguous.
fn try_fuse_ubfx(
    first: &MachineInst,
    second: &MachineInst,
    later: &[MachineInst],
    term: &MachineTerminator,
    _config: BackendConfig,
) -> Option<MachineInst> {
    // First: IntBinary { op: ShrU, dst: shift_dst, lhs: Reg(src), rhs: Imm64(shift) }
    let (width, shift_dst, src, shift_amount) = match first.kind {
        MachineInstKind::IntBinary {
            width,
            op: MachineIntBinaryOp::ShrU,
            dst,
            lhs: MachineValue::Reg(src),
            rhs: MachineValue::Imm64(shift),
        } if shift <= max_shift(width) as u64 => (width, dst, src, shift as u8),
        _ => return None,
    };

    // Second: IntBinary { op: And, dst: and_dst, lhs: Reg(shift_dst), rhs: Imm64(mask) }
    let (and_dst, mask) = match second.kind {
        MachineInstKind::IntBinary {
            width: w2,
            op: MachineIntBinaryOp::And,
            dst,
            lhs: MachineValue::Reg(lhs),
            rhs: MachineValue::Imm64(mask),
        } if w2 == width && lhs == shift_dst => (dst, mask),
        _ => return None,
    };

    // Check mask is contiguous and the combined lsb+width fits.
    let mask_width = contiguous_mask_width(mask, width)?;
    let max = max_shift(width) + 1;
    if shift_amount as u16 + mask_width as u16 > max as u16 {
        return None;
    }

    // shift_dst must be safely eliminable (linear value + dead after fusion).
    if !can_eliminate_reg(first.kind.def_owner(), shift_dst, and_dst, later, term) {
        return None;
    }

    Some(MachineInst {
        kind: MachineInstKind::BitfieldExtractU {
            width,
            dst: and_dst,
            src,
            lsb: shift_amount,
            bits: mask_width,
        },
    })
}

// ── Pattern 2: Shifted operand ───────────────────────────────────────────────

/// Match `shift(src, #amount) + binop(reg, shifted)`.
fn try_fuse_shifted_binop(
    first: &MachineInst,
    second: &MachineInst,
    later: &[MachineInst],
    term: &MachineTerminator,
    _config: BackendConfig,
) -> Option<MachineInst> {
    // First: IntBinary { op: Shl/ShrU/ShrS, dst: shift_dst, lhs: Reg(shift_src), rhs: Imm64(amount) }
    let (width, shift_dst, shift_src, shift, amount) = match first.kind {
        MachineInstKind::IntBinary {
            width,
            op,
            dst,
            lhs: MachineValue::Reg(src),
            rhs: MachineValue::Imm64(amt),
        } => {
            let s = shift_op(op)?;
            if amt > max_shift(width) as u64 {
                return None;
            }
            (width, dst, src, s, amt as u8)
        }
        _ => return None,
    };

    // Second: IntBinary { op: Add/Sub/And/Or/Xor, using shift_dst as one operand }
    let (binop, binop_dst, lhs_reg, rhs_is_shifted) = match second.kind {
        MachineInstKind::IntBinary {
            width: w2,
            op,
            dst,
            lhs: MachineValue::Reg(lhs),
            rhs: MachineValue::Reg(rhs),
        } if w2 == width && supports_shifted_reg(op) => {
            if rhs == shift_dst {
                // Normal: lhs OP (rhs << N)
                (op, dst, lhs, true)
            } else if lhs == shift_dst && is_commutative(op) {
                // Commutative swap: rhs OP (lhs << N) = (lhs << N) OP rhs
                (op, dst, rhs, true)
            } else {
                return None;
            }
        }
        _ => return None,
    };

    if !rhs_is_shifted {
        return None;
    }

    // The non-shifted operand must not reference the shift destination register.
    // Since we eliminate the shift instruction, lhs_reg would read stale data.
    if lhs_reg == shift_dst {
        return None;
    }

    // shift_dst must be safely eliminable (linear value + dead after fusion).
    if !can_eliminate_reg(first.kind.def_owner(), shift_dst, binop_dst, later, term) {
        return None;
    }

    Some(MachineInst {
        kind: MachineInstKind::IntBinaryShifted {
            width,
            op: binop,
            dst: binop_dst,
            lhs: lhs_reg,
            rhs: shift_src,
            shift,
            amount,
        },
    })
}

// ── Pattern 3: TST ───────────────────────────────────────────────────────────

/// Match `And(src, mask) + IntCompare(and_result, 0, Eq/Ne)`.
fn try_fuse_test_bits(
    first: &MachineInst,
    second: &MachineInst,
    later: &[MachineInst],
    term: &MachineTerminator,
    _config: BackendConfig,
) -> Option<MachineInst> {
    // First: IntBinary { op: And, dst: and_dst, lhs: src, rhs: mask }
    let (width, and_dst, src, mask) = match first.kind {
        MachineInstKind::IntBinary {
            width,
            op: MachineIntBinaryOp::And,
            dst,
            lhs: MachineValue::Reg(src),
            rhs: mask,
        } => (width, dst, src, mask),
        _ => return None,
    };

    // Second: IntCompare { kind: Eq|Ne, dst: cmp_dst, lhs: Reg(and_dst), rhs: Imm64(0) }
    let (cmp_kind, cmp_dst) = match second.kind {
        MachineInstKind::IntCompare {
            width: w2,
            kind,
            sign: MachineSign::Unsigned,
            dst,
            lhs: MachineValue::Reg(lhs),
            rhs: MachineValue::Imm64(0),
        } if w2 == width
            && lhs == and_dst
            && matches!(kind, MachineCompareKind::Eq | MachineCompareKind::Ne) =>
        {
            (kind, dst)
        }
        _ => return None,
    };

    // and_dst must be safely eliminable (linear value + dead after fusion).
    if !can_eliminate_reg(first.kind.def_owner(), and_dst, cmp_dst, later, term) {
        return None;
    }

    Some(MachineInst {
        kind: MachineInstKind::TestBits {
            width,
            kind: cmp_kind,
            dst: cmp_dst,
            src,
            mask,
        },
    })
}
