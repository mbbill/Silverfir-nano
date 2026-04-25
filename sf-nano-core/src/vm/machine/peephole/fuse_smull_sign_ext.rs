//! Fuse `i64.mul` of two `i64.extend_i32_s` operands into a single signed
//! 32×32→64 multiply (`Int64MulFromSignExt32`).
//!
//! # Pattern
//!
//! Lowering `i64.extend_i32_s(x)` on a 32-bit GP backend emits two MIR ops:
//! a `Move` for the low half (= the original i32 value) and an
//! `IntBinary{ShrS, _, x, 31}` for the high half (= sign of x). After
//! `copy_propagate` and the `gp32` lowering pipeline finishes, the typical
//! pattern around `i64.mul` looks like one of:
//!
//! ```text
//!     // Squaring case (lhs == rhs, e.g. zx*zx):
//!     IntBinary{ShrS, R6, Reg(R4), Imm64(31)}    ; R6 = sign(R4)
//!     Store [fp+72] U32 <- Reg(R4)               ; spill R4
//!     Store [fp+76] U32 <- Reg(R6)               ; spill R6
//!     Load R4 <- [fp+72] U32                     ; reload (no-op)
//!     Load R6 <- [fp+76] U32                     ; reload (no-op)
//!     Int64PairBinary{Mul, ..., Reg(R4), Reg(R6), Reg(R4), Reg(R6)}
//!
//!     // Mixed case (lhs != rhs, e.g. zx*zy) — note the Move that aliases
//!     // R6 to R4 before the sign-extend of R4:
//!     Move R6 <- Reg(R4)                         ; R6 := R4
//!     IntBinary{ShrS, R7, Reg(R4), Imm64(31)}    ; R7 = sign(R4)
//!     Store [fp+96] U32 <- Reg(R6)               ; spill R6 (= R4)
//!     Store [fp+100] U32 <- Reg(R7)              ; spill R7
//!     Load R6 <- [fp+96] U32                     ; reload
//!     Load R7 <- [fp+100] U32                    ; reload
//!     Int64PairBinary{Mul, ..., Reg(R6), Reg(R7), ...}
//! ```
//!
//! The mul lowers naively to `UMULL + MLA + MLA + 2 movs` on arm32 — but
//! when both i64 operands are sign-extended-from-i32, the entire 64-bit
//! product collapses to a single signed `SMULL`. That's the rewrite this
//! pass enables.
//!
//! # State tracked
//!
//! Two per-register maps are maintained alongside a small recent-spill
//! table, all updated in a single forward sweep:
//!
//! - `sign_ext_of[r] = Some(s)` — `r` currently holds `sign(s)`. Set by
//!   `IntBinary{ShrS, r, Reg(s), 31}`. Invalidated by any other write to r.
//! - `value_alias[r] = a` — `r` currently holds the same value that `a`
//!   does. Initialized to self (`a == r`). Followed by `Move r <- Reg(s)`
//!   (then `value_alias[r] = value_alias[s]`).
//!
//! At each `Int64PairBinary{Mul}` we check
//! `sign_ext_of[lhs_hi] == Some(value_alias[lhs_lo])`, similarly for rhs.
//! If both sides match, rewrite to `Int64MulFromSignExt32` using the
//! canonical aliases as operands.
//!
//! ## Spill/reload
//!
//! Self-spill+self-reload (`Store [slot] U32 <- Reg(R); Load R <- [slot]
//! U32`) is value-preserving for `R`. The `TrackedSpill` table captures
//! both `sign_ext_of[R]` and `value_alias[R]` at store time so the load
//! can restore them.
//!
//! # Why state must be tracked at the point of each mul
//!
//! Inside one block there can be multiple muls with intervening writes
//! that invalidate sign-ext / alias relationships. A "build a final-state
//! map, then scan muls" approach therefore reports the wrong state for
//! early muls. This pass interleaves state updates and rewrites in a
//! single forward sweep.
//!
//! # Ordering
//!
//! Must run **after** `copy_propagate`.

use crate::collections;

use crate::vm::machine::machine_ir::{
    MachineAddr, MachineBlock, MachineInstKind, MachineIntBinaryOp, MachineMemWidth, MachineReg,
    MachineValue,
};

#[derive(Clone, Copy)]
struct TrackedSpill {
    addr: MachineAddr,
    stored_reg: MachineReg,
    sign_ext_of: Option<MachineReg>,
    value_alias: MachineReg,
}

pub(super) fn fuse_smull_sign_ext(block: &mut MachineBlock, total_reg_count: usize) {
    let has_mul = block.ops.iter().any(|inst| {
        matches!(
            inst.kind,
            MachineInstKind::Int64PairBinary {
                op: MachineIntBinaryOp::Mul,
                ..
            }
        )
    });
    if !has_mul {
        return;
    }

    let mut sign_ext_of: collections::Vec<Option<MachineReg>> =
        collections::vec![None; total_reg_count];
    let mut value_alias: collections::Vec<MachineReg> =
        (0..total_reg_count as u16).map(MachineReg).collect();
    let mut spills: collections::Vec<TrackedSpill> = collections::Vec::new();

    let canon = |r: MachineReg, alias: &[MachineReg]| -> MachineReg {
        let idx = r.0 as usize;
        if idx < alias.len() {
            alias[idx]
        } else {
            r
        }
    };

    for inst in &mut block.ops {
        // Try fusion first using the current state.
        if let MachineInstKind::Int64PairBinary {
            op: MachineIntBinaryOp::Mul,
            dst_lo,
            dst_hi,
            lhs_lo: MachineValue::Reg(lhs_lo_r),
            lhs_hi: MachineValue::Reg(lhs_hi_r),
            rhs_lo: MachineValue::Reg(rhs_lo_r),
            rhs_hi: MachineValue::Reg(rhs_hi_r),
        } = &inst.kind
        {
            let lhs_lo_canon = canon(*lhs_lo_r, &value_alias);
            let rhs_lo_canon = canon(*rhs_lo_r, &value_alias);
            let l_src = sign_ext_of.get(lhs_hi_r.0 as usize).copied().flatten();
            let r_src = sign_ext_of.get(rhs_hi_r.0 as usize).copied().flatten();
            if l_src == Some(lhs_lo_canon) && r_src == Some(rhs_lo_canon) {
                let new_dst_lo = *dst_lo;
                let new_dst_hi = *dst_hi;
                inst.kind = MachineInstKind::Int64MulFromSignExt32 {
                    dst_lo: new_dst_lo,
                    dst_hi: new_dst_hi,
                    lhs: MachineValue::Reg(lhs_lo_canon),
                    rhs: MachineValue::Reg(rhs_lo_canon),
                };
                // The mul writes dst_lo and dst_hi — invalidate.
                for r in [new_dst_lo, new_dst_hi] {
                    let idx = r.0 as usize;
                    if idx < sign_ext_of.len() {
                        sign_ext_of[idx] = None;
                        value_alias[idx] = r;
                    }
                }
                continue;
            }
        }

        // Otherwise update state from this instruction's effects.
        match &inst.kind {
            MachineInstKind::IntBinary {
                op: MachineIntBinaryOp::ShrS,
                dst,
                lhs: MachineValue::Reg(src),
                rhs: MachineValue::Imm64(31),
                ..
            } => {
                let src_canon = canon(*src, &value_alias);
                let idx = dst.0 as usize;
                if idx < sign_ext_of.len() {
                    sign_ext_of[idx] = Some(src_canon);
                    value_alias[idx] = *dst; // ShrS produces a fresh value
                }
            }
            MachineInstKind::Move {
                dst,
                src: MachineValue::Reg(src),
                ..
            } => {
                let src_canon = canon(*src, &value_alias);
                let src_sign = sign_ext_of.get(src.0 as usize).copied().flatten();
                let idx = dst.0 as usize;
                if idx < sign_ext_of.len() {
                    sign_ext_of[idx] = src_sign;
                    value_alias[idx] = src_canon;
                }
            }
            MachineInstKind::Store {
                addr,
                src: MachineValue::Reg(src_reg),
                width: MachineMemWidth::U32,
                ..
            } => {
                spills.retain(|e| !(e.addr.base == addr.base && e.addr.offset == addr.offset));
                let captured_sign = sign_ext_of.get(src_reg.0 as usize).copied().flatten();
                let captured_alias = canon(*src_reg, &value_alias);
                spills.push(TrackedSpill {
                    addr: *addr,
                    stored_reg: *src_reg,
                    sign_ext_of: captured_sign,
                    value_alias: captured_alias,
                });
            }
            MachineInstKind::Store { addr, .. } => {
                spills.retain(|e| !(e.addr.base == addr.base && e.addr.offset == addr.offset));
            }
            MachineInstKind::Load {
                dst,
                addr,
                width: MachineMemWidth::U32,
                ..
            } => {
                let recovered = spills.iter().rev().find(|e| {
                    e.addr.base == addr.base && e.addr.offset == addr.offset && e.stored_reg == *dst
                });
                let idx = dst.0 as usize;
                if idx < sign_ext_of.len() {
                    if let Some(spill) = recovered {
                        sign_ext_of[idx] = spill.sign_ext_of;
                        value_alias[idx] = spill.value_alias;
                    } else {
                        sign_ext_of[idx] = None;
                        value_alias[idx] = *dst;
                    }
                }
            }
            other => {
                super::helpers::for_each_defined_reg(other, |reg| {
                    let idx = reg.0 as usize;
                    if idx < sign_ext_of.len() {
                        sign_ext_of[idx] = None;
                        value_alias[idx] = reg;
                    }
                });
            }
        }
    }
}
