//! Constant deduplication pass.
//!
//! Within a block, if the same constant value is materialized into multiple
//! registers (via `Move { src: Imm64 }` or `FloatConst`), the second and
//! subsequent materializations are replaced with register-to-register copies.

use alloc::vec::Vec;

use crate::vm::machine::machine_ir::{
    MachineBlock, MachineFloatWidth, MachineInstKind, MachineReg, MachineStorageType,
    MachineValue,
};

use super::helpers::for_each_defined_reg;

pub(super) fn deduplicate_constants(block: &mut MachineBlock, first_fp_reg: u16) {
    let mut gp_consts: Vec<(u64, MachineReg)> = Vec::new();
    let mut fp_consts: Vec<(u64, MachineFloatWidth, MachineReg)> = Vec::new();

    for inst in &mut block.ops {
        if matches!(inst.kind, MachineInstKind::CallHelper(_)) {
            gp_consts.clear();
            fp_consts.clear();
            continue;
        }

        let mut new_gp = None;
        let mut new_fp = None;

        match &mut inst.kind {
            MachineInstKind::Move {
                dst,
                src: src @ MachineValue::Imm64(..),
                ..
            } if dst.0 < first_fp_reg => {
                let bits = match *src {
                    MachineValue::Imm64(b) => b,
                    _ => unreachable!(),
                };
                // Skip zero: fold_constants can inline Imm64(0) into consumers
                // for free (str xzr, cmp #0, etc.), so dedup would be a regression.
                if bits != 0 {
                    if let Some(&(_, prev)) =
                        gp_consts.iter().find(|(b, r)| *b == bits && *r != *dst)
                    {
                        *src = MachineValue::Reg(prev);
                    }
                    new_gp = Some((bits, *dst));
                }
            }
            MachineInstKind::FloatConst { dst, bits, width } => {
                let (d, b, w) = (*dst, *bits, *width);
                // Skip zero: fcmp d, #0.0 is free when folded as Imm64(0).
                if b != 0 {
                    if let Some(&(_, _, prev)) = fp_consts
                        .iter()
                        .find(|(bb, ww, r)| *bb == b && *ww == w && *r != d)
                    {
                        inst.kind = MachineInstKind::Move {
                            ty: match w {
                                MachineFloatWidth::F32 => MachineStorageType::Fp32,
                                MachineFloatWidth::F64 => MachineStorageType::Fp64,
                            },
                            dst: d,
                            src: MachineValue::Reg(prev),
                        };
                    }
                    new_fp = Some((b, w, d));
                }
            }
            _ => {}
        }

        // Invalidate tracking for any register redefined by this instruction.
        for_each_defined_reg(&inst.kind, |def| {
            gp_consts.retain(|(_, r)| *r != def);
            fp_consts.retain(|(_, _, r)| *r != def);
        });

        if let Some(e) = new_gp {
            gp_consts.push(e);
        }
        if let Some(e) = new_fp {
            fp_consts.push(e);
        }
    }
}
