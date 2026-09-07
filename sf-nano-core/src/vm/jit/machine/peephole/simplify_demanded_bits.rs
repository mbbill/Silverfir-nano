//! Discard unobserved constant bits, then simplify redundant bitfield masks.
//!
//! Proofs are local to a block. Every escaping register starts fully demanded;
//! opaque operations reset the proof. No memory operation is moved or removed.

use crate::collections;
use crate::vm::jit::backend::BackendConfig;
use crate::vm::jit::machine::machine_ir::{
    MachineBlock, MachineInstKind, MachineIntBinaryOp, MachineIntWidth, MachineReg, MachineShiftOp,
    MachineStorageType, MachineValue, MACHINE_FP_REG,
};

fn low_bits(bits: u8) -> u64 {
    u64::MAX.checked_shr(u32::from(64 - bits)).unwrap_or(0)
}

fn width_bits(width: MachineIntWidth) -> u8 {
    match width {
        MachineIntWidth::I32 => 32,
        MachineIntWidth::I64 => 64,
    }
}

fn read(state: &[u64], reg: MachineReg) -> u64 {
    state.get(usize::from(reg.0)).copied().unwrap_or(u64::MAX)
}

fn write(state: &mut [u64], reg: MachineReg, bits: u64) {
    if let Some(slot) = state.get_mut(usize::from(reg.0)) {
        *slot = bits;
    }
}

fn value_bits(state: &[u64], value: MachineValue) -> u64 {
    match value {
        MachineValue::Reg(reg) => read(state, reg),
        MachineValue::Imm64(bits) => bits,
        _ => u64::MAX,
    }
}

fn demand(state: &mut [u64], value: MachineValue, bits: u64) {
    if let MachineValue::Reg(reg) = value {
        write(state, reg, read(state, reg) | bits);
    }
}

fn take(state: &mut [u64], dst: MachineReg) -> u64 {
    let bits = read(state, dst);
    write(state, dst, 0);
    bits
}

fn scalar_demand(bits: u64, width: MachineIntWidth) -> u64 {
    // On a sign-extending backend, observing upper native-word bits can
    // observe the i32 sign bit. Conservatively require all scalar bits.
    let mask = low_bits(width_bits(width));
    if bits & !mask != 0 {
        mask
    } else {
        bits
    }
}

fn unshift(bits: u64, shift: MachineShiftOp, amount: u8, width: MachineIntWidth) -> u64 {
    let mask = low_bits(width_bits(width));
    let amount = u32::from(amount & (width_bits(width) - 1));
    match shift {
        MachineShiftOp::Lsl => bits >> amount,
        MachineShiftOp::Lsr => (bits << amount) & mask,
        // Signed shifts and rotations retain the original operand conservatively.
        _ => mask,
    }
}

pub(super) fn simplify_demanded_bits(
    block: &mut MachineBlock,
    config: BackendConfig,
    scratch: &mut collections::Vec<u64>,
) {
    // Narrow constants only where doing so can remove an extraction mask.
    // Most blocks have no such operation and need no backward dataflow scan.
    if !block
        .ops
        .iter()
        .any(|inst| matches!(inst.kind, MachineInstKind::BitfieldExtractU { .. }))
    {
        return;
    }
    scratch.resize(usize::from(config.total_reg_count()), u64::MAX);
    scratch.fill(u64::MAX);
    let mut changed = false;
    for inst in block.ops.iter_mut().rev() {
        match &mut inst.kind {
            MachineInstKind::Move {
                ty: MachineStorageType::GpWord,
                dst,
                src,
                ..
            } => {
                let needed = take(scratch, *dst);
                demand(scratch, *src, needed);
            }
            MachineInstKind::Select {
                ty: MachineStorageType::GpWord,
                dst,
                on_true,
                on_false,
                cond,
            } => {
                let needed = take(scratch, *dst);
                demand(scratch, *on_true, needed);
                demand(scratch, *on_false, needed);
                demand(scratch, *cond, u64::MAX);
            }
            MachineInstKind::BitfieldExtractU {
                width,
                dst,
                src,
                lsb,
                bits,
            } if *width == MachineIntWidth::I32 || config.gp_unit_bytes == 8 => {
                let needed = scalar_demand(take(scratch, *dst), *width);
                demand(
                    scratch,
                    MachineValue::Reg(*src),
                    (needed & low_bits(*bits)) << *lsb,
                );
            }
            MachineInstKind::IntBinary {
                width,
                op,
                dst,
                lhs,
                rhs,
            } if (*width == MachineIntWidth::I32 || config.gp_unit_bytes == 8)
                && matches!(
                    op,
                    MachineIntBinaryOp::And | MachineIntBinaryOp::Or | MachineIntBinaryOp::Xor
                ) =>
            {
                let needed = scalar_demand(take(scratch, *dst), *width);
                let mask = low_bits(width_bits(*width));
                for value in [&mut *lhs, &mut *rhs] {
                    if let MachineValue::Imm64(bits) = value {
                        let narrowed = *bits & (needed | !mask);
                        changed |= narrowed != *bits;
                        *bits = narrowed;
                    }
                }
                let operand_demand = |other| match (*op, other) {
                    (MachineIntBinaryOp::And, MachineValue::Imm64(bits)) => needed & bits,
                    (MachineIntBinaryOp::Or, MachineValue::Imm64(bits)) => needed & !bits,
                    _ => needed,
                };
                demand(scratch, *lhs, operand_demand(*rhs));
                demand(scratch, *rhs, operand_demand(*lhs));
            }
            MachineInstKind::IntBinaryShifted {
                width,
                op,
                dst,
                lhs,
                rhs,
                shift,
                amount,
            } if (*width == MachineIntWidth::I32 || config.gp_unit_bytes == 8)
                && matches!(
                    op,
                    MachineIntBinaryOp::And | MachineIntBinaryOp::Or | MachineIntBinaryOp::Xor
                ) =>
            {
                let needed = scalar_demand(take(scratch, *dst), *width);
                demand(scratch, MachineValue::Reg(*lhs), needed);
                demand(
                    scratch,
                    MachineValue::Reg(*rhs),
                    unshift(needed, *shift, *amount, *width),
                );
            }
            MachineInstKind::Store {
                ty: MachineStorageType::GpWord,
                addr,
                src,
                ..
            } if addr.base == MACHINE_FP_REG => {
                demand(scratch, *src, u64::MAX);
                demand(scratch, MachineValue::Reg(addr.base), u64::MAX);
            }
            _ => scratch.fill(u64::MAX),
        }
    }
    if !changed {
        return;
    }

    // Here bits describe possible ones, so their complement is known zero.
    scratch.fill(u64::MAX);
    for inst in &mut block.ops {
        let (dst, possible) = match inst.kind {
            MachineInstKind::Move {
                ty: MachineStorageType::GpWord,
                dst,
                src,
                ..
            } => (dst, value_bits(scratch, src)),
            MachineInstKind::Select {
                ty: MachineStorageType::GpWord,
                dst,
                on_true,
                on_false,
                ..
            } => (
                dst,
                value_bits(scratch, on_true) | value_bits(scratch, on_false),
            ),
            MachineInstKind::IntBinary {
                width,
                op,
                dst,
                lhs,
                rhs,
            } if width == MachineIntWidth::I32 || config.gp_unit_bytes == 8 => {
                let mask = low_bits(width_bits(width));
                let a = value_bits(scratch, lhs) & mask;
                let b = value_bits(scratch, rhs) & mask;
                let possible = match (op, rhs) {
                    (MachineIntBinaryOp::And, _) => a & b,
                    (MachineIntBinaryOp::Or | MachineIntBinaryOp::Xor, _) => a | b,
                    (MachineIntBinaryOp::ShrU, MachineValue::Imm64(n)) => {
                        a >> (n & (u64::from(width_bits(width)) - 1))
                    }
                    (MachineIntBinaryOp::Shl, MachineValue::Imm64(n)) => {
                        a << (n & (u64::from(width_bits(width)) - 1))
                    }
                    _ => mask,
                };
                (dst, native_result_bits(possible & mask, width, config))
            }
            MachineInstKind::BitfieldExtractU {
                width,
                dst,
                src,
                lsb,
                bits,
            } if width == MachineIntWidth::I32 || config.gp_unit_bytes == 8 => {
                let input = read(scratch, src) & low_bits(width_bits(width));
                if lsb != 0 && input & !low_bits(lsb + bits) == 0 {
                    inst.kind = MachineInstKind::IntBinary {
                        width,
                        op: MachineIntBinaryOp::ShrU,
                        dst,
                        lhs: MachineValue::Reg(src),
                        rhs: MachineValue::Imm64(u64::from(lsb)),
                    };
                }
                (
                    dst,
                    native_result_bits((input >> lsb) & low_bits(bits), width, config),
                )
            }
            MachineInstKind::IntBinaryShifted {
                width,
                op,
                dst,
                lhs,
                rhs,
                shift,
                amount,
            } if width == MachineIntWidth::I32 || config.gp_unit_bytes == 8 => {
                let mask = low_bits(width_bits(width));
                let a = read(scratch, lhs) & mask;
                let b = read(scratch, rhs) & mask;
                let amount = amount & (width_bits(width) - 1);
                let shifted = match shift {
                    MachineShiftOp::Lsl => (b << amount) & mask,
                    MachineShiftOp::Lsr => b >> amount,
                    _ => mask,
                };
                let possible = match op {
                    MachineIntBinaryOp::And => a & shifted,
                    MachineIntBinaryOp::Or | MachineIntBinaryOp::Xor => a | shifted,
                    _ => mask,
                };
                (dst, native_result_bits(possible, width, config))
            }
            MachineInstKind::Store {
                ty: MachineStorageType::GpWord,
                addr,
                ..
            } if addr.base == MACHINE_FP_REG => continue,
            _ => {
                scratch.fill(u64::MAX);
                continue;
            }
        };
        write(scratch, dst, possible);
    }
}

fn native_result_bits(bits: u64, width: MachineIntWidth, config: BackendConfig) -> u64 {
    if width == MachineIntWidth::I32 && config.gp_unit_bytes == 8 && !config.gp32_defs_zero_extend {
        bits | !u64::from(u32::MAX)
    } else {
        bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::jit::machine::machine_ir::{
        MachineAddr, MachineBlockId, MachineCallRuntime, MachineConstId, MachineInst,
        MachineMemWidth, MachineRegOwner, MachineTerminator,
    };

    fn inst(kind: MachineInstKind) -> MachineInst {
        MachineInst { kind }
    }

    fn copy(dst: u16, value: u64) -> MachineInst {
        inst(MachineInstKind::Move {
            owner: MachineRegOwner::CachedCell,
            ty: MachineStorageType::GpWord,
            dst: MachineReg(dst),
            src: MachineValue::Imm64(value),
        })
    }

    fn fixture(width: MachineIntWidth, op: MachineIntBinaryOp, shifted: bool) -> MachineBlock {
        let mut ops = collections::vec![
            inst(MachineInstKind::IntBinary {
                width,
                op: MachineIntBinaryOp::And,
                dst: MachineReg(5),
                lhs: MachineValue::Reg(MachineReg(4)),
                rhs: MachineValue::Imm64(255),
            }),
            inst(MachineInstKind::IntBinary {
                width,
                op,
                dst: MachineReg(6),
                lhs: MachineValue::Reg(MachineReg(5)),
                rhs: MachineValue::Imm64(0xabcd_0000_ffff_0041),
            }),
        ];
        if shifted {
            ops.push(inst(MachineInstKind::IntBinaryShifted {
                width,
                op: MachineIntBinaryOp::Xor,
                dst: MachineReg(7),
                lhs: MachineReg(6),
                rhs: MachineReg(4),
                shift: MachineShiftOp::Lsr,
                amount: 3,
            }));
            ops.push(inst(MachineInstKind::IntBinary {
                width,
                op: MachineIntBinaryOp::And,
                dst: MachineReg(7),
                lhs: MachineValue::Reg(MachineReg(7)),
                rhs: MachineValue::Imm64(1),
            }));
        }
        ops.extend([
            inst(MachineInstKind::Select {
                ty: MachineStorageType::GpWord,
                dst: MachineReg(5),
                on_true: MachineValue::Reg(MachineReg(6)),
                on_false: MachineValue::Reg(MachineReg(5)),
                cond: MachineValue::Reg(MachineReg(if shifted { 7 } else { 8 })),
            }),
            inst(MachineInstKind::BitfieldExtractU {
                width,
                dst: MachineReg(6),
                src: MachineReg(5),
                lsb: 3,
                bits: 5,
            }),
            copy(5, 0),
        ]);
        MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops,
            terminator: MachineTerminator::Return,
        }
    }

    fn execute(block: &MachineBlock, mut regs: [u64; 16], config: BackendConfig) -> [u64; 16] {
        let value = |regs: &[u64; 16], v| match v {
            MachineValue::Reg(r) => regs[usize::from(r.0)],
            MachineValue::Imm64(bits) => bits,
            MachineValue::ReservedReg(_) => panic!("not a readable value"),
        };
        let scalar = |bits: u64, width| {
            if width == MachineIntWidth::I64 {
                bits
            } else if config.gp_unit_bytes == 8 && !config.gp32_defs_zero_extend {
                bits as i32 as i64 as u64
            } else {
                u64::from(bits as u32)
            }
        };
        for op in &block.ops {
            let (dst, result) = match op.kind {
                MachineInstKind::Move { dst, src, .. } => (dst, value(&regs, src)),
                MachineInstKind::Select {
                    dst,
                    on_true,
                    on_false,
                    cond,
                    ..
                } => (
                    dst,
                    value(
                        &regs,
                        if value(&regs, cond) as u32 != 0 {
                            on_true
                        } else {
                            on_false
                        },
                    ),
                ),
                MachineInstKind::IntBinary {
                    width,
                    op,
                    dst,
                    lhs,
                    rhs,
                } => {
                    let a = value(&regs, lhs);
                    let b = value(&regs, rhs);
                    let bits = match op {
                        MachineIntBinaryOp::And => a & b,
                        MachineIntBinaryOp::Or => a | b,
                        MachineIntBinaryOp::Xor => a ^ b,
                        MachineIntBinaryOp::ShrU => {
                            (a & low_bits(width_bits(width)))
                                >> (b & (u64::from(width_bits(width)) - 1))
                        }
                        _ => panic!("unsupported test operation"),
                    };
                    (dst, scalar(bits, width))
                }
                MachineInstKind::IntBinaryShifted {
                    width,
                    op: MachineIntBinaryOp::Xor,
                    dst,
                    lhs,
                    rhs,
                    shift: MachineShiftOp::Lsr,
                    amount,
                } => {
                    let a = regs[usize::from(lhs.0)];
                    let b = regs[usize::from(rhs.0)] & low_bits(width_bits(width));
                    (dst, scalar(a ^ (b >> amount), width))
                }
                MachineInstKind::BitfieldExtractU {
                    width,
                    dst,
                    src,
                    lsb,
                    bits,
                } => {
                    let input = regs[usize::from(src.0)] & low_bits(width_bits(width));
                    (dst, scalar((input >> lsb) & low_bits(bits), width))
                }
                _ => panic!("unsupported test operation"),
            };
            regs[usize::from(dst.0)] = if config.gp_unit_bytes == 4 {
                u64::from(result as u32)
            } else {
                result
            };
        }
        regs
    }

    #[test]
    fn randomized_inputs_preserve_every_escaping_lane_at_both_scalar_widths() {
        let mut random = 0x6217_c19d_039b_a5e3u64;
        for gp_bytes in [4, 8] {
            for zero_extend in [false, true] {
                let mut config = BackendConfig::new(gp_bytes, 10, 0, 0);
                config.gp32_defs_zero_extend = zero_extend;
                for width in [MachineIntWidth::I32, MachineIntWidth::I64] {
                    if width == MachineIntWidth::I64 && gp_bytes == 4 {
                        continue;
                    }
                    for op in [
                        MachineIntBinaryOp::And,
                        MachineIntBinaryOp::Or,
                        MachineIntBinaryOp::Xor,
                    ] {
                        for shifted in [false, true] {
                            let before = fixture(width, op, shifted);
                            let mut after = before.clone();
                            simplify_demanded_bits(
                                &mut after,
                                config,
                                &mut collections::Vec::new(),
                            );
                            assert_ne!(after, before);
                            assert!(!after.ops.iter().any(|i| matches!(
                                i.kind,
                                MachineInstKind::BitfieldExtractU { .. }
                            )));
                            for _ in 0..1000 {
                                let mut regs = [0; 16];
                                for r in &mut regs {
                                    random ^= random << 13;
                                    random ^= random >> 7;
                                    random ^= random << 17;
                                    *r = if gp_bytes == 4 {
                                        u64::from(random as u32)
                                    } else {
                                        random
                                    };
                                }
                                // Exercise either select arm, including high condition bits.
                                regs[8] &= 0xffff_ffff_0000_0001;
                                assert_eq!(
                                    execute(&before, regs, config),
                                    execute(&after, regs, config)
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn keeps_bits_observed_by_published_stores_barriers_or_escaping_registers() {
        for case in 0..3 {
            let mut block = fixture(MachineIntWidth::I32, MachineIntBinaryOp::Xor, false);
            match case {
                0 => block.ops.insert(
                    3,
                    inst(MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: MachineAddr {
                            base: MACHINE_FP_REG,
                            offset: 0,
                        },
                        width: MachineMemWidth::U64,
                        src: MachineValue::Reg(MachineReg(5)),
                    }),
                ),
                1 => block.ops.insert(
                    3,
                    inst(MachineInstKind::CallRuntime(MachineCallRuntime {
                        metadata: MachineConstId(0),
                    })),
                ),
                2 => {
                    block.ops.pop();
                }
                _ => unreachable!(),
            }
            let before = block.clone();
            simplify_demanded_bits(
                &mut block,
                BackendConfig::new(8, 10, 0, 0),
                &mut collections::Vec::new(),
            );
            assert_eq!(block, before, "observable bits in case {case}");
        }
    }
}
