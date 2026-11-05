//! Fold a chain of immediate adds into one.
//!
//! Wasm producers address a struct field of a struct field as
//! `i32.add(i32.add(p, #8), #40)`, and the chain reaches MachineIR intact:
//!
//! ```text
//! i32.Add t <- x #c1
//! i32.Add d <- t #c2
//! ```
//!
//! Integer addition wraps at the operation width, so the pair is exactly
//! `i32.Add d <- x #(c1 + c2)` whenever nothing else reads `t`. Subtraction
//! of an immediate folds the same way with the negated constant.
//!
//! Only same-block pairs within a short window are rewritten. `t` must be
//! read by nothing between the two adds and be dead after the second, or be
//! the second add's own destination. `x` must keep its value up to the
//! second add. The first add is removed.

use crate::vm::jit::machine::machine_ir::{
    MachineBlock, MachineInstKind, MachineIntBinaryOp, MachineIntWidth, MachineReg, MachineValue,
};

use super::helpers::{inst_defines, inst_uses_value, reg_live_after};

/// How far back from the second add to look for the first.
const WINDOW: usize = 4;

pub(super) fn fold_add_chains(block: &mut MachineBlock) {
    let mut removed = crate::collections::Vec::new();
    for second in 1..block.ops.len() {
        let Some((width, dst, t, c2)) = offset_add(&block.ops[second].kind) else {
            continue;
        };
        let Some(first) = first_add(block, second, t) else {
            continue;
        };
        let Some((first_width, first_dst, x, c1)) = offset_add(&block.ops[first].kind) else {
            continue;
        };
        debug_assert_eq!(first_dst, t);
        // Cheapest checks first: the liveness scan walks the rest of the
        // block.
        if first_width != width
            || removed.contains(&first)
            || block.ops[first + 1..second]
                .iter()
                .any(|op| inst_defines(&op.kind, x))
            || (t != dst && reg_live_after(&block.ops[second + 1..], &block.terminator, t))
        {
            continue;
        }
        block.ops[second].kind = MachineInstKind::IntBinary {
            width,
            op: MachineIntBinaryOp::Add,
            dst,
            lhs: MachineValue::Reg(x),
            rhs: MachineValue::Imm64(wrap(width, c1.wrapping_add(c2))),
        };
        removed.push(first);
    }
    if !removed.is_empty() {
        let mut index = 0;
        block.ops.retain(|_| {
            let keep = !removed.contains(&index);
            index += 1;
            keep
        });
    }
}

/// The nearest earlier immediate add in the window that defines `t`,
/// provided no op between it and `second` reads or writes `t`.
fn first_add(block: &MachineBlock, second: usize, t: MachineReg) -> Option<usize> {
    let start = second.saturating_sub(WINDOW);
    // Most adds have no such partner; the pattern match rejects them before
    // any operand walk.
    let first = (start..second).rev().find(
        |&index| matches!(offset_add(&block.ops[index].kind), Some((_, dst, _, _)) if dst == t),
    )?;
    block.ops[first + 1..second]
        .iter()
        .all(|op| !inst_defines(&op.kind, t) && !inst_uses_value(&op.kind, t))
        .then_some(first)
}

/// `dst <- src ± imm` as `(width, dst, src, signed addend)`, the addend
/// already negated for a subtraction.
fn offset_add(kind: &MachineInstKind) -> Option<(MachineIntWidth, MachineReg, MachineReg, u64)> {
    let MachineInstKind::IntBinary {
        width,
        op,
        dst,
        lhs: MachineValue::Reg(src),
        rhs: MachineValue::Imm64(imm),
    } = *kind
    else {
        return None;
    };
    let addend = match op {
        MachineIntBinaryOp::Add => imm,
        MachineIntBinaryOp::Sub => imm.wrapping_neg(),
        _ => return None,
    };
    Some((width, dst, src, addend))
}

/// An I32 immediate is carried as its zero-extended 32-bit pattern.
fn wrap(width: MachineIntWidth, value: u64) -> u64 {
    match width {
        MachineIntWidth::I32 => u64::from(value as u32),
        MachineIntWidth::I64 => value,
    }
}

#[cfg(test)]
mod tests {
    use super::fold_add_chains;
    use crate::collections;
    use crate::vm::jit::machine::machine_ir::{
        MachineBlock, MachineBlockId, MachineEdge, MachineInst, MachineInstKind,
        MachineIntBinaryOp, MachineIntWidth, MachineReg, MachineTerminator, MachineValue,
    };

    fn add(
        width: MachineIntWidth,
        op: MachineIntBinaryOp,
        dst: u16,
        src: u16,
        imm: u64,
    ) -> MachineInst {
        MachineInst {
            kind: MachineInstKind::IntBinary {
                width,
                op,
                dst: MachineReg(dst),
                lhs: MachineValue::Reg(MachineReg(src)),
                rhs: MachineValue::Imm64(imm),
            },
        }
    }

    fn i32_add(dst: u16, src: u16, imm: u64) -> MachineInst {
        add(MachineIntWidth::I32, MachineIntBinaryOp::Add, dst, src, imm)
    }

    fn block(ops: collections::Vec<MachineInst>, live_out: &[u16]) -> MachineBlock {
        MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops,
            terminator: MachineTerminator::Jump(MachineEdge {
                target: MachineBlockId(1),
                args: live_out
                    .iter()
                    .map(|&reg| MachineValue::Reg(MachineReg(reg)))
                    .collect(),
            }),
        }
    }

    #[test]
    fn folds_a_chain_into_its_destination() {
        // i32.Add r4 <- r11 8; i32.Add r4 <- r4 40
        let mut b = block(
            collections::vec![i32_add(4, 11, 8), i32_add(4, 4, 40)],
            &[4],
        );
        fold_add_chains(&mut b);
        assert_eq!(b.ops, collections::vec![i32_add(4, 11, 48)]);
    }

    #[test]
    fn folds_through_a_dead_temporary_and_wraps() {
        let mut b = block(
            collections::vec![
                add(MachineIntWidth::I32, MachineIntBinaryOp::Sub, 6, 5, 0x10),
                i32_add(7, 6, 0xFFFF_FFF8),
            ],
            &[7],
        );
        fold_add_chains(&mut b);
        // x - 16 + (-8) = x - 24, as a 32-bit pattern.
        assert_eq!(
            b.ops,
            collections::vec![i32_add(7, 5, u64::from(24u32.wrapping_neg()))]
        );

        let mut b = block(
            collections::vec![
                add(
                    MachineIntWidth::I64,
                    MachineIntBinaryOp::Add,
                    6,
                    5,
                    u64::MAX
                ),
                add(MachineIntWidth::I64, MachineIntBinaryOp::Sub, 7, 6, 2),
            ],
            &[7],
        );
        fold_add_chains(&mut b);
        assert_eq!(
            b.ops,
            collections::vec![add(
                MachineIntWidth::I64,
                MachineIntBinaryOp::Add,
                7,
                5,
                3u64.wrapping_neg()
            )]
        );
    }

    #[test]
    fn keeps_a_temporary_that_is_read_again() {
        // r6 is still live out of the block.
        let original = collections::vec![i32_add(6, 5, 8), i32_add(7, 6, 40)];
        let mut b = block(original.clone(), &[6, 7]);
        fold_add_chains(&mut b);
        assert_eq!(b.ops, original);

        // r6 is read between the two adds.
        let original = collections::vec![i32_add(6, 5, 8), i32_add(9, 6, 1), i32_add(7, 6, 40)];
        let mut b = block(original.clone(), &[7, 9]);
        fold_add_chains(&mut b);
        assert_eq!(b.ops, original);

        // r6 is redefined between the two adds by something other than an
        // immediate add, so the second add reads that value instead.
        let original = collections::vec![
            i32_add(6, 5, 8),
            add(MachineIntWidth::I32, MachineIntBinaryOp::Mul, 6, 9, 3),
            i32_add(7, 6, 40),
        ];
        let mut b = block(original.clone(), &[7]);
        fold_add_chains(&mut b);
        assert_eq!(b.ops, original);
    }

    #[test]
    fn keeps_a_chain_whose_source_changes_or_whose_widths_differ() {
        // r5 is redefined between the two adds.
        let original = collections::vec![i32_add(6, 5, 8), i32_add(5, 9, 1), i32_add(7, 6, 40)];
        let mut b = block(original.clone(), &[5, 7]);
        fold_add_chains(&mut b);
        assert_eq!(b.ops, original);

        let original = collections::vec![
            add(MachineIntWidth::I64, MachineIntBinaryOp::Add, 6, 5, 8),
            i32_add(7, 6, 40),
        ];
        let mut b = block(original.clone(), &[7]);
        fold_add_chains(&mut b);
        assert_eq!(b.ops, original);
    }

    #[test]
    fn folds_a_self_increment_chain() {
        // r4 <- r4 + 8; r4 <- r4 + 40: the first add's source is its own
        // destination, which still holds the original value once it goes.
        let mut b = block(collections::vec![i32_add(4, 4, 8), i32_add(4, 4, 40)], &[4]);
        fold_add_chains(&mut b);
        assert_eq!(b.ops, collections::vec![i32_add(4, 4, 48)]);
    }
}
