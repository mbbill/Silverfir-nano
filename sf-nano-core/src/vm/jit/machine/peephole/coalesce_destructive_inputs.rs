//! Reuse a dead linear input for a short, entirely local integer result.
//!
//! Two-operand backends otherwise copy that input before the ALU operation.
//! Ownership, both lifetimes, and a closed set of pure instructions must all
//! agree. No cached input, fixed lane, call boundary, or CFG argument changes.

use crate::vm::jit::{
    backend::BackendConfig,
    machine::{
        machine_ir::{
            MachineBlock, MachineInst, MachineInstKind, MachineIntBinaryOp, MachineReg,
            MachineRegOwner, MachineStorageType, MachineTerminator, MachineValue,
        },
        ownership::DynamicOwnershipTracker,
    },
};

use super::helpers::{inst_uses_value, terminator_uses_reg};

// Limit proof work per candidate independently of the size of the block.
const LOOKAHEAD: usize = 16;

fn candidate(kind: &MachineInstKind) -> Option<(MachineReg, MachineReg)> {
    let (op, dst, lhs) = match *kind {
        MachineInstKind::IntBinary {
            op,
            dst,
            lhs: MachineValue::Reg(lhs),
            rhs,
            ..
        } => {
            if matches!(
                op,
                MachineIntBinaryOp::And
                    | MachineIntBinaryOp::Or
                    | MachineIntBinaryOp::Xor
                    | MachineIntBinaryOp::Mul
            ) && rhs == MachineValue::Reg(dst)
            {
                // The backend can already use the other commutative input.
                return None;
            }
            (op, dst, lhs)
        }
        MachineInstKind::IntBinaryShifted { op, dst, lhs, .. } => (op, dst, lhs),
        _ => return None,
    };
    if dst == lhs
        || !matches!(
            op,
            MachineIntBinaryOp::Sub
                | MachineIntBinaryOp::Mul
                | MachineIntBinaryOp::And
                | MachineIntBinaryOp::Or
                | MachineIntBinaryOp::Xor
                | MachineIntBinaryOp::Shl
                | MachineIntBinaryOp::ShrS
                | MachineIntBinaryOp::ShrU
                | MachineIntBinaryOp::Rotl
                | MachineIntBinaryOp::Rotr
        )
    {
        return None;
    }
    Some((dst, lhs))
}

fn pure_destination(kind: &MachineInstKind) -> Option<MachineReg> {
    let dst = match *kind {
        MachineInstKind::Move {
            owner: MachineRegOwner::LinearValue,
            ty: MachineStorageType::GpWord | MachineStorageType::GpI64,
            dst,
            ..
        }
        | MachineInstKind::IntUnary { dst, .. }
        | MachineInstKind::IntCompare { dst, .. }
        | MachineInstKind::BitfieldExtractU { dst, .. }
        | MachineInstKind::TestBits { dst, .. }
        | MachineInstKind::Select {
            ty: MachineStorageType::GpWord | MachineStorageType::GpI64,
            dst,
            ..
        } => Some(dst),
        MachineInstKind::IntBinary { op, dst, .. }
        | MachineInstKind::IntBinaryShifted { op, dst, .. }
            if !matches!(
                op,
                MachineIntBinaryOp::DivS
                    | MachineIntBinaryOp::DivU
                    | MachineIntBinaryOp::RemS
                    | MachineIntBinaryOp::RemU
            ) =>
        {
            Some(dst)
        }
        _ => None,
    }?;
    (dst.0 >= BackendConfig::FIXED).then_some(dst)
}

fn dies_before_use(ops: &[MachineInst], term: &MachineTerminator, reg: MachineReg) -> bool {
    for inst in ops.iter().take(LOOKAHEAD) {
        let Some(dst) = pure_destination(&inst.kind) else {
            return false;
        };
        if inst_uses_value(&inst.kind, reg) {
            return false;
        }
        if dst == reg {
            return true;
        }
    }
    ops.len() <= LOOKAHEAD && !terminator_uses_reg(term, reg)
}

fn interval_end(
    block: &MachineBlock,
    start: usize,
    tmp: MachineReg,
    input: MachineReg,
) -> Option<usize> {
    let limit = block.ops.len().min(start + 1 + LOOKAHEAD);
    for index in start + 1..limit {
        let kind = &block.ops[index].kind;
        let dst = pure_destination(kind)?;
        // This checks reads before defs, including read-modify-write forms.
        if inst_uses_value(kind, input) {
            return None;
        }
        if dst == input {
            return dies_before_use(&block.ops[index + 1..], &block.terminator, tmp)
                .then_some(index + 1);
        }
        if dst == tmp && !inst_uses_value(kind, tmp) {
            return dies_before_use(&block.ops[index..], &block.terminator, input).then_some(index);
        }
    }
    (limit == block.ops.len()
        && !terminator_uses_reg(&block.terminator, input)
        && !terminator_uses_reg(&block.terminator, tmp))
    .then_some(limit)
}

fn rename_value(value: &mut MachineValue, old: MachineReg, new: MachineReg) {
    if *value == MachineValue::Reg(old) {
        *value = MachineValue::Reg(new);
    }
}

fn rename_reg(reg: &mut MachineReg, old: MachineReg, new: MachineReg) {
    if *reg == old {
        *reg = new;
    }
}

fn rename(kind: &mut MachineInstKind, old: MachineReg, new: MachineReg, sources: bool) {
    match kind {
        MachineInstKind::Move { dst, src, .. } | MachineInstKind::IntUnary { dst, src, .. } => {
            if sources {
                rename_value(src, old, new);
            }
            rename_reg(dst, old, new);
        }
        MachineInstKind::IntBinary { dst, lhs, rhs, .. }
        | MachineInstKind::IntCompare { dst, lhs, rhs, .. } => {
            if sources {
                rename_value(lhs, old, new);
                rename_value(rhs, old, new);
            }
            rename_reg(dst, old, new);
        }
        MachineInstKind::IntBinaryShifted { dst, lhs, rhs, .. } => {
            if sources {
                rename_reg(lhs, old, new);
                rename_reg(rhs, old, new);
            }
            rename_reg(dst, old, new);
        }
        MachineInstKind::BitfieldExtractU { dst, src, .. } => {
            if sources {
                rename_reg(src, old, new);
            }
            rename_reg(dst, old, new);
        }
        MachineInstKind::TestBits { dst, src, mask, .. } => {
            if sources {
                rename_reg(src, old, new);
                rename_value(mask, old, new);
            }
            rename_reg(dst, old, new);
        }
        MachineInstKind::Select {
            dst,
            cond,
            on_true,
            on_false,
            ..
        } => {
            if sources {
                rename_value(cond, old, new);
                rename_value(on_true, old, new);
                rename_value(on_false, old, new);
            }
            rename_reg(dst, old, new);
        }
        _ => unreachable!("coalescing only rewrites proven pure scalar instructions"),
    }
}

pub(super) fn coalesce(
    block: &mut MachineBlock,
    config: BackendConfig,
    ownership: &mut DynamicOwnershipTracker,
) {
    ownership.reset_for_block(block, config);
    let volatile_end = BackendConfig::FIXED + u16::from(config.gp_volatile_dynamic);
    let allocatable_end = BackendConfig::FIXED + u16::from(config.allocatable_gp_dynamic_budget());
    for index in 0..block.ops.len() {
        if let Some((tmp, input)) = candidate(&block.ops[index].kind) {
            if (BackendConfig::FIXED..volatile_end).contains(&input.0)
                && (BackendConfig::FIXED..allocatable_end).contains(&tmp.0)
                && ownership.is_linear_value_reg(input, config)
            {
                if let Some(end) = interval_end(block, index, tmp, input) {
                    // The original destination can also be an input to this
                    // first operation. Only its definition changes here.
                    rename(&mut block.ops[index].kind, tmp, input, false);
                    for inst in &mut block.ops[index + 1..end] {
                        rename(&mut inst.kind, tmp, input, true);
                    }
                }
            }
        }
        ownership.apply_inst(&block.ops[index].kind, config);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        collections,
        vm::jit::machine::machine_ir::{
            MachineAddr, MachineBlockId, MachineBlockParam, MachineBranchCond, MachineCallRuntime,
            MachineConstId, MachineIntWidth, MachineMemWidth, MachineResultSrc, MachineReturnValue,
            MachineShiftOp, MachineTrapKind, MACHINE_FP_REG,
        },
    };

    fn config() -> BackendConfig {
        BackendConfig::with_volatility(8, 5, 2, 1, 2, 0, 4, 2, true, 3).with_destructive_gp_binary()
    }

    fn block(width: MachineIntWidth) -> MachineBlock {
        MachineBlock {
            id: MachineBlockId(0),
            params: (4..8)
                .map(|r| MachineBlockParam::gp_word(MachineReg(r)))
                .collect(),
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::IntBinary {
                        width,
                        op: MachineIntBinaryOp::Xor,
                        dst: MachineReg(7),
                        lhs: MachineValue::Reg(MachineReg(4)),
                        rhs: MachineValue::Reg(MachineReg(5)),
                    }
                },
                MachineInst {
                    kind: MachineInstKind::IntBinary {
                        width,
                        op: MachineIntBinaryOp::And,
                        dst: MachineReg(7),
                        lhs: MachineValue::Reg(MachineReg(7)),
                        rhs: MachineValue::Imm64(13),
                    }
                },
                MachineInst {
                    kind: MachineInstKind::Select {
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(4),
                        cond: MachineValue::Reg(MachineReg(7)),
                        on_true: MachineValue::Reg(MachineReg(5)),
                        on_false: MachineValue::Reg(MachineReg(6)),
                    }
                },
            ],
            terminator: MachineTerminator::ReturnScalar {
                value: MachineReturnValue::ScalarGp {
                    src: MachineResultSrc::Reg(MachineReg(4)),
                    ty: MachineStorageType::GpWord,
                },
            },
        }
    }

    fn optimize(block: &mut MachineBlock) {
        let config = config();
        coalesce(
            block,
            config,
            &mut DynamicOwnershipTracker::new(config.total_reg_count() as usize),
        );
    }

    fn read(value: MachineValue, regs: &[u64; 32]) -> u64 {
        match value {
            MachineValue::Reg(r) => regs[r.0 as usize],
            MachineValue::Imm64(value) => value,
            MachineValue::ReservedReg(_) => panic!("test interpreter needs ordinary registers"),
        }
    }

    fn execute(block: &MachineBlock, mut regs: [u64; 32]) -> u64 {
        for inst in &block.ops {
            match inst.kind {
                MachineInstKind::IntBinary {
                    width,
                    op,
                    dst,
                    lhs,
                    rhs,
                } => {
                    let lhs = read(lhs, &regs);
                    let rhs = read(rhs, &regs);
                    let result = match op {
                        MachineIntBinaryOp::Xor => lhs ^ rhs,
                        MachineIntBinaryOp::And => lhs & rhs,
                        MachineIntBinaryOp::Sub => lhs.wrapping_sub(rhs),
                        _ => panic!("unexpected test binary operation"),
                    };
                    regs[dst.0 as usize] = if width == MachineIntWidth::I32 {
                        result as u32 as u64
                    } else {
                        result
                    };
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
                    let lhs = regs[lhs.0 as usize];
                    let rhs = regs[rhs.0 as usize];
                    regs[dst.0 as usize] = if width == MachineIntWidth::I32 {
                        ((lhs as u32) ^ ((rhs as u32) >> amount)) as u64
                    } else {
                        lhs ^ (rhs >> amount)
                    };
                }
                MachineInstKind::Move { dst, src, .. } => regs[dst.0 as usize] = read(src, &regs),
                MachineInstKind::Select {
                    dst,
                    cond,
                    on_true,
                    on_false,
                    ..
                } => {
                    regs[dst.0 as usize] = read(
                        if read(cond, &regs) != 0 {
                            on_true
                        } else {
                            on_false
                        },
                        &regs,
                    );
                }
                _ => panic!("unexpected test instruction"),
            }
        }
        let MachineTerminator::ReturnScalar {
            value:
                MachineReturnValue::ScalarGp {
                    src: MachineResultSrc::Reg(reg),
                    ..
                },
        } = block.terminator
        else {
            panic!("test returns a GP register")
        };
        regs[reg.0 as usize]
    }

    #[test]
    fn reuses_dead_inputs_without_changing_full_width_results_or_aliased_rhs() {
        for width in [MachineIntWidth::I32, MachineIntWidth::I64] {
            for form in 0..3 {
                let mut candidate = block(width);
                match form {
                    1 => {
                        candidate.ops[0].kind = MachineInstKind::IntBinary {
                            width,
                            op: MachineIntBinaryOp::Sub,
                            dst: MachineReg(7),
                            lhs: MachineValue::Reg(MachineReg(4)),
                            rhs: MachineValue::Reg(MachineReg(7)),
                        }
                    }
                    2 => {
                        candidate.ops[0].kind = MachineInstKind::IntBinaryShifted {
                            width,
                            op: MachineIntBinaryOp::Xor,
                            dst: MachineReg(7),
                            lhs: MachineReg(4),
                            rhs: MachineReg(5),
                            shift: MachineShiftOp::Lsr,
                            amount: 3,
                        }
                    }
                    _ => {}
                }
                let original = candidate.clone();
                optimize(&mut candidate);
                assert_eq!(
                    pure_destination(&candidate.ops[0].kind),
                    Some(MachineReg(4))
                );
                let mut seed = 0x91df_382c_a87b_d104u64;
                for input in [0, 1, u32::MAX as u64, 1 << 32, i64::MAX as u64, u64::MAX] {
                    for _ in 0..64 {
                        let mut regs = [0; 32];
                        for reg in &mut regs {
                            seed ^= seed << 13;
                            seed ^= seed >> 7;
                            seed ^= seed << 17;
                            *reg = seed;
                        }
                        regs[4] = input;
                        assert_eq!(execute(&original, regs), execute(&candidate, regs));
                    }
                }
            }
        }
    }

    #[test]
    fn rejects_live_inputs_escaping_results_and_cached_owners() {
        for case in 0..4 {
            let mut candidate = block(MachineIntWidth::I64);
            match case {
                0 => candidate.ops.insert(
                    1,
                    MachineInst {
                        kind: MachineInstKind::Move {
                            owner: MachineRegOwner::LinearValue,
                            ty: MachineStorageType::GpWord,
                            dst: MachineReg(6),
                            src: MachineValue::Reg(MachineReg(4)),
                        },
                    },
                ),
                1 => {
                    candidate.terminator = MachineTerminator::ReturnScalar {
                        value: MachineReturnValue::ScalarGp {
                            src: MachineResultSrc::Reg(MachineReg(7)),
                            ty: MachineStorageType::GpWord,
                        },
                    }
                }
                2 => candidate.params[0].owner = MachineRegOwner::CachedCell,
                3 => {
                    candidate.params.clear();
                }
                _ => unreachable!(),
            }
            let original = candidate.clone();
            optimize(&mut candidate);
            assert_eq!(candidate, original);
        }
    }

    #[test]
    fn rejects_observers_fixed_writes_and_unbounded_proofs() {
        for barrier in [
            MachineInstKind::Store {
                ty: MachineStorageType::GpWord,
                addr: MachineAddr {
                    base: MACHINE_FP_REG,
                    offset: 0,
                },
                width: MachineMemWidth::U64,
                src: MachineValue::Reg(MachineReg(6)),
            },
            MachineInstKind::CallRuntime(MachineCallRuntime {
                metadata: MachineConstId(0),
            }),
            MachineInstKind::TrapIf {
                cond: MachineBranchCond::Value(MachineValue::Reg(MachineReg(6))),
                kind: MachineTrapKind::Unreachable,
            },
            MachineInstKind::Move {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: MACHINE_FP_REG,
                src: MachineValue::Reg(MachineReg(6)),
            },
        ] {
            let mut candidate = block(MachineIntWidth::I64);
            candidate.ops.insert(1, MachineInst { kind: barrier });
            let original = candidate.clone();
            optimize(&mut candidate);
            assert_eq!(candidate, original);
        }
        let mut candidate = block(MachineIntWidth::I64);
        for _ in 0..LOOKAHEAD {
            candidate.ops.insert(
                1,
                MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(6),
                        src: MachineValue::Imm64(17),
                    },
                },
            );
        }
        let original = candidate.clone();
        optimize(&mut candidate);
        assert_eq!(candidate, original);
    }

    #[test]
    fn stops_renaming_before_an_independent_temporary_definition() {
        let mut candidate = block(MachineIntWidth::I64);
        candidate.ops.insert(
            2,
            MachineInst {
                kind: MachineInstKind::Move {
                    owner: MachineRegOwner::LinearValue,
                    ty: MachineStorageType::GpWord,
                    dst: MachineReg(7),
                    src: MachineValue::Imm64(0),
                },
            },
        );
        let original = candidate.clone();
        optimize(&mut candidate);
        assert_eq!(
            pure_destination(&candidate.ops[0].kind),
            Some(MachineReg(4))
        );
        assert_eq!(candidate.ops[2], original.ops[2]);
        assert_eq!(candidate.ops[3], original.ops[3]);
        assert_eq!(
            execute(&candidate, [u64::MAX; 32]),
            execute(&original, [u64::MAX; 32])
        );
    }

    #[test]
    fn can_remove_a_preserved_temporary_but_never_adds_a_preserved_input_write() {
        let mut candidate = block(MachineIntWidth::I64);
        for inst in &mut candidate.ops {
            rename(&mut inst.kind, MachineReg(7), MachineReg(9), true);
        }
        let original = candidate.clone();
        optimize(&mut candidate);
        assert_eq!(
            pure_destination(&candidate.ops[0].kind),
            Some(MachineReg(4))
        );
        assert_eq!(execute(&candidate, [37; 32]), execute(&original, [37; 32]));

        let mut candidate = block(MachineIntWidth::I64);
        candidate.params[0].reg = MachineReg(9);
        for inst in &mut candidate.ops {
            rename(&mut inst.kind, MachineReg(4), MachineReg(9), true);
        }
        let original = candidate.clone();
        optimize(&mut candidate);
        assert_eq!(candidate, original);
    }
}
