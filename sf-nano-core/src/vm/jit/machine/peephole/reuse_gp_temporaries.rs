//! Reassign registers inside bounded, side-effect-free integer fragments.
//!
//! A temporary write to a cached cell can force a snapshot of its old value
//! into another register. Here both values are ordinary immutable identities
//! until the next observer. Register copies become identity aliases, and a
//! consumed input can donate its physical register to a later result. Every
//! live output and its ownership are restored before leaving the fragment.
//! Memory, calls, traps, fixed registers and lowering scratch are boundaries.

use crate::collections;
use crate::vm::jit::backend::BackendConfig;
use crate::vm::jit::machine::machine_ir::{
    MachineBlock, MachineInst, MachineInstKind, MachineIntBinaryOp, MachineReg, MachineRegOwner,
    MachineStorageType, MachineValue,
};

use super::helpers::{visit_source_values, visit_terminator_source_regs};

const MAX_FRAGMENT: usize = 64;

struct Value {
    reg: Option<MachineReg>,
    last_use: Option<usize>,
    output_hint: Option<MachineReg>,
}

struct Node {
    // Input register numbers temporarily name Value indices, not physical lanes.
    inst: MachineInst,
    result: usize,
}

#[derive(Clone, Copy)]
struct Copy {
    dst: MachineReg,
    src: MachineReg,
    owner: MachineRegOwner,
}

#[derive(Default)]
struct Scratch {
    fragments: collections::Vec<(usize, usize)>,
    live_after: collections::Vec<u64>,
    values: collections::Vec<Value>,
    current: collections::Vec<usize>,
    physical: collections::Vec<Option<usize>>,
    owners: collections::Vec<Option<MachineRegOwner>>,
    output_owners: collections::Vec<Option<MachineRegOwner>>,
    nodes: collections::Vec<Node>,
    output: collections::Vec<MachineInst>,
    copies: collections::Vec<Copy>,
}

pub(super) fn reuse_gp_temporaries(blocks: &mut [MachineBlock], config: BackendConfig) {
    // Whole-word copies are identities on native 64-bit GP backends. Pair
    // legalization and targets with other upper-word rules stay untouched.
    let end = BackendConfig::FIXED + u16::from(config.allocatable_gp_dynamic_budget());
    if config.gp_unit_bytes != 8 || !config.gp32_defs_zero_extend || end > 64 {
        return;
    }
    let mut scratch = Scratch::default();
    for block in blocks {
        if !block.ops.iter().any(|inst| matches!(inst.kind,
            MachineInstKind::Move { owner: MachineRegOwner::CachedCell, dst, src: MachineValue::Reg(src), .. }
                if dst != src))
        {
            continue;
        }
        scratch.fragments.clear();
        let mut start = 0;
        while start < block.ops.len() {
            if !eligible(&block.ops[start].kind, end) {
                start += 1;
                continue;
            }
            let mut finish = start;
            let mut moves = 0;
            let mut cached = false;
            while finish < block.ops.len() && eligible(&block.ops[finish].kind, end) {
                moves += usize::from(real_copy(&block.ops[finish].kind));
                cached |= matches!(
                    block.ops[finish].kind,
                    MachineInstKind::Move {
                        owner: MachineRegOwner::CachedCell,
                        ..
                    }
                );
                finish += 1;
            }
            if finish - start <= MAX_FRAGMENT && moves >= 2 && cached {
                scratch.fragments.push((start, finish));
            }
            start = finish;
        }
        if scratch.fragments.is_empty() {
            continue;
        }
        scratch.live_after.clear();
        scratch.live_after.resize(block.ops.len(), 0);
        let mut live = 0;
        visit_terminator_source_regs(&block.terminator, |reg| live |= bit(reg));
        for (index, inst) in block.ops.iter().enumerate().rev() {
            scratch.live_after[index] = live;
            inst.kind.for_each_defined_reg(|reg| live &= !bit(reg));
            visit_source_values(&inst.kind, |value| {
                if let MachineValue::Reg(reg) = value {
                    live |= bit(*reg);
                }
            });
        }
        // Replacing a suffix cannot invalidate original indices in an earlier
        // fragment, and each replacement preserves the original live-in set.
        while let Some((start, finish)) = scratch.fragments.pop() {
            let original = &block.ops[start..finish];
            if rewrite(original, scratch.live_after[finish - 1], end, &mut scratch) {
                block.ops.splice(start..finish, scratch.output.drain(..));
            }
        }
    }
}

fn bit(reg: MachineReg) -> u64 {
    1u64.checked_shl(u32::from(reg.0)).unwrap_or(0)
}

fn real_copy(kind: &MachineInstKind) -> bool {
    matches!(kind, MachineInstKind::Move { dst, src: MachineValue::Reg(src), .. } if dst != src)
}

fn eligible(kind: &MachineInstKind, end: u16) -> bool {
    let dst = match kind {
        MachineInstKind::Move {
            ty: MachineStorageType::GpWord | MachineStorageType::GpI64,
            dst,
            ..
        }
        | MachineInstKind::BitfieldExtractU { dst, .. }
        | MachineInstKind::IntBinaryShifted { dst, .. }
        | MachineInstKind::IntCompare { dst, .. }
        | MachineInstKind::TestBits { dst, .. } => *dst,
        MachineInstKind::IntBinary { dst, op, .. }
            if !matches!(
                op,
                MachineIntBinaryOp::DivS
                    | MachineIntBinaryOp::DivU
                    | MachineIntBinaryOp::RemS
                    | MachineIntBinaryOp::RemU
            ) =>
        {
            *dst
        }
        _ => return false,
    };
    let dynamic = |reg: MachineReg| reg.0 >= BackendConfig::FIXED && reg.0 < end;
    let mut valid = dynamic(dst);
    visit_source_values(kind, |value| match value {
        MachineValue::Reg(reg) => valid &= dynamic(*reg),
        MachineValue::ReservedReg(_) => valid = false,
        MachineValue::Imm64(_) => {}
    });
    valid
}

fn destination(kind: &mut MachineInstKind) -> &mut MachineReg {
    match kind {
        MachineInstKind::Move { dst, .. }
        | MachineInstKind::IntBinary { dst, .. }
        | MachineInstKind::BitfieldExtractU { dst, .. }
        | MachineInstKind::IntBinaryShifted { dst, .. }
        | MachineInstKind::IntCompare { dst, .. }
        | MachineInstKind::TestBits { dst, .. } => dst,
        _ => unreachable!("validated pure integer fragment"),
    }
}

fn inputs(kind: &mut MachineInstKind, mut visit: impl FnMut(&mut MachineReg)) {
    fn value(value: &mut MachineValue, visit: &mut impl FnMut(&mut MachineReg)) {
        if let MachineValue::Reg(reg) = value {
            visit(reg);
        }
    }
    match kind {
        MachineInstKind::Move { src, .. } => value(src, &mut visit),
        MachineInstKind::IntBinary { lhs, rhs, .. }
        | MachineInstKind::IntCompare { lhs, rhs, .. } => {
            value(lhs, &mut visit);
            value(rhs, &mut visit);
        }
        MachineInstKind::BitfieldExtractU { src, .. } => visit(src),
        MachineInstKind::IntBinaryShifted { lhs, rhs, .. } => {
            visit(lhs);
            visit(rhs);
        }
        MachineInstKind::TestBits { src, mask, .. } => {
            visit(src);
            value(mask, &mut visit);
        }
        _ => unreachable!("validated pure integer fragment"),
    }
}

fn rewrite(original: &[MachineInst], live: u64, end: u16, scratch: &mut Scratch) -> bool {
    let count = usize::from(end);
    scratch.values.clear();
    scratch.current.clear();
    scratch.physical.clear();
    scratch.owners.clear();
    scratch.owners.resize(count, None);
    scratch.output_owners.clear();
    scratch.output_owners.resize(count, None);
    scratch.nodes.clear();
    scratch.output.clear();
    scratch.copies.clear();
    for index in 0..count {
        scratch.values.push(Value {
            reg: Some(MachineReg(index as u16)),
            last_use: None,
            output_hint: None,
        });
        scratch.current.push(index);
        scratch.physical.push(Some(index));
    }
    let mut written = 0;
    for original_inst in original {
        let mut inst = original_inst.clone();
        let dst = *destination(&mut inst.kind);
        written |= bit(dst);
        scratch.output_owners[usize::from(dst.0)] = inst.kind.def_owner();
        if let MachineInstKind::Move {
            src: MachineValue::Reg(src),
            ..
        } = inst.kind
        {
            scratch.current[usize::from(dst.0)] = scratch.current[usize::from(src.0)];
            continue;
        }
        let position = scratch.nodes.len();
        inputs(&mut inst.kind, |reg| {
            let id = scratch.current[usize::from(reg.0)];
            scratch.values[id].last_use = Some(position);
            *reg = MachineReg(id as u16);
        });
        let result = scratch.values.len();
        scratch.values.push(Value {
            reg: None,
            last_use: None,
            output_hint: None,
        });
        scratch.current[usize::from(dst.0)] = result;
        scratch.nodes.push(Node { inst, result });
    }
    let outputs = live & written;
    for reg in BackendConfig::FIXED..end {
        if outputs & bit(MachineReg(reg)) != 0 {
            let value = &mut scratch.values[scratch.current[usize::from(reg)]];
            value.last_use = Some(scratch.nodes.len());
            value.output_hint.get_or_insert(MachineReg(reg));
        }
    }
    for (position, node) in scratch.nodes.iter().enumerate() {
        let mut inst = node.inst.clone();
        let original_dst = *destination(&mut inst.kind);
        let mut missing = false;
        let mut first_input = None;
        inputs(&mut inst.kind, |reg| {
            if let Some(physical) = scratch.values[usize::from(reg.0)].reg {
                *reg = physical;
                first_input.get_or_insert(physical);
            } else {
                missing = true;
            }
        });
        if missing {
            return false;
        }
        let available = |reg: MachineReg| {
            written & bit(reg) != 0
                && scratch.physical[usize::from(reg.0)].is_none_or(|id| {
                    scratch.values[id]
                        .last_use
                        .is_none_or(|last| last <= position)
                })
        };
        let preferred = [
            scratch.values[node.result].output_hint,
            first_input,
            Some(original_dst),
        ];
        let Some(dst) = preferred
            .into_iter()
            .flatten()
            .find(|&reg| available(reg))
            .or_else(|| {
                (BackendConfig::FIXED..end)
                    .map(MachineReg)
                    .find(|&reg| available(reg))
            })
        else {
            return false;
        };
        *destination(&mut inst.kind) = dst;
        // Prefer the donated register as the unshifted left input for ordinary
        // commutative ALU operations; all reads still precede the write.
        if let MachineInstKind::IntBinary { op, lhs, rhs, .. } = &mut inst.kind {
            if matches!(
                op,
                MachineIntBinaryOp::Add
                    | MachineIntBinaryOp::Mul
                    | MachineIntBinaryOp::And
                    | MachineIntBinaryOp::Or
                    | MachineIntBinaryOp::Xor
            ) && *rhs == MachineValue::Reg(dst)
                && *lhs != MachineValue::Reg(dst)
            {
                core::mem::swap(lhs, rhs);
            }
        }
        if let Some(old) = scratch.physical[usize::from(dst.0)] {
            scratch.values[old].reg = None;
        }
        scratch.physical[usize::from(dst.0)] = Some(node.result);
        scratch.values[node.result].reg = Some(dst);
        scratch.owners[usize::from(dst.0)] = inst.kind.def_owner();
        scratch.output.push(inst);
    }
    for reg in BackendConfig::FIXED..end {
        let dst = MachineReg(reg);
        if outputs & bit(dst) == 0 {
            continue;
        }
        let Some(src) = scratch.values[scratch.current[usize::from(reg)]].reg else {
            return false;
        };
        let owner = scratch.output_owners[usize::from(reg)].expect("written fragment output");
        if src != dst {
            scratch.copies.push(Copy { dst, src, owner });
        } else if scratch.owners[usize::from(reg)] != Some(owner) {
            // A same-register move is metadata-only at this word width.
            scratch.output.push(copy_inst(Copy { dst, src, owner }));
        }
    }
    while !scratch.copies.is_empty() {
        if let Some(index) = scratch
            .copies
            .iter()
            .position(|copy| !scratch.copies.iter().any(|other| other.src == copy.dst))
        {
            let copy = scratch.copies.remove(index);
            scratch.output.push(copy_inst(copy));
            continue;
        }
        let occupied = scratch
            .copies
            .iter()
            .fold(outputs, |mask, copy| mask | bit(copy.src));
        let Some(temp) = (BackendConfig::FIXED..end)
            .map(MachineReg)
            .find(|reg| written & bit(*reg) != 0 && occupied & bit(*reg) == 0)
        else {
            return false;
        };
        let source = scratch.copies[0].src;
        scratch.output.push(copy_inst(Copy {
            dst: temp,
            src: source,
            owner: MachineRegOwner::LinearValue,
        }));
        for copy in &mut scratch.copies {
            if copy.src == source {
                copy.src = temp;
            }
        }
    }
    // Pay for all boundary copies. Equal instruction counts do not justify
    // changing register placement; native measurements decide actual benefit.
    let emitted = |ops: &[MachineInst]| {
        ops.iter()
            .filter(|inst| {
                !matches!(inst.kind,
        MachineInstKind::Move { dst, src: MachineValue::Reg(src), .. } if dst == src)
            })
            .count()
    };
    emitted(&scratch.output) < emitted(original)
}

fn copy_inst(copy: Copy) -> MachineInst {
    MachineInst {
        kind: MachineInstKind::Move {
            owner: copy.owner,
            ty: MachineStorageType::GpWord,
            dst: copy.dst,
            src: MachineValue::Reg(copy.src),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::jit::machine::machine_ir::{
        MachineAddr, MachineBlockId, MachineCompareKind, MachineEdge, MachineIntWidth,
        MachineMemWidth, MachineShiftOp, MachineSign, MachineTerminator, MACHINE_FP_REG,
    };

    fn mv(dst: u16, src: u16, owner: MachineRegOwner) -> MachineInst {
        copy_inst(Copy {
            dst: MachineReg(dst),
            src: MachineReg(src),
            owner,
        })
    }

    fn binary(
        width: MachineIntWidth,
        op: MachineIntBinaryOp,
        dst: u16,
        lhs: u16,
        rhs: MachineValue,
    ) -> MachineInst {
        MachineInst {
            kind: MachineInstKind::IntBinary {
                width,
                op,
                dst: MachineReg(dst),
                lhs: MachineValue::Reg(MachineReg(lhs)),
                rhs,
            },
        }
    }

    fn snapshot(width: MachineIntWidth) -> collections::Vec<MachineInst> {
        collections::vec![
            binary(
                width,
                MachineIntBinaryOp::Mul,
                7,
                4,
                MachineValue::Reg(MachineReg(5))
            ),
            mv(8, 6, MachineRegOwner::LinearValue),
            mv(6, 7, MachineRegOwner::CachedCell),
            MachineInst {
                kind: MachineInstKind::BitfieldExtractU {
                    width,
                    dst: MachineReg(7),
                    src: MachineReg(6),
                    lsb: 3,
                    bits: 7
                }
            },
            MachineInst {
                kind: MachineInstKind::BitfieldExtractU {
                    width,
                    dst: MachineReg(9),
                    src: MachineReg(6),
                    lsb: 11,
                    bits: 9
                }
            },
            binary(
                width,
                MachineIntBinaryOp::Xor,
                7,
                7,
                MachineValue::Reg(MachineReg(9))
            ),
            binary(
                width,
                MachineIntBinaryOp::Add,
                6,
                8,
                MachineValue::Reg(MachineReg(7))
            ),
        ]
    }

    fn operand(regs: &[u64; 16], value: MachineValue) -> u64 {
        match value {
            MachineValue::Reg(reg) => regs[usize::from(reg.0)],
            MachineValue::Imm64(value) => value,
            MachineValue::ReservedReg(_) => panic!("reserved lane is not a value"),
        }
    }

    fn calculate(width: MachineIntWidth, op: MachineIntBinaryOp, lhs: u64, rhs: u64) -> u64 {
        let narrow = width == MachineIntWidth::I32;
        let mask = if narrow { u32::MAX as u64 } else { u64::MAX };
        let lhs = lhs & mask;
        let rhs = rhs & mask;
        let shift = (rhs & if narrow { 31 } else { 63 }) as u32;
        let result = match op {
            MachineIntBinaryOp::Add => lhs.wrapping_add(rhs),
            MachineIntBinaryOp::Sub => lhs.wrapping_sub(rhs),
            MachineIntBinaryOp::Mul => lhs.wrapping_mul(rhs),
            MachineIntBinaryOp::And => lhs & rhs,
            MachineIntBinaryOp::Or => lhs | rhs,
            MachineIntBinaryOp::Xor => lhs ^ rhs,
            MachineIntBinaryOp::Shl => lhs << shift,
            MachineIntBinaryOp::ShrU => lhs >> shift,
            MachineIntBinaryOp::ShrS if narrow => ((lhs as u32 as i32) >> shift) as u64,
            MachineIntBinaryOp::ShrS => ((lhs as i64) >> shift) as u64,
            MachineIntBinaryOp::Rotl if narrow => (lhs as u32).rotate_left(shift) as u64,
            MachineIntBinaryOp::Rotl => lhs.rotate_left(shift),
            MachineIntBinaryOp::Rotr if narrow => (lhs as u32).rotate_right(shift) as u64,
            MachineIntBinaryOp::Rotr => lhs.rotate_right(shift),
            _ => panic!("division is a fragment boundary"),
        };
        result & mask
    }

    fn compare(
        width: MachineIntWidth,
        kind: MachineCompareKind,
        sign: MachineSign,
        lhs: u64,
        rhs: u64,
    ) -> u64 {
        let ordering = match (width, sign) {
            (MachineIntWidth::I32, MachineSign::Signed) => {
                (lhs as u32 as i32).cmp(&(rhs as u32 as i32))
            }
            (MachineIntWidth::I32, MachineSign::Unsigned) => (lhs as u32).cmp(&(rhs as u32)),
            (MachineIntWidth::I64, MachineSign::Signed) => (lhs as i64).cmp(&(rhs as i64)),
            (MachineIntWidth::I64, MachineSign::Unsigned) => lhs.cmp(&rhs),
        };
        u64::from(match kind {
            MachineCompareKind::Eq => ordering.is_eq(),
            MachineCompareKind::Ne => !ordering.is_eq(),
            MachineCompareKind::Lt => ordering.is_lt(),
            MachineCompareKind::Le => !ordering.is_gt(),
            MachineCompareKind::Gt => ordering.is_gt(),
            MachineCompareKind::Ge => !ordering.is_lt(),
        })
    }

    fn execute(
        ops: &[MachineInst],
        mut regs: [u64; 16],
    ) -> ([u64; 16], collections::Vec<(i32, u64)>) {
        let mut stores = collections::Vec::new();
        for inst in ops {
            let (dst, value) = match inst.kind {
                MachineInstKind::Move { dst, src, .. } => (dst, operand(&regs, src)),
                MachineInstKind::IntBinary {
                    width,
                    op,
                    dst,
                    lhs,
                    rhs,
                } => (
                    dst,
                    calculate(width, op, operand(&regs, lhs), operand(&regs, rhs)),
                ),
                MachineInstKind::IntBinaryShifted {
                    width,
                    op,
                    dst,
                    lhs,
                    rhs,
                    shift,
                    amount,
                } => {
                    let shift = match shift {
                        MachineShiftOp::Lsl => MachineIntBinaryOp::Shl,
                        MachineShiftOp::Lsr => MachineIntBinaryOp::ShrU,
                        MachineShiftOp::Asr => MachineIntBinaryOp::ShrS,
                        MachineShiftOp::Ror => MachineIntBinaryOp::Rotr,
                    };
                    let shifted =
                        calculate(width, shift, regs[usize::from(rhs.0)], u64::from(amount));
                    (dst, calculate(width, op, regs[usize::from(lhs.0)], shifted))
                }
                MachineInstKind::BitfieldExtractU {
                    width,
                    dst,
                    src,
                    lsb,
                    bits,
                } => {
                    let shifted = calculate(
                        width,
                        MachineIntBinaryOp::ShrU,
                        regs[usize::from(src.0)],
                        u64::from(lsb),
                    );
                    (dst, shifted & (u64::MAX >> (64 - bits)))
                }
                MachineInstKind::IntCompare {
                    width,
                    kind,
                    sign,
                    dst,
                    lhs,
                    rhs,
                } => (
                    dst,
                    compare(width, kind, sign, operand(&regs, lhs), operand(&regs, rhs)),
                ),
                MachineInstKind::TestBits {
                    width,
                    kind,
                    dst,
                    src,
                    mask,
                } => {
                    let masked = calculate(
                        width,
                        MachineIntBinaryOp::And,
                        regs[usize::from(src.0)],
                        operand(&regs, mask),
                    );
                    (dst, compare(width, kind, MachineSign::Unsigned, masked, 0))
                }
                MachineInstKind::Store {
                    addr,
                    width: MachineMemWidth::U64,
                    src,
                    ..
                } => {
                    assert_eq!(addr.base, MACHINE_FP_REG);
                    stores.push((addr.offset, operand(&regs, src)));
                    continue;
                }
                _ => panic!("unsupported fixture instruction: {:?}", inst.kind),
            };
            regs[usize::from(dst.0)] = value;
        }
        (regs, stores)
    }

    fn verify(original: &[MachineInst], replacement: &[MachineInst], live: u64, regs: [u64; 16]) {
        let (before, old_stores) = execute(original, regs);
        let (after, new_stores) = execute(replacement, regs);
        assert_eq!(old_stores, new_stores);
        let mut written = 0;
        let mut before_owners = [None; 16];
        let mut after_owners = [None; 16];
        for inst in original {
            inst.kind.for_each_defined_reg(|reg| {
                written |= bit(reg);
                before_owners[usize::from(reg.0)] = inst.kind.def_owner();
            });
        }
        for inst in replacement {
            inst.kind.for_each_defined_reg(|reg| {
                assert_ne!(written & bit(reg), 0, "new physical clobber");
                after_owners[usize::from(reg.0)] = inst.kind.def_owner();
            });
        }
        for reg in 0..16 {
            if live & (1 << reg) != 0 || written & (1 << reg) == 0 {
                assert_eq!(before[reg], after[reg], "live register r{reg}");
            }
            if live & written & (1 << reg) != 0 {
                assert_eq!(before_owners[reg], after_owners[reg], "output owner r{reg}");
            }
        }
    }

    #[test]
    fn preserves_old_cached_snapshot_and_full_width_results() {
        for width in [MachineIntWidth::I32, MachineIntWidth::I64] {
            let original = snapshot(width);
            let live = bit(MachineReg(6));
            let mut scratch = Scratch::default();
            assert!(rewrite(&original, live, 11, &mut scratch));
            assert!(!scratch.output.iter().any(|inst| real_copy(&inst.kind)));
            for seed in [0, 1, u64::MAX, 0x1234_5678_ffff_ffff, 0x8000_0000_8000_0000] {
                let regs = core::array::from_fn(|index| {
                    seed.wrapping_mul(index as u64 + 1)
                        .rotate_left(index as u32)
                });
                verify(&original, &scratch.output, live, regs);
            }
        }
    }

    #[test]
    fn restores_parallel_output_cycles_using_only_an_originally_written_lane() {
        let original = collections::vec![
            mv(7, 4, MachineRegOwner::LinearValue),
            mv(4, 7, MachineRegOwner::CachedCell),
            mv(6, 4, MachineRegOwner::CachedCell),
            mv(4, 5, MachineRegOwner::LinearValue),
            mv(5, 6, MachineRegOwner::CachedCell),
        ];
        let live = bit(MachineReg(4)) | bit(MachineReg(5)) | bit(MachineReg(6));
        let mut scratch = Scratch::default();
        assert!(rewrite(&original, live, 11, &mut scratch));
        verify(
            &original,
            &scratch.output,
            live,
            core::array::from_fn(|index| 0x1234_ffff_ffff_0000 + index as u64),
        );
    }

    #[test]
    fn preserves_values_seen_by_memory_and_cfg_edges() {
        let mut ops = snapshot(MachineIntWidth::I32);
        ops.push(MachineInst {
            kind: MachineInstKind::Store {
                ty: MachineStorageType::GpWord,
                addr: MachineAddr {
                    base: MACHINE_FP_REG,
                    offset: 8,
                },
                width: MachineMemWidth::U64,
                src: MachineValue::Reg(MachineReg(6)),
            },
        });
        ops.extend(snapshot(MachineIntWidth::I64));
        let mut block = MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops,
            terminator: MachineTerminator::Jump(MachineEdge {
                target: MachineBlockId(1),
                args: collections::vec![MachineValue::Reg(MachineReg(6))],
            }),
        };
        let before = block.clone();
        reuse_gp_temporaries(
            core::slice::from_mut(&mut block),
            BackendConfig::new(8, 8, 0, 0).with_gp32_zero_extending_defs(),
        );
        assert_eq!(block.terminator, before.terminator);
        assert!(block.ops.len() < before.ops.len());
        verify(
            &before.ops,
            &block.ops,
            bit(MachineReg(6)),
            core::array::from_fn(|index| u64::MAX.wrapping_sub(index as u64 * 13)),
        );
    }

    #[test]
    fn excludes_trapping_arithmetic_fixed_registers_and_lowering_reserve() {
        let reg = MachineValue::Reg(MachineReg(5));
        for op in [
            MachineIntBinaryOp::DivS,
            MachineIntBinaryOp::DivU,
            MachineIntBinaryOp::RemS,
            MachineIntBinaryOp::RemU,
        ] {
            assert!(!eligible(
                &binary(MachineIntWidth::I64, op, 6, 4, reg).kind,
                11
            ));
        }
        for (dst, src) in [(1, 4), (4, 1), (11, 4), (4, 11)] {
            assert!(!eligible(
                &mv(dst, src, MachineRegOwner::CachedCell).kind,
                11
            ));
        }
        let mut block = MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: snapshot(MachineIntWidth::I32),
            terminator: MachineTerminator::Return,
        };
        let before = block.clone();
        for config in [
            BackendConfig::new(4, 8, 0, 0).with_gp32_zero_extending_defs(),
            BackendConfig::new(8, 8, 0, 0),
        ] {
            reuse_gp_temporaries(core::slice::from_mut(&mut block), config);
            assert_eq!(block, before);
        }
    }

    #[test]
    fn generated_alias_lifetimes_match_independent_execution() {
        let mut state = 0x96ea_17c5_0da1_7723u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let operators = [
            MachineIntBinaryOp::Add,
            MachineIntBinaryOp::Sub,
            MachineIntBinaryOp::Mul,
            MachineIntBinaryOp::And,
            MachineIntBinaryOp::Or,
            MachineIntBinaryOp::Xor,
            MachineIntBinaryOp::Shl,
            MachineIntBinaryOp::ShrU,
            MachineIntBinaryOp::ShrS,
            MachineIntBinaryOp::Rotl,
            MachineIntBinaryOp::Rotr,
        ];
        let mut accepted = 0;
        let mut scratch = Scratch::default();
        for _ in 0..512 {
            let mut original = snapshot(MachineIntWidth::I32);
            for _ in 0..24 {
                let width = if next() & 1 == 0 {
                    MachineIntWidth::I32
                } else {
                    MachineIntWidth::I64
                };
                let dst = 4 + (next() % 7) as u16;
                let lhs = 4 + (next() % 7) as u16;
                let rhs = if next() & 1 == 0 {
                    MachineValue::Imm64(next())
                } else {
                    MachineValue::Reg(MachineReg(4 + (next() % 7) as u16))
                };
                let kind = match next() % 8 {
                    0 => mv(dst, lhs, MachineRegOwner::CachedCell).kind,
                    1 => MachineInstKind::IntCompare {
                        width,
                        dst: MachineReg(dst),
                        lhs: MachineValue::Reg(MachineReg(lhs)),
                        rhs,
                        kind: [
                            MachineCompareKind::Eq,
                            MachineCompareKind::Ne,
                            MachineCompareKind::Lt,
                            MachineCompareKind::Le,
                            MachineCompareKind::Gt,
                            MachineCompareKind::Ge,
                        ][next() as usize % 6],
                        sign: if next() & 1 == 0 {
                            MachineSign::Signed
                        } else {
                            MachineSign::Unsigned
                        },
                    },
                    2 => MachineInstKind::TestBits {
                        width,
                        dst: MachineReg(dst),
                        src: MachineReg(lhs),
                        mask: rhs,
                        kind: if next() & 1 == 0 {
                            MachineCompareKind::Eq
                        } else {
                            MachineCompareKind::Ne
                        },
                    },
                    3 => MachineInstKind::IntBinaryShifted {
                        width,
                        dst: MachineReg(dst),
                        lhs: MachineReg(lhs),
                        rhs: MachineReg(4 + (next() % 7) as u16),
                        op: [
                            MachineIntBinaryOp::Add,
                            MachineIntBinaryOp::Sub,
                            MachineIntBinaryOp::And,
                            MachineIntBinaryOp::Or,
                            MachineIntBinaryOp::Xor,
                        ][next() as usize % 5],
                        shift: [
                            MachineShiftOp::Lsl,
                            MachineShiftOp::Lsr,
                            MachineShiftOp::Asr,
                            MachineShiftOp::Ror,
                        ][next() as usize % 4],
                        amount: (next() % 32) as u8,
                    },
                    _ => {
                        binary(
                            width,
                            operators[next() as usize % operators.len()],
                            dst,
                            lhs,
                            rhs,
                        )
                        .kind
                    }
                };
                original.push(MachineInst { kind });
            }
            let live = next() & 0x7f0;
            if rewrite(&original, live, 11, &mut scratch) {
                accepted += 1;
                for _ in 0..8 {
                    verify(
                        &original,
                        &scratch.output,
                        live,
                        core::array::from_fn(|_| next()),
                    );
                }
            }
        }
        assert!(
            accepted > 100,
            "coverage must include successful rewrites: {accepted}"
        );
    }
}
