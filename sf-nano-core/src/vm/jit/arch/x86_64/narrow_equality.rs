//! Compare a zero-extended byte/halfword with a truncated GP value directly.
//!
//! Only a closed block suffix is skipped. A dead load can become CMP's memory
//! operand across the pure mask/XOR; every skipped result is dead on both paths.
use super::fusion::LoadAluMem;
use crate::vm::jit::machine::machine_ir::{
    MachineBlock, MachineBranchCond, MachineCompareKind, MachineEdge, MachineInstKind,
    MachineIntBinaryOp, MachineIntWidth, MachineLoadExtension, MachineMemWidth, MachineReg,
    MachineTerminator, MachineValue, MACHINE_FIXED_REG_COUNT,
};

#[derive(Clone, Copy)]
pub(super) struct NarrowEquality {
    pub first_skipped: usize,
    pub width: MachineMemWidth,
    pub loaded: MachineReg,
    pub source: MachineReg,
    pub memory: Option<LoadAluMem>,
}

fn dynamic_gp(reg: MachineReg) -> bool {
    reg.0 >= MACHINE_FIXED_REG_COUNT && usize::from(reg.0) < super::abi::max_gp_mapped_regs()
}

fn unordered_pair(lhs: MachineValue, rhs: MachineValue, a: MachineReg, b: MachineReg) -> bool {
    (lhs == MachineValue::Reg(a) && rhs == MachineValue::Reg(b))
        || (lhs == MachineValue::Reg(b) && rhs == MachineValue::Reg(a))
}

fn dead_on_edge(blocks: &[MachineBlock], edge: &MachineEdge, reg: MachineReg) -> bool {
    if edge
        .args
        .iter()
        .any(|arg| matches!(arg, MachineValue::Reg(r) | MachineValue::ReservedReg(r) if *r == reg))
    {
        return false;
    }
    blocks.get(edge.target.as_usize()).is_some_and(|target| {
        target.params.iter().any(|param| param.reg == reg)
            || !crate::vm::jit::machine::peephole::helpers::reg_live_after(
                &target.ops,
                &target.terminator,
                reg,
            )
    })
}

pub(super) fn narrow_equality(
    block: &MachineBlock,
    blocks: &[MachineBlock],
) -> Option<NarrowEquality> {
    let last = block.ops.last()?;
    if !matches!(
        last.kind,
        MachineInstKind::IntBinary {
            width: MachineIntWidth::I32,
            op: MachineIntBinaryOp::And | MachineIntBinaryOp::Xor,
            ..
        }
    ) {
        return None;
    }
    let MachineTerminator::Branch {
        cond:
            MachineBranchCond::IntCompare {
                width: MachineIntWidth::I32,
                kind: MachineCompareKind::Eq | MachineCompareKind::Ne,
                lhs,
                rhs,
                ..
            },
        then_edge,
        else_edge,
    } = &block.terminator
    else {
        return None;
    };
    let dead = |reg| dead_on_edge(blocks, then_edge, reg) && dead_on_edge(blocks, else_edge, reg);
    let (mask_index, compare_lhs, compare_rhs) = match last.kind {
        MachineInstKind::IntBinary {
            width: MachineIntWidth::I32,
            op: MachineIntBinaryOp::Xor,
            dst,
            lhs: a,
            rhs: b,
        } => {
            if !dynamic_gp(dst)
                || !((*lhs == MachineValue::Reg(dst) && *rhs == MachineValue::Imm64(0))
                    || (*rhs == MachineValue::Reg(dst) && *lhs == MachineValue::Imm64(0)))
                || !dead(dst)
            {
                return None;
            }
            (block.ops.len().checked_sub(2)?, a, b)
        }
        _ => (block.ops.len() - 1, *lhs, *rhs),
    };
    let MachineInstKind::IntBinary {
        width: MachineIntWidth::I32,
        op: MachineIntBinaryOp::And,
        dst: masked,
        lhs,
        rhs,
    } = block.ops[mask_index].kind
    else {
        return None;
    };
    let (source, mask) = match (lhs, rhs) {
        (MachineValue::Reg(source), MachineValue::Imm64(mask))
        | (MachineValue::Imm64(mask), MachineValue::Reg(source)) => (source, mask),
        _ => return None,
    };
    let load_index = mask_index.checked_sub(1)?;
    let (loaded, width, memory) = match block.ops[load_index].kind {
        MachineInstKind::Load {
            dst,
            addr,
            width: width @ (MachineMemWidth::U8 | MachineMemWidth::U16),
            extension: MachineLoadExtension::None | MachineLoadExtension::ZeroExtend,
            ..
        } => (dst, width, LoadAluMem::Base(addr)),
        MachineInstKind::IndexedLoad {
            dst,
            base,
            index,
            index_extend,
            offset,
            width: width @ (MachineMemWidth::U8 | MachineMemWidth::U16),
            extension: MachineLoadExtension::None | MachineLoadExtension::ZeroExtend,
            ..
        } => (
            dst,
            width,
            LoadAluMem::Indexed {
                base,
                index,
                extend: index_extend,
                offset,
            },
        ),
        _ => return None,
    };
    let expected_mask = (1u64 << (width.bytes() * 8)) - 1;
    if mask != expected_mask
        || loaded == masked
        || !dynamic_gp(loaded)
        || !dynamic_gp(masked)
        || !dynamic_gp(source)
        || !unordered_pair(compare_lhs, compare_rhs, loaded, masked)
        || !dead(masked)
    {
        return None;
    }
    // If the load overwrote the compared source, CMP must still read that
    // loaded value. Likewise retain the load when a successor observes it.
    let memory = (source != loaded && dead(loaded)).then_some(memory);
    Some(NarrowEquality {
        first_skipped: if memory.is_some() {
            load_index
        } else {
            mask_index
        },
        width,
        loaded,
        source,
        memory,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collections;
    use crate::vm::jit::machine::machine_ir::{
        MachineAddr, MachineBlockId, MachineInst, MachineRegOwner, MachineSign, MachineStorageType,
    };

    fn blocks(width: MachineMemWidth, xor: bool) -> collections::Vec<MachineBlock> {
        let mut ops = collections::vec![
            MachineInst {
                kind: MachineInstKind::Load {
                    owner: MachineRegOwner::LinearValue,
                    ty: MachineStorageType::GpWord,
                    dst: MachineReg(5),
                    addr: MachineAddr {
                        base: MachineReg(2),
                        offset: 16
                    },
                    width,
                    extension: MachineLoadExtension::ZeroExtend,
                }
            },
            MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: MachineIntWidth::I32,
                    op: MachineIntBinaryOp::And,
                    dst: MachineReg(7),
                    lhs: MachineValue::Reg(MachineReg(6)),
                    rhs: MachineValue::Imm64((1 << (width.bytes() * 8)) - 1),
                }
            },
        ];
        if xor {
            ops.push(MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: MachineIntWidth::I32,
                    op: MachineIntBinaryOp::Xor,
                    dst: MachineReg(5),
                    lhs: MachineValue::Reg(MachineReg(5)),
                    rhs: MachineValue::Reg(MachineReg(7)),
                },
            });
        }
        let edge = |id| MachineEdge {
            target: MachineBlockId(id),
            args: collections::vec![],
        };
        collections::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: collections::vec![],
                ops,
                terminator: MachineTerminator::Branch {
                    cond: MachineBranchCond::IntCompare {
                        width: MachineIntWidth::I32,
                        kind: MachineCompareKind::Eq,
                        sign: MachineSign::Signed,
                        lhs: MachineValue::Reg(MachineReg(5)),
                        rhs: if xor {
                            MachineValue::Imm64(0)
                        } else {
                            MachineValue::Reg(MachineReg(7))
                        },
                    },
                    then_edge: edge(1),
                    else_edge: edge(2),
                },
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: collections::vec![],
                ops: collections::vec![],
                terminator: MachineTerminator::Return
            },
            MachineBlock {
                id: MachineBlockId(2),
                params: collections::vec![],
                ops: collections::vec![],
                terminator: MachineTerminator::Return
            },
        ]
    }

    #[test]
    fn narrow_equality_preserves_all_byte_values_and_halfword_boundaries() {
        for width in [MachineMemWidth::U8, MachineMemWidth::U16] {
            for xor in [false, true] {
                for swapped in [false, true] {
                    let mut b = blocks(width, xor);
                    if swapped {
                        if let MachineInstKind::IntBinary { lhs, rhs, .. } = &mut b[0].ops[1].kind {
                            core::mem::swap(lhs, rhs);
                        }
                        if let MachineTerminator::Branch {
                            cond: MachineBranchCond::IntCompare { lhs, rhs, .. },
                            ..
                        } = &mut b[0].terminator
                        {
                            core::mem::swap(lhs, rhs);
                        }
                    }
                    let plan = narrow_equality(&b[0], &b).unwrap();
                    assert_eq!(plan.first_skipped, 0);
                    assert!(plan.memory.is_some());
                    assert_eq!(
                        (plan.loaded, plan.source, plan.width),
                        (MachineReg(5), MachineReg(6), width)
                    );
                    let mask = (1u32 << (width.bytes() * 8)) - 1;
                    for loaded in 0..=mask {
                        for low in [0, loaded, loaded ^ 1, mask] {
                            for high in [0, 0x1234_0000, 0xffff_ff00] {
                                let source = (high & !mask) | low;
                                let masked = source & mask;
                                let original = if xor {
                                    loaded ^ masked == 0
                                } else {
                                    loaded == masked
                                };
                                let native = match width {
                                    MachineMemWidth::U8 => loaded as u8 == source as u8,
                                    MachineMemWidth::U16 => loaded as u16 == source as u16,
                                    _ => unreachable!(),
                                };
                                assert_eq!(native, original);
                            }
                        }
                    }
                }
            }
        }
        // Exhaust every possible pair of low bytes independently of the
        // boundary-oriented halfword sweep above.
        for lhs in 0..=255u32 {
            for rhs in 0..=255u32 {
                let rhs = rhs | 0xaabb_cc00;
                assert_eq!(lhs == (rhs & 255), lhs as u8 == rhs as u8);
            }
        }
    }

    #[test]
    fn narrow_equality_rejects_observed_results_and_unproven_ranges() {
        for xor in [false, true] {
            for bad in 0..8 {
                let mut b = blocks(MachineMemWidth::U8, xor);
                match bad {
                    0 => {
                        if let MachineInstKind::Load { extension, .. } = &mut b[0].ops[0].kind {
                            *extension = MachineLoadExtension::SignExtend;
                        }
                    }
                    1 => {
                        if let MachineInstKind::IntBinary { rhs, .. } = &mut b[0].ops[1].kind {
                            *rhs = MachineValue::Imm64(127);
                        }
                    }
                    2 => {
                        if let MachineInstKind::IntBinary { dst, .. } = &mut b[0].ops[1].kind {
                            *dst = MachineReg(5);
                        }
                    }
                    3 => {
                        if let MachineInstKind::IntBinary { lhs, .. } = &mut b[0].ops[1].kind {
                            *lhs = MachineValue::Reg(MachineReg(2));
                        }
                    }
                    4 => {
                        if let MachineTerminator::Branch { then_edge, .. } = &mut b[0].terminator {
                            then_edge.args.push(MachineValue::Reg(MachineReg(7)));
                        }
                    }
                    5 => b[1].ops.push(MachineInst {
                        kind: MachineInstKind::IntBinary {
                            width: MachineIntWidth::I32,
                            op: MachineIntBinaryOp::Add,
                            dst: MachineReg(7),
                            lhs: MachineValue::Reg(MachineReg(7)),
                            rhs: MachineValue::Imm64(1),
                        },
                    }),
                    6 => {
                        if let MachineTerminator::Branch {
                            cond: MachineBranchCond::IntCompare { kind, .. },
                            ..
                        } = &mut b[0].terminator
                        {
                            *kind = MachineCompareKind::Lt;
                        }
                    }
                    7 => {
                        if let MachineInstKind::Load { width, .. } = &mut b[0].ops[0].kind {
                            *width = MachineMemWidth::U32;
                        }
                    }
                    _ => unreachable!(),
                }
                assert!(narrow_equality(&b[0], &b).is_none(), "xor={xor}, bad={bad}");
            }
        }
        let mut b = blocks(MachineMemWidth::U8, true);
        if let MachineTerminator::Branch { else_edge, .. } = &mut b[0].terminator {
            else_edge.args.push(MachineValue::Reg(MachineReg(5)));
        }
        assert!(narrow_equality(&b[0], &b).is_none());
    }

    #[test]
    fn narrow_compare_encodings_select_low_bytes_and_extended_registers() {
        use super::super::{enc, reg::X86Reg};
        use crate::vm::jit::arch::common::text_emitter::TextEmitter;
        let mut text = TextEmitter::new();
        enc::cmp_rr_8(&mut text, X86Reg::RSI, X86Reg::RDI);
        enc::cmp_rr_8(&mut text, X86Reg::R8, X86Reg::R9);
        enc::cmp_rr_16(&mut text, X86Reg::R8, X86Reg::RDI);
        enc::cmp_rr_16(&mut text, X86Reg::RSI, X86Reg::R9);
        assert_eq!(
            text.finish(),
            [0x40, 0x3a, 0xf7, 0x45, 0x3a, 0xc1, 0x66, 0x44, 0x3b, 0xc7, 0x66, 0x41, 0x3b, 0xf1]
        );
    }

    #[test]
    fn memory_equality_keeps_a_load_that_is_observed_or_overwrites_the_source() {
        let mut b = blocks(MachineMemWidth::U8, false);
        if let MachineTerminator::Branch { then_edge, .. } = &mut b[0].terminator {
            then_edge.args.push(MachineValue::Reg(MachineReg(5)));
        }
        b[1].params
            .push(crate::vm::jit::machine::machine_ir::MachineBlockParam::gp_i64(MachineReg(8)));
        let plan = narrow_equality(&b[0], &b).unwrap();
        assert_eq!(plan.first_skipped, 1);
        assert!(plan.memory.is_none());

        let mut b = blocks(MachineMemWidth::U16, false);
        if let MachineInstKind::IntBinary { lhs, .. } = &mut b[0].ops[1].kind {
            *lhs = MachineValue::Reg(MachineReg(5));
        }
        let plan = narrow_equality(&b[0], &b).unwrap();
        assert_eq!(plan.first_skipped, 1);
        assert!(plan.memory.is_none());
    }

    #[test]
    fn narrow_memory_encodings_cover_rex_sib_and_zero_displacement() {
        use super::super::{enc, reg::X86Reg};
        use crate::vm::jit::arch::common::text_emitter::TextEmitter;
        let mut text = TextEmitter::new();
        enc::cmp_rm_narrow(&mut text, false, X86Reg::RSI, X86Reg::RBP, None, 0);
        enc::cmp_rm_narrow(
            &mut text,
            false,
            X86Reg::R8,
            X86Reg::R12,
            Some(X86Reg::RDI),
            0,
        );
        enc::cmp_rm_narrow(
            &mut text,
            true,
            X86Reg::RDI,
            X86Reg::R12,
            Some(X86Reg::R8),
            2,
        );
        assert_eq!(
            text.finish(),
            [0x40, 0x3a, 0x75, 0x00, 0x45, 0x3a, 0x04, 0x3c, 0x66, 0x43, 0x3b, 0x7c, 0x04, 0x02]
        );
    }

    #[test]
    fn executes_narrow_memory_equality_with_high_bits_and_both_address_forms() {
        use super::super::{
            abi::{C_ARG0, C_ARG1, C_ARG2},
            enc,
            reg::X86Reg,
        };
        use crate::vm::jit::{
            arch::common::text_emitter::TextEmitter, runtime::code_buf::CodeBuffer,
        };
        let memory: [u8; 64] = core::array::from_fn(|n| ((n * 73) ^ (n >> 2)) as u8);
        for word in [false, true] {
            for indexed in [false, true] {
                for offset in [-15, 0, 17] {
                    let mut text = TextEmitter::new();
                    enc::cmp_rm_narrow(
                        &mut text,
                        word,
                        C_ARG1,
                        C_ARG0,
                        indexed.then_some(C_ARG2),
                        offset,
                    );
                    enc::setcc(&mut text, enc::Cc::E, X86Reg::RAX);
                    enc::movzx_r32_r8(&mut text, X86Reg::RAX, X86Reg::RAX);
                    enc::ret(&mut text);
                    let bytes = text.finish();
                    let mut code = CodeBuffer::with_capacity(4096).unwrap();
                    code.begin_write();
                    code.emit_bytes(&bytes);
                    code.finish_write(0, bytes.len());
                    // This leaf uses only C argument registers and RAX. Every
                    // tested address (including the full U16 read) is in memory.
                    let compare: unsafe extern "C" fn(*const u8, u64, usize) -> u32 =
                        unsafe { code.fn_ptr(0) };
                    for index in [0, 1, 5] {
                        let address = (16 + offset) as usize + if indexed { index } else { 0 };
                        let loaded = if word {
                            u64::from(u16::from_le_bytes([memory[address], memory[address + 1]]))
                        } else {
                            u64::from(memory[address])
                        };
                        let mask = if word { 65535 } else { 255 };
                        for source in [
                            loaded,
                            loaded | 0xabcd_0000_0000_0000,
                            loaded ^ 1,
                            0,
                            u64::MAX,
                        ] {
                            let actual = unsafe { compare(memory.as_ptr().add(16), source, index) };
                            assert_eq!(actual, u32::from(loaded == source & mask));
                        }
                    }
                }
            }
        }
    }
}
