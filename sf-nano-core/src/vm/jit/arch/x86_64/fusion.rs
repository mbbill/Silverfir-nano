//! x86_64 condition-code mapping and instruction-pair fusion probes.

use super::enc::{Cc, MemAluOp};
use crate::vm::jit::machine::machine_ir::{
    MachineAddr, MachineCompareKind, MachineIndexExtend, MachineInst, MachineInstKind,
    MachineIntBinaryOp, MachineIntWidth, MachineLoadExtension, MachineMemWidth, MachineReg,
    MachineRegOwner, MachineSign, MachineStorageType, MachineTerminator, MachineValue,
    MACHINE_FIXED_REG_COUNT,
};

/// Read through an immediately preceding GP snapshot without materializing
/// the copy. This is an emitter-only substitution: cached-value ownership in
/// MachineIR stays intact. Narrow loads cannot participate in load+ALU fusion.
pub(super) fn copied_narrow_load(
    copy: &MachineInst,
    load: &MachineInst,
    after: &[MachineInst],
    term: &MachineTerminator,
) -> Option<MachineInst> {
    let MachineInstKind::Move {
        owner: MachineRegOwner::LinearValue,
        ty: MachineStorageType::GpWord | MachineStorageType::GpI64,
        dst: copied,
        src: MachineValue::Reg(source),
    } = copy.kind
    else {
        return None;
    };
    let dynamic_gp = |reg: MachineReg| {
        reg.0 >= MACHINE_FIXED_REG_COUNT && usize::from(reg.0) < super::abi::max_gp_mapped_regs()
    };
    if copied == source || !dynamic_gp(copied) || !dynamic_gp(source) {
        return None;
    }
    let MachineInstKind::IndexedLoad {
        dst,
        base,
        index,
        index_extend,
        offset,
        width: width @ (MachineMemWidth::U8 | MachineMemWidth::U16),
        extension,
    } = load.kind
    else {
        return None;
    };
    if !dynamic_gp(dst)
        || (base != copied && index != copied)
        || (dst != copied
            && crate::vm::jit::machine::peephole::helpers::reg_live_after(after, term, copied))
    {
        return None;
    }
    Some(MachineInst {
        kind: MachineInstKind::IndexedLoad {
            dst,
            base: if base == copied { source } else { base },
            index: if index == copied { source } else { index },
            index_extend,
            offset,
            width,
            extension,
        },
    })
}

/// A fused `load + ALU` pair: `dst <- dst OP [mem]` in one instruction,
/// replacing a load into a transient plus a reg-reg ALU op. The loaded
/// register must additionally be proven dead by the caller.
pub(super) struct LoadAluFusion {
    pub op: MemAluOp,
    pub w32: bool,
    /// The ALU's destination, which is also its register operand.
    pub dst: MachineReg,
    /// The load's transient destination; dead after the ALU.
    pub loaded: MachineReg,
    pub mem: LoadAluMem,
}

#[derive(Clone, Copy)]
pub(super) enum LoadAluMem {
    Base(MachineAddr),
    Indexed {
        base: MachineReg,
        index: MachineReg,
        extend: MachineIndexExtend,
        offset: i32,
    },
}

/// Probe `prev = load, next = int-binary` for the memory-operand form.
/// The x86 shape is `dst = dst OP mem`, so the ALU's register operand
/// must be its destination; commutative ops accept either operand order.
/// Width must match exactly — the memory operand reads exactly the ALU
/// width — which also excludes every extending load except the
/// zero-extending 32-bit form consumed by a 32-bit op.
pub(super) fn load_alu_fusion(prev: &MachineInst, next: &MachineInst) -> Option<LoadAluFusion> {
    let (loaded, mem, mem_width) = match &prev.kind {
        MachineInstKind::Load {
            owner: MachineRegOwner::LinearValue,
            dst,
            addr,
            width,
            extension,
            ..
        } if load_form_matches(*width, *extension) => (*dst, LoadAluMem::Base(*addr), *width),
        MachineInstKind::IndexedLoad {
            dst,
            base,
            index,
            index_extend,
            offset,
            width,
            extension,
        } if load_form_matches(*width, *extension) => (
            *dst,
            LoadAluMem::Indexed {
                base: *base,
                index: *index,
                extend: *index_extend,
                offset: *offset,
            },
            *width,
        ),
        _ => return None,
    };
    let MachineInstKind::IntBinary {
        width,
        op,
        dst,
        lhs,
        rhs,
    } = &next.kind
    else {
        return None;
    };
    let (alu_op, commutative) = match op {
        MachineIntBinaryOp::Add => (MemAluOp::Add, true),
        MachineIntBinaryOp::Sub => (MemAluOp::Sub, false),
        MachineIntBinaryOp::And => (MemAluOp::And, true),
        MachineIntBinaryOp::Or => (MemAluOp::Or, true),
        MachineIntBinaryOp::Xor => (MemAluOp::Xor, true),
        _ => return None,
    };
    let w32 = match (width, mem_width) {
        (MachineIntWidth::I64, MachineMemWidth::U64) => false,
        (MachineIntWidth::I32, MachineMemWidth::U32) => true,
        _ => return None,
    };
    let dst = *dst;
    if dst == loaded {
        return None;
    }
    let direct = *lhs == MachineValue::Reg(dst) && *rhs == MachineValue::Reg(loaded);
    let swapped =
        commutative && *lhs == MachineValue::Reg(loaded) && *rhs == MachineValue::Reg(dst);
    if !direct && !swapped {
        return None;
    }
    Some(LoadAluFusion {
        op: alu_op,
        w32,
        dst,
        loaded,
        mem,
    })
}

/// Cheap buffering predicate: only loads that could take part in the
/// memory-operand fusion are worth holding in the lookahead slot.
pub(super) fn fusible_load(kind: &MachineInstKind) -> bool {
    match kind {
        MachineInstKind::Load {
            owner: MachineRegOwner::LinearValue,
            width,
            extension,
            ..
        }
        | MachineInstKind::IndexedLoad {
            width, extension, ..
        } => load_form_matches(*width, *extension),
        _ => false,
    }
}

fn load_form_matches(width: MachineMemWidth, extension: MachineLoadExtension) -> bool {
    match (width, extension) {
        (MachineMemWidth::U64, MachineLoadExtension::None)
        | (MachineMemWidth::U64, MachineLoadExtension::ZeroExtend)
        | (MachineMemWidth::U32, MachineLoadExtension::None)
        | (MachineMemWidth::U32, MachineLoadExtension::ZeroExtend) => true,
        _ => false,
    }
}

/// A fusible `IntCompare + Select` pair: the compare's boolean feeds the
/// select's condition and nothing else, so the select's CMOV can consume
/// the compare's flags directly — no SETcc materialization and no re-TEST.
/// The caller must additionally prove `bool_reg` dead after the select.
pub(super) struct IntCompareSelect {
    pub width: MachineIntWidth,
    pub kind: MachineCompareKind,
    pub sign: MachineSign,
    pub bool_reg: MachineReg,
    pub lhs: MachineValue,
    pub rhs: MachineValue,
    pub select_result: MachineReg,
}

pub(super) fn int_compare_select_fusion(
    compare: &MachineInst,
    select: &MachineInst,
) -> Option<IntCompareSelect> {
    let MachineInstKind::IntCompare {
        width,
        kind,
        sign,
        dst: bool_reg,
        lhs,
        rhs,
    } = compare.kind
    else {
        return None;
    };
    let MachineInstKind::Select {
        ty,
        dst: select_result,
        on_true,
        on_false,
        cond: MachineValue::Reg(cond),
    } = select.kind
    else {
        return None;
    };
    // GP selects only: the FP select path branches and re-tests, and V128
    // has no CMOV form.
    if ty.float_width().is_some()
        || ty == MachineStorageType::V128
        || cond != bool_reg
        || on_true == MachineValue::Reg(bool_reg)
        || on_false == MachineValue::Reg(bool_reg)
    {
        return None;
    }
    Some(IntCompareSelect {
        width,
        kind,
        sign,
        bool_reg,
        lhs,
        rhs,
        select_result,
    })
}

pub(super) fn map_int_cond(kind: MachineCompareKind, sign: MachineSign) -> Cc {
    match (kind, sign) {
        (MachineCompareKind::Eq, _) => Cc::E,
        (MachineCompareKind::Ne, _) => Cc::NE,
        (MachineCompareKind::Lt, MachineSign::Signed) => Cc::L,
        (MachineCompareKind::Lt, MachineSign::Unsigned) => Cc::B,
        (MachineCompareKind::Gt, MachineSign::Signed) => Cc::G,
        (MachineCompareKind::Gt, MachineSign::Unsigned) => Cc::A,
        (MachineCompareKind::Le, MachineSign::Signed) => Cc::LE,
        (MachineCompareKind::Le, MachineSign::Unsigned) => Cc::BE,
        (MachineCompareKind::Ge, MachineSign::Signed) => Cc::GE,
        (MachineCompareKind::Ge, MachineSign::Unsigned) => Cc::AE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::jit::machine::machine_ir::{MachineBlockId, MachineEdge};

    fn copy() -> MachineInst {
        MachineInst {
            kind: MachineInstKind::Move {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: MachineReg(7),
                src: MachineValue::Reg(MachineReg(9)),
            },
        }
    }

    fn load() -> MachineInst {
        MachineInst {
            kind: MachineInstKind::IndexedLoad {
                dst: MachineReg(4),
                base: MachineReg(2),
                index: MachineReg(7),
                index_extend: MachineIndexExtend::ZeroExtend32,
                offset: 0,
                width: MachineMemWidth::U8,
                extension: MachineLoadExtension::ZeroExtend,
            },
        }
    }

    fn address(inst: &MachineInst, regs: &[u64; 12]) -> u64 {
        let MachineInstKind::IndexedLoad {
            base,
            index,
            index_extend,
            offset,
            ..
        } = inst.kind
        else {
            panic!("expected indexed load")
        };
        let index = regs[usize::from(index.0)];
        let index = match index_extend {
            MachineIndexExtend::None => index,
            MachineIndexExtend::ZeroExtend32 => u64::from(index as u32),
        };
        regs[usize::from(base.0)]
            .wrapping_add(index)
            .wrapping_add(offset as u64)
    }

    #[test]
    fn copied_address_preserves_aliasing_high_bits_and_displacements() {
        for base in [2, 7, 9] {
            for index in [7, 9] {
                if base != 7 && index != 7 {
                    continue;
                }
                for dst in [4, 7, 9] {
                    for extend in [MachineIndexExtend::None, MachineIndexExtend::ZeroExtend32] {
                        for offset in [i32::MIN, -257, 0, 65535, i32::MAX] {
                            for width in [MachineMemWidth::U8, MachineMemWidth::U16] {
                                for extension in [
                                    MachineLoadExtension::None,
                                    MachineLoadExtension::ZeroExtend,
                                    MachineLoadExtension::SignExtend,
                                ] {
                                    let original = MachineInst {
                                        kind: MachineInstKind::IndexedLoad {
                                            dst: MachineReg(dst),
                                            base: MachineReg(base),
                                            index: MachineReg(index),
                                            index_extend: extend,
                                            offset,
                                            width,
                                            extension,
                                        },
                                    };
                                    let rewritten = copied_narrow_load(
                                        &copy(),
                                        &original,
                                        &[],
                                        &MachineTerminator::Return,
                                    )
                                    .unwrap();
                                    let mut regs = [0x1234_5678_8000_0042u64; 12];
                                    regs[7] = 0xdead_beef;
                                    let actual = address(&rewritten, &regs);
                                    regs[7] = regs[9];
                                    assert_eq!(actual, address(&original, &regs));
                                    let MachineInstKind::IndexedLoad {
                                        dst: new_dst,
                                        width: new_width,
                                        extension: new_extension,
                                        ..
                                    } = rewritten.kind
                                    else {
                                        unreachable!()
                                    };
                                    assert_eq!(
                                        (new_dst, new_width, new_extension),
                                        (MachineReg(dst), width, extension)
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn copied_address_requires_a_dead_linear_gp_snapshot() {
        let copy = copy();
        let load = load();
        let edge = MachineTerminator::Jump(MachineEdge {
            target: MachineBlockId(1),
            args: crate::collections::vec![MachineValue::Reg(MachineReg(7))],
        });
        assert!(copied_narrow_load(&copy, &load, &[], &edge).is_none());
        let use_copy = MachineInst {
            kind: MachineInstKind::Move {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: MachineReg(5),
                src: MachineValue::Reg(MachineReg(7)),
            },
        };
        assert!(
            copied_narrow_load(&copy, &load, &[use_copy], &MachineTerminator::Return).is_none()
        );
        let mut overwrite = load.clone();
        if let MachineInstKind::IndexedLoad { dst, .. } = &mut overwrite.kind {
            *dst = MachineReg(7);
        }
        assert!(copied_narrow_load(&copy, &overwrite, &[], &edge).is_some());
        for reg in [0, 1, 2, 3, 12, 40] {
            let mut invalid = copy.clone();
            if let MachineInstKind::Move { src, .. } = &mut invalid.kind {
                *src = MachineValue::Reg(MachineReg(reg));
            }
            assert!(copied_narrow_load(&invalid, &load, &[], &MachineTerminator::Return).is_none());
        }
        let mut cached = copy.clone();
        if let MachineInstKind::Move { owner, .. } = &mut cached.kind {
            *owner = MachineRegOwner::CachedCell;
        }
        assert!(copied_narrow_load(&cached, &load, &[], &MachineTerminator::Return).is_none());
        for width in [MachineMemWidth::U32, MachineMemWidth::U64] {
            let mut wide = load.clone();
            if let MachineInstKind::IndexedLoad { width: w, .. } = &mut wide.kind {
                *w = width;
            }
            assert!(copied_narrow_load(&copy, &wide, &[], &MachineTerminator::Return).is_none());
        }
    }
}
