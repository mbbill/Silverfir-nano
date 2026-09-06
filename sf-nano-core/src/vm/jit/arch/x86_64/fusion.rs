//! x86_64 condition-code mapping and instruction-pair fusion probes.

use super::enc::{Cc, MemAluOp};
use crate::vm::jit::machine::machine_ir::{
    MachineAddr, MachineCompareKind, MachineIndexExtend, MachineInst, MachineInstKind,
    MachineIntBinaryOp, MachineIntWidth, MachineLoadExtension, MachineMemWidth, MachineReg,
    MachineRegOwner, MachineSign, MachineStorageType, MachineValue, MACHINE_FP_REG,
};

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
/// An aligned native frame word is the sole narrowing exception: an i32
/// consumer can read its low half directly. Both widths touch the same page
/// of the aligned, runtime-owned frame, so no guest access or trap is elided.
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
        (MachineIntWidth::I32, MachineMemWidth::U64)
            if matches!(&mem, LoadAluMem::Base(addr)
                if addr.base == MACHINE_FP_REG && addr.offset % 8 == 0) =>
        {
            true
        }
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

    #[test]
    fn i32_alu_can_read_low_half_of_aligned_native_frame_word() {
        let alu = MachineInst {
            kind: MachineInstKind::IntBinary {
                width: MachineIntWidth::I32,
                op: MachineIntBinaryOp::Add,
                dst: MachineReg(5),
                lhs: MachineValue::Reg(MachineReg(5)),
                rhs: MachineValue::Reg(MachineReg(4)),
            },
        };
        for (base, offset, width, extension, accepted) in [
            (
                MACHINE_FP_REG,
                24,
                MachineMemWidth::U64,
                MachineLoadExtension::None,
                true,
            ),
            (
                MACHINE_FP_REG,
                24,
                MachineMemWidth::U64,
                MachineLoadExtension::ZeroExtend,
                true,
            ),
            (
                MACHINE_FP_REG,
                28,
                MachineMemWidth::U64,
                MachineLoadExtension::None,
                false,
            ),
            (
                MachineReg(6),
                24,
                MachineMemWidth::U64,
                MachineLoadExtension::None,
                false,
            ),
            (
                MACHINE_FP_REG,
                24,
                MachineMemWidth::U16,
                MachineLoadExtension::None,
                false,
            ),
            (
                MACHINE_FP_REG,
                24,
                MachineMemWidth::U32,
                MachineLoadExtension::SignExtend,
                false,
            ),
        ] {
            let load = MachineInst {
                kind: MachineInstKind::Load {
                    owner: MachineRegOwner::LinearValue,
                    ty: MachineStorageType::GpWord,
                    dst: MachineReg(4),
                    addr: MachineAddr { base, offset },
                    width,
                    extension,
                },
            };
            let fusion = load_alu_fusion(&load, &alu);
            assert_eq!(
                fusion.is_some(),
                accepted,
                "{base:?} + {offset}, {width:?}, {extension:?}"
            );
            if let Some(fusion) = fusion {
                assert!(fusion.w32);
            }
        }
    }
}
