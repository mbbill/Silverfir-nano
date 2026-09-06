//! x86_64 condition-code mapping and instruction-pair fusion probes.

use super::enc::{Cc, MemAluOp};
use crate::vm::jit::machine::machine_ir::{
    MachineAddr, MachineCompareKind, MachineIndexExtend, MachineInst, MachineInstKind,
    MachineIntBinaryOp, MachineIntWidth, MachineLoadExtension, MachineMemWidth, MachineReg,
    MachineRegOwner, MachineSign, MachineStorageType, MachineValue,
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

#[derive(Clone, Copy, PartialEq, Eq)]
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
    let (loaded, mem, mem_width) = transient_load(prev)?;
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

fn transient_load(inst: &MachineInst) -> Option<(MachineReg, LoadAluMem, MachineMemWidth)> {
    Some(match &inst.kind {
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
    })
}

/// A non-atomic memory update replacing an adjacent load, immediate ALU and
/// exact store. The caller must prove the transient value dead after the store.
pub(super) struct MemoryUpdateFusion {
    pub op: MemAluOp,
    pub w32: bool,
    pub immediate: i32,
    pub loaded: MachineReg,
    pub mem: LoadAluMem,
}

pub(super) fn memory_update_fusion(
    load: &MachineInst,
    alu: &MachineInst,
    store: &MachineInst,
) -> Option<MemoryUpdateFusion> {
    let (loaded, mem, mem_width) = transient_load(load)?;
    let MachineInstKind::IntBinary {
        width,
        op,
        dst,
        lhs,
        rhs,
    } = alu.kind
    else {
        return None;
    };
    let (op, commutative) = match op {
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
    if dst != loaded {
        return None;
    }
    let immediate = match (lhs, rhs) {
        (MachineValue::Reg(reg), MachineValue::Imm64(value)) if reg == loaded => value,
        (MachineValue::Imm64(value), MachineValue::Reg(reg)) if commutative && reg == loaded => {
            value
        }
        _ => return None,
    };
    let encoded = immediate as i32;
    if !w32 && encoded as i64 as u64 != immediate {
        return None;
    }
    let (stored_mem, stored_width, src) = match store.kind {
        MachineInstKind::Store {
            ty: MachineStorageType::GpWord | MachineStorageType::GpI64,
            addr,
            width,
            src,
        } => (LoadAluMem::Base(addr), width, src),
        MachineInstKind::IndexedStore {
            base,
            index,
            index_extend,
            offset,
            width,
            src,
        } => (
            LoadAluMem::Indexed {
                base,
                index,
                extend: index_extend,
                offset,
            },
            width,
            src,
        ),
        _ => return None,
    };
    if stored_mem != mem || stored_width != mem_width || src != MachineValue::Reg(loaded) {
        return None;
    }
    // Syntactically equal addresses need not denote the same location if
    // the load/ALU overwrote an address register in between.
    if match mem {
        LoadAluMem::Base(addr) => addr.base == loaded,
        LoadAluMem::Indexed { base, index, .. } => base == loaded || index == loaded,
    } {
        return None;
    }
    Some(MemoryUpdateFusion {
        op,
        w32,
        immediate: encoded,
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
mod memory_update_tests {
    use super::*;

    fn sequence() -> [MachineInst; 3] {
        let addr = MachineAddr {
            base: MachineReg(5),
            offset: 8,
        };
        [
            MachineInst {
                kind: MachineInstKind::Load {
                    owner: MachineRegOwner::LinearValue,
                    ty: MachineStorageType::GpWord,
                    dst: MachineReg(4),
                    addr,
                    width: MachineMemWidth::U32,
                    extension: MachineLoadExtension::ZeroExtend,
                },
            },
            MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: MachineIntWidth::I32,
                    op: MachineIntBinaryOp::Add,
                    dst: MachineReg(4),
                    lhs: MachineValue::Reg(MachineReg(4)),
                    rhs: MachineValue::Imm64(1),
                },
            },
            MachineInst {
                kind: MachineInstKind::Store {
                    ty: MachineStorageType::GpWord,
                    addr,
                    width: MachineMemWidth::U32,
                    src: MachineValue::Reg(MachineReg(4)),
                },
            },
        ]
    }

    #[test]
    fn requires_exact_width_address_and_unmodified_address_registers() {
        let s = sequence();
        assert!(memory_update_fusion(&s[0], &s[1], &s[2]).is_some());
        for case in 0..6 {
            let mut s = sequence();
            match case {
                0 => {
                    if let MachineInstKind::Store { addr, .. } = &mut s[2].kind {
                        addr.offset += 4;
                    }
                }
                1 => {
                    if let MachineInstKind::Store { width, .. } = &mut s[2].kind {
                        *width = MachineMemWidth::U64;
                    }
                }
                2 => {
                    if let MachineInstKind::Load { addr, .. } = &mut s[0].kind {
                        addr.base = MachineReg(4);
                    }
                    if let MachineInstKind::Store { addr, .. } = &mut s[2].kind {
                        addr.base = MachineReg(4);
                    }
                }
                3 => {
                    if let MachineInstKind::IntBinary { dst, .. } = &mut s[1].kind {
                        *dst = MachineReg(6);
                    }
                }
                4 => {
                    if let MachineInstKind::IntBinary { op, .. } = &mut s[1].kind {
                        *op = MachineIntBinaryOp::DivS;
                    }
                }
                5 => {
                    if let MachineInstKind::Store { src, .. } = &mut s[2].kind {
                        *src = MachineValue::Reg(MachineReg(6));
                    }
                }
                _ => unreachable!(),
            }
            assert!(memory_update_fusion(&s[0], &s[1], &s[2]).is_none());
        }
    }

    #[test]
    fn indexed_updates_preserve_index_extension_and_address_inputs() {
        for (index, stored_extend, accepted) in [
            (MachineReg(5), MachineIndexExtend::ZeroExtend32, true),
            (MachineReg(4), MachineIndexExtend::ZeroExtend32, false),
            (MachineReg(5), MachineIndexExtend::None, false),
        ] {
            let mut s = sequence();
            s[0].kind = MachineInstKind::IndexedLoad {
                dst: MachineReg(4),
                base: MachineReg(2),
                index,
                index_extend: MachineIndexExtend::ZeroExtend32,
                offset: 8,
                width: MachineMemWidth::U32,
                extension: MachineLoadExtension::ZeroExtend,
            };
            s[2].kind = MachineInstKind::IndexedStore {
                base: MachineReg(2),
                index,
                index_extend: stored_extend,
                offset: 8,
                width: MachineMemWidth::U32,
                src: MachineValue::Reg(MachineReg(4)),
            };
            assert_eq!(
                memory_update_fusion(&s[0], &s[1], &s[2]).is_some(),
                accepted
            );
        }
    }

    #[test]
    fn wide_updates_require_exact_sign_extended_immediates() {
        for (immediate, accepted) in [(0x8000_0000, false), (0xffff_ffff_8000_0000, true)] {
            let mut s = sequence();
            if let MachineInstKind::Load { width, .. } = &mut s[0].kind {
                *width = MachineMemWidth::U64;
            }
            if let MachineInstKind::IntBinary { width, rhs, .. } = &mut s[1].kind {
                *width = MachineIntWidth::I64;
                *rhs = MachineValue::Imm64(immediate);
            }
            if let MachineInstKind::Store { width, .. } = &mut s[2].kind {
                *width = MachineMemWidth::U64;
            }
            assert_eq!(
                memory_update_fusion(&s[0], &s[1], &s[2]).is_some(),
                accepted
            );
        }
    }
}
