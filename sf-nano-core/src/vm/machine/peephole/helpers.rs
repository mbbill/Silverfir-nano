//! Shared helper functions for peephole optimization passes.

use crate::collections;

use crate::vm::backend::BackendConfig;
use crate::vm::machine::machine_ir::{
    is_fp_reg, is_gp_reg, MachineAddr, MachineArgSrc, MachineArgSrcPair, MachineBranchCond,
    MachineCallArgs, MachineCallLaneArg, MachineCallResults, MachineCallTarget, MachineEdge,
    MachineInst, MachineInstKind, MachineIntUnaryOp, MachineMemWidth, MachineReg, MachineResultDst,
    MachineResultSrc, MachineReturnValue, MachineStorageType, MachineTerminator, MachineValue,
};

// --- Instruction analysis ---

/// Return the single destination register defined by an instruction, if any.
pub(super) fn defined_reg(kind: &MachineInstKind) -> Option<MachineReg> {
    match kind {
        MachineInstKind::Move { dst, .. }
        | MachineInstKind::FloatConst { dst, .. }
        | MachineInstKind::Load { dst, .. }
        | MachineInstKind::IntUnary { dst, .. }
        | MachineInstKind::IntBinary { dst, .. }
        | MachineInstKind::IntCompare { dst, .. }
        | MachineInstKind::FloatUnary { dst, .. }
        | MachineInstKind::FloatBinary { dst, .. }
        | MachineInstKind::FloatCompare { dst, .. }
        | MachineInstKind::Convert { dst, .. }
        | MachineInstKind::Select { dst, .. }
        | MachineInstKind::IndexedLoad { dst, .. }
        | MachineInstKind::BitfieldExtractU { dst, .. }
        | MachineInstKind::IntBinaryShifted { dst, .. }
        | MachineInstKind::TestBits { dst, .. }
        | MachineInstKind::EhAllocExnRef { dst, .. }
        | MachineInstKind::RefFunc { dst, .. }
        | MachineInstKind::RefAsNonNull { dst, .. }
        | MachineInstKind::RefEq { dst, .. }
        | MachineInstKind::RefI31 { dst, .. }
        | MachineInstKind::I31GetS { dst, .. }
        | MachineInstKind::I31GetU { dst, .. }
        | MachineInstKind::AnyConvertExtern { dst, .. }
        | MachineInstKind::ExternConvertAny { dst, .. }
        | MachineInstKind::RefTest { dst, .. }
        | MachineInstKind::RefCast { dst, .. }
        | MachineInstKind::StructNew { dst, .. }
        | MachineInstKind::StructNewDefault { dst, .. }
        | MachineInstKind::ArrayNew { dst, .. }
        | MachineInstKind::ArrayNewDefault { dst, .. }
        | MachineInstKind::ArrayNewFixed { dst, .. }
        | MachineInstKind::ArrayNewData { dst, .. }
        | MachineInstKind::ArrayNewElem { dst, .. }
        | MachineInstKind::ArrayLen { dst, .. } => Some(*dst),
        #[cfg(sf_has_simd)]
        MachineInstKind::V128Const { dst, .. }
        | MachineInstKind::V128FromRaw { dst, .. }
        | MachineInstKind::V128ToRaw { dst, .. }
        | MachineInstKind::SimdUnary { dst, .. }
        | MachineInstKind::SimdBinary { dst, .. }
        | MachineInstKind::SimdTernary { dst, .. }
        | MachineInstKind::SimdShift { dst, .. }
        | MachineInstKind::SimdExtractLane { dst, .. }
        | MachineInstKind::SimdReplaceLane { dst, .. }
        | MachineInstKind::SimdShuffle { dst, .. }
        | MachineInstKind::SimdLoad { dst, .. }
        | MachineInstKind::SimdLoadLane { dst, .. } => Some(*dst),
        MachineInstKind::StructGet { dst, dst_hi, .. }
        | MachineInstKind::ArrayGet { dst, dst_hi, .. } => dst_hi.is_none().then_some(*dst),
        MachineInstKind::MemoryGrow { dst, .. } | MachineInstKind::TableGrow { dst, .. } => {
            Some(*dst)
        }
        MachineInstKind::MemoryFill { .. }
        | MachineInstKind::MemoryCopy { .. }
        | MachineInstKind::MemoryInit { .. }
        | MachineInstKind::DataDrop { .. }
        | MachineInstKind::TableFill { .. }
        | MachineInstKind::TableCopy { .. }
        | MachineInstKind::TableInit { .. }
        | MachineInstKind::ElemDrop { .. }
        | MachineInstKind::StructSet { .. }
        | MachineInstKind::ArraySet { .. }
        | MachineInstKind::ArrayFill { .. }
        | MachineInstKind::ArrayCopy { .. }
        | MachineInstKind::ArrayInitData { .. }
        | MachineInstKind::ArrayInitElem { .. } => None,
        MachineInstKind::Int64PairBinary { .. } => None,
        MachineInstKind::Int64PairUnary { .. } => None,
        MachineInstKind::Int64PairDivRem { .. } => None,
        MachineInstKind::Int64PairShift { .. } => None,
        MachineInstKind::Int64MulFromSignExt32 { .. } => None,
        MachineInstKind::Int64PairCompare { dst, .. } => Some(*dst),
        MachineInstKind::ConvertFloatToI64Pair { .. } => None,
        MachineInstKind::ConvertI64PairToFloat { dst, .. }
        | MachineInstKind::ReinterpretI64PairToF64 { dst, .. } => Some(*dst),
        MachineInstKind::ReinterpretF64ToI64Pair { .. } => None,
        MachineInstKind::Store { .. }
        | MachineInstKind::IndexedStore { .. }
        | MachineInstKind::TrapIf { .. }
        | MachineInstKind::CallRuntime(_)
        | MachineInstKind::EhThrow { .. }
        | MachineInstKind::EhThrowRef { .. } => None,
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdStore { .. } | MachineInstKind::SimdStoreLane { .. } => None,
    }
}

/// Visit every register defined by an instruction, including both halves of
/// legalized i64 pair destinations on 32-bit backends.
pub(super) fn for_each_defined_reg(kind: &MachineInstKind, mut f: impl FnMut(MachineReg)) {
    if let Some(dst) = defined_reg(kind) {
        f(dst);
    }
    match kind {
        MachineInstKind::Int64PairBinary { dst_lo, dst_hi, .. }
        | MachineInstKind::Int64PairUnary { dst_lo, dst_hi, .. }
        | MachineInstKind::Int64PairDivRem { dst_lo, dst_hi, .. }
        | MachineInstKind::Int64PairShift { dst_lo, dst_hi, .. }
        | MachineInstKind::Int64MulFromSignExt32 { dst_lo, dst_hi, .. }
        | MachineInstKind::ConvertFloatToI64Pair { dst_lo, dst_hi, .. }
        | MachineInstKind::ReinterpretF64ToI64Pair { dst_lo, dst_hi, .. }
        | MachineInstKind::StructGet {
            dst: dst_lo,
            dst_hi: Some(dst_hi),
            ..
        }
        | MachineInstKind::ArrayGet {
            dst: dst_lo,
            dst_hi: Some(dst_hi),
            ..
        } => {
            f(*dst_lo);
            f(*dst_hi);
        }
        _ => {}
    }
}

/// Check if `kind` defines (writes to) `reg`.
pub(super) fn inst_defines(kind: &MachineInstKind, reg: MachineReg) -> bool {
    match kind {
        MachineInstKind::Move { dst, .. }
        | MachineInstKind::FloatConst { dst, .. }
        | MachineInstKind::Load { dst, .. }
        | MachineInstKind::IntUnary { dst, .. }
        | MachineInstKind::IntBinary { dst, .. }
        | MachineInstKind::IntCompare { dst, .. }
        | MachineInstKind::FloatUnary { dst, .. }
        | MachineInstKind::FloatBinary { dst, .. }
        | MachineInstKind::FloatCompare { dst, .. }
        | MachineInstKind::Convert { dst, .. }
        | MachineInstKind::Select { dst, .. }
        | MachineInstKind::IndexedLoad { dst, .. }
        | MachineInstKind::BitfieldExtractU { dst, .. }
        | MachineInstKind::IntBinaryShifted { dst, .. }
        | MachineInstKind::TestBits { dst, .. } => *dst == reg,
        #[cfg(sf_has_simd)]
        MachineInstKind::V128Const { dst, .. }
        | MachineInstKind::V128FromRaw { dst, .. }
        | MachineInstKind::V128ToRaw { dst, .. }
        | MachineInstKind::SimdUnary { dst, .. }
        | MachineInstKind::SimdBinary { dst, .. }
        | MachineInstKind::SimdTernary { dst, .. }
        | MachineInstKind::SimdShift { dst, .. }
        | MachineInstKind::SimdExtractLane { dst, .. }
        | MachineInstKind::SimdReplaceLane { dst, .. }
        | MachineInstKind::SimdShuffle { dst, .. }
        | MachineInstKind::SimdLoad { dst, .. }
        | MachineInstKind::SimdLoadLane { dst, .. } => *dst == reg,
        MachineInstKind::MemoryGrow { dst, .. }
        | MachineInstKind::TableGrow { dst, .. }
        | MachineInstKind::EhAllocExnRef { dst, .. } => *dst == reg,
        MachineInstKind::MemoryFill { .. }
        | MachineInstKind::MemoryCopy { .. }
        | MachineInstKind::MemoryInit { .. }
        | MachineInstKind::DataDrop { .. }
        | MachineInstKind::TableFill { .. }
        | MachineInstKind::TableCopy { .. }
        | MachineInstKind::TableInit { .. }
        | MachineInstKind::ElemDrop { .. } => false,
        MachineInstKind::Int64PairBinary { dst_lo, dst_hi, .. } => *dst_lo == reg || *dst_hi == reg,
        MachineInstKind::Int64PairUnary { dst_lo, dst_hi, .. } => *dst_lo == reg || *dst_hi == reg,
        MachineInstKind::Int64PairDivRem { dst_lo, dst_hi, .. } => *dst_lo == reg || *dst_hi == reg,
        MachineInstKind::Int64PairShift { dst_lo, dst_hi, .. } => *dst_lo == reg || *dst_hi == reg,
        MachineInstKind::Int64MulFromSignExt32 { dst_lo, dst_hi, .. } => {
            *dst_lo == reg || *dst_hi == reg
        }
        MachineInstKind::Int64PairCompare { dst, .. } => *dst == reg,
        MachineInstKind::ConvertFloatToI64Pair { dst_lo, dst_hi, .. } => {
            *dst_lo == reg || *dst_hi == reg
        }
        MachineInstKind::ReinterpretF64ToI64Pair { dst_lo, dst_hi, .. } => {
            *dst_lo == reg || *dst_hi == reg
        }
        MachineInstKind::ConvertI64PairToFloat { dst, .. }
        | MachineInstKind::ReinterpretI64PairToF64 { dst, .. } => *dst == reg,
        MachineInstKind::RefFunc { dst, .. }
        | MachineInstKind::RefAsNonNull { dst, .. }
        | MachineInstKind::RefEq { dst, .. }
        | MachineInstKind::RefI31 { dst, .. }
        | MachineInstKind::I31GetS { dst, .. }
        | MachineInstKind::I31GetU { dst, .. }
        | MachineInstKind::AnyConvertExtern { dst, .. }
        | MachineInstKind::ExternConvertAny { dst, .. }
        | MachineInstKind::RefTest { dst, .. }
        | MachineInstKind::RefCast { dst, .. }
        | MachineInstKind::StructNew { dst, .. }
        | MachineInstKind::StructNewDefault { dst, .. }
        | MachineInstKind::ArrayNew { dst, .. }
        | MachineInstKind::ArrayNewDefault { dst, .. }
        | MachineInstKind::ArrayNewFixed { dst, .. }
        | MachineInstKind::ArrayNewData { dst, .. }
        | MachineInstKind::ArrayNewElem { dst, .. }
        | MachineInstKind::ArrayLen { dst, .. } => *dst == reg,
        MachineInstKind::StructGet { dst, dst_hi, .. }
        | MachineInstKind::ArrayGet { dst, dst_hi, .. } => {
            *dst == reg || dst_hi.is_some_and(|dst_hi| dst_hi == reg)
        }
        MachineInstKind::Store { .. }
        | MachineInstKind::IndexedStore { .. }
        | MachineInstKind::TrapIf { .. }
        | MachineInstKind::CallRuntime(_)
        | MachineInstKind::EhThrow { .. }
        | MachineInstKind::EhThrowRef { .. }
        | MachineInstKind::StructSet { .. }
        | MachineInstKind::ArraySet { .. }
        | MachineInstKind::ArrayFill { .. }
        | MachineInstKind::ArrayCopy { .. }
        | MachineInstKind::ArrayInitData { .. }
        | MachineInstKind::ArrayInitElem { .. } => false,
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdStore { .. } | MachineInstKind::SimdStoreLane { .. } => false,
    }
}

/// Check if an instruction uses `reg` as a source operand.
pub(super) fn inst_uses_value(kind: &MachineInstKind, reg: MachineReg) -> bool {
    let mut found = false;
    visit_source_values(kind, |v| {
        if matches!(v, MachineValue::Reg(r) if *r == reg) {
            found = true;
        }
    });
    found
}

/// Count how many times `reg` appears as a source operand in `kind`.
pub(super) fn count_value_uses(kind: &MachineInstKind, reg: MachineReg) -> usize {
    let mut count = 0;
    visit_source_values(kind, |v| {
        if matches!(v, MachineValue::Reg(r) if *r == reg) {
            count += 1;
        }
    });
    count
}

/// Visit all source (read) values in an instruction.
pub(crate) fn visit_source_values(kind: &MachineInstKind, mut f: impl FnMut(&MachineValue)) {
    match kind {
        MachineInstKind::Move { src, .. } => f(src),
        MachineInstKind::FloatConst { .. } => {}
        #[cfg(sf_has_simd)]
        MachineInstKind::V128Const { .. } => {}
        #[cfg(sf_has_simd)]
        MachineInstKind::V128FromRaw { raw, .. } => f(raw),
        #[cfg(sf_has_simd)]
        MachineInstKind::V128ToRaw { src, .. } => f(src),
        MachineInstKind::Load { addr, .. } => {
            f(&MachineValue::Reg(addr.base));
        }
        MachineInstKind::Store { addr, src, .. } => {
            f(&MachineValue::Reg(addr.base));
            f(src);
        }
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdLoad { addr, .. } => {
            f(&MachineValue::Reg(addr.base));
        }
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdStore { addr, src, .. } => {
            f(&MachineValue::Reg(addr.base));
            f(src);
        }
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdLoadLane { addr, vector, .. } => {
            f(&MachineValue::Reg(addr.base));
            f(vector);
        }
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdStoreLane { addr, vector, .. } => {
            f(&MachineValue::Reg(addr.base));
            f(vector);
        }
        MachineInstKind::IndexedLoad { base, index, .. } => {
            f(&MachineValue::Reg(*base));
            f(&MachineValue::Reg(*index));
        }
        MachineInstKind::IndexedStore {
            base, index, src, ..
        } => {
            f(&MachineValue::Reg(*base));
            f(&MachineValue::Reg(*index));
            f(src);
        }
        MachineInstKind::IntUnary { src, .. }
        | MachineInstKind::FloatUnary { src, .. }
        | MachineInstKind::Convert { src, .. } => f(src),
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdUnary { src, .. } | MachineInstKind::SimdExtractLane { src, .. } => {
            f(src)
        }
        MachineInstKind::IntBinary { lhs, rhs, .. }
        | MachineInstKind::IntCompare { lhs, rhs, .. }
        | MachineInstKind::FloatBinary { lhs, rhs, .. }
        | MachineInstKind::FloatCompare { lhs, rhs, .. } => {
            f(lhs);
            f(rhs);
        }
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdBinary { lhs, rhs, .. } => {
            f(lhs);
            f(rhs);
        }
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdTernary { a, b, c, .. } => {
            f(a);
            f(b);
            f(c);
        }
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdShift { vector, shift, .. } => {
            f(vector);
            f(shift);
        }
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdReplaceLane { vector, scalar, .. } => {
            f(vector);
            f(scalar);
        }
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdShuffle { lhs, rhs, .. } => {
            f(lhs);
            f(rhs);
        }
        MachineInstKind::Int64PairBinary {
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        } => {
            f(lhs_lo);
            f(lhs_hi);
            f(rhs_lo);
            f(rhs_hi);
        }
        MachineInstKind::Int64PairUnary {
            op:
                MachineIntUnaryOp::Extend8S
                | MachineIntUnaryOp::Extend16S
                | MachineIntUnaryOp::Extend32S,
            src_lo,
            ..
        } => {
            f(src_lo);
        }
        MachineInstKind::Int64PairUnary { src_lo, src_hi, .. } => {
            f(src_lo);
            f(src_hi);
        }
        MachineInstKind::Int64PairDivRem {
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        } => {
            f(lhs_lo);
            f(lhs_hi);
            f(rhs_lo);
            f(rhs_hi);
        }
        MachineInstKind::Int64PairShift {
            lhs_lo,
            lhs_hi,
            rhs,
            ..
        } => {
            f(lhs_lo);
            f(lhs_hi);
            f(rhs);
        }
        MachineInstKind::Int64MulFromSignExt32 { lhs, rhs, .. } => {
            f(lhs);
            f(rhs);
        }
        MachineInstKind::Int64PairCompare {
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        } => {
            f(lhs_lo);
            f(lhs_hi);
            f(rhs_lo);
            f(rhs_hi);
        }
        MachineInstKind::ConvertI64PairToFloat { src_lo, src_hi, .. } => {
            f(src_lo);
            f(src_hi);
        }
        MachineInstKind::ConvertFloatToI64Pair { src, .. }
        | MachineInstKind::ReinterpretF64ToI64Pair { src, .. } => f(src),
        MachineInstKind::ReinterpretI64PairToF64 { src_lo, src_hi, .. } => {
            f(src_lo);
            f(src_hi);
        }
        MachineInstKind::Select {
            on_true,
            on_false,
            cond,
            ..
        } => {
            f(on_true);
            f(on_false);
            f(cond);
        }
        MachineInstKind::BitfieldExtractU { src, .. } => {
            f(&MachineValue::Reg(*src));
        }
        MachineInstKind::IntBinaryShifted { lhs, rhs, .. } => {
            f(&MachineValue::Reg(*lhs));
            f(&MachineValue::Reg(*rhs));
        }
        MachineInstKind::TestBits { src, mask, .. } => {
            f(&MachineValue::Reg(*src));
            f(mask);
        }
        MachineInstKind::TrapIf { cond, .. } => visit_branch_cond_values(cond, &mut f),
        MachineInstKind::CallRuntime(_)
        | MachineInstKind::EhThrow { .. }
        | MachineInstKind::EhThrowRef { .. }
        | MachineInstKind::EhAllocExnRef { .. } => {}
        MachineInstKind::RefFunc { .. } => {}
        MachineInstKind::StructNew { fields, .. } => {
            for (value_lo, value_hi) in fields {
                f(value_lo);
                if let Some(value_hi) = value_hi {
                    f(value_hi);
                }
            }
        }
        MachineInstKind::StructNewDefault { .. } => {}
        MachineInstKind::RefAsNonNull { src, .. }
        | MachineInstKind::RefI31 { src, .. }
        | MachineInstKind::I31GetS { src, .. }
        | MachineInstKind::I31GetU { src, .. }
        | MachineInstKind::AnyConvertExtern { src, .. }
        | MachineInstKind::ExternConvertAny { src, .. }
        | MachineInstKind::RefTest { src, .. }
        | MachineInstKind::RefCast { src, .. }
        | MachineInstKind::StructGet { src, .. }
        | MachineInstKind::ArrayNewDefault { length: src, .. }
        | MachineInstKind::ArrayLen { src, .. } => f(src),
        MachineInstKind::ArrayNew {
            init_lo,
            init_hi,
            length,
            ..
        } => {
            f(init_lo);
            if let Some(init_hi) = init_hi {
                f(init_hi);
            }
            f(length);
        }
        MachineInstKind::ArrayNewFixed { elements, .. } => {
            for (value_lo, value_hi) in elements {
                f(value_lo);
                if let Some(value_hi) = value_hi {
                    f(value_hi);
                }
            }
        }
        MachineInstKind::ArrayNewData { src, len, .. }
        | MachineInstKind::ArrayNewElem { src, len, .. } => {
            f(src);
            f(len);
        }
        MachineInstKind::RefEq { lhs, rhs, .. } => {
            f(lhs);
            f(rhs);
        }
        MachineInstKind::ArrayGet { ref_src, index, .. } => {
            f(ref_src);
            f(index);
        }
        MachineInstKind::ArraySet {
            ref_src,
            index,
            value_lo,
            value_hi,
            ..
        } => {
            f(ref_src);
            f(index);
            f(value_lo);
            if let Some(value_hi) = value_hi {
                f(value_hi);
            }
        }
        MachineInstKind::ArrayFill {
            ref_src,
            index,
            value_lo,
            value_hi,
            len,
            ..
        } => {
            f(ref_src);
            f(index);
            f(value_lo);
            if let Some(value_hi) = value_hi {
                f(value_hi);
            }
            f(len);
        }
        MachineInstKind::ArrayCopy {
            dst_ref,
            dst_index,
            src_ref,
            src_index,
            len,
            ..
        } => {
            f(dst_ref);
            f(dst_index);
            f(src_ref);
            f(src_index);
            f(len);
        }
        MachineInstKind::ArrayInitData {
            ref_src,
            dst_index,
            src_index,
            len,
            ..
        }
        | MachineInstKind::ArrayInitElem {
            ref_src,
            dst_index,
            src_index,
            len,
            ..
        } => {
            f(ref_src);
            f(dst_index);
            f(src_index);
            f(len);
        }
        MachineInstKind::StructSet {
            ref_src,
            value_lo,
            value_hi,
            ..
        } => {
            f(ref_src);
            f(value_lo);
            if let Some(value_hi) = value_hi {
                f(value_hi);
            }
        }
        MachineInstKind::MemoryGrow { delta, .. } => {
            f(delta);
        }
        MachineInstKind::MemoryFill { dest, val, len, .. }
        | MachineInstKind::TableFill {
            start: dest,
            val,
            len,
            ..
        } => {
            f(dest);
            f(val);
            f(len);
        }
        MachineInstKind::MemoryCopy { dest, src, len, .. }
        | MachineInstKind::MemoryInit { dest, src, len, .. }
        | MachineInstKind::TableCopy { dest, src, len, .. }
        | MachineInstKind::TableInit { dest, src, len, .. } => {
            f(dest);
            f(src);
            f(len);
        }
        MachineInstKind::TableGrow {
            init_val, delta, ..
        } => {
            f(init_val);
            f(delta);
        }
        MachineInstKind::DataDrop { .. } | MachineInstKind::ElemDrop { .. } => {}
    }
}

pub(super) fn visit_branch_cond_values(cond: &MachineBranchCond, mut f: impl FnMut(&MachineValue)) {
    match cond {
        MachineBranchCond::Value(value) => f(value),
        MachineBranchCond::IntCompare { lhs, rhs, .. } => {
            f(lhs);
            f(rhs);
        }
        MachineBranchCond::TestBits { src, mask, .. } => {
            f(src);
            f(mask);
        }
    }
}

pub(super) fn value_is_reg(v: &MachineValue, reg: MachineReg) -> bool {
    matches!(v, MachineValue::Reg(r) if *r == reg)
}

// --- Liveness queries ---

/// Check if `reg` is used by any instruction in `ops` or the terminator before
/// being redefined.
pub(super) fn reg_live_after(
    ops: &[MachineInst],
    term: &MachineTerminator,
    reg: MachineReg,
) -> bool {
    for inst in ops {
        if inst_uses_value(&inst.kind, reg) {
            return true;
        }
        if inst_defines(&inst.kind, reg) {
            return false;
        }
    }
    terminator_uses_reg(term, reg)
}

/// Check if a terminator reads from `reg`.
pub(super) fn terminator_uses_reg(term: &MachineTerminator, reg: MachineReg) -> bool {
    match term {
        MachineTerminator::Jump(edge) => edge_uses_reg(edge, reg),
        MachineTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => {
            branch_cond_uses_reg(cond, reg)
                || edge_uses_reg(then_edge, reg)
                || edge_uses_reg(else_edge, reg)
        }
        MachineTerminator::JumpTable { index, entries } => {
            value_is_reg(index, reg) || entries.iter().any(|e| edge_uses_reg(e, reg))
        }
        MachineTerminator::Call {
            target,
            args,
            results,
            success,
            ..
        } => {
            call_target_uses_reg(target, reg)
                || call_args_use_reg(args, reg)
                || call_success_uses_reg(success, results, reg)
        }
        MachineTerminator::TailCall { target, args } => {
            call_target_uses_reg(target, reg) || call_args_use_reg(args, reg)
        }
        MachineTerminator::Return | MachineTerminator::Trap { .. } => false,
        MachineTerminator::ReturnScalar { value } => return_value_uses_reg(value, reg),
    }
}

fn call_target_uses_reg(target: &MachineCallTarget, reg: MachineReg) -> bool {
    match target {
        MachineCallTarget::Direct(_) => false,
        MachineCallTarget::Indirect {
            callee_target,
            callee_entry,
        } => *callee_target == reg || *callee_entry == reg,
    }
}

fn call_args_use_reg(args: &MachineCallArgs, reg: MachineReg) -> bool {
    args.lane_args.iter().any(|arg| match arg {
        MachineCallLaneArg::Gp { src, .. } | MachineCallLaneArg::Fp { src, .. } => {
            arg_src_uses_reg(src, reg)
        }
        MachineCallLaneArg::GpPair { src, .. } => arg_src_pair_uses_reg(src, reg),
    })
}

fn arg_src_pair_uses_reg(src: &MachineArgSrcPair, reg: MachineReg) -> bool {
    arg_src_uses_reg(&src.lo, reg) || arg_src_uses_reg(&src.hi, reg)
}

fn arg_src_uses_reg(src: &MachineArgSrc, reg: MachineReg) -> bool {
    matches!(src, MachineArgSrc::Reg(src) if *src == reg)
}

fn return_value_uses_reg(value: &MachineReturnValue, reg: MachineReg) -> bool {
    match value {
        MachineReturnValue::ScalarGp { src, .. } | MachineReturnValue::ScalarFp { src, .. } => {
            result_src_uses_reg(src, reg)
        }
        MachineReturnValue::ScalarGpPair { lo, hi } => {
            result_src_uses_reg(lo, reg) || result_src_uses_reg(hi, reg)
        }
    }
}

fn result_src_uses_reg(src: &MachineResultSrc, reg: MachineReg) -> bool {
    matches!(src, MachineResultSrc::Reg(src) if *src == reg)
}

fn call_success_uses_reg(
    success: &MachineEdge,
    results: &MachineCallResults,
    reg: MachineReg,
) -> bool {
    success
        .args
        .iter()
        .any(|v| value_is_reg(v, reg) && !call_results_define_reg(results, reg))
}

fn call_results_define_reg(results: &MachineCallResults, reg: MachineReg) -> bool {
    match results {
        MachineCallResults::None | MachineCallResults::FrameFallback { .. } => false,
        MachineCallResults::ScalarGp { dst, .. } | MachineCallResults::ScalarFp { dst, .. } => {
            result_dst_is_reg(*dst, reg)
        }
        MachineCallResults::ScalarGpPair { lo, hi } => {
            result_dst_is_reg(*lo, reg) || result_dst_is_reg(*hi, reg)
        }
    }
}

fn result_dst_is_reg(dst: MachineResultDst, reg: MachineReg) -> bool {
    matches!(dst, MachineResultDst::Reg(dst) if dst == reg)
}

fn edge_uses_reg(edge: &MachineEdge, reg: MachineReg) -> bool {
    edge.args.iter().any(|v| value_is_reg(v, reg))
}

fn branch_cond_uses_reg(cond: &MachineBranchCond, reg: MachineReg) -> bool {
    match cond {
        MachineBranchCond::Value(v) => value_is_reg(v, reg),
        MachineBranchCond::IntCompare { lhs, rhs, .. } => {
            value_is_reg(lhs, reg) || value_is_reg(rhs, reg)
        }
        MachineBranchCond::TestBits { src, mask, .. } => {
            value_is_reg(src, reg) || value_is_reg(mask, reg)
        }
    }
}

// --- Move rewrite support ---

pub(super) fn rewrite_move_storage_type(
    dst: MachineReg,
    src: MachineValue,
    ty: MachineStorageType,
    config: BackendConfig,
) -> Option<MachineStorageType> {
    if is_fp_reg(dst, config) != ty.is_fp() {
        return None;
    }
    move_rewrite_supported(dst, src, config).then_some(ty)
}

fn move_rewrite_supported(dst: MachineReg, src: MachineValue, config: BackendConfig) -> bool {
    match src {
        MachineValue::Reg(src_reg) => reg_move_rewrite_supported(dst, src_reg, config),
        MachineValue::ReservedReg(_) => false,
        MachineValue::Imm64(_) => is_gp_reg(dst, config),
    }
}

fn reg_move_rewrite_supported(dst: MachineReg, src: MachineReg, config: BackendConfig) -> bool {
    let dst_is_fp = is_fp_reg(dst, config);
    let src_is_fp = is_fp_reg(src, config);
    !dst_is_fp || src_is_fp
}

// --- Memory overlap ---

pub(super) fn addrs_overlap(
    lhs_addr: MachineAddr,
    lhs_width: MachineMemWidth,
    rhs_addr: MachineAddr,
    rhs_width: MachineMemWidth,
) -> bool {
    if lhs_addr.base != rhs_addr.base {
        return false;
    }

    let lhs_start = i64::from(lhs_addr.offset);
    let lhs_end = lhs_start + i64::from(lhs_width.bytes());
    let rhs_start = i64::from(rhs_addr.offset);
    let rhs_end = rhs_start + i64::from(rhs_width.bytes());

    lhs_start < rhs_end && rhs_start < lhs_end
}

// --- Tracker invalidation ---

pub(super) fn kill_tracked_stores_by_reg(
    tracked: &mut collections::Vec<super::TrackedStore>,
    reg: MachineReg,
) {
    tracked.retain(|entry| entry.addr.base != reg && !value_is_reg(&entry.src, reg));
}

pub(super) fn kill_tracked_loads_by_reg(
    tracked: &mut collections::Vec<super::TrackedLoad>,
    reg: MachineReg,
) {
    tracked.retain(|entry| entry.addr.base != reg && entry.reg != reg);
}
