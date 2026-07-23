//! Shared helper functions for peephole optimization passes.

use crate::collections;

use crate::vm::backend::BackendConfig;
use crate::vm::machine::machine_ir::{
    is_fp_reg, is_gp_reg, MachineAddr, MachineArgSrc, MachineBranchCond, MachineCallArgs,
    MachineCallLaneArg, MachineCallResults, MachineCallTarget, MachineEdge, MachineInst,
    MachineInstKind, MachineIntUnaryOp, MachineMemWidth, MachineReg, MachineResultDst,
    MachineResultSrc, MachineReturnValue, MachineStorageType, MachineTerminator, MachineValue,
    MACHINE_CTX_REG, MACHINE_FP_REG,
};

// --- Instruction analysis ---

/// Check if `kind` defines (writes to) `reg`.
pub(super) fn inst_defines(kind: &MachineInstKind, reg: MachineReg) -> bool {
    let mut defines = false;
    kind.for_each_defined_reg(|def| {
        if def == reg {
            defines = true;
        }
    });
    defines
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
pub(crate) fn reg_live_after(
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
pub(crate) fn terminator_uses_reg(term: &MachineTerminator, reg: MachineReg) -> bool {
    let mut found = false;
    visit_terminator_source_regs(term, |source| {
        found |= source == reg;
    });
    found
}

/// Visit every register read by a terminator.
///
/// Call result destinations are definitions, not reads, even when they also
/// appear in the success edge's post-call arguments.
pub(crate) fn visit_terminator_source_regs(
    term: &MachineTerminator,
    mut f: impl FnMut(MachineReg),
) {
    match term {
        MachineTerminator::Jump(edge) => visit_edge_source_regs(edge, &mut f),
        MachineTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => {
            visit_branch_cond_source_regs(cond, &mut f);
            visit_edge_source_regs(then_edge, &mut f);
            visit_edge_source_regs(else_edge, &mut f);
        }
        MachineTerminator::JumpTable { index, entries } => {
            visit_value_source_reg(index, &mut f);
            for edge in entries {
                visit_edge_source_regs(edge, &mut f);
            }
        }
        MachineTerminator::Call {
            target,
            args,
            results,
            success,
            ..
        } => {
            visit_call_target_source_regs(target, &mut f);
            visit_call_arg_source_regs(args, &mut f);
            for value in &success.args {
                if let MachineValue::Reg(reg) = *value {
                    if !call_results_define_reg(results, reg) {
                        f(reg);
                    }
                }
            }
        }
        MachineTerminator::TailCall { target, args } => {
            visit_call_target_source_regs(target, &mut f);
            visit_call_arg_source_regs(args, &mut f);
        }
        MachineTerminator::Return | MachineTerminator::Trap { .. } => {}
        MachineTerminator::ReturnScalar { value } => {
            visit_return_value_source_regs(value, &mut f);
        }
    }
}

fn visit_call_target_source_regs(target: &MachineCallTarget, f: &mut impl FnMut(MachineReg)) {
    match target {
        MachineCallTarget::Direct(_) => {}
        MachineCallTarget::Indirect {
            callee_target,
            callee_entry,
        } => {
            f(*callee_target);
            f(*callee_entry);
        }
    }
}

fn visit_call_arg_source_regs(args: &MachineCallArgs, f: &mut impl FnMut(MachineReg)) {
    for arg in &args.lane_args {
        match arg {
            MachineCallLaneArg::Gp { src, .. } | MachineCallLaneArg::Fp { src, .. } => {
                visit_arg_source_reg(src, f);
            }
            MachineCallLaneArg::GpPair { src, .. } => {
                visit_arg_source_reg(&src.lo, f);
                visit_arg_source_reg(&src.hi, f);
            }
        }
    }
}

fn visit_arg_source_reg(src: &MachineArgSrc, f: &mut impl FnMut(MachineReg)) {
    if let MachineArgSrc::Reg(reg) = *src {
        f(reg);
    }
}

fn visit_return_value_source_regs(value: &MachineReturnValue, f: &mut impl FnMut(MachineReg)) {
    match value {
        MachineReturnValue::ScalarGp { src, .. } | MachineReturnValue::ScalarFp { src, .. } => {
            visit_result_source_reg(src, f);
        }
        MachineReturnValue::ScalarGpPair { lo, hi } => {
            visit_result_source_reg(lo, f);
            visit_result_source_reg(hi, f);
        }
    }
}

fn visit_result_source_reg(src: &MachineResultSrc, f: &mut impl FnMut(MachineReg)) {
    if let MachineResultSrc::Reg(reg) = *src {
        f(reg);
    }
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

fn visit_edge_source_regs(edge: &MachineEdge, f: &mut impl FnMut(MachineReg)) {
    for value in &edge.args {
        visit_value_source_reg(value, f);
    }
}

fn visit_branch_cond_source_regs(cond: &MachineBranchCond, f: &mut impl FnMut(MachineReg)) {
    visit_branch_cond_values(cond, |value| visit_value_source_reg(value, f));
}

fn visit_value_source_reg(value: &MachineValue, f: &mut impl FnMut(MachineReg)) {
    if let MachineValue::Reg(reg) = *value {
        f(reg);
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

/// True if `base` is a runtime-owned address space that a Wasm-visible store
/// (linear memory, table) can never write: the frame and the runtime context.
fn is_runtime_owned_base(base: MachineReg) -> bool {
    base == MACHINE_FP_REG || base == MACHINE_CTX_REG
}

/// Conservative may-alias between a tracked entry and a plain `Store`.
///
/// Same base register: precise range overlap. Different base registers: two
/// Wasm-side bases (e.g. the linear-memory base and a computed pointer into
/// it) can address the same bytes, so they conservatively alias; a
/// runtime-owned base on either side cannot alias the other.
pub(super) fn store_may_alias(
    entry_addr: MachineAddr,
    entry_width: MachineMemWidth,
    store_addr: MachineAddr,
    store_width: MachineMemWidth,
) -> bool {
    if entry_addr.base == store_addr.base {
        return addrs_overlap(entry_addr, entry_width, store_addr, store_width);
    }
    !is_runtime_owned_base(entry_addr.base) && !is_runtime_owned_base(store_addr.base)
}

/// Conservative may-alias between a tracked entry and a store whose target
/// range is unknown (indexed store, bulk-memory op, table write). Only
/// runtime-owned bases are provably out of reach.
pub(super) fn unknown_store_may_alias(entry_base: MachineReg) -> bool {
    !is_runtime_owned_base(entry_base)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::machine::machine_ir::MachineBlockId;

    #[test]
    fn visits_call_sources_once_without_treating_results_as_reads() {
        let result = MachineReg(5);
        let survivor = MachineReg(6);
        let term = MachineTerminator::Call {
            target: MachineCallTarget::Indirect {
                callee_target: MachineReg(2),
                callee_entry: MachineReg(3),
            },
            frame_delta: 0,
            args: MachineCallArgs {
                frame_params: Default::default(),
                lane_args: collections::vec![MachineCallLaneArg::Gp {
                    param_index: 0,
                    lane: 0,
                    src: MachineArgSrc::Reg(MachineReg(4)),
                    ty: MachineStorageType::GpWord,
                }],
            },
            results: MachineCallResults::ScalarGp {
                dst: MachineResultDst::Reg(result),
                ty: MachineStorageType::GpWord,
            },
            success: MachineEdge {
                target: MachineBlockId(1),
                args: collections::vec![MachineValue::Reg(result), MachineValue::Reg(survivor)],
            },
        };

        let mut sources = collections::Vec::new();
        visit_terminator_source_regs(&term, |reg| sources.push(reg));

        assert_eq!(
            sources,
            collections::vec![MachineReg(2), MachineReg(3), MachineReg(4), survivor]
        );
        assert!(!terminator_uses_reg(&term, result));
        assert!(terminator_uses_reg(&term, survivor));
    }
}
