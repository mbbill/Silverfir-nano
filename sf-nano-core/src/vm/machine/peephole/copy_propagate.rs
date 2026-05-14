//! Copy propagation pass.
//!
//! Tracks linear-value register aliases within a block and rewrites later uses
//! to the original source register. Cached-local and fixed-register writes are
//! preserved, but their sources are still canonicalized.
//!
//! Also folds single-use `move rX <- Imm64(C)` into consumer operands as
//! inline immediates.

use crate::collections;

use crate::vm::backend::BackendConfig;
use crate::vm::machine::machine_ir::{
    self, MachineAddr, MachineArgSrc, MachineBlock, MachineBranchCond, MachineCallArgs,
    MachineCallLaneArg, MachineCallResults, MachineCallTarget, MachineConvertOp, MachineEdge,
    MachineInst, MachineInstKind, MachineMemWidth, MachineReg, MachineRegOwner, MachineResultDst,
    MachineResultSrc, MachineReturnValue, MachineTerminator, MachineValue,
};
use crate::vm::machine::ownership::DynamicOwnershipTracker;

use super::helpers::{
    count_value_uses, for_each_defined_reg, inst_defines, reg_live_after, terminator_uses_reg,
};

/// Reusable scratch buffers for copy_propagate to avoid per-block allocation.
pub(super) struct CopyPropagateScratch {
    aliases: collections::Vec<Option<MachineReg>>,
    float_aliases: collections::Vec<Option<MachineReg>>,
    ownership: DynamicOwnershipTracker,
}

impl CopyPropagateScratch {
    pub(super) fn new(reg_count: usize) -> Self {
        Self {
            aliases: collections::vec![None; reg_count],
            float_aliases: collections::vec![None; reg_count],
            ownership: DynamicOwnershipTracker::new(reg_count),
        }
    }

    fn clear(&mut self) {
        for a in &mut self.aliases {
            *a = None;
        }
        for a in &mut self.float_aliases {
            *a = None;
        }
    }
}

pub(super) fn copy_propagate(
    block: &mut MachineBlock,
    config: BackendConfig,
    scratch: &mut CopyPropagateScratch,
) {
    scratch.clear();
    let aliases = &mut scratch.aliases;
    let float_aliases = &mut scratch.float_aliases;
    let ownership = &mut scratch.ownership;
    ownership.reset_for_block(block, config);

    let len = block.ops.len();
    let mut read = 0usize;
    let mut write = 0usize;

    while read < len {
        let mut inst = block.ops[read].clone();

        rewrite_sources(&mut inst.kind, aliases);
        rewrite_float_alias_sources(&mut inst.kind, float_aliases);

        if matches!(
            inst.kind,
            MachineInstKind::CallRuntime(_)
                | MachineInstKind::RefFunc { .. }
                | MachineInstKind::RefAsNonNull { .. }
                | MachineInstKind::RefEq { .. }
                | MachineInstKind::RefI31 { .. }
                | MachineInstKind::I31GetS { .. }
                | MachineInstKind::I31GetU { .. }
                | MachineInstKind::AnyConvertExtern { .. }
                | MachineInstKind::ExternConvertAny { .. }
                | MachineInstKind::RefTest { .. }
                | MachineInstKind::RefCast { .. }
                | MachineInstKind::StructNew { .. }
                | MachineInstKind::StructNewDefault { .. }
                | MachineInstKind::StructGet { .. }
                | MachineInstKind::StructSet { .. }
                | MachineInstKind::ArrayNew { .. }
                | MachineInstKind::ArrayNewDefault { .. }
                | MachineInstKind::ArrayNewFixed { .. }
                | MachineInstKind::ArrayNewData { .. }
                | MachineInstKind::ArrayNewElem { .. }
                | MachineInstKind::ArrayGet { .. }
                | MachineInstKind::ArraySet { .. }
                | MachineInstKind::ArrayFill { .. }
                | MachineInstKind::ArrayCopy { .. }
                | MachineInstKind::ArrayInitData { .. }
                | MachineInstKind::ArrayInitElem { .. }
                | MachineInstKind::ArrayLen { .. }
                | MachineInstKind::EhAllocExnRef { .. }
        ) {
            clear_aliases(aliases);
            clear_aliases(float_aliases);
            block.ops[write] = inst;
            read += 1;
            write += 1;
            continue;
        }

        for_each_defined_reg(&inst.kind, |dst| {
            kill_alias(aliases, dst);
            kill_alias(float_aliases, dst);
        });

        match &inst.kind {
            MachineInstKind::Move {
                owner: MachineRegOwner::LinearValue,
                ty: _,
                dst,
                src: MachineValue::Reg(src),
            } => {
                if *dst == *src {
                    read += 1;
                    continue;
                }
                // Only linear-value moves are safe to elide here. Cached-local
                // or fixed-register moves often materialize snapshots, so they
                // must remain explicit.
                if ownership.is_linear_value_reg(*src, config)
                    && machine_ir::same_reg_bank(*dst, *src, config)
                    && can_elide_reg_move(&block.ops, &block.terminator, read, *dst, *src)
                {
                    aliases[dst.0 as usize] = Some(*src);
                    read += 1;
                    continue;
                }
                if !machine_ir::is_fp_reg(*dst, config) && machine_ir::is_fp_reg(*src, config) {
                    float_aliases[dst.0 as usize] = Some(*src);
                }
            }
            _ => {}
        }

        ownership.apply_inst(&inst.kind, config);
        block.ops[write] = inst;
        read += 1;
        write += 1;
    }

    rewrite_terminator_sources(&mut block.terminator, aliases);
    rewrite_float_alias_terminator_sources(&mut block.terminator, float_aliases);
    block.ops.truncate(write);
}

fn can_elide_reg_move(
    ops: &[MachineInst],
    terminator: &MachineTerminator,
    start_idx: usize,
    dst: MachineReg,
    src: MachineReg,
) -> bool {
    let mut source_stable = true;

    for (later_index, inst) in ops[start_idx + 1..].iter().enumerate() {
        if count_value_uses(&inst.kind, dst) > 0 && !source_stable {
            return false;
        }
        if inst_defines(&inst.kind, dst) {
            return true;
        }
        if matches!(
            inst.kind,
            MachineInstKind::CallRuntime(_)
                | MachineInstKind::RefFunc { .. }
                | MachineInstKind::RefAsNonNull { .. }
                | MachineInstKind::RefEq { .. }
                | MachineInstKind::RefI31 { .. }
                | MachineInstKind::I31GetS { .. }
                | MachineInstKind::I31GetU { .. }
                | MachineInstKind::AnyConvertExtern { .. }
                | MachineInstKind::ExternConvertAny { .. }
                | MachineInstKind::RefTest { .. }
                | MachineInstKind::RefCast { .. }
                | MachineInstKind::StructNew { .. }
                | MachineInstKind::StructNewDefault { .. }
                | MachineInstKind::StructGet { .. }
                | MachineInstKind::StructSet { .. }
                | MachineInstKind::ArrayNew { .. }
                | MachineInstKind::ArrayNewDefault { .. }
                | MachineInstKind::ArrayNewFixed { .. }
                | MachineInstKind::ArrayNewData { .. }
                | MachineInstKind::ArrayNewElem { .. }
                | MachineInstKind::ArrayGet { .. }
                | MachineInstKind::ArraySet { .. }
                | MachineInstKind::ArrayFill { .. }
                | MachineInstKind::ArrayCopy { .. }
                | MachineInstKind::ArrayInitData { .. }
                | MachineInstKind::ArrayInitElem { .. }
                | MachineInstKind::ArrayLen { .. }
                | MachineInstKind::EhAllocExnRef { .. }
        ) {
            // copy_propagate clears aliases at helper calls, so a move can only
            // disappear here if its destination is dead after the barrier.
            let remaining = &ops[start_idx + 1 + later_index + 1..];
            return !reg_live_after(remaining, terminator, dst);
        }
        if inst_defines(&inst.kind, src) {
            source_stable = false;
        }
    }

    if terminator_call_args_use_reg(terminator, dst) {
        return false;
    }
    source_stable || !terminator_uses_reg(terminator, dst)
}

fn terminator_call_args_use_reg(terminator: &MachineTerminator, reg: MachineReg) -> bool {
    match terminator {
        MachineTerminator::Call { args, .. } | MachineTerminator::TailCall { args, .. } => {
            call_args_use_reg(args, reg)
        }
        _ => false,
    }
}

fn call_args_use_reg(args: &MachineCallArgs, reg: MachineReg) -> bool {
    args.lane_args.iter().any(|arg| match arg {
        MachineCallLaneArg::Gp { src, .. } | MachineCallLaneArg::Fp { src, .. } => {
            arg_src_uses_reg(src, reg)
        }
        MachineCallLaneArg::GpPair { src, .. } => {
            arg_src_uses_reg(&src.lo, reg) || arg_src_uses_reg(&src.hi, reg)
        }
    })
}

fn arg_src_uses_reg(src: &MachineArgSrc, reg: MachineReg) -> bool {
    matches!(src, MachineArgSrc::Reg(src_reg) if *src_reg == reg)
}

// --- Alias rewriting ---

fn rewrite_sources(kind: &mut MachineInstKind, aliases: &[Option<MachineReg>]) {
    match kind {
        MachineInstKind::Move { src, .. }
        | MachineInstKind::IntUnary { src, .. }
        | MachineInstKind::FloatUnary { src, .. }
        | MachineInstKind::Convert { src, .. }
        | MachineInstKind::ConvertFloatToI64Pair { src, .. }
        | MachineInstKind::ReinterpretF64ToI64Pair { src, .. } => rewrite_value(src, aliases),
        #[cfg(sf_has_simd)]
        MachineInstKind::V128FromRaw { raw: src, .. }
        | MachineInstKind::V128ToRaw { src, .. }
        | MachineInstKind::SimdUnary { src, .. }
        | MachineInstKind::SimdExtractLane { src, .. } => rewrite_value(src, aliases),
        MachineInstKind::ConvertI64PairToFloat { src_lo, src_hi, .. }
        | MachineInstKind::ReinterpretI64PairToF64 { src_lo, src_hi, .. } => {
            rewrite_value(src_lo, aliases);
            rewrite_value(src_hi, aliases);
        }
        MachineInstKind::FloatConst { .. } => {}
        #[cfg(sf_has_simd)]
        MachineInstKind::V128Const { .. } => {}
        MachineInstKind::Load { addr, .. } => rewrite_addr(addr, aliases),
        MachineInstKind::Store { addr, src, .. } => {
            rewrite_addr(addr, aliases);
            rewrite_value(src, aliases);
        }
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdLoad { addr, .. } => rewrite_addr(addr, aliases),
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdStore { addr, src, .. } => {
            rewrite_addr(addr, aliases);
            rewrite_value(src, aliases);
        }
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdLoadLane { addr, vector, .. } => {
            rewrite_addr(addr, aliases);
            rewrite_value(vector, aliases);
        }
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdStoreLane { addr, vector, .. } => {
            rewrite_addr(addr, aliases);
            rewrite_value(vector, aliases);
        }
        MachineInstKind::IndexedLoad { base, index, .. } => {
            *base = resolve_alias(*base, aliases);
            *index = resolve_alias(*index, aliases);
        }
        MachineInstKind::IndexedStore {
            base, index, src, ..
        } => {
            *base = resolve_alias(*base, aliases);
            *index = resolve_alias(*index, aliases);
            rewrite_value(src, aliases);
        }
        MachineInstKind::IntBinary { lhs, rhs, .. }
        | MachineInstKind::IntCompare { lhs, rhs, .. }
        | MachineInstKind::FloatBinary { lhs, rhs, .. }
        | MachineInstKind::FloatCompare { lhs, rhs, .. } => {
            rewrite_value(lhs, aliases);
            rewrite_value(rhs, aliases);
        }
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdBinary { lhs, rhs, .. } => {
            rewrite_value(lhs, aliases);
            rewrite_value(rhs, aliases);
        }
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdTernary { a, b, c, .. } => {
            rewrite_value(a, aliases);
            rewrite_value(b, aliases);
            rewrite_value(c, aliases);
        }
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdShift { vector, shift, .. } => {
            rewrite_value(vector, aliases);
            rewrite_value(shift, aliases);
        }
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdReplaceLane { vector, scalar, .. } => {
            rewrite_value(vector, aliases);
            rewrite_value(scalar, aliases);
        }
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdShuffle { lhs, rhs, .. } => {
            rewrite_value(lhs, aliases);
            rewrite_value(rhs, aliases);
        }
        MachineInstKind::Int64PairBinary {
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        } => {
            rewrite_value(lhs_lo, aliases);
            rewrite_value(lhs_hi, aliases);
            rewrite_value(rhs_lo, aliases);
            rewrite_value(rhs_hi, aliases);
        }
        MachineInstKind::Int64PairDivRem {
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        } => {
            rewrite_value(lhs_lo, aliases);
            rewrite_value(lhs_hi, aliases);
            rewrite_value(rhs_lo, aliases);
            rewrite_value(rhs_hi, aliases);
        }
        MachineInstKind::Int64PairUnary { src_lo, src_hi, .. } => {
            rewrite_value(src_lo, aliases);
            rewrite_value(src_hi, aliases);
        }
        MachineInstKind::Int64MulFromSignExt32 { lhs, rhs, .. } => {
            rewrite_value(lhs, aliases);
            rewrite_value(rhs, aliases);
        }
        MachineInstKind::Int64PairShift {
            lhs_lo,
            lhs_hi,
            rhs,
            ..
        } => {
            rewrite_value(lhs_lo, aliases);
            rewrite_value(lhs_hi, aliases);
            rewrite_value(rhs, aliases);
        }
        MachineInstKind::Int64PairCompare {
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        } => {
            rewrite_value(lhs_lo, aliases);
            rewrite_value(lhs_hi, aliases);
            rewrite_value(rhs_lo, aliases);
            rewrite_value(rhs_hi, aliases);
        }
        MachineInstKind::Select {
            on_true,
            on_false,
            cond,
            ..
        } => {
            rewrite_value(on_true, aliases);
            rewrite_value(on_false, aliases);
            rewrite_value(cond, aliases);
        }
        MachineInstKind::BitfieldExtractU { src, .. } => {
            *src = resolve_alias(*src, aliases);
        }
        MachineInstKind::IntBinaryShifted { lhs, rhs, .. } => {
            *lhs = resolve_alias(*lhs, aliases);
            *rhs = resolve_alias(*rhs, aliases);
        }
        MachineInstKind::TestBits { src, mask, .. } => {
            *src = resolve_alias(*src, aliases);
            rewrite_value(mask, aliases);
        }
        MachineInstKind::TrapIf { cond, .. } => rewrite_branch_cond(cond, aliases),
        MachineInstKind::CallRuntime(_)
        | MachineInstKind::EhThrow { .. }
        | MachineInstKind::EhThrowRef { .. }
        | MachineInstKind::EhAllocExnRef { .. } => {}
        MachineInstKind::RefFunc { .. } => {}
        MachineInstKind::StructNew { fields, .. } => {
            for (value_lo, value_hi) in fields.iter_mut() {
                rewrite_value(value_lo, aliases);
                if let Some(value_hi) = value_hi {
                    rewrite_value(value_hi, aliases);
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
        | MachineInstKind::ArrayLen { src, .. } => {
            rewrite_value(src, aliases);
        }
        MachineInstKind::ArrayNew {
            init_lo,
            init_hi,
            length,
            ..
        } => {
            rewrite_value(init_lo, aliases);
            if let Some(init_hi) = init_hi {
                rewrite_value(init_hi, aliases);
            }
            rewrite_value(length, aliases);
        }
        MachineInstKind::ArrayNewFixed { elements, .. } => {
            for (value_lo, value_hi) in elements {
                rewrite_value(value_lo, aliases);
                if let Some(value_hi) = value_hi {
                    rewrite_value(value_hi, aliases);
                }
            }
        }
        MachineInstKind::ArrayNewData { src, len, .. }
        | MachineInstKind::ArrayNewElem { src, len, .. } => {
            rewrite_value(src, aliases);
            rewrite_value(len, aliases);
        }
        MachineInstKind::RefEq { lhs, rhs, .. } => {
            rewrite_value(lhs, aliases);
            rewrite_value(rhs, aliases);
        }
        MachineInstKind::ArrayGet { ref_src, index, .. } => {
            rewrite_value(ref_src, aliases);
            rewrite_value(index, aliases);
        }
        MachineInstKind::ArraySet {
            ref_src,
            index,
            value_lo,
            value_hi,
            ..
        } => {
            rewrite_value(ref_src, aliases);
            rewrite_value(index, aliases);
            rewrite_value(value_lo, aliases);
            if let Some(value_hi) = value_hi {
                rewrite_value(value_hi, aliases);
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
            rewrite_value(ref_src, aliases);
            rewrite_value(index, aliases);
            rewrite_value(value_lo, aliases);
            if let Some(value_hi) = value_hi {
                rewrite_value(value_hi, aliases);
            }
            rewrite_value(len, aliases);
        }
        MachineInstKind::ArrayCopy {
            dst_ref,
            dst_index,
            src_ref,
            src_index,
            len,
            ..
        } => {
            rewrite_value(dst_ref, aliases);
            rewrite_value(dst_index, aliases);
            rewrite_value(src_ref, aliases);
            rewrite_value(src_index, aliases);
            rewrite_value(len, aliases);
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
            rewrite_value(ref_src, aliases);
            rewrite_value(dst_index, aliases);
            rewrite_value(src_index, aliases);
            rewrite_value(len, aliases);
        }
        MachineInstKind::StructSet {
            ref_src,
            value_lo,
            value_hi,
            ..
        } => {
            rewrite_value(ref_src, aliases);
            rewrite_value(value_lo, aliases);
            if let Some(value_hi) = value_hi {
                rewrite_value(value_hi, aliases);
            }
        }
        MachineInstKind::MemoryGrow { delta, .. } => {
            rewrite_value(delta, aliases);
        }
        MachineInstKind::MemoryFill { dest, val, len, .. }
        | MachineInstKind::TableFill {
            start: dest,
            val,
            len,
            ..
        } => {
            rewrite_value(dest, aliases);
            rewrite_value(val, aliases);
            rewrite_value(len, aliases);
        }
        MachineInstKind::MemoryCopy { dest, src, len, .. }
        | MachineInstKind::MemoryInit { dest, src, len, .. }
        | MachineInstKind::TableCopy { dest, src, len, .. }
        | MachineInstKind::TableInit { dest, src, len, .. } => {
            rewrite_value(dest, aliases);
            rewrite_value(src, aliases);
            rewrite_value(len, aliases);
        }
        MachineInstKind::TableGrow {
            init_val, delta, ..
        } => {
            rewrite_value(init_val, aliases);
            rewrite_value(delta, aliases);
        }
        MachineInstKind::DataDrop { .. } | MachineInstKind::ElemDrop { .. } => {}
    }
}

fn rewrite_terminator_sources(term: &mut MachineTerminator, aliases: &[Option<MachineReg>]) {
    match term {
        MachineTerminator::Jump(edge) => rewrite_edge(edge, aliases),
        MachineTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => {
            rewrite_branch_cond(cond, aliases);
            rewrite_edge(then_edge, aliases);
            rewrite_edge(else_edge, aliases);
        }
        MachineTerminator::JumpTable { index, entries } => {
            rewrite_value(index, aliases);
            for edge in entries {
                rewrite_edge(edge, aliases);
            }
        }
        MachineTerminator::Call {
            target,
            args,
            results,
            success,
            ..
        } => {
            rewrite_call_target(target, aliases);
            let _ = args;
            rewrite_call_success_sources(success, results, aliases);
        }
        MachineTerminator::TailCall { target, args } => {
            rewrite_call_target(target, aliases);
            let _ = args;
        }
        MachineTerminator::Return | MachineTerminator::Trap { .. } => {}
        MachineTerminator::ReturnScalar { value } => rewrite_return_value(value, aliases),
    }
}

fn rewrite_call_target(target: &mut MachineCallTarget, aliases: &[Option<MachineReg>]) {
    if let MachineCallTarget::Indirect {
        callee_target,
        callee_entry,
    } = target
    {
        *callee_target = resolve_alias(*callee_target, aliases);
        *callee_entry = resolve_alias(*callee_entry, aliases);
    }
}

fn rewrite_call_success_sources(
    success: &mut MachineEdge,
    results: &MachineCallResults,
    aliases: &[Option<MachineReg>],
) {
    for arg in &mut success.args {
        if let MachineValue::Reg(reg) = arg {
            if !call_results_define_reg(results, *reg) {
                *reg = resolve_alias(*reg, aliases);
            }
        }
    }
}

fn rewrite_return_value(value: &mut MachineReturnValue, aliases: &[Option<MachineReg>]) {
    match value {
        MachineReturnValue::ScalarGp { src, .. } | MachineReturnValue::ScalarFp { src, .. } => {
            rewrite_result_src(src, aliases);
        }
        MachineReturnValue::ScalarGpPair { lo, hi } => {
            rewrite_result_src(lo, aliases);
            rewrite_result_src(hi, aliases);
        }
    }
}

fn rewrite_result_src(src: &mut MachineResultSrc, aliases: &[Option<MachineReg>]) {
    if let MachineResultSrc::Reg(reg) = src {
        *reg = resolve_alias(*reg, aliases);
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

fn rewrite_float_alias_terminator_sources(
    term: &mut MachineTerminator,
    aliases: &[Option<MachineReg>],
) {
    match term {
        MachineTerminator::Branch { cond, .. } => rewrite_float_alias_branch_cond(cond, aliases),
        MachineTerminator::Jump(_)
        | MachineTerminator::JumpTable { .. }
        | MachineTerminator::Call { .. }
        | MachineTerminator::TailCall { .. }
        | MachineTerminator::Return
        | MachineTerminator::Trap { .. } => {}
        MachineTerminator::ReturnScalar { value } => rewrite_return_value(value, aliases),
    }
}

fn rewrite_float_alias_sources(kind: &mut MachineInstKind, aliases: &[Option<MachineReg>]) {
    match kind {
        MachineInstKind::FloatUnary { src, .. } => rewrite_float_alias_value(src, aliases),
        MachineInstKind::FloatBinary { lhs, rhs, .. }
        | MachineInstKind::FloatCompare { lhs, rhs, .. } => {
            rewrite_float_alias_value(lhs, aliases);
            rewrite_float_alias_value(rhs, aliases);
        }
        MachineInstKind::Store { width, src, .. }
            if matches!(width, MachineMemWidth::U32 | MachineMemWidth::U64) =>
        {
            rewrite_float_alias_value(src, aliases);
        }
        MachineInstKind::TrapIf { cond, .. } => rewrite_float_alias_branch_cond(cond, aliases),
        MachineInstKind::Convert { op, src, .. } if convert_src_accepts_fp(*op) => {
            rewrite_float_alias_value(src, aliases);
        }
        _ => {}
    }
}

fn rewrite_branch_cond(cond: &mut MachineBranchCond, aliases: &[Option<MachineReg>]) {
    match cond {
        MachineBranchCond::Value(value) => rewrite_value(value, aliases),
        MachineBranchCond::IntCompare { lhs, rhs, .. } => {
            rewrite_value(lhs, aliases);
            rewrite_value(rhs, aliases);
        }
        MachineBranchCond::TestBits { src, mask, .. } => {
            rewrite_value(src, aliases);
            rewrite_value(mask, aliases);
        }
    }
}

fn rewrite_float_alias_branch_cond(cond: &mut MachineBranchCond, aliases: &[Option<MachineReg>]) {
    let _ = (cond, aliases);
}

fn rewrite_edge(edge: &mut MachineEdge, aliases: &[Option<MachineReg>]) {
    for arg in &mut edge.args {
        rewrite_value(arg, aliases);
    }
}

fn rewrite_addr(addr: &mut MachineAddr, aliases: &[Option<MachineReg>]) {
    addr.base = resolve_alias(addr.base, aliases);
}

fn rewrite_value(value: &mut MachineValue, aliases: &[Option<MachineReg>]) {
    if let MachineValue::Reg(reg) = value {
        *reg = resolve_alias(*reg, aliases);
    }
}

fn rewrite_float_alias_value(value: &mut MachineValue, aliases: &[Option<MachineReg>]) {
    let MachineValue::Reg(reg) = value else {
        return;
    };
    if let Some(Some(src)) = aliases.get(reg.0 as usize) {
        *value = MachineValue::Reg(*src);
    }
}

fn resolve_alias(reg: MachineReg, aliases: &[Option<MachineReg>]) -> MachineReg {
    let mut resolved = reg;
    while let Some(Some(next)) = aliases.get(resolved.0 as usize) {
        if *next == resolved {
            break;
        }
        resolved = *next;
    }
    resolved
}

fn convert_src_accepts_fp(op: MachineConvertOp) -> bool {
    matches!(
        op,
        MachineConvertOp::I32TruncF32S
            | MachineConvertOp::I32TruncF32U
            | MachineConvertOp::I32TruncF64S
            | MachineConvertOp::I32TruncF64U
            | MachineConvertOp::I64TruncF32S
            | MachineConvertOp::I64TruncF32U
            | MachineConvertOp::I64TruncF64S
            | MachineConvertOp::I64TruncF64U
            | MachineConvertOp::I32TruncSatF32S
            | MachineConvertOp::I32TruncSatF32U
            | MachineConvertOp::I32TruncSatF64S
            | MachineConvertOp::I32TruncSatF64U
            | MachineConvertOp::I64TruncSatF32S
            | MachineConvertOp::I64TruncSatF32U
            | MachineConvertOp::I64TruncSatF64S
            | MachineConvertOp::I64TruncSatF64U
            | MachineConvertOp::F32DemoteF64
            | MachineConvertOp::F64PromoteF32
            | MachineConvertOp::I32ReinterpretF32
            | MachineConvertOp::I64ReinterpretF64
    )
}

fn kill_alias(aliases: &mut [Option<MachineReg>], reg: MachineReg) {
    if let Some(slot) = aliases.get_mut(reg.0 as usize) {
        *slot = None;
    }
    for alias in aliases.iter_mut() {
        if *alias == Some(reg) {
            *alias = None;
        }
    }
}

fn clear_aliases(aliases: &mut [Option<MachineReg>]) {
    for alias in aliases.iter_mut() {
        *alias = None;
    }
}
