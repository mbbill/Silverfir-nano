//! MachineIR liveness facts for 32-bit lowering of i64 pairs.
//!
//! This analysis is intentionally MachineIR-level: it reasons about
//! `Int64Pair*` low/high destination registers after 32-bit GP legalization,
//! but before any backend chooses native instruction encodings.

use crate::collections;
use crate::vm::machine::{
    machine_ir::{
        MachineArgSrc, MachineArgSrcPair, MachineBlockId, MachineBranchCond, MachineCallArgs,
        MachineCallLaneArg, MachineCallResults, MachineCallTarget, MachineEdge, MachineFunction,
        MachineInstKind, MachineIntBinaryOp, MachineIntUnaryOp, MachineReg, MachineResultDst,
        MachineResultSrc, MachineReturnValue, MachineTerminator, MachineValue,
    },
    peephole::helpers as mir_helpers,
};

/// Per-instruction facts for i64-pair operations whose high result register is
/// not demanded by any later required computation.
#[derive(Clone, Debug, Default)]
pub(crate) struct Low32DeadHiDefs {
    dead_hi_defs: collections::Vec<collections::Vec<bool>>,
}

impl Low32DeadHiDefs {
    pub(crate) fn compute(function: &MachineFunction, reg_count: usize) -> Self {
        let block_count = function
            .program
            .blocks
            .iter()
            .map(|block| block.id.0 as usize + 1)
            .max()
            .unwrap_or(0);
        let mut demand_in = collections::vec![collections::vec![false; reg_count]; block_count];

        loop {
            let mut changed = false;
            for block in function.program.blocks.iter().rev() {
                let mut demand =
                    Self::terminator_demand(function, &block.terminator, &demand_in, reg_count);
                for inst in block.ops.iter().rev() {
                    Self::transfer_inst_demand(&inst.kind, &mut demand);
                }
                let block_index = block.id.0 as usize;
                if block_index < demand_in.len() && demand_in[block_index] != demand {
                    demand_in[block_index] = demand;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        let mut dead_hi_defs = collections::Vec::new();
        dead_hi_defs.resize_with(block_count, collections::Vec::new);
        for block in &function.program.blocks {
            let block_index = block.id.0 as usize;
            if block_index >= dead_hi_defs.len() {
                continue;
            }
            dead_hi_defs[block_index] = collections::vec![false; block.ops.len()];
            let mut demand =
                Self::terminator_demand(function, &block.terminator, &demand_in, reg_count);
            for (index, inst) in block.ops.iter().enumerate().rev() {
                dead_hi_defs[block_index][index] =
                    Self::inst_low32_can_ignore_hi(&inst.kind, &demand);
                Self::transfer_inst_demand(&inst.kind, &mut demand);
            }
        }

        Self { dead_hi_defs }
    }

    pub(crate) fn is_dead_at(
        &self,
        current_block: Option<MachineBlockId>,
        current_op_index: Option<usize>,
    ) -> bool {
        let Some(block) = current_block else {
            return false;
        };
        let Some(index) = current_op_index else {
            return false;
        };
        self.dead_hi_defs
            .get(block.0 as usize)
            .and_then(|block| block.get(index))
            .copied()
            .unwrap_or(false)
    }

    fn inst_low32_can_ignore_hi(kind: &MachineInstKind, demand_after: &[bool]) -> bool {
        match kind {
            MachineInstKind::Int64PairBinary { op, dst_hi, .. }
                if matches!(
                    op,
                    MachineIntBinaryOp::Add | MachineIntBinaryOp::Sub | MachineIntBinaryOp::And
                ) =>
            {
                !Self::reg_demanded(demand_after, *dst_hi)
            }
            MachineInstKind::Int64PairUnary {
                op: MachineIntUnaryOp::Extend32S,
                dst_hi,
                ..
            } => !Self::reg_demanded(demand_after, *dst_hi),
            MachineInstKind::Int64PairShift {
                op,
                dst_hi,
                rhs: MachineValue::Imm64(count),
                ..
            } if matches!(op, MachineIntBinaryOp::ShrU | MachineIntBinaryOp::ShrS)
                && (1..32).contains(&((*count as u32) & 63)) =>
            {
                !Self::reg_demanded(demand_after, *dst_hi)
            }
            _ => false,
        }
    }

    fn terminator_demand(
        function: &MachineFunction,
        term: &MachineTerminator,
        demand_in: &[collections::Vec<bool>],
        reg_count: usize,
    ) -> collections::Vec<bool> {
        let mut demand = collections::vec![false; reg_count];
        match term {
            MachineTerminator::Jump(edge) => {
                Self::demand_edge(function, edge, demand_in, &mut demand);
            }
            MachineTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            } => {
                Self::demand_branch_cond(cond, &mut demand);
                Self::demand_edge(function, then_edge, demand_in, &mut demand);
                Self::demand_edge(function, else_edge, demand_in, &mut demand);
            }
            MachineTerminator::JumpTable { index, entries } => {
                Self::demand_value(&mut demand, index);
                for edge in entries {
                    Self::demand_edge(function, edge, demand_in, &mut demand);
                }
            }
            MachineTerminator::Call {
                target,
                args,
                results,
                success,
                ..
            } => {
                Self::demand_call_target(target, &mut demand);
                Self::demand_call_args(args, &mut demand);
                Self::demand_call_success_sources(success, results, &mut demand);
            }
            MachineTerminator::TailCall { target, args } => {
                Self::demand_call_target(target, &mut demand);
                Self::demand_call_args(args, &mut demand);
            }
            MachineTerminator::Return | MachineTerminator::Trap { .. } => {}
            MachineTerminator::ReturnScalar { value } => {
                Self::demand_return_value(value, &mut demand);
            }
        }
        demand
    }

    fn demand_edge(
        function: &MachineFunction,
        edge: &MachineEdge,
        demand_in: &[collections::Vec<bool>],
        demand: &mut [bool],
    ) {
        let Some(target) = function.program.blocks.get(edge.target.0 as usize) else {
            return;
        };
        let Some(target_demand) = demand_in.get(edge.target.0 as usize) else {
            return;
        };
        for (arg, param) in edge.args.iter().zip(&target.params) {
            if Self::reg_demanded(target_demand, param.reg) {
                Self::demand_value(demand, arg);
            }
        }
    }

    fn transfer_inst_demand(kind: &MachineInstKind, demand: &mut [bool]) {
        match kind {
            MachineInstKind::Move { dst, src, .. } => {
                let needed = Self::take_reg_demand(demand, *dst);
                if needed {
                    Self::demand_value(demand, src);
                }
            }
            MachineInstKind::Int64PairBinary {
                op,
                dst_lo,
                dst_hi,
                lhs_lo,
                lhs_hi,
                rhs_lo,
                rhs_hi,
            } => {
                let lo_needed = Self::take_reg_demand(demand, *dst_lo);
                let hi_needed = Self::take_reg_demand(demand, *dst_hi);
                match op {
                    MachineIntBinaryOp::Add | MachineIntBinaryOp::Sub => {
                        if lo_needed || hi_needed {
                            Self::demand_value(demand, lhs_lo);
                            Self::demand_value(demand, rhs_lo);
                        }
                        if hi_needed {
                            Self::demand_value(demand, lhs_hi);
                            Self::demand_value(demand, rhs_hi);
                        }
                    }
                    MachineIntBinaryOp::And | MachineIntBinaryOp::Or | MachineIntBinaryOp::Xor => {
                        if lo_needed {
                            Self::demand_value(demand, lhs_lo);
                            Self::demand_value(demand, rhs_lo);
                        }
                        if hi_needed {
                            Self::demand_value(demand, lhs_hi);
                            Self::demand_value(demand, rhs_hi);
                        }
                    }
                    MachineIntBinaryOp::Mul => {
                        if lo_needed || hi_needed {
                            Self::demand_value(demand, lhs_lo);
                            Self::demand_value(demand, rhs_lo);
                        }
                        if hi_needed {
                            Self::demand_value(demand, lhs_hi);
                            Self::demand_value(demand, rhs_hi);
                        }
                    }
                    _ => {
                        if lo_needed || hi_needed {
                            Self::demand_value(demand, lhs_lo);
                            Self::demand_value(demand, lhs_hi);
                            Self::demand_value(demand, rhs_lo);
                            Self::demand_value(demand, rhs_hi);
                        }
                    }
                }
            }
            MachineInstKind::Int64PairUnary {
                op: MachineIntUnaryOp::Extend32S,
                dst_lo,
                dst_hi,
                src_lo,
                ..
            } => {
                let lo_needed = Self::take_reg_demand(demand, *dst_lo);
                let hi_needed = Self::take_reg_demand(demand, *dst_hi);
                if lo_needed || hi_needed {
                    Self::demand_value(demand, src_lo);
                }
            }
            MachineInstKind::Int64MulFromSignExt32 {
                dst_lo,
                dst_hi,
                lhs,
                rhs,
            } => {
                let needed =
                    Self::take_reg_demand(demand, *dst_lo) | Self::take_reg_demand(demand, *dst_hi);
                if needed {
                    Self::demand_value(demand, lhs);
                    Self::demand_value(demand, rhs);
                }
            }
            MachineInstKind::Int64PairShift {
                op,
                dst_lo,
                dst_hi,
                lhs_lo,
                lhs_hi,
                rhs,
            } => {
                let lo_needed = Self::take_reg_demand(demand, *dst_lo);
                let hi_needed = Self::take_reg_demand(demand, *dst_hi);
                if lo_needed || hi_needed {
                    Self::demand_value(demand, rhs);
                }
                match (op, rhs) {
                    (
                        MachineIntBinaryOp::ShrU | MachineIntBinaryOp::ShrS,
                        MachineValue::Imm64(count),
                    ) if (1..32).contains(&((*count as u32) & 63)) => {
                        if lo_needed {
                            Self::demand_value(demand, lhs_lo);
                            Self::demand_value(demand, lhs_hi);
                        }
                        if hi_needed {
                            Self::demand_value(demand, lhs_hi);
                        }
                    }
                    _ => {
                        if lo_needed || hi_needed {
                            Self::demand_value(demand, lhs_lo);
                            Self::demand_value(demand, lhs_hi);
                        }
                    }
                }
            }
            _ => {
                let output_needed = Self::kill_defs_and_report_need(kind, demand);
                if output_needed || !Self::has_register_defs(kind) {
                    mir_helpers::visit_source_values(kind, |value| {
                        Self::demand_value(demand, value);
                    });
                }
            }
        }
    }

    fn kill_defs_and_report_need(kind: &MachineInstKind, demand: &mut [bool]) -> bool {
        let mut needed = false;
        match kind {
            MachineInstKind::FloatConst { dst, .. }
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
            | MachineInstKind::MemoryGrow { dst, .. }
            | MachineInstKind::TableGrow { dst, .. }
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
            | MachineInstKind::ArrayLen { dst, .. } => {
                needed |= Self::take_reg_demand(demand, *dst);
            }
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
            | MachineInstKind::SimdLoadLane { dst, .. } => {
                needed |= Self::take_reg_demand(demand, *dst);
            }
            MachineInstKind::StructGet { dst, dst_hi, .. }
            | MachineInstKind::ArrayGet { dst, dst_hi, .. } => {
                needed |= Self::take_reg_demand(demand, *dst);
                if let Some(dst_hi) = dst_hi {
                    needed |= Self::take_reg_demand(demand, *dst_hi);
                }
            }
            MachineInstKind::Int64PairDivRem { dst_lo, dst_hi, .. }
            | MachineInstKind::ConvertFloatToI64Pair { dst_lo, dst_hi, .. }
            | MachineInstKind::ReinterpretF64ToI64Pair { dst_lo, dst_hi, .. } => {
                needed |= Self::take_reg_demand(demand, *dst_lo);
                needed |= Self::take_reg_demand(demand, *dst_hi);
            }
            MachineInstKind::Int64PairCompare { dst, .. }
            | MachineInstKind::ConvertI64PairToFloat { dst, .. }
            | MachineInstKind::ReinterpretI64PairToF64 { dst, .. } => {
                needed |= Self::take_reg_demand(demand, *dst);
            }
            MachineInstKind::Store { .. }
            | MachineInstKind::IndexedStore { .. }
            | MachineInstKind::TrapIf { .. }
            | MachineInstKind::CallRuntime(_)
            | MachineInstKind::MemoryFill { .. }
            | MachineInstKind::MemoryCopy { .. }
            | MachineInstKind::MemoryInit { .. }
            | MachineInstKind::DataDrop { .. }
            | MachineInstKind::TableFill { .. }
            | MachineInstKind::TableCopy { .. }
            | MachineInstKind::TableInit { .. }
            | MachineInstKind::ElemDrop { .. }
            | MachineInstKind::EhThrow { .. }
            | MachineInstKind::EhThrowRef { .. }
            | MachineInstKind::StructSet { .. }
            | MachineInstKind::ArraySet { .. }
            | MachineInstKind::ArrayFill { .. }
            | MachineInstKind::ArrayCopy { .. }
            | MachineInstKind::ArrayInitData { .. }
            | MachineInstKind::ArrayInitElem { .. } => {}
            #[cfg(sf_has_simd)]
            MachineInstKind::SimdStore { .. } | MachineInstKind::SimdStoreLane { .. } => {}
            MachineInstKind::Move { .. }
            | MachineInstKind::Int64PairBinary { .. }
            | MachineInstKind::Int64PairUnary { .. }
            | MachineInstKind::Int64PairShift { .. }
            | MachineInstKind::Int64MulFromSignExt32 { .. } => {}
        }
        needed
    }

    fn has_register_defs(kind: &MachineInstKind) -> bool {
        match kind {
            MachineInstKind::Store { .. }
            | MachineInstKind::IndexedStore { .. }
            | MachineInstKind::TrapIf { .. }
            | MachineInstKind::CallRuntime(_)
            | MachineInstKind::MemoryFill { .. }
            | MachineInstKind::MemoryCopy { .. }
            | MachineInstKind::MemoryInit { .. }
            | MachineInstKind::DataDrop { .. }
            | MachineInstKind::TableFill { .. }
            | MachineInstKind::TableCopy { .. }
            | MachineInstKind::TableInit { .. }
            | MachineInstKind::ElemDrop { .. }
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
            _ => true,
        }
    }

    fn demand_branch_cond(cond: &MachineBranchCond, demand: &mut [bool]) {
        match cond {
            MachineBranchCond::Value(value) => Self::demand_value(demand, value),
            MachineBranchCond::IntCompare { lhs, rhs, .. } => {
                Self::demand_value(demand, lhs);
                Self::demand_value(demand, rhs);
            }
            MachineBranchCond::TestBits { src, mask, .. } => {
                Self::demand_value(demand, src);
                Self::demand_value(demand, mask);
            }
        }
    }

    fn demand_call_target(target: &MachineCallTarget, demand: &mut [bool]) {
        if let MachineCallTarget::Indirect {
            callee_target,
            callee_entry,
        } = target
        {
            Self::demand_reg(demand, *callee_target);
            Self::demand_reg(demand, *callee_entry);
        }
    }

    fn demand_call_args(args: &MachineCallArgs, demand: &mut [bool]) {
        for arg in &args.lane_args {
            match arg {
                MachineCallLaneArg::Gp { src, .. } | MachineCallLaneArg::Fp { src, .. } => {
                    Self::demand_arg_src(src, demand);
                }
                MachineCallLaneArg::GpPair { src, .. } => {
                    Self::demand_arg_src_pair(src, demand);
                }
            }
        }
    }

    fn demand_arg_src_pair(src: &MachineArgSrcPair, demand: &mut [bool]) {
        Self::demand_arg_src(&src.lo, demand);
        Self::demand_arg_src(&src.hi, demand);
    }

    fn demand_arg_src(src: &MachineArgSrc, demand: &mut [bool]) {
        if let MachineArgSrc::Reg(reg) = src {
            Self::demand_reg(demand, *reg);
        }
    }

    fn demand_return_value(value: &MachineReturnValue, demand: &mut [bool]) {
        match value {
            MachineReturnValue::ScalarGp { src, .. } | MachineReturnValue::ScalarFp { src, .. } => {
                Self::demand_result_src(src, demand);
            }
            MachineReturnValue::ScalarGpPair { lo, hi } => {
                Self::demand_result_src(lo, demand);
                Self::demand_result_src(hi, demand);
            }
        }
    }

    fn demand_result_src(src: &MachineResultSrc, demand: &mut [bool]) {
        if let MachineResultSrc::Reg(reg) = src {
            Self::demand_reg(demand, *reg);
        }
    }

    fn demand_call_success_sources(
        success: &MachineEdge,
        results: &MachineCallResults,
        demand: &mut [bool],
    ) {
        for arg in &success.args {
            if let MachineValue::Reg(reg) = arg {
                if !Self::call_results_define_reg(results, *reg) {
                    Self::demand_reg(demand, *reg);
                }
            }
        }
    }

    fn call_results_define_reg(results: &MachineCallResults, reg: MachineReg) -> bool {
        match results {
            MachineCallResults::None | MachineCallResults::FrameFallback { .. } => false,
            MachineCallResults::ScalarGp { dst, .. } | MachineCallResults::ScalarFp { dst, .. } => {
                Self::result_dst_is_reg(*dst, reg)
            }
            MachineCallResults::ScalarGpPair { lo, hi } => {
                Self::result_dst_is_reg(*lo, reg) || Self::result_dst_is_reg(*hi, reg)
            }
        }
    }

    fn result_dst_is_reg(dst: MachineResultDst, reg: MachineReg) -> bool {
        matches!(dst, MachineResultDst::Reg(dst) if dst == reg)
    }

    fn demand_value(demand: &mut [bool], value: &MachineValue) {
        if let MachineValue::Reg(reg) = value {
            Self::demand_reg(demand, *reg);
        }
    }

    fn demand_reg(demand: &mut [bool], reg: MachineReg) {
        if let Some(slot) = demand.get_mut(reg.0 as usize) {
            *slot = true;
        }
    }

    fn reg_demanded(demand: &[bool], reg: MachineReg) -> bool {
        demand.get(reg.0 as usize).copied().unwrap_or(false)
    }

    fn take_reg_demand(demand: &mut [bool], reg: MachineReg) -> bool {
        let Some(slot) = demand.get_mut(reg.0 as usize) else {
            return false;
        };
        let needed = *slot;
        *slot = false;
        needed
    }
}
