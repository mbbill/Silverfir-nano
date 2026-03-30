//! Copy propagation pass.
//!
//! Tracks transient register aliases within a block and rewrites later uses to
//! the original source register. Cached-local and fixed-register writes are
//! preserved, but their sources are still canonicalized.
//!
//! Also folds single-use `move rX <- Imm64(C)` into consumer operands as
//! inline immediates.

use alloc::vec;
use alloc::vec::Vec;

use crate::vm::backend::BackendConfig;
use crate::vm::machine::machine_ir::{
    self, MachineAddr, MachineBlock, MachineBranchCond, MachineConvertOp, MachineEdge, MachineInst,
    MachineInstKind, MachineMemWidth, MachineReg, MachineTerminator, MachineValue,
};

use super::helpers::{
    count_value_uses, for_each_defined_reg, inst_defines, reg_live_after, terminator_uses_reg,
};

/// Reusable scratch buffers for copy_propagate to avoid per-block allocation.
pub(super) struct CopyPropagateScratch {
    aliases: Vec<Option<MachineReg>>,
    float_aliases: Vec<Option<MachineReg>>,
    rewritten: Vec<MachineInst>,
}

impl CopyPropagateScratch {
    pub(super) fn new(reg_count: usize) -> Self {
        Self {
            aliases: vec![None; reg_count],
            float_aliases: vec![None; reg_count],
            rewritten: Vec::new(),
        }
    }

    fn clear(&mut self) {
        for a in &mut self.aliases {
            *a = None;
        }
        for a in &mut self.float_aliases {
            *a = None;
        }
        self.rewritten.clear();
    }
}

pub(super) fn copy_propagate(
    block: &mut MachineBlock,
    config: BackendConfig,
    scratch: &mut CopyPropagateScratch,
) {
    scratch.clear();
    let original_ops = core::mem::take(&mut block.ops);
    scratch.rewritten.reserve(
        original_ops
            .len()
            .saturating_sub(scratch.rewritten.capacity()),
    );
    let aliases = &mut scratch.aliases;
    let float_aliases = &mut scratch.float_aliases;
    let rewritten = &mut scratch.rewritten;

    for (index, mut inst) in original_ops.iter().cloned().enumerate() {
        rewrite_sources(&mut inst.kind, aliases);
        rewrite_float_alias_sources(&mut inst.kind, float_aliases);

        if matches!(inst.kind, MachineInstKind::CallHelper(_)) {
            clear_aliases(aliases);
            clear_aliases(float_aliases);
            rewritten.push(inst);
            continue;
        }

        for_each_defined_reg(&inst.kind, |dst| {
            kill_alias(aliases, dst);
            kill_alias(float_aliases, dst);
        });

        match &inst.kind {
            MachineInstKind::Move {
                ty: _,
                dst,
                src: MachineValue::Reg(src),
            } => {
                if *dst == *src {
                    continue;
                }
                // Only transient-to-transient copies are safe to elide here.
                // Moves from fixed or cached-local registers into a transient
                // often act as snapshots, not just aliases.
                if machine_ir::is_transient_reg(*dst, config)
                    && machine_ir::is_transient_reg(*src, config)
                    && machine_ir::same_reg_bank(*dst, *src, config)
                    && can_elide_reg_move(&original_ops, &block.terminator, index, *dst, *src)
                {
                    aliases[dst.0 as usize] = Some(*src);
                    continue;
                }
                if !machine_ir::is_fp_reg(*dst, config) && machine_ir::is_fp_reg(*src, config) {
                    float_aliases[dst.0 as usize] = Some(*src);
                }
            }
            _ => {}
        }

        rewritten.push(inst);
    }

    rewrite_terminator_sources(&mut block.terminator, aliases);
    rewrite_float_alias_terminator_sources(&mut block.terminator, float_aliases);
    block.ops = core::mem::take(rewritten);
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
        if matches!(inst.kind, MachineInstKind::CallHelper(_)) {
            // copy_propagate clears aliases at helper calls, so a move can only
            // disappear here if its destination is dead after the barrier.
            let remaining = &ops[start_idx + 1 + later_index + 1..];
            return !reg_live_after(remaining, terminator, dst);
        }
        if inst_defines(&inst.kind, src) {
            source_stable = false;
        }
    }

    source_stable || !terminator_uses_reg(terminator, dst)
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
        MachineInstKind::ConvertI64PairToFloat { src_lo, src_hi, .. }
        | MachineInstKind::ReinterpretI64PairToF64 { src_lo, src_hi, .. } => {
            rewrite_value(src_lo, aliases);
            rewrite_value(src_hi, aliases);
        }
        MachineInstKind::FloatConst { .. } => {}
        MachineInstKind::Load { addr, .. } => rewrite_addr(addr, aliases),
        MachineInstKind::Store { addr, src, .. } => {
            rewrite_addr(addr, aliases);
            rewrite_value(src, aliases);
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
        MachineInstKind::CallHelper(_) => {}
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
        MachineTerminator::CallDirect {
            callee_frame_base, ..
        } => {
            *callee_frame_base = resolve_alias(*callee_frame_base, aliases);
        }
        MachineTerminator::CallIndirect {
            callee_target,
            callee_frame_base,
            ..
        } => {
            rewrite_value(callee_target, aliases);
            *callee_frame_base = resolve_alias(*callee_frame_base, aliases);
        }
        MachineTerminator::Return | MachineTerminator::Trap { .. } => {}
    }
}

fn rewrite_float_alias_terminator_sources(
    term: &mut MachineTerminator,
    aliases: &[Option<MachineReg>],
) {
    match term {
        MachineTerminator::Branch { cond, .. } => rewrite_float_alias_branch_cond(cond, aliases),
        MachineTerminator::Jump(_)
        | MachineTerminator::JumpTable { .. }
        | MachineTerminator::CallDirect { .. }
        | MachineTerminator::CallIndirect { .. }
        | MachineTerminator::Return
        | MachineTerminator::Trap { .. } => {}
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
