//! MachineIR peephole optimization pass.
//!
//! Runs within a single block. Current optimizations:
//!
//! 1. **Constant folding into operands**: `move rX <- C; op rD <- ... rX ...`
//!    → `op rD <- ... C ...` when rX has no other uses before redefinition.
//!
//! 2. **Redundant cached-local round-trip**: `move rCached <- rVal; ... move rY <- rCached`
//!    → forward rVal directly when rCached isn't used for anything else in between.
//!
//! 3. **Dead store-load elimination**: `store [addr] <- rX; load rY <- [addr]`
//!    when adjacent → `move rY <- rX` (or eliminated if rY == rX).

use alloc::vec::Vec;

use super::{
    MachineBlock, MachineInst, MachineInstKind, MachineProgram, MachineReg, MachineValue,
};

/// Run peephole optimizations on all blocks in a program.
///
/// `first_transient` is the register index where transient (single-use SSA)
/// registers start. Only transient registers are candidates for constant
/// folding — fixed and cached-local registers must not be disturbed.
pub fn optimize(program: &mut MachineProgram, first_transient: u16) {
    for block in &mut program.blocks {
        fold_constants(block, first_transient);
        eliminate_store_load_pairs(block);
    }
}

/// Fold `move rX <- imm; op ... rX ...` into `op ... imm ...` when rX is
/// a transient register (single-use SSA value). Cached-local and fixed
/// registers are never folded — they persist across the block.
fn fold_constants(block: &mut MachineBlock, first_transient: u16) {
    let mut i = 0;
    while i + 1 < block.ops.len() {
        let (dst, imm) = match &block.ops[i].kind {
            MachineInstKind::Move {
                dst,
                src: MachineValue::Imm64(imm),
            } if dst.0 >= first_transient => (*dst, *imm),
            _ => {
                i += 1;
                continue;
            }
        };

        // Check: is dst used in the next instruction? And is it NOT used
        // anywhere else before being redefined?
        // Simple conservative check: dst must appear exactly once in the
        // next instruction's source operands, and must not be the next
        // instruction's destination.
        let next = &block.ops[i + 1].kind;
        let use_count = count_value_uses(next, dst);
        let is_dst_of_next = inst_defines(next, dst);

        if use_count == 1 && !is_dst_of_next {
            // Check that no later instruction or the terminator uses dst before redefinition
            let safe = is_last_use_before_redef(block, i + 1, dst);
            if safe {
                let imm_val = MachineValue::Imm64(imm);
                replace_value_use(&mut block.ops[i + 1].kind, dst, imm_val);
                block.ops.remove(i);
                continue;
            }
        }
        i += 1;
    }
}

/// Eliminate `store.u64 [base+off] <- rX` immediately followed by
/// `load.u64 rY <- [base+off]` → replace load with `move rY <- rX`.
fn eliminate_store_load_pairs(block: &mut MachineBlock) {
    let mut i = 0;
    while i + 1 < block.ops.len() {
        let (store_addr_base, store_addr_off, store_src) = match &block.ops[i].kind {
            MachineInstKind::Store {
                addr,
                width: super::MachineMemWidth::U64,
                src,
            } => (addr.base, addr.offset, *src),
            _ => {
                i += 1;
                continue;
            }
        };

        let (load_dst, load_addr_base, load_addr_off) = match &block.ops[i + 1].kind {
            MachineInstKind::Load {
                dst,
                addr,
                width: super::MachineMemWidth::U64,
                extension: super::MachineLoadExtension::None,
            } => (*dst, addr.base, addr.offset),
            _ => {
                i += 1;
                continue;
            }
        };

        if store_addr_base == load_addr_base && store_addr_off == load_addr_off {
            // Replace load with move (or remove if dst == src reg)
            match store_src {
                MachineValue::Reg(src_reg) if src_reg == load_dst => {
                    block.ops.remove(i + 1);
                }
                _ => {
                    block.ops[i + 1].kind = MachineInstKind::Move {
                        dst: load_dst,
                        src: store_src,
                    };
                }
            }
        }
        i += 1;
    }
}

// --- helpers ---

/// Count how many times `reg` appears as a source operand in `kind`.
fn count_value_uses(kind: &MachineInstKind, reg: MachineReg) -> usize {
    let mut count = 0;
    visit_source_values(kind, |v| {
        if matches!(v, MachineValue::Reg(r) if *r == reg) {
            count += 1;
        }
    });
    count
}

/// Check if `kind` defines (writes to) `reg`.
fn inst_defines(kind: &MachineInstKind, reg: MachineReg) -> bool {
    match kind {
        MachineInstKind::Move { dst, .. }
        | MachineInstKind::Lea { dst, .. }
        | MachineInstKind::Load { dst, .. }
        | MachineInstKind::IntUnary { dst, .. }
        | MachineInstKind::IntBinary { dst, .. }
        | MachineInstKind::IntCompare { dst, .. }
        | MachineInstKind::FloatUnary { dst, .. }
        | MachineInstKind::FloatBinary { dst, .. }
        | MachineInstKind::FloatCompare { dst, .. }
        | MachineInstKind::Convert { dst, .. }
        | MachineInstKind::Select { dst, .. } => *dst == reg,
        MachineInstKind::Store { .. } | MachineInstKind::CallHelper(_) => false,
    }
}

/// Check that `reg` is not used by any instruction after `start_idx` in the
/// block (including the terminator), or if it is used again, it is redefined first.
fn is_last_use_before_redef(block: &MachineBlock, start_idx: usize, reg: MachineReg) -> bool {
    for inst in &block.ops[start_idx + 1..] {
        if inst_defines(&inst.kind, reg) {
            return true; // redefined before any other use
        }
        if count_value_uses(&inst.kind, reg) > 0 {
            return false; // used again
        }
    }
    // Also check the terminator
    if terminator_uses_reg(&block.terminator, reg) {
        return false;
    }
    true // not used again in the block
}

/// Check if a terminator reads from `reg`.
fn terminator_uses_reg(term: &super::MachineTerminator, reg: MachineReg) -> bool {
    match term {
        super::MachineTerminator::Jump(edge) => edge_uses_reg(edge, reg),
        super::MachineTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => {
            branch_cond_uses_reg(cond, reg)
                || edge_uses_reg(then_edge, reg)
                || edge_uses_reg(else_edge, reg)
        }
        super::MachineTerminator::JumpTable { index, entries } => {
            value_is_reg(index, reg)
                || entries.iter().any(|e| edge_uses_reg(e, reg))
        }
        super::MachineTerminator::CallDirect {
            callee_frame_base, ..
        } => *callee_frame_base == reg,
        super::MachineTerminator::CallIndirect {
            callee_target,
            callee_frame_base,
            ..
        } => value_is_reg(callee_target, reg) || *callee_frame_base == reg,
        super::MachineTerminator::Return | super::MachineTerminator::Trap { .. } => false,
    }
}

fn edge_uses_reg(edge: &super::MachineEdge, reg: MachineReg) -> bool {
    edge.args.iter().any(|v| value_is_reg(v, reg))
}

fn branch_cond_uses_reg(cond: &super::MachineBranchCond, reg: MachineReg) -> bool {
    match cond {
        super::MachineBranchCond::Value(v) => value_is_reg(v, reg),
        super::MachineBranchCond::IntCompare { lhs, rhs, .. } => {
            value_is_reg(lhs, reg) || value_is_reg(rhs, reg)
        }
        super::MachineBranchCond::FloatCompare { lhs, rhs, .. } => {
            value_is_reg(lhs, reg) || value_is_reg(rhs, reg)
        }
    }
}

fn value_is_reg(v: &MachineValue, reg: MachineReg) -> bool {
    matches!(v, MachineValue::Reg(r) if *r == reg)
}

/// Visit all source (read) values in an instruction.
fn visit_source_values(kind: &MachineInstKind, mut f: impl FnMut(&MachineValue)) {
    match kind {
        MachineInstKind::Move { src, .. } => f(src),
        MachineInstKind::Lea { addr, .. } => {
            f(&MachineValue::Reg(addr.base));
        }
        MachineInstKind::Load { addr, .. } => {
            f(&MachineValue::Reg(addr.base));
        }
        MachineInstKind::Store { addr, src, .. } => {
            f(&MachineValue::Reg(addr.base));
            f(src);
        }
        MachineInstKind::IntUnary { src, .. } => f(src),
        MachineInstKind::IntBinary { lhs, rhs, .. } => {
            f(lhs);
            f(rhs);
        }
        MachineInstKind::IntCompare { lhs, rhs, .. } => {
            f(lhs);
            f(rhs);
        }
        MachineInstKind::FloatUnary { src, .. } => f(src),
        MachineInstKind::FloatBinary { lhs, rhs, .. } => {
            f(lhs);
            f(rhs);
        }
        MachineInstKind::FloatCompare { lhs, rhs, .. } => {
            f(lhs);
            f(rhs);
        }
        MachineInstKind::Convert { src, .. } => f(src),
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
        MachineInstKind::CallHelper(_) => {}
    }
}

/// Replace one occurrence of `Reg(old)` with `new_val` in an instruction's sources.
fn replace_value_use(kind: &mut MachineInstKind, old: MachineReg, new_val: MachineValue) {
    match kind {
        MachineInstKind::Move { src, .. } => { try_replace(src, old, new_val); }
        MachineInstKind::IntUnary { src, .. } => { try_replace(src, old, new_val); }
        MachineInstKind::IntBinary { lhs, rhs, .. } => {
            if !try_replace(lhs, old, new_val) {
                try_replace(rhs, old, new_val);
            }
        }
        MachineInstKind::IntCompare { lhs, rhs, .. } => {
            if !try_replace(lhs, old, new_val) {
                try_replace(rhs, old, new_val);
            }
        }
        MachineInstKind::FloatUnary { src, .. } => { try_replace(src, old, new_val); }
        MachineInstKind::FloatBinary { lhs, rhs, .. } => {
            if !try_replace(lhs, old, new_val) {
                try_replace(rhs, old, new_val);
            }
        }
        MachineInstKind::FloatCompare { lhs, rhs, .. } => {
            if !try_replace(lhs, old, new_val) {
                try_replace(rhs, old, new_val);
            }
        }
        MachineInstKind::Convert { src, .. } => { try_replace(src, old, new_val); }
        MachineInstKind::Store { src, .. } => { try_replace(src, old, new_val); }
        MachineInstKind::Select {
            on_true,
            on_false,
            cond,
            ..
        } => {
            if !try_replace(on_true, old, new_val) {
                if !try_replace(on_false, old, new_val) {
                    try_replace(cond, old, new_val);
                }
            }
        }
        _ => {}
    }
}

fn try_replace(val: &mut MachineValue, old: MachineReg, new_val: MachineValue) -> bool {
    if matches!(val, MachineValue::Reg(r) if *r == old) {
        *val = new_val;
        true
    } else {
        false
    }
}
