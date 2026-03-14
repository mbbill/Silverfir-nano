//! MachineIR peephole optimization pass.
//!
//! Runs within a single block. Current optimizations:
//!
//! 1. **Constant folding into operands**: `move rX <- C; op rD <- ... rX ...`
//!    → `op rD <- ... C ...` when rX has no other uses before redefinition.
//!
//! 2. **Copy propagation**: `move rTmp <- rSrc; op ... rTmp ...`
//!    → rewrite later uses to `rSrc` and remove the transient move.
//!
//! 3. **Store-to-load forwarding**: `store.u64 [addr] <- X; ...; load.u64 rY <- [addr]`
//!    → `move rY <- X` when intervening ops cannot invalidate the exact address.

use alloc::vec;
use alloc::vec::Vec;

use super::{
    MachineAddr, MachineBlock, MachineBranchCond, MachineEdge, MachineInst, MachineInstKind,
    MachineProgram, MachineReg, MachineTerminator, MachineValue,
};

#[derive(Clone, Copy)]
struct TrackedStore {
    addr: MachineAddr,
    src: MachineValue,
}

#[derive(Clone, Copy)]
struct TrackedLoad {
    addr: MachineAddr,
    width: super::MachineMemWidth,
    extension: super::MachineLoadExtension,
    reg: MachineReg,
}

/// Run peephole optimizations on all blocks in a program.
///
/// `first_transient` is the register index where transient (single-use SSA)
/// registers start. Only transient registers are candidates for constant
/// folding — fixed and cached-local registers must not be disturbed.
pub fn optimize(program: &mut MachineProgram, first_transient: u16) {
    for block in &mut program.blocks {
        fold_constants(block, first_transient);
        copy_propagate(block, program.reg_count, first_transient);
        forward_stored_values(block);
        reuse_loaded_values(block);
        fold_constants(block, first_transient);
        copy_propagate(block, program.reg_count, first_transient);
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

/// Forward exact `store.u64` values into later exact `load.u64` instructions
/// within a block when no intervening instruction can change the address or
/// the stored source value.
fn forward_stored_values(block: &mut MachineBlock) {
    let mut tracked = Vec::<TrackedStore>::new();
    let mut rewritten = Vec::with_capacity(block.ops.len());

    for mut inst in block.ops.drain(..) {
        let mut keep_inst = true;

        match &mut inst.kind {
            MachineInstKind::Load {
                dst,
                addr,
                width: super::MachineMemWidth::U64,
                extension: super::MachineLoadExtension::None,
            } => {
                if let Some(src) = tracked
                    .iter()
                    .rev()
                    .find(|entry| entry.addr == *addr)
                    .map(|entry| entry.src)
                {
                    if matches!(src, MachineValue::Reg(src_reg) if src_reg == *dst) {
                        keep_inst = false;
                    } else {
                        inst.kind = MachineInstKind::Move { dst: *dst, src };
                    }
                }
            }
            MachineInstKind::Store { addr, width, src } => {
                tracked.retain(|entry| {
                    !addrs_overlap(entry.addr, super::MachineMemWidth::U64, *addr, *width)
                });
                if *width == super::MachineMemWidth::U64 {
                    tracked.push(TrackedStore {
                        addr: *addr,
                        src: *src,
                    });
                }
            }
            MachineInstKind::CallHelper(_) => {
                tracked.clear();
            }
            _ => {}
        }

        if keep_inst {
            if let Some(dst) = defined_reg(&inst.kind) {
                kill_tracked_stores_by_reg(&mut tracked, dst);
            }
            rewritten.push(inst);
        }
    }

    block.ops = rewritten;
}

/// Reuse an earlier exact load result for a later identical load when no
/// intervening store, helper call, or register redefinition can invalidate it.
fn reuse_loaded_values(block: &mut MachineBlock) {
    let mut tracked = Vec::<TrackedLoad>::new();
    let mut rewritten = Vec::with_capacity(block.ops.len());

    for mut inst in block.ops.drain(..) {
        let mut keep_inst = true;
        let mut produced_load = None;
        let mut rewrite_load = None;

        match &inst.kind {
            MachineInstKind::Load {
                dst,
                addr,
                width,
                extension,
            } => {
                if let Some(src_reg) = tracked
                    .iter()
                    .rev()
                    .find(|entry| {
                        entry.addr == *addr
                            && entry.width == *width
                            && entry.extension == *extension
                    })
                    .map(|entry| entry.reg)
                {
                    if src_reg == *dst {
                        keep_inst = false;
                    } else {
                        rewrite_load = Some((*dst, src_reg));
                        produced_load = Some(TrackedLoad {
                            addr: *addr,
                            width: *width,
                            extension: *extension,
                            reg: *dst,
                        });
                    }
                } else {
                    produced_load = Some(TrackedLoad {
                        addr: *addr,
                        width: *width,
                        extension: *extension,
                        reg: *dst,
                    });
                }
            }
            MachineInstKind::Store { addr, width, .. } => {
                tracked.retain(|entry| !addrs_overlap(entry.addr, entry.width, *addr, *width));
            }
            MachineInstKind::CallHelper(_) => {
                tracked.clear();
            }
            _ => {}
        }

        if keep_inst {
            if let Some((dst, src_reg)) = rewrite_load {
                inst.kind = MachineInstKind::Move {
                    dst,
                    src: MachineValue::Reg(src_reg),
                };
            }
            if let Some(dst) = defined_reg(&inst.kind) {
                kill_tracked_loads_by_reg(&mut tracked, dst);
            }
            if let Some(load) = produced_load {
                tracked.push(load);
            }
            rewritten.push(inst);
        }
    }

    block.ops = rewritten;
}

/// Track transient register aliases within a block and rewrite later uses to
/// the original source register. Cached-local and fixed-register writes are
/// preserved, but their sources are still canonicalized.
fn copy_propagate(block: &mut MachineBlock, reg_count: u16, first_transient: u16) {
    let original_ops = core::mem::take(&mut block.ops);
    let mut aliases = vec![None; reg_count as usize];
    let mut rewritten = Vec::with_capacity(original_ops.len());

    for (index, mut inst) in original_ops.iter().cloned().enumerate() {
        rewrite_sources(&mut inst.kind, &aliases);

        if matches!(inst.kind, MachineInstKind::CallHelper(_)) {
            // Helpers preserve the abstract machine-register file, but they can
            // mutate canonical frame slots that later register reloads observe.
            // Dropping aliases here keeps the pass aligned with MachineIR's
            // memory-visible semantics instead of assuming helper purity.
            clear_aliases(&mut aliases);
            rewritten.push(inst);
            continue;
        }

        if let Some(dst) = defined_reg(&inst.kind) {
            kill_alias(&mut aliases, dst);
        }

        match &inst.kind {
            MachineInstKind::Move {
                dst,
                src: MachineValue::Reg(src),
            } => {
                if *dst == *src {
                    continue;
                }
                if dst.0 >= first_transient
                    && can_elide_reg_move(&original_ops, &block.terminator, index, *dst, *src)
                {
                    aliases[dst.0 as usize] = Some(*src);
                    continue;
                }
            }
            _ => {}
        }

        rewritten.push(inst);
    }

    rewrite_terminator_sources(&mut block.terminator, &aliases);
    block.ops = rewritten;
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

fn defined_reg(kind: &MachineInstKind) -> Option<MachineReg> {
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
        | MachineInstKind::Select { dst, .. } => Some(*dst),
        MachineInstKind::Store { .. } | MachineInstKind::CallHelper(_) => None,
    }
}

/// Check that `reg` is not used by any instruction after `start_idx` in the
/// block (including the terminator), or if it is used again, it is redefined first.
fn is_last_use_before_redef(block: &MachineBlock, start_idx: usize, reg: MachineReg) -> bool {
    for inst in &block.ops[start_idx + 1..] {
        if count_value_uses(&inst.kind, reg) > 0 {
            return false; // used again, even if the instruction also redefines it
        }
        if inst_defines(&inst.kind, reg) {
            return true; // redefined before any other use
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

fn rewrite_sources(kind: &mut MachineInstKind, aliases: &[Option<MachineReg>]) {
    match kind {
        MachineInstKind::Move { src, .. }
        | MachineInstKind::IntUnary { src, .. }
        | MachineInstKind::FloatUnary { src, .. }
        | MachineInstKind::Convert { src, .. } => rewrite_value(src, aliases),
        MachineInstKind::Lea { addr, .. } | MachineInstKind::Load { addr, .. } => {
            rewrite_addr(addr, aliases);
        }
        MachineInstKind::Store { addr, src, .. } => {
            rewrite_addr(addr, aliases);
            rewrite_value(src, aliases);
        }
        MachineInstKind::IntBinary { lhs, rhs, .. }
        | MachineInstKind::IntCompare { lhs, rhs, .. }
        | MachineInstKind::FloatBinary { lhs, rhs, .. }
        | MachineInstKind::FloatCompare { lhs, rhs, .. } => {
            rewrite_value(lhs, aliases);
            rewrite_value(rhs, aliases);
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

fn rewrite_branch_cond(cond: &mut MachineBranchCond, aliases: &[Option<MachineReg>]) {
    match cond {
        MachineBranchCond::Value(value) => rewrite_value(value, aliases),
        MachineBranchCond::IntCompare { lhs, rhs, .. }
        | MachineBranchCond::FloatCompare { lhs, rhs, .. } => {
            rewrite_value(lhs, aliases);
            rewrite_value(rhs, aliases);
        }
    }
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

fn can_elide_reg_move(
    ops: &[MachineInst],
    terminator: &MachineTerminator,
    start_idx: usize,
    dst: MachineReg,
    src: MachineReg,
) -> bool {
    let mut source_stable = true;

    for inst in &ops[start_idx + 1..] {
        if count_value_uses(&inst.kind, dst) > 0 && !source_stable {
            return false;
        }
        if inst_defines(&inst.kind, dst) {
            return true;
        }
        if inst_defines(&inst.kind, src) {
            source_stable = false;
        }
    }

    source_stable || !terminator_uses_reg(terminator, dst)
}

fn kill_tracked_stores_by_reg(tracked: &mut Vec<TrackedStore>, reg: MachineReg) {
    tracked.retain(|entry| entry.addr.base != reg && !value_is_reg(&entry.src, reg));
}

fn kill_tracked_loads_by_reg(tracked: &mut Vec<TrackedLoad>, reg: MachineReg) {
    tracked.retain(|entry| entry.addr.base != reg && entry.reg != reg);
}

fn addrs_overlap(
    lhs_addr: MachineAddr,
    lhs_width: super::MachineMemWidth,
    rhs_addr: MachineAddr,
    rhs_width: super::MachineMemWidth,
) -> bool {
    if lhs_addr.base != rhs_addr.base {
        return false;
    }

    let lhs_start = i64::from(lhs_addr.offset);
    let lhs_end = lhs_start + mem_width_bytes(lhs_width);
    let rhs_start = i64::from(rhs_addr.offset);
    let rhs_end = rhs_start + mem_width_bytes(rhs_width);

    lhs_start < rhs_end && rhs_start < lhs_end
}

fn mem_width_bytes(width: super::MachineMemWidth) -> i64 {
    match width {
        super::MachineMemWidth::U8 => 1,
        super::MachineMemWidth::U16 => 2,
        super::MachineMemWidth::U32 => 4,
        super::MachineMemWidth::U64 => 8,
    }
}
