//! MachineIR peephole optimization pass.
//!
//! Runs within a single block. Current optimizations:
//!
//! 1. **Constant deduplication**: When the same constant value is materialized
//!    multiple times (`Move { src: Imm64 }` or `FloatConst`), subsequent
//!    occurrences are replaced with register copies from the first.
//!
//! 2. **Constant folding into operands**: `move rX <- C; op rD <- ... rX ...`
//!    → `op rD <- ... C ...` when rX has no other uses before redefinition.
//!
//! 3. **Copy propagation**: `move rTmp <- rSrc; op ... rTmp ...`
//!    → rewrite later uses to `rSrc` and remove the transient move.
//!
//! 4. **Store-to-load forwarding**: `store.u64 [addr] <- X; ...; load.u64 rY <- [addr]`
//!    → `move rY <- X` when intervening ops cannot invalidate the exact address.
//!
//! 5. **Compare-and-branch fusion**: `IntCompare { dst } + Branch { Value(Reg(dst)) }`
//!    → `Branch { IntCompare { ... } }` when the compare result register is dead in
//!    both successor blocks. Eliminates the boolean materialization (CSET on ARM64,
//!    SETCC+MOVZX on x86_64) and enables hardware compare-and-branch fusion.
//!    FloatCompare is NOT fused because x86_64 requires multi-instruction NaN
//!    handling that cannot be expressed as a single conditional branch.

use alloc::vec;
use alloc::vec::Vec;

use super::{
    MachineAddr, MachineBlock, MachineBlockId, MachineBranchCond, MachineEdge, MachineFloatWidth,
    MachineInst, MachineInstKind, MachineProgram, MachineReg, MachineStorageType,
    MachineTerminator, MachineValue,
};

#[derive(Clone, Copy)]
struct TrackedStore {
    addr: MachineAddr,
    src: MachineValue,
}

#[derive(Clone, Copy)]
struct TrackedLoad {
    addr: MachineAddr,
    ty: MachineStorageType,
    width: super::MachineMemWidth,
    extension: super::MachineLoadExtension,
    reg: MachineReg,
}

/// Run peephole optimizations on all blocks in a program.
///
/// `first_transient` is the first GP transient register. FP transients are the
/// prefix of the FP bank with length `program.fp_transient_count`. Only
/// transient registers are candidates for copy/constant rewriting; fixed and
/// cached-local registers must not be disturbed.
pub fn optimize(program: &mut MachineProgram, first_transient: u16, gp_reg_width: u8) {
    let fp_transient_end = program.first_fp_reg + program.fp_transient_count;
    let mut cp_scratch = CopyPropagateScratch::new(program.reg_count as usize);
    for block in &mut program.blocks {
        deduplicate_constants(block, program.first_fp_reg);
        fold_constants(
            block,
            first_transient,
            program.first_fp_reg,
            fp_transient_end,
        );
        copy_propagate(
            block,
            first_transient,
            program.first_fp_reg,
            program.fp_transient_count,
            gp_reg_width,
            &mut cp_scratch,
        );
        forward_stored_values(block, program.first_fp_reg);
        reuse_loaded_values(block, program.first_fp_reg);
        fold_constants(
            block,
            first_transient,
            program.first_fp_reg,
            fp_transient_end,
        );
        copy_propagate(
            block,
            first_transient,
            program.first_fp_reg,
            program.fp_transient_count,
            gp_reg_width,
            &mut cp_scratch,
        );
    }
    // Compare-and-branch fusion needs cross-block liveness, so it runs after
    // the per-block passes when the instruction stream is stable.
    fuse_compare_branch(&mut program.blocks, gp_reg_width);
}

/// Replace duplicate constant materializations with register copies.
///
/// Within a block, if the same constant value is materialized into multiple
/// registers (via `Move { src: Imm64 }` or `FloatConst`), the second and
/// subsequent materializations are replaced with register-to-register copies.
///
/// Runs before `fold_constants` so that shared constants keep their defining
/// instruction alive, preventing fold from inlining the same expensive constant
/// into multiple consumers independently.
fn deduplicate_constants(block: &mut MachineBlock, first_fp_reg: u16) {
    let mut gp_consts: Vec<(u64, MachineReg)> = Vec::new();
    let mut fp_consts: Vec<(u64, MachineFloatWidth, MachineReg)> = Vec::new();

    for inst in &mut block.ops {
        if matches!(inst.kind, MachineInstKind::CallHelper(_)) {
            gp_consts.clear();
            fp_consts.clear();
            continue;
        }

        let mut new_gp = None;
        let mut new_fp = None;

        match &mut inst.kind {
            MachineInstKind::Move {
                dst,
                src: src @ MachineValue::Imm64(..),
                ..
            } if dst.0 < first_fp_reg => {
                let bits = match *src {
                    MachineValue::Imm64(b) => b,
                    _ => unreachable!(),
                };
                // Skip zero: fold_constants can inline Imm64(0) into consumers
                // for free (str xzr, cmp #0, etc.), so dedup would be a regression.
                if bits != 0 {
                    if let Some(&(_, prev)) =
                        gp_consts.iter().find(|(b, r)| *b == bits && *r != *dst)
                    {
                        *src = MachineValue::Reg(prev);
                    }
                    new_gp = Some((bits, *dst));
                }
            }
            MachineInstKind::FloatConst { dst, bits, width } => {
                let (d, b, w) = (*dst, *bits, *width);
                // Skip zero: fcmp d, #0.0 is free when folded as Imm64(0).
                if b != 0 {
                    if let Some(&(_, _, prev)) = fp_consts
                        .iter()
                        .find(|(bb, ww, r)| *bb == b && *ww == w && *r != d)
                    {
                        inst.kind = MachineInstKind::Move {
                            ty: match w {
                                MachineFloatWidth::F32 => MachineStorageType::Fp32,
                                MachineFloatWidth::F64 => MachineStorageType::Fp64,
                            },
                            dst: d,
                            src: MachineValue::Reg(prev),
                        };
                    }
                    new_fp = Some((b, w, d));
                }
            }
            _ => {}
        }

        // Invalidate tracking for any register redefined by this instruction.
        if let Some(def) = defined_reg(&inst.kind) {
            gp_consts.retain(|(_, r)| *r != def);
            fp_consts.retain(|(_, _, r)| *r != def);
        }

        if let Some(e) = new_gp {
            gp_consts.push(e);
        }
        if let Some(e) = new_fp {
            fp_consts.push(e);
        }
    }
}

/// Fold `move rX <- imm; op ... rX ...` into `op ... imm ...` when rX is
/// a transient register (single-use SSA value). Also folds
/// `FloatConst rX <- bits; FloatCompare/op ... rX ...` into `op ... Imm64(bits) ...`.
/// Cached-local and fixed registers are never folded — they persist across the block.
fn fold_constants(
    block: &mut MachineBlock,
    first_transient: u16,
    first_fp_reg: u16,
    fp_transient_end: u16,
) {
    let mut i = 0;
    while i + 1 < block.ops.len() {
        let (dst, imm) = match &block.ops[i].kind {
            // GP transient constant: move rX <- imm
            MachineInstKind::Move {
                dst,
                src: MachineValue::Imm64(imm),
                ..
            } if dst.0 >= first_transient && dst.0 < first_fp_reg => (*dst, *imm),
            // FP transient constant: FloatConst rX <- bits
            MachineInstKind::FloatConst { dst, bits, .. }
                if dst.0 >= first_fp_reg && dst.0 < fp_transient_end =>
            {
                (*dst, *bits)
            }
            _ => {
                i += 1;
                continue;
            }
        };

        let next = &block.ops[i + 1].kind;
        let use_count = count_value_uses(next, dst);
        let is_dst_of_next = inst_defines(next, dst);

        if use_count == 1 && !is_dst_of_next {
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
fn forward_stored_values(block: &mut MachineBlock, first_fp_reg: u16) {
    let mut tracked = Vec::<TrackedStore>::new();
    let mut rewritten = Vec::with_capacity(block.ops.len());

    for mut inst in block.ops.drain(..) {
        let mut keep_inst = true;

        match &mut inst.kind {
            MachineInstKind::Load {
                ty,
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
                    } else if let Some(move_ty) =
                        rewrite_move_storage_type(*dst, src, *ty, first_fp_reg)
                    {
                        inst.kind = MachineInstKind::Move {
                            ty: move_ty,
                            dst: *dst,
                            src,
                        };
                    }
                }
            }
            MachineInstKind::Store {
                addr, width, src, ..
            } => {
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
fn reuse_loaded_values(block: &mut MachineBlock, first_fp_reg: u16) {
    let mut tracked = Vec::<TrackedLoad>::new();
    let mut rewritten = Vec::with_capacity(block.ops.len());

    for mut inst in block.ops.drain(..) {
        let mut keep_inst = true;
        let mut produced_load = None;
        let mut rewrite_load = None;

        match &inst.kind {
            MachineInstKind::Load {
                ty,
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
                            && entry.ty == *ty
                            && entry.width == *width
                            && entry.extension == *extension
                    })
                    .map(|entry| entry.reg)
                {
                    if src_reg == *dst {
                        keep_inst = false;
                    } else if let Some(move_ty) = rewrite_move_storage_type(
                        *dst,
                        MachineValue::Reg(src_reg),
                        *ty,
                        first_fp_reg,
                    ) {
                        rewrite_load = Some((*dst, src_reg, move_ty));
                        produced_load = Some(TrackedLoad {
                            addr: *addr,
                            ty: *ty,
                            width: *width,
                            extension: *extension,
                            reg: *dst,
                        });
                    }
                } else {
                    produced_load = Some(TrackedLoad {
                        addr: *addr,
                        ty: *ty,
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
            if let Some((dst, src_reg, ty)) = rewrite_load {
                inst.kind = MachineInstKind::Move {
                    ty,
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
/// Reusable scratch buffers for copy_propagate to avoid per-block allocation.
struct CopyPropagateScratch {
    aliases: Vec<Option<MachineReg>>,
    float_aliases: Vec<Option<MachineReg>>,
    rewritten: Vec<MachineInst>,
}

impl CopyPropagateScratch {
    fn new(reg_count: usize) -> Self {
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

fn copy_propagate(
    block: &mut MachineBlock,
    first_transient: u16,
    first_fp_reg: u16,
    fp_transient_count: u16,
    gp_reg_width: u8,
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

        if let Some(dst) = defined_reg(&inst.kind) {
            kill_alias(aliases, dst);
            kill_alias(float_aliases, dst);
        }

        match &inst.kind {
            MachineInstKind::Move {
                ty,
                dst,
                src: MachineValue::Reg(src),
            } => {
                if *dst == *src {
                    continue;
                }
                if is_transient_reg(*dst, first_transient, first_fp_reg, fp_transient_count)
                    && (gp_reg_width != 4 || ty.is_fp())
                    && same_reg_bank(*dst, *src, first_fp_reg)
                    && can_elide_reg_move(&original_ops, &block.terminator, index, *dst, *src)
                {
                    aliases[dst.0 as usize] = Some(*src);
                    continue;
                }
                if dst.0 < first_fp_reg && src.0 >= first_fp_reg {
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
        | MachineInstKind::FloatConst { dst, .. }
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
        MachineInstKind::IntMulWide { dst_lo, dst_hi, .. } => *dst_lo == reg || *dst_hi == reg,
        MachineInstKind::Int64PairUnary { dst_lo, dst_hi, .. } => *dst_lo == reg || *dst_hi == reg,
        MachineInstKind::Int64PairDivRem { dst_lo, dst_hi, .. } => *dst_lo == reg || *dst_hi == reg,
        MachineInstKind::Int64PairShift { dst_lo, dst_hi, .. } => *dst_lo == reg || *dst_hi == reg,
        MachineInstKind::ConvertFloatToI64Pair { dst_lo, dst_hi, .. } => {
            *dst_lo == reg || *dst_hi == reg
        }
        MachineInstKind::ReinterpretF64ToI64Pair { dst_lo, dst_hi, .. } => {
            *dst_lo == reg || *dst_hi == reg
        }
        MachineInstKind::ConvertI64PairToFloat { dst, .. }
        | MachineInstKind::ReinterpretI64PairToF64 { dst, .. } => *dst == reg,
        MachineInstKind::Store { .. }
        | MachineInstKind::TrapIf { .. }
        | MachineInstKind::CallHelper(_) => false,
    }
}

fn defined_reg(kind: &MachineInstKind) -> Option<MachineReg> {
    match kind {
        MachineInstKind::Move { dst, .. }
        | MachineInstKind::FloatConst { dst, .. }
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
        MachineInstKind::IntMulWide { .. } => None,
        MachineInstKind::Int64PairUnary { .. } => None,
        MachineInstKind::Int64PairDivRem { .. } => None,
        MachineInstKind::Int64PairShift { .. } => None,
        MachineInstKind::ConvertFloatToI64Pair { .. } => None,
        MachineInstKind::ConvertI64PairToFloat { dst, .. }
        | MachineInstKind::ReinterpretI64PairToF64 { dst, .. } => Some(*dst),
        MachineInstKind::ReinterpretF64ToI64Pair { .. } => None,
        MachineInstKind::Store { .. }
        | MachineInstKind::TrapIf { .. }
        | MachineInstKind::CallHelper(_) => None,
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
            value_is_reg(index, reg) || entries.iter().any(|e| edge_uses_reg(e, reg))
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
        MachineInstKind::FloatConst { .. } => {}
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
        MachineInstKind::IntMulWide { lhs, rhs, .. } => {
            f(lhs);
            f(rhs);
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
        MachineInstKind::TrapIf { cond, .. } => visit_branch_cond_values(cond, &mut f),
        MachineInstKind::CallHelper(_) => {}
    }
}

/// Replace one occurrence of `Reg(old)` with `new_val` in an instruction's sources.
fn replace_value_use(kind: &mut MachineInstKind, old: MachineReg, new_val: MachineValue) {
    match kind {
        MachineInstKind::Move { src, .. } => {
            try_replace(src, old, new_val);
        }
        MachineInstKind::IntUnary { src, .. } => {
            try_replace(src, old, new_val);
        }
        MachineInstKind::IntBinary { lhs, rhs, .. } => {
            if !try_replace(lhs, old, new_val) {
                try_replace(rhs, old, new_val);
            }
        }
        MachineInstKind::IntMulWide { lhs, rhs, .. } => {
            if !try_replace(lhs, old, new_val) {
                try_replace(rhs, old, new_val);
            }
        }
        MachineInstKind::Int64PairUnary { src_lo, src_hi, .. } => {
            if !try_replace(src_lo, old, new_val) {
                try_replace(src_hi, old, new_val);
            }
        }
        MachineInstKind::Int64PairDivRem {
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        } => {
            if !try_replace(lhs_lo, old, new_val) {
                if !try_replace(lhs_hi, old, new_val) {
                    if !try_replace(rhs_lo, old, new_val) {
                        try_replace(rhs_hi, old, new_val);
                    }
                }
            }
        }
        MachineInstKind::Int64PairShift {
            lhs_lo,
            lhs_hi,
            rhs,
            ..
        } => {
            if !try_replace(lhs_lo, old, new_val) {
                if !try_replace(lhs_hi, old, new_val) {
                    try_replace(rhs, old, new_val);
                }
            }
        }
        MachineInstKind::IntCompare { lhs, rhs, .. } => {
            if !try_replace(lhs, old, new_val) {
                try_replace(rhs, old, new_val);
            }
        }
        MachineInstKind::FloatUnary { src, .. } => {
            try_replace(src, old, new_val);
        }
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
        MachineInstKind::Convert { src, .. } => {
            try_replace(src, old, new_val);
        }
        MachineInstKind::ConvertI64PairToFloat { src_lo, src_hi, .. } => {
            if !try_replace(src_lo, old, new_val) {
                try_replace(src_hi, old, new_val);
            }
        }
        MachineInstKind::ConvertFloatToI64Pair { src, .. }
        | MachineInstKind::ReinterpretF64ToI64Pair { src, .. } => {
            try_replace(src, old, new_val);
        }
        MachineInstKind::ReinterpretI64PairToF64 { src_lo, src_hi, .. } => {
            if !try_replace(src_lo, old, new_val) {
                try_replace(src_hi, old, new_val);
            }
        }
        MachineInstKind::Store { src, .. } => {
            try_replace(src, old, new_val);
        }
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
        MachineInstKind::TrapIf { cond, .. } => replace_branch_cond_value(cond, old, new_val),
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
        | MachineInstKind::Convert { src, .. }
        | MachineInstKind::ConvertFloatToI64Pair { src, .. }
        | MachineInstKind::ReinterpretF64ToI64Pair { src, .. } => rewrite_value(src, aliases),
        MachineInstKind::ConvertI64PairToFloat { src_lo, src_hi, .. }
        | MachineInstKind::ReinterpretI64PairToF64 { src_lo, src_hi, .. } => {
            rewrite_value(src_lo, aliases);
            rewrite_value(src_hi, aliases);
        }
        MachineInstKind::FloatConst { .. } => {}
        MachineInstKind::Lea { addr, .. } | MachineInstKind::Load { addr, .. } => {
            rewrite_addr(addr, aliases);
        }
        MachineInstKind::Store { addr, src, .. } => {
            rewrite_addr(addr, aliases);
            rewrite_value(src, aliases);
        }
        MachineInstKind::IntBinary { lhs, rhs, .. }
        | MachineInstKind::IntMulWide { lhs, rhs, .. }
        | MachineInstKind::IntCompare { lhs, rhs, .. }
        | MachineInstKind::FloatBinary { lhs, rhs, .. }
        | MachineInstKind::FloatCompare { lhs, rhs, .. } => {
            rewrite_value(lhs, aliases);
            rewrite_value(rhs, aliases);
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

fn rewrite_float_alias_branch_cond(cond: &mut MachineBranchCond, aliases: &[Option<MachineReg>]) {
    if let MachineBranchCond::FloatCompare { lhs, rhs, .. } = cond {
        rewrite_float_alias_value(lhs, aliases);
        rewrite_float_alias_value(rhs, aliases);
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

fn rewrite_float_alias_sources(kind: &mut MachineInstKind, aliases: &[Option<MachineReg>]) {
    match kind {
        MachineInstKind::FloatUnary { src, .. } => rewrite_float_alias_value(src, aliases),
        MachineInstKind::FloatBinary { lhs, rhs, .. }
        | MachineInstKind::FloatCompare { lhs, rhs, .. } => {
            rewrite_float_alias_value(lhs, aliases);
            rewrite_float_alias_value(rhs, aliases);
        }
        MachineInstKind::Store { width, src, .. }
            if matches!(
                width,
                super::MachineMemWidth::U32 | super::MachineMemWidth::U64
            ) =>
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

fn visit_branch_cond_values(cond: &MachineBranchCond, mut f: impl FnMut(&MachineValue)) {
    match cond {
        MachineBranchCond::Value(value) => f(value),
        MachineBranchCond::IntCompare { lhs, rhs, .. }
        | MachineBranchCond::FloatCompare { lhs, rhs, .. } => {
            f(lhs);
            f(rhs);
        }
    }
}

fn replace_branch_cond_value(cond: &mut MachineBranchCond, old: MachineReg, new_val: MachineValue) {
    match cond {
        MachineBranchCond::Value(value) => {
            try_replace(value, old, new_val);
        }
        MachineBranchCond::IntCompare { lhs, rhs, .. }
        | MachineBranchCond::FloatCompare { lhs, rhs, .. } => {
            if !try_replace(lhs, old, new_val) {
                try_replace(rhs, old, new_val);
            }
        }
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

fn convert_src_accepts_fp(op: super::MachineConvertOp) -> bool {
    matches!(
        op,
        super::MachineConvertOp::I32TruncF32S
            | super::MachineConvertOp::I32TruncF32U
            | super::MachineConvertOp::I32TruncF64S
            | super::MachineConvertOp::I32TruncF64U
            | super::MachineConvertOp::I64TruncF32S
            | super::MachineConvertOp::I64TruncF32U
            | super::MachineConvertOp::I64TruncF64S
            | super::MachineConvertOp::I64TruncF64U
            | super::MachineConvertOp::I32TruncSatF32S
            | super::MachineConvertOp::I32TruncSatF32U
            | super::MachineConvertOp::I32TruncSatF64S
            | super::MachineConvertOp::I32TruncSatF64U
            | super::MachineConvertOp::I64TruncSatF32S
            | super::MachineConvertOp::I64TruncSatF32U
            | super::MachineConvertOp::I64TruncSatF64S
            | super::MachineConvertOp::I64TruncSatF64U
            | super::MachineConvertOp::F32DemoteF64
            | super::MachineConvertOp::F64PromoteF32
            | super::MachineConvertOp::I32ReinterpretF32
            | super::MachineConvertOp::I64ReinterpretF64
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

fn same_reg_bank(lhs: MachineReg, rhs: MachineReg, first_fp_reg: u16) -> bool {
    (lhs.0 >= first_fp_reg) == (rhs.0 >= first_fp_reg)
}

fn is_transient_reg(
    reg: MachineReg,
    first_gp_transient: u16,
    first_fp_reg: u16,
    fp_transient_count: u16,
) -> bool {
    if reg.0 < first_fp_reg {
        return reg.0 >= first_gp_transient;
    }
    reg.0 < first_fp_reg.saturating_add(fp_transient_count)
}

fn reg_move_rewrite_supported(dst: MachineReg, src: MachineReg, first_fp_reg: u16) -> bool {
    let dst_is_fp = dst.0 >= first_fp_reg;
    let src_is_fp = src.0 >= first_fp_reg;
    !dst_is_fp || src_is_fp
}

fn move_rewrite_supported(dst: MachineReg, src: MachineValue, first_fp_reg: u16) -> bool {
    match src {
        MachineValue::Reg(src_reg) => reg_move_rewrite_supported(dst, src_reg, first_fp_reg),
        MachineValue::Imm64(_) => dst.0 < first_fp_reg,
    }
}

fn rewrite_move_storage_type(
    dst: MachineReg,
    src: MachineValue,
    ty: MachineStorageType,
    first_fp_reg: u16,
) -> Option<MachineStorageType> {
    if (dst.0 >= first_fp_reg) != ty.is_fp() {
        return None;
    }
    move_rewrite_supported(dst, src, first_fp_reg).then_some(ty)
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

// ---------------------------------------------------------------------------
// Compare-and-branch fusion
// ---------------------------------------------------------------------------

/// Rewrite `IntCompare/FloatCompare { dst } + Branch { Value(Reg(dst)) }`
/// into `Branch { IntCompare/FloatCompare { ... } }` when the compare result
/// register is provably dead in both successor blocks.
///
/// This is a cross-block pass: it reads successor blocks to check liveness,
/// so it must run after the per-block optimizations are done.
fn fuse_compare_branch(blocks: &mut [MachineBlock], gp_reg_width: u8) {
    for idx in 0..blocks.len() {
        // Check the last op and the terminator of this block.
        let last_op = match blocks[idx].ops.last() {
            Some(op) => op,
            None => continue,
        };

        // Terminator must be Branch { cond: Value(Reg(cond_reg)) }.
        let (cond_reg, then_target, else_target) = match &blocks[idx].terminator {
            MachineTerminator::Branch {
                cond: MachineBranchCond::Value(MachineValue::Reg(r)),
                then_edge,
                else_edge,
            } => (*r, then_edge.target, else_edge.target),
            _ => continue,
        };

        // Build the fused branch condition, or skip.
        //
        // Only IntCompare is fused here. FloatCompare is NOT fused because
        // on x86_64 Wasm float comparisons require multi-instruction NaN
        // handling (SETCC+SETNP+AND) that cannot be expressed as a single
        // conditional branch. ARM64's FCMP condition codes handle NaN
        // correctly with a single B.cond, but since this is a shared pass
        // it must be safe for all backends.
        let fused_cond = match &last_op.kind {
            MachineInstKind::IntCompare {
                width,
                kind,
                sign,
                dst,
                lhs,
                rhs,
            } if *width == super::MachineIntWidth::I64 && gp_reg_width == 4 => continue,
            MachineInstKind::IntCompare {
                width,
                kind,
                sign,
                dst,
                lhs,
                rhs,
            } if *dst == cond_reg => MachineBranchCond::IntCompare {
                width: *width,
                kind: *kind,
                sign: *sign,
                lhs: *lhs,
                rhs: *rhs,
            },
            _ => continue,
        };

        // Reject if any edge passes dst as an arg.
        if term_edge_uses_value(&blocks[idx].terminator, cond_reg) {
            continue;
        }

        // Reject if dst is live-in to either successor.
        if !reg_dead_at_block_entry(blocks, then_target, cond_reg) {
            continue;
        }
        if !reg_dead_at_block_entry(blocks, else_target, cond_reg) {
            continue;
        }

        // Safe to fuse: remove the compare op and rewrite the terminator.
        blocks[idx].ops.pop();
        if let MachineTerminator::Branch { cond, .. } = &mut blocks[idx].terminator {
            *cond = fused_cond;
        }
    }
}

/// Check whether any edge arg in the terminator references `reg`.
fn term_edge_uses_value(term: &MachineTerminator, reg: MachineReg) -> bool {
    let check = |e: &MachineEdge| e.args.iter().any(|a| value_is_reg(a, reg));
    match term {
        MachineTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } => check(then_edge) || check(else_edge),
        MachineTerminator::Jump(edge) => check(edge),
        MachineTerminator::JumpTable { entries, .. } => entries.iter().any(|e| check(e)),
        _ => false,
    }
}

/// Returns true if `reg` is provably dead at the beginning of `target`:
/// either the block defines it before any use, the block has it as a
/// parameter, or the block never touches it.
pub fn reg_dead_at_block_entry(
    blocks: &[MachineBlock],
    target: MachineBlockId,
    reg: MachineReg,
) -> bool {
    let Some(block) = blocks.get(target.as_usize()) else {
        return false;
    };
    // If the target has reg as a param, it will be defined by the edge.
    if block.params.iter().any(|p| p.reg == reg) {
        return true;
    }
    // Scan ops: defined before used → dead at entry.
    for op in &block.ops {
        if inst_defines(&op.kind, reg) {
            return true;
        }
        if count_value_uses(&op.kind, reg) > 0 {
            return false;
        }
    }
    // Reached terminator without touching reg.
    !terminator_uses_reg(&block.terminator, reg)
}
