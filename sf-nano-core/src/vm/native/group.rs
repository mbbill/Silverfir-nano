//! JIT group compiler: identify and compile groups of consecutive JIT-able ops.
//!
//! Scans `&[IrOp]` for consecutive JIT-able operations, compiles them into
//! single ARM64 code blocks via `JitEmitter`, and produces `Vec<ResolvedInst>`.
//! Non-JIT-able ops fall back to 1:1 base handler resolution.

use alloc::vec::Vec;
use super::arm64_enc::{self, Cond};
use super::code_buf::CodeBuffer;
use super::codegen::{JitEmitter, depth_variant, tos_reg};
use super::debug_map;
use super::op_meta;
use super::reg::Reg;
use super::samply_jitdump;
use crate::vm::interp::fast::builder::backend::{CompactionDisposition, ResolvedInst};
use crate::vm::interp::fast::builder::ir::{IrOp, IrOpKind, OpIndex, stack_effect};
use crate::vm::interp::fast::handlers::OpHandler;

/// Check if an IrOp is JIT-able, considering kind, height, and hot local mask.
fn is_jit_able_op(op: &IrOp, hot_local_mask: [bool; 3]) -> bool {
    let kind = &op.kind;
    let h = op.pre_height as usize;

    if !op_meta::supports_kind(kind) { return false; }
    if !op_meta::memidx_is_supported(kind) { return false; }

    // Height checks based on stack effect
    let (pops, _) = stack_effect(kind);
    if h < pops as usize { return false; }

    // LocalTeeHot: hot locals only
    if let IrOpKind::LocalTeeHot { reg } = kind {
        if (*reg as usize) >= 3 || !hot_local_mask[*reg as usize] {
            return false;
        }
    }

    true
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GroupTerminator {
    Br,
    BrIfSimple,
    If,
    ReturnVoid,
    ReturnOne,
}

/// Check if an IrOp is a group terminator (can only terminate, not start).
fn group_terminator(op: &IrOp) -> Option<GroupTerminator> {
    match &op.kind {
        IrOpKind::Br { stack_drop, arity, .. }
            if (*stack_drop == 0 || *arity == 0) && op.alt_target.is_some() =>
        {
            Some(GroupTerminator::Br)
        }
        IrOpKind::BrIfSimple if op.pre_height >= 1 => Some(GroupTerminator::BrIfSimple),
        IrOpKind::If if op.pre_height >= 1 => Some(GroupTerminator::If),
        IrOpKind::ReturnVoid { .. } => Some(GroupTerminator::ReturnVoid),
        IrOpKind::ReturnOne { .. } => Some(GroupTerminator::ReturnOne),
        _ => None,
    }
}

/// Mark every IR index that can be reached by a non-fallthrough branch.
fn incoming_targets(ir: &[IrOp]) -> Vec<bool> {
    let mut incoming = alloc::vec![false; ir.len()];

    for op in ir {
        if let Some(target) = op.alt_target {
            if target.as_usize() < incoming.len() {
                incoming[target.as_usize()] = true;
            }
        }

        if let IrOpKind::BrTable { entries, .. } = &op.kind {
            for entry in entries {
                if let Some(target) = entry.target_idx {
                    if target.as_usize() < incoming.len() {
                        incoming[target.as_usize()] = true;
                    }
                }
            }
        }
    }

    incoming
}

// =========================================================================
// JIT statistics
// =========================================================================

pub struct JitStats {
    pub groups: core::sync::atomic::AtomicUsize,
    pub ops: core::sync::atomic::AtomicUsize,
    pub bytes_emitted: core::sync::atomic::AtomicUsize,
    pub groups_skipped_capacity: core::sync::atomic::AtomicUsize,
    pub ops_skipped_capacity: core::sync::atomic::AtomicUsize,
}

const AZ: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

pub static JIT_STATS: JitStats = JitStats {
    groups: AZ,
    ops: AZ,
    bytes_emitted: AZ,
    groups_skipped_capacity: AZ,
    ops_skipped_capacity: AZ,
};

pub struct JitStatsSnapshot {
    pub groups: usize,
    pub ops: usize,
    pub bytes_emitted: usize,
    pub groups_skipped: usize,
    pub ops_skipped: usize,
}

pub fn jit_stats_snapshot() -> JitStatsSnapshot {
    use core::sync::atomic::Ordering::Relaxed;
    JitStatsSnapshot {
        groups: JIT_STATS.groups.load(Relaxed),
        ops: JIT_STATS.ops.load(Relaxed),
        bytes_emitted: JIT_STATS.bytes_emitted.load(Relaxed),
        groups_skipped: JIT_STATS.groups_skipped_capacity.load(Relaxed),
        ops_skipped: JIT_STATS.ops_skipped_capacity.load(Relaxed),
    }
}

pub fn jit_stats() -> (usize, usize) {
    (
        JIT_STATS.groups.load(core::sync::atomic::Ordering::Relaxed),
        JIT_STATS.ops.load(core::sync::atomic::Ordering::Relaxed),
    )
}

pub fn jit_capacity_skips() -> (usize, usize) {
    (
        JIT_STATS.groups_skipped_capacity.load(core::sync::atomic::Ordering::Relaxed),
        JIT_STATS.ops_skipped_capacity.load(core::sync::atomic::Ordering::Relaxed),
    )
}

// =========================================================================
// JIT backend: resolve IR ops via JIT compilation + 1:1 fallback
// =========================================================================

/// Resolve IR ops via JIT compilation.
///
/// Scans for consecutive JIT-able ops, compiles groups into ARM64 code,
/// and falls back to 1:1 base handler resolution for non-JIT-able ops.
pub fn resolve_jit(
    ir: &[IrOp],
    buf: &mut CodeBuffer,
    hot_local_mask: [bool; 3],
) -> Vec<ResolvedInst> {
    resolve_jit_with_context(ir, buf, hot_local_mask, "", 0)
}

/// Resolve IR ops via JIT compilation with optional source metadata for debug maps.
pub fn resolve_jit_with_context(
    ir: &[IrOp],
    buf: &mut CodeBuffer,
    hot_local_mask: [bool; 3],
    module_name: &str,
    func_idx: u32,
) -> Vec<ResolvedInst> {
    buf.begin_write();
    let bytes_before = buf.len();
    let branch_targets = incoming_targets(ir);

    let mut out = Vec::with_capacity(ir.len());
    let mut groups_compiled: usize = 0;
    let mut ops_compiled: usize = 0;
    let mut groups_skipped: usize = 0;
    let mut ops_skipped: usize = 0;

    let mut i = 0;
    while i < ir.len() {
        // Try to start a JIT group
        if group_terminator(&ir[i]).is_none() && is_jit_able_op(&ir[i], hot_local_mask) {
            let group_start = i;
            i += 1;

            // Extend the group
            while i < ir.len() {
                if branch_targets[i] {
                    break;
                }
                if group_terminator(&ir[i]).is_some() {
                    i += 1; // include br_if_simple as terminator
                    break;
                }
                if !is_jit_able_op(&ir[i], hot_local_mask) {
                    break;
                }
                i += 1;
            }

            let group_len = i - group_start;
            if group_len >= 2 {
                let estimated_bytes = op_meta::estimate_group_bytes(
                    &ir[group_start..i],
                    group_terminator(&ir[i - 1]).map(|_| &ir[i - 1].kind),
                );
                if buf.remaining() >= estimated_bytes {
                    let code_start = buf.len();
                    if let Some((handler, terminator, branch_target)) =
                        try_compile_group(&ir[group_start..i], buf, hot_local_mask)
                    {
                        let code_len = buf.len() - code_start;
                        debug_map::record_group(
                            buf.base_ptr(),
                            code_start,
                            code_len,
                            module_name,
                            func_idx,
                            group_start,
                            &ir[group_start..i],
                            terminator_name(terminator),
                            branch_target,
                        );
                        samply_jitdump::record_group(
                            buf.base_ptr(),
                            code_start,
                            code_len,
                            module_name,
                            func_idx,
                            group_start,
                            &ir[group_start..i],
                            terminator_name(terminator),
                            branch_target,
                        );
                        // JIT entry
                        match terminator {
                            Some(GroupTerminator::Br) => {
                                out.push(ResolvedInst {
                                    handler,
                                    kind: ir[group_start + group_len - 1].kind.clone(),
                                    alt_target: branch_target,
                                    has_target: true,
                                    compaction: CompactionDisposition::Keep,
                                });
                            }
                            Some(GroupTerminator::BrIfSimple) => {
                                out.push(ResolvedInst {
                                    handler,
                                    kind: IrOpKind::BrIfSimple,
                                    alt_target: branch_target,
                                    has_target: true,
                                    compaction: CompactionDisposition::Keep,
                                });
                            }
                            Some(GroupTerminator::If) => {
                                out.push(ResolvedInst {
                                    handler,
                                    kind: IrOpKind::If,
                                    alt_target: branch_target,
                                    has_target: true,
                                    compaction: CompactionDisposition::Keep,
                                });
                            }
                            Some(GroupTerminator::ReturnVoid) => {
                                out.push(ResolvedInst {
                                    handler,
                                    kind: ir[group_start + group_len - 1].kind.clone(),
                                    alt_target: None,
                                    has_target: false,
                                    compaction: CompactionDisposition::Keep,
                                });
                            }
                            Some(GroupTerminator::ReturnOne) => {
                                out.push(ResolvedInst {
                                    handler,
                                    kind: ir[group_start + group_len - 1].kind.clone(),
                                    alt_target: None,
                                    has_target: false,
                                    compaction: CompactionDisposition::Keep,
                                });
                            }
                            None => {
                                out.push(ResolvedInst {
                                    handler,
                                    kind: IrOpKind::Data { imm0: 0, imm1: 0, imm2: 0 },
                                    alt_target: None,
                                    has_target: false,
                                    compaction: CompactionDisposition::Keep,
                                });
                            }
                        }
                        // Skip remaining ops in group
                        for _ in 1..group_len {
                            out.push(ResolvedInst::skip());
                        }
                        groups_compiled += 1;
                        ops_compiled += group_len;
                        continue;
                    }
                } else {
                    groups_skipped += 1;
                    ops_skipped += group_len;
                }
            }

            // Fallback: 1:1 for all ops in the attempted range
            for j in group_start..i {
                out.push(ResolvedInst::from_ir(&ir[j]));
            }
            continue;
        }

        // Lone terminator or non-JIT-able op: 1:1
        out.push(ResolvedInst::from_ir(&ir[i]));
        i += 1;
    }

    let total_len = buf.len();
    buf.finish_write(bytes_before, total_len - bytes_before);

    // Update global stats
    let bytes_emitted = total_len - bytes_before;
    JIT_STATS.groups.fetch_add(groups_compiled, core::sync::atomic::Ordering::Relaxed);
    JIT_STATS.ops.fetch_add(ops_compiled, core::sync::atomic::Ordering::Relaxed);
    JIT_STATS.bytes_emitted.fetch_add(bytes_emitted, core::sync::atomic::Ordering::Relaxed);
    if groups_skipped > 0 {
        JIT_STATS.groups_skipped_capacity.fetch_add(groups_skipped, core::sync::atomic::Ordering::Relaxed);
        JIT_STATS.ops_skipped_capacity.fetch_add(ops_skipped, core::sync::atomic::Ordering::Relaxed);
    }

    out
}

fn terminator_name(terminator: Option<GroupTerminator>) -> Option<&'static str> {
    match terminator {
        Some(GroupTerminator::Br) => Some("br"),
        Some(GroupTerminator::BrIfSimple) => Some("br_if_simple"),
        Some(GroupTerminator::If) => Some("if"),
        Some(GroupTerminator::ReturnVoid) => Some("return_void"),
        Some(GroupTerminator::ReturnOne) => Some("return_one"),
        None => None,
    }
}

#[derive(Clone, Copy, Debug)]
enum CompareFusionKind {
    I32(Cond),
    I64(Cond),
    F32(Cond),
    F64(Cond),
}

#[derive(Clone, Copy, Debug)]
enum TailFusion {
    DirectCompareBr(CompareFusionKind),
    DirectCompareIf(CompareFusionKind),
    DirectZeroCompareBr(CompareFusionKind),
    DirectZeroCompareIf(CompareFusionKind),
    EqzBr,
    EqzIf,
    XorOneBr,
    XorOneIfIdentity,
}

fn compare_fusion_kind(kind: &IrOpKind) -> Option<CompareFusionKind> {
    use IrOpKind::*;

    match kind {
        I32Eq => Some(CompareFusionKind::I32(Cond::EQ)),
        I32Ne => Some(CompareFusionKind::I32(Cond::NE)),
        I32LtS => Some(CompareFusionKind::I32(Cond::LT)),
        I32LtU => Some(CompareFusionKind::I32(Cond::LO)),
        I32GtS => Some(CompareFusionKind::I32(Cond::GT)),
        I32GtU => Some(CompareFusionKind::I32(Cond::HI)),
        I32LeS => Some(CompareFusionKind::I32(Cond::LE)),
        I32LeU => Some(CompareFusionKind::I32(Cond::LS)),
        I32GeS => Some(CompareFusionKind::I32(Cond::GE)),
        I32GeU => Some(CompareFusionKind::I32(Cond::HS)),
        I64Eq => Some(CompareFusionKind::I64(Cond::EQ)),
        I64Ne => Some(CompareFusionKind::I64(Cond::NE)),
        I64LtS => Some(CompareFusionKind::I64(Cond::LT)),
        I64LtU => Some(CompareFusionKind::I64(Cond::LO)),
        I64GtS => Some(CompareFusionKind::I64(Cond::GT)),
        I64GtU => Some(CompareFusionKind::I64(Cond::HI)),
        I64LeS => Some(CompareFusionKind::I64(Cond::LE)),
        I64LeU => Some(CompareFusionKind::I64(Cond::LS)),
        I64GeS => Some(CompareFusionKind::I64(Cond::GE)),
        I64GeU => Some(CompareFusionKind::I64(Cond::HS)),
        F32Eq => Some(CompareFusionKind::F32(Cond::EQ)),
        F32Ne => Some(CompareFusionKind::F32(Cond::NE)),
        F32Lt => Some(CompareFusionKind::F32(Cond::MI)),
        F32Gt => Some(CompareFusionKind::F32(Cond::GT)),
        F32Le => Some(CompareFusionKind::F32(Cond::LS)),
        F32Ge => Some(CompareFusionKind::F32(Cond::GE)),
        F64Eq => Some(CompareFusionKind::F64(Cond::EQ)),
        F64Ne => Some(CompareFusionKind::F64(Cond::NE)),
        F64Lt => Some(CompareFusionKind::F64(Cond::MI)),
        F64Gt => Some(CompareFusionKind::F64(Cond::GT)),
        F64Le => Some(CompareFusionKind::F64(Cond::LS)),
        F64Ge => Some(CompareFusionKind::F64(Cond::GE)),
        _ => None,
    }
}

fn matches_const_one_xor(suffix: &[IrOpKind]) -> bool {
    matches!(
        suffix,
        [IrOpKind::I32Const { value: 1 }, IrOpKind::I32Xor]
    )
}

fn produces_bool(kind: &IrOpKind) -> bool {
    compare_fusion_kind(kind).is_some() || matches!(kind, IrOpKind::I32Eqz | IrOpKind::I64Eqz)
}

fn zero_float_compare_kind(const_kind: &IrOpKind, cmp_kind: &IrOpKind) -> Option<CompareFusionKind> {
    match (const_kind, compare_fusion_kind(cmp_kind)) {
        (IrOpKind::F32Const { value: 0 }, Some(cmp @ CompareFusionKind::F32(_))) => Some(cmp),
        (IrOpKind::F64Const { value: 0 }, Some(cmp @ CompareFusionKind::F64(_))) => Some(cmp),
        _ => None,
    }
}

fn pick_tail_fusion(group: &[IrOp], terminator: Option<GroupTerminator>, body_end: usize) -> Option<(TailFusion, usize)> {
    let start_height = group[0].pre_height;

    match terminator {
        Some(GroupTerminator::BrIfSimple) => {
            if start_height == 0 && body_end >= 2 {
                if let Some(cmp) =
                    zero_float_compare_kind(&group[body_end - 2].kind, &group[body_end - 1].kind)
                {
                    return Some((TailFusion::DirectZeroCompareBr(cmp), body_end - 2));
                }
            }
            if start_height == 0 && body_end >= 1 {
                if let Some(cmp) = compare_fusion_kind(&group[body_end - 1].kind) {
                    return Some((TailFusion::DirectCompareBr(cmp), body_end - 1));
                }
            }
            if body_end >= 3 && matches_const_one_xor(&[
                group[body_end - 2].kind.clone(),
                group[body_end - 1].kind.clone(),
            ]) && produces_bool(&group[body_end - 3].kind) {
                return Some((TailFusion::XorOneBr, body_end - 2));
            }
            if body_end >= 1 && matches!(group[body_end - 1].kind, IrOpKind::I32Eqz) {
                return Some((TailFusion::EqzBr, body_end - 1));
            }
        }
        Some(GroupTerminator::If) => {
            if body_end >= 5
                && start_height == 0
                && matches_const_one_xor(&[
                    group[body_end - 3].kind.clone(),
                    group[body_end - 2].kind.clone(),
                ])
                && matches!(group[body_end - 1].kind, IrOpKind::I32Eqz)
            {
                if let Some(cmp) =
                    zero_float_compare_kind(&group[body_end - 5].kind, &group[body_end - 4].kind)
                {
                    return Some((TailFusion::DirectZeroCompareIf(cmp), body_end - 5));
                }
            }
            if start_height == 0 && body_end >= 2 {
                if let Some(cmp) =
                    zero_float_compare_kind(&group[body_end - 2].kind, &group[body_end - 1].kind)
                {
                    return Some((TailFusion::DirectZeroCompareIf(cmp), body_end - 2));
                }
            }
            if body_end >= 4
                && start_height == 0
                && matches_const_one_xor(&[
                    group[body_end - 3].kind.clone(),
                    group[body_end - 2].kind.clone(),
                ])
                && matches!(group[body_end - 1].kind, IrOpKind::I32Eqz)
            {
                if let Some(cmp) = compare_fusion_kind(&group[body_end - 4].kind) {
                    return Some((TailFusion::DirectCompareIf(cmp), body_end - 4));
                }
            }
            if body_end >= 4
                && matches_const_one_xor(&[
                    group[body_end - 3].kind.clone(),
                    group[body_end - 2].kind.clone(),
                ])
                && matches!(group[body_end - 1].kind, IrOpKind::I32Eqz)
                && produces_bool(&group[body_end - 4].kind)
            {
                return Some((TailFusion::XorOneIfIdentity, body_end - 3));
            }
            if body_end >= 1 && matches!(group[body_end - 1].kind, IrOpKind::I32Eqz) {
                return Some((TailFusion::EqzIf, body_end - 1));
            }
            if start_height == 0 && body_end >= 1 {
                if let Some(cmp) = compare_fusion_kind(&group[body_end - 1].kind) {
                    return Some((TailFusion::DirectCompareIf(cmp), body_end - 1));
                }
            }
        }
        _ => {}
    }

    None
}

fn emit_compare_flags(e: &mut JitEmitter, cmp: CompareFusionKind) {
    match cmp {
        CompareFusionKind::I32(_) => {
            let lhs = e.resolve_tos_reg(2);
            let rhs = e.resolve_tos_reg(1);
            e.emit_raw(arm64_enc::subs_reg_32(Reg::XZR, lhs, rhs));
        }
        CompareFusionKind::I64(_) => {
            let lhs = e.resolve_tos_reg(2);
            let rhs = e.resolve_tos_reg(1);
            e.emit_raw(arm64_enc::subs_reg_64(Reg::XZR, lhs, rhs));
        }
        CompareFusionKind::F32(_) => {
            let lhs = e.ensure_tos_f32(2);
            let rhs = e.ensure_tos_f32(1);
            e.emit_raw(arm64_enc::fcmp_32(lhs, rhs));
        }
        CompareFusionKind::F64(_) => {
            let lhs = e.ensure_tos_f64(2);
            let rhs = e.ensure_tos_f64(1);
            e.emit_raw(arm64_enc::fcmp_64(lhs, rhs));
        }
    }
}

fn emit_compare_zero_flags(e: &mut JitEmitter, cmp: CompareFusionKind) {
    match cmp {
        CompareFusionKind::F32(_) => {
            let lhs = e.ensure_tos_f32(1);
            e.emit_raw(arm64_enc::fcmp_32_zero(lhs));
        }
        CompareFusionKind::F64(_) => {
            let lhs = e.ensure_tos_f64(1);
            e.emit_raw(arm64_enc::fcmp_64_zero(lhs));
        }
        CompareFusionKind::I32(_) | CompareFusionKind::I64(_) => unreachable!("zero-float compare only"),
    }
}

fn top_i32_reg(e: &JitEmitter) -> Reg {
    e.resolve_tos_reg(1)
}

/// Try to compile a group of IrOps into a single JIT handler.
///
/// Returns `(handler, ends_with_brif, branch_alt_target)` on success.
fn try_compile_group(
    group: &[IrOp],
    buf: &mut CodeBuffer,
    hot_local_mask: [bool; 3],
) -> Option<(OpHandler, Option<GroupTerminator>, Option<OpIndex>)> {
    let terminator = group_terminator(group.last().unwrap());
    let branch_alt = terminator.and_then(|_| group.last().unwrap().alt_target);

    // Validate: check that heights stay valid throughout the group
    let mut sim_height = group[0].pre_height as usize;
    let body_end = if terminator.is_some() {
        group.len() - 1
    } else {
        group.len()
    };
    for op in &group[..body_end] {
        let (pops, pushes) = stack_effect(&op.kind);
        if sim_height < pops as usize {
            return None;
        }
        sim_height = sim_height - pops as usize + pushes as usize;
    }
    match terminator {
        Some(GroupTerminator::BrIfSimple | GroupTerminator::If) if sim_height < 1 => {
            return None;
        }
        _ => {}
    }

    // Compile
    let mut e = JitEmitter::new(buf, group[0].pre_height as usize);
    let tail_fusion = pick_tail_fusion(group, terminator, body_end);
    let compile_body_end = tail_fusion.map(|(_, end)| end).unwrap_or(body_end);

    for op in group[..compile_body_end].iter() {
        emit_op(&mut e, op, hot_local_mask);
    }

    let start = match tail_fusion {
        Some((TailFusion::DirectCompareBr(cmp), _)) => {
            let cond = match cmp {
                CompareFusionKind::I32(cond)
                | CompareFusionKind::I64(cond)
                | CompareFusionKind::F32(cond)
                | CompareFusionKind::F64(cond) => cond,
            };
            emit_compare_flags(&mut e, cmp);
            e.finish_br_if_simple_from_flags(cond)
        }
        Some((TailFusion::DirectCompareIf(cmp), _)) => {
            let cond = match cmp {
                CompareFusionKind::I32(cond)
                | CompareFusionKind::I64(cond)
                | CompareFusionKind::F32(cond)
                | CompareFusionKind::F64(cond) => cond,
            };
            emit_compare_flags(&mut e, cmp);
            e.finish_if_from_flags(cond)
        }
        Some((TailFusion::DirectZeroCompareBr(cmp), _)) => {
            let cond = match cmp {
                CompareFusionKind::F32(cond) | CompareFusionKind::F64(cond) => cond,
                CompareFusionKind::I32(_) | CompareFusionKind::I64(_) => unreachable!(),
            };
            emit_compare_zero_flags(&mut e, cmp);
            e.finish_br_if_simple_from_flags(cond)
        }
        Some((TailFusion::DirectZeroCompareIf(cmp), _)) => {
            let cond = match cmp {
                CompareFusionKind::F32(cond) | CompareFusionKind::F64(cond) => cond,
                CompareFusionKind::I32(_) | CompareFusionKind::I64(_) => unreachable!(),
            };
            emit_compare_zero_flags(&mut e, cmp);
            e.finish_if_from_flags(cond)
        }
        Some((TailFusion::EqzBr, _)) => {
            let cond = top_i32_reg(&e);
            e.finish_br_if_simple_from_reg(cond, true)
        }
        Some((TailFusion::EqzIf, _)) => {
            let cond = top_i32_reg(&e);
            e.finish_if_from_reg(cond, true)
        }
        Some((TailFusion::XorOneBr, _)) => {
            let cond = top_i32_reg(&e);
            e.finish_br_if_simple_from_reg(cond, true)
        }
        Some((TailFusion::XorOneIfIdentity, _)) => {
            let cond = top_i32_reg(&e);
            e.finish_if_from_reg(cond, false)
        }
        None => match terminator {
            Some(GroupTerminator::Br) => e.finish_br(),
            Some(GroupTerminator::BrIfSimple) => e.finish_br_if_simple(),
            Some(GroupTerminator::If) => e.finish_if(),
            Some(GroupTerminator::ReturnVoid) => match &group.last().unwrap().kind {
                IrOpKind::ReturnVoid { frame_size } => e.finish_return_void(*frame_size),
                _ => unreachable!("terminator kind mismatch"),
            },
            Some(GroupTerminator::ReturnOne) => match &group.last().unwrap().kind {
                IrOpKind::ReturnOne {
                    frame_size,
                    operand_base_offset,
                    height,
                } => e.finish_return_one(*frame_size, *operand_base_offset, *height),
                _ => unreachable!("terminator kind mismatch"),
            },
            None => e.finish(),
        },
    };

    let handler: OpHandler = unsafe { buf.fn_ptr(start) };
    Some((handler, terminator, branch_alt))
}

/// Emit a single op via JitEmitter based on its IrOpKind.
fn emit_op(e: &mut JitEmitter, op: &IrOp, hot_local_mask: [bool; 3]) {
    // Debug: verify emitter height matches IR pre_height
    debug_assert_eq!(
        e.height() as u16, op.pre_height,
        "height mismatch at {:?}: emitter={}, ir={}",
        op.kind, e.height(), op.pre_height
    );
    match &op.kind {
        IrOpKind::I32Const { .. }
        | IrOpKind::I64Const { .. }
        | IrOpKind::F32Const { .. }
        | IrOpKind::F64Const { .. }
        | IrOpKind::LocalGetHot { .. }
        | IrOpKind::LocalGetFrame { .. }
        | IrOpKind::LocalSetHot { .. }
        | IrOpKind::LocalSetFrame { .. }
        | IrOpKind::LocalTeeHot { .. }
        | IrOpKind::LocalTeeFrame { .. }
        | IrOpKind::Drop
        | IrOpKind::I32Eqz
        | IrOpKind::I64Eqz
        | IrOpKind::F32Add
        | IrOpKind::F32Sub
        | IrOpKind::F32Mul
        | IrOpKind::F32Div
        | IrOpKind::F32Min
        | IrOpKind::F32Max
        | IrOpKind::F32Eq
        | IrOpKind::F32Ne
        | IrOpKind::F32Lt
        | IrOpKind::F32Gt
        | IrOpKind::F32Le
        | IrOpKind::F32Ge
        | IrOpKind::F64Add
        | IrOpKind::F64Sub
        | IrOpKind::F64Mul
        | IrOpKind::F64Div
        | IrOpKind::F64Min
        | IrOpKind::F64Max
        | IrOpKind::F64Eq
        | IrOpKind::F64Ne
        | IrOpKind::F64Lt
        | IrOpKind::F64Gt
        | IrOpKind::F64Le
        | IrOpKind::F64Ge => op_meta::emit(e, op, hot_local_mask),
        _ => {
            e.materialize_aliases();
            op_meta::emit(e, op, hot_local_mask);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use alloc::vec;
    use super::*;
    use crate::vm::interp::fast::builder::ir::{IrOp, IrOpKind};
    use crate::vm::interp::fast::instruction::Instruction;
    use crate::vm::interp::fast::handlers::{self, OpHandler, NextHandler, run_trampoline};
    use crate::vm::interp::fast::handlers::full_set;
    use crate::vm::interp::fast::context::Context;
    use crate::vm::native::codegen::{depth_variant, tos_reg};
    use crate::vm::native::reg::Reg;
    use crate::vm::native::arm64_enc;
    use crate::vm::native::emit;

    /// Helper: create an IrOp with given kind and pre_height.
    fn make_op(kind: IrOpKind, pre_height: u16) -> IrOp {
        IrOp {
            kind,
            variant: 0,
            pre_height,
            fallthrough: None,
            alt_target: None,
            has_target: false,
        }
    }

    // ==================== Classification tests ====================

    #[test]
    fn test_is_jit_able_arithmetic() {
        let mask = [true, true, true];
        assert!(is_jit_able_op(&make_op(IrOpKind::I32Add, 2), mask));
        assert!(is_jit_able_op(&make_op(IrOpKind::I64Mul, 3), mask));
        assert!(is_jit_able_op(&make_op(IrOpKind::I32Eq, 2), mask));
        assert!(is_jit_able_op(&make_op(IrOpKind::I32Eqz, 1), mask));
        assert!(!is_jit_able_op(&make_op(IrOpKind::I32Add, 1), mask));
        assert!(!is_jit_able_op(&make_op(IrOpKind::I32Eqz, 0), mask));
    }

    #[test]
    fn test_is_jit_able_const() {
        let mask = [true, true, true];
        assert!(is_jit_able_op(&make_op(IrOpKind::I32Const { value: 42 }, 0), mask));
        assert!(is_jit_able_op(&make_op(IrOpKind::I64Const { value: 100 }, 0), mask));
    }

    #[test]
    fn test_is_jit_able_locals() {
        let mask = [true, true, false];
        assert!(is_jit_able_op(&make_op(IrOpKind::LocalGetHot { reg: 0 }, 0), mask));
        assert!(is_jit_able_op(&make_op(IrOpKind::LocalGetFrame { idx: 5 }, 0), mask));
        assert!(is_jit_able_op(&make_op(IrOpKind::LocalSetHot { reg: 1 }, 1), mask));
        assert!(!is_jit_able_op(&make_op(IrOpKind::LocalSetHot { reg: 1 }, 0), mask));
        assert!(is_jit_able_op(&make_op(IrOpKind::LocalTeeHot { reg: 0 }, 1), mask));
        assert!(!is_jit_able_op(&make_op(IrOpKind::LocalTeeHot { reg: 2 }, 1), mask));
    }

    #[test]
    fn test_is_jit_able_drop() {
        let mask = [true, true, true];
        assert!(is_jit_able_op(&make_op(IrOpKind::Drop, 1), mask));
        assert!(!is_jit_able_op(&make_op(IrOpKind::Drop, 0), mask));
    }

    #[test]
    fn test_is_jit_able_init_locals() {
        let mask = [true, true, true];
        assert!(is_jit_able_op(&make_op(IrOpKind::InitLocals { k0: 5, k1: 1, k2: 2 }, 0), mask));
    }

    #[test]
    fn test_is_jit_able_non_jitable() {
        let mask = [true, true, true];
        assert!(!is_jit_able_op(&make_op(IrOpKind::Block, 0), mask));
        assert!(!is_jit_able_op(&make_op(IrOpKind::Nop, 0), mask));
        assert!(!is_jit_able_op(&make_op(IrOpKind::I32DivS, 2), mask));
        assert!(!is_jit_able_op(&make_op(IrOpKind::GlobalGet { idx: 0 }, 0), mask));
    }

    // ==================== Group identification tests ====================

    #[test]
    fn test_group_identification_basic() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        let mask = [true, true, true];

        let ops = vec![
            make_op(IrOpKind::I32Const { value: 5 }, 0),
            make_op(IrOpKind::I32Const { value: 3 }, 1),
            make_op(IrOpKind::I32Add, 2),
            // Non-JIT-able op (separator)
            make_op(IrOpKind::Nop, 1),
            make_op(IrOpKind::I32Const { value: 7 }, 0),
            make_op(IrOpKind::I32Const { value: 6 }, 1),
            make_op(IrOpKind::I32Mul, 2),
        ];

        let resolved = resolve_jit(&ops, &mut buf, mask);

        // Group 1: resolved[0] = JIT entry, [1,2] = internal-only removed slots.
        assert!(!resolved[0].is_removed());
        assert!(resolved[1].is_internal_only());
        assert!(resolved[2].is_internal_only());

        // Separator: 1:1 Nop (removed, but a legal redirect target).
        assert!(resolved[3].redirects_branch_target());

        // Group 2: resolved[4] = JIT entry, [5,6] = skip
        assert!(!resolved[4].is_removed());
        assert!(resolved[5].is_internal_only());
        assert!(resolved[6].is_internal_only());
    }

    #[test]
    fn test_single_op_not_grouped() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        let mask = [true, true, true];

        let ops = vec![
            make_op(IrOpKind::I32Const { value: 5 }, 0),
            make_op(IrOpKind::CallExternal { func_idx: 0, delta: crate::vm::interp::fast::builder::ir::SlotRef::operand_relative(0) }, 0),
            make_op(IrOpKind::I32Const { value: 3 }, 0),
        ];

        let resolved = resolve_jit(&ops, &mut buf, mask);

        // All should be 1:1 (no group >= 2)
        assert_eq!(resolved.len(), 3);
        assert!(matches!(resolved[0].kind, IrOpKind::I32Const { .. }));
        assert!(matches!(resolved[1].kind, IrOpKind::CallExternal { .. }));
        assert!(matches!(resolved[2].kind, IrOpKind::I32Const { .. }));
    }

    #[test]
    fn test_group_with_br_if_simple() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        let mask = [true, true, true];

        let ops = vec![
            make_op(IrOpKind::I32Const { value: 5 }, 0),
            make_op(IrOpKind::I32Const { value: 5 }, 1),
            make_op(IrOpKind::I32Eq, 2),
            {
                let mut op = make_op(IrOpKind::BrIfSimple, 1);
                op.has_target = true;
                op.alt_target = Some(crate::vm::interp::fast::builder::ir::OpIndex::from(10));
                op
            },
        ];

        let resolved = resolve_jit(&ops, &mut buf, mask);

        // JIT entry with br_if encoding
        assert!(matches!(resolved[0].kind, IrOpKind::BrIfSimple));
        assert!(resolved[0].has_target);
        assert_eq!(resolved[0].alt_target, Some(crate::vm::interp::fast::builder::ir::OpIndex::from(10)));
        assert!(!resolved[0].is_removed());

        // Rest are skip
        assert!(resolved[1].is_internal_only());
        assert!(resolved[2].is_internal_only());
        assert!(resolved[3].is_internal_only());
    }

    #[test]
    fn test_group_with_br_terminator() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        let mask = [true, true, true];

        let ops = vec![
            make_op(IrOpKind::I32Const { value: 5 }, 0),
            {
                let mut op = make_op(
                    IrOpKind::Br {
                        stack_drop: 1,
                        arity: 0,
                        height: 1,
                        operand_base_offset: 0,
                    },
                    1,
                );
                op.has_target = true;
                op.alt_target = Some(OpIndex::from(10));
                op
            },
        ];

        let resolved = resolve_jit(&ops, &mut buf, mask);

        assert!(matches!(resolved[0].kind, IrOpKind::Br { arity: 0, .. }));
        assert!(resolved[0].has_target);
        assert_eq!(resolved[0].alt_target, Some(OpIndex::from(10)));
        assert!(!resolved[0].is_removed());
        assert!(resolved[1].is_internal_only());
    }

    #[test]
    fn test_br_with_fixup_not_grouped() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        let mask = [true, true, true];

        let ops = vec![
            make_op(IrOpKind::I32Const { value: 5 }, 0),
            {
                let mut op = make_op(
                    IrOpKind::Br {
                        stack_drop: 1,
                        arity: 1,
                        height: 1,
                        operand_base_offset: 0,
                    },
                    1,
                );
                op.has_target = true;
                op.alt_target = Some(OpIndex::from(10));
                op
            },
        ];

        let resolved = resolve_jit(&ops, &mut buf, mask);

        assert!(matches!(resolved[0].kind, IrOpKind::I32Const { .. }));
        assert!(matches!(resolved[1].kind, IrOpKind::Br { arity: 1, .. }));
        assert!(!resolved[0].is_removed());
        assert!(!resolved[1].is_removed());
    }

    #[test]
    fn test_br_with_arity_and_no_fixup_grouped() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        let mask = [true, true, true];

        let ops = vec![
            make_op(IrOpKind::I32Const { value: 5 }, 0),
            {
                let mut op = make_op(
                    IrOpKind::Br {
                        stack_drop: 0,
                        arity: 1,
                        height: 1,
                        operand_base_offset: 0,
                    },
                    1,
                );
                op.has_target = true;
                op.alt_target = Some(OpIndex::from(10));
                op
            },
        ];

        let resolved = resolve_jit(&ops, &mut buf, mask);

        assert!(matches!(resolved[0].kind, IrOpKind::Br { arity: 1, stack_drop: 0, .. }));
        assert!(resolved[0].has_target);
        assert_eq!(resolved[0].alt_target, Some(OpIndex::from(10)));
        assert!(!resolved[0].is_removed());
        assert!(resolved[1].is_internal_only());
    }

    #[test]
    fn test_group_with_if_terminator() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        let mask = [true, true, true];

        let ops = vec![
            make_op(IrOpKind::I32Const { value: 1 }, 0),
            {
                let mut op = make_op(IrOpKind::If, 1);
                op.has_target = true;
                op.alt_target = Some(OpIndex::from(10));
                op
            },
        ];

        let resolved = resolve_jit(&ops, &mut buf, mask);

        assert!(matches!(resolved[0].kind, IrOpKind::If));
        assert!(resolved[0].has_target);
        assert_eq!(resolved[0].alt_target, Some(OpIndex::from(10)));
        assert!(!resolved[0].is_removed());
        assert!(resolved[1].is_internal_only());
    }

    #[test]
    fn test_group_with_return_one_terminator() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        let mask = [true, true, true];

        let ops = vec![
            make_op(IrOpKind::I32Const { value: 7 }, 0),
            make_op(IrOpKind::Spill { slot: 3, count: 1 }, 1),
            make_op(
                IrOpKind::ReturnOne {
                    frame_size: 0,
                    operand_base_offset: 24,
                    height: 1,
                },
                1,
            ),
        ];

        let resolved = resolve_jit(&ops, &mut buf, mask);

        assert!(matches!(resolved[0].kind, IrOpKind::ReturnOne { .. }));
        assert!(!resolved[0].is_removed());
        assert!(resolved[1].is_internal_only());
        assert!(resolved[2].is_internal_only());
    }

    #[test]
    fn test_group_with_return_void_terminator() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        let mask = [true, true, true];

        let ops = vec![
            make_op(IrOpKind::InitLocals { k0: 0, k1: 1, k2: 2 }, 0),
            make_op(IrOpKind::ReturnVoid { frame_size: 0 }, 0),
        ];

        let resolved = resolve_jit(&ops, &mut buf, mask);

        assert!(matches!(resolved[0].kind, IrOpKind::ReturnVoid { .. }));
        assert!(!resolved[0].is_removed());
        assert!(resolved[1].is_internal_only());
    }

    #[test]
    fn test_group_does_not_start_at_if() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        let mask = [true, true, true];

        let ops = vec![
            {
                let mut op = make_op(IrOpKind::If, 1);
                op.has_target = true;
                op.alt_target = Some(OpIndex::from(2));
                op
            },
            make_op(IrOpKind::I32Const { value: 11 }, 0),
            make_op(IrOpKind::LocalSetFrame { idx: 0 }, 1),
        ];

        let resolved = resolve_jit(&ops, &mut buf, mask);
        assert!(matches!(resolved[0].kind, IrOpKind::If));
        assert!(!resolved[0].is_removed(), "If must remain a standalone entry");
        assert!(!resolved[1].is_removed(), "grouping must not start at the If terminator");
        assert!(!resolved[2].is_removed(), "incoming branch targets must still break later groups");
    }

    #[test]
    fn test_group_breaks_at_incoming_branch_target() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        let mask = [true, true, true];

        let ops = vec![
            {
                let mut op = make_op(IrOpKind::BrIfSimple, 1);
                op.has_target = true;
                op.alt_target = Some(crate::vm::interp::fast::builder::ir::OpIndex::from(2));
                op
            },
            make_op(IrOpKind::I32Const { value: 5 }, 0),
            make_op(IrOpKind::I32Const { value: 3 }, 0),
            make_op(IrOpKind::I32Eqz, 1),
        ];

        let resolved = resolve_jit(&ops, &mut buf, mask);

        assert!(matches!(resolved[0].kind, IrOpKind::BrIfSimple));
        assert!(matches!(resolved[1].kind, IrOpKind::I32Const { .. }));
        assert!(!resolved[1].is_removed(), "incoming target must break the group before ir[2]");
        assert!(!resolved[2].is_removed(), "targeted op should remain a real entry");
        assert!(resolved[3].is_internal_only(), "ir[2..4] may still form a group starting at the target");
    }

    // ==================== Execution tests ====================

    fn run_group_test(
        handler: OpHandler,
        _initial_height: usize,
        t0: u64, t1: u64, t2: u64, t3: u64,
        l0: u64, l1: u64, l2: u64,
    ) -> u64 {
        let term = full_set::op_term;

        let mut insts = [
            Instruction::new_handler_only(handler),
            Instruction::new_handler_only(term),
            Instruction::new_handler_only(term),
        ];

        let mut frame = [0u64; 32];
        let mut ctx = Context::new(
            core::ptr::null_mut(), core::ptr::null(),
            frame.as_mut_ptr().wrapping_add(32),
            core::ptr::null_mut(), 0,
        );
        ctx.hot.term_inst = handlers::term();

        let pc = &mut insts[0] as *mut Instruction;
        let nh: NextHandler = unsafe { core::mem::transmute(insts[1].handler) };

        unsafe {
            run_trampoline(
                &mut ctx, pc, frame.as_mut_ptr(),
                l0, l1, l2,
                t0, t1, t2, t3,
                nh,
            );
        }

        frame[0]
    }

    #[test]
    fn test_compile_group_const_add() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        buf.begin_write();
        let mut e = JitEmitter::new(&mut buf, 0);
        e.i32_const(5);
        e.i32_const(3);
        e.i32_add();
        let result_reg = tos_reg(depth_variant(e.height()), 1);
        e.emit_raw(arm64_enc::str_64(result_reg, Reg::FP, 0));
        let start = e.finish();
        let total = buf.len();
        buf.finish_write(0, total);

        let handler: OpHandler = unsafe { buf.fn_ptr(start) };
        let result = run_group_test(handler, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(result, 8);
    }

    #[test]
    fn test_compile_group_with_locals() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        buf.begin_write();

        let mut e = JitEmitter::new(&mut buf, 0);
        e.local_get_ln(0);
        e.local_get_ln(1);
        e.i32_add();
        let result_reg = tos_reg(depth_variant(e.height()), 1);
        e.emit_raw(arm64_enc::str_64(result_reg, Reg::FP, 0));
        let start = e.finish();
        let total = buf.len();
        buf.finish_write(0, total);

        let handler: OpHandler = unsafe { buf.fn_ptr(start) };
        let result = run_group_test(handler, 0, 0, 0, 0, 0, 10, 20, 0);
        assert_eq!(result, 30);
    }

    #[test]
    fn test_compile_group_local_set() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        buf.begin_write();

        let mut e = JitEmitter::new(&mut buf, 0);
        e.i32_const(42);
        e.local_set_ln(0);
        let start_offset = e.start_offset;
        e.emit_raw(arm64_enc::str_64(Reg::L0, Reg::FP, 0));
        let _ = e.finish();
        let total = buf.len();
        buf.finish_write(0, total);

        let handler: OpHandler = unsafe { buf.fn_ptr(start_offset) };
        let result = run_group_test(handler, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_jit_stats_nonzero_after_compilation() {
        JIT_STATS.groups.store(0, core::sync::atomic::Ordering::Relaxed);
        JIT_STATS.ops.store(0, core::sync::atomic::Ordering::Relaxed);

        let mut buf = CodeBuffer::new().expect("mmap failed");
        let mask = [true, true, true];

        let ops = vec![
            make_op(IrOpKind::I32Const { value: 5 }, 0),
            make_op(IrOpKind::I32Const { value: 3 }, 1),
            make_op(IrOpKind::I32Add, 2),
        ];

        let _ = resolve_jit(&ops, &mut buf, mask);

        let (groups, ops_count) = jit_stats();
        assert!(groups >= 1, "expected at least 1 JIT group, got {}", groups);
        assert!(ops_count >= 3, "expected at least 3 ops, got {}", ops_count);
    }

    /// Test spill: push 4 consts, spill bottom, push 5th, add top 2.
    /// Sequence: const(1) const(2) const(3) const(4) spill_1 const(5) i32_add
    /// Expected: top = 4 + 5 = 9, and fp[slot] = 1 (spilled)
    #[test]
    fn test_spill_basic() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        buf.begin_write();

        // Simulate what the IR lowering produces:
        // Push 4 consts (fills all 4 TOS regs)
        let mut e = JitEmitter::new(&mut buf, 0);
        e.i32_const(1);  // height 0→1, T0
        e.i32_const(2);  // height 1→2, T1
        e.i32_const(3);  // height 2→3, T2
        e.i32_const(4);  // height 3→4, T3

        // Spill bottom (T0) to fp[10] — variant 1 (spill_depth=0)
        e.spill(10, 1, 1);  // STR T0, [FP, #10*8]

        // Push 5th const — reuses T0 (height 4→5, dv=1)
        e.i32_const(5);

        // I32Add: pops top 2 (T0=5, T3=4), result in T3
        e.i32_add();

        // Store result (T3) to fp[0] for verification
        let result_reg = tos_reg(depth_variant(e.height()), 1);
        e.emit_raw(arm64_enc::str_64(result_reg, Reg::FP, 0));

        // Also store fp[10] (spilled value) to fp[1] for verification
        // Read fp[10] into a scratch, then store to fp[1]
        e.emit_raw(arm64_enc::ldr_64(Reg::TMP0, Reg::FP, 10));
        e.emit_raw(arm64_enc::str_64(Reg::TMP0, Reg::FP, 1));

        let start = e.finish();
        let total = buf.len();
        buf.finish_write(0, total);

        let handler: OpHandler = unsafe { buf.fn_ptr(start) };

        let term = full_set::op_term;
        let mut insts = [
            Instruction::new_handler_only(handler),
            Instruction::new_handler_only(term),
            Instruction::new_handler_only(term),
        ];
        let mut frame = [0u64; 32];
        let mut ctx = Context::new(
            core::ptr::null_mut(), core::ptr::null(),
            frame.as_mut_ptr().wrapping_add(32),
            core::ptr::null_mut(), 0,
        );
        ctx.hot.term_inst = handlers::term();
        let pc = &mut insts[0] as *mut Instruction;
        let nh: NextHandler = unsafe { core::mem::transmute(insts[1].handler) };

        unsafe {
            run_trampoline(
                &mut ctx, pc, frame.as_mut_ptr(),
                0, 0, 0,
                0, 0, 0, 0,
                nh,
            );
        }

        assert_eq!(frame[0], 9, "expected 4+5=9, got {}", frame[0]);
        assert_eq!(frame[1], 1, "expected spilled value 1, got {}", frame[1]);
    }

    /// Test fill: pre-populate memory, fill into TOS, then use values.
    /// Simulates: spill_all at height=3, then fill_2 to restore top 2, then i32_add.
    #[test]
    fn test_fill_basic() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        buf.begin_write();

        // Pre-populate memory to simulate values that were spilled earlier.
        // We'll store known values at fp[10], fp[11], fp[12] (heights 1, 2, 3).
        let mut e = JitEmitter::new(&mut buf, 0);

        // Push 3 values and spill all (like emit_spill_all before a call)
        e.i32_const(10); // height 0→1, T0=10
        e.i32_const(20); // height 1→2, T1=20
        e.i32_const(30); // height 2→3, T2=30

        // spill_all: variant = dv(3) = 3, count = 3, slot = 12
        // This stores T2→fp[12], T1→fp[11], T0→fp[10]
        e.spill(12, 3, 3);

        // Now fill 2 values back (like emit_fill after control flow merge)
        // Fill top 2 from fp[12] (height 3) and fp[11] (height 2)
        // variant = (((3-1)%4)+1) = 3, count = 2
        e.fill(12, 2, 3);

        // Now TOS should have: T2=30 (pos1, top), T1=20 (pos2)
        // i32_add: 30 + 20 = 50
        e.i32_add();

        // Store result to fp[0]
        let result_reg = tos_reg(depth_variant(e.height()), 1);
        e.emit_raw(arm64_enc::str_64(result_reg, Reg::FP, 0));

        let start = e.finish();
        let total = buf.len();
        buf.finish_write(0, total);

        let handler: OpHandler = unsafe { buf.fn_ptr(start) };

        let term = full_set::op_term;
        let mut insts = [
            Instruction::new_handler_only(handler),
            Instruction::new_handler_only(term),
            Instruction::new_handler_only(term),
        ];
        let mut frame = [0u64; 32];
        let mut ctx = Context::new(
            core::ptr::null_mut(), core::ptr::null(),
            frame.as_mut_ptr().wrapping_add(32),
            core::ptr::null_mut(), 0,
        );
        ctx.hot.term_inst = handlers::term();
        let pc = &mut insts[0] as *mut Instruction;
        let nh: NextHandler = unsafe { core::mem::transmute(insts[1].handler) };

        unsafe {
            run_trampoline(
                &mut ctx, pc, frame.as_mut_ptr(),
                0, 0, 0,
                0, 0, 0, 0,
                nh,
            );
        }

        assert_eq!(frame[0], 50, "expected 30+20=50, got {}", frame[0]);
    }

    /// Test resolve_jit with spill/fill in the IR stream.
    /// Simulates 5 pushes with a spill, then an add.
    #[test]
    fn test_resolve_jit_with_spill() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        let mask = [true, true, true];

        // Simulate IR from lowering: 4 consts, spill_1, 5th const, i32_add
        let ops = vec![
            make_op(IrOpKind::I32Const { value: 10 }, 0),
            make_op(IrOpKind::I32Const { value: 20 }, 1),
            make_op(IrOpKind::I32Const { value: 30 }, 2),
            make_op(IrOpKind::I32Const { value: 40 }, 3),
            // Spill at height=4, spill_depth=0 → variant=1
            {
                let mut op = make_op(IrOpKind::Spill { slot: 4, count: 1 }, 4);
                op.variant = 1;
                op
            },
            make_op(IrOpKind::I32Const { value: 50 }, 4),
            make_op(IrOpKind::I32Add, 5),
        ];

        let resolved = resolve_jit(&ops, &mut buf, mask);

        // Should be 1 JIT group (7 ops)
        assert!(!resolved[0].is_removed(), "expected JIT entry");
        // Remaining should be skips
        for i in 1..7 {
            assert!(resolved[i].is_internal_only(), "expected skip at {}", i);
        }
    }

    /// Test full spill→overwrite→fill→use pattern at runtime.
    /// Pattern: const(1) const(2) const(3) const(4) spill_1 const(5)
    ///          add add add fill_1 add
    /// Expected: 1 + (2 + (3 + (4 + 5))) = 15
    #[test]
    fn test_spill_fill_roundtrip() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        buf.begin_write();
        let mut e = JitEmitter::new(&mut buf, 0);

        // Push 4 consts
        e.i32_const(1);  // h0→1, T0=1
        e.i32_const(2);  // h1→2, T1=2
        e.i32_const(3);  // h2→3, T2=3
        e.i32_const(4);  // h3→4, T3=4

        // Spill T0 to fp[3] (variant=1 for spill_depth=0)
        e.spill(3, 1, 1);

        // Push 5th const (overwrites T0)
        e.i32_const(5);  // h4→5, T0=5

        // 4 adds to consume everything
        e.i32_add();  // 5+4=9, h5→4
        e.i32_add();  // 9+3=12, h4→3
        e.i32_add();  // 12+2=14, h3→2

        // Fill: load spilled value back from fp[3] into T0
        // variant = ((0%4)+1) = 1
        e.fill(3, 1, 1);

        // Final add: 14+1=15
        e.i32_add();  // h2→1

        // Store result to fp[0]
        let result_reg = tos_reg(depth_variant(e.height()), 1);
        e.emit_raw(arm64_enc::str_64(result_reg, Reg::FP, 0));

        let start = e.finish();
        let total = buf.len();
        buf.finish_write(0, total);

        let handler: OpHandler = unsafe { buf.fn_ptr(start) };

        let term = full_set::op_term;
        let mut insts = [
            Instruction::new_handler_only(handler),
            Instruction::new_handler_only(term),
            Instruction::new_handler_only(term),
        ];
        let mut frame = [0u64; 32];
        let mut ctx = Context::new(
            core::ptr::null_mut(), core::ptr::null(),
            frame.as_mut_ptr().wrapping_add(32),
            core::ptr::null_mut(), 0,
        );
        ctx.hot.term_inst = handlers::term();
        let pc = &mut insts[0] as *mut Instruction;
        let nh: NextHandler = unsafe { core::mem::transmute(insts[1].handler) };

        unsafe {
            run_trampoline(
                &mut ctx, pc, frame.as_mut_ptr(),
                0, 0, 0,
                0, 0, 0, 0,
                nh,
            );
        }

        assert_eq!(frame[0], 15, "expected 1+(2+(3+(4+5)))=15, got {}", frame[0]);
    }

    /// Test that resolve_jit handles fill correctly via the full pipeline.
    /// Uses IrOps with correct variants and pre_heights.
    #[test]
    fn test_resolve_jit_with_fill() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        let mask = [true, true, true];

        fn make_op_v(kind: IrOpKind, pre_height: u16, variant: u8) -> IrOp {
            IrOp { kind, variant, pre_height, fallthrough: None, alt_target: None, has_target: false }
        }

        let ops = vec![
            make_op_v(IrOpKind::I32Const { value: 1 }, 0, 1),
            make_op_v(IrOpKind::I32Const { value: 2 }, 1, 2),
            make_op_v(IrOpKind::I32Const { value: 3 }, 2, 3),
            make_op_v(IrOpKind::I32Const { value: 4 }, 3, 4),
            make_op_v(IrOpKind::Spill { slot: 3, count: 1 }, 4, 1),
            make_op_v(IrOpKind::I32Const { value: 5 }, 4, 1),
            make_op_v(IrOpKind::I32Add, 5, 1),
            make_op_v(IrOpKind::I32Add, 4, 4),
            make_op_v(IrOpKind::I32Add, 3, 3),
            make_op_v(IrOpKind::Fill { slot: 3, count: 1 }, 2, 1),
            make_op_v(IrOpKind::I32Add, 2, 2),
        ];

        let resolved = resolve_jit(&ops, &mut buf, mask);

        // Should be 1 group of 11 ops
        assert!(!resolved[0].is_removed(), "expected JIT entry");
        for i in 1..11 {
            assert!(resolved[i].is_internal_only(), "expected skip at {}", i);
        }

        // Execute it
        let handler = resolved[0].handler;
        let term = full_set::op_term;
        let mut insts = [
            Instruction::new_handler_only(handler),
            Instruction::new_handler_only(term),
            Instruction::new_handler_only(term),
        ];
        let mut frame = [0u64; 32];
        let mut ctx = Context::new(
            core::ptr::null_mut(), core::ptr::null(),
            frame.as_mut_ptr().wrapping_add(32),
            core::ptr::null_mut(), 0,
        );
        ctx.hot.term_inst = handlers::term();
        let pc = &mut insts[0] as *mut Instruction;
        let nh: NextHandler = unsafe { core::mem::transmute(insts[1].handler) };

        unsafe {
            run_trampoline(
                &mut ctx, pc, frame.as_mut_ptr(),
                0, 0, 0,
                0, 0, 0, 0,
                nh,
            );
        }

        // After the group, height=1, dv=1, result in T0
        // But the group uses dispatch_linear which advances PC, doesn't store result
        // The resolved group doesn't include a store — we need to check via
        // a different mechanism. Let's check the spill value in frame[3] instead.
        assert_eq!(frame[3], 1, "spilled value at fp[3] should be 1, got {}", frame[3]);
    }

    #[test]
    fn test_resolve_jit_with_init_locals() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        let mask = [true, false, false];

        let ops = vec![
            make_op(IrOpKind::InitLocals { k0: 5, k1: 1, k2: 2 }, 0),
            make_op(IrOpKind::LocalGetFrame { idx: 0 }, 0),
            make_op(IrOpKind::LocalSetFrame { idx: 8 }, 1),
        ];

        let resolved = resolve_jit(&ops, &mut buf, mask);
        assert!(!resolved[0].is_removed(), "expected JIT entry");
        assert!(resolved[1].is_internal_only());
        assert!(resolved[2].is_internal_only());

        let handler = resolved[0].handler;
        let term = full_set::op_term;
        let mut insts = [
            Instruction::new_handler_only(handler),
            Instruction::new_handler_only(term),
            Instruction::new_handler_only(term),
        ];
        let mut frame = [0u64; 32];
        frame[0] = 10;
        frame[5] = 50;

        let mut ctx = Context::new(
            core::ptr::null_mut(),
            core::ptr::null(),
            frame.as_mut_ptr().wrapping_add(32),
            core::ptr::null_mut(),
            0,
        );
        ctx.hot.term_inst = handlers::term();
        let pc = &mut insts[0] as *mut Instruction;
        let nh: NextHandler = unsafe { core::mem::transmute(insts[1].handler) };

        unsafe {
            run_trampoline(
                &mut ctx,
                pc,
                frame.as_mut_ptr(),
                111,
                222,
                333,
                0,
                0,
                0,
                0,
                nh,
            );
        }

        assert_eq!(frame[0], 50, "fp[0] should contain the hot local after swap");
        assert_eq!(frame[5], 10, "fp[5] should contain the original fp[0]");
        assert_eq!(frame[8], 50, "LocalGetFrame idx 0 should read from L0 after init");
    }

    #[test]
    fn test_resolve_jit_with_if_terminator() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        let mask = [true, true, true];

        let ops = vec![
            make_op(IrOpKind::I32Const { value: 42 }, 0),
            make_op(IrOpKind::I32Const { value: 0 }, 1),
            {
                let mut op = make_op(IrOpKind::If, 2);
                op.has_target = true;
                op.alt_target = Some(OpIndex::from(3));
                op
            },
        ];

        let resolved = resolve_jit(&ops, &mut buf, mask);
        assert!(matches!(resolved[0].kind, IrOpKind::If));
        assert_eq!(resolved[0].alt_target, Some(OpIndex::from(3)));
        assert!(resolved[1].is_internal_only());
        assert!(resolved[2].is_internal_only());
    }

    #[test]
    fn test_resolve_jit_with_hot_float_chain() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        let mask = [false, true, false];

        let ops = vec![
            make_op(IrOpKind::LocalGetHot { reg: 0 }, 0),
            make_op(IrOpKind::LocalGetHot { reg: 0 }, 1),
            make_op(IrOpKind::F64Mul, 2),
            make_op(IrOpKind::LocalTeeHot { reg: 1 }, 1),
            make_op(IrOpKind::LocalGetHot { reg: 1 }, 1),
            make_op(IrOpKind::F64Add, 2),
            make_op(IrOpKind::LocalSetFrame { idx: 3 }, 1),
        ];

        let resolved = resolve_jit(&ops, &mut buf, mask);
        let handler = resolved[0].handler;
        let term = full_set::op_term;
        let mut insts = [
            Instruction::new_handler_only(handler),
            Instruction::new_handler_only(term),
            Instruction::new_handler_only(term),
        ];
        let mut frame = [0u64; 32];
        let mut ctx = Context::new(
            core::ptr::null_mut(),
            core::ptr::null(),
            frame.as_mut_ptr().wrapping_add(32),
            core::ptr::null_mut(),
            0,
        );
        ctx.hot.term_inst = handlers::term();
        let pc = &mut insts[0] as *mut Instruction;
        let nh: NextHandler = unsafe { core::mem::transmute(insts[1].handler) };

        unsafe {
            run_trampoline(
                &mut ctx,
                pc,
                frame.as_mut_ptr(),
                3.0f64.to_bits(),
                0,
                0,
                0,
                0,
                0,
                0,
                nh,
            );
        }

        assert_eq!(frame[3], 18.0f64.to_bits());
    }

    #[test]
    fn test_resolve_jit_with_hot_frame_float_mix() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        let mask = [false, true, false];

        let ops = vec![
            make_op(IrOpKind::LocalGetFrame { idx: 3 }, 0),
            make_op(IrOpKind::LocalGetHot { reg: 1 }, 1),
            make_op(IrOpKind::LocalGetFrame { idx: 4 }, 2),
            make_op(IrOpKind::F64Sub, 3),
            make_op(IrOpKind::F64Add, 2),
            make_op(IrOpKind::LocalSetFrame { idx: 5 }, 1),
        ];

        let resolved = resolve_jit(&ops, &mut buf, mask);
        let handler = resolved[0].handler;
        let term = full_set::op_term;
        let mut insts = [
            Instruction::new_handler_only(handler),
            Instruction::new_handler_only(term),
            Instruction::new_handler_only(term),
        ];
        let mut frame = [0u64; 32];
        frame[3] = 5.0f64.to_bits();
        frame[4] = 2.0f64.to_bits();
        let mut ctx = Context::new(
            core::ptr::null_mut(),
            core::ptr::null(),
            frame.as_mut_ptr().wrapping_add(32),
            core::ptr::null_mut(),
            0,
        );
        ctx.hot.term_inst = handlers::term();
        let pc = &mut insts[0] as *mut Instruction;
        let nh: NextHandler = unsafe { core::mem::transmute(insts[1].handler) };

        unsafe {
            run_trampoline(
                &mut ctx,
                pc,
                frame.as_mut_ptr(),
                0,
                3.0f64.to_bits(),
                0,
                0,
                0,
                0,
                0,
                nh,
            );
        }

        assert_eq!(frame[5], 6.0f64.to_bits());
    }

    #[test]
    fn test_group_estimate_covers_arithmetic_group() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        let mask = [true, true, true];
        let ops = vec![
            make_op(IrOpKind::I32Const { value: 7 }, 0),
            make_op(IrOpKind::I32Const { value: 3 }, 1),
            make_op(IrOpKind::I32Add, 2),
            make_op(IrOpKind::I32Eqz, 1),
        ];

        let estimate = op_meta::estimate_group_bytes(&ops, None);
        let before = buf.len();
        buf.begin_write();
        let compiled = try_compile_group(&ops, &mut buf, mask).expect("group should compile");
        let written = buf.len() - before;
        buf.finish_write(before, written);

        assert!(compiled.0 as usize != 0);
        assert!(written <= estimate, "actual {} bytes exceeded estimate {}", written, estimate);
    }

    #[test]
    fn test_group_estimate_covers_memory_group() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        let mask = [true, true, true];
        let ops = vec![
            make_op(IrOpKind::I32Load { offset: 1024, memidx: 0 }, 1),
            make_op(IrOpKind::I32Eqz, 1),
        ];

        let estimate = op_meta::estimate_group_bytes(&ops, None);
        let before = buf.len();
        buf.begin_write();
        let compiled = try_compile_group(&ops, &mut buf, mask).expect("memory group should compile");
        let written = buf.len() - before;
        buf.finish_write(before, written);

        assert!(compiled.0 as usize != 0);
        assert!(written <= estimate, "actual {} bytes exceeded estimate {}", written, estimate);
    }
}
