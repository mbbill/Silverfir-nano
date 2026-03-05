//! JIT group compiler: identify and compile groups of consecutive JIT-able ops.
//!
//! Scans `&[IrOp]` for consecutive JIT-able operations, compiles them into
//! single ARM64 code blocks via `JitEmitter`, and produces `Vec<ResolvedInst>`.
//! Non-JIT-able ops fall back to 1:1 base handler resolution.

use alloc::vec::Vec;
use super::code_buf::CodeBuffer;
use super::codegen::JitEmitter;
use crate::vm::interp::fast::builder::backend::ResolvedInst;
use crate::vm::interp::fast::builder::ir::{IrOp, IrOpKind, stack_effect};
use crate::vm::interp::fast::handlers::OpHandler;

// =========================================================================
// JIT-ability classification (JIT-specific policy, not IR truth)
// =========================================================================

/// Whether an IR op kind can be JIT-compiled into ARM64 native code.
fn is_jit_able_kind(kind: &IrOpKind) -> bool {
    use IrOpKind::*;
    matches!(
        kind,
        // i32/i64 arithmetic (not div/rem — they can trap)
        I32Add | I32Sub | I32Mul |
        I32And | I32Or | I32Xor |
        I32Shl | I32ShrS | I32ShrU | I32Rotl | I32Rotr |
        I64Add | I64Sub | I64Mul |
        I64And | I64Or | I64Xor |
        I64Shl | I64ShrS | I64ShrU | I64Rotl | I64Rotr |

        // i32/i64 comparisons
        I32Eq | I32Ne | I32LtS | I32LtU | I32GtS | I32GtU |
        I32LeS | I32LeU | I32GeS | I32GeU |
        I64Eq | I64Ne | I64LtS | I64LtU | I64GtS | I64GtU |
        I64LeS | I64LeU | I64GeS | I64GeU |

        // i32/i64 unary
        I32Eqz | I32Clz | I32Ctz |
        I64Eqz | I64Clz | I64Ctz |

        // Constants (i32/i64 only — not f32/f64)
        I32Const { .. } | I64Const { .. } |

        // Locals (hot and frame get/set, hot tee only)
        LocalGetHot { .. } | LocalGetFrame { .. } |
        LocalSetHot { .. } | LocalSetFrame { .. } |
        LocalTeeHot { .. } |

        // Drop
        Drop |

        // Memory loads (all variants — memidx checked separately)
        I32Load { .. } | I64Load { .. } | F32Load { .. } | F64Load { .. } |
        I32Load8S { .. } | I32Load8U { .. } | I32Load16S { .. } | I32Load16U { .. } |
        I64Load8S { .. } | I64Load8U { .. } | I64Load16S { .. } | I64Load16U { .. } |
        I64Load32S { .. } | I64Load32U { .. } |

        // Memory stores (all variants — memidx checked separately)
        I32Store { .. } | I64Store { .. } | F32Store { .. } | F64Store { .. } |
        I32Store8 { .. } | I32Store16 { .. } |
        I64Store8 { .. } | I64Store16 { .. } | I64Store32 { .. }
    )
}

/// Check memidx constraint for JIT (only mem0 supported).
fn is_jit_able_mem(kind: &IrOpKind) -> bool {
    use IrOpKind::*;
    match kind {
        I32Load { memidx, .. } | I64Load { memidx, .. } |
        F32Load { memidx, .. } | F64Load { memidx, .. } |
        I32Load8S { memidx, .. } | I32Load8U { memidx, .. } |
        I32Load16S { memidx, .. } | I32Load16U { memidx, .. } |
        I64Load8S { memidx, .. } | I64Load8U { memidx, .. } |
        I64Load16S { memidx, .. } | I64Load16U { memidx, .. } |
        I64Load32S { memidx, .. } | I64Load32U { memidx, .. } |
        I32Store { memidx, .. } | I64Store { memidx, .. } |
        F32Store { memidx, .. } | F64Store { memidx, .. } |
        I32Store8 { memidx, .. } | I32Store16 { memidx, .. } |
        I64Store8 { memidx, .. } | I64Store16 { memidx, .. } |
        I64Store32 { memidx, .. } => *memidx == 0,
        _ => true,
    }
}

/// Check if an IrOp is JIT-able, considering kind, height, and hot local mask.
fn is_jit_able_op(op: &IrOp, hot_local_mask: [bool; 3]) -> bool {
    let kind = &op.kind;
    let h = op.pre_height as usize;

    if !is_jit_able_kind(kind) { return false; }
    if !is_jit_able_mem(kind) { return false; }

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

/// Check if an IrOp is a br_if_simple group terminator (can only terminate, not start).
fn is_br_if_simple(op: &IrOp) -> bool {
    op.pre_height >= 1 && matches!(op.kind, IrOpKind::BrIfSimple)
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

pub static JIT_STATS: JitStats = JitStats {
    groups: core::sync::atomic::AtomicUsize::new(0),
    ops: core::sync::atomic::AtomicUsize::new(0),
    bytes_emitted: core::sync::atomic::AtomicUsize::new(0),
    groups_skipped_capacity: core::sync::atomic::AtomicUsize::new(0),
    ops_skipped_capacity: core::sync::atomic::AtomicUsize::new(0),
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
    buf.begin_write();
    let bytes_before = buf.len();

    let mut out = Vec::with_capacity(ir.len());
    let mut groups_compiled: usize = 0;
    let mut ops_compiled: usize = 0;
    let mut groups_skipped: usize = 0;
    let mut ops_skipped: usize = 0;

    let mut i = 0;
    while i < ir.len() {
        // Try to start a JIT group
        if is_jit_able_op(&ir[i], hot_local_mask) {
            let group_start = i;
            i += 1;

            // Extend the group
            while i < ir.len() {
                if is_br_if_simple(&ir[i]) {
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
                let estimated_bytes = group_len * 256 + 256;
                if buf.remaining() >= estimated_bytes {
                    if let Some((handler, ends_brif, branch_target)) =
                        try_compile_group(&ir[group_start..i], buf, hot_local_mask)
                    {
                        // JIT entry
                        if ends_brif {
                            out.push(ResolvedInst {
                                handler,
                                kind: IrOpKind::BrIfSimple,
                                alt_target: branch_target,
                                has_target: true,
                                structural: false,
                            });
                        } else {
                            out.push(ResolvedInst {
                                handler,
                                kind: IrOpKind::Data { imm0: 0, imm1: 0, imm2: 0 },
                                alt_target: None,
                                has_target: false,
                                structural: false,
                            });
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

        // Lone br_if_simple or non-JIT-able op: 1:1
        out.push(ResolvedInst::from_ir(&ir[i]));
        i += 1;
    }

    let total_len = buf.len();
    buf.finish_write(0, total_len);

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

/// Try to compile a group of IrOps into a single JIT handler.
///
/// Returns `(handler, ends_with_brif, branch_alt_target)` on success.
fn try_compile_group(
    group: &[IrOp],
    buf: &mut CodeBuffer,
    hot_local_mask: [bool; 3],
) -> Option<(OpHandler, bool, Option<usize>)> {
    let ends_with_brif = is_br_if_simple(group.last().unwrap());
    let branch_alt = if ends_with_brif {
        group.last().unwrap().alt_target
    } else {
        None
    };

    // Validate: check that heights stay valid throughout the group
    let mut sim_height = group[0].pre_height as usize;
    let body_end = if ends_with_brif { group.len() - 1 } else { group.len() };
    for op in &group[..body_end] {
        let (pops, pushes) = stack_effect(&op.kind);
        if sim_height < pops as usize {
            return None;
        }
        sim_height = sim_height - pops as usize + pushes as usize;
    }
    if ends_with_brif && sim_height < 1 {
        return None;
    }

    // Compile
    let mut e = JitEmitter::new(buf, group[0].pre_height as usize);
    for op in group[..body_end].iter() {
        emit_op(&mut e, op, hot_local_mask);
    }

    let start = if ends_with_brif {
        e.finish_br_if_simple()
    } else {
        e.finish()
    };

    let handler: OpHandler = unsafe { buf.fn_ptr(start) };
    Some((handler, ends_with_brif, branch_alt))
}

/// Emit a single op via JitEmitter based on its IrOpKind.
fn emit_op(e: &mut JitEmitter, op: &IrOp, hot_local_mask: [bool; 3]) {
    match &op.kind {
        // i32 binary
        IrOpKind::I32Add => e.i32_add(),
        IrOpKind::I32Sub => e.i32_sub(),
        IrOpKind::I32Mul => e.i32_mul(),
        IrOpKind::I32And => e.i32_and(),
        IrOpKind::I32Or => e.i32_or(),
        IrOpKind::I32Xor => e.i32_xor(),
        IrOpKind::I32Shl => e.i32_shl(),
        IrOpKind::I32ShrU => e.i32_shr_u(),
        IrOpKind::I32ShrS => e.i32_shr_s(),
        IrOpKind::I32Rotl => e.i32_rotl(),
        IrOpKind::I32Rotr => e.i32_rotr(),

        // i64 binary
        IrOpKind::I64Add => e.i64_add(),
        IrOpKind::I64Sub => e.i64_sub(),
        IrOpKind::I64Mul => e.i64_mul(),
        IrOpKind::I64And => e.i64_and(),
        IrOpKind::I64Or => e.i64_or(),
        IrOpKind::I64Xor => e.i64_xor(),
        IrOpKind::I64Shl => e.i64_shl(),
        IrOpKind::I64ShrU => e.i64_shr_u(),
        IrOpKind::I64ShrS => e.i64_shr_s(),
        IrOpKind::I64Rotl => e.i64_rotl(),
        IrOpKind::I64Rotr => e.i64_rotr(),

        // i32 comparisons
        IrOpKind::I32Eq => e.i32_eq(),
        IrOpKind::I32Ne => e.i32_ne(),
        IrOpKind::I32LtS => e.i32_lt_s(),
        IrOpKind::I32LtU => e.i32_lt_u(),
        IrOpKind::I32GtS => e.i32_gt_s(),
        IrOpKind::I32GtU => e.i32_gt_u(),
        IrOpKind::I32LeS => e.i32_le_s(),
        IrOpKind::I32LeU => e.i32_le_u(),
        IrOpKind::I32GeS => e.i32_ge_s(),
        IrOpKind::I32GeU => e.i32_ge_u(),

        // i64 comparisons
        IrOpKind::I64Eq => e.i64_eq(),
        IrOpKind::I64Ne => e.i64_ne(),
        IrOpKind::I64LtS => e.i64_lt_s(),
        IrOpKind::I64LtU => e.i64_lt_u(),
        IrOpKind::I64GtS => e.i64_gt_s(),
        IrOpKind::I64GtU => e.i64_gt_u(),
        IrOpKind::I64LeS => e.i64_le_s(),
        IrOpKind::I64LeU => e.i64_le_u(),
        IrOpKind::I64GeS => e.i64_ge_s(),
        IrOpKind::I64GeU => e.i64_ge_u(),

        // i32 unary
        IrOpKind::I32Eqz => e.i32_eqz(),
        IrOpKind::I32Clz => e.i32_clz(),
        IrOpKind::I32Ctz => e.i32_ctz(),

        // i64 unary
        IrOpKind::I64Eqz => e.i64_eqz(),
        IrOpKind::I64Clz => e.i64_clz(),
        IrOpKind::I64Ctz => e.i64_ctz(),

        // Constants
        IrOpKind::I32Const { value } => e.i32_const(*value as u32),
        IrOpKind::I64Const { value } => e.i64_const(*value),

        // Locals — hot vs frame
        IrOpKind::LocalGetHot { reg } => e.local_get_ln(*reg),
        IrOpKind::LocalGetFrame { idx } => {
            if (*idx as usize) < 3 && hot_local_mask[*idx as usize] {
                e.local_get_ln(*idx as u8);
            } else {
                e.local_get(*idx);
            }
        }
        IrOpKind::LocalSetHot { reg } => e.local_set_ln(*reg),
        IrOpKind::LocalSetFrame { idx } => {
            if (*idx as usize) < 3 && hot_local_mask[*idx as usize] {
                e.local_set_ln(*idx as u8);
            } else {
                e.local_set(*idx);
            }
        }
        IrOpKind::LocalTeeHot { reg } => e.local_tee_ln(*reg),

        // Drop
        IrOpKind::Drop => e.drop_val(),

        // Memory loads
        IrOpKind::I32Load { offset, .. } => e.i32_load(*offset),
        IrOpKind::I32Load8S { offset, .. } => e.i32_load8_s(*offset),
        IrOpKind::I32Load8U { offset, .. } => e.i32_load8_u(*offset),
        IrOpKind::I32Load16S { offset, .. } => e.i32_load16_s(*offset),
        IrOpKind::I32Load16U { offset, .. } => e.i32_load16_u(*offset),
        IrOpKind::I64Load { offset, .. } => e.i64_load(*offset),
        IrOpKind::I64Load8S { offset, .. } => e.i64_load8_s(*offset),
        IrOpKind::I64Load8U { offset, .. } => e.i64_load8_u(*offset),
        IrOpKind::I64Load16S { offset, .. } => e.i64_load16_s(*offset),
        IrOpKind::I64Load16U { offset, .. } => e.i64_load16_u(*offset),
        IrOpKind::I64Load32S { offset, .. } => e.i64_load32_s(*offset),
        IrOpKind::I64Load32U { offset, .. } => e.i64_load32_u(*offset),
        IrOpKind::F32Load { offset, .. } => e.i32_load(*offset),
        IrOpKind::F64Load { offset, .. } => e.i64_load(*offset),

        // Memory stores
        IrOpKind::I32Store { offset, .. } => e.i32_store(*offset),
        IrOpKind::I32Store8 { offset, .. } => e.i32_store8(*offset),
        IrOpKind::I32Store16 { offset, .. } => e.i32_store16(*offset),
        IrOpKind::I64Store { offset, .. } => e.i64_store(*offset),
        IrOpKind::I64Store8 { offset, .. } => e.i64_store8(*offset),
        IrOpKind::I64Store16 { offset, .. } => e.i64_store16(*offset),
        IrOpKind::I64Store32 { offset, .. } => e.i64_store32(*offset),
        IrOpKind::F32Store { offset, .. } => e.i32_store(*offset),
        IrOpKind::F64Store { offset, .. } => e.i64_store(*offset),

        _ => {} // Should not reach here (is_jit_able_op filters)
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
    use crate::vm::interp::fast::jit::codegen::{depth_variant, tos_reg};
    use crate::vm::interp::fast::jit::reg::Reg;
    use crate::vm::interp::fast::jit::arm64_enc;
    use crate::vm::interp::fast::jit::emit;

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

        // Group 1: resolved[0] = JIT entry (non-structural), [1,2] = skip (structural)
        assert!(!resolved[0].structural);
        assert!(resolved[1].structural);
        assert!(resolved[2].structural);

        // Separator: 1:1 Nop (structural)
        assert!(resolved[3].structural);

        // Group 2: resolved[4] = JIT entry, [5,6] = skip
        assert!(!resolved[4].structural);
        assert!(resolved[5].structural);
        assert!(resolved[6].structural);
    }

    #[test]
    fn test_single_op_not_grouped() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        let mask = [true, true, true];

        let ops = vec![
            make_op(IrOpKind::I32Const { value: 5 }, 0),
            make_op(IrOpKind::CallExternal { func_idx: 0, delta: 0 }, 0),
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
                op.alt_target = Some(10);
                op
            },
        ];

        let resolved = resolve_jit(&ops, &mut buf, mask);

        // JIT entry with br_if encoding
        assert!(matches!(resolved[0].kind, IrOpKind::BrIfSimple));
        assert!(resolved[0].has_target);
        assert_eq!(resolved[0].alt_target, Some(10));
        assert!(!resolved[0].structural);

        // Rest are skip
        assert!(resolved[1].structural);
        assert!(resolved[2].structural);
        assert!(resolved[3].structural);
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
        ctx.term_inst = handlers::term() as *mut u8;

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
}
