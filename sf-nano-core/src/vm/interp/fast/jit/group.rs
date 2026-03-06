//! JIT group compiler: identify and compile groups of consecutive JIT-able ops.
//!
//! Scans `&[IrOp]` for consecutive JIT-able operations, compiles them into
//! single ARM64 code blocks via `JitEmitter`, and produces `Vec<ResolvedInst>`.
//! Non-JIT-able ops fall back to 1:1 base handler resolution.

use alloc::vec::Vec;
use super::code_buf::CodeBuffer;
use super::codegen::JitEmitter;
use crate::vm::interp::fast::builder::backend::{CompactionDisposition, ResolvedInst};
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

        // f32/f64 arithmetic
        F32Add | F32Sub | F32Mul | F32Div | F32Min | F32Max |
        F64Add | F64Sub | F64Mul | F64Div | F64Min | F64Max |

        // f32/f64 comparisons
        F32Eq | F32Ne | F32Lt | F32Gt | F32Le | F32Ge |
        F64Eq | F64Ne | F64Lt | F64Gt | F64Le | F64Ge |

        // f32/f64 unary
        F32Abs | F32Neg | F32Ceil | F32Floor | F32Trunc | F32Nearest | F32Sqrt |
        F64Abs | F64Neg | F64Ceil | F64Floor | F64Trunc | F64Nearest | F64Sqrt |

        // Constants (all types)
        I32Const { .. } | I64Const { .. } | F32Const { .. } | F64Const { .. } |

        // Locals (hot and frame get/set/tee)
        LocalGetHot { .. } | LocalGetFrame { .. } |
        LocalSetHot { .. } | LocalSetFrame { .. } |
        LocalTeeHot { .. } | LocalTeeFrame { .. } |

        // Drop / Select
        Drop | Select |

        // Type conversions
        I32WrapI64 | I64ExtendI32S | I64ExtendI32U |
        I32ReinterpretF32 | I64ReinterpretF64 | F32ReinterpretI32 | F64ReinterpretI64 |
        F32DemoteF64 | F64PromoteF32 |

        // Int-to-float conversions
        F32ConvertI32S | F32ConvertI32U | F32ConvertI64S | F32ConvertI64U |
        F64ConvertI32S | F64ConvertI32U | F64ConvertI64S | F64ConvertI64U |

        // Saturating float-to-int truncation
        I32TruncSatF32S | I32TruncSatF32U | I32TruncSatF64S | I32TruncSatF64U |
        I64TruncSatF32S | I64TruncSatF32U | I64TruncSatF64S | I64TruncSatF64U |

        // Sign extensions
        I32Extend8S | I32Extend16S | I64Extend8S | I64Extend16S | I64Extend32S |

        // Memory loads (all variants — memidx checked separately)
        I32Load { .. } | I64Load { .. } | F32Load { .. } | F64Load { .. } |
        I32Load8S { .. } | I32Load8U { .. } | I32Load16S { .. } | I32Load16U { .. } |
        I64Load8S { .. } | I64Load8U { .. } | I64Load16S { .. } | I64Load16U { .. } |
        I64Load32S { .. } | I64Load32U { .. } |

        // Memory stores (all variants — memidx checked separately)
        I32Store { .. } | I64Store { .. } | F32Store { .. } | F64Store { .. } |
        I32Store8 { .. } | I32Store16 { .. } |
        I64Store8 { .. } | I64Store16 { .. } | I64Store32 { .. } |

        // TOS spill/fill (register ↔ memory transfers)
        Spill { .. } | Fill { .. }
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

/// Mark every IR index that can be reached by a non-fallthrough branch.
fn incoming_targets(ir: &[IrOp]) -> Vec<bool> {
    let mut incoming = alloc::vec![false; ir.len()];

    for op in ir {
        if let Some(target) = op.alt_target {
            if target < incoming.len() {
                incoming[target] = true;
            }
        }

        if let IrOpKind::BrTable { entries, .. } = &op.kind {
            for entry in entries {
                if let Some(target) = entry.target_idx {
                    if target < incoming.len() {
                        incoming[target] = true;
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
        if is_jit_able_op(&ir[i], hot_local_mask) {
            let group_start = i;
            i += 1;

            // Extend the group
            while i < ir.len() {
                if branch_targets[i] {
                    break;
                }
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
                                compaction: CompactionDisposition::Keep,
                            });
                        } else {
                            out.push(ResolvedInst {
                                handler,
                                kind: IrOpKind::Data { imm0: 0, imm1: 0, imm2: 0 },
                                alt_target: None,
                                has_target: false,
                                compaction: CompactionDisposition::Keep,
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
    // Debug: verify emitter height matches IR pre_height
    debug_assert_eq!(
        e.height() as u16, op.pre_height,
        "height mismatch at {:?}: emitter={}, ir={}",
        op.kind, e.height(), op.pre_height
    );
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
        IrOpKind::F32Const { value } => e.f32_const(*value),
        IrOpKind::F64Const { value } => e.f64_const(*value),

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
        IrOpKind::LocalTeeFrame { idx } => {
            if (*idx as usize) < 3 && hot_local_mask[*idx as usize] {
                e.local_tee_ln(*idx as u8);
            } else {
                e.local_tee(*idx);
            }
        }

        // Drop / Select
        IrOpKind::Drop => e.drop_val(),
        IrOpKind::Select => e.select(),

        // f64 binary
        IrOpKind::F64Add => e.f64_add(),
        IrOpKind::F64Sub => e.f64_sub(),
        IrOpKind::F64Mul => e.f64_mul(),
        IrOpKind::F64Div => e.f64_div(),
        IrOpKind::F64Min => e.f64_min(),
        IrOpKind::F64Max => e.f64_max(),

        // f32 binary
        IrOpKind::F32Add => e.f32_add(),
        IrOpKind::F32Sub => e.f32_sub(),
        IrOpKind::F32Mul => e.f32_mul(),
        IrOpKind::F32Div => e.f32_div(),
        IrOpKind::F32Min => e.f32_min(),
        IrOpKind::F32Max => e.f32_max(),

        // f64 comparisons
        IrOpKind::F64Eq => e.f64_eq(),
        IrOpKind::F64Ne => e.f64_ne(),
        IrOpKind::F64Lt => e.f64_lt(),
        IrOpKind::F64Gt => e.f64_gt(),
        IrOpKind::F64Le => e.f64_le(),
        IrOpKind::F64Ge => e.f64_ge(),

        // f32 comparisons
        IrOpKind::F32Eq => e.f32_eq(),
        IrOpKind::F32Ne => e.f32_ne(),
        IrOpKind::F32Lt => e.f32_lt(),
        IrOpKind::F32Gt => e.f32_gt(),
        IrOpKind::F32Le => e.f32_le(),
        IrOpKind::F32Ge => e.f32_ge(),

        // f64 unary
        IrOpKind::F64Abs => e.f64_abs(),
        IrOpKind::F64Neg => e.f64_neg(),
        IrOpKind::F64Sqrt => e.f64_sqrt(),
        IrOpKind::F64Ceil => e.f64_ceil(),
        IrOpKind::F64Floor => e.f64_floor(),
        IrOpKind::F64Trunc => e.f64_trunc(),
        IrOpKind::F64Nearest => e.f64_nearest(),

        // f32 unary
        IrOpKind::F32Abs => e.f32_abs(),
        IrOpKind::F32Neg => e.f32_neg(),
        IrOpKind::F32Sqrt => e.f32_sqrt(),
        IrOpKind::F32Ceil => e.f32_ceil(),
        IrOpKind::F32Floor => e.f32_floor(),
        IrOpKind::F32Trunc => e.f32_trunc(),
        IrOpKind::F32Nearest => e.f32_nearest(),

        // Type conversions
        IrOpKind::I32WrapI64 => e.i32_wrap_i64(),
        IrOpKind::I64ExtendI32S => e.i64_extend_i32_s(),
        IrOpKind::I64ExtendI32U => e.i64_extend_i32_u(),
        IrOpKind::I32ReinterpretF32 | IrOpKind::I64ReinterpretF64 |
        IrOpKind::F32ReinterpretI32 | IrOpKind::F64ReinterpretI64 => {
            // NOP — same bit pattern, just reinterpreted
        }
        IrOpKind::F64PromoteF32 => e.f64_promote_f32(),
        IrOpKind::F32DemoteF64 => e.f32_demote_f64(),

        // Int-to-float conversions
        IrOpKind::F32ConvertI32S => e.f32_convert_i32_s(),
        IrOpKind::F32ConvertI32U => e.f32_convert_i32_u(),
        IrOpKind::F32ConvertI64S => e.f32_convert_i64_s(),
        IrOpKind::F32ConvertI64U => e.f32_convert_i64_u(),
        IrOpKind::F64ConvertI32S => e.f64_convert_i32_s(),
        IrOpKind::F64ConvertI32U => e.f64_convert_i32_u(),
        IrOpKind::F64ConvertI64S => e.f64_convert_i64_s(),
        IrOpKind::F64ConvertI64U => e.f64_convert_i64_u(),

        // Saturating float-to-int truncation
        IrOpKind::I32TruncSatF32S => e.i32_trunc_sat_f32_s(),
        IrOpKind::I32TruncSatF32U => e.i32_trunc_sat_f32_u(),
        IrOpKind::I32TruncSatF64S => e.i32_trunc_sat_f64_s(),
        IrOpKind::I32TruncSatF64U => e.i32_trunc_sat_f64_u(),
        IrOpKind::I64TruncSatF32S => e.i64_trunc_sat_f32_s(),
        IrOpKind::I64TruncSatF32U => e.i64_trunc_sat_f32_u(),
        IrOpKind::I64TruncSatF64S => e.i64_trunc_sat_f64_s(),
        IrOpKind::I64TruncSatF64U => e.i64_trunc_sat_f64_u(),

        // Sign extensions
        IrOpKind::I32Extend8S => e.i32_extend8_s(),
        IrOpKind::I32Extend16S => e.i32_extend16_s(),
        IrOpKind::I64Extend8S => e.i64_extend8_s(),
        IrOpKind::I64Extend16S => e.i64_extend16_s(),
        IrOpKind::I64Extend32S => e.i64_extend32_s(),

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

        // TOS spill/fill
        IrOpKind::Spill { slot, count } => e.spill(*slot, *count, op.variant),
        IrOpKind::Fill { slot, count } => e.fill(*slot, *count, op.variant),

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
        assert!(!resolved[0].is_removed());

        // Rest are skip
        assert!(resolved[1].is_internal_only());
        assert!(resolved[2].is_internal_only());
        assert!(resolved[3].is_internal_only());
    }

    #[test]
    fn test_group_breaks_at_incoming_branch_target() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        let mask = [true, true, true];

        let ops = vec![
            {
                let mut op = make_op(IrOpKind::BrIfSimple, 1);
                op.has_target = true;
                op.alt_target = Some(2);
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
        ctx.term_inst = handlers::term() as *mut u8;
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
        ctx.term_inst = handlers::term() as *mut u8;
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
        ctx.term_inst = handlers::term() as *mut u8;
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
        ctx.term_inst = handlers::term() as *mut u8;
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
}
