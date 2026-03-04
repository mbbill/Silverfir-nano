//! JIT group compiler: identify and compile groups of consecutive JIT-able ops.
//!
//! After `decode_and_dispatch` produces `Vec<TempInst>`, this module scans for
//! consecutive JIT-able operations, compiles them into single ARM64 code blocks
//! via `JitEmitter`, and replaces the TempInsts. The finalizer then removes the
//! NOP-ified remainder.

use alloc::vec::Vec;
use super::code_buf::CodeBuffer;
use super::codegen::JitEmitter;
use crate::opcodes::{Opcode, WasmOpcode};
use crate::vm::interp::fast::builder::{Handler, TempInst};
use crate::vm::interp::fast::encoding::PatternData;
use crate::vm::interp::fast::handlers::full_set::op_nop;

/// Check if a TempInst has BrIfSimple pattern (can only terminate a group).
/// Requires pre_height >= 1 since it reads the condition from TOS top.
fn is_br_if_simple(t: &TempInst) -> bool {
    t.pre_height >= 1
        && matches!(t.wasm_op, WasmOpcode::OP(Opcode::BR_IF))
        && matches!(t.data, PatternData::BrIfSimple { .. })
}

/// Classify whether a TempInst can be JIT-compiled.
///
/// Returns `true` for arithmetic, comparison, unary, constant, local, and drop ops.
/// br_if_simple is handled separately as a group terminator.
///
/// Height checks: ops that consume TOS values require sufficient pre_height.
/// At height < required, operands are on the frame (not in TOS registers),
/// so the JIT can't handle them.
fn is_jit_able(t: &TempInst, hot_local_mask: [bool; 3]) -> bool {
    let h = t.pre_height as usize;
    // All non-fused ops use PatternData::Raw. Fused patterns (e.g., AddConst,
    // ConstLoad) keep the first opcode but have specialized PatternData —
    // those must NOT be JIT-compiled.
    let is_raw = matches!(t.data, PatternData::Raw { .. });
    match t.wasm_op {
        WasmOpcode::OP(op) => match op {
            // i32 binary (pop 2, push 1) — need height >= 2, non-fused only
            Opcode::I32_ADD | Opcode::I32_SUB | Opcode::I32_MUL |
            Opcode::I32_AND | Opcode::I32_OR | Opcode::I32_XOR |
            Opcode::I32_SHL | Opcode::I32_SHR_U | Opcode::I32_SHR_S |
            Opcode::I32_ROTL | Opcode::I32_ROTR => is_raw && h >= 2,

            // i64 binary (pop 2, push 1) — need height >= 2, non-fused only
            Opcode::I64_ADD | Opcode::I64_SUB | Opcode::I64_MUL |
            Opcode::I64_AND | Opcode::I64_OR | Opcode::I64_XOR |
            Opcode::I64_SHL | Opcode::I64_SHR_U | Opcode::I64_SHR_S |
            Opcode::I64_ROTL | Opcode::I64_ROTR => is_raw && h >= 2,

            // i32 comparisons (pop 2, push 1) — need height >= 2, non-fused only
            Opcode::I32_EQ | Opcode::I32_NE |
            Opcode::I32_LT_S | Opcode::I32_LT_U |
            Opcode::I32_GT_S | Opcode::I32_GT_U |
            Opcode::I32_LE_S | Opcode::I32_LE_U |
            Opcode::I32_GE_S | Opcode::I32_GE_U => is_raw && h >= 2,

            // i64 comparisons (pop 2, push 1) — need height >= 2, non-fused only
            Opcode::I64_EQ | Opcode::I64_NE |
            Opcode::I64_LT_S | Opcode::I64_LT_U |
            Opcode::I64_GT_S | Opcode::I64_GT_U |
            Opcode::I64_LE_S | Opcode::I64_LE_U |
            Opcode::I64_GE_S | Opcode::I64_GE_U => is_raw && h >= 2,

            // i32 unary (pop 1, push 1) — need height >= 1, non-fused only
            Opcode::I32_EQZ | Opcode::I32_CLZ | Opcode::I32_CTZ => is_raw && h >= 1,

            // i64 unary (pop 1, push 1) — need height >= 1, non-fused only
            Opcode::I64_EQZ | Opcode::I64_CLZ | Opcode::I64_CTZ => is_raw && h >= 1,

            // Constants (push 1) — must have Const data (fused patterns like ConstLoad have different data)
            Opcode::I32_CONST | Opcode::I64_CONST => matches!(t.data, PatternData::Const { .. }),

            // Drop (pop 1) — need height >= 1, non-fused only
            Opcode::DROP => is_raw && h >= 1,

            // Local get (push 1) — always OK
            Opcode::LOCAL_GET => matches!(t.data, PatternData::LocalGet { .. }),

            // Local set (pop 1) — need height >= 1
            Opcode::LOCAL_SET => {
                h >= 1 && matches!(t.data, PatternData::LocalSet { .. })
            },

            // Local tee (read TOS top, hot locals only) — need height >= 1
            Opcode::LOCAL_TEE => match t.data {
                PatternData::LocalTee { idx } => {
                    h >= 1 && (idx as usize) < 3 && hot_local_mask[idx as usize]
                }
                _ => false,
            },

            // Memory loads (pop 1, push 1) — need height >= 1, mem0 only
            Opcode::I32_LOAD | Opcode::I64_LOAD | Opcode::F32_LOAD | Opcode::F64_LOAD |
            Opcode::I32_LOAD8_S | Opcode::I32_LOAD8_U |
            Opcode::I32_LOAD16_S | Opcode::I32_LOAD16_U |
            Opcode::I64_LOAD8_S | Opcode::I64_LOAD8_U |
            Opcode::I64_LOAD16_S | Opcode::I64_LOAD16_U |
            Opcode::I64_LOAD32_S | Opcode::I64_LOAD32_U => {
                h >= 1 && matches!(t.data, PatternData::Load { memidx, .. } if memidx == 0)
            },

            // Memory stores (pop 2, push 0) — need height >= 2, mem0 only
            Opcode::I32_STORE | Opcode::I64_STORE | Opcode::F32_STORE | Opcode::F64_STORE |
            Opcode::I32_STORE8 | Opcode::I32_STORE16 |
            Opcode::I64_STORE8 | Opcode::I64_STORE16 | Opcode::I64_STORE32 => {
                h >= 2 && matches!(t.data, PatternData::Store { memidx, .. } if memidx == 0)
            },

            _ => false,
        },
        _ => false,
    }
}

/// Global JIT compilation statistics (for diagnostics).
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

/// Snapshot of JIT compilation statistics.
pub struct JitStatsSnapshot {
    pub groups: usize,
    pub ops: usize,
    pub bytes_emitted: usize,
    pub groups_skipped: usize,
    pub ops_skipped: usize,
}

/// Return a snapshot of all JIT statistics.
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

/// Return (groups_compiled, ops_compiled) since process start.
pub fn jit_stats() -> (usize, usize) {
    (
        JIT_STATS.groups.load(core::sync::atomic::Ordering::Relaxed),
        JIT_STATS.ops.load(core::sync::atomic::Ordering::Relaxed),
    )
}

/// Return (groups_skipped, ops_skipped) due to buffer capacity exhaustion.
pub fn jit_capacity_skips() -> (usize, usize) {
    (
        JIT_STATS.groups_skipped_capacity.load(core::sync::atomic::Ordering::Relaxed),
        JIT_STATS.ops_skipped_capacity.load(core::sync::atomic::Ordering::Relaxed),
    )
}

/// Scan for consecutive JIT-able ops, compile groups into ARM64 code.
///
/// Groups of 2+ consecutive JIT-able instructions are compiled into a single
/// ARM64 code block. The first TempInst gets the JIT handler, remaining ones
/// become NOPs (removed by the finalizer).
pub fn compile_jit_groups(
    temps: &mut Vec<TempInst>,
    buf: &mut CodeBuffer,
    hot_local_mask: [bool; 3],
) {
    buf.begin_write();
    let bytes_before = buf.len();

    let mut groups_compiled: usize = 0;
    let mut ops_compiled: usize = 0;
    let mut groups_skipped: usize = 0;
    let mut ops_skipped: usize = 0;

    let mut i = 0;
    while i < temps.len() {
        if is_jit_able(&temps[i], hot_local_mask) || is_br_if_simple(&temps[i]) {
            let group_start = i;

            // br_if_simple as first op: only valid as terminator, skip
            if is_br_if_simple(&temps[i]) {
                i += 1;
                // Single br_if_simple is not a group (size < 2)
                continue;
            }

            i += 1;
            // Extend the group
            while i < temps.len() {
                if is_br_if_simple(&temps[i]) {
                    i += 1; // include br_if_simple as terminator
                    break;
                }
                if !is_jit_able(&temps[i], hot_local_mask) {
                    break;
                }
                i += 1;
            }

            let group_len = i - group_start;
            if group_len >= 2 {
                // Conservative capacity check: ~256 bytes per op + 256 bytes for
                // dispatch stub + trap stub overhead.
                let estimated_bytes = group_len * 256 + 256;
                if buf.remaining() >= estimated_bytes {
                    compile_group(&mut temps[group_start..i], buf, hot_local_mask);
                    groups_compiled += 1;
                    ops_compiled += group_len;
                } else {
                    groups_skipped += 1;
                    ops_skipped += group_len;
                }
            }
        } else {
            i += 1;
        }
    }

    let total_len = buf.len();
    // Always finish_write to restore thread to execute mode (macOS JIT protection).
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
}

/// Compute the stack effect of a JIT-able op: (pops, pushes).
fn op_stack_effect(t: &TempInst) -> (usize, usize) {
    match t.wasm_op {
        WasmOpcode::OP(op) => match op {
            // Binops / comparisons: pop 2, push 1
            Opcode::I32_ADD | Opcode::I32_SUB | Opcode::I32_MUL |
            Opcode::I32_AND | Opcode::I32_OR | Opcode::I32_XOR |
            Opcode::I32_SHL | Opcode::I32_SHR_U | Opcode::I32_SHR_S |
            Opcode::I32_ROTL | Opcode::I32_ROTR |
            Opcode::I64_ADD | Opcode::I64_SUB | Opcode::I64_MUL |
            Opcode::I64_AND | Opcode::I64_OR | Opcode::I64_XOR |
            Opcode::I64_SHL | Opcode::I64_SHR_U | Opcode::I64_SHR_S |
            Opcode::I64_ROTL | Opcode::I64_ROTR |
            Opcode::I32_EQ | Opcode::I32_NE |
            Opcode::I32_LT_S | Opcode::I32_LT_U |
            Opcode::I32_GT_S | Opcode::I32_GT_U |
            Opcode::I32_LE_S | Opcode::I32_LE_U |
            Opcode::I32_GE_S | Opcode::I32_GE_U |
            Opcode::I64_EQ | Opcode::I64_NE |
            Opcode::I64_LT_S | Opcode::I64_LT_U |
            Opcode::I64_GT_S | Opcode::I64_GT_U |
            Opcode::I64_LE_S | Opcode::I64_LE_U |
            Opcode::I64_GE_S | Opcode::I64_GE_U => (2, 1),
            // Unary: pop 1, push 1
            Opcode::I32_EQZ | Opcode::I32_CLZ | Opcode::I32_CTZ |
            Opcode::I64_EQZ | Opcode::I64_CLZ | Opcode::I64_CTZ => (1, 1),
            // Constants / local_get: push 1
            Opcode::I32_CONST | Opcode::I64_CONST | Opcode::LOCAL_GET => (0, 1),
            // Drop / local_set: pop 1
            Opcode::DROP | Opcode::LOCAL_SET => (1, 0),
            // Local tee: read top, write local (no height change)
            Opcode::LOCAL_TEE => (0, 0),

            // Memory loads: pop 1, push 1
            Opcode::I32_LOAD | Opcode::I64_LOAD | Opcode::F32_LOAD | Opcode::F64_LOAD |
            Opcode::I32_LOAD8_S | Opcode::I32_LOAD8_U |
            Opcode::I32_LOAD16_S | Opcode::I32_LOAD16_U |
            Opcode::I64_LOAD8_S | Opcode::I64_LOAD8_U |
            Opcode::I64_LOAD16_S | Opcode::I64_LOAD16_U |
            Opcode::I64_LOAD32_S | Opcode::I64_LOAD32_U => (1, 1),

            // Memory stores: pop 2, push 0
            Opcode::I32_STORE | Opcode::I64_STORE | Opcode::F32_STORE | Opcode::F64_STORE |
            Opcode::I32_STORE8 | Opcode::I32_STORE16 |
            Opcode::I64_STORE8 | Opcode::I64_STORE16 | Opcode::I64_STORE32 => (2, 0),

            _ => (0, 0),
        },
        _ => (0, 0),
    }
}

/// Compile a group of TempInsts into a single JIT handler.
fn compile_group(
    group: &mut [TempInst],
    buf: &mut CodeBuffer,
    hot_local_mask: [bool; 3],
) {
    let ends_with_brif = is_br_if_simple(group.last().unwrap());
    let branch_alt_idx = if ends_with_brif {
        group.last().unwrap().alt_idx
    } else {
        None
    };

    // Validate: check that heights stay valid throughout the group.
    let mut sim_height = group[0].pre_height as usize;
    let body_end_idx = if ends_with_brif { group.len() - 1 } else { group.len() };
    for t in &group[..body_end_idx] {
        let (pops, pushes) = op_stack_effect(t);
        if sim_height < pops {
            return; // Height underflow — bail, don't JIT this group
        }
        sim_height = sim_height - pops + pushes;
    }
    if ends_with_brif && sim_height < 1 {
        return; // br_if_simple needs height >= 1
    }

    let mut e = JitEmitter::new(buf, group[0].pre_height as usize);

    // Compile body ops (all ops if linear, all-but-last if br_if_simple terminator)
    let body_end = if ends_with_brif { group.len() - 1 } else { group.len() };

    for t in group[..body_end].iter() {
        emit_op(&mut e, t, hot_local_mask);
    }

    // Finish: dispatch stub
    let start = if ends_with_brif {
        e.finish_br_if_simple()
    } else {
        e.finish()
    };

    // Get JIT handler
    let jit_handler: Handler = unsafe { buf.fn_ptr(start) };

    // Replace first TempInst with JIT handler
    group[0].handler = jit_handler;
    if ends_with_brif {
        group[0].data = PatternData::BrIfSimple {};
        group[0].has_target = true;
        group[0].alt_idx = branch_alt_idx;
    } else {
        group[0].data = PatternData::Raw { imm0: 0, imm1: 0, imm2: 0 };
        group[0].has_target = false;
    }
    // Keep original wasm_op so finalizer doesn't remove it

    // NOP-ify remaining TempInsts
    for t in &mut group[1..] {
        t.handler = op_nop;
        t.wasm_op = WasmOpcode::OP(Opcode::NOP);
        t.has_target = false;
        t.alt_idx = None;
    }
}

/// Emit a single op via JitEmitter based on its wasm_op and pattern data.
fn emit_op(e: &mut JitEmitter, t: &TempInst, hot_local_mask: [bool; 3]) {
    match t.wasm_op {
        WasmOpcode::OP(op) => match op {
            // i32 binary
            Opcode::I32_ADD => e.i32_add(),
            Opcode::I32_SUB => e.i32_sub(),
            Opcode::I32_MUL => e.i32_mul(),
            Opcode::I32_AND => e.i32_and(),
            Opcode::I32_OR => e.i32_or(),
            Opcode::I32_XOR => e.i32_xor(),
            Opcode::I32_SHL => e.i32_shl(),
            Opcode::I32_SHR_U => e.i32_shr_u(),
            Opcode::I32_SHR_S => e.i32_shr_s(),
            Opcode::I32_ROTL => e.i32_rotl(),
            Opcode::I32_ROTR => e.i32_rotr(),

            // i64 binary
            Opcode::I64_ADD => e.i64_add(),
            Opcode::I64_SUB => e.i64_sub(),
            Opcode::I64_MUL => e.i64_mul(),
            Opcode::I64_AND => e.i64_and(),
            Opcode::I64_OR => e.i64_or(),
            Opcode::I64_XOR => e.i64_xor(),
            Opcode::I64_SHL => e.i64_shl(),
            Opcode::I64_SHR_U => e.i64_shr_u(),
            Opcode::I64_SHR_S => e.i64_shr_s(),
            Opcode::I64_ROTL => e.i64_rotl(),
            Opcode::I64_ROTR => e.i64_rotr(),

            // i32 comparisons
            Opcode::I32_EQ => e.i32_eq(),
            Opcode::I32_NE => e.i32_ne(),
            Opcode::I32_LT_S => e.i32_lt_s(),
            Opcode::I32_LT_U => e.i32_lt_u(),
            Opcode::I32_GT_S => e.i32_gt_s(),
            Opcode::I32_GT_U => e.i32_gt_u(),
            Opcode::I32_LE_S => e.i32_le_s(),
            Opcode::I32_LE_U => e.i32_le_u(),
            Opcode::I32_GE_S => e.i32_ge_s(),
            Opcode::I32_GE_U => e.i32_ge_u(),

            // i64 comparisons
            Opcode::I64_EQ => e.i64_eq(),
            Opcode::I64_NE => e.i64_ne(),
            Opcode::I64_LT_S => e.i64_lt_s(),
            Opcode::I64_LT_U => e.i64_lt_u(),
            Opcode::I64_GT_S => e.i64_gt_s(),
            Opcode::I64_GT_U => e.i64_gt_u(),
            Opcode::I64_LE_S => e.i64_le_s(),
            Opcode::I64_LE_U => e.i64_le_u(),
            Opcode::I64_GE_S => e.i64_ge_s(),
            Opcode::I64_GE_U => e.i64_ge_u(),

            // i32 unary
            Opcode::I32_EQZ => e.i32_eqz(),
            Opcode::I32_CLZ => e.i32_clz(),
            Opcode::I32_CTZ => e.i32_ctz(),

            // i64 unary
            Opcode::I64_EQZ => e.i64_eqz(),
            Opcode::I64_CLZ => e.i64_clz(),
            Opcode::I64_CTZ => e.i64_ctz(),

            // Constants
            Opcode::I32_CONST => {
                if let PatternData::Const { value } = t.data {
                    e.i32_const(value as u32);
                }
            }
            Opcode::I64_CONST => {
                if let PatternData::Const { value } = t.data {
                    e.i64_const(value);
                }
            }

            // Locals
            Opcode::LOCAL_GET => {
                if let PatternData::LocalGet { idx } = t.data {
                    if (idx as usize) < 3 && hot_local_mask[idx as usize] {
                        e.local_get_ln(idx as u8);
                    } else {
                        e.local_get(idx);
                    }
                }
            }
            Opcode::LOCAL_SET => {
                if let PatternData::LocalSet { idx } = t.data {
                    if (idx as usize) < 3 && hot_local_mask[idx as usize] {
                        e.local_set_ln(idx as u8);
                    } else {
                        e.local_set(idx);
                    }
                }
            }
            Opcode::LOCAL_TEE => {
                if let PatternData::LocalTee { idx } = t.data {
                    e.local_tee_ln(idx as u8);
                }
            }

            // Drop
            Opcode::DROP => e.drop_val(),

            // Memory loads
            Opcode::I32_LOAD => {
                if let PatternData::Load { offset, .. } = t.data { e.i32_load(offset); }
            }
            Opcode::I32_LOAD8_S => {
                if let PatternData::Load { offset, .. } = t.data { e.i32_load8_s(offset); }
            }
            Opcode::I32_LOAD8_U => {
                if let PatternData::Load { offset, .. } = t.data { e.i32_load8_u(offset); }
            }
            Opcode::I32_LOAD16_S => {
                if let PatternData::Load { offset, .. } = t.data { e.i32_load16_s(offset); }
            }
            Opcode::I32_LOAD16_U => {
                if let PatternData::Load { offset, .. } = t.data { e.i32_load16_u(offset); }
            }
            Opcode::I64_LOAD => {
                if let PatternData::Load { offset, .. } = t.data { e.i64_load(offset); }
            }
            Opcode::I64_LOAD8_S => {
                if let PatternData::Load { offset, .. } = t.data { e.i64_load8_s(offset); }
            }
            Opcode::I64_LOAD8_U => {
                if let PatternData::Load { offset, .. } = t.data { e.i64_load8_u(offset); }
            }
            Opcode::I64_LOAD16_S => {
                if let PatternData::Load { offset, .. } = t.data { e.i64_load16_s(offset); }
            }
            Opcode::I64_LOAD16_U => {
                if let PatternData::Load { offset, .. } = t.data { e.i64_load16_u(offset); }
            }
            Opcode::I64_LOAD32_S => {
                if let PatternData::Load { offset, .. } = t.data { e.i64_load32_s(offset); }
            }
            Opcode::I64_LOAD32_U => {
                if let PatternData::Load { offset, .. } = t.data { e.i64_load32_u(offset); }
            }
            Opcode::F32_LOAD => {
                if let PatternData::Load { offset, .. } = t.data { e.i32_load(offset); }
            }
            Opcode::F64_LOAD => {
                if let PatternData::Load { offset, .. } = t.data { e.i64_load(offset); }
            }

            // Memory stores
            Opcode::I32_STORE => {
                if let PatternData::Store { offset, .. } = t.data { e.i32_store(offset); }
            }
            Opcode::I32_STORE8 => {
                if let PatternData::Store { offset, .. } = t.data { e.i32_store8(offset); }
            }
            Opcode::I32_STORE16 => {
                if let PatternData::Store { offset, .. } = t.data { e.i32_store16(offset); }
            }
            Opcode::I64_STORE => {
                if let PatternData::Store { offset, .. } = t.data { e.i64_store(offset); }
            }
            Opcode::I64_STORE8 => {
                if let PatternData::Store { offset, .. } = t.data { e.i64_store8(offset); }
            }
            Opcode::I64_STORE16 => {
                if let PatternData::Store { offset, .. } = t.data { e.i64_store16(offset); }
            }
            Opcode::I64_STORE32 => {
                if let PatternData::Store { offset, .. } = t.data { e.i64_store32(offset); }
            }
            Opcode::F32_STORE => {
                if let PatternData::Store { offset, .. } = t.data { e.i32_store(offset); }
            }
            Opcode::F64_STORE => {
                if let PatternData::Store { offset, .. } = t.data { e.i64_store(offset); }
            }

            _ => {} // Should not reach here (is_jit_able filters)
        },
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use alloc::vec;
    use super::*;
    use crate::vm::interp::fast::builder::TempInst;
    use crate::vm::interp::fast::encoding::PatternData;
    use crate::vm::interp::fast::handlers::full_set;
    use crate::vm::interp::fast::instruction::Instruction;
    use crate::vm::interp::fast::handlers::{self, OpHandler, NextHandler, run_trampoline};
    use crate::vm::interp::fast::context::Context;
    use crate::vm::interp::fast::jit::codegen::{depth_variant, tos_reg};
    use crate::vm::interp::fast::jit::reg::Reg;
    use crate::vm::interp::fast::jit::arm64_enc;
    use crate::vm::interp::fast::jit::emit;

    /// Helper: create a TempInst for a given opcode and pattern data.
    fn make_temp(op: Opcode, data: PatternData) -> TempInst {
        let mut t = TempInst::new(
            full_set::op_nop,
            data,
            WasmOpcode::OP(op),
        );
        t.pre_height = 0;
        t
    }

    // ==================== Classification tests ====================

    #[test]
    fn test_is_jit_able_arithmetic() {
        let mask = [true, true, true];
        // Binops need height >= 2
        let t = make_temp_with_height(Opcode::I32_ADD, PatternData::Raw { imm0: 0, imm1: 0, imm2: 0 }, 2);
        assert!(is_jit_able(&t, mask));

        let t = make_temp_with_height(Opcode::I64_MUL, PatternData::Raw { imm0: 0, imm1: 0, imm2: 0 }, 3);
        assert!(is_jit_able(&t, mask));

        let t = make_temp_with_height(Opcode::I32_EQ, PatternData::Raw { imm0: 0, imm1: 0, imm2: 0 }, 2);
        assert!(is_jit_able(&t, mask));

        // Unary needs height >= 1
        let t = make_temp_with_height(Opcode::I32_EQZ, PatternData::Raw { imm0: 0, imm1: 0, imm2: 0 }, 1);
        assert!(is_jit_able(&t, mask));

        // Binop at height 1 → NOT jit-able (insufficient TOS values)
        let t = make_temp_with_height(Opcode::I32_ADD, PatternData::Raw { imm0: 0, imm1: 0, imm2: 0 }, 1);
        assert!(!is_jit_able(&t, mask));

        // Unary at height 0 → NOT jit-able
        let t = make_temp_with_height(Opcode::I32_EQZ, PatternData::Raw { imm0: 0, imm1: 0, imm2: 0 }, 0);
        assert!(!is_jit_able(&t, mask));
    }

    #[test]
    fn test_is_jit_able_const() {
        let mask = [true, true, true];
        let t = make_temp(Opcode::I32_CONST, PatternData::Const { value: 42 });
        assert!(is_jit_able(&t, mask));

        let t = make_temp(Opcode::I64_CONST, PatternData::Const { value: 100 });
        assert!(is_jit_able(&t, mask));
    }

    #[test]
    fn test_is_jit_able_locals() {
        let mask = [true, true, false];

        // Hot local get (idx=0, mask[0]=true) — push, any height
        let t = make_temp(Opcode::LOCAL_GET, PatternData::LocalGet { idx: 0 });
        assert!(is_jit_able(&t, mask));

        // Non-hot local get (idx=5) — push, any height
        let t = make_temp(Opcode::LOCAL_GET, PatternData::LocalGet { idx: 5 });
        assert!(is_jit_able(&t, mask));

        // Hot local set — needs height >= 1
        let t = make_temp_with_height(Opcode::LOCAL_SET, PatternData::LocalSet { idx: 1 }, 1);
        assert!(is_jit_able(&t, mask));

        // Non-hot local set (idx=2, mask[2]=false) — needs height >= 1
        let t = make_temp_with_height(Opcode::LOCAL_SET, PatternData::LocalSet { idx: 2 }, 1);
        assert!(is_jit_able(&t, mask));

        // Local set at height 0 → NOT jit-able
        let t = make_temp_with_height(Opcode::LOCAL_SET, PatternData::LocalSet { idx: 1 }, 0);
        assert!(!is_jit_able(&t, mask));

        // Hot local tee (idx=0, mask[0]=true) — needs height >= 1
        let t = make_temp_with_height(Opcode::LOCAL_TEE, PatternData::LocalTee { idx: 0 }, 1);
        assert!(is_jit_able(&t, mask));

        // Non-hot local tee (idx=2, mask[2]=false) → NOT jit-able
        let t = make_temp_with_height(Opcode::LOCAL_TEE, PatternData::LocalTee { idx: 2 }, 1);
        assert!(!is_jit_able(&t, mask));
    }

    #[test]
    fn test_is_jit_able_drop() {
        let mask = [true, true, true];
        // Drop needs height >= 1
        let t = make_temp_with_height(Opcode::DROP, PatternData::Raw { imm0: 0, imm1: 0, imm2: 0 }, 1);
        assert!(is_jit_able(&t, mask));

        // Drop at height 0 → NOT jit-able
        let t = make_temp(Opcode::DROP, PatternData::Raw { imm0: 0, imm1: 0, imm2: 0 });
        assert!(!is_jit_able(&t, mask));
    }

    #[test]
    fn test_is_jit_able_non_jitable() {
        let mask = [true, true, true];

        // CALL
        let t = make_temp(Opcode::CALL, PatternData::Raw { imm0: 0, imm1: 0, imm2: 0 });
        assert!(!is_jit_able(&t, mask));

        // BLOCK
        let t = make_temp(Opcode::BLOCK, PatternData::Raw { imm0: 0, imm1: 0, imm2: 0 });
        assert!(!is_jit_able(&t, mask));

        // NOP (spill/fill marker)
        let t = make_temp(Opcode::NOP, PatternData::Raw { imm0: 0, imm1: 0, imm2: 0 });
        assert!(!is_jit_able(&t, mask));

        // I32_DIV_S (not supported by JIT)
        let t = make_temp(Opcode::I32_DIV_S, PatternData::Raw { imm0: 0, imm1: 0, imm2: 0 });
        assert!(!is_jit_able(&t, mask));

        // GLOBAL_GET
        let t = make_temp(Opcode::GLOBAL_GET, PatternData::Raw { imm0: 0, imm1: 0, imm2: 0 });
        assert!(!is_jit_able(&t, mask));
    }

    // ==================== Group identification tests ====================

    #[test]
    fn test_group_identification_basic() {
        // [const, const, add, SPILL, const, const, mul]
        // Should produce 2 groups: [const,const,add] and [const,const,mul]
        let mut buf = CodeBuffer::new().expect("mmap failed");
        let mask = [true, true, true];

        let mut temps = vec![
            make_temp_with_height(Opcode::I32_CONST, PatternData::Const { value: 5 }, 0),
            make_temp_with_height(Opcode::I32_CONST, PatternData::Const { value: 3 }, 1),
            make_temp_with_height(Opcode::I32_ADD, PatternData::Raw { imm0: 0, imm1: 0, imm2: 0 }, 2),
            // Spill (uses NOP wasm_op)
            make_temp_with_height(Opcode::NOP, PatternData::Raw { imm0: 0, imm1: 0, imm2: 0 }, 1),
            make_temp_with_height(Opcode::I32_CONST, PatternData::Const { value: 7 }, 0),
            make_temp_with_height(Opcode::I32_CONST, PatternData::Const { value: 6 }, 1),
            make_temp_with_height(Opcode::I32_MUL, PatternData::Raw { imm0: 0, imm1: 0, imm2: 0 }, 2),
        ];

        compile_jit_groups(&mut temps, &mut buf, mask);

        // Group 1: temps[0] has JIT handler, temps[1] and temps[2] are NOP
        assert_ne!(temps[0].handler as usize, op_nop as usize, "first group should have JIT handler");
        assert_eq!(temps[1].handler as usize, op_nop as usize, "second in group should be NOP");
        assert_eq!(temps[2].handler as usize, op_nop as usize, "third in group should be NOP");

        // Spill untouched
        assert_eq!(temps[3].wasm_op, WasmOpcode::OP(Opcode::NOP));

        // Group 2: temps[4] has JIT handler, temps[5] and temps[6] are NOP
        assert_ne!(temps[4].handler as usize, op_nop as usize, "second group should have JIT handler");
        assert_eq!(temps[5].handler as usize, op_nop as usize);
        assert_eq!(temps[6].handler as usize, op_nop as usize);
    }

    #[test]
    fn test_single_op_not_grouped() {
        // [const, CALL, const] — single ops, should NOT be JIT-compiled
        let mut buf = CodeBuffer::new().expect("mmap failed");
        let mask = [true, true, true];

        let original_handler = full_set::op_nop as usize;

        let mut temps = vec![
            make_temp(Opcode::I32_CONST, PatternData::Const { value: 5 }),
            make_temp(Opcode::CALL, PatternData::Raw { imm0: 0, imm1: 0, imm2: 0 }),
            make_temp(Opcode::I32_CONST, PatternData::Const { value: 3 }),
        ];

        compile_jit_groups(&mut temps, &mut buf, mask);

        // None should be modified (all single ops, no group >= 2)
        assert_eq!(temps[0].handler as usize, original_handler);
        assert_eq!(temps[1].handler as usize, original_handler);
        assert_eq!(temps[2].handler as usize, original_handler);
    }

    #[test]
    fn test_group_with_br_if_simple() {
        // [const, const, eq, br_if_simple] → group of 4 with br_if terminator
        let mut buf = CodeBuffer::new().expect("mmap failed");
        let mask = [true, true, true];

        let mut temps = vec![
            make_temp_with_height(Opcode::I32_CONST, PatternData::Const { value: 5 }, 0),
            make_temp_with_height(Opcode::I32_CONST, PatternData::Const { value: 5 }, 1),
            make_temp_with_height(Opcode::I32_EQ, PatternData::Raw { imm0: 0, imm1: 0, imm2: 0 }, 2),
            {
                let mut t = make_temp_with_height(Opcode::BR_IF, PatternData::BrIfSimple {}, 1);
                t.has_target = true;
                t.alt_idx = Some(10);
                t
            },
        ];

        compile_jit_groups(&mut temps, &mut buf, mask);

        // First should have JIT handler with br_if_simple pattern
        assert_ne!(temps[0].handler as usize, op_nop as usize);
        assert!(matches!(temps[0].data, PatternData::BrIfSimple { .. }));
        assert!(temps[0].has_target);
        assert_eq!(temps[0].alt_idx, Some(10));

        // Rest are NOP
        assert_eq!(temps[1].handler as usize, op_nop as usize);
        assert_eq!(temps[2].handler as usize, op_nop as usize);
        assert_eq!(temps[3].handler as usize, op_nop as usize);
    }

    // ==================== Execution tests ====================

    /// Run a JIT group handler and return TOS top value.
    fn run_group_test(
        handler: OpHandler,
        initial_height: usize,
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

    /// Helper: create TempInst with pre_height set.
    fn make_temp_with_height(op: Opcode, data: PatternData, height: u16) -> TempInst {
        let mut t = TempInst::new(
            full_set::op_nop,
            data,
            WasmOpcode::OP(op),
        );
        t.pre_height = height;
        t
    }

    #[test]
    fn test_compile_group_const_add() {
        // Group: const(5), const(3), i32_add → result = 8
        let mut buf = CodeBuffer::new().expect("mmap failed");
        let mask = [true, true, true];

        let mut temps = vec![
            make_temp_with_height(Opcode::I32_CONST, PatternData::Const { value: 5 }, 0),
            make_temp_with_height(Opcode::I32_CONST, PatternData::Const { value: 3 }, 1),
            make_temp_with_height(Opcode::I32_ADD, PatternData::Raw { imm0: 0, imm1: 0, imm2: 0 }, 2),
        ];

        compile_jit_groups(&mut temps, &mut buf, mask);

        // The JIT handler is now in temps[0]. But to test execution, we need
        // to build a proper handler that stores the result. Instead, let's
        // manually compile a group that includes a store to fp[0] for verification.
        let mut buf2 = CodeBuffer::new().expect("mmap failed");
        buf2.begin_write();
        let mut e = JitEmitter::new(&mut buf2, 0);
        e.i32_const(5);
        e.i32_const(3);
        e.i32_add();
        // Store result to fp[0]
        let result_reg = tos_reg(depth_variant(e.height()), 1);
        e.emit_raw(arm64_enc::str_64(result_reg, Reg::FP, 0));
        let start = e.finish();
        let total = buf2.len();
        buf2.finish_write(0, total);

        let handler: OpHandler = unsafe { buf2.fn_ptr(start) };
        let result = run_group_test(handler, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(result, 8);
    }

    #[test]
    fn test_compile_group_with_locals() {
        // Group: local_get_l0, local_get_l1, i32_add → l0 + l1
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
        // Group: const(42), local_set_l0 → l0 should be 42
        let mut buf = CodeBuffer::new().expect("mmap failed");
        buf.begin_write();

        let mut e = JitEmitter::new(&mut buf, 0);
        e.i32_const(42);
        e.local_set_ln(0);
        // Store l0 to fp[0] for verification
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
        // Reset stats
        JIT_STATS.groups.store(0, core::sync::atomic::Ordering::Relaxed);
        JIT_STATS.ops.store(0, core::sync::atomic::Ordering::Relaxed);

        let mut buf = CodeBuffer::new().expect("mmap failed");
        let mask = [true, true, true];

        let mut temps = vec![
            make_temp_with_height(Opcode::I32_CONST, PatternData::Const { value: 5 }, 0),
            make_temp_with_height(Opcode::I32_CONST, PatternData::Const { value: 3 }, 1),
            make_temp_with_height(Opcode::I32_ADD, PatternData::Raw { imm0: 0, imm1: 0, imm2: 0 }, 2),
        ];

        compile_jit_groups(&mut temps, &mut buf, mask);

        let (groups, ops) = jit_stats();
        assert!(groups >= 1, "expected at least 1 JIT group compiled, got {}", groups);
        assert!(ops >= 3, "expected at least 3 ops compiled, got {}", ops);
    }
}
