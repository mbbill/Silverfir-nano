use super::semantics::SemanticsEmitter;
use crate::vm::lir::{IrOp, IrOpKind, current_variant_from_window};

const LINEAR_FINISH_BYTES: usize = 16;
const COND_TERMINATOR_FINISH_BYTES: usize = 40;
const BR_TERMINATOR_FINISH_BYTES: usize = 20;
const RETURN_TERMINATOR_FINISH_BYTES: usize = 64;
const TRAP_STUB_MAX_BYTES: usize = 32;

/// Whether an IR op kind can be lowered by the ARM64 micro-JIT at all.
pub fn supports_kind(kind: &IrOpKind) -> bool {
    use IrOpKind::*;

    matches!(
        kind,
        I32Add | I32Sub | I32Mul | I32DivS | I32DivU | I32RemS | I32RemU |
        I32And | I32Or | I32Xor |
        I32Shl | I32ShrS | I32ShrU | I32Rotl | I32Rotr |
        I64Add | I64Sub | I64Mul | I64DivS | I64DivU | I64RemS | I64RemU |
        I64And | I64Or | I64Xor |
        I64Shl | I64ShrS | I64ShrU | I64Rotl | I64Rotr |
        I32Eq | I32Ne | I32LtS | I32LtU | I32GtS | I32GtU |
        I32LeS | I32LeU | I32GeS | I32GeU |
        I64Eq | I64Ne | I64LtS | I64LtU | I64GtS | I64GtU |
        I64LeS | I64LeU | I64GeS | I64GeU |
        I32Eqz | I32Clz | I32Ctz |
        I64Eqz | I64Clz | I64Ctz |
        F32Add | F32Sub | F32Mul | F32Div | F32Min | F32Max |
        F64Add | F64Sub | F64Mul | F64Div | F64Min | F64Max |
        F32Eq | F32Ne | F32Lt | F32Gt | F32Le | F32Ge |
        F64Eq | F64Ne | F64Lt | F64Gt | F64Le | F64Ge |
        F32Abs | F32Neg | F32Ceil | F32Floor | F32Trunc | F32Nearest | F32Sqrt |
        F64Abs | F64Neg | F64Ceil | F64Floor | F64Trunc | F64Nearest | F64Sqrt |
        I32Const { .. } | I64Const { .. } | F32Const { .. } | F64Const { .. } |
        LocalGetHot { .. } | LocalGetFrame { .. } |
        LocalSetHot { .. } | LocalSetFrame { .. } |
        LocalTeeHot { .. } | LocalTeeFrame { .. } |
        Drop | Select |
        I32WrapI64 | I64ExtendI32S | I64ExtendI32U |
        I32ReinterpretF32 | I64ReinterpretF64 | F32ReinterpretI32 | F64ReinterpretI64 |
        F32DemoteF64 | F64PromoteF32 |
        F32ConvertI32S | F32ConvertI32U | F32ConvertI64S | F32ConvertI64U |
        F64ConvertI32S | F64ConvertI32U | F64ConvertI64S | F64ConvertI64U |
        I32TruncSatF32S | I32TruncSatF32U | I32TruncSatF64S | I32TruncSatF64U |
        I64TruncSatF32S | I64TruncSatF32U | I64TruncSatF64S | I64TruncSatF64U |
        I32Extend8S | I32Extend16S | I64Extend8S | I64Extend16S | I64Extend32S |
        I32Load { .. } | I64Load { .. } | F32Load { .. } | F64Load { .. } |
        I32Load8S { .. } | I32Load8U { .. } | I32Load16S { .. } | I32Load16U { .. } |
        I64Load8S { .. } | I64Load8U { .. } | I64Load16S { .. } | I64Load16U { .. } |
        I64Load32S { .. } | I64Load32U { .. } |
        I32Store { .. } | I64Store { .. } | F32Store { .. } | F64Store { .. } |
        I32Store8 { .. } | I32Store16 { .. } |
        I64Store8 { .. } | I64Store16 { .. } | I64Store32 { .. } |
        MemorySize { mem_idx: 0 } |
        InitLocals { .. } |
        Spill { .. } | Fill { .. }
    )
}

/// Memory ops are only supported for memidx 0 by the current micro-JIT.
pub fn memidx_is_supported(kind: &IrOpKind) -> bool {
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

pub fn uses_trap_stub(kind: &IrOpKind) -> bool {
    use IrOpKind::*;

    matches!(
        kind,
        I32Load { .. } | I64Load { .. } | F32Load { .. } | F64Load { .. } |
        I32Load8S { .. } | I32Load8U { .. } | I32Load16S { .. } | I32Load16U { .. } |
        I64Load8S { .. } | I64Load8U { .. } | I64Load16S { .. } | I64Load16U { .. } |
        I64Load32S { .. } | I64Load32U { .. } |
        I32Store { .. } | I64Store { .. } | F32Store { .. } | F64Store { .. } |
        I32Store8 { .. } | I32Store16 { .. } |
        I64Store8 { .. } | I64Store16 { .. } | I64Store32 { .. }
    )
}

pub fn max_emitted_bytes(op: &IrOp) -> usize {
    use IrOpKind::*;

    match &op.kind {
        Drop |
        I32ReinterpretF32 | I64ReinterpretF64 |
        F32ReinterpretI32 | F64ReinterpretI64 => 0,
        LocalGetHot { .. } | LocalGetFrame { .. } |
        LocalSetHot { .. } | LocalSetFrame { .. } |
        LocalTeeHot { .. } | LocalTeeFrame { .. } => 4,
        Spill { count, .. } | Fill { count, .. } => (*count as usize) * 4,
        InitLocals { .. } => 48,
        I32Const { .. } | I64Const { .. } | F32Const { .. } | F64Const { .. } => 16,
        Select => 24,
        I32Load { .. } | I64Load { .. } | F32Load { .. } | F64Load { .. } |
        I32Load8S { .. } | I32Load8U { .. } | I32Load16S { .. } | I32Load16U { .. } |
        I64Load8S { .. } | I64Load8U { .. } | I64Load16S { .. } | I64Load16U { .. } |
        I64Load32S { .. } | I64Load32U { .. } |
        I32Store { .. } | I64Store { .. } | F32Store { .. } | F64Store { .. } |
        I32Store8 { .. } | I32Store16 { .. } |
        I64Store8 { .. } | I64Store16 { .. } | I64Store32 { .. } => 80,
        I32DivS | I32DivU | I32RemS | I32RemU |
        I64DivS | I64DivU | I64RemS | I64RemU => 96,
        MemorySize { .. } => 8,
        _ => 32,
    }
}

pub fn estimate_group_bytes(group: &[IrOp], terminator: Option<&IrOpKind>) -> usize {
    let body_end = if terminator.is_some() {
        group.len().saturating_sub(1)
    } else {
        group.len()
    };
    let body = group[..body_end]
        .iter()
        .map(max_emitted_bytes)
        .sum::<usize>();
    let finish = match terminator {
        Some(IrOpKind::BrIfSimple | IrOpKind::If) => COND_TERMINATOR_FINISH_BYTES,
        Some(IrOpKind::Br { .. }) => BR_TERMINATOR_FINISH_BYTES,
        Some(IrOpKind::ReturnVoid { .. } | IrOpKind::ReturnOne { .. } | IrOpKind::Return { .. }) => {
            RETURN_TERMINATOR_FINISH_BYTES
        }
        _ => LINEAR_FINISH_BYTES,
    };
    let trap = if group[..body_end].iter().any(|op| uses_trap_stub(&op.kind)) {
        TRAP_STUB_MAX_BYTES
    } else {
        0
    };

    body + finish + trap
}

/// Emit one already-validated JIT op.
pub fn emit<E: SemanticsEmitter>(e: &mut E, op: &IrOp, hot_local_mask: [bool; 3]) {
    match &op.kind {
        IrOpKind::I32Add => e.i32_add(),
        IrOpKind::I32Sub => e.i32_sub(),
        IrOpKind::I32Mul => e.i32_mul(),
        IrOpKind::I32DivS => e.i32_div_s(),
        IrOpKind::I32DivU => e.i32_div_u(),
        IrOpKind::I32RemS => e.i32_rem_s(),
        IrOpKind::I32RemU => e.i32_rem_u(),
        IrOpKind::I32And => e.i32_and(),
        IrOpKind::I32Or => e.i32_or(),
        IrOpKind::I32Xor => e.i32_xor(),
        IrOpKind::I32Shl => e.i32_shl(),
        IrOpKind::I32ShrU => e.i32_shr_u(),
        IrOpKind::I32ShrS => e.i32_shr_s(),
        IrOpKind::I32Rotl => e.i32_rotl(),
        IrOpKind::I32Rotr => e.i32_rotr(),
        IrOpKind::I64Add => e.i64_add(),
        IrOpKind::I64Sub => e.i64_sub(),
        IrOpKind::I64Mul => e.i64_mul(),
        IrOpKind::I64DivS => e.i64_div_s(),
        IrOpKind::I64DivU => e.i64_div_u(),
        IrOpKind::I64RemS => e.i64_rem_s(),
        IrOpKind::I64RemU => e.i64_rem_u(),
        IrOpKind::I64And => e.i64_and(),
        IrOpKind::I64Or => e.i64_or(),
        IrOpKind::I64Xor => e.i64_xor(),
        IrOpKind::I64Shl => e.i64_shl(),
        IrOpKind::I64ShrU => e.i64_shr_u(),
        IrOpKind::I64ShrS => e.i64_shr_s(),
        IrOpKind::I64Rotl => e.i64_rotl(),
        IrOpKind::I64Rotr => e.i64_rotr(),
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
        IrOpKind::I32Eqz => e.i32_eqz(),
        IrOpKind::I32Clz => e.i32_clz(),
        IrOpKind::I32Ctz => e.i32_ctz(),
        IrOpKind::I64Eqz => e.i64_eqz(),
        IrOpKind::I64Clz => e.i64_clz(),
        IrOpKind::I64Ctz => e.i64_ctz(),
        IrOpKind::I32Const { value } => e.i32_const(*value as u32),
        IrOpKind::I64Const { value } => e.i64_const(*value),
        IrOpKind::F32Const { value } => e.f32_const(*value),
        IrOpKind::F64Const { value } => e.f64_const(*value),
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
        IrOpKind::Drop => e.drop_val(),
        IrOpKind::Select => e.select(),
        IrOpKind::F64Add => e.f64_add(),
        IrOpKind::F64Sub => e.f64_sub(),
        IrOpKind::F64Mul => e.f64_mul(),
        IrOpKind::F64Div => e.f64_div(),
        IrOpKind::F64Min => e.f64_min(),
        IrOpKind::F64Max => e.f64_max(),
        IrOpKind::F32Add => e.f32_add(),
        IrOpKind::F32Sub => e.f32_sub(),
        IrOpKind::F32Mul => e.f32_mul(),
        IrOpKind::F32Div => e.f32_div(),
        IrOpKind::F32Min => e.f32_min(),
        IrOpKind::F32Max => e.f32_max(),
        IrOpKind::F64Eq => e.f64_eq(),
        IrOpKind::F64Ne => e.f64_ne(),
        IrOpKind::F64Lt => e.f64_lt(),
        IrOpKind::F64Gt => e.f64_gt(),
        IrOpKind::F64Le => e.f64_le(),
        IrOpKind::F64Ge => e.f64_ge(),
        IrOpKind::F32Eq => e.f32_eq(),
        IrOpKind::F32Ne => e.f32_ne(),
        IrOpKind::F32Lt => e.f32_lt(),
        IrOpKind::F32Gt => e.f32_gt(),
        IrOpKind::F32Le => e.f32_le(),
        IrOpKind::F32Ge => e.f32_ge(),
        IrOpKind::F64Abs => e.f64_abs(),
        IrOpKind::F64Neg => e.f64_neg(),
        IrOpKind::F64Sqrt => e.f64_sqrt(),
        IrOpKind::F64Ceil => e.f64_ceil(),
        IrOpKind::F64Floor => e.f64_floor(),
        IrOpKind::F64Trunc => e.f64_trunc(),
        IrOpKind::F64Nearest => e.f64_nearest(),
        IrOpKind::F32Abs => e.f32_abs(),
        IrOpKind::F32Neg => e.f32_neg(),
        IrOpKind::F32Sqrt => e.f32_sqrt(),
        IrOpKind::F32Ceil => e.f32_ceil(),
        IrOpKind::F32Floor => e.f32_floor(),
        IrOpKind::F32Trunc => e.f32_trunc(),
        IrOpKind::F32Nearest => e.f32_nearest(),
        IrOpKind::I32WrapI64 => e.i32_wrap_i64(),
        IrOpKind::I64ExtendI32S => e.i64_extend_i32_s(),
        IrOpKind::I64ExtendI32U => e.i64_extend_i32_u(),
        IrOpKind::I32ReinterpretF32 | IrOpKind::I64ReinterpretF64 |
        IrOpKind::F32ReinterpretI32 | IrOpKind::F64ReinterpretI64 => {}
        IrOpKind::F64PromoteF32 => e.f64_promote_f32(),
        IrOpKind::F32DemoteF64 => e.f32_demote_f64(),
        IrOpKind::F32ConvertI32S => e.f32_convert_i32_s(),
        IrOpKind::F32ConvertI32U => e.f32_convert_i32_u(),
        IrOpKind::F32ConvertI64S => e.f32_convert_i64_s(),
        IrOpKind::F32ConvertI64U => e.f32_convert_i64_u(),
        IrOpKind::F64ConvertI32S => e.f64_convert_i32_s(),
        IrOpKind::F64ConvertI32U => e.f64_convert_i32_u(),
        IrOpKind::F64ConvertI64S => e.f64_convert_i64_s(),
        IrOpKind::F64ConvertI64U => e.f64_convert_i64_u(),
        IrOpKind::I32TruncSatF32S => e.i32_trunc_sat_f32_s(),
        IrOpKind::I32TruncSatF32U => e.i32_trunc_sat_f32_u(),
        IrOpKind::I32TruncSatF64S => e.i32_trunc_sat_f64_s(),
        IrOpKind::I32TruncSatF64U => e.i32_trunc_sat_f64_u(),
        IrOpKind::I64TruncSatF32S => e.i64_trunc_sat_f32_s(),
        IrOpKind::I64TruncSatF32U => e.i64_trunc_sat_f32_u(),
        IrOpKind::I64TruncSatF64S => e.i64_trunc_sat_f64_s(),
        IrOpKind::I64TruncSatF64U => e.i64_trunc_sat_f64_u(),
        IrOpKind::I32Extend8S => e.i32_extend8_s(),
        IrOpKind::I32Extend16S => e.i32_extend16_s(),
        IrOpKind::I64Extend8S => e.i64_extend8_s(),
        IrOpKind::I64Extend16S => e.i64_extend16_s(),
        IrOpKind::I64Extend32S => e.i64_extend32_s(),
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
        IrOpKind::F32Load { offset, .. } => e.f32_load(*offset),
        IrOpKind::F64Load { offset, .. } => e.f64_load(*offset),
        IrOpKind::I32Store { offset, .. } => e.i32_store(*offset),
        IrOpKind::I32Store8 { offset, .. } => e.i32_store8(*offset),
        IrOpKind::I32Store16 { offset, .. } => e.i32_store16(*offset),
        IrOpKind::I64Store { offset, .. } => e.i64_store(*offset),
        IrOpKind::I64Store8 { offset, .. } => e.i64_store8(*offset),
        IrOpKind::I64Store16 { offset, .. } => e.i64_store16(*offset),
        IrOpKind::I64Store32 { offset, .. } => e.i64_store32(*offset),
        IrOpKind::F32Store { offset, .. } => e.i32_store(*offset),
        IrOpKind::F64Store { offset, .. } => e.i64_store(*offset),
        IrOpKind::MemorySize { mem_idx } => e.memory_size(*mem_idx),
        IrOpKind::InitLocals { k0, k1, k2 } => e.init_locals(*k0, *k1, *k2),
        IrOpKind::Spill { slot, count } => e.spill(*slot, *count, current_variant_from_window(op.window)),
        IrOpKind::Fill { slot, count } => e.fill(*slot, *count, current_variant_from_window(op.window)),
        _ => unreachable!("unsupported JIT op: {:?}", op.kind),
    }
}
