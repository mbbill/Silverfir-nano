//! Free helper functions shared across instruction and control-flow emitters.

use crate::{
    error::WasmError,
    vm::{
        machine::machine_ir::{
            MachineConvertOp, MachineFloatWidth, MachineTrapKind, MachineValue,
        },
        runtime::context::ctx_offset,
    },
};

use super::{
    abi::{
        emit_shared_epilogue, fp_machine_reg, map_fixed_reg, map_reg, FP_SCRATCH0, FP_SCRATCH1,
        FP_SCRATCH2, SCRATCH0, SCRATCH1,
    },
    armv7a_raise_trap,
    enc::{self, Cond},
    reg::Arm32Reg,
};

use crate::vm::machine::machine_ir::MACHINE_CTX_REG;
use super::compile::FunctionCompiler;

// ─── Trap-site encoding ─────────────────────────────────────────────────────

pub(super) const TRAP_SITE_BLOCK_BITS: u32 = 12;
pub(super) const TRAP_SITE_BLOCK_MASK: u32 = (1 << TRAP_SITE_BLOCK_BITS) - 1;
pub(super) const TRAP_SITE_UNKNOWN_BLOCK: u32 = TRAP_SITE_BLOCK_MASK;

#[inline]
pub(super) fn encode_trap_site(func_idx: u32, block_idx: Option<u32>) -> u32 {
    let block = block_idx
        .filter(|&block| block < TRAP_SITE_UNKNOWN_BLOCK)
        .unwrap_or(TRAP_SITE_UNKNOWN_BLOCK);
    (func_idx << TRAP_SITE_BLOCK_BITS) | block
}

// ─── Host call ──────────────────────────────────────────────────────────────

#[inline]
pub(super) fn emit_host_call(fc: &mut FunctionCompiler<'_>, target: usize) {
    fc.emit_load_addr(SCRATCH0, target);
    fc.text.emit_u32(enc::blx_reg(SCRATCH0));
}

// ─── GP value materialization ───────────────────────────────────────────────

pub(super) fn emit_move_gp_value(
    fc: &mut FunctionCompiler<'_>,
    dst: Arm32Reg,
    value: &MachineValue,
) -> Result<(), WasmError> {
    match value {
        MachineValue::Reg(r) => {
            let src = map_reg(*r)?;
            if dst != src {
                fc.text.emit_u32(enc::mov_reg(dst, src));
            }
        }
        MachineValue::Imm64(value) => fc.emit_load_u32(dst, *value as u32),
    }
    Ok(())
}

pub(super) fn materialize_gp_value(
    fc: &mut FunctionCompiler<'_>,
    value: &MachineValue,
    scratch: Arm32Reg,
) -> Result<Arm32Reg, WasmError> {
    match value {
        MachineValue::Reg(r) => map_reg(*r),
        MachineValue::Imm64(v) => {
            fc.emit_load_u32(scratch, *v as u32);
            Ok(scratch)
        }
    }
}

pub(super) fn materialize_gp_into(
    fc: &mut FunctionCompiler<'_>,
    dst: Arm32Reg,
    value: &MachineValue,
) -> Result<(), WasmError> {
    match value {
        MachineValue::Reg(r) => {
            let src = map_reg(*r)?;
            if dst != src {
                fc.text.emit_u32(enc::mov_reg(dst, src));
            }
        }
        MachineValue::Imm64(v) => {
            fc.emit_load_u32(dst, *v as u32);
        }
    }
    Ok(())
}

// ─── FP value materialization ───────────────────────────────────────────────

pub(super) fn materialize_float_value_dreg(
    fc: &mut FunctionCompiler<'_>,
    width: MachineFloatWidth,
    val: &MachineValue,
    scratch_d: u32,
) -> Result<u32, WasmError> {
    match val {
        MachineValue::Reg(r) => fc.map_fp_dreg(*r),
        MachineValue::Imm64(bits) => {
            match width {
                MachineFloatWidth::F64 => {
                    fc.emit_load_u32(Arm32Reg::R0, *bits as u32);
                    fc.emit_load_u32(Arm32Reg::R1, (*bits >> 32) as u32);
                    fc.text
                        .emit_u32(enc::vmov_d_rr(scratch_d, Arm32Reg::R0, Arm32Reg::R1));
                }
                MachineFloatWidth::F32 => {
                    fc.emit_load_u32(SCRATCH0, *bits as u32);
                    fc.text.emit_u32(enc::vmov_s_r(scratch_d * 2, SCRATCH0));
                }
            }
            Ok(scratch_d)
        }
    }
}

// ─── Pair argument / result shuffling ───────────────────────────────────────

pub(super) fn emit_pair_args_to_r0_r1(
    fc: &mut FunctionCompiler<'_>,
    src_lo: &MachineValue,
    src_hi: &MachineValue,
) -> Result<(), WasmError> {
    let src_lo_reg = match src_lo {
        MachineValue::Reg(r) => Some(map_reg(*r)?),
        MachineValue::Imm64(_) => None,
    };
    let src_hi_reg = match src_hi {
        MachineValue::Reg(r) => Some(map_reg(*r)?),
        MachineValue::Imm64(_) => None,
    };

    if matches!(src_lo_reg, Some(Arm32Reg::R1)) && matches!(src_hi_reg, Some(Arm32Reg::R0)) {
        fc.text.emit_u32(enc::mov_reg(SCRATCH0, Arm32Reg::R0));
        fc.text.emit_u32(enc::mov_reg(Arm32Reg::R0, Arm32Reg::R1));
        fc.text.emit_u32(enc::mov_reg(Arm32Reg::R1, SCRATCH0));
        return Ok(());
    }

    let moved_hi =
        matches!(src_hi_reg, Some(Arm32Reg::R0)) && !matches!(src_lo_reg, Some(Arm32Reg::R0));
    if moved_hi {
        fc.text.emit_u32(enc::mov_reg(Arm32Reg::R1, Arm32Reg::R0));
    }

    if !matches!(src_lo_reg, Some(Arm32Reg::R0)) {
        emit_move_gp_value(fc, Arm32Reg::R0, src_lo)?;
    }
    if !moved_hi && !matches!(src_hi_reg, Some(Arm32Reg::R1)) {
        emit_move_gp_value(fc, Arm32Reg::R1, src_hi)?;
    }
    Ok(())
}

pub(super) fn emit_pair_results_from_r0_r1(
    fc: &mut FunctionCompiler<'_>,
    dst_lo: crate::vm::machine::machine_ir::MachineReg,
    dst_hi: crate::vm::machine::machine_ir::MachineReg,
) -> Result<(), WasmError> {
    let dst_lo_hw = map_reg(dst_lo)?;
    let dst_hi_hw = map_reg(dst_hi)?;

    if dst_lo_hw == Arm32Reg::R1 && dst_hi_hw == Arm32Reg::R0 {
        fc.text.emit_u32(enc::mov_reg(SCRATCH0, Arm32Reg::R0));
        fc.text.emit_u32(enc::mov_reg(Arm32Reg::R0, Arm32Reg::R1));
        fc.text.emit_u32(enc::mov_reg(Arm32Reg::R1, SCRATCH0));
        return Ok(());
    }

    let moved_lo = dst_hi_hw == Arm32Reg::R0 && dst_lo_hw != Arm32Reg::R0;
    if moved_lo {
        fc.text.emit_u32(enc::mov_reg(dst_lo_hw, Arm32Reg::R0));
    }

    let moved_hi = dst_lo_hw == Arm32Reg::R1 && dst_hi_hw != Arm32Reg::R1;
    if moved_hi {
        fc.text.emit_u32(enc::mov_reg(dst_hi_hw, Arm32Reg::R1));
    }

    if !moved_lo && dst_lo_hw != Arm32Reg::R0 {
        fc.text.emit_u32(enc::mov_reg(dst_lo_hw, Arm32Reg::R0));
    }
    if !moved_hi && dst_hi_hw != Arm32Reg::R1 {
        fc.text.emit_u32(enc::mov_reg(dst_hi_hw, Arm32Reg::R1));
    }
    Ok(())
}

// ─── Caller-saved register spill / restore ──────────────────────────────────

pub(super) fn spill_caller_saved_gp_regs(fc: &mut FunctionCompiler<'_>) {
    // R0-R3 participate in the allocatable GP bank on ARMv7A. Any backend
    // sequence that repurposes them as helper arguments or staging temporaries
    // must preserve live values first, then restore every register that is not
    // an explicit destination of that sequence.
    fc.text
        .emit_u32(enc::sub_imm(Arm32Reg::SP, Arm32Reg::SP, 16, 0));
    fc.text
        .emit_u32(enc::str_imm(Arm32Reg::R0, Arm32Reg::SP, 0));
    fc.text
        .emit_u32(enc::str_imm(Arm32Reg::R1, Arm32Reg::SP, 4));
    fc.text
        .emit_u32(enc::str_imm(Arm32Reg::R2, Arm32Reg::SP, 8));
    fc.text
        .emit_u32(enc::str_imm(Arm32Reg::R3, Arm32Reg::SP, 12));
}

pub(super) fn restore_caller_saved_gp_regs(fc: &mut FunctionCompiler<'_>, preserved: &[Arm32Reg]) {
    if !preserved.contains(&Arm32Reg::R0) {
        fc.text
            .emit_u32(enc::ldr_imm(Arm32Reg::R0, Arm32Reg::SP, 0));
    }
    if !preserved.contains(&Arm32Reg::R1) {
        fc.text
            .emit_u32(enc::ldr_imm(Arm32Reg::R1, Arm32Reg::SP, 4));
    }
    if !preserved.contains(&Arm32Reg::R2) {
        fc.text
            .emit_u32(enc::ldr_imm(Arm32Reg::R2, Arm32Reg::SP, 8));
    }
    if !preserved.contains(&Arm32Reg::R3) {
        fc.text
            .emit_u32(enc::ldr_imm(Arm32Reg::R3, Arm32Reg::SP, 12));
    }
    fc.text
        .emit_u32(enc::add_imm(Arm32Reg::SP, Arm32Reg::SP, 16, 0));
}

// ─── Stack-staged value moves ───────────────────────────────────────────────

pub(super) fn emit_values_to_regs_via_stack(
    fc: &mut FunctionCompiler<'_>,
    regs: &[Arm32Reg],
    values: &[&MachineValue],
) -> Result<(), WasmError> {
    if regs.len() != values.len() {
        return Err(WasmError::internal(
            "armv7a stack-staged value move requires matching regs and values".into(),
        ));
    }
    let scratch_mask = 1 << SCRATCH0.idx();
    for value in values {
        emit_move_gp_value(fc, SCRATCH0, value)?;
        fc.text.emit_u32(enc::push(scratch_mask));
    }
    for reg in regs.iter().rev() {
        fc.text.emit_u32(enc::pop(1 << reg.idx()));
    }
    Ok(())
}

pub(super) fn emit_quad_args_to_r0_r3(
    fc: &mut FunctionCompiler<'_>,
    value0: &MachineValue,
    value1: &MachineValue,
    value2: &MachineValue,
    value3: &MachineValue,
) -> Result<(), WasmError> {
    emit_values_to_regs_via_stack(
        fc,
        &[Arm32Reg::R0, Arm32Reg::R1, Arm32Reg::R2, Arm32Reg::R3],
        &[value0, value1, value2, value3],
    )
}

// ─── Compare / bool helpers ─────────────────────────────────────────────────

pub(super) fn emit_cmp_gp_values(
    fc: &mut FunctionCompiler<'_>,
    lhs: &MachineValue,
    rhs: &MachineValue,
) -> Result<(), WasmError> {
    emit_values_to_regs_via_stack(fc, &[Arm32Reg::R0, Arm32Reg::R1], &[lhs, rhs])?;
    fc.text.emit_u32(enc::cmp_reg(Arm32Reg::R0, Arm32Reg::R1));
    Ok(())
}

pub(super) fn emit_set_bool_immediate(fc: &mut FunctionCompiler<'_>, dst: Arm32Reg, value: bool) {
    fc.emit_load_u32(dst, u32::from(value));
}

// ─── Stack temp alloc/free ──────────────────────────────────────────────────

pub(super) fn emit_stack_temp_alloc(fc: &mut FunctionCompiler<'_>, bytes: u32) {
    fc.text
        .emit_u32(enc::sub_imm(Arm32Reg::SP, Arm32Reg::SP, bytes, 0));
}

pub(super) fn emit_stack_temp_free(fc: &mut FunctionCompiler<'_>, bytes: u32) {
    fc.text
        .emit_u32(enc::add_imm(Arm32Reg::SP, Arm32Reg::SP, bytes, 0));
}

pub(super) fn emit_trunc_result_buffer_alloc(fc: &mut FunctionCompiler<'_>) {
    emit_stack_temp_alloc(fc, 16);
}

pub(super) fn emit_trunc_result_buffer_free(fc: &mut FunctionCompiler<'_>) {
    emit_stack_temp_free(fc, 16);
}

// ─── Memory word helpers ────────────────────────────────────────────────────

pub(super) fn emit_load_word_from_addr(
    fc: &mut FunctionCompiler<'_>,
    dst: Arm32Reg,
    base: Arm32Reg,
    offset: i32,
) {
    if (-4095..=4095).contains(&offset) {
        fc.text.emit_u32(enc::ldr_imm(dst, base, offset));
    } else {
        fc.emit_load_u32(SCRATCH0, offset as u32);
        fc.text.emit_u32(enc::add_reg(SCRATCH0, base, SCRATCH0));
        fc.text.emit_u32(enc::ldr_imm(dst, SCRATCH0, 0));
    }
}

pub(super) fn emit_store_word_to_addr(
    fc: &mut FunctionCompiler<'_>,
    src: Arm32Reg,
    base: Arm32Reg,
    offset: i32,
) {
    if (-4095..=4095).contains(&offset) {
        fc.text.emit_u32(enc::str_imm(src, base, offset));
    } else {
        let addr_tmp = if src == SCRATCH0 { SCRATCH1 } else { SCRATCH0 };
        fc.emit_load_u32(addr_tmp, offset as u32);
        fc.text.emit_u32(enc::add_reg(addr_tmp, base, addr_tmp));
        fc.text.emit_u32(enc::str_imm(src, addr_tmp, 0));
    }
}

// ─── Trap helpers ───────────────────────────────────────────────────────────

pub(super) fn emit_raise_trap_and_return(
    fc: &mut FunctionCompiler<'_>,
    trap_kind: u32,
) -> Result<(), WasmError> {
    fc.text
        .emit_u32(enc::mov_reg(Arm32Reg::R0, map_fixed_reg(MACHINE_CTX_REG)));
    fc.emit_load_u32(Arm32Reg::R1, trap_kind);
    fc.emit_load_u32(Arm32Reg::R2, fc.current_trap_site());
    emit_host_call(fc, armv7a_raise_trap as usize);
    fc.emit_load_u32(Arm32Reg::R0, 1);
    emit_shared_epilogue(&mut fc.text);
    Ok(())
}

pub(super) fn trap_kind_to_u32(kind: MachineTrapKind) -> u32 {
    match kind {
        MachineTrapKind::Unreachable => 0,
        MachineTrapKind::MemoryOutOfBounds => 1,
        MachineTrapKind::TableOutOfBounds => 2,
        MachineTrapKind::InvalidFunctionReference => 3,
        MachineTrapKind::IndirectCallTypeMismatch => 4,
        MachineTrapKind::IntegerDivideByZero => 5,
        MachineTrapKind::IntegerOverflow => 6,
        MachineTrapKind::StackOverflow => 7,
        MachineTrapKind::HelperFailure => 8,
    }
}

// ─── Convert op code ────────────────────────────────────────────────────────

pub(super) fn convert_op_code(op: MachineConvertOp) -> u32 {
    match op {
        MachineConvertOp::I32TruncF32S => 0,
        MachineConvertOp::I32TruncF32U => 1,
        MachineConvertOp::I32TruncF64S => 2,
        MachineConvertOp::I32TruncF64U => 3,
        MachineConvertOp::I64TruncF32S => 4,
        MachineConvertOp::I64TruncF32U => 5,
        MachineConvertOp::I64TruncF64S => 6,
        MachineConvertOp::I64TruncF64U => 7,
        MachineConvertOp::I32TruncSatF32S => 8,
        MachineConvertOp::I32TruncSatF32U => 9,
        MachineConvertOp::I32TruncSatF64S => 10,
        MachineConvertOp::I32TruncSatF64U => 11,
        MachineConvertOp::I64TruncSatF32S => 12,
        MachineConvertOp::I64TruncSatF32U => 13,
        MachineConvertOp::I64TruncSatF64S => 14,
        MachineConvertOp::I64TruncSatF64U => 15,
        _ => u32::MAX,
    }
}

/// RBIT Rd, Rm (reverse bits, ARMv6T2+)
pub(super) fn rbit(dst: Arm32Reg, src: Arm32Reg) -> u32 {
    // RBIT: cond 0110 1111 1111 Rd 1111 0011 Rm
    enc::cond_bits(Cond::Al)
        | (0b01101111 << 20)
        | (0b1111 << 16)
        | ((dst.idx()) << 12)
        | (0b11110011 << 4)
        | src.idx()
}
