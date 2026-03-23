//! Instruction emission for the ARMv7-A backend.
//!
//! Each `compile_*` function handles one or more `MachineInstKind` variants,
//! emitting ARM32 machine code via the `FunctionCompiler`.

use crate::{
    error::WasmError,
    vm::machine::machine_ir::{
        MachineAddr, MachineBranchCond, MachineCompareKind, MachineConvertOp,
        MachineFloatBinaryOp, MachineFloatUnaryOp, MachineFloatWidth, MachineHelperCall,
        MachineInst, MachineInstKind, MachineIntBinaryOp, MachineIntUnaryOp, MachineIntWidth,
        MachineLoadExtension, MachineMemWidth, MachineReg, MachineSign, MachineStorageType,
        MachineTrapKind, MachineValue,
    },
    vm::runtime::helpers::resolve_helper_entry,
};

use super::{
    abi::{
        emit_shared_epilogue, map_fixed_reg, map_reg, FP_SCRATCH0, FP_SCRATCH1, FP_SCRATCH2,
        SCRATCH0, SCRATCH1,
    },
    armv7a_f32_ceil, armv7a_f32_floor, armv7a_f32_nearest_bits, armv7a_f32_trunc,
    armv7a_f64_ceil, armv7a_f64_floor, armv7a_f64_nearest_bits, armv7a_f64_trunc,
    armv7a_i64_clz, armv7a_i64_ctz, armv7a_i64_div_s, armv7a_i64_div_u, armv7a_i64_mul,
    armv7a_i64_popcnt, armv7a_i64_rem_s, armv7a_i64_rem_u, armv7a_i64_rotl, armv7a_i64_rotr,
    armv7a_i64_shl, armv7a_i64_shr_s, armv7a_i64_shr_u, armv7a_i64s_to_f32, armv7a_i64s_to_f64,
    armv7a_i64u_to_f32, armv7a_i64u_to_f64, armv7a_raise_trap, armv7a_saturating_trunc,
    armv7a_sdiv, armv7a_smul_wide, armv7a_trapping_trunc, armv7a_udiv, armv7a_umul_wide,
    enc::{self, Cond},
    reg::Arm32Reg,
};

use crate::vm::machine::machine_ir::{MACHINE_CTX_REG, MACHINE_FP_REG};
use super::compile::{BranchFixupKind, FunctionCompiler, LabelKind};

use super::compile_control::compile_branch_condition;

use super::compile_helpers::{
    convert_op_code, emit_cmp_gp_values, emit_host_call, emit_load_word_from_addr,
    emit_move_gp_value, emit_pair_args_to_r0_r1, emit_pair_results_from_r0_r1,
    emit_quad_args_to_r0_r3, emit_raise_trap_and_return, emit_set_bool_immediate,
    emit_stack_temp_alloc, emit_stack_temp_free, emit_store_word_to_addr,
    emit_trunc_result_buffer_alloc, emit_trunc_result_buffer_free,
    emit_values_to_regs_via_stack, materialize_float_value_dreg, materialize_gp_into,
    materialize_gp_value, rbit, restore_caller_saved_gp_regs, spill_caller_saved_gp_regs,
    trap_kind_to_u32,
};

// ─── Top-level instruction dispatch ─────────────────────────────────────────

pub(super) fn compile_inst(
    fc: &mut FunctionCompiler<'_>,
    inst: &MachineInst,
) -> Result<(), WasmError> {
    match &inst.kind {
        MachineInstKind::Move { dst, src, .. } => {
            let dst_is_fp = fc.is_fp_machine_reg(*dst);
            let src_is_fp = match src {
                MachineValue::Reg(r) => fc.is_fp_machine_reg(*r),
                MachineValue::Imm64(_) => false,
            };

            if dst_is_fp && src_is_fp {
                // FP → FP move (D-register)
                let dd = fc.map_fp_dreg(*dst)?;
                let dm = fc.map_fp_dreg(match src {
                    MachineValue::Reg(r) => *r,
                    _ => unreachable!(),
                })?;
                if dd != dm {
                    fc.text.emit_u32(enc::vmov_d(dd, dm));
                }
            } else if dst_is_fp {
                // GP/Imm → FP: load to GP scratch then VMOV to D-reg
                let dd = fc.map_fp_dreg(*dst)?;
                match src {
                    MachineValue::Reg(r) => {
                        let src_hw = map_reg(*r)?;
                        // Move GP value to low half of D-register (as 64-bit with zero-extended high)
                        fc.emit_load_u32(Arm32Reg::R1, 0);
                        fc.text.emit_u32(enc::vmov_d_rr(dd, src_hw, Arm32Reg::R1));
                    }
                    MachineValue::Imm64(imm) => {
                        let lo = *imm as u32;
                        let hi = (*imm >> 32) as u32;
                        fc.emit_load_u32(Arm32Reg::R0, lo);
                        fc.emit_load_u32(Arm32Reg::R1, hi);
                        fc.text
                            .emit_u32(enc::vmov_d_rr(dd, Arm32Reg::R0, Arm32Reg::R1));
                    }
                }
            } else if src_is_fp {
                // FP → GP: VMOV from D-reg low word to GP
                let dst_hw = map_reg(*dst)?;
                let dm = fc.map_fp_dreg(match src {
                    MachineValue::Reg(r) => *r,
                    _ => unreachable!(),
                })?;
                fc.text.emit_u32(enc::vmov_rr_d(dst_hw, Arm32Reg::R1, dm));
                // dst_hw now has the low 32 bits
            } else {
                // GP → GP or Imm → GP
                let dst_hw = map_reg(*dst)?;
                match src {
                    MachineValue::Reg(r) => {
                        let src_hw = map_reg(*r)?;
                        if dst_hw != src_hw {
                            fc.text.emit_u32(enc::mov_reg(dst_hw, src_hw));
                        }
                    }
                    MachineValue::Imm64(imm) => {
                        fc.emit_load_u32(dst_hw, *imm as u32);
                    }
                }
            }
        }

        MachineInstKind::FloatConst { width, dst, bits } => {
            // Load FP constant: put bits in GP scratch, then VMOV to FP reg
            let dd = fc.map_fp_dreg(*dst)?;
            match width {
                MachineFloatWidth::F32 => {
                    let lo = *bits as u32;
                    fc.emit_load_u32(SCRATCH0, lo);
                    fc.text.emit_u32(enc::vmov_s_r(dd * 2, SCRATCH0));
                }
                MachineFloatWidth::F64 => {
                    let lo = *bits as u32;
                    let hi = (*bits >> 32) as u32;
                    fc.emit_load_u32(Arm32Reg::R0, lo);
                    fc.emit_load_u32(Arm32Reg::R1, hi);
                    fc.text
                        .emit_u32(enc::vmov_d_rr(dd, Arm32Reg::R0, Arm32Reg::R1));
                }
            }
        }

        MachineInstKind::Lea { dst, addr } => {
            let dst_hw = map_reg(*dst)?;
            let base_hw = map_reg(addr.base)?;
            if addr.offset == 0 {
                if dst_hw != base_hw {
                    fc.text.emit_u32(enc::mov_reg(dst_hw, base_hw));
                }
            } else if let Some((imm8, rot)) = enc::encode_arm_imm(addr.offset as u32) {
                fc.text.emit_u32(enc::add_imm(dst_hw, base_hw, imm8, rot));
            } else {
                fc.emit_load_u32(SCRATCH0, addr.offset as u32);
                fc.text.emit_u32(enc::add_reg(dst_hw, base_hw, SCRATCH0));
            }
        }

        MachineInstKind::Load {
            ty: _,
            dst,
            addr,
            width,
            extension,
        } => {
            compile_load(fc, *dst, addr, *width, *extension)?;
        }

        MachineInstKind::Store {
            ty,
            addr,
            width,
            src,
        } => {
            compile_store(fc, *ty, addr, *width, src)?;
        }

        MachineInstKind::IntBinary {
            width,
            op,
            dst,
            lhs,
            rhs,
        } => {
            compile_int_binary(fc, *width, *op, *dst, lhs, rhs)?;
        }

        MachineInstKind::IntMulWide {
            sign,
            dst_lo,
            dst_hi,
            lhs,
            rhs,
        } => {
            compile_int_mul_wide(fc, *sign, *dst_lo, *dst_hi, lhs, rhs)?;
        }

        MachineInstKind::Int64PairBinary {
            op,
            dst_lo,
            dst_hi,
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
        } => {
            compile_int64_pair_binary(fc, *op, *dst_lo, *dst_hi, lhs_lo, lhs_hi, rhs_lo, rhs_hi)?;
        }

        MachineInstKind::Int64PairUnary {
            op,
            dst_lo,
            dst_hi,
            src_lo,
            src_hi,
        } => {
            compile_int64_pair_unary(fc, *op, *dst_lo, *dst_hi, src_lo, src_hi)?;
        }

        MachineInstKind::Int64PairDivRem {
            sign,
            rem,
            dst_lo,
            dst_hi,
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
        } => {
            compile_int64_pair_div_rem(
                fc, *sign, *rem, *dst_lo, *dst_hi, lhs_lo, lhs_hi, rhs_lo, rhs_hi,
            )?;
        }

        MachineInstKind::Int64PairShift {
            op,
            dst_lo,
            dst_hi,
            lhs_lo,
            lhs_hi,
            rhs,
        } => {
            compile_int64_pair_shift(fc, *op, *dst_lo, *dst_hi, lhs_lo, lhs_hi, rhs)?;
        }

        MachineInstKind::IntUnary {
            width,
            op,
            dst,
            src,
        } => {
            compile_int_unary(fc, *width, *op, *dst, src)?;
        }

        MachineInstKind::IntCompare {
            width,
            kind,
            sign,
            dst,
            lhs,
            rhs,
        } => {
            compile_int_compare(fc, *width, *kind, *sign, *dst, lhs, rhs)?;
        }

        MachineInstKind::Int64PairCompare {
            kind,
            sign,
            dst,
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
        } => {
            compile_int64_pair_compare(fc, *kind, *sign, *dst, lhs_lo, lhs_hi, rhs_lo, rhs_hi)?;
        }

        MachineInstKind::FloatBinary {
            width,
            op,
            dst,
            lhs,
            rhs,
        } => {
            compile_float_binary(fc, *width, *op, *dst, lhs, rhs)?;
        }

        MachineInstKind::FloatUnary {
            width,
            op,
            dst,
            src,
        } => {
            compile_float_unary(fc, *width, *op, *dst, src)?;
        }

        MachineInstKind::FloatCompare {
            width,
            kind,
            dst,
            lhs,
            rhs,
        } => {
            compile_float_compare(fc, *width, *kind, *dst, lhs, rhs)?;
        }

        MachineInstKind::Convert { op, dst, src } => {
            compile_convert(fc, *op, *dst, src)?;
        }

        MachineInstKind::ConvertI64PairToFloat {
            width,
            sign,
            dst,
            src_lo,
            src_hi,
        } => {
            compile_convert_i64_pair_to_float(fc, *width, *sign, *dst, src_lo, src_hi)?;
        }

        MachineInstKind::ConvertFloatToI64Pair {
            op,
            dst_lo,
            dst_hi,
            src,
        } => {
            compile_convert_float_to_i64_pair(fc, *op, *dst_lo, *dst_hi, src)?;
        }

        MachineInstKind::ReinterpretF64ToI64Pair {
            dst_lo,
            dst_hi,
            src,
        } => {
            compile_reinterpret_f64_to_i64_pair(fc, *dst_lo, *dst_hi, src)?;
        }

        MachineInstKind::ReinterpretI64PairToF64 {
            dst,
            src_lo,
            src_hi,
        } => {
            compile_reinterpret_i64_pair_to_f64(fc, *dst, src_lo, src_hi)?;
        }

        MachineInstKind::Select {
            dst,
            on_true,
            on_false,
            cond,
            ..
        } => {
            compile_select(fc, *dst, cond, on_true, on_false)?;
        }

        MachineInstKind::TrapIf { kind, cond } => {
            compile_trap_if(fc, *kind, cond)?;
        }

        MachineInstKind::CallHelper(call) => {
            compile_call_helper(fc, call)?;
        }
        MachineInstKind::IndexedLoad { .. } | MachineInstKind::IndexedStore { .. } => {
            todo!("armv7a: emit IndexedLoad / IndexedStore")
        }
    }
    Ok(())
}

// ─── Load/Store helpers ─────────────────────────────────────────────────────

fn compile_load(
    fc: &mut FunctionCompiler<'_>,
    dst: MachineReg,
    addr: &MachineAddr,
    width: MachineMemWidth,
    extension: MachineLoadExtension,
) -> Result<(), WasmError> {
    let base_hw = map_reg(addr.base)?;
    let offset = addr.offset;

    // ARMv7 VFP loads/stores require alignment that Wasm memory does not
    // guarantee. MachineAddr does not currently preserve enough provenance to
    // distinguish "provably aligned frame/context slot" from "possibly
    // unaligned Wasm memory address", so use a GP-word bridge here for
    // correctness.
    if fc.is_fp_machine_reg(dst) {
        let dd = fc.map_fp_dreg(dst)?;
        match width {
            MachineMemWidth::U64 => {
                emit_load_word_from_addr(fc, SCRATCH0, base_hw, offset);
                emit_load_word_from_addr(fc, SCRATCH1, base_hw, offset + 4);
                fc.text.emit_u32(enc::vmov_d_rr(dd, SCRATCH0, SCRATCH1));
            }
            MachineMemWidth::U32 => {
                emit_load_word_from_addr(fc, SCRATCH0, base_hw, offset);
                fc.text.emit_u32(enc::vmov_s_r(dd * 2, SCRATCH0));
            }
            _ => {
                return Err(WasmError::invalid(alloc::format!(
                    "armv7a: unsupported FP load width {:?}",
                    width
                )));
            }
        }
        return Ok(());
    }

    // GP destination: use LDR/LDRB/LDRH etc.
    let dst_hw = map_reg(dst)?;
    match width {
        MachineMemWidth::U8 => match extension {
            MachineLoadExtension::SignExtend => {
                fc.text.emit_u32(enc::ldrsb_imm(dst_hw, base_hw, offset));
            }
            _ => {
                fc.text.emit_u32(enc::ldrb_imm(dst_hw, base_hw, offset));
            }
        },
        MachineMemWidth::U16 => match extension {
            MachineLoadExtension::SignExtend => {
                fc.text.emit_u32(enc::ldrsh_imm(dst_hw, base_hw, offset));
            }
            _ => {
                fc.text.emit_u32(enc::ldrh_imm(dst_hw, base_hw, offset));
            }
        },
        MachineMemWidth::U32 => {
            fc.text.emit_u32(enc::ldr_imm(dst_hw, base_hw, offset));
        }
        MachineMemWidth::U64 => {
            // 64-bit load to GP: load low 32 bits only
            fc.text.emit_u32(enc::ldr_imm(dst_hw, base_hw, offset));
        }
    }
    Ok(())
}

fn compile_store(
    fc: &mut FunctionCompiler<'_>,
    ty: MachineStorageType,
    addr: &MachineAddr,
    width: MachineMemWidth,
    src: &MachineValue,
) -> Result<(), WasmError> {
    let base_hw = map_reg(addr.base)?;
    let offset = addr.offset;

    if matches!(ty, MachineStorageType::Fp32 | MachineStorageType::Fp64) {
        match src {
            MachineValue::Reg(r) if fc.is_fp_machine_reg(*r) => {
                let dd = fc.map_fp_dreg(*r)?;
                match width {
                    MachineMemWidth::U64 => {
                        fc.text.emit_u32(enc::vmov_rr_d(SCRATCH0, SCRATCH1, dd));
                        emit_store_word_to_addr(fc, SCRATCH0, base_hw, offset);
                        emit_store_word_to_addr(fc, SCRATCH1, base_hw, offset + 4);
                    }
                    MachineMemWidth::U32 => {
                        fc.text.emit_u32(enc::vmov_r_s(SCRATCH0, dd * 2));
                        emit_store_word_to_addr(fc, SCRATCH0, base_hw, offset);
                    }
                    _ => {
                        return Err(WasmError::invalid(alloc::format!(
                            "armv7a: unsupported FP store width {:?}",
                            width
                        )));
                    }
                }
            }
            MachineValue::Imm64(bits) => match width {
                MachineMemWidth::U64 => {
                    fc.emit_load_u32(SCRATCH0, *bits as u32);
                    fc.emit_load_u32(SCRATCH1, (*bits >> 32) as u32);
                    emit_store_word_to_addr(fc, SCRATCH0, base_hw, offset);
                    emit_store_word_to_addr(fc, SCRATCH1, base_hw, offset + 4);
                }
                MachineMemWidth::U32 => {
                    fc.emit_load_u32(SCRATCH0, *bits as u32);
                    emit_store_word_to_addr(fc, SCRATCH0, base_hw, offset);
                }
                _ => {
                    return Err(WasmError::invalid(alloc::format!(
                        "armv7a: unsupported FP store width {:?}",
                        width
                    )));
                }
            },
            MachineValue::Reg(r) => {
                return Err(WasmError::invalid(alloc::format!(
                    "armv7a: FP store expects an FP register source, got GP machine reg {}",
                    r.0
                )));
            }
        }
        return Ok(());
    }

    // GP source
    let src_hw = match src {
        MachineValue::Reg(r) => map_reg(*r)?,
        MachineValue::Imm64(imm) => {
            fc.emit_load_u32(SCRATCH0, *imm as u32);
            SCRATCH0
        }
    };

    match width {
        MachineMemWidth::U8 => {
            fc.text.emit_u32(enc::strb_imm(src_hw, base_hw, offset));
        }
        MachineMemWidth::U16 => {
            fc.text.emit_u32(enc::strh_imm(src_hw, base_hw, offset));
        }
        MachineMemWidth::U32 => {
            fc.text.emit_u32(enc::str_imm(src_hw, base_hw, offset));
        }
        MachineMemWidth::U64 => {
            // Store low word, then zero high word
            fc.text.emit_u32(enc::str_imm(src_hw, base_hw, offset));
            fc.emit_load_u32(SCRATCH0, 0);
            fc.text
                .emit_u32(enc::str_imm(SCRATCH0, base_hw, offset + 4));
        }
    }
    Ok(())
}

// ─── Integer ALU ────────────────────────────────────────────────────────────

fn compile_int_binary(
    fc: &mut FunctionCompiler<'_>,
    width: MachineIntWidth,
    op: MachineIntBinaryOp,
    dst: MachineReg,
    lhs: &MachineValue,
    rhs: &MachineValue,
) -> Result<(), WasmError> {
    let dst_hw = map_reg(dst)?;

    let lhs_hw = match lhs {
        MachineValue::Reg(r) => map_reg(*r)?,
        MachineValue::Imm64(v) => {
            fc.emit_load_u32(dst_hw, *v as u32);
            dst_hw
        }
    };

    match op {
        MachineIntBinaryOp::Add => match rhs {
            MachineValue::Imm64(imm) => {
                if let Some((imm8, rot)) = enc::encode_arm_imm(*imm as u32) {
                    fc.text.emit_u32(enc::add_imm(dst_hw, lhs_hw, imm8, rot));
                } else {
                    fc.emit_load_u32(SCRATCH0, *imm as u32);
                    fc.text.emit_u32(enc::add_reg(dst_hw, lhs_hw, SCRATCH0));
                }
            }
            MachineValue::Reg(r) => {
                fc.text.emit_u32(enc::add_reg(dst_hw, lhs_hw, map_reg(*r)?));
            }
        },
        MachineIntBinaryOp::Sub => match rhs {
            MachineValue::Imm64(imm) => {
                if let Some((imm8, rot)) = enc::encode_arm_imm(*imm as u32) {
                    fc.text.emit_u32(enc::sub_imm(dst_hw, lhs_hw, imm8, rot));
                } else {
                    fc.emit_load_u32(SCRATCH0, *imm as u32);
                    fc.text.emit_u32(enc::sub_reg(dst_hw, lhs_hw, SCRATCH0));
                }
            }
            MachineValue::Reg(r) => {
                fc.text.emit_u32(enc::sub_reg(dst_hw, lhs_hw, map_reg(*r)?));
            }
        },
        MachineIntBinaryOp::Mul => {
            let rhs_hw = match rhs {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            fc.text.emit_u32(enc::mul(dst_hw, lhs_hw, rhs_hw));
        }
        MachineIntBinaryOp::And => {
            let rhs_hw = match rhs {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    if let Some((imm8, rot)) = enc::encode_arm_imm(*v as u32) {
                        fc.text.emit_u32(enc::and_imm(dst_hw, lhs_hw, imm8, rot));
                        return Ok(());
                    }
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            fc.text.emit_u32(enc::and_reg(dst_hw, lhs_hw, rhs_hw));
        }
        MachineIntBinaryOp::Or => {
            let rhs_hw = match rhs {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    if let Some((imm8, rot)) = enc::encode_arm_imm(*v as u32) {
                        fc.text.emit_u32(enc::orr_imm(dst_hw, lhs_hw, imm8, rot));
                        return Ok(());
                    }
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            fc.text.emit_u32(enc::orr_reg(dst_hw, lhs_hw, rhs_hw));
        }
        MachineIntBinaryOp::Xor => {
            let rhs_hw = match rhs {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    if let Some((imm8, rot)) = enc::encode_arm_imm(*v as u32) {
                        fc.text.emit_u32(enc::eor_imm(dst_hw, lhs_hw, imm8, rot));
                        return Ok(());
                    }
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            fc.text.emit_u32(enc::eor_reg(dst_hw, lhs_hw, rhs_hw));
        }
        MachineIntBinaryOp::Shl => {
            let rhs_hw = match rhs {
                MachineValue::Imm64(v) => {
                    let shift = (*v as u32) & 31;
                    fc.text.emit_u32(enc::lsl_imm(dst_hw, lhs_hw, shift));
                    return Ok(());
                }
                MachineValue::Reg(r) => map_reg(*r)?,
            };
            // Mask shift amount to 5 bits (wasm i32 semantics)
            fc.text.emit_u32(enc::and_imm(SCRATCH0, rhs_hw, 31, 0));
            fc.text.emit_u32(enc::lsl_reg(dst_hw, lhs_hw, SCRATCH0));
        }
        MachineIntBinaryOp::ShrU => {
            let rhs_hw = match rhs {
                MachineValue::Imm64(v) => {
                    let shift = (*v as u32) & 31;
                    fc.text.emit_u32(enc::lsr_imm(dst_hw, lhs_hw, shift));
                    return Ok(());
                }
                MachineValue::Reg(r) => map_reg(*r)?,
            };
            fc.text.emit_u32(enc::and_imm(SCRATCH0, rhs_hw, 31, 0));
            fc.text.emit_u32(enc::lsr_reg(dst_hw, lhs_hw, SCRATCH0));
        }
        MachineIntBinaryOp::ShrS => {
            let rhs_hw = match rhs {
                MachineValue::Imm64(v) => {
                    let shift = (*v as u32) & 31;
                    fc.text.emit_u32(enc::asr_imm(dst_hw, lhs_hw, shift));
                    return Ok(());
                }
                MachineValue::Reg(r) => map_reg(*r)?,
            };
            fc.text.emit_u32(enc::and_imm(SCRATCH0, rhs_hw, 31, 0));
            fc.text.emit_u32(enc::asr_reg(dst_hw, lhs_hw, SCRATCH0));
        }
        MachineIntBinaryOp::Rotl => {
            // rotl(x, k) = rotr(x, 32-k)
            let rhs_hw = match rhs {
                MachineValue::Imm64(v) => {
                    let shift = (32 - ((*v as u32) & 31)) & 31;
                    fc.text.emit_u32(enc::ror_imm(dst_hw, lhs_hw, shift));
                    return Ok(());
                }
                MachineValue::Reg(r) => map_reg(*r)?,
            };
            fc.text.emit_u32(enc::and_imm(SCRATCH0, rhs_hw, 31, 0));
            fc.text.emit_u32(enc::rsb_imm(SCRATCH0, SCRATCH0, 32, 0));
            fc.text.emit_u32(enc::ror_reg(dst_hw, lhs_hw, SCRATCH0));
        }
        MachineIntBinaryOp::Rotr => {
            let rhs_hw = match rhs {
                MachineValue::Imm64(v) => {
                    let shift = (*v as u32) & 31;
                    fc.text.emit_u32(enc::ror_imm(dst_hw, lhs_hw, shift));
                    return Ok(());
                }
                MachineValue::Reg(r) => map_reg(*r)?,
            };
            fc.text.emit_u32(enc::and_imm(SCRATCH0, rhs_hw, 31, 0));
            fc.text.emit_u32(enc::ror_reg(dst_hw, lhs_hw, SCRATCH0));
        }
        MachineIntBinaryOp::DivU => {
            spill_caller_saved_gp_regs(fc);
            emit_values_to_regs_via_stack(fc, &[Arm32Reg::R0, Arm32Reg::R1], &[lhs, rhs])?;
            fc.text.emit_u32(enc::cmp_imm(Arm32Reg::R1, 0, 0));
            let ok = fc.alloc_label(LabelKind::Block);
            let trap_div_zero = fc.alloc_label(LabelKind::Block);
            let done = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::BCond(Cond::Ne), ok);
            fc.emit_branch(BranchFixupKind::B, trap_div_zero);
            // Call armv7a_udiv(num, den) -> quotient in R0
            fc.bind_label(ok);
            emit_host_call(fc, armv7a_udiv as usize);
            if dst_hw != Arm32Reg::R0 {
                fc.text.emit_u32(enc::mov_reg(dst_hw, Arm32Reg::R0));
            }
            restore_caller_saved_gp_regs(fc, &[dst_hw]);
            fc.emit_branch(BranchFixupKind::B, done);
            fc.bind_label(trap_div_zero);
            emit_stack_temp_free(fc, 16);
            emit_raise_trap_and_return(fc, 5)?;
            fc.bind_label(done);
        }
        MachineIntBinaryOp::DivS => {
            spill_caller_saved_gp_regs(fc);
            emit_values_to_regs_via_stack(fc, &[Arm32Reg::R0, Arm32Reg::R1], &[lhs, rhs])?;
            // Trap on divide by zero
            fc.text.emit_u32(enc::cmp_imm(Arm32Reg::R1, 0, 0));
            let not_zero = fc.alloc_label(LabelKind::Block);
            let trap_div_zero = fc.alloc_label(LabelKind::Block);
            let trap_overflow = fc.alloc_label(LabelKind::Block);
            let after_traps = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::BCond(Cond::Ne), not_zero);
            fc.emit_branch(BranchFixupKind::B, trap_div_zero);
            fc.bind_label(not_zero);
            // Trap on INT_MIN / -1 (integer overflow)
            fc.emit_load_u32(SCRATCH0, 0x80000000u32);
            fc.text.emit_u32(enc::cmp_reg(Arm32Reg::R0, SCRATCH0));
            let not_overflow = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::BCond(Cond::Ne), not_overflow);
            fc.text.emit_u32(enc::cmn_imm(Arm32Reg::R1, 1, 0)); // CMN rhs, #1 == CMP rhs, #-1
            let not_overflow2 = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::BCond(Cond::Ne), not_overflow2);
            fc.emit_branch(BranchFixupKind::B, trap_overflow);
            fc.bind_label(not_overflow);
            fc.bind_label(not_overflow2);
            fc.emit_branch(BranchFixupKind::B, after_traps);
            fc.bind_label(trap_div_zero);
            emit_stack_temp_free(fc, 16);
            emit_raise_trap_and_return(fc, 5)?;
            fc.bind_label(trap_overflow);
            emit_stack_temp_free(fc, 16);
            emit_raise_trap_and_return(fc, 6)?;
            fc.bind_label(after_traps);
            // Call armv7a_sdiv(num, den)
            emit_host_call(fc, armv7a_sdiv as usize);
            if dst_hw != Arm32Reg::R0 {
                fc.text.emit_u32(enc::mov_reg(dst_hw, Arm32Reg::R0));
            }
            restore_caller_saved_gp_regs(fc, &[dst_hw]);
        }
        MachineIntBinaryOp::RemU => {
            spill_caller_saved_gp_regs(fc);
            emit_values_to_regs_via_stack(fc, &[Arm32Reg::R0, Arm32Reg::R1], &[lhs, rhs])?;
            // Trap on divide by zero
            fc.text.emit_u32(enc::cmp_imm(Arm32Reg::R1, 0, 0));
            let ok = fc.alloc_label(LabelKind::Block);
            let trap_div_zero = fc.alloc_label(LabelKind::Block);
            let done = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::BCond(Cond::Ne), ok);
            fc.emit_branch(BranchFixupKind::B, trap_div_zero);
            fc.bind_label(ok);
            emit_stack_temp_alloc(fc, 8);
            fc.text
                .emit_u32(enc::str_imm(Arm32Reg::R0, Arm32Reg::SP, 0));
            fc.text
                .emit_u32(enc::str_imm(Arm32Reg::R1, Arm32Reg::SP, 4));
            // rem = lhs - (lhs / rhs) * rhs
            emit_host_call(fc, armv7a_udiv as usize);
            // R0 = quotient. Restore lhs, rhs
            fc.text
                .emit_u32(enc::ldr_imm(Arm32Reg::R2, Arm32Reg::SP, 0));
            fc.text
                .emit_u32(enc::ldr_imm(Arm32Reg::R3, Arm32Reg::SP, 4));
            emit_stack_temp_free(fc, 8);
            fc.text
                .emit_u32(enc::mul(SCRATCH0, Arm32Reg::R0, Arm32Reg::R3));
            fc.text
                .emit_u32(enc::sub_reg(dst_hw, Arm32Reg::R2, SCRATCH0));
            restore_caller_saved_gp_regs(fc, &[dst_hw]);
            fc.emit_branch(BranchFixupKind::B, done);
            fc.bind_label(trap_div_zero);
            emit_stack_temp_free(fc, 16);
            emit_raise_trap_and_return(fc, 5)?;
            fc.bind_label(done);
        }
        MachineIntBinaryOp::RemS => {
            spill_caller_saved_gp_regs(fc);
            emit_values_to_regs_via_stack(fc, &[Arm32Reg::R0, Arm32Reg::R1], &[lhs, rhs])?;
            // Trap on divide by zero
            fc.text.emit_u32(enc::cmp_imm(Arm32Reg::R1, 0, 0));
            let ok = fc.alloc_label(LabelKind::Block);
            let trap_div_zero = fc.alloc_label(LabelKind::Block);
            let done = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::BCond(Cond::Ne), ok);
            fc.emit_branch(BranchFixupKind::B, trap_div_zero);
            fc.bind_label(ok);
            // INT_MIN % -1 == 0 in wasm (no trap, just returns 0)
            // rem = num - (num / den) * den — this naturally gives 0
            // Save lhs and rhs, call sdiv, compute remainder
            emit_stack_temp_alloc(fc, 8);
            fc.text
                .emit_u32(enc::str_imm(Arm32Reg::R0, Arm32Reg::SP, 0));
            fc.text
                .emit_u32(enc::str_imm(Arm32Reg::R1, Arm32Reg::SP, 4));
            emit_host_call(fc, armv7a_sdiv as usize);
            // R0 = quotient. Restore lhs, rhs
            fc.text
                .emit_u32(enc::ldr_imm(Arm32Reg::R2, Arm32Reg::SP, 0));
            fc.text
                .emit_u32(enc::ldr_imm(Arm32Reg::R3, Arm32Reg::SP, 4));
            emit_stack_temp_free(fc, 8);
            // rem = lhs - quotient * rhs: MLS dst, R0, R3, R2
            fc.text
                .emit_u32(enc::mul(SCRATCH0, Arm32Reg::R0, Arm32Reg::R3));
            fc.text
                .emit_u32(enc::sub_reg(dst_hw, Arm32Reg::R2, SCRATCH0));
            restore_caller_saved_gp_regs(fc, &[dst_hw]);
            fc.emit_branch(BranchFixupKind::B, done);
            fc.bind_label(trap_div_zero);
            emit_stack_temp_free(fc, 16);
            emit_raise_trap_and_return(fc, 5)?;
            fc.bind_label(done);
        }
    }

    // For I32 width, mask to 32 bits (already natural on ARM32)
    // For I64 width, we only handle low 32 bits currently
    Ok(())
}

fn compile_int_unary(
    fc: &mut FunctionCompiler<'_>,
    width: MachineIntWidth,
    op: MachineIntUnaryOp,
    dst: MachineReg,
    src: &MachineValue,
) -> Result<(), WasmError> {
    let dst_hw = map_reg(dst)?;
    let src_hw = match src {
        MachineValue::Reg(r) => map_reg(*r)?,
        MachineValue::Imm64(v) => {
            fc.emit_load_u32(dst_hw, *v as u32);
            dst_hw
        }
    };

    match op {
        MachineIntUnaryOp::Eqz => {
            // dst = (src == 0) ? 1 : 0
            fc.text.emit_u32(enc::cmp_imm(src_hw, 0, 0));
            // Seed the false arm after the compare so `dst == src` stays safe.
            fc.emit_load_u32(dst_hw, 0);
            // MOV{EQ} dst, #1
            let (imm8, rot) = enc::encode_arm_imm(1).unwrap();
            fc.text.emit_u32(enc::dp_imm_cond(
                Cond::Eq,
                0b1101,
                false,
                dst_hw,
                Arm32Reg::R0,
                imm8,
                rot,
            ));
        }
        MachineIntUnaryOp::Clz => {
            fc.text.emit_u32(enc::clz(dst_hw, src_hw));
        }
        MachineIntUnaryOp::Ctz => {
            // ctz(x) = 31 - clz(x & -x) when x != 0, else 32
            // RBIT + CLZ on ARMv7
            // Actually ARMv7 has RBIT: reverse bits, then CLZ
            fc.text.emit_u32(rbit(dst_hw, src_hw));
            fc.text.emit_u32(enc::clz(dst_hw, dst_hw));
        }
        MachineIntUnaryOp::Popcnt => {
            let mask_tmp = SCRATCH1;
            // Hamming weight using parallel bit counting
            // x = x - ((x >> 1) & 0x55555555)
            fc.text.emit_u32(enc::lsr_imm(SCRATCH0, src_hw, 1));
            fc.emit_load_u32(mask_tmp, 0x55555555);
            fc.text.emit_u32(enc::and_reg(SCRATCH0, SCRATCH0, mask_tmp));
            fc.text.emit_u32(enc::sub_reg(dst_hw, src_hw, SCRATCH0));
            // x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
            fc.emit_load_u32(mask_tmp, 0x33333333);
            fc.text.emit_u32(enc::lsr_imm(SCRATCH0, dst_hw, 2));
            fc.text.emit_u32(enc::and_reg(SCRATCH0, SCRATCH0, mask_tmp));
            fc.text.emit_u32(enc::and_reg(dst_hw, dst_hw, mask_tmp));
            fc.text.emit_u32(enc::add_reg(dst_hw, dst_hw, SCRATCH0));
            // x = (x + (x >> 4)) & 0x0F0F0F0F
            fc.text.emit_u32(enc::lsr_imm(SCRATCH0, dst_hw, 4));
            fc.text.emit_u32(enc::add_reg(dst_hw, dst_hw, SCRATCH0));
            fc.emit_load_u32(mask_tmp, 0x0F0F0F0F);
            fc.text.emit_u32(enc::and_reg(dst_hw, dst_hw, mask_tmp));
            // x = x * 0x01010101 >> 24
            fc.emit_load_u32(mask_tmp, 0x01010101);
            fc.text.emit_u32(enc::mul(dst_hw, dst_hw, mask_tmp));
            fc.text.emit_u32(enc::lsr_imm(dst_hw, dst_hw, 24));
        }
        MachineIntUnaryOp::Extend8S => {
            fc.text.emit_u32(enc::sxtb(dst_hw, src_hw));
        }
        MachineIntUnaryOp::Extend16S => {
            fc.text.emit_u32(enc::sxth(dst_hw, src_hw));
        }
        MachineIntUnaryOp::Extend32S => {
            // On 32-bit, this is a no-op (value is already 32 bits)
            if dst_hw != src_hw {
                fc.text.emit_u32(enc::mov_reg(dst_hw, src_hw));
            }
        }
    }
    Ok(())
}

// ─── Integer widening multiply ──────────────────────────────────────────────

fn compile_int_mul_wide(
    fc: &mut FunctionCompiler<'_>,
    sign: MachineSign,
    dst_lo: MachineReg,
    dst_hi: MachineReg,
    lhs: &MachineValue,
    rhs: &MachineValue,
) -> Result<(), WasmError> {
    let dst_lo_hw = map_reg(dst_lo)?;
    let dst_hi_hw = map_reg(dst_hi)?;
    spill_caller_saved_gp_regs(fc);
    emit_move_gp_value(fc, Arm32Reg::R0, lhs)?;
    emit_move_gp_value(fc, Arm32Reg::R1, rhs)?;
    emit_host_call(
        fc,
        match sign {
            MachineSign::Signed => armv7a_smul_wide as usize,
            MachineSign::Unsigned => armv7a_umul_wide as usize,
        },
    );
    emit_pair_results_from_r0_r1(fc, dst_lo, dst_hi)?;
    restore_caller_saved_gp_regs(fc, &[dst_lo_hw, dst_hi_hw]);
    Ok(())
}

// ─── I64 pair binary ────────────────────────────────────────────────────────

fn compile_int64_pair_binary(
    fc: &mut FunctionCompiler<'_>,
    op: MachineIntBinaryOp,
    dst_lo: MachineReg,
    dst_hi: MachineReg,
    lhs_lo: &MachineValue,
    lhs_hi: &MachineValue,
    rhs_lo: &MachineValue,
    rhs_hi: &MachineValue,
) -> Result<(), WasmError> {
    match op {
        MachineIntBinaryOp::Add => {
            let dst_lo_hw = map_reg(dst_lo)?;
            let dst_hi_hw = map_reg(dst_hi)?;
            spill_caller_saved_gp_regs(fc);
            emit_quad_args_to_r0_r3(fc, lhs_lo, lhs_hi, rhs_lo, rhs_hi)?;
            fc.text
                .emit_u32(enc::adds_reg(Arm32Reg::R0, Arm32Reg::R0, Arm32Reg::R2));
            fc.text
                .emit_u32(enc::adc_reg(Arm32Reg::R1, Arm32Reg::R1, Arm32Reg::R3));
            emit_pair_results_from_r0_r1(fc, dst_lo, dst_hi)?;
            restore_caller_saved_gp_regs(fc, &[dst_lo_hw, dst_hi_hw]);
            Ok(())
        }
        MachineIntBinaryOp::Sub => {
            let dst_lo_hw = map_reg(dst_lo)?;
            let dst_hi_hw = map_reg(dst_hi)?;
            spill_caller_saved_gp_regs(fc);
            emit_quad_args_to_r0_r3(fc, lhs_lo, lhs_hi, rhs_lo, rhs_hi)?;
            fc.text
                .emit_u32(enc::subs_reg(Arm32Reg::R0, Arm32Reg::R0, Arm32Reg::R2));
            fc.text
                .emit_u32(enc::sbc_reg(Arm32Reg::R1, Arm32Reg::R1, Arm32Reg::R3));
            emit_pair_results_from_r0_r1(fc, dst_lo, dst_hi)?;
            restore_caller_saved_gp_regs(fc, &[dst_lo_hw, dst_hi_hw]);
            Ok(())
        }
        MachineIntBinaryOp::Mul => {
            let dst_lo_hw = map_reg(dst_lo)?;
            let dst_hi_hw = map_reg(dst_hi)?;
            spill_caller_saved_gp_regs(fc);
            emit_quad_args_to_r0_r3(fc, lhs_lo, lhs_hi, rhs_lo, rhs_hi)?;
            emit_host_call(fc, armv7a_i64_mul as usize);
            emit_pair_results_from_r0_r1(fc, dst_lo, dst_hi)?;
            restore_caller_saved_gp_regs(fc, &[dst_lo_hw, dst_hi_hw]);
            Ok(())
        }
        MachineIntBinaryOp::And | MachineIntBinaryOp::Or | MachineIntBinaryOp::Xor => {
            let dst_lo_hw = map_reg(dst_lo)?;
            let dst_hi_hw = map_reg(dst_hi)?;
            materialize_gp_into(fc, dst_lo_hw, lhs_lo)?;
            materialize_gp_into(fc, dst_hi_hw, lhs_hi)?;
            let rhs_lo_hw = materialize_gp_value(fc, rhs_lo, SCRATCH0)?;
            let rhs_hi_hw = materialize_gp_value(fc, rhs_hi, SCRATCH1)?;
            let emit = match op {
                MachineIntBinaryOp::And => enc::and_reg,
                MachineIntBinaryOp::Or => enc::orr_reg,
                MachineIntBinaryOp::Xor => enc::eor_reg,
                _ => unreachable!(),
            };
            fc.text.emit_u32(emit(dst_lo_hw, dst_lo_hw, rhs_lo_hw));
            fc.text.emit_u32(emit(dst_hi_hw, dst_hi_hw, rhs_hi_hw));
            Ok(())
        }
        other => Err(WasmError::invalid(alloc::format!(
            "armv7a: unsupported i64 pair binary op {:?}",
            other
        ))),
    }
}

// ─── I64 pair unary ─────────────────────────────────────────────────────────

fn compile_int64_pair_unary(
    fc: &mut FunctionCompiler<'_>,
    op: MachineIntUnaryOp,
    dst_lo: MachineReg,
    dst_hi: MachineReg,
    src_lo: &MachineValue,
    src_hi: &MachineValue,
) -> Result<(), WasmError> {
    let dst_lo_hw = map_reg(dst_lo)?;
    let dst_hi_hw = map_reg(dst_hi)?;
    spill_caller_saved_gp_regs(fc);
    match op {
        MachineIntUnaryOp::Clz | MachineIntUnaryOp::Ctz | MachineIntUnaryOp::Popcnt => {
            emit_pair_args_to_r0_r1(fc, src_lo, src_hi)?;
            emit_host_call(
                fc,
                match op {
                    MachineIntUnaryOp::Clz => armv7a_i64_clz as usize,
                    MachineIntUnaryOp::Ctz => armv7a_i64_ctz as usize,
                    MachineIntUnaryOp::Popcnt => armv7a_i64_popcnt as usize,
                    _ => unreachable!(),
                },
            );
            emit_pair_results_from_r0_r1(fc, dst_lo, dst_hi)?;
            restore_caller_saved_gp_regs(fc, &[dst_lo_hw, dst_hi_hw]);
            Ok(())
        }
        MachineIntUnaryOp::Extend8S => {
            let src_lo_hw = materialize_gp_value(fc, src_lo, SCRATCH0)?;
            fc.text.emit_u32(enc::sxtb(dst_lo_hw, src_lo_hw));
            fc.text.emit_u32(enc::asr_imm(dst_hi_hw, dst_lo_hw, 31));
            restore_caller_saved_gp_regs(fc, &[dst_lo_hw, dst_hi_hw]);
            Ok(())
        }
        MachineIntUnaryOp::Extend16S => {
            let src_lo_hw = materialize_gp_value(fc, src_lo, SCRATCH0)?;
            fc.text.emit_u32(enc::sxth(dst_lo_hw, src_lo_hw));
            fc.text.emit_u32(enc::asr_imm(dst_hi_hw, dst_lo_hw, 31));
            restore_caller_saved_gp_regs(fc, &[dst_lo_hw, dst_hi_hw]);
            Ok(())
        }
        MachineIntUnaryOp::Extend32S => {
            let src_lo_hw = materialize_gp_value(fc, src_lo, SCRATCH0)?;
            if dst_lo_hw != src_lo_hw {
                fc.text.emit_u32(enc::mov_reg(dst_lo_hw, src_lo_hw));
            }
            fc.text.emit_u32(enc::asr_imm(dst_hi_hw, dst_lo_hw, 31));
            restore_caller_saved_gp_regs(fc, &[dst_lo_hw, dst_hi_hw]);
            Ok(())
        }
        other => Err(WasmError::invalid(alloc::format!(
            "armv7a: unsupported i64 pair unary op {:?}",
            other
        ))),
    }
}

// ─── I64 pair div/rem ───────────────────────────────────────────────────────

fn compile_int64_pair_div_rem(
    fc: &mut FunctionCompiler<'_>,
    sign: MachineSign,
    rem: bool,
    dst_lo: MachineReg,
    dst_hi: MachineReg,
    lhs_lo: &MachineValue,
    lhs_hi: &MachineValue,
    rhs_lo: &MachineValue,
    rhs_hi: &MachineValue,
) -> Result<(), WasmError> {
    let dst_lo_hw = map_reg(dst_lo)?;
    let dst_hi_hw = map_reg(dst_hi)?;
    spill_caller_saved_gp_regs(fc);
    let trap_div_zero = fc.alloc_label(LabelKind::Block);
    let trap_overflow = fc.alloc_label(LabelKind::Block);
    let after_traps = fc.alloc_label(LabelKind::Block);
    emit_quad_args_to_r0_r3(fc, lhs_lo, lhs_hi, rhs_lo, rhs_hi)?;

    fc.text
        .emit_u32(enc::orr_reg(SCRATCH0, Arm32Reg::R2, Arm32Reg::R3));
    fc.text.emit_u32(enc::cmp_imm(SCRATCH0, 0, 0));
    let non_zero = fc.alloc_label(LabelKind::Block);
    fc.emit_branch(BranchFixupKind::BCond(Cond::Ne), non_zero);
    fc.emit_branch(BranchFixupKind::B, trap_div_zero);
    fc.bind_label(non_zero);

    if matches!(sign, MachineSign::Signed) && !rem {
        fc.emit_load_u32(SCRATCH0, 0x8000_0000);
        fc.text.emit_u32(enc::cmp_reg(Arm32Reg::R1, SCRATCH0));
        let no_overflow = fc.alloc_label(LabelKind::Block);
        fc.emit_branch(BranchFixupKind::BCond(Cond::Ne), no_overflow);
        fc.text.emit_u32(enc::cmp_imm(Arm32Reg::R0, 0, 0));
        let no_overflow_lo = fc.alloc_label(LabelKind::Block);
        fc.emit_branch(BranchFixupKind::BCond(Cond::Ne), no_overflow_lo);
        fc.emit_load_u32(SCRATCH0, u32::MAX);
        fc.text.emit_u32(enc::cmp_reg(Arm32Reg::R2, SCRATCH0));
        let no_overflow_rhs_lo = fc.alloc_label(LabelKind::Block);
        fc.emit_branch(BranchFixupKind::BCond(Cond::Ne), no_overflow_rhs_lo);
        fc.text.emit_u32(enc::cmp_reg(Arm32Reg::R3, SCRATCH0));
        let no_overflow_rhs_hi = fc.alloc_label(LabelKind::Block);
        fc.emit_branch(BranchFixupKind::BCond(Cond::Ne), no_overflow_rhs_hi);
        fc.emit_branch(BranchFixupKind::B, trap_overflow);
        fc.bind_label(no_overflow);
        fc.bind_label(no_overflow_lo);
        fc.bind_label(no_overflow_rhs_lo);
        fc.bind_label(no_overflow_rhs_hi);
    }
    fc.emit_branch(BranchFixupKind::B, after_traps);

    fc.bind_label(trap_div_zero);
    emit_stack_temp_free(fc, 16);
    emit_raise_trap_and_return(fc, 5)?;

    fc.bind_label(trap_overflow);
    emit_stack_temp_free(fc, 16);
    emit_raise_trap_and_return(fc, 6)?;

    fc.bind_label(after_traps);

    emit_host_call(
        fc,
        match (sign, rem) {
            (MachineSign::Signed, false) => armv7a_i64_div_s as usize,
            (MachineSign::Unsigned, false) => armv7a_i64_div_u as usize,
            (MachineSign::Signed, true) => armv7a_i64_rem_s as usize,
            (MachineSign::Unsigned, true) => armv7a_i64_rem_u as usize,
        },
    );
    emit_pair_results_from_r0_r1(fc, dst_lo, dst_hi)?;
    restore_caller_saved_gp_regs(fc, &[dst_lo_hw, dst_hi_hw]);
    Ok(())
}

// ─── I64 pair shift ─────────────────────────────────────────────────────────

fn compile_int64_pair_shift(
    fc: &mut FunctionCompiler<'_>,
    op: MachineIntBinaryOp,
    dst_lo: MachineReg,
    dst_hi: MachineReg,
    lhs_lo: &MachineValue,
    lhs_hi: &MachineValue,
    rhs: &MachineValue,
) -> Result<(), WasmError> {
    let dst_lo_hw = map_reg(dst_lo)?;
    let dst_hi_hw = map_reg(dst_hi)?;
    spill_caller_saved_gp_regs(fc);
    emit_move_gp_value(fc, Arm32Reg::R2, rhs)?;
    emit_pair_args_to_r0_r1(fc, lhs_lo, lhs_hi)?;
    emit_host_call(
        fc,
        match op {
            MachineIntBinaryOp::Shl => armv7a_i64_shl as usize,
            MachineIntBinaryOp::ShrS => armv7a_i64_shr_s as usize,
            MachineIntBinaryOp::ShrU => armv7a_i64_shr_u as usize,
            MachineIntBinaryOp::Rotl => armv7a_i64_rotl as usize,
            MachineIntBinaryOp::Rotr => armv7a_i64_rotr as usize,
            other => {
                return Err(WasmError::invalid(alloc::format!(
                    "armv7a: unsupported i64 pair shift op {:?}",
                    other
                )));
            }
        },
    );
    emit_pair_results_from_r0_r1(fc, dst_lo, dst_hi)?;
    restore_caller_saved_gp_regs(fc, &[dst_lo_hw, dst_hi_hw]);
    Ok(())
}

// ─── I64 pair compare ───────────────────────────────────────────────────────

fn compile_int64_pair_compare(
    fc: &mut FunctionCompiler<'_>,
    kind: MachineCompareKind,
    sign: MachineSign,
    dst: MachineReg,
    lhs_lo: &MachineValue,
    lhs_hi: &MachineValue,
    rhs_lo: &MachineValue,
    rhs_hi: &MachineValue,
) -> Result<(), WasmError> {
    let dst_hw = map_reg(dst)?;
    spill_caller_saved_gp_regs(fc);
    let set_true = fc.alloc_label(LabelKind::Block);
    let set_false = fc.alloc_label(LabelKind::Block);
    let done = fc.alloc_label(LabelKind::Block);

    let hi_lt = match sign {
        MachineSign::Signed => Cond::Lt,
        MachineSign::Unsigned => Cond::Cc,
    };
    let hi_gt = match sign {
        MachineSign::Signed => Cond::Gt,
        MachineSign::Unsigned => Cond::Hi,
    };

    match kind {
        MachineCompareKind::Eq => {
            emit_cmp_gp_values(fc, lhs_hi, rhs_hi)?;
            fc.emit_branch(BranchFixupKind::BCond(Cond::Ne), set_false);
            emit_cmp_gp_values(fc, lhs_lo, rhs_lo)?;
            fc.emit_branch(BranchFixupKind::BCond(Cond::Eq), set_true);
        }
        MachineCompareKind::Ne => {
            emit_cmp_gp_values(fc, lhs_hi, rhs_hi)?;
            fc.emit_branch(BranchFixupKind::BCond(Cond::Ne), set_true);
            emit_cmp_gp_values(fc, lhs_lo, rhs_lo)?;
            fc.emit_branch(BranchFixupKind::BCond(Cond::Ne), set_true);
        }
        MachineCompareKind::Lt => {
            emit_cmp_gp_values(fc, lhs_hi, rhs_hi)?;
            fc.emit_branch(BranchFixupKind::BCond(hi_lt), set_true);
            fc.emit_branch(BranchFixupKind::BCond(hi_gt), set_false);
            emit_cmp_gp_values(fc, lhs_lo, rhs_lo)?;
            fc.emit_branch(BranchFixupKind::BCond(Cond::Cc), set_true);
        }
        MachineCompareKind::Le => {
            emit_cmp_gp_values(fc, lhs_hi, rhs_hi)?;
            fc.emit_branch(BranchFixupKind::BCond(hi_lt), set_true);
            fc.emit_branch(BranchFixupKind::BCond(hi_gt), set_false);
            emit_cmp_gp_values(fc, lhs_lo, rhs_lo)?;
            fc.emit_branch(BranchFixupKind::BCond(Cond::Ls), set_true);
        }
        MachineCompareKind::Gt => {
            emit_cmp_gp_values(fc, lhs_hi, rhs_hi)?;
            fc.emit_branch(BranchFixupKind::BCond(hi_gt), set_true);
            fc.emit_branch(BranchFixupKind::BCond(hi_lt), set_false);
            emit_cmp_gp_values(fc, lhs_lo, rhs_lo)?;
            fc.emit_branch(BranchFixupKind::BCond(Cond::Hi), set_true);
        }
        MachineCompareKind::Ge => {
            emit_cmp_gp_values(fc, lhs_hi, rhs_hi)?;
            fc.emit_branch(BranchFixupKind::BCond(hi_gt), set_true);
            fc.emit_branch(BranchFixupKind::BCond(hi_lt), set_false);
            emit_cmp_gp_values(fc, lhs_lo, rhs_lo)?;
            fc.emit_branch(BranchFixupKind::BCond(Cond::Cs), set_true);
        }
    }

    fc.emit_branch(BranchFixupKind::B, set_false);
    fc.bind_label(set_true);
    emit_set_bool_immediate(fc, dst_hw, true);
    fc.emit_branch(BranchFixupKind::B, done);
    fc.bind_label(set_false);
    emit_set_bool_immediate(fc, dst_hw, false);
    fc.bind_label(done);
    restore_caller_saved_gp_regs(fc, &[dst_hw]);
    Ok(())
}

// ─── I64 pair → float conversion ────────────────────────────────────────────

fn compile_convert_i64_pair_to_float(
    fc: &mut FunctionCompiler<'_>,
    width: MachineFloatWidth,
    sign: MachineSign,
    dst: MachineReg,
    src_lo: &MachineValue,
    src_hi: &MachineValue,
) -> Result<(), WasmError> {
    spill_caller_saved_gp_regs(fc);
    emit_pair_args_to_r0_r1(fc, src_lo, src_hi)?;
    emit_host_call(
        fc,
        match (width, sign) {
            (MachineFloatWidth::F32, MachineSign::Signed) => armv7a_i64s_to_f32 as usize,
            (MachineFloatWidth::F32, MachineSign::Unsigned) => armv7a_i64u_to_f32 as usize,
            (MachineFloatWidth::F64, MachineSign::Signed) => armv7a_i64s_to_f64 as usize,
            (MachineFloatWidth::F64, MachineSign::Unsigned) => armv7a_i64u_to_f64 as usize,
        },
    );

    match width {
        MachineFloatWidth::F32 => {
            let dst_s = fc.map_fp_dreg(dst)? * 2;
            let s0 = FP_SCRATCH0 * 2;
            if dst_s != s0 {
                fc.text.emit_u32(enc::vmov_s(dst_s, s0));
            }
        }
        MachineFloatWidth::F64 => {
            let dst_d = fc.map_fp_dreg(dst)?;
            if dst_d != FP_SCRATCH0 {
                fc.text.emit_u32(enc::vmov_d(dst_d, FP_SCRATCH0));
            }
        }
    }
    restore_caller_saved_gp_regs(fc, &[]);
    Ok(())
}

// ─── Float → I64 pair conversion ────────────────────────────────────────────

fn compile_convert_float_to_i64_pair(
    fc: &mut FunctionCompiler<'_>,
    op: MachineConvertOp,
    dst_lo: MachineReg,
    dst_hi: MachineReg,
    src: &MachineValue,
) -> Result<(), WasmError> {
    let src_is_f32 = matches!(
        op,
        MachineConvertOp::I64TruncF32S
            | MachineConvertOp::I64TruncF32U
            | MachineConvertOp::I64TruncSatF32S
            | MachineConvertOp::I64TruncSatF32U
    );
    let src_d = materialize_float_value_dreg(
        fc,
        if src_is_f32 {
            MachineFloatWidth::F32
        } else {
            MachineFloatWidth::F64
        },
        src,
        FP_SCRATCH1,
    )?;

    if src_is_f32 {
        let src_s = src_d * 2;
        let s0 = FP_SCRATCH0 * 2;
        if src_s != s0 {
            fc.text.emit_u32(enc::vmov_s(s0, src_s));
        }
        fc.text.emit_u32(enc::vmov_r_s(Arm32Reg::R0, s0));
        fc.emit_load_u32(Arm32Reg::R1, 0);
    } else {
        if src_d != FP_SCRATCH0 {
            fc.text.emit_u32(enc::vmov_d(FP_SCRATCH0, src_d));
        }
        fc.text
            .emit_u32(enc::vmov_rr_d(Arm32Reg::R0, Arm32Reg::R1, FP_SCRATCH0));
    }
    fc.emit_load_u32(Arm32Reg::R2, convert_op_code(op));

    if matches!(
        op,
        MachineConvertOp::I64TruncSatF32S
            | MachineConvertOp::I64TruncSatF32U
            | MachineConvertOp::I64TruncSatF64S
            | MachineConvertOp::I64TruncSatF64U
    ) {
        emit_host_call(fc, armv7a_saturating_trunc as usize);
        return emit_pair_results_from_r0_r1(fc, dst_lo, dst_hi);
    }

    emit_trunc_result_buffer_alloc(fc);
    fc.text
        .emit_u32(enc::mov_reg(Arm32Reg::R3, map_fixed_reg(MACHINE_CTX_REG)));
    fc.text.emit_u32(enc::add_imm(SCRATCH0, Arm32Reg::SP, 8, 0));
    fc.text.emit_u32(enc::str_imm(SCRATCH0, Arm32Reg::SP, 0));
    emit_host_call(fc, armv7a_trapping_trunc as usize);
    fc.text.emit_u32(enc::cmp_imm(Arm32Reg::R0, 0, 0));
    let ok = fc.alloc_label(LabelKind::Block);
    fc.emit_branch(BranchFixupKind::BCond(Cond::Eq), ok);
    emit_trunc_result_buffer_free(fc);
    fc.emit_load_u32(Arm32Reg::R0, 1);
    emit_shared_epilogue(&mut fc.text);
    fc.bind_label(ok);
    fc.text
        .emit_u32(enc::ldr_imm(Arm32Reg::R0, Arm32Reg::SP, 8));
    fc.text
        .emit_u32(enc::ldr_imm(Arm32Reg::R1, Arm32Reg::SP, 12));
    emit_trunc_result_buffer_free(fc);
    emit_pair_results_from_r0_r1(fc, dst_lo, dst_hi)
}

// ─── Float → I32 conversion ────────────────────────────────────────────────

fn compile_convert_float_to_i32(
    fc: &mut FunctionCompiler<'_>,
    op: MachineConvertOp,
    dst: MachineReg,
    src: &MachineValue,
) -> Result<(), WasmError> {
    let src_is_f32 = matches!(
        op,
        MachineConvertOp::I32TruncF32S
            | MachineConvertOp::I32TruncF32U
            | MachineConvertOp::I32TruncSatF32S
            | MachineConvertOp::I32TruncSatF32U
    );
    let src_d = materialize_float_value_dreg(
        fc,
        if src_is_f32 {
            MachineFloatWidth::F32
        } else {
            MachineFloatWidth::F64
        },
        src,
        FP_SCRATCH1,
    )?;
    let dst_hw = map_reg(dst)?;

    if src_is_f32 {
        let src_s = src_d * 2;
        let s0 = FP_SCRATCH0 * 2;
        if src_s != s0 {
            fc.text.emit_u32(enc::vmov_s(s0, src_s));
        }
        fc.text.emit_u32(enc::vmov_r_s(Arm32Reg::R0, s0));
        fc.emit_load_u32(Arm32Reg::R1, 0);
    } else {
        if src_d != FP_SCRATCH0 {
            fc.text.emit_u32(enc::vmov_d(FP_SCRATCH0, src_d));
        }
        fc.text
            .emit_u32(enc::vmov_rr_d(Arm32Reg::R0, Arm32Reg::R1, FP_SCRATCH0));
    }
    fc.emit_load_u32(Arm32Reg::R2, convert_op_code(op));

    if matches!(
        op,
        MachineConvertOp::I32TruncSatF32S
            | MachineConvertOp::I32TruncSatF32U
            | MachineConvertOp::I32TruncSatF64S
            | MachineConvertOp::I32TruncSatF64U
    ) {
        emit_host_call(fc, armv7a_saturating_trunc as usize);
        if dst_hw != Arm32Reg::R0 {
            fc.text.emit_u32(enc::mov_reg(dst_hw, Arm32Reg::R0));
        }
        return Ok(());
    }

    emit_trunc_result_buffer_alloc(fc);
    fc.text
        .emit_u32(enc::mov_reg(Arm32Reg::R3, map_fixed_reg(MACHINE_CTX_REG)));
    fc.text.emit_u32(enc::add_imm(SCRATCH0, Arm32Reg::SP, 8, 0));
    fc.text.emit_u32(enc::str_imm(SCRATCH0, Arm32Reg::SP, 0));
    emit_host_call(fc, armv7a_trapping_trunc as usize);
    fc.text.emit_u32(enc::cmp_imm(Arm32Reg::R0, 0, 0));
    let ok = fc.alloc_label(LabelKind::Block);
    fc.emit_branch(BranchFixupKind::BCond(Cond::Eq), ok);
    emit_trunc_result_buffer_free(fc);
    fc.emit_load_u32(Arm32Reg::R0, 1);
    emit_shared_epilogue(&mut fc.text);
    fc.bind_label(ok);
    fc.text
        .emit_u32(enc::ldr_imm(Arm32Reg::R0, Arm32Reg::SP, 8));
    emit_trunc_result_buffer_free(fc);
    if dst_hw != Arm32Reg::R0 {
        fc.text.emit_u32(enc::mov_reg(dst_hw, Arm32Reg::R0));
    }
    Ok(())
}

// ─── Reinterpret F64 ↔ I64 pair ────────────────────────────────────────────

fn compile_reinterpret_f64_to_i64_pair(
    fc: &mut FunctionCompiler<'_>,
    dst_lo: MachineReg,
    dst_hi: MachineReg,
    src: &MachineValue,
) -> Result<(), WasmError> {
    match src {
        MachineValue::Reg(reg) => {
            let dm = fc.map_fp_dreg(*reg)?;
            let dst_lo_hw = map_reg(dst_lo)?;
            let dst_hi_hw = map_reg(dst_hi)?;
            fc.text.emit_u32(enc::vmov_rr_d(dst_lo_hw, dst_hi_hw, dm));
        }
        MachineValue::Imm64(bits) => {
            emit_move_gp_value(
                fc,
                map_reg(dst_lo)?,
                &MachineValue::Imm64(u64::from(*bits as u32)),
            )?;
            emit_move_gp_value(
                fc,
                map_reg(dst_hi)?,
                &MachineValue::Imm64(u64::from((*bits >> 32) as u32)),
            )?;
        }
    }
    Ok(())
}

fn compile_reinterpret_i64_pair_to_f64(
    fc: &mut FunctionCompiler<'_>,
    dst: MachineReg,
    src_lo: &MachineValue,
    src_hi: &MachineValue,
) -> Result<(), WasmError> {
    let dd = fc.map_fp_dreg(dst)?;
    spill_caller_saved_gp_regs(fc);
    emit_pair_args_to_r0_r1(fc, src_lo, src_hi)?;
    fc.text
        .emit_u32(enc::vmov_d_rr(dd, Arm32Reg::R0, Arm32Reg::R1));
    restore_caller_saved_gp_regs(fc, &[]);
    Ok(())
}

// ─── Float ALU ──────────────────────────────────────────────────────────────

fn compile_float_binary(
    fc: &mut FunctionCompiler<'_>,
    width: MachineFloatWidth,
    op: MachineFloatBinaryOp,
    dst: MachineReg,
    lhs: &MachineValue,
    rhs: &MachineValue,
) -> Result<(), WasmError> {
    let dn = materialize_float_value_dreg(fc, width, lhs, FP_SCRATCH1)?;
    let dm = materialize_float_value_dreg(fc, width, rhs, FP_SCRATCH2)?;

    let dd = fc.map_fp_dreg(dst)?;

    match (width, op) {
        (MachineFloatWidth::F64, MachineFloatBinaryOp::Add) => {
            fc.text.emit_u32(enc::vadd_d(dd, dn, dm));
        }
        (MachineFloatWidth::F64, MachineFloatBinaryOp::Sub) => {
            fc.text.emit_u32(enc::vsub_d(dd, dn, dm));
        }
        (MachineFloatWidth::F64, MachineFloatBinaryOp::Mul) => {
            fc.text.emit_u32(enc::vmul_d(dd, dn, dm));
        }
        (MachineFloatWidth::F64, MachineFloatBinaryOp::Div) => {
            fc.text.emit_u32(enc::vdiv_d(dd, dn, dm));
        }
        (MachineFloatWidth::F32, MachineFloatBinaryOp::Add) => {
            fc.text.emit_u32(enc::vadd_s(dd * 2, dn * 2, dm * 2));
        }
        (MachineFloatWidth::F32, MachineFloatBinaryOp::Sub) => {
            fc.text.emit_u32(enc::vsub_s(dd * 2, dn * 2, dm * 2));
        }
        (MachineFloatWidth::F32, MachineFloatBinaryOp::Mul) => {
            fc.text.emit_u32(enc::vmul_s(dd * 2, dn * 2, dm * 2));
        }
        (MachineFloatWidth::F32, MachineFloatBinaryOp::Div) => {
            fc.text.emit_u32(enc::vdiv_s(dd * 2, dn * 2, dm * 2));
        }

        // Min/Max: compare, handle NaN, select
        (MachineFloatWidth::F64, MachineFloatBinaryOp::Min) => {
            // wasm min: if either is NaN → NaN; min(-0,+0) → -0
            fc.text.emit_u32(enc::vcmp_d(dn, dm));
            fc.text.emit_u32(enc::vmrs_apsr());
            // If unordered (NaN): result = lhs + rhs (propagates NaN)
            let no_nan = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::BCond(Cond::Vc), no_nan);
            fc.text.emit_u32(enc::vadd_d(dd, dn, dm));
            let done = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::B, done);
            fc.bind_label(no_nan);
            // MI = lhs < rhs → lhs; GT = lhs > rhs → rhs; EQ = equal, pick rhs for -0 handling
            // Use VBSL or conditional: select lhs if MI, else rhs
            // Simplest: if lhs < rhs then dd=dn else dd=dm
            if dd != dm {
                fc.text.emit_u32(enc::vmov_d(dd, dm));
            }
            fc.text.emit_u32(enc::vcmp_d(dn, dm));
            fc.text.emit_u32(enc::vmrs_apsr());
            // If MI (lhs < rhs), overwrite with lhs
            let skip = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::BCond(Cond::Pl), skip);
            fc.text.emit_u32(enc::vmov_d(dd, dn));
            fc.bind_label(skip);
            fc.bind_label(done);
        }
        (MachineFloatWidth::F64, MachineFloatBinaryOp::Max) => {
            fc.text.emit_u32(enc::vcmp_d(dn, dm));
            fc.text.emit_u32(enc::vmrs_apsr());
            let no_nan = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::BCond(Cond::Vc), no_nan);
            fc.text.emit_u32(enc::vadd_d(dd, dn, dm));
            let done = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::B, done);
            fc.bind_label(no_nan);
            if dd != dm {
                fc.text.emit_u32(enc::vmov_d(dd, dm));
            }
            fc.text.emit_u32(enc::vcmp_d(dn, dm));
            fc.text.emit_u32(enc::vmrs_apsr());
            let skip = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::BCond(Cond::Mi), skip); // lhs < rhs → keep rhs
            fc.text.emit_u32(enc::vmov_d(dd, dn)); // lhs >= rhs → lhs
            fc.bind_label(skip);
            fc.bind_label(done);
        }
        (MachineFloatWidth::F32, MachineFloatBinaryOp::Min) => {
            let sdd = dd * 2;
            let sdn = dn * 2;
            let sdm = dm * 2;
            fc.text.emit_u32(enc::vcmp_s(sdn, sdm));
            fc.text.emit_u32(enc::vmrs_apsr());
            let no_nan = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::BCond(Cond::Vc), no_nan);
            fc.text.emit_u32(enc::vadd_s(sdd, sdn, sdm));
            let done = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::B, done);
            fc.bind_label(no_nan);
            if sdd != sdm {
                fc.text.emit_u32(enc::vmov_s(sdd, sdm));
            }
            fc.text.emit_u32(enc::vcmp_s(sdn, sdm));
            fc.text.emit_u32(enc::vmrs_apsr());
            let skip = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::BCond(Cond::Pl), skip);
            fc.text.emit_u32(enc::vmov_s(sdd, sdn));
            fc.bind_label(skip);
            fc.bind_label(done);
        }
        (MachineFloatWidth::F32, MachineFloatBinaryOp::Max) => {
            let sdd = dd * 2;
            let sdn = dn * 2;
            let sdm = dm * 2;
            fc.text.emit_u32(enc::vcmp_s(sdn, sdm));
            fc.text.emit_u32(enc::vmrs_apsr());
            let no_nan = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::BCond(Cond::Vc), no_nan);
            fc.text.emit_u32(enc::vadd_s(sdd, sdn, sdm));
            let done = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::B, done);
            fc.bind_label(no_nan);
            if sdd != sdm {
                fc.text.emit_u32(enc::vmov_s(sdd, sdm));
            }
            fc.text.emit_u32(enc::vcmp_s(sdn, sdm));
            fc.text.emit_u32(enc::vmrs_apsr());
            let skip = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::BCond(Cond::Mi), skip);
            fc.text.emit_u32(enc::vmov_s(sdd, sdn));
            fc.bind_label(skip);
            fc.bind_label(done);
        }

        // Copysign: take magnitude from lhs, sign from rhs
        (MachineFloatWidth::F64, MachineFloatBinaryOp::Copysign) => {
            let (sign_imm8, sign_rot) = enc::encode_arm_imm(0x8000_0000).unwrap();
            // Extract sign bit of rhs (bit 63) into R0
            fc.text
                .emit_u32(enc::vmov_rr_d(Arm32Reg::R0, Arm32Reg::R1, dm));
            // R1 has the high word with the sign bit
            // Extract magnitude of lhs
            fc.text
                .emit_u32(enc::vmov_rr_d(Arm32Reg::R2, Arm32Reg::R3, dn));
            // Clear sign bit of lhs high word, insert sign bit from rhs
            fc.text.emit_u32(enc::bic_imm(
                Arm32Reg::R3,
                Arm32Reg::R3,
                sign_imm8,
                sign_rot,
            ));
            fc.text.emit_u32(enc::and_imm(
                Arm32Reg::R1,
                Arm32Reg::R1,
                sign_imm8,
                sign_rot,
            ));
            fc.text
                .emit_u32(enc::orr_reg(Arm32Reg::R3, Arm32Reg::R3, Arm32Reg::R1));
            fc.text
                .emit_u32(enc::vmov_d_rr(dd, Arm32Reg::R2, Arm32Reg::R3));
        }
        (MachineFloatWidth::F32, MachineFloatBinaryOp::Copysign) => {
            let (sign_imm8, sign_rot) = enc::encode_arm_imm(0x8000_0000).unwrap();
            let sdn = dn * 2;
            let sdm = dm * 2;
            let sdd = dd * 2;
            fc.text.emit_u32(enc::vmov_r_s(Arm32Reg::R0, sdn)); // lhs bits
            fc.text.emit_u32(enc::vmov_r_s(Arm32Reg::R1, sdm)); // rhs bits
            fc.text.emit_u32(enc::bic_imm(
                Arm32Reg::R0,
                Arm32Reg::R0,
                sign_imm8,
                sign_rot,
            ));
            fc.text.emit_u32(enc::and_imm(
                Arm32Reg::R1,
                Arm32Reg::R1,
                sign_imm8,
                sign_rot,
            ));
            fc.text
                .emit_u32(enc::orr_reg(Arm32Reg::R0, Arm32Reg::R0, Arm32Reg::R1));
            fc.text.emit_u32(enc::vmov_s_r(sdd, Arm32Reg::R0));
        }
    }
    Ok(())
}

// ─── Float unary ────────────────────────────────────────────────────────────

fn compile_float_unary(
    fc: &mut FunctionCompiler<'_>,
    width: MachineFloatWidth,
    op: MachineFloatUnaryOp,
    dst: MachineReg,
    src: &MachineValue,
) -> Result<(), WasmError> {
    let dm = materialize_float_value_dreg(fc, width, src, FP_SCRATCH1)?;
    let dd = fc.map_fp_dreg(dst)?;

    match (width, op) {
        (MachineFloatWidth::F64, MachineFloatUnaryOp::Abs) => {
            fc.text.emit_u32(enc::vabs_d(dd, dm));
        }
        (MachineFloatWidth::F64, MachineFloatUnaryOp::Neg) => {
            fc.text.emit_u32(enc::vneg_d(dd, dm));
        }
        (MachineFloatWidth::F64, MachineFloatUnaryOp::Ceil) => {
            if dm != FP_SCRATCH0 {
                fc.text.emit_u32(enc::vmov_d(FP_SCRATCH0, dm));
            }
            emit_host_call(fc, armv7a_f64_ceil as usize);
            if dd != FP_SCRATCH0 {
                fc.text.emit_u32(enc::vmov_d(dd, FP_SCRATCH0));
            }
        }
        (MachineFloatWidth::F64, MachineFloatUnaryOp::Floor) => {
            if dm != FP_SCRATCH0 {
                fc.text.emit_u32(enc::vmov_d(FP_SCRATCH0, dm));
            }
            emit_host_call(fc, armv7a_f64_floor as usize);
            if dd != FP_SCRATCH0 {
                fc.text.emit_u32(enc::vmov_d(dd, FP_SCRATCH0));
            }
        }
        (MachineFloatWidth::F64, MachineFloatUnaryOp::Trunc) => {
            if dm != FP_SCRATCH0 {
                fc.text.emit_u32(enc::vmov_d(FP_SCRATCH0, dm));
            }
            emit_host_call(fc, armv7a_f64_trunc as usize);
            if dd != FP_SCRATCH0 {
                fc.text.emit_u32(enc::vmov_d(dd, FP_SCRATCH0));
            }
        }
        (MachineFloatWidth::F64, MachineFloatUnaryOp::Nearest) => {
            fc.text
                .emit_u32(enc::vmov_rr_d(Arm32Reg::R0, Arm32Reg::R1, dm));
            emit_host_call(fc, armv7a_f64_nearest_bits as usize);
            fc.text
                .emit_u32(enc::vmov_d_rr(dd, Arm32Reg::R0, Arm32Reg::R1));
        }
        (MachineFloatWidth::F64, MachineFloatUnaryOp::Sqrt) => {
            fc.text.emit_u32(enc::vsqrt_d(dd, dm));
        }
        (MachineFloatWidth::F32, MachineFloatUnaryOp::Abs) => {
            fc.text.emit_u32(enc::vabs_s(dd * 2, dm * 2));
        }
        (MachineFloatWidth::F32, MachineFloatUnaryOp::Neg) => {
            fc.text.emit_u32(enc::vneg_s(dd * 2, dm * 2));
        }
        (MachineFloatWidth::F32, MachineFloatUnaryOp::Ceil) => {
            let src_s = dm * 2;
            let dst_s = dd * 2;
            let s0 = FP_SCRATCH0 * 2;
            if src_s != s0 {
                fc.text.emit_u32(enc::vmov_s(s0, src_s));
            }
            emit_host_call(fc, armv7a_f32_ceil as usize);
            if dst_s != s0 {
                fc.text.emit_u32(enc::vmov_s(dst_s, s0));
            }
        }
        (MachineFloatWidth::F32, MachineFloatUnaryOp::Floor) => {
            let src_s = dm * 2;
            let dst_s = dd * 2;
            let s0 = FP_SCRATCH0 * 2;
            if src_s != s0 {
                fc.text.emit_u32(enc::vmov_s(s0, src_s));
            }
            emit_host_call(fc, armv7a_f32_floor as usize);
            if dst_s != s0 {
                fc.text.emit_u32(enc::vmov_s(dst_s, s0));
            }
        }
        (MachineFloatWidth::F32, MachineFloatUnaryOp::Trunc) => {
            let src_s = dm * 2;
            let dst_s = dd * 2;
            let s0 = FP_SCRATCH0 * 2;
            if src_s != s0 {
                fc.text.emit_u32(enc::vmov_s(s0, src_s));
            }
            emit_host_call(fc, armv7a_f32_trunc as usize);
            if dst_s != s0 {
                fc.text.emit_u32(enc::vmov_s(dst_s, s0));
            }
        }
        (MachineFloatWidth::F32, MachineFloatUnaryOp::Nearest) => {
            let src_s = dm * 2;
            let dst_s = dd * 2;
            fc.text.emit_u32(enc::vmov_r_s(Arm32Reg::R0, src_s));
            emit_host_call(fc, armv7a_f32_nearest_bits as usize);
            fc.text.emit_u32(enc::vmov_s_r(dst_s, Arm32Reg::R0));
        }
        (MachineFloatWidth::F32, MachineFloatUnaryOp::Sqrt) => {
            fc.text.emit_u32(enc::vsqrt_s(dd * 2, dm * 2));
        }
        _ => {
            return Err(WasmError::invalid(alloc::format!(
                "armv7a: unsupported float unary op {:?} {:?}",
                width,
                op
            )));
        }
    }
    Ok(())
}

// ─── Float compare ──────────────────────────────────────────────────────────

fn compile_float_compare(
    fc: &mut FunctionCompiler<'_>,
    width: MachineFloatWidth,
    kind: MachineCompareKind,
    dst: MachineReg,
    lhs: &MachineValue,
    rhs: &MachineValue,
) -> Result<(), WasmError> {
    let lhs_d = materialize_float_value_dreg(fc, width, lhs, FP_SCRATCH1)?;
    let rhs_d = materialize_float_value_dreg(fc, width, rhs, FP_SCRATCH2)?;
    let dst_hw = map_reg(dst)?;

    match width {
        MachineFloatWidth::F64 => {
            fc.text.emit_u32(enc::vcmp_d(lhs_d, rhs_d));
        }
        MachineFloatWidth::F32 => {
            fc.text.emit_u32(enc::vcmp_s(lhs_d * 2, rhs_d * 2));
        }
    }
    fc.text.emit_u32(enc::vmrs_apsr());

    let cond = match kind {
        MachineCompareKind::Eq => Cond::Eq,
        MachineCompareKind::Ne => Cond::Ne,
        MachineCompareKind::Lt => Cond::Mi,
        MachineCompareKind::Gt => Cond::Gt,
        MachineCompareKind::Le => Cond::Ls,
        MachineCompareKind::Ge => Cond::Ge,
    };

    fc.emit_load_u32(dst_hw, 0);
    let (imm8, rot) = enc::encode_arm_imm(1).unwrap();
    fc.text.emit_u32(enc::dp_imm_cond(
        cond,
        0b1101,
        false,
        dst_hw,
        Arm32Reg::R0,
        imm8,
        rot,
    ));
    Ok(())
}

// ─── Convert ────────────────────────────────────────────────────────────────

fn compile_convert(
    fc: &mut FunctionCompiler<'_>,
    op: MachineConvertOp,
    dst: MachineReg,
    src: &MachineValue,
) -> Result<(), WasmError> {
    match op {
        // ─── Integer wrapping/extending (GP → GP) ────────────────────────
        MachineConvertOp::I32WrapI64 => {
            let dst_hw = map_reg(dst)?;
            match src {
                MachineValue::Reg(r) => {
                    let src_hw = map_reg(*r)?;
                    if dst_hw != src_hw {
                        fc.text.emit_u32(enc::mov_reg(dst_hw, src_hw));
                    }
                }
                MachineValue::Imm64(v) => fc.emit_load_u32(dst_hw, *v as u32),
            }
        }
        MachineConvertOp::I64ExtendI32U | MachineConvertOp::I64ExtendI32S => {
            let dst_hw = map_reg(dst)?;
            match src {
                MachineValue::Reg(r) => {
                    let src_hw = map_reg(*r)?;
                    if dst_hw != src_hw {
                        fc.text.emit_u32(enc::mov_reg(dst_hw, src_hw));
                    }
                }
                MachineValue::Imm64(v) => fc.emit_load_u32(dst_hw, *v as u32),
            }
        }

        // ─── F64/F32 → I32 (helper-backed for Wasm trap/sat semantics) ───
        MachineConvertOp::I32TruncF64S
        | MachineConvertOp::I32TruncF64U
        | MachineConvertOp::I32TruncSatF64S
        | MachineConvertOp::I32TruncSatF64U
        | MachineConvertOp::I32TruncF32S
        | MachineConvertOp::I32TruncF32U
        | MachineConvertOp::I32TruncSatF32S
        | MachineConvertOp::I32TruncSatF32U => {
            compile_convert_float_to_i32(fc, op, dst, src)?;
        }

        // ─── F64/F32 → I64 (via helper call, returns low 32 bits) ──────
        MachineConvertOp::I64TruncF64S
        | MachineConvertOp::I64TruncSatF64S
        | MachineConvertOp::I64TruncF64U
        | MachineConvertOp::I64TruncSatF64U
        | MachineConvertOp::I64TruncF32S
        | MachineConvertOp::I64TruncSatF32S
        | MachineConvertOp::I64TruncF32U
        | MachineConvertOp::I64TruncSatF32U => {
            return Err(WasmError::internal(
                "armv7a direct i64 trunc convert should be legalized to ConvertFloatToI64Pair"
                    .into(),
            ));
        }

        // ─── I32 → F64 (GP src → FP dst) ────────────────────────────────
        MachineConvertOp::F64ConvertI32S => {
            let dd = fc.map_fp_dreg(dst)?;
            let src_hw = match src {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            let sd_tmp = FP_SCRATCH0 * 2;
            fc.text.emit_u32(enc::vmov_s_r(sd_tmp, src_hw));
            fc.text.emit_u32(enc::vcvt_d_s32(dd, sd_tmp));
        }
        MachineConvertOp::F64ConvertI32U => {
            let dd = fc.map_fp_dreg(dst)?;
            let src_hw = match src {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            let sd_tmp = FP_SCRATCH0 * 2;
            fc.text.emit_u32(enc::vmov_s_r(sd_tmp, src_hw));
            fc.text.emit_u32(enc::vcvt_d_u32(dd, sd_tmp));
        }

        // ─── I32 → F32 (GP src → FP dst) ────────────────────────────────
        MachineConvertOp::F32ConvertI32S => {
            let sd = fc.map_fp_dreg(dst)? * 2; // S-register
            let src_hw = match src {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            let sd_tmp = FP_SCRATCH0 * 2;
            fc.text.emit_u32(enc::vmov_s_r(sd_tmp, src_hw));
            fc.text.emit_u32(enc::vcvt_s_s32(sd, sd_tmp));
        }
        MachineConvertOp::F32ConvertI32U => {
            let sd = fc.map_fp_dreg(dst)? * 2;
            let src_hw = match src {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            let sd_tmp = FP_SCRATCH0 * 2;
            fc.text.emit_u32(enc::vmov_s_r(sd_tmp, src_hw));
            fc.text.emit_u32(enc::vcvt_s_u32(sd, sd_tmp));
        }

        // ─── F32 ↔ F64 (FP → FP) ───────────────────────────────────────
        MachineConvertOp::F64PromoteF32 => {
            let dd = fc.map_fp_dreg(dst)?;
            let sm =
                materialize_float_value_dreg(fc, MachineFloatWidth::F32, src, FP_SCRATCH1)? * 2;
            fc.text.emit_u32(enc::vcvt_d_s(dd, sm));
        }
        MachineConvertOp::F32DemoteF64 => {
            let sd = fc.map_fp_dreg(dst)? * 2;
            let dm = materialize_float_value_dreg(fc, MachineFloatWidth::F64, src, FP_SCRATCH1)?;
            fc.text.emit_u32(enc::vcvt_s_d(sd, dm));
        }

        // ─── I64 → F64/F32 (via helper call) ─────────────────────────────
        // On ARM32, the GP register holds the low 32 bits of the i64.
        // We sign/zero-extend from the 32-bit value to form the full i64,
        // then call a helper that does the conversion.
        MachineConvertOp::F64ConvertI64S => {
            let dd = fc.map_fp_dreg(dst)?;
            let src_hw = match src {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            // R0 = lo, R1 = hi (sign-extend: hi = lo >> 31, arithmetic shift)
            fc.text.emit_u32(enc::mov_reg(Arm32Reg::R0, src_hw));
            fc.text.emit_u32(enc::asr_imm(Arm32Reg::R1, src_hw, 31));
            emit_host_call(fc, armv7a_i64s_to_f64 as usize);
            // Result is in D0 (EABI: f64 returned in D0)
            if dd != FP_SCRATCH0 {
                fc.text.emit_u32(enc::vmov_d(dd, FP_SCRATCH0));
            }
        }
        MachineConvertOp::F64ConvertI64U => {
            let dd = fc.map_fp_dreg(dst)?;
            let src_hw = match src {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            // R0 = lo, R1 = 0 (zero-extend)
            fc.text.emit_u32(enc::mov_reg(Arm32Reg::R0, src_hw));
            fc.emit_load_u32(Arm32Reg::R1, 0);
            emit_host_call(fc, armv7a_i64u_to_f64 as usize);
            if dd != FP_SCRATCH0 {
                fc.text.emit_u32(enc::vmov_d(dd, FP_SCRATCH0));
            }
        }
        MachineConvertOp::F32ConvertI64S => {
            let sd = fc.map_fp_dreg(dst)? * 2;
            let src_hw = match src {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            fc.text.emit_u32(enc::mov_reg(Arm32Reg::R0, src_hw));
            fc.text.emit_u32(enc::asr_imm(Arm32Reg::R1, src_hw, 31));
            emit_host_call(fc, armv7a_i64s_to_f32 as usize);
            // Result in S0 (EABI: f32 returned in S0)
            let s0 = FP_SCRATCH0 * 2;
            if sd != s0 {
                fc.text.emit_u32(enc::vmov_s(sd, s0));
            }
        }
        MachineConvertOp::F32ConvertI64U => {
            let sd = fc.map_fp_dreg(dst)? * 2;
            let src_hw = match src {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            fc.text.emit_u32(enc::mov_reg(Arm32Reg::R0, src_hw));
            fc.emit_load_u32(Arm32Reg::R1, 0);
            emit_host_call(fc, armv7a_i64u_to_f32 as usize);
            let s0 = FP_SCRATCH0 * 2;
            if sd != s0 {
                fc.text.emit_u32(enc::vmov_s(sd, s0));
            }
        }

        // ─── Reinterpret (bit cast, no conversion) ──────────────────────
        MachineConvertOp::I32ReinterpretF32 => {
            let dst_hw = map_reg(dst)?;
            let sm =
                materialize_float_value_dreg(fc, MachineFloatWidth::F32, src, FP_SCRATCH1)? * 2;
            fc.text.emit_u32(enc::vmov_r_s(dst_hw, sm));
        }
        MachineConvertOp::F32ReinterpretI32 => {
            let sd = fc.map_fp_dreg(dst)? * 2;
            let src_hw = match src {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            fc.text.emit_u32(enc::vmov_s_r(sd, src_hw));
        }
        MachineConvertOp::I64ReinterpretF64 => {
            // F64 (D-reg) → I64 (GP low 32 bits on ARM32)
            let dst_hw = map_reg(dst)?;
            let dm = materialize_float_value_dreg(fc, MachineFloatWidth::F64, src, FP_SCRATCH1)?;
            // VMOV Rlo, Rhi, Dm — extract low 32 bits to dst
            fc.text.emit_u32(enc::vmov_rr_d(dst_hw, Arm32Reg::R1, dm));
        }
        MachineConvertOp::F64ReinterpretI64 => {
            // I64 (GP low 32 bits) → F64 (D-reg)
            let dd = fc.map_fp_dreg(dst)?;
            match src {
                MachineValue::Reg(r) => {
                    let src_hw = map_reg(*r)?;
                    fc.emit_load_u32(Arm32Reg::R1, 0);
                    fc.text.emit_u32(enc::vmov_d_rr(dd, src_hw, Arm32Reg::R1));
                }
                MachineValue::Imm64(v) => {
                    fc.emit_load_u32(Arm32Reg::R0, *v as u32);
                    fc.emit_load_u32(Arm32Reg::R1, (*v >> 32) as u32);
                    fc.text
                        .emit_u32(enc::vmov_d_rr(dd, Arm32Reg::R0, Arm32Reg::R1));
                }
            }
        }

        _ => {
            return Err(WasmError::invalid(alloc::format!(
                "armv7a: unsupported convert op {:?}",
                op
            )));
        }
    }
    Ok(())
}

// ─── Select ─────────────────────────────────────────────────────────────────

fn compile_select(
    fc: &mut FunctionCompiler<'_>,
    dst: MachineReg,
    condition: &MachineValue,
    true_val: &MachineValue,
    false_val: &MachineValue,
) -> Result<(), WasmError> {
    fn gp_value_aliases_dst(
        fc: &FunctionCompiler<'_>,
        value: &MachineValue,
        dst_hw: Arm32Reg,
    ) -> Result<bool, WasmError> {
        match value {
            MachineValue::Reg(r) if !fc.is_fp_machine_reg(*r) => Ok(map_reg(*r)? == dst_hw),
            _ => Ok(false),
        }
    }

    fn emit_gp_select_value(
        fc: &mut FunctionCompiler<'_>,
        dst_hw: Arm32Reg,
        value: &MachineValue,
    ) -> Result<(), WasmError> {
        match value {
            MachineValue::Reg(r) if fc.is_fp_machine_reg(*r) => {
                let sd = fc.map_fp_dreg(*r)?;
                fc.text.emit_u32(enc::vmov_rr_d(dst_hw, Arm32Reg::R1, sd));
            }
            MachineValue::Reg(r) => {
                let src = map_reg(*r)?;
                if dst_hw != src {
                    fc.text.emit_u32(enc::mov_reg(dst_hw, src));
                }
            }
            MachineValue::Imm64(v) => {
                fc.emit_load_u32(dst_hw, *v as u32);
            }
        }
        Ok(())
    }

    fn emit_gp_select_value_cond(
        fc: &mut FunctionCompiler<'_>,
        dst_hw: Arm32Reg,
        value: &MachineValue,
        cond: Cond,
    ) -> Result<(), WasmError> {
        match value {
            MachineValue::Reg(r) if fc.is_fp_machine_reg(*r) => {
                let skip = fc.alloc_label(LabelKind::Block);
                fc.emit_branch(BranchFixupKind::BCond(cond.invert()), skip);
                let sd = fc.map_fp_dreg(*r)?;
                fc.text.emit_u32(enc::vmov_rr_d(dst_hw, Arm32Reg::R1, sd));
                fc.bind_label(skip);
            }
            MachineValue::Reg(r) => {
                let src = map_reg(*r)?;
                fc.text.emit_u32(enc::mov_reg_cond(cond, dst_hw, src));
            }
            MachineValue::Imm64(v) => {
                fc.emit_load_u32(SCRATCH0, *v as u32);
                fc.text.emit_u32(enc::mov_reg_cond(cond, dst_hw, SCRATCH0));
            }
        }
        Ok(())
    }

    if fc.is_fp_machine_reg(dst) {
        // FP select: use branch-based approach since ARM32 has no conditional VMOV
        let dd = fc.map_fp_dreg(dst)?;

        // Test condition first
        let cond_hw = match condition {
            MachineValue::Reg(r) => map_reg(*r)?,
            MachineValue::Imm64(v) => {
                fc.emit_load_u32(SCRATCH0, *v as u32);
                SCRATCH0
            }
        };
        fc.text.emit_u32(enc::cmp_imm(cond_hw, 0, 0));

        let true_label = fc.alloc_label(LabelKind::Block);
        let done_label = fc.alloc_label(LabelKind::Block);
        fc.emit_branch(BranchFixupKind::BCond(Cond::Ne), true_label);

        // False path: load false_val to dd
        match false_val {
            MachineValue::Reg(r) if fc.is_fp_machine_reg(*r) => {
                let sd = fc.map_fp_dreg(*r)?;
                if dd != sd {
                    fc.text.emit_u32(enc::vmov_d(dd, sd));
                }
            }
            MachineValue::Reg(r) => {
                let src = map_reg(*r)?;
                fc.emit_load_u32(Arm32Reg::R1, 0);
                fc.text.emit_u32(enc::vmov_d_rr(dd, src, Arm32Reg::R1));
            }
            MachineValue::Imm64(v) => {
                fc.emit_load_u32(Arm32Reg::R0, *v as u32);
                fc.emit_load_u32(Arm32Reg::R1, (*v >> 32) as u32);
                fc.text
                    .emit_u32(enc::vmov_d_rr(dd, Arm32Reg::R0, Arm32Reg::R1));
            }
        }
        fc.emit_branch(BranchFixupKind::B, done_label);

        // True path: load true_val to dd
        fc.bind_label(true_label);
        match true_val {
            MachineValue::Reg(r) if fc.is_fp_machine_reg(*r) => {
                let sd = fc.map_fp_dreg(*r)?;
                if dd != sd {
                    fc.text.emit_u32(enc::vmov_d(dd, sd));
                }
            }
            MachineValue::Reg(r) => {
                let src = map_reg(*r)?;
                fc.emit_load_u32(Arm32Reg::R1, 0);
                fc.text.emit_u32(enc::vmov_d_rr(dd, src, Arm32Reg::R1));
            }
            MachineValue::Imm64(v) => {
                fc.emit_load_u32(Arm32Reg::R0, *v as u32);
                fc.emit_load_u32(Arm32Reg::R1, (*v >> 32) as u32);
                fc.text
                    .emit_u32(enc::vmov_d_rr(dd, Arm32Reg::R0, Arm32Reg::R1));
            }
        }
        fc.bind_label(done_label);
        return Ok(());
    }

    // GP select
    let dst_hw = map_reg(dst)?;

    // Test condition before touching dst so dst == cond is safe.
    let cond_hw = match condition {
        MachineValue::Reg(r) => map_reg(*r)?,
        MachineValue::Imm64(v) => {
            fc.emit_load_u32(SCRATCH0, *v as u32);
            SCRATCH0
        }
    };
    fc.text.emit_u32(enc::cmp_imm(cond_hw, 0, 0));

    if gp_value_aliases_dst(fc, true_val, dst_hw)? {
        // Loading the false arm first would clobber the live true source when
        // `dst` reuses that register. Seed `dst` with the true arm, then
        // overwrite it on the false path.
        emit_gp_select_value(fc, dst_hw, true_val)?;
        emit_gp_select_value_cond(fc, dst_hw, false_val, Cond::Eq)?;
    } else {
        emit_gp_select_value(fc, dst_hw, false_val)?;
        emit_gp_select_value_cond(fc, dst_hw, true_val, Cond::Ne)?;
    }

    Ok(())
}

// ─── IntCompare ─────────────────────────────────────────────────────────

fn compile_int_compare(
    fc: &mut FunctionCompiler<'_>,
    _width: MachineIntWidth,
    kind: MachineCompareKind,
    sign: MachineSign,
    dst: MachineReg,
    lhs: &MachineValue,
    rhs: &MachineValue,
) -> Result<(), WasmError> {
    let dst_hw = map_reg(dst)?;
    let lhs_hw = match lhs {
        MachineValue::Reg(r) => map_reg(*r)?,
        MachineValue::Imm64(v) => {
            fc.emit_load_u32(SCRATCH0, *v as u32);
            SCRATCH0
        }
    };

    match rhs {
        MachineValue::Reg(r) => {
            fc.text.emit_u32(enc::cmp_reg(lhs_hw, map_reg(*r)?));
        }
        MachineValue::Imm64(v) => {
            if let Some((imm8, rot)) = enc::encode_arm_imm(*v as u32) {
                fc.text.emit_u32(enc::cmp_imm(lhs_hw, imm8, rot));
            } else {
                let tmp = if lhs_hw == SCRATCH0 {
                    Arm32Reg::R3
                } else {
                    SCRATCH0
                };
                fc.emit_load_u32(tmp, *v as u32);
                fc.text.emit_u32(enc::cmp_reg(lhs_hw, tmp));
            }
        }
    }

    let cond = match (kind, sign) {
        (MachineCompareKind::Eq, _) => Cond::Eq,
        (MachineCompareKind::Ne, _) => Cond::Ne,
        (MachineCompareKind::Lt, MachineSign::Signed) => Cond::Lt,
        (MachineCompareKind::Lt, MachineSign::Unsigned) => Cond::Cc,
        (MachineCompareKind::Gt, MachineSign::Signed) => Cond::Gt,
        (MachineCompareKind::Gt, MachineSign::Unsigned) => Cond::Hi,
        (MachineCompareKind::Le, MachineSign::Signed) => Cond::Le,
        (MachineCompareKind::Le, MachineSign::Unsigned) => Cond::Ls,
        (MachineCompareKind::Ge, MachineSign::Signed) => Cond::Ge,
        (MachineCompareKind::Ge, MachineSign::Unsigned) => Cond::Cs,
    };

    fc.emit_load_u32(dst_hw, 0);
    let (imm8, rot) = enc::encode_arm_imm(1).unwrap();
    fc.text.emit_u32(enc::dp_imm_cond(
        cond,
        0b1101,
        false,
        dst_hw,
        Arm32Reg::R0,
        imm8,
        rot,
    ));
    Ok(())
}

// ─── TrapIf ─────────────────────────────────────────────────────────────

fn compile_trap_if(
    fc: &mut FunctionCompiler<'_>,
    kind: MachineTrapKind,
    cond: &MachineBranchCond,
) -> Result<(), WasmError> {
    let arm_cond = compile_branch_condition(fc, cond)?;
    // Skip trap if condition is NOT met
    let skip_label = fc.alloc_label(LabelKind::Block);
    let inv_cond = arm_cond.invert();
    fc.emit_branch(BranchFixupKind::BCond(inv_cond), skip_label);

    // Emit trap inline
    fc.text
        .emit_u32(enc::mov_reg(Arm32Reg::R0, map_fixed_reg(MACHINE_CTX_REG)));
    let trap_code = trap_kind_to_u32(kind);
    fc.emit_load_u32(Arm32Reg::R1, trap_code);
    fc.emit_load_u32(Arm32Reg::R2, fc.current_trap_site());
    emit_host_call(fc, armv7a_raise_trap as usize);
    fc.emit_load_u32(Arm32Reg::R0, 1);
    emit_shared_epilogue(&mut fc.text);

    fc.bind_label(skip_label);
    Ok(())
}

// ─── CallHelper ─────────────────────────────────────────────────────────

fn compile_call_helper(
    fc: &mut FunctionCompiler<'_>,
    call: &MachineHelperCall,
) -> Result<(), WasmError> {
    let binding = fc
        .compiled
        .module()
        .externs
        .get(call.target.0 as usize)
        .ok_or_else(|| {
            WasmError::internal(alloc::format!(
                "armv7a: extern id {} not found",
                call.target.0
            ))
        })?;
    let metadata = fc
        .compiled
        .const_ptr(call.metadata)
        .ok_or_else(|| WasmError::internal("armv7a: helper metadata is out of range".into()))?;

    let helper_ptr = resolve_helper_entry(binding.symbol) as usize;

    // EABI: fn(ctx: *mut NativeContext, frame: *mut u64, metadata: *const u8) -> u32
    fc.text
        .emit_u32(enc::mov_reg(Arm32Reg::R0, map_fixed_reg(MACHINE_CTX_REG)));
    fc.text
        .emit_u32(enc::mov_reg(Arm32Reg::R1, map_fixed_reg(MACHINE_FP_REG)));
    fc.emit_load_addr(Arm32Reg::R2, metadata as usize);

    emit_host_call(fc, helper_ptr);

    // Check return value: if non-zero, return error
    fc.text.emit_u32(enc::cmp_imm(Arm32Reg::R0, 0, 0));
    let ok_label = fc.alloc_label(LabelKind::Block);
    fc.emit_branch(BranchFixupKind::BCond(Cond::Eq), ok_label);
    emit_shared_epilogue(&mut fc.text);
    fc.bind_label(ok_label);

    Ok(())
}
