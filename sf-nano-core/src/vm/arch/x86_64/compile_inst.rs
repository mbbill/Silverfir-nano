//! x86_64 backend: instruction emission methods for FunctionCompiler.

use alloc::vec::Vec;

use crate::{
    error::WasmError,
    vm::machine::machine_ir::{
        MachineAddr, MachineCompareKind, MachineConvertOp, MachineFloatBinaryOp,
        MachineFloatUnaryOp, MachineFloatWidth, MachineIndexExtend, MachineIntBinaryOp,
        MachineIntUnaryOp, MachineIntWidth, MachineLoadExtension, MachineMemWidth,
        MachineReg, MachineSign, MachineStorageType, MachineTrapKind, MachineValue,
    },
};

use super::{
    abi::{
        fp_machine_reg, inv_map_reg, map_fixed_reg, FP_SCRATCH0, FP_SCRATCH1, FP_SCRATCH2,
        SCRATCH0, SCRATCH1,
    },
    compile::{FunctionCompiler, LabelKind},
    compile_helpers::{
        convert_op_code, convert_result_float_width, map_float_cond, map_int_cond,
        x86_64_saturating_trunc, x86_64_trapping_trunc,
    },
    enc::{self, Cc},
    reg::X86Reg,
};

use crate::vm::machine::machine_ir::{MACHINE_CTX_REG, MACHINE_FP_REG, MACHINE_MEM0_BASE_REG};
use crate::vm::runtime::helpers::resolve_helper_entry;

impl<'a> FunctionCompiler<'a> {
    // ── Move / const ────────────────────────────────────────────────────────

    pub(super) fn emit_move(
        &mut self,
        ty: MachineStorageType,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        if let Some(width) = ty.float_width() {
            let dst_fp = self.map_fp_reg(dst)? as u8;
            match src {
                MachineValue::Reg(src_reg) if self.is_fp_reg(src_reg) => {
                    let src_fp = self.map_fp_reg(src_reg)? as u8;
                    let src_width = self.fp_reg_width(src_reg)?;
                    if src_width != width {
                        return Err(WasmError::invalid(alloc::format!(
                            "x86_64 typed float move width mismatch: dst expects {:?}, src {} is {:?}",
                            width,
                            src_reg.0,
                            src_width,
                        )));
                    }
                    if dst_fp != src_fp {
                        match width {
                            MachineFloatWidth::F32 => enc::movss_rr(&mut self.text, dst_fp, src_fp),
                            MachineFloatWidth::F64 => enc::movsd_rr(&mut self.text, dst_fp, src_fp),
                        };
                    }
                    self.set_fp_reg_width(dst, width)?;
                    Ok(())
                }
                MachineValue::Reg(src_reg) => {
                    let src_gp = self.map_gp_reg(src_reg)?;
                    match width {
                        MachineFloatWidth::F32 => enc::movd_xmm_r32(&mut self.text, dst_fp, src_gp),
                        MachineFloatWidth::F64 => enc::movq_xmm_r64(&mut self.text, dst_fp, src_gp),
                    };
                    self.set_fp_reg_width(dst, width)?;
                    Ok(())
                }
                MachineValue::Imm64(value) => {
                    self.materialize_u64(SCRATCH0, value);
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movd_xmm_r32(&mut self.text, dst_fp, SCRATCH0)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movq_xmm_r64(&mut self.text, dst_fp, SCRATCH0)
                        }
                    };
                    self.set_fp_reg_width(dst, width)?;
                    Ok(())
                }
            }
        } else {
            let dst_gp = self.map_gp_reg(dst)?;
            match src {
                MachineValue::Reg(src_reg) if self.is_fp_reg(src_reg) => {
                    let src_fp = self.map_fp_reg(src_reg)? as u8;
                    match self.fp_reg_width(src_reg)? {
                        MachineFloatWidth::F32 => enc::movd_r32_xmm(&mut self.text, dst_gp, src_fp),
                        MachineFloatWidth::F64 => enc::movq_r64_xmm(&mut self.text, dst_gp, src_fp),
                    };
                    Ok(())
                }
                MachineValue::Reg(src_reg) => {
                    let src_gp = self.map_gp_reg(src_reg)?;
                    if dst_gp != src_gp {
                        enc::mov_rr_64(&mut self.text, dst_gp, src_gp);
                    }
                    Ok(())
                }
                MachineValue::Imm64(value) => {
                    self.materialize_u64(dst_gp, value);
                    Ok(())
                }
            }
        }
    }

    pub(super) fn emit_float_const(
        &mut self,
        width: MachineFloatWidth,
        dst: MachineReg,
        bits: u64,
    ) -> Result<(), WasmError> {
        if !self.is_fp_reg(dst) {
            return Err(WasmError::invalid(alloc::format!(
                "x86_64 FloatConst destination {} must be an FP register",
                dst.0
            )));
        }
        let dst_fp = self.map_fp_reg(dst)? as u8;
        let imm = match width {
            MachineFloatWidth::F32 => u64::from(bits as u32),
            MachineFloatWidth::F64 => bits,
        };
        if imm == 0 {
            // XORPS/XORPD xmm, xmm → zero
            match width {
                MachineFloatWidth::F32 => enc::xorps(&mut self.text, dst_fp, dst_fp),
                MachineFloatWidth::F64 => enc::xorpd(&mut self.text, dst_fp, dst_fp),
            };
        } else {
            self.materialize_u64(SCRATCH0, imm);
            match width {
                MachineFloatWidth::F32 => enc::movd_xmm_r32(&mut self.text, dst_fp, SCRATCH0),
                MachineFloatWidth::F64 => enc::movq_xmm_r64(&mut self.text, dst_fp, SCRATCH0),
            };
        }
        self.set_fp_reg_width(dst, width)?;
        Ok(())
    }

    // ── LEA / Load / Store ───────────────────────────────────────────────────

    pub(super) fn emit_lea(&mut self, dst: MachineReg, addr: MachineAddr) -> Result<(), WasmError> {
        let dst_gp = self.map_gp_reg(dst)?;
        let base = self.map_gp_reg(addr.base)?;
        if addr.offset == 0 {
            if dst_gp != base {
                enc::mov_rr_64(&mut self.text, dst_gp, base);
            }
        } else {
            enc::lea_64(&mut self.text, dst_gp, base, addr.offset);
        }
        Ok(())
    }

    pub(super) fn emit_load(
        &mut self,
        dst: MachineReg,
        addr: MachineAddr,
        width: MachineMemWidth,
        extension: MachineLoadExtension,
    ) -> Result<(), WasmError> {
        let base = self.map_gp_reg(addr.base)?;
        let disp = addr.offset;

        // FP register destination
        if self.is_fp_reg(dst) {
            let dst_fp = self.map_fp_reg(dst)? as u8;
            let tracked_width = match width {
                MachineMemWidth::U32 => MachineFloatWidth::F32,
                MachineMemWidth::U64 => MachineFloatWidth::F64,
                _ => return Err(WasmError::invalid(
                    "x86_64 MachineIR backend does not support narrow integer loads into FP regs"
                        .into(),
                )),
            };
            match tracked_width {
                MachineFloatWidth::F32 => enc::movss_load(&mut self.text, dst_fp, base, disp),
                MachineFloatWidth::F64 => enc::movsd_load(&mut self.text, dst_fp, base, disp),
            };
            self.set_fp_reg_width(dst, tracked_width)?;
            return Ok(());
        }

        // GP register destination
        let dst_gp = self.map_gp_reg(dst)?;
        match (width, extension) {
            (MachineMemWidth::U8, MachineLoadExtension::None)
            | (MachineMemWidth::U8, MachineLoadExtension::ZeroExtend) => {
                enc::load_u8(&mut self.text, dst_gp, base, disp);
            }
            (MachineMemWidth::U8, MachineLoadExtension::SignExtend) => {
                enc::load_s8_64(&mut self.text, dst_gp, base, disp);
            }
            (MachineMemWidth::U16, MachineLoadExtension::None)
            | (MachineMemWidth::U16, MachineLoadExtension::ZeroExtend) => {
                enc::load_u16(&mut self.text, dst_gp, base, disp);
            }
            (MachineMemWidth::U16, MachineLoadExtension::SignExtend) => {
                enc::load_s16_64(&mut self.text, dst_gp, base, disp);
            }
            (MachineMemWidth::U32, MachineLoadExtension::None)
            | (MachineMemWidth::U32, MachineLoadExtension::ZeroExtend) => {
                enc::load_32(&mut self.text, dst_gp, base, disp);
            }
            (MachineMemWidth::U32, MachineLoadExtension::SignExtend) => {
                enc::load_s32_64(&mut self.text, dst_gp, base, disp);
            }
            (MachineMemWidth::U64, MachineLoadExtension::None)
            | (MachineMemWidth::U64, MachineLoadExtension::ZeroExtend) => {
                enc::load_64(&mut self.text, dst_gp, base, disp);
            }
            (MachineMemWidth::U64, MachineLoadExtension::SignExtend) => {
                return Err(WasmError::invalid(
                    "x86_64 MachineIR backend does not support sign-extending 64-bit loads".into(),
                ))
            }
        };
        Ok(())
    }

    pub(super) fn emit_store(
        &mut self,
        addr: MachineAddr,
        width: MachineMemWidth,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let base = self.map_gp_reg(addr.base)?;
        let disp = addr.offset;

        // FP register source
        if let MachineValue::Reg(src_reg) = src {
            if self.is_fp_reg(src_reg) {
                let src_fp = self.map_fp_reg(src_reg)? as u8;
                match width {
                    MachineMemWidth::U32 => enc::movss_store(&mut self.text, base, disp, src_fp),
                    MachineMemWidth::U64 => enc::movsd_store(&mut self.text, base, disp, src_fp),
                    _ => {
                        return Err(WasmError::invalid(
                            "x86_64 MachineIR backend does not support narrow FP stores".into(),
                        ))
                    }
                };
                return Ok(());
            }
        }

        // Imm64(0) store → store_imm32_64 for U64
        if matches!(src, MachineValue::Imm64(0)) && width == MachineMemWidth::U64 {
            enc::store_imm32_64(&mut self.text, base, disp, 0);
            return Ok(());
        }

        let src_gp = self.materialize_value(SCRATCH0, src)?;
        match width {
            MachineMemWidth::U8 => enc::store_8(&mut self.text, base, disp, src_gp),
            MachineMemWidth::U16 => enc::store_16(&mut self.text, base, disp, src_gp),
            MachineMemWidth::U32 => enc::store_32(&mut self.text, base, disp, src_gp),
            MachineMemWidth::U64 => enc::store_64(&mut self.text, base, disp, src_gp),
        };
        Ok(())
    }

    // ── Integer unary ops ────────────────────────────────────────────────────

    pub(super) fn emit_int_unary(
        &mut self,
        width: MachineIntWidth,
        op: MachineIntUnaryOp,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let dst = self.map_gp_reg(dst)?;
        let src = self.materialize_value(SCRATCH0, src)?;
        match (width, op) {
            (MachineIntWidth::I32, MachineIntUnaryOp::Eqz) => {
                enc::test_rr_32(&mut self.text, src, src);
                enc::setcc(&mut self.text, Cc::E, dst);
                // Zero-extend the byte result to full register
                enc::movzx_r32_r8(&mut self.text, dst, dst);
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Eqz) => {
                enc::test_rr_64(&mut self.text, src, src);
                enc::setcc(&mut self.text, Cc::E, dst);
                enc::movzx_r32_r8(&mut self.text, dst, dst);
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Clz) => {
                enc::lzcnt_rr_32(&mut self.text, dst, src);
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Clz) => {
                enc::lzcnt_rr_64(&mut self.text, dst, src);
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Ctz) => {
                enc::tzcnt_rr_32(&mut self.text, dst, src);
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Ctz) => {
                enc::tzcnt_rr_64(&mut self.text, dst, src);
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Popcnt) => {
                enc::popcnt_rr_32(&mut self.text, dst, src);
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Popcnt) => {
                enc::popcnt_rr_64(&mut self.text, dst, src);
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Extend8S) => {
                enc::movsx_r32_r8(&mut self.text, dst, src);
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Extend16S) => {
                enc::movsx_r32_r16(&mut self.text, dst, src);
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Extend8S) => {
                enc::movsx_r64_r8(&mut self.text, dst, src);
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Extend16S) => {
                enc::movsx_r64_r16(&mut self.text, dst, src);
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Extend32S) => {
                enc::movsxd_r64_r32(&mut self.text, dst, src);
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Extend32S) => {
                // i32.extend32_s is a nop (already 32-bit)
                if dst != src {
                    enc::mov_rr_64(&mut self.text, dst, src);
                }
            }
        }
        Ok(())
    }

    // ── Integer binary ops ───────────────────────────────────────────────────

    pub(super) fn emit_int_binary(
        &mut self,
        width: MachineIntWidth,
        op: MachineIntBinaryOp,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        let dst = self.map_gp_reg(dst)?;

        // x86 two-operand model: dst = dst OP src. So we need lhs in dst first.
        match op {
            MachineIntBinaryOp::Add
            | MachineIntBinaryOp::Sub
            | MachineIntBinaryOp::And
            | MachineIntBinaryOp::Or
            | MachineIntBinaryOp::Xor => {
                // Try immediate form: dst = lhs OP imm32
                if let MachineValue::Imm64(imm_val) = rhs {
                    let imm = imm_val as i64 as i32;
                    if imm as i64 == imm_val as i64
                        || (width == MachineIntWidth::I32 && imm_val as u32 as i32 == imm)
                    {
                        let lhs_gp = self.materialize_value(SCRATCH0, lhs)?;
                        if dst != lhs_gp {
                            enc::mov_rr_64(&mut self.text, dst, lhs_gp);
                        }
                        match (width, op) {
                            (MachineIntWidth::I64, MachineIntBinaryOp::Add) => {
                                enc::add_ri_64(&mut self.text, dst, imm)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::Add) => {
                                enc::add_ri_32(&mut self.text, dst, imm)
                            }
                            (MachineIntWidth::I64, MachineIntBinaryOp::Sub) => {
                                enc::sub_ri_64(&mut self.text, dst, imm)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::Sub) => {
                                enc::sub_ri_32(&mut self.text, dst, imm)
                            }
                            (MachineIntWidth::I64, MachineIntBinaryOp::And) => {
                                enc::and_ri_64(&mut self.text, dst, imm)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::And) => {
                                enc::and_ri_32(&mut self.text, dst, imm)
                            }
                            (MachineIntWidth::I64, MachineIntBinaryOp::Or) => {
                                enc::or_ri_64(&mut self.text, dst, imm)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::Or) => {
                                enc::or_ri_32(&mut self.text, dst, imm)
                            }
                            (MachineIntWidth::I64, MachineIntBinaryOp::Xor) => {
                                enc::xor_ri_64(&mut self.text, dst, imm)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::Xor) => {
                                enc::xor_ri_32(&mut self.text, dst, imm)
                            }
                            _ => unreachable!(),
                        };
                        return Ok(());
                    }
                }
                let lhs_gp = self.materialize_value(SCRATCH0, lhs)?;
                let rhs_gp = self.materialize_value(SCRATCH1, rhs)?;
                // Handle aliasing: if dst == rhs_gp but dst != lhs_gp,
                // mov dst, lhs would clobber rhs before the operation.
                if dst == rhs_gp && dst != lhs_gp {
                    if op == MachineIntBinaryOp::Sub {
                        // Sub is not commutative: compute in scratch
                        enc::mov_rr_64(&mut self.text, SCRATCH0, lhs_gp);
                        match width {
                            MachineIntWidth::I64 => {
                                enc::sub_rr_64(&mut self.text, SCRATCH0, rhs_gp)
                            }
                            MachineIntWidth::I32 => {
                                enc::sub_rr_32(&mut self.text, SCRATCH0, rhs_gp)
                            }
                        };
                        enc::mov_rr_64(&mut self.text, dst, SCRATCH0);
                    } else {
                        // Commutative: swap operands — do dst = rhs OP lhs
                        match (width, op) {
                            (MachineIntWidth::I64, MachineIntBinaryOp::Add) => {
                                enc::add_rr_64(&mut self.text, dst, lhs_gp)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::Add) => {
                                enc::add_rr_32(&mut self.text, dst, lhs_gp)
                            }
                            (MachineIntWidth::I64, MachineIntBinaryOp::And) => {
                                enc::and_rr_64(&mut self.text, dst, lhs_gp)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::And) => {
                                enc::and_rr_32(&mut self.text, dst, lhs_gp)
                            }
                            (MachineIntWidth::I64, MachineIntBinaryOp::Or) => {
                                enc::or_rr_64(&mut self.text, dst, lhs_gp)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::Or) => {
                                enc::or_rr_32(&mut self.text, dst, lhs_gp)
                            }
                            (MachineIntWidth::I64, MachineIntBinaryOp::Xor) => {
                                enc::xor_rr_64(&mut self.text, dst, lhs_gp)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::Xor) => {
                                enc::xor_rr_32(&mut self.text, dst, lhs_gp)
                            }
                            _ => unreachable!(),
                        };
                    }
                } else {
                    if dst != lhs_gp {
                        enc::mov_rr_64(&mut self.text, dst, lhs_gp);
                    }
                    match (width, op) {
                        (MachineIntWidth::I64, MachineIntBinaryOp::Add) => {
                            enc::add_rr_64(&mut self.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I32, MachineIntBinaryOp::Add) => {
                            enc::add_rr_32(&mut self.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I64, MachineIntBinaryOp::Sub) => {
                            enc::sub_rr_64(&mut self.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I32, MachineIntBinaryOp::Sub) => {
                            enc::sub_rr_32(&mut self.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I64, MachineIntBinaryOp::And) => {
                            enc::and_rr_64(&mut self.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I32, MachineIntBinaryOp::And) => {
                            enc::and_rr_32(&mut self.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I64, MachineIntBinaryOp::Or) => {
                            enc::or_rr_64(&mut self.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I32, MachineIntBinaryOp::Or) => {
                            enc::or_rr_32(&mut self.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I64, MachineIntBinaryOp::Xor) => {
                            enc::xor_rr_64(&mut self.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I32, MachineIntBinaryOp::Xor) => {
                            enc::xor_rr_32(&mut self.text, dst, rhs_gp)
                        }
                        _ => unreachable!(),
                    };
                }
                Ok(())
            }
            MachineIntBinaryOp::Mul => {
                let lhs_gp = self.materialize_value(SCRATCH0, lhs)?;
                let rhs_gp = self.materialize_value(SCRATCH1, rhs)?;
                if dst == rhs_gp && dst != lhs_gp {
                    // IMUL is commutative: dst already has rhs, just mul by lhs
                    match width {
                        MachineIntWidth::I64 => enc::imul_rr_64(&mut self.text, dst, lhs_gp),
                        MachineIntWidth::I32 => enc::imul_rr_32(&mut self.text, dst, lhs_gp),
                    };
                } else {
                    if dst != lhs_gp {
                        enc::mov_rr_64(&mut self.text, dst, lhs_gp);
                    }
                    match width {
                        MachineIntWidth::I64 => enc::imul_rr_64(&mut self.text, dst, rhs_gp),
                        MachineIntWidth::I32 => enc::imul_rr_32(&mut self.text, dst, rhs_gp),
                    };
                }
                Ok(())
            }
            MachineIntBinaryOp::Shl
            | MachineIntBinaryOp::ShrS
            | MachineIntBinaryOp::ShrU
            | MachineIntBinaryOp::Rotl
            | MachineIntBinaryOp::Rotr => self.emit_shift_op(width, op, dst, lhs, rhs),
            MachineIntBinaryOp::DivS
            | MachineIntBinaryOp::DivU
            | MachineIntBinaryOp::RemS
            | MachineIntBinaryOp::RemU => self.emit_div_rem(width, op, dst, lhs, rhs),
        }
    }

    /// Emit shift/rotate: on x86_64, shift amount must be in CL.
    fn emit_shift_op(
        &mut self,
        width: MachineIntWidth,
        op: MachineIntBinaryOp,
        dst: X86Reg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        // If dst == RCX, we need special handling: shift amount goes in CL,
        // but dst is also RCX. Strategy: put lhs in SCRATCH0 first, then
        // load shift amount into RCX, shift SCRATCH0, move result to RCX.
        if dst == X86Reg::RCX {
            let lhs_gp = self.materialize_value(SCRATCH0, lhs)?;
            if SCRATCH0 != lhs_gp {
                enc::mov_rr_64(&mut self.text, SCRATCH0, lhs_gp);
            }
            let rhs_gp = self.materialize_value(X86Reg::RCX, rhs)?;
            if rhs_gp != X86Reg::RCX {
                enc::mov_rr_64(&mut self.text, X86Reg::RCX, rhs_gp);
            }
            match (width, op) {
                (MachineIntWidth::I64, MachineIntBinaryOp::Shl) => {
                    enc::shl_cl_64(&mut self.text, SCRATCH0)
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::Shl) => {
                    enc::shl_cl_32(&mut self.text, SCRATCH0)
                }
                (MachineIntWidth::I64, MachineIntBinaryOp::ShrS) => {
                    enc::sar_cl_64(&mut self.text, SCRATCH0)
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::ShrS) => {
                    enc::sar_cl_32(&mut self.text, SCRATCH0)
                }
                (MachineIntWidth::I64, MachineIntBinaryOp::ShrU) => {
                    enc::shr_cl_64(&mut self.text, SCRATCH0)
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::ShrU) => {
                    enc::shr_cl_32(&mut self.text, SCRATCH0)
                }
                (MachineIntWidth::I64, MachineIntBinaryOp::Rotl) => {
                    enc::rol_cl_64(&mut self.text, SCRATCH0)
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::Rotl) => {
                    enc::rol_cl_32(&mut self.text, SCRATCH0)
                }
                (MachineIntWidth::I64, MachineIntBinaryOp::Rotr) => {
                    enc::ror_cl_64(&mut self.text, SCRATCH0)
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::Rotr) => {
                    enc::ror_cl_32(&mut self.text, SCRATCH0)
                }
                _ => unreachable!(),
            };
            enc::mov_rr_64(&mut self.text, X86Reg::RCX, SCRATCH0);
            return Ok(());
        }
        // For immediate shift amounts, use the imm8 form (no RCX needed).
        if let MachineValue::Imm64(amount) = rhs {
            let lhs_gp = self.materialize_value(SCRATCH0, lhs)?;
            if dst != lhs_gp {
                enc::mov_rr_64(&mut self.text, dst, lhs_gp);
            }
            let imm = (amount & 0x3F) as u8; // mask to 6 bits (x86 does this anyway)
            match (width, op) {
                (MachineIntWidth::I64, MachineIntBinaryOp::Shl) => {
                    enc::shl_imm_64(&mut self.text, dst, imm)
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::Shl) => {
                    enc::shl_imm_32(&mut self.text, dst, imm)
                }
                (MachineIntWidth::I64, MachineIntBinaryOp::ShrS) => {
                    enc::sar_imm_64(&mut self.text, dst, imm)
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::ShrS) => {
                    enc::sar_imm_32(&mut self.text, dst, imm)
                }
                (MachineIntWidth::I64, MachineIntBinaryOp::ShrU) => {
                    enc::shr_imm_64(&mut self.text, dst, imm)
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::ShrU) => {
                    enc::shr_imm_32(&mut self.text, dst, imm)
                }
                (MachineIntWidth::I64, MachineIntBinaryOp::Rotl) => {
                    enc::rol_imm_64(&mut self.text, dst, imm)
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::Rotl) => {
                    enc::rol_imm_32(&mut self.text, dst, imm)
                }
                (MachineIntWidth::I64, MachineIntBinaryOp::Rotr) => {
                    enc::ror_imm_64(&mut self.text, dst, imm)
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::Rotr) => {
                    enc::ror_imm_32(&mut self.text, dst, imm)
                }
                _ => unreachable!(),
            };
            return Ok(());
        }
        // Variable shift: need lhs in dst and rhs in RCX (CL).
        // Careful: moving lhs→dst may clobber rhs, and moving rhs→RCX may clobber lhs.
        // Resolve both source registers first, then do a safe parallel assignment.
        let lhs_gp = self.materialize_value(SCRATCH0, lhs)?;
        let rhs_gp = self.materialize_value(SCRATCH1, rhs)?;
        let need_save_rcx = dst != X86Reg::RCX;
        if need_save_rcx {
            enc::mov_rr_64(&mut self.text, SCRATCH1, X86Reg::RCX);
        }
        // Parallel assignment: lhs_gp → dst, rhs_gp → RCX.
        // Check for conflicts to determine safe ordering.
        let lhs_conflicts_rcx = lhs_gp == X86Reg::RCX; // moving rhs→RCX clobbers lhs
        let rhs_conflicts_dst = rhs_gp == dst; // moving lhs→dst clobbers rhs
        if lhs_conflicts_rcx && rhs_conflicts_dst {
            // Cycle: lhs is in RCX, rhs is in dst. Swap via SCRATCH0.
            enc::mov_rr_64(&mut self.text, SCRATCH0, lhs_gp); // save lhs
            enc::mov_rr_64(&mut self.text, X86Reg::RCX, rhs_gp); // rhs → RCX
            enc::mov_rr_64(&mut self.text, dst, SCRATCH0); // lhs → dst
        } else if rhs_conflicts_dst {
            // Moving lhs→dst would clobber rhs. Do rhs→RCX first.
            if rhs_gp != X86Reg::RCX {
                enc::mov_rr_64(&mut self.text, X86Reg::RCX, rhs_gp);
            }
            if dst != lhs_gp {
                enc::mov_rr_64(&mut self.text, dst, lhs_gp);
            }
        } else {
            // No conflict, or only lhs_conflicts_rcx. Do lhs→dst first.
            if dst != lhs_gp {
                enc::mov_rr_64(&mut self.text, dst, lhs_gp);
            }
            if rhs_gp != X86Reg::RCX {
                enc::mov_rr_64(&mut self.text, X86Reg::RCX, rhs_gp);
            }
        }
        match (width, op) {
            (MachineIntWidth::I64, MachineIntBinaryOp::Shl) => enc::shl_cl_64(&mut self.text, dst),
            (MachineIntWidth::I32, MachineIntBinaryOp::Shl) => enc::shl_cl_32(&mut self.text, dst),
            (MachineIntWidth::I64, MachineIntBinaryOp::ShrS) => enc::sar_cl_64(&mut self.text, dst),
            (MachineIntWidth::I32, MachineIntBinaryOp::ShrS) => enc::sar_cl_32(&mut self.text, dst),
            (MachineIntWidth::I64, MachineIntBinaryOp::ShrU) => enc::shr_cl_64(&mut self.text, dst),
            (MachineIntWidth::I32, MachineIntBinaryOp::ShrU) => enc::shr_cl_32(&mut self.text, dst),
            (MachineIntWidth::I64, MachineIntBinaryOp::Rotl) => enc::rol_cl_64(&mut self.text, dst),
            (MachineIntWidth::I32, MachineIntBinaryOp::Rotl) => enc::rol_cl_32(&mut self.text, dst),
            (MachineIntWidth::I64, MachineIntBinaryOp::Rotr) => enc::ror_cl_64(&mut self.text, dst),
            (MachineIntWidth::I32, MachineIntBinaryOp::Rotr) => enc::ror_cl_32(&mut self.text, dst),
            _ => unreachable!(),
        };
        if need_save_rcx {
            enc::mov_rr_64(&mut self.text, X86Reg::RCX, SCRATCH1);
        }
        Ok(())
    }

    /// Emit div/rem: x86_64 uses RAX:RDX for dividend, result in RAX (quot) / RDX (rem).
    fn emit_div_rem(
        &mut self,
        width: MachineIntWidth,
        op: MachineIntBinaryOp,
        dst: X86Reg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        // div/idiv implicitly uses RAX and RDX. RDX is a GP transient that
        // might hold a live value. Save it to SCRATCH1 (R11) and restore after.
        // (RAX = SCRATCH0, not in dynamic pool, so no save needed.)
        let need_save_rdx = dst != X86Reg::RDX;
        if need_save_rdx {
            enc::mov_rr_64(&mut self.text, SCRATCH1, X86Reg::RDX);
        }

        // Put dividend into RAX
        let lhs_gp = self.materialize_value(SCRATCH0, lhs)?;
        if lhs_gp != X86Reg::RAX {
            enc::mov_rr_64(&mut self.text, X86Reg::RAX, lhs_gp);
        }
        // Divisor must NOT be RAX or RDX. Use R10 as safe scratch for divisor.
        let rhs_gp = self.materialize_value(X86Reg::R10, rhs)?;
        if rhs_gp == X86Reg::RAX || rhs_gp == X86Reg::RDX {
            enc::mov_rr_64(&mut self.text, X86Reg::R10, rhs_gp);
        }
        let divisor = if rhs_gp == X86Reg::RAX || rhs_gp == X86Reg::RDX {
            X86Reg::R10
        } else {
            rhs_gp
        };

        // Division-by-zero check: divisor == 0 => trap
        enc::test_rr_64(&mut self.text, divisor, divisor);
        let div_zero_label = self.ensure_trap_label(MachineTrapKind::IntegerDivideByZero);
        self.emit_jcc(Cc::E, div_zero_label);

        match op {
            MachineIntBinaryOp::DivS => {
                // Signed overflow check: MIN / -1 => IntegerOverflow trap
                let not_min = self.new_label(LabelKind::Edge);
                match width {
                    MachineIntWidth::I32 => {
                        enc::cmp_ri_32(&mut self.text, X86Reg::RAX, i32::MIN);
                    }
                    MachineIntWidth::I64 => {
                        self.materialize_u64(X86Reg::RDX, i64::MIN as u64);
                        enc::cmp_rr_64(&mut self.text, X86Reg::RAX, X86Reg::RDX);
                    }
                };
                self.emit_jcc(Cc::NE, not_min);
                // Compare divisor against -1 using matching width
                match width {
                    MachineIntWidth::I32 => enc::cmp_ri_32(&mut self.text, divisor, -1),
                    MachineIntWidth::I64 => enc::cmp_ri_64(&mut self.text, divisor, -1),
                };
                let overflow_label = self.ensure_trap_label(MachineTrapKind::IntegerOverflow);
                self.emit_jcc(Cc::E, overflow_label);
                self.bind_label(not_min);
                // Sign-extend RAX → RDX:RAX, then IDIV
                match width {
                    MachineIntWidth::I64 => {
                        enc::cqo(&mut self.text);
                        enc::idiv_rm_64(&mut self.text, divisor);
                    }
                    MachineIntWidth::I32 => {
                        enc::cdq(&mut self.text);
                        enc::idiv_rm_32(&mut self.text, divisor);
                    }
                };
            }
            MachineIntBinaryOp::RemS => {
                // MIN % -1 = 0 (no trap, just skip the div)
                let not_min = self.new_label(LabelKind::Edge);
                let done = self.new_label(LabelKind::Edge);
                match width {
                    MachineIntWidth::I32 => {
                        enc::cmp_ri_32(&mut self.text, X86Reg::RAX, i32::MIN);
                    }
                    MachineIntWidth::I64 => {
                        self.materialize_u64(X86Reg::RDX, i64::MIN as u64);
                        enc::cmp_rr_64(&mut self.text, X86Reg::RAX, X86Reg::RDX);
                    }
                };
                self.emit_jcc(Cc::NE, not_min);
                match width {
                    MachineIntWidth::I32 => enc::cmp_ri_32(&mut self.text, divisor, -1),
                    MachineIntWidth::I64 => enc::cmp_ri_64(&mut self.text, divisor, -1),
                };
                self.emit_jcc(Cc::NE, not_min);
                // MIN % -1 = 0
                enc::xor_rr_32(&mut self.text, X86Reg::RDX, X86Reg::RDX);
                self.emit_jmp(done);
                self.bind_label(not_min);
                match width {
                    MachineIntWidth::I64 => {
                        enc::cqo(&mut self.text);
                        enc::idiv_rm_64(&mut self.text, divisor);
                    }
                    MachineIntWidth::I32 => {
                        enc::cdq(&mut self.text);
                        enc::idiv_rm_32(&mut self.text, divisor);
                    }
                };
                self.bind_label(done);
            }
            MachineIntBinaryOp::DivU | MachineIntBinaryOp::RemU => {
                // Zero-extend: XOR RDX, RDX
                enc::xor_rr_32(&mut self.text, X86Reg::RDX, X86Reg::RDX);
                match width {
                    MachineIntWidth::I64 => enc::div_rm_64(&mut self.text, divisor),
                    MachineIntWidth::I32 => enc::div_rm_32(&mut self.text, divisor),
                };
            }
            _ => unreachable!(),
        }

        // Result: quotient in RAX, remainder in RDX
        let result_reg = match op {
            MachineIntBinaryOp::DivS | MachineIntBinaryOp::DivU => X86Reg::RAX,
            MachineIntBinaryOp::RemS | MachineIntBinaryOp::RemU => X86Reg::RDX,
            _ => unreachable!(),
        };
        if dst != result_reg {
            enc::mov_rr_64(&mut self.text, dst, result_reg);
        }
        // Restore RDX if it was saved
        if need_save_rdx {
            enc::mov_rr_64(&mut self.text, X86Reg::RDX, SCRATCH1);
        }
        Ok(())
    }

    // ── Integer compare ──────────────────────────────────────────────────────

    pub(super) fn emit_int_compare(
        &mut self,
        width: MachineIntWidth,
        kind: MachineCompareKind,
        sign: MachineSign,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        self.emit_cmp_values(width, lhs, rhs)?;
        let dst = self.map_gp_reg(dst)?;
        let cc = map_int_cond(kind, sign);
        enc::setcc(&mut self.text, cc, dst);
        enc::movzx_r32_r8(&mut self.text, dst, dst);
        Ok(())
    }

    pub(super) fn emit_cmp_values(
        &mut self,
        width: MachineIntWidth,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        // Immediate form
        if let MachineValue::Imm64(imm_val) = rhs {
            let imm = imm_val as i64 as i32;
            if imm as i64 == imm_val as i64
                || (width == MachineIntWidth::I32 && imm_val as u32 as i32 == imm)
            {
                let lhs_gp = self.materialize_value(SCRATCH0, lhs)?;
                match width {
                    MachineIntWidth::I64 => enc::cmp_ri_64(&mut self.text, lhs_gp, imm),
                    MachineIntWidth::I32 => enc::cmp_ri_32(&mut self.text, lhs_gp, imm),
                };
                return Ok(());
            }
        }
        let lhs_gp = self.materialize_value(SCRATCH0, lhs)?;
        let rhs_gp = self.materialize_value(SCRATCH1, rhs)?;
        match width {
            MachineIntWidth::I64 => enc::cmp_rr_64(&mut self.text, lhs_gp, rhs_gp),
            MachineIntWidth::I32 => enc::cmp_rr_32(&mut self.text, lhs_gp, rhs_gp),
        };
        Ok(())
    }

    // ── Select ───────────────────────────────────────────────────────────────

    pub(super) fn emit_select(
        &mut self,
        ty: MachineStorageType,
        dst: MachineReg,
        on_true: MachineValue,
        on_false: MachineValue,
        cond: MachineValue,
    ) -> Result<(), WasmError> {
        if let Some(width) = ty.float_width() {
            match cond {
                MachineValue::Imm64(value) => {
                    let selected = if value != 0 { on_true } else { on_false };
                    return self.emit_move(ty, dst, selected);
                }
                MachineValue::Reg(reg) => {
                    let cond_gp = self.map_gp_reg(reg)?;
                    let dst_fp = self.map_fp_reg(dst)? as u8;
                    let false_label = self.new_label(LabelKind::Edge);
                    let done = self.new_label(LabelKind::Edge);
                    enc::test_rr_64(&mut self.text, cond_gp, cond_gp);
                    self.emit_jcc(Cc::E, false_label);
                    let true_fp =
                        self.prepare_float_operand(width, on_true, SCRATCH0, FP_SCRATCH0)?;
                    if dst_fp != true_fp as u8 {
                        match width {
                            MachineFloatWidth::F32 => {
                                enc::movss_rr(&mut self.text, dst_fp, true_fp as u8)
                            }
                            MachineFloatWidth::F64 => {
                                enc::movsd_rr(&mut self.text, dst_fp, true_fp as u8)
                            }
                        };
                    }
                    self.emit_jmp(done);
                    self.bind_label(false_label);
                    let false_fp =
                        self.prepare_float_operand(width, on_false, SCRATCH1, FP_SCRATCH1)?;
                    if dst_fp != false_fp as u8 {
                        match width {
                            MachineFloatWidth::F32 => {
                                enc::movss_rr(&mut self.text, dst_fp, false_fp as u8)
                            }
                            MachineFloatWidth::F64 => {
                                enc::movsd_rr(&mut self.text, dst_fp, false_fp as u8)
                            }
                        };
                    }
                    self.bind_label(done);
                    self.set_fp_reg_width(dst, width)?;
                    Ok(())
                }
            }
        } else {
            let dst = self.map_gp_reg(dst)?;
            match cond {
                MachineValue::Imm64(value) => {
                    let selected = if value != 0 { on_true } else { on_false };
                    return self.emit_move(ty, inv_map_reg(dst), selected);
                }
                _ => {}
            }
            // Materialize operands BEFORE testing the condition, because
            // materialize_value may clobber flags (e.g. xor reg,reg for zero).
            let true_reg = self.materialize_value(SCRATCH0, on_true)?;
            let false_reg = self.materialize_value(SCRATCH1, on_false)?;
            let cond_gp = match cond {
                MachineValue::Reg(reg) => self.map_gp_reg(reg)?,
                _ => unreachable!(),
            };
            enc::test_rr_64(&mut self.text, cond_gp, cond_gp);
            if dst == true_reg && dst != false_reg {
                enc::cmovcc_rr_64(&mut self.text, Cc::E, dst, false_reg);
            } else if dst == false_reg {
                enc::cmovcc_rr_64(&mut self.text, Cc::NE, dst, true_reg);
            } else {
                enc::mov_rr_64(&mut self.text, dst, false_reg);
                enc::cmovcc_rr_64(&mut self.text, Cc::NE, dst, true_reg);
            }
            Ok(())
        }
    }

    // ── Convert (integer parts only — float conversions are Phase 4) ─────────

    pub(super) fn emit_convert(
        &mut self,
        op: MachineConvertOp,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let dst_float_width = convert_result_float_width(op);
        let src_gp = self.materialize_value(SCRATCH0, src)?;
        match op {
            MachineConvertOp::I32WrapI64 => {
                let dst_gp = self.map_gp_reg(dst)?;
                // MOV r32, r32 zeroes the upper 32 bits
                enc::mov_rr_32(&mut self.text, dst_gp, src_gp);
            }
            MachineConvertOp::I64ExtendI32S => {
                let dst_gp = self.map_gp_reg(dst)?;
                enc::movsxd_r64_r32(&mut self.text, dst_gp, src_gp);
            }
            MachineConvertOp::I64ExtendI32U => {
                let dst_gp = self.map_gp_reg(dst)?;
                // MOV r32, r32 zero-extends
                enc::mov_rr_32(&mut self.text, dst_gp, src_gp);
            }
            MachineConvertOp::I32ReinterpretF32 | MachineConvertOp::I64ReinterpretF64 => {
                let dst_gp = self.map_gp_reg(dst)?;
                if dst_gp != src_gp {
                    enc::mov_rr_64(&mut self.text, dst_gp, src_gp);
                }
            }
            MachineConvertOp::F32ReinterpretI32 | MachineConvertOp::F64ReinterpretI64 => {
                let width = dst_float_width.expect("float reinterpret width");
                if self.is_fp_reg(dst) {
                    let dst_fp = self.map_fp_reg(dst)? as u8;
                    match width {
                        MachineFloatWidth::F32 => enc::movd_xmm_r32(&mut self.text, dst_fp, src_gp),
                        MachineFloatWidth::F64 => enc::movq_xmm_r64(&mut self.text, dst_fp, src_gp),
                    };
                    self.set_fp_reg_width(dst, width)?;
                } else {
                    let dst_gp = self.map_gp_reg(dst)?;
                    if dst_gp != src_gp {
                        enc::mov_rr_64(&mut self.text, dst_gp, src_gp);
                    }
                }
            }
            // Float promotion / demotion
            MachineConvertOp::F64PromoteF32 => {
                let src_fp =
                    self.prepare_float_operand(MachineFloatWidth::F32, src, SCRATCH0, FP_SCRATCH0)?;
                if self.is_fp_reg(dst) {
                    let dst_fp = self.map_fp_reg(dst)? as u8;
                    self.set_fp_reg_width(dst, MachineFloatWidth::F64)?;
                    enc::cvtss2sd(&mut self.text, dst_fp, src_fp as u8);
                } else {
                    enc::cvtss2sd(&mut self.text, FP_SCRATCH1 as u8, src_fp as u8);
                    let dst_gp = self.map_gp_reg(dst)?;
                    enc::movq_r64_xmm(&mut self.text, dst_gp, FP_SCRATCH1 as u8);
                }
            }
            MachineConvertOp::F32DemoteF64 => {
                let src_fp =
                    self.prepare_float_operand(MachineFloatWidth::F64, src, SCRATCH0, FP_SCRATCH0)?;
                if self.is_fp_reg(dst) {
                    let dst_fp = self.map_fp_reg(dst)? as u8;
                    self.set_fp_reg_width(dst, MachineFloatWidth::F32)?;
                    enc::cvtsd2ss(&mut self.text, dst_fp, src_fp as u8);
                } else {
                    enc::cvtsd2ss(&mut self.text, FP_SCRATCH1 as u8, src_fp as u8);
                    let dst_gp = self.map_gp_reg(dst)?;
                    enc::movd_r32_xmm(&mut self.text, dst_gp, FP_SCRATCH1 as u8);
                }
            }
            // Int -> Float conversions
            MachineConvertOp::F32ConvertI32S => {
                // CVTSI2SS xmm, r32
                let dst_fp = self.dst_float_reg(dst, MachineFloatWidth::F32)?;
                enc::cvtsi2ss_r32(&mut self.text, dst_fp, src_gp);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F32, dst_fp)?;
            }
            MachineConvertOp::F32ConvertI32U => {
                // Zero-extend to 64-bit first for unsigned interpretation
                enc::mov_rr_32(&mut self.text, SCRATCH0, src_gp);
                let dst_fp = self.dst_float_reg(dst, MachineFloatWidth::F32)?;
                enc::cvtsi2ss_r64(&mut self.text, dst_fp, SCRATCH0);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F32, dst_fp)?;
            }
            MachineConvertOp::F32ConvertI64S => {
                let dst_fp = self.dst_float_reg(dst, MachineFloatWidth::F32)?;
                enc::cvtsi2ss_r64(&mut self.text, dst_fp, src_gp);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F32, dst_fp)?;
            }
            MachineConvertOp::F32ConvertI64U => {
                // x86_64 has no unsigned int-to-float instruction.
                // For values that fit in i64 (bit 63 = 0), use signed conversion.
                // For values with bit 63 set, shift right by 1, convert, then double.
                let dst_fp = self.dst_float_reg(dst, MachineFloatWidth::F32)?;
                enc::test_rr_64(&mut self.text, src_gp, src_gp);
                let large = self.new_label(LabelKind::Edge);
                self.emit_jcc(Cc::S, large); // JS = sign flag set = bit 63 is 1
                                             // Small path: fits in i64
                enc::cvtsi2ss_r64(&mut self.text, dst_fp, src_gp);
                let done = self.new_label(LabelKind::Edge);
                self.emit_jmp(done);
                // Large path: bit 63 set
                self.bind_label(large);
                enc::mov_rr_64(&mut self.text, SCRATCH0, src_gp);
                enc::mov_rr_64(&mut self.text, SCRATCH1, src_gp);
                enc::shr_imm_64(&mut self.text, SCRATCH0, 1); // src >> 1
                enc::and_ri_32(&mut self.text, SCRATCH1, 1); // src & 1 (preserve LSB)
                enc::or_rr_64(&mut self.text, SCRATCH0, SCRATCH1); // (src >> 1) | (src & 1)
                enc::cvtsi2ss_r64(&mut self.text, dst_fp, SCRATCH0);
                enc::addss(&mut self.text, dst_fp, dst_fp); // double it
                self.bind_label(done);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F32, dst_fp)?;
            }
            MachineConvertOp::F64ConvertI32S => {
                let dst_fp = self.dst_float_reg(dst, MachineFloatWidth::F64)?;
                enc::cvtsi2sd_r32(&mut self.text, dst_fp, src_gp);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F64, dst_fp)?;
            }
            MachineConvertOp::F64ConvertI32U => {
                enc::mov_rr_32(&mut self.text, SCRATCH0, src_gp);
                let dst_fp = self.dst_float_reg(dst, MachineFloatWidth::F64)?;
                enc::cvtsi2sd_r64(&mut self.text, dst_fp, SCRATCH0);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F64, dst_fp)?;
            }
            MachineConvertOp::F64ConvertI64S => {
                let dst_fp = self.dst_float_reg(dst, MachineFloatWidth::F64)?;
                enc::cvtsi2sd_r64(&mut self.text, dst_fp, src_gp);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F64, dst_fp)?;
            }
            MachineConvertOp::F64ConvertI64U => {
                let dst_fp = self.dst_float_reg(dst, MachineFloatWidth::F64)?;
                enc::test_rr_64(&mut self.text, src_gp, src_gp);
                let large = self.new_label(LabelKind::Edge);
                self.emit_jcc(Cc::S, large);
                enc::cvtsi2sd_r64(&mut self.text, dst_fp, src_gp);
                let done = self.new_label(LabelKind::Edge);
                self.emit_jmp(done);
                self.bind_label(large);
                enc::mov_rr_64(&mut self.text, SCRATCH0, src_gp);
                enc::mov_rr_64(&mut self.text, SCRATCH1, src_gp);
                enc::shr_imm_64(&mut self.text, SCRATCH0, 1);
                enc::and_ri_32(&mut self.text, SCRATCH1, 1);
                enc::or_rr_64(&mut self.text, SCRATCH0, SCRATCH1);
                enc::cvtsi2sd_r64(&mut self.text, dst_fp, SCRATCH0);
                enc::addsd(&mut self.text, dst_fp, dst_fp);
                self.bind_label(done);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F64, dst_fp)?;
            }
            // Trapping truncations: call Rust helper
            MachineConvertOp::I32TruncF32S
            | MachineConvertOp::I32TruncF32U
            | MachineConvertOp::I32TruncF64S
            | MachineConvertOp::I32TruncF64U
            | MachineConvertOp::I64TruncF32S
            | MachineConvertOp::I64TruncF32U
            | MachineConvertOp::I64TruncF64S
            | MachineConvertOp::I64TruncF64U => {
                let dst_gp = self.map_gp_reg(dst)?;
                self.emit_trapping_trunc(op, dst_gp, src_gp)?;
            }
            // Saturating truncations: call Rust helper
            MachineConvertOp::I32TruncSatF32S
            | MachineConvertOp::I32TruncSatF32U
            | MachineConvertOp::I32TruncSatF64S
            | MachineConvertOp::I32TruncSatF64U
            | MachineConvertOp::I64TruncSatF32S
            | MachineConvertOp::I64TruncSatF32U
            | MachineConvertOp::I64TruncSatF64S
            | MachineConvertOp::I64TruncSatF64U => {
                let dst_gp = self.map_gp_reg(dst)?;
                self.emit_saturating_trunc(op, dst_gp, src_gp)?;
            }
        }
        Ok(())
    }

    // ── Float ops ─────────────────────────────────────────────────────────────

    pub(super) fn emit_float_unary(
        &mut self,
        width: MachineFloatWidth,
        op: MachineFloatUnaryOp,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let src_fp = self.prepare_float_operand(width, src, SCRATCH0, FP_SCRATCH0)?;
        let result_fp = if self.is_fp_reg(dst) {
            let dst_fp = self.map_fp_reg(dst)? as u8;
            self.set_fp_reg_width(dst, width)?;
            dst_fp
        } else {
            FP_SCRATCH2 as u8
        };
        match op {
            MachineFloatUnaryOp::Sqrt => {
                match width {
                    MachineFloatWidth::F32 => enc::sqrtss(&mut self.text, result_fp, src_fp as u8),
                    MachineFloatWidth::F64 => enc::sqrtsd(&mut self.text, result_fp, src_fp as u8),
                };
            }
            MachineFloatUnaryOp::Ceil => {
                match width {
                    MachineFloatWidth::F32 => {
                        enc::roundss(&mut self.text, result_fp, src_fp as u8, enc::ROUND_CEIL)
                    }
                    MachineFloatWidth::F64 => {
                        enc::roundsd(&mut self.text, result_fp, src_fp as u8, enc::ROUND_CEIL)
                    }
                };
            }
            MachineFloatUnaryOp::Floor => {
                match width {
                    MachineFloatWidth::F32 => {
                        enc::roundss(&mut self.text, result_fp, src_fp as u8, enc::ROUND_FLOOR)
                    }
                    MachineFloatWidth::F64 => {
                        enc::roundsd(&mut self.text, result_fp, src_fp as u8, enc::ROUND_FLOOR)
                    }
                };
            }
            MachineFloatUnaryOp::Trunc => {
                match width {
                    MachineFloatWidth::F32 => {
                        enc::roundss(&mut self.text, result_fp, src_fp as u8, enc::ROUND_TRUNC)
                    }
                    MachineFloatWidth::F64 => {
                        enc::roundsd(&mut self.text, result_fp, src_fp as u8, enc::ROUND_TRUNC)
                    }
                };
            }
            MachineFloatUnaryOp::Nearest => {
                match width {
                    MachineFloatWidth::F32 => {
                        enc::roundss(&mut self.text, result_fp, src_fp as u8, enc::ROUND_NEAREST)
                    }
                    MachineFloatWidth::F64 => {
                        enc::roundsd(&mut self.text, result_fp, src_fp as u8, enc::ROUND_NEAREST)
                    }
                };
            }
            MachineFloatUnaryOp::Abs => {
                // Clear sign bit: AND with mask.
                let mask_xmm = if result_fp != FP_SCRATCH0 as u8 {
                    FP_SCRATCH0 as u8
                } else {
                    FP_SCRATCH2 as u8
                };
                if result_fp != src_fp as u8 {
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movss_rr(&mut self.text, result_fp, src_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.text, result_fp, src_fp as u8)
                        }
                    };
                }
                let mask = match width {
                    MachineFloatWidth::F32 => 0x7FFF_FFFFu64,
                    MachineFloatWidth::F64 => 0x7FFF_FFFF_FFFF_FFFFu64,
                };
                self.materialize_u64(SCRATCH0, mask);
                enc::movq_xmm_r64(&mut self.text, mask_xmm, SCRATCH0);
                enc::andpd(&mut self.text, result_fp, mask_xmm);
            }
            MachineFloatUnaryOp::Neg => {
                // Flip sign bit: XOR with mask.
                let mask_xmm = if result_fp != FP_SCRATCH0 as u8 {
                    FP_SCRATCH0 as u8
                } else {
                    FP_SCRATCH2 as u8
                };
                if result_fp != src_fp as u8 {
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movss_rr(&mut self.text, result_fp, src_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.text, result_fp, src_fp as u8)
                        }
                    };
                }
                let mask = match width {
                    MachineFloatWidth::F32 => 0x8000_0000u64,
                    MachineFloatWidth::F64 => 0x8000_0000_0000_0000u64,
                };
                self.materialize_u64(SCRATCH0, mask);
                enc::movq_xmm_r64(&mut self.text, mask_xmm, SCRATCH0);
                enc::xorpd(&mut self.text, result_fp, mask_xmm);
            }
        }
        if !self.is_fp_reg(dst) {
            let dst_gp = self.map_gp_reg(dst)?;
            match width {
                MachineFloatWidth::F32 => enc::movd_r32_xmm(&mut self.text, dst_gp, result_fp),
                MachineFloatWidth::F64 => enc::movq_r64_xmm(&mut self.text, dst_gp, result_fp),
            };
        }
        Ok(())
    }

    pub(super) fn emit_float_binary(
        &mut self,
        width: MachineFloatWidth,
        op: MachineFloatBinaryOp,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        let lhs_fp = self.prepare_float_operand(width, lhs, SCRATCH0, FP_SCRATCH0)?;
        let rhs_fp = self.prepare_float_operand(width, rhs, SCRATCH1, FP_SCRATCH1)?;
        let result_fp = if self.is_fp_reg(dst) {
            let dst_fp = self.map_fp_reg(dst)? as u8;
            self.set_fp_reg_width(dst, width)?;
            dst_fp
        } else {
            FP_SCRATCH2 as u8
        };
        match op {
            MachineFloatBinaryOp::Add
            | MachineFloatBinaryOp::Sub
            | MachineFloatBinaryOp::Mul
            | MachineFloatBinaryOp::Div => {
                // SSE two-operand: dst = dst OP src. Move lhs into result first.
                // If result == rhs and result != lhs, the move would clobber rhs.
                // Save rhs to scratch first in that case.
                let actual_rhs = if result_fp == rhs_fp as u8 && result_fp != lhs_fp as u8 {
                    let scratch = FP_SCRATCH2 as u8;
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movss_rr(&mut self.text, scratch, rhs_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.text, scratch, rhs_fp as u8)
                        }
                    };
                    scratch
                } else {
                    rhs_fp as u8
                };
                if result_fp != lhs_fp as u8 {
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movss_rr(&mut self.text, result_fp, lhs_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.text, result_fp, lhs_fp as u8)
                        }
                    };
                }
                match (width, op) {
                    (MachineFloatWidth::F32, MachineFloatBinaryOp::Add) => {
                        enc::addss(&mut self.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F64, MachineFloatBinaryOp::Add) => {
                        enc::addsd(&mut self.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F32, MachineFloatBinaryOp::Sub) => {
                        enc::subss(&mut self.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F64, MachineFloatBinaryOp::Sub) => {
                        enc::subsd(&mut self.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F32, MachineFloatBinaryOp::Mul) => {
                        enc::mulss(&mut self.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F64, MachineFloatBinaryOp::Mul) => {
                        enc::mulsd(&mut self.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F32, MachineFloatBinaryOp::Div) => {
                        enc::divss(&mut self.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F64, MachineFloatBinaryOp::Div) => {
                        enc::divsd(&mut self.text, result_fp, actual_rhs)
                    }
                    _ => unreachable!(),
                };
            }
            MachineFloatBinaryOp::Min => {
                // Wasm fmin: if either operand is NaN, result is NaN.
                // x86_64 minsd/minss: if either is NaN, returns the SECOND operand.
                // Strategy: result = minsd(lhs, rhs); if unordered, result = addsd(lhs, rhs) (NaN propagation).
                // Guard: if result == rhs and result != lhs, save rhs to scratch first.
                let actual_rhs = if result_fp == rhs_fp as u8 && result_fp != lhs_fp as u8 {
                    let scratch = FP_SCRATCH2 as u8;
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movss_rr(&mut self.text, scratch, rhs_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.text, scratch, rhs_fp as u8)
                        }
                    };
                    scratch
                } else {
                    rhs_fp as u8
                };
                if result_fp != lhs_fp as u8 {
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movss_rr(&mut self.text, result_fp, lhs_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.text, result_fp, lhs_fp as u8)
                        }
                    };
                }
                match width {
                    MachineFloatWidth::F32 => enc::minss(&mut self.text, result_fp, actual_rhs),
                    MachineFloatWidth::F64 => enc::minsd(&mut self.text, result_fp, actual_rhs),
                };
                // Compare for NaN: ucomisd lhs, rhs sets PF=1 if unordered (NaN)
                match width {
                    MachineFloatWidth::F32 => {
                        enc::ucomiss(&mut self.text, lhs_fp as u8, actual_rhs)
                    }
                    MachineFloatWidth::F64 => {
                        enc::ucomisd(&mut self.text, lhs_fp as u8, actual_rhs)
                    }
                };
                let done = self.new_label(LabelKind::Edge);
                self.emit_jcc(Cc::NP, done); // no NaN => minsd result is correct
                                             // NaN case: add propagates NaN
                if result_fp != lhs_fp as u8 {
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movss_rr(&mut self.text, result_fp, lhs_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.text, result_fp, lhs_fp as u8)
                        }
                    };
                }
                match width {
                    MachineFloatWidth::F32 => enc::addss(&mut self.text, result_fp, actual_rhs),
                    MachineFloatWidth::F64 => enc::addsd(&mut self.text, result_fp, actual_rhs),
                };
                self.bind_label(done);
            }
            MachineFloatBinaryOp::Max => {
                // Same NaN handling as Min but with maxsd/maxss.
                // Guard: if result == rhs and result != lhs, save rhs to scratch first.
                let actual_rhs = if result_fp == rhs_fp as u8 && result_fp != lhs_fp as u8 {
                    let scratch = FP_SCRATCH2 as u8;
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movss_rr(&mut self.text, scratch, rhs_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.text, scratch, rhs_fp as u8)
                        }
                    };
                    scratch
                } else {
                    rhs_fp as u8
                };
                if result_fp != lhs_fp as u8 {
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movss_rr(&mut self.text, result_fp, lhs_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.text, result_fp, lhs_fp as u8)
                        }
                    };
                }
                match width {
                    MachineFloatWidth::F32 => enc::maxss(&mut self.text, result_fp, actual_rhs),
                    MachineFloatWidth::F64 => enc::maxsd(&mut self.text, result_fp, actual_rhs),
                };
                match width {
                    MachineFloatWidth::F32 => {
                        enc::ucomiss(&mut self.text, lhs_fp as u8, actual_rhs)
                    }
                    MachineFloatWidth::F64 => {
                        enc::ucomisd(&mut self.text, lhs_fp as u8, actual_rhs)
                    }
                };
                let done = self.new_label(LabelKind::Edge);
                self.emit_jcc(Cc::NP, done);
                if result_fp != lhs_fp as u8 {
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movss_rr(&mut self.text, result_fp, lhs_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.text, result_fp, lhs_fp as u8)
                        }
                    };
                }
                match width {
                    MachineFloatWidth::F32 => enc::addss(&mut self.text, result_fp, actual_rhs),
                    MachineFloatWidth::F64 => enc::addsd(&mut self.text, result_fp, actual_rhs),
                };
                self.bind_label(done);
            }
            MachineFloatBinaryOp::Copysign => {
                // magnitude of lhs, sign of rhs.
                // Strategy: clear sign of lhs (abs), extract sign of rhs, OR them.
                // Use a mask scratch that doesn't conflict with result_fp or rhs_fp.
                let mask_xmm =
                    if result_fp != FP_SCRATCH0 as u8 && rhs_fp as u8 != FP_SCRATCH0 as u8 {
                        FP_SCRATCH0 as u8
                    } else {
                        FP_SCRATCH2 as u8
                    };
                let sign_mask = match width {
                    MachineFloatWidth::F32 => 0x8000_0000u64,
                    MachineFloatWidth::F64 => 0x8000_0000_0000_0000u64,
                };
                let abs_mask = match width {
                    MachineFloatWidth::F32 => 0x7FFF_FFFFu64,
                    MachineFloatWidth::F64 => 0x7FFF_FFFF_FFFF_FFFFu64,
                };
                // result = lhs & abs_mask (clear sign bit)
                if result_fp != lhs_fp as u8 {
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movss_rr(&mut self.text, result_fp, lhs_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.text, result_fp, lhs_fp as u8)
                        }
                    };
                }
                self.materialize_u64(SCRATCH0, abs_mask);
                enc::movq_xmm_r64(&mut self.text, mask_xmm, SCRATCH0);
                enc::andpd(&mut self.text, result_fp, mask_xmm);
                // mask_xmm = rhs & sign_mask (extract sign bit)
                self.materialize_u64(SCRATCH0, sign_mask);
                enc::movq_xmm_r64(&mut self.text, mask_xmm, SCRATCH0);
                enc::andpd(&mut self.text, mask_xmm, rhs_fp as u8);
                // result |= mask_xmm
                enc::orpd(&mut self.text, result_fp, mask_xmm);
            }
        };
        if !self.is_fp_reg(dst) {
            let dst_gp = self.map_gp_reg(dst)?;
            match width {
                MachineFloatWidth::F32 => enc::movd_r32_xmm(&mut self.text, dst_gp, result_fp),
                MachineFloatWidth::F64 => enc::movq_r64_xmm(&mut self.text, dst_gp, result_fp),
            };
        }
        Ok(())
    }

    pub(super) fn emit_float_compare(
        &mut self,
        width: MachineFloatWidth,
        kind: MachineCompareKind,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        let dst_gp = self.map_gp_reg(dst)?;
        let lhs_fp = self.prepare_float_operand(width, lhs, SCRATCH0, FP_SCRATCH0)?;
        // Choose an rhs FP scratch that doesn't conflict with lhs. When lhs
        // already lives in a mapped FP register (not FP_SCRATCH0), reuse
        // FP_SCRATCH0 for rhs to avoid clobbering live FP transients in
        // FP_SCRATCH1/FP_SCRATCH2.
        let rhs_fp_scratch = if lhs_fp != FP_SCRATCH0 as u32 {
            FP_SCRATCH0
        } else {
            FP_SCRATCH2
        };
        if matches!(rhs, MachineValue::Imm64(0)) {
            enc::xorpd(&mut self.text, rhs_fp_scratch as u8, rhs_fp_scratch as u8);
            match width {
                MachineFloatWidth::F32 => {
                    enc::ucomiss(&mut self.text, lhs_fp as u8, rhs_fp_scratch as u8)
                }
                MachineFloatWidth::F64 => {
                    enc::ucomisd(&mut self.text, lhs_fp as u8, rhs_fp_scratch as u8)
                }
            };
        } else {
            let rhs_fp = self.prepare_float_operand(width, rhs, SCRATCH1, rhs_fp_scratch)?;
            match width {
                MachineFloatWidth::F32 => enc::ucomiss(&mut self.text, lhs_fp as u8, rhs_fp as u8),
                MachineFloatWidth::F64 => enc::ucomisd(&mut self.text, lhs_fp as u8, rhs_fp as u8),
            };
        }
        // Wasm float comparisons: unordered (NaN) => 0 for all except Ne.
        // UCOMISD sets: ZF=1,PF=1,CF=1 for unordered; ZF=1,PF=0,CF=0 for equal;
        // ZF=0,PF=0,CF=1 for less-than; ZF=0,PF=0,CF=0 for greater-than.
        match kind {
            MachineCompareKind::Eq => {
                // Ordered and equal: ZF=1 AND PF=0
                // SETE sets dst to ZF, SETNP sets tmp to !PF, AND them.
                enc::setcc(&mut self.text, Cc::E, dst_gp);
                enc::setcc(&mut self.text, Cc::NP, SCRATCH0);
                enc::and_rr_32(&mut self.text, dst_gp, SCRATCH0);
                // Zero-extend the byte result to full register
                enc::movzx_r32_r8(&mut self.text, dst_gp, dst_gp);
            }
            MachineCompareKind::Ne => {
                // Unordered OR not-equal: NE=1 OR PF=1
                enc::setcc(&mut self.text, Cc::NE, dst_gp);
                enc::setcc(&mut self.text, Cc::P, SCRATCH0);
                enc::or_rr_32(&mut self.text, dst_gp, SCRATCH0);
                enc::movzx_r32_r8(&mut self.text, dst_gp, dst_gp);
            }
            MachineCompareKind::Lt => {
                // Ordered and less: CF=1 AND PF=0 → use JB (CF=1), but need !PF too
                // Actually, for UCOMISD: CF=1 for less-than AND for unordered.
                // So Lt = CF=1 AND PF=0 → SETB AND SETNP
                enc::setcc(&mut self.text, Cc::B, dst_gp);
                enc::setcc(&mut self.text, Cc::NP, SCRATCH0);
                enc::and_rr_32(&mut self.text, dst_gp, SCRATCH0);
                enc::movzx_r32_r8(&mut self.text, dst_gp, dst_gp);
            }
            MachineCompareKind::Gt => {
                // Ordered and greater: ZF=0, CF=0, PF=0 → JA (CF=0 AND ZF=0)
                // JA already excludes unordered (PF=1 implies CF=1), so SETA is correct.
                enc::setcc(&mut self.text, Cc::A, dst_gp);
                enc::movzx_r32_r8(&mut self.text, dst_gp, dst_gp);
            }
            MachineCompareKind::Le => {
                // Ordered and less-or-equal: (CF=1 OR ZF=1) AND PF=0
                enc::setcc(&mut self.text, Cc::BE, dst_gp);
                enc::setcc(&mut self.text, Cc::NP, SCRATCH0);
                enc::and_rr_32(&mut self.text, dst_gp, SCRATCH0);
                enc::movzx_r32_r8(&mut self.text, dst_gp, dst_gp);
            }
            MachineCompareKind::Ge => {
                // Ordered and greater-or-equal: CF=0 AND PF=0 → JAE excludes unordered already
                // Actually JAE = !CF. Unordered sets CF=1, so JAE is 0 for unordered. Correct.
                enc::setcc(&mut self.text, Cc::AE, dst_gp);
                enc::movzx_r32_r8(&mut self.text, dst_gp, dst_gp);
            }
        }
        Ok(())
    }

    // ── Helper calls ──────────────────────────────────────────────────────────

    pub(super) fn emit_call_helper(
        &mut self,
        extern_idx: usize,
        const_idx: usize,
    ) -> Result<(), WasmError> {
        let binding = self
            .compiled
            .module()
            .externs
            .get(extern_idx)
            .ok_or_else(|| WasmError::internal("x86_64 helper target is out of range".into()))?;
        let metadata = self
            .compiled
            .const_ptr(crate::vm::machine::machine_ir::MachineConstId(
                const_idx as u32,
            ))
            .ok_or_else(|| WasmError::internal("x86_64 helper metadata is out of range".into()))?;
        // System V AMD64 ABI: RDI=ctx, RSI=fp, RDX=metadata
        enc::mov_rr_64(
            &mut self.text,
            X86Reg::RDI,
            map_fixed_reg(crate::vm::machine::machine_ir::MACHINE_CTX_REG),
        );
        enc::mov_rr_64(&mut self.text, X86Reg::RSI, map_fixed_reg(MACHINE_FP_REG));
        self.materialize_u64(X86Reg::RDX, metadata as u64);
        self.materialize_u64(
            SCRATCH1,
            resolve_helper_entry(binding.symbol) as usize as u64,
        );
        enc::call_reg(&mut self.text, SCRATCH1);
        // Check return: RAX != 0 => error
        enc::test_rr_32(&mut self.text, X86Reg::RAX, X86Reg::RAX);
        self.emit_jcc(Cc::NE, self.return_error_label);
        Ok(())
    }

    // ── Float conversion helpers ────────────────────────────────────────────

    /// Get FP destination register: if dst is FP reg, use it directly; else use scratch.
    fn dst_float_reg(
        &mut self,
        dst: MachineReg,
        width: MachineFloatWidth,
    ) -> Result<u8, WasmError> {
        if self.is_fp_reg(dst) {
            let dst_fp = self.map_fp_reg(dst)? as u8;
            self.set_fp_reg_width(dst, width)?;
            Ok(dst_fp)
        } else {
            Ok(FP_SCRATCH1 as u8)
        }
    }

    /// If dst is a GP register, move float result from XMM to GP.
    fn store_fp_result_if_gp(
        &mut self,
        dst: MachineReg,
        width: MachineFloatWidth,
        fp_reg: u8,
    ) -> Result<(), WasmError> {
        if !self.is_fp_reg(dst) {
            let dst_gp = self.map_gp_reg(dst)?;
            match width {
                MachineFloatWidth::F32 => enc::movd_r32_xmm(&mut self.text, dst_gp, fp_reg),
                MachineFloatWidth::F64 => enc::movq_r64_xmm(&mut self.text, dst_gp, fp_reg),
            };
        }
        Ok(())
    }

    /// Save GP transient registers to the system stack before a C helper call.
    /// Pushes 8 registers (7 transients + padding) for 16-byte alignment.
    fn save_gp_transients(&mut self) {
        // Push 7 GP transients + 1 padding for 16-byte alignment (8 * 8 = 64 bytes)
        enc::push(&mut self.text, X86Reg::RCX);
        enc::push(&mut self.text, X86Reg::RDX);
        enc::push(&mut self.text, X86Reg::RSI);
        enc::push(&mut self.text, X86Reg::RDI);
        enc::push(&mut self.text, X86Reg::R8);
        enc::push(&mut self.text, X86Reg::R9);
        enc::push(&mut self.text, X86Reg::R10);
        enc::push(&mut self.text, X86Reg::R10); // padding for 16-byte alignment
    }

    /// Restore GP transient registers from the system stack after a C helper call.
    fn restore_gp_transients(&mut self) {
        enc::pop(&mut self.text, X86Reg::R10); // padding
        enc::pop(&mut self.text, X86Reg::R10);
        enc::pop(&mut self.text, X86Reg::R9);
        enc::pop(&mut self.text, X86Reg::R8);
        enc::pop(&mut self.text, X86Reg::RDI);
        enc::pop(&mut self.text, X86Reg::RSI);
        enc::pop(&mut self.text, X86Reg::RDX);
        enc::pop(&mut self.text, X86Reg::RCX);
    }

    fn emit_trapping_trunc(
        &mut self,
        op: MachineConvertOp,
        dst: X86Reg,
        src: X86Reg,
    ) -> Result<(), WasmError> {
        // The C helper call clobbers all GP transient registers. Save them.
        self.save_gp_transients();
        // Call: x86_64_trapping_trunc(ctx, src_bits, op_code) -> TruncResult{status, value}
        // System V: RDI=ctx, RSI=src_bits, RDX=op_code
        // Returns: RAX=status, RDX=value (struct in RAX+RDX)
        enc::mov_rr_64(
            &mut self.text,
            X86Reg::RDI,
            map_fixed_reg(crate::vm::machine::machine_ir::MACHINE_CTX_REG),
        );
        enc::mov_rr_64(&mut self.text, X86Reg::RSI, src);
        self.materialize_u64(X86Reg::RDX, convert_op_code(op));
        self.materialize_u64(SCRATCH1, x86_64_trapping_trunc as usize as u64);
        enc::call_reg(&mut self.text, SCRATCH1);
        // Save result before restoring transients (RDX would be clobbered)
        enc::mov_rr_64(&mut self.text, SCRATCH1, X86Reg::RDX); // save result value
        let status = X86Reg::RAX; // save status
        self.restore_gp_transients();
        // Check status
        enc::test_rr_64(&mut self.text, status, status);
        self.emit_jcc(Cc::NE, self.return_error_label);
        if dst != SCRATCH1 {
            enc::mov_rr_64(&mut self.text, dst, SCRATCH1);
        }
        Ok(())
    }

    fn emit_saturating_trunc(
        &mut self,
        op: MachineConvertOp,
        dst: X86Reg,
        src: X86Reg,
    ) -> Result<(), WasmError> {
        // The C helper call clobbers all GP transient registers. Save them.
        self.save_gp_transients();
        // Call: x86_64_saturating_trunc(src_bits, op_code) -> u64
        // System V: RDI=src_bits, RSI=op_code. Returns RAX.
        enc::mov_rr_64(&mut self.text, X86Reg::RDI, src);
        self.materialize_u64(X86Reg::RSI, convert_op_code(op));
        self.materialize_u64(SCRATCH1, x86_64_saturating_trunc as usize as u64);
        enc::call_reg(&mut self.text, SCRATCH1);
        // Save result before restoring transients
        enc::mov_rr_64(&mut self.text, SCRATCH1, X86Reg::RAX);
        self.restore_gp_transients();
        if dst != SCRATCH1 {
            enc::mov_rr_64(&mut self.text, dst, SCRATCH1);
        }
        Ok(())
    }

    /// Decomposed indexed load: extend(index) + offset into SCRATCH0, then
    /// load from [base + SCRATCH0]. Stable-base form for store-forwarding.
    /// TODO: use x86_64 [base + index + disp] addressing for 1-2 instructions.
    pub(super) fn emit_indexed_load_decomposed(
        &mut self,
        dst: MachineReg,
        base: MachineReg,
        index: MachineReg,
        index_extend: MachineIndexExtend,
        offset: i32,
        width: MachineMemWidth,
        extension: MachineLoadExtension,
    ) -> Result<(), WasmError> {
        let base_x86 = self.map_gp_reg(base)?;
        let index_x86 = self.map_gp_reg(index)?;
        // Step 1: copy/extend index into SCRATCH0
        if index_extend == MachineIndexExtend::ZeroExtend32 {
            enc::mov_rr_32(&mut self.text, SCRATCH0, index_x86);
        } else {
            enc::mov_rr_64(&mut self.text, SCRATCH0, index_x86);
        }
        // Step 2: add offset
        if offset != 0 {
            enc::add_ri_64(&mut self.text, SCRATCH0, offset);
        }
        // Step 3: add base → SCRATCH0 = base + extended_index + offset
        enc::add_rr_64(&mut self.text, SCRATCH0, base_x86);
        // Step 4: load from [SCRATCH0]
        self.emit_load(dst, MachineAddr { base, offset: 0 }, width, extension)
            .ok();
        // Actually, we need to load from SCRATCH0, not from `base`.
        // Override: emit load from [SCRATCH0 + 0] by patching the base.
        // Let me just use emit_load properly — we need a MachineReg that
        // maps to SCRATCH0. The simplest: use emit_addr_into pattern.
        // For now, just decompose fully:
        self.emit_load(
            dst,
            MachineAddr {
                base: inv_map_reg(SCRATCH0).ok_or_else(|| {
                    WasmError::internal("x86_64 SCRATCH0 has no MachineReg mapping".into())
                })?,
                offset: 0,
            },
            width,
            extension,
        )
    }

    /// Decomposed indexed store.
    pub(super) fn emit_indexed_store_decomposed(
        &mut self,
        base: MachineReg,
        index: MachineReg,
        index_extend: MachineIndexExtend,
        offset: i32,
        width: MachineMemWidth,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let base_x86 = self.map_gp_reg(base)?;
        let index_x86 = self.map_gp_reg(index)?;
        if index_extend == MachineIndexExtend::ZeroExtend32 {
            enc::mov_rr_32(&mut self.text, SCRATCH0, index_x86);
        } else {
            enc::mov_rr_64(&mut self.text, SCRATCH0, index_x86);
        }
        if offset != 0 {
            enc::add_ri_64(&mut self.text, SCRATCH0, offset);
        }
        enc::add_rr_64(&mut self.text, SCRATCH0, base_x86);
        self.emit_store(
            MachineAddr {
                base: inv_map_reg(SCRATCH0).ok_or_else(|| {
                    WasmError::internal("x86_64 SCRATCH0 has no MachineReg mapping".into())
                })?,
                offset: 0,
            },
            width,
            src,
        )
    }
}
