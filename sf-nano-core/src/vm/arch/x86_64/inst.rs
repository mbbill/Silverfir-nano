//! x86_64 backend: instruction lowering methods for X86_64Backend.

use alloc::vec::Vec;

use crate::{
    error::WasmError,
    vm::machine::machine_ir::{
        MachineAddr, MachineCompareKind, MachineConvertOp, MachineFloatBinaryOp,
        MachineFloatUnaryOp, MachineFloatWidth, MachineIndexExtend, MachineInst,
        MachineInstKind, MachineIntBinaryOp,
        MachineIntUnaryOp, MachineIntWidth, MachineLoadExtension, MachineMemWidth,
        MachineReg, MachineSign, MachineStorageType, MachineTrapKind, MachineValue,
    },
};

use super::abi::{C_ARG0, C_ARG1, C_ARG2};
#[cfg(target_os = "windows")]
use super::abi::C_ARG3;
#[cfg(target_os = "windows")]
use super::helpers::x86_64_trapping_trunc_win;

use super::{
    abi::{fp_machine_reg, map_fixed_reg},
    backend::X86_64Backend,
    enc::{self, Cc},
    fusion::{map_float_cond, map_int_cond},
    helpers::{x86_64_saturating_trunc, x86_64_trapping_trunc},
    reg::X86Reg,
};
use crate::vm::arch::common::helpers::{convert_op_code, convert_result_float_width};

use crate::vm::machine::machine_ir::{MACHINE_CTX_REG, MACHINE_FP_REG, MACHINE_MEM0_BASE_REG};
use crate::vm::runtime::helpers::resolve_helper_entry;

impl<'a> X86_64Backend<'a> {
    pub(super) fn lower_inst_dispatch(&mut self, inst: &MachineInst) -> Result<(), WasmError> {
        match &inst.kind {
            MachineInstKind::Move { dst, src, ty } => self.lower_move(*ty, *dst, *src),
            MachineInstKind::FloatConst { width, dst, bits } => {
                self.lower_float_const(*width, *dst, *bits)
            }
            MachineInstKind::Load {
                ty: _,
                dst,
                addr,
                width,
                extension,
            } => self.lower_load(*dst, *addr, *width, *extension),
            MachineInstKind::Store {
                ty: _,
                addr,
                width,
                src,
            } => self.lower_store(*addr, *width, *src),
            MachineInstKind::IntUnary {
                width,
                op,
                dst,
                src,
            } => self.lower_int_unary(*width, *op, *dst, *src),
            MachineInstKind::IntBinary {
                width,
                op,
                dst,
                lhs,
                rhs,
            } => self.lower_int_binary(*width, *op, *dst, *lhs, *rhs),
            MachineInstKind::Int64PairBinary { .. } => Err(WasmError::internal(
                "x86_64 backend received Int64PairBinary; 32-bit legalized MachineIR should not reach x86_64 codegen".into(),
            )),
            MachineInstKind::Int64PairUnary { .. } => Err(WasmError::internal(
                "x86_64 backend received Int64PairUnary; 32-bit legalized MachineIR should not reach x86_64 codegen".into(),
            )),
            MachineInstKind::Int64PairDivRem { .. } => Err(WasmError::internal(
                "x86_64 backend received Int64PairDivRem; 32-bit legalized MachineIR should not reach x86_64 codegen".into(),
            )),
            MachineInstKind::Int64PairShift { .. } => Err(WasmError::internal(
                "x86_64 backend received Int64PairShift; 32-bit legalized MachineIR should not reach x86_64 codegen".into(),
            )),
            MachineInstKind::IntCompare {
                width,
                kind,
                sign,
                dst,
                lhs,
                rhs,
            } => self.lower_int_compare(*width, *kind, *sign, *dst, *lhs, *rhs),
            MachineInstKind::Select {
                ty,
                dst,
                on_true,
                on_false,
                cond,
                ..
            } => self.lower_select(*ty, *dst, *on_true, *on_false, *cond),
            MachineInstKind::TrapIf { kind, cond } => {
                let trap_label = self.core.ensure_trap_label(*kind);
                self.lower_branch_if(cond, trap_label)
            }
            MachineInstKind::CallHelper(call) => {
                self.lower_call_helper(call.target.0 as usize, call.metadata.0 as usize)
            }
            MachineInstKind::FloatUnary {
                width,
                op,
                dst,
                src,
            } => self.lower_float_unary(*width, *op, *dst, *src),
            MachineInstKind::FloatBinary {
                width,
                op,
                dst,
                lhs,
                rhs,
            } => self.lower_float_binary(*width, *op, *dst, *lhs, *rhs),
            MachineInstKind::FloatCompare {
                width,
                kind,
                dst,
                lhs,
                rhs,
            } => self.lower_float_compare(*width, *kind, *dst, *lhs, *rhs),
            MachineInstKind::Convert { op, dst, src } => self.lower_convert(*op, *dst, *src),
            MachineInstKind::ConvertI64PairToFloat { .. } => Err(WasmError::internal(
                "x86_64 backend received ConvertI64PairToFloat; 32-bit legalized MachineIR should not reach x86_64 codegen".into(),
            )),
            MachineInstKind::Int64PairCompare { .. } => Err(WasmError::internal(
                "x86_64 backend received Int64PairCompare; 32-bit legalized MachineIR should not reach x86_64 codegen".into(),
            )),
            MachineInstKind::ConvertFloatToI64Pair { .. } => Err(WasmError::internal(
                "x86_64 backend received ConvertFloatToI64Pair; 32-bit legalized MachineIR should not reach x86_64 codegen".into(),
            )),
            MachineInstKind::ReinterpretF64ToI64Pair { .. } => Err(WasmError::internal(
                "x86_64 backend received ReinterpretF64ToI64Pair; 32-bit legalized MachineIR should not reach x86_64 codegen".into(),
            )),
            MachineInstKind::ReinterpretI64PairToF64 { .. } => Err(WasmError::internal(
                "x86_64 backend received ReinterpretI64PairToF64; 32-bit legalized MachineIR should not reach x86_64 codegen".into(),
            )),
            MachineInstKind::IndexedLoad {
                dst,
                base,
                index,
                index_extend,
                offset,
                width,
                extension,
            } => {
                self.lower_indexed_load_decomposed(*dst, *base, *index, *index_extend, *offset, *width, *extension)
            }
            MachineInstKind::IndexedStore {
                base,
                index,
                index_extend,
                offset,
                width,
                src,
            } => {
                self.lower_indexed_store_decomposed(*base, *index, *index_extend, *offset, *width, *src)
            }
        }
    }
    // ── Move / const ────────────────────────────────────────────────────────

    pub(super) fn lower_move(
        &mut self,
        ty: MachineStorageType,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        if let Some(width) = ty.float_width() {
            let dst_fp = self.map_fp_reg(dst)? as u8;
            match src {
                MachineValue::Reg(src_reg) if self.core.is_fp_reg(src_reg) => {
                    let src_fp = self.map_fp_reg(src_reg)? as u8;
                    let src_width = self.core.fp_reg_width(src_reg)?;
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
                            MachineFloatWidth::F32 => enc::movss_rr(&mut self.core.text, dst_fp, src_fp),
                            MachineFloatWidth::F64 => enc::movsd_rr(&mut self.core.text, dst_fp, src_fp),
                        };
                    }
                    self.core.set_fp_reg_width(dst, width)?;
                    Ok(())
                }
                MachineValue::Reg(src_reg) => {
                    let src_gp = self.map_gp_reg(src_reg)?;
                    match width {
                        MachineFloatWidth::F32 => enc::movd_xmm_r32(&mut self.core.text, dst_fp, src_gp),
                        MachineFloatWidth::F64 => enc::movq_xmm_r64(&mut self.core.text, dst_fp, src_gp),
                    };
                    self.core.set_fp_reg_width(dst, width)?;
                    Ok(())
                }
                MachineValue::Imm64(value) => {
                    self.materialize_u64(self.gp_scratch.reg(0), value);
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movd_xmm_r32(&mut self.core.text, dst_fp, self.gp_scratch.reg(0))
                        }
                        MachineFloatWidth::F64 => {
                            enc::movq_xmm_r64(&mut self.core.text, dst_fp, self.gp_scratch.reg(0))
                        }
                    };
                    self.core.set_fp_reg_width(dst, width)?;
                    Ok(())
                }
            }
        } else {
            let dst_gp = self.map_gp_reg(dst)?;
            match src {
                MachineValue::Reg(src_reg) if self.core.is_fp_reg(src_reg) => {
                    let src_fp = self.map_fp_reg(src_reg)? as u8;
                    match self.core.fp_reg_width(src_reg)? {
                        MachineFloatWidth::F32 => enc::movd_r32_xmm(&mut self.core.text, dst_gp, src_fp),
                        MachineFloatWidth::F64 => enc::movq_r64_xmm(&mut self.core.text, dst_gp, src_fp),
                    };
                    Ok(())
                }
                MachineValue::Reg(src_reg) => {
                    let src_gp = self.map_gp_reg(src_reg)?;
                    if dst_gp != src_gp {
                        enc::mov_rr_64(&mut self.core.text, dst_gp, src_gp);
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

    pub(super) fn lower_float_const(
        &mut self,
        width: MachineFloatWidth,
        dst: MachineReg,
        bits: u64,
    ) -> Result<(), WasmError> {
        if !self.core.is_fp_reg(dst) {
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
                MachineFloatWidth::F32 => enc::xorps(&mut self.core.text, dst_fp, dst_fp),
                MachineFloatWidth::F64 => enc::xorpd(&mut self.core.text, dst_fp, dst_fp),
            };
        } else {
            self.materialize_u64(self.gp_scratch.reg(0), imm);
            match width {
                MachineFloatWidth::F32 => enc::movd_xmm_r32(&mut self.core.text, dst_fp, self.gp_scratch.reg(0)),
                MachineFloatWidth::F64 => enc::movq_xmm_r64(&mut self.core.text, dst_fp, self.gp_scratch.reg(0)),
            };
        }
        self.core.set_fp_reg_width(dst, width)?;
        Ok(())
    }

    // ── LEA / Load / Store ───────────────────────────────────────────────────

    pub(super) fn lower_lea(&mut self, dst: MachineReg, addr: MachineAddr) -> Result<(), WasmError> {
        let dst_gp = self.map_gp_reg(dst)?;
        let base = self.map_gp_reg(addr.base)?;
        if addr.offset == 0 {
            if dst_gp != base {
                enc::mov_rr_64(&mut self.core.text, dst_gp, base);
            }
        } else {
            enc::lea_64(&mut self.core.text, dst_gp, base, addr.offset);
        }
        Ok(())
    }

    pub(super) fn lower_load(
        &mut self,
        dst: MachineReg,
        addr: MachineAddr,
        width: MachineMemWidth,
        extension: MachineLoadExtension,
    ) -> Result<(), WasmError> {
        let base = self.map_gp_reg(addr.base)?;
        self.lower_load_from(dst, base, addr.offset, width, extension)
    }

    /// Load from a physical base register + displacement.
    fn lower_load_from(
        &mut self,
        dst: MachineReg,
        base: X86Reg,
        disp: i32,
        width: MachineMemWidth,
        extension: MachineLoadExtension,
    ) -> Result<(), WasmError> {
        // FP register destination
        if self.core.is_fp_reg(dst) {
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
                MachineFloatWidth::F32 => enc::movss_load(&mut self.core.text, dst_fp, base, disp),
                MachineFloatWidth::F64 => enc::movsd_load(&mut self.core.text, dst_fp, base, disp),
            };
            self.core.set_fp_reg_width(dst, tracked_width)?;
            return Ok(());
        }

        // GP register destination
        let dst_gp = self.map_gp_reg(dst)?;
        match (width, extension) {
            (MachineMemWidth::U8, MachineLoadExtension::None)
            | (MachineMemWidth::U8, MachineLoadExtension::ZeroExtend) => {
                enc::load_u8(&mut self.core.text, dst_gp, base, disp);
            }
            (MachineMemWidth::U8, MachineLoadExtension::SignExtend) => {
                enc::load_s8_64(&mut self.core.text, dst_gp, base, disp);
            }
            (MachineMemWidth::U16, MachineLoadExtension::None)
            | (MachineMemWidth::U16, MachineLoadExtension::ZeroExtend) => {
                enc::load_u16(&mut self.core.text, dst_gp, base, disp);
            }
            (MachineMemWidth::U16, MachineLoadExtension::SignExtend) => {
                enc::load_s16_64(&mut self.core.text, dst_gp, base, disp);
            }
            (MachineMemWidth::U32, MachineLoadExtension::None)
            | (MachineMemWidth::U32, MachineLoadExtension::ZeroExtend) => {
                enc::load_32(&mut self.core.text, dst_gp, base, disp);
            }
            (MachineMemWidth::U32, MachineLoadExtension::SignExtend) => {
                enc::load_s32_64(&mut self.core.text, dst_gp, base, disp);
            }
            (MachineMemWidth::U64, MachineLoadExtension::None)
            | (MachineMemWidth::U64, MachineLoadExtension::ZeroExtend) => {
                enc::load_64(&mut self.core.text, dst_gp, base, disp);
            }
            (MachineMemWidth::U64, MachineLoadExtension::SignExtend) => {
                return Err(WasmError::invalid(
                    "x86_64 MachineIR backend does not support sign-extending 64-bit loads".into(),
                ))
            }
        };
        Ok(())
    }

    pub(super) fn lower_store(
        &mut self,
        addr: MachineAddr,
        width: MachineMemWidth,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let base = self.map_gp_reg(addr.base)?;
        self.lower_store_to(base, addr.offset, width, src)
    }

    /// Store to a physical base register + displacement.
    fn lower_store_to(
        &mut self,
        base: X86Reg,
        disp: i32,
        width: MachineMemWidth,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        // FP register source
        if let MachineValue::Reg(src_reg) = src {
            if self.core.is_fp_reg(src_reg) {
                let src_fp = self.map_fp_reg(src_reg)? as u8;
                match width {
                    MachineMemWidth::U32 => enc::movss_store(&mut self.core.text, base, disp, src_fp),
                    MachineMemWidth::U64 => enc::movsd_store(&mut self.core.text, base, disp, src_fp),
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
            enc::store_imm32_64(&mut self.core.text, base, disp, 0);
            return Ok(());
        }

        let src_gp = self.materialize_value(self.gp_scratch.reg(0), src)?;
        match width {
            MachineMemWidth::U8 => enc::store_8(&mut self.core.text, base, disp, src_gp),
            MachineMemWidth::U16 => enc::store_16(&mut self.core.text, base, disp, src_gp),
            MachineMemWidth::U32 => enc::store_32(&mut self.core.text, base, disp, src_gp),
            MachineMemWidth::U64 => enc::store_64(&mut self.core.text, base, disp, src_gp),
        };
        Ok(())
    }

    // ── Integer unary ops ────────────────────────────────────────────────────

    pub(super) fn lower_int_unary(
        &mut self,
        width: MachineIntWidth,
        op: MachineIntUnaryOp,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let dst = self.map_gp_reg(dst)?;
        let src = self.materialize_value(self.gp_scratch.reg(0), src)?;
        match (width, op) {
            (MachineIntWidth::I32, MachineIntUnaryOp::Eqz) => {
                enc::test_rr_32(&mut self.core.text, src, src);
                enc::setcc(&mut self.core.text, Cc::E, dst);
                // Zero-extend the byte result to full register
                enc::movzx_r32_r8(&mut self.core.text, dst, dst);
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Eqz) => {
                enc::test_rr_64(&mut self.core.text, src, src);
                enc::setcc(&mut self.core.text, Cc::E, dst);
                enc::movzx_r32_r8(&mut self.core.text, dst, dst);
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Clz) => {
                enc::lzcnt_rr_32(&mut self.core.text, dst, src);
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Clz) => {
                enc::lzcnt_rr_64(&mut self.core.text, dst, src);
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Ctz) => {
                enc::tzcnt_rr_32(&mut self.core.text, dst, src);
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Ctz) => {
                enc::tzcnt_rr_64(&mut self.core.text, dst, src);
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Popcnt) => {
                enc::popcnt_rr_32(&mut self.core.text, dst, src);
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Popcnt) => {
                enc::popcnt_rr_64(&mut self.core.text, dst, src);
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Extend8S) => {
                enc::movsx_r32_r8(&mut self.core.text, dst, src);
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Extend16S) => {
                enc::movsx_r32_r16(&mut self.core.text, dst, src);
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Extend8S) => {
                enc::movsx_r64_r8(&mut self.core.text, dst, src);
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Extend16S) => {
                enc::movsx_r64_r16(&mut self.core.text, dst, src);
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Extend32S) => {
                enc::movsxd_r64_r32(&mut self.core.text, dst, src);
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Extend32S) => {
                // i32.extend32_s is a nop (already 32-bit)
                if dst != src {
                    enc::mov_rr_64(&mut self.core.text, dst, src);
                }
            }
        }
        Ok(())
    }

    // ── Integer binary ops ───────────────────────────────────────────────────

    pub(super) fn lower_int_binary(
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
                        let lhs_gp = self.materialize_value(self.gp_scratch.reg(0), lhs)?;
                        if dst != lhs_gp {
                            enc::mov_rr_64(&mut self.core.text, dst, lhs_gp);
                        }
                        match (width, op) {
                            (MachineIntWidth::I64, MachineIntBinaryOp::Add) => {
                                enc::add_ri_64(&mut self.core.text, dst, imm)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::Add) => {
                                enc::add_ri_32(&mut self.core.text, dst, imm)
                            }
                            (MachineIntWidth::I64, MachineIntBinaryOp::Sub) => {
                                enc::sub_ri_64(&mut self.core.text, dst, imm)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::Sub) => {
                                enc::sub_ri_32(&mut self.core.text, dst, imm)
                            }
                            (MachineIntWidth::I64, MachineIntBinaryOp::And) => {
                                enc::and_ri_64(&mut self.core.text, dst, imm)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::And) => {
                                enc::and_ri_32(&mut self.core.text, dst, imm)
                            }
                            (MachineIntWidth::I64, MachineIntBinaryOp::Or) => {
                                enc::or_ri_64(&mut self.core.text, dst, imm)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::Or) => {
                                enc::or_ri_32(&mut self.core.text, dst, imm)
                            }
                            (MachineIntWidth::I64, MachineIntBinaryOp::Xor) => {
                                enc::xor_ri_64(&mut self.core.text, dst, imm)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::Xor) => {
                                enc::xor_ri_32(&mut self.core.text, dst, imm)
                            }
                            _ => unreachable!(),
                        };
                        return Ok(());
                    }
                }
                let lhs_gp = self.materialize_value(self.gp_scratch.reg(0), lhs)?;
                let rhs_gp = self.materialize_value(self.gp_scratch.reg(1), rhs)?;
                // Handle aliasing: if dst == rhs_gp but dst != lhs_gp,
                // mov dst, lhs would clobber rhs before the operation.
                if dst == rhs_gp && dst != lhs_gp {
                    if op == MachineIntBinaryOp::Sub {
                        // Sub is not commutative: compute in scratch
                        enc::mov_rr_64(&mut self.core.text, self.gp_scratch.reg(0), lhs_gp);
                        match width {
                            MachineIntWidth::I64 => {
                                enc::sub_rr_64(&mut self.core.text, self.gp_scratch.reg(0), rhs_gp)
                            }
                            MachineIntWidth::I32 => {
                                enc::sub_rr_32(&mut self.core.text, self.gp_scratch.reg(0), rhs_gp)
                            }
                        };
                        enc::mov_rr_64(&mut self.core.text, dst, self.gp_scratch.reg(0));
                    } else {
                        // Commutative: swap operands — do dst = rhs OP lhs
                        match (width, op) {
                            (MachineIntWidth::I64, MachineIntBinaryOp::Add) => {
                                enc::add_rr_64(&mut self.core.text, dst, lhs_gp)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::Add) => {
                                enc::add_rr_32(&mut self.core.text, dst, lhs_gp)
                            }
                            (MachineIntWidth::I64, MachineIntBinaryOp::And) => {
                                enc::and_rr_64(&mut self.core.text, dst, lhs_gp)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::And) => {
                                enc::and_rr_32(&mut self.core.text, dst, lhs_gp)
                            }
                            (MachineIntWidth::I64, MachineIntBinaryOp::Or) => {
                                enc::or_rr_64(&mut self.core.text, dst, lhs_gp)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::Or) => {
                                enc::or_rr_32(&mut self.core.text, dst, lhs_gp)
                            }
                            (MachineIntWidth::I64, MachineIntBinaryOp::Xor) => {
                                enc::xor_rr_64(&mut self.core.text, dst, lhs_gp)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::Xor) => {
                                enc::xor_rr_32(&mut self.core.text, dst, lhs_gp)
                            }
                            _ => unreachable!(),
                        };
                    }
                } else {
                    if dst != lhs_gp {
                        enc::mov_rr_64(&mut self.core.text, dst, lhs_gp);
                    }
                    match (width, op) {
                        (MachineIntWidth::I64, MachineIntBinaryOp::Add) => {
                            enc::add_rr_64(&mut self.core.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I32, MachineIntBinaryOp::Add) => {
                            enc::add_rr_32(&mut self.core.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I64, MachineIntBinaryOp::Sub) => {
                            enc::sub_rr_64(&mut self.core.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I32, MachineIntBinaryOp::Sub) => {
                            enc::sub_rr_32(&mut self.core.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I64, MachineIntBinaryOp::And) => {
                            enc::and_rr_64(&mut self.core.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I32, MachineIntBinaryOp::And) => {
                            enc::and_rr_32(&mut self.core.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I64, MachineIntBinaryOp::Or) => {
                            enc::or_rr_64(&mut self.core.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I32, MachineIntBinaryOp::Or) => {
                            enc::or_rr_32(&mut self.core.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I64, MachineIntBinaryOp::Xor) => {
                            enc::xor_rr_64(&mut self.core.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I32, MachineIntBinaryOp::Xor) => {
                            enc::xor_rr_32(&mut self.core.text, dst, rhs_gp)
                        }
                        _ => unreachable!(),
                    };
                }
                Ok(())
            }
            MachineIntBinaryOp::Mul => {
                let lhs_gp = self.materialize_value(self.gp_scratch.reg(0), lhs)?;
                let rhs_gp = self.materialize_value(self.gp_scratch.reg(1), rhs)?;
                if dst == rhs_gp && dst != lhs_gp {
                    // IMUL is commutative: dst already has rhs, just mul by lhs
                    match width {
                        MachineIntWidth::I64 => enc::imul_rr_64(&mut self.core.text, dst, lhs_gp),
                        MachineIntWidth::I32 => enc::imul_rr_32(&mut self.core.text, dst, lhs_gp),
                    };
                } else {
                    if dst != lhs_gp {
                        enc::mov_rr_64(&mut self.core.text, dst, lhs_gp);
                    }
                    match width {
                        MachineIntWidth::I64 => enc::imul_rr_64(&mut self.core.text, dst, rhs_gp),
                        MachineIntWidth::I32 => enc::imul_rr_32(&mut self.core.text, dst, rhs_gp),
                    };
                }
                Ok(())
            }
            MachineIntBinaryOp::Shl
            | MachineIntBinaryOp::ShrS
            | MachineIntBinaryOp::ShrU
            | MachineIntBinaryOp::Rotl
            | MachineIntBinaryOp::Rotr => self.lower_shift_op(width, op, dst, lhs, rhs),
            MachineIntBinaryOp::DivS
            | MachineIntBinaryOp::DivU
            | MachineIntBinaryOp::RemS
            | MachineIntBinaryOp::RemU => self.lower_div_rem(width, op, dst, lhs, rhs),
        }
    }

    /// Emit shift/rotate: on x86_64, shift amount must be in CL.
    fn lower_shift_op(
        &mut self,
        width: MachineIntWidth,
        op: MachineIntBinaryOp,
        dst: X86Reg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        // If dst == RCX, we need special handling: shift amount goes in CL,
        // but dst is also RCX. Strategy: put lhs in self.gp_scratch.reg(0) first, then
        // load shift amount into RCX, shift self.gp_scratch.reg(0), move result to RCX.
        if dst == X86Reg::RCX {
            let lhs_gp = self.materialize_value(self.gp_scratch.reg(0), lhs)?;
            if self.gp_scratch.reg(0) != lhs_gp {
                enc::mov_rr_64(&mut self.core.text, self.gp_scratch.reg(0), lhs_gp);
            }
            let rhs_gp = self.materialize_value(X86Reg::RCX, rhs)?;
            if rhs_gp != X86Reg::RCX {
                enc::mov_rr_64(&mut self.core.text, X86Reg::RCX, rhs_gp);
            }
            match (width, op) {
                (MachineIntWidth::I64, MachineIntBinaryOp::Shl) => {
                    enc::shl_cl_64(&mut self.core.text, self.gp_scratch.reg(0))
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::Shl) => {
                    enc::shl_cl_32(&mut self.core.text, self.gp_scratch.reg(0))
                }
                (MachineIntWidth::I64, MachineIntBinaryOp::ShrS) => {
                    enc::sar_cl_64(&mut self.core.text, self.gp_scratch.reg(0))
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::ShrS) => {
                    enc::sar_cl_32(&mut self.core.text, self.gp_scratch.reg(0))
                }
                (MachineIntWidth::I64, MachineIntBinaryOp::ShrU) => {
                    enc::shr_cl_64(&mut self.core.text, self.gp_scratch.reg(0))
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::ShrU) => {
                    enc::shr_cl_32(&mut self.core.text, self.gp_scratch.reg(0))
                }
                (MachineIntWidth::I64, MachineIntBinaryOp::Rotl) => {
                    enc::rol_cl_64(&mut self.core.text, self.gp_scratch.reg(0))
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::Rotl) => {
                    enc::rol_cl_32(&mut self.core.text, self.gp_scratch.reg(0))
                }
                (MachineIntWidth::I64, MachineIntBinaryOp::Rotr) => {
                    enc::ror_cl_64(&mut self.core.text, self.gp_scratch.reg(0))
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::Rotr) => {
                    enc::ror_cl_32(&mut self.core.text, self.gp_scratch.reg(0))
                }
                _ => unreachable!(),
            };
            enc::mov_rr_64(&mut self.core.text, X86Reg::RCX, self.gp_scratch.reg(0));
            return Ok(());
        }
        // For immediate shift amounts, use the imm8 form (no RCX needed).
        if let MachineValue::Imm64(amount) = rhs {
            let lhs_gp = self.materialize_value(self.gp_scratch.reg(0), lhs)?;
            if dst != lhs_gp {
                enc::mov_rr_64(&mut self.core.text, dst, lhs_gp);
            }
            let imm = (amount & 0x3F) as u8; // mask to 6 bits (x86 does this anyway)
            match (width, op) {
                (MachineIntWidth::I64, MachineIntBinaryOp::Shl) => {
                    enc::shl_imm_64(&mut self.core.text, dst, imm)
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::Shl) => {
                    enc::shl_imm_32(&mut self.core.text, dst, imm)
                }
                (MachineIntWidth::I64, MachineIntBinaryOp::ShrS) => {
                    enc::sar_imm_64(&mut self.core.text, dst, imm)
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::ShrS) => {
                    enc::sar_imm_32(&mut self.core.text, dst, imm)
                }
                (MachineIntWidth::I64, MachineIntBinaryOp::ShrU) => {
                    enc::shr_imm_64(&mut self.core.text, dst, imm)
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::ShrU) => {
                    enc::shr_imm_32(&mut self.core.text, dst, imm)
                }
                (MachineIntWidth::I64, MachineIntBinaryOp::Rotl) => {
                    enc::rol_imm_64(&mut self.core.text, dst, imm)
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::Rotl) => {
                    enc::rol_imm_32(&mut self.core.text, dst, imm)
                }
                (MachineIntWidth::I64, MachineIntBinaryOp::Rotr) => {
                    enc::ror_imm_64(&mut self.core.text, dst, imm)
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::Rotr) => {
                    enc::ror_imm_32(&mut self.core.text, dst, imm)
                }
                _ => unreachable!(),
            };
            return Ok(());
        }
        // Variable shift: need lhs in dst and rhs in RCX (CL).
        // Careful: moving lhs→dst may clobber rhs, and moving rhs→RCX may clobber lhs.
        // Resolve both source registers first, then do a safe parallel assignment.
        let lhs_gp = self.materialize_value(self.gp_scratch.reg(0), lhs)?;
        let rhs_gp = self.materialize_value(self.gp_scratch.reg(1), rhs)?;
        let need_save_rcx = dst != X86Reg::RCX;
        if need_save_rcx {
            enc::mov_rr_64(&mut self.core.text, self.gp_scratch.reg(1), X86Reg::RCX);
        }
        // Parallel assignment: lhs_gp → dst, rhs_gp → RCX.
        // Check for conflicts to determine safe ordering.
        let lhs_conflicts_rcx = lhs_gp == X86Reg::RCX; // moving rhs→RCX clobbers lhs
        let rhs_conflicts_dst = rhs_gp == dst; // moving lhs→dst clobbers rhs
        if lhs_conflicts_rcx && rhs_conflicts_dst {
            // Cycle: lhs is in RCX, rhs is in dst. Swap via self.gp_scratch.reg(0).
            enc::mov_rr_64(&mut self.core.text, self.gp_scratch.reg(0), lhs_gp); // save lhs
            enc::mov_rr_64(&mut self.core.text, X86Reg::RCX, rhs_gp); // rhs → RCX
            enc::mov_rr_64(&mut self.core.text, dst, self.gp_scratch.reg(0)); // lhs → dst
        } else if rhs_conflicts_dst {
            // Moving lhs→dst would clobber rhs. Do rhs→RCX first.
            if rhs_gp != X86Reg::RCX {
                enc::mov_rr_64(&mut self.core.text, X86Reg::RCX, rhs_gp);
            }
            if dst != lhs_gp {
                enc::mov_rr_64(&mut self.core.text, dst, lhs_gp);
            }
        } else {
            // No conflict, or only lhs_conflicts_rcx. Do lhs→dst first.
            if dst != lhs_gp {
                enc::mov_rr_64(&mut self.core.text, dst, lhs_gp);
            }
            if rhs_gp != X86Reg::RCX {
                enc::mov_rr_64(&mut self.core.text, X86Reg::RCX, rhs_gp);
            }
        }
        match (width, op) {
            (MachineIntWidth::I64, MachineIntBinaryOp::Shl) => enc::shl_cl_64(&mut self.core.text, dst),
            (MachineIntWidth::I32, MachineIntBinaryOp::Shl) => enc::shl_cl_32(&mut self.core.text, dst),
            (MachineIntWidth::I64, MachineIntBinaryOp::ShrS) => enc::sar_cl_64(&mut self.core.text, dst),
            (MachineIntWidth::I32, MachineIntBinaryOp::ShrS) => enc::sar_cl_32(&mut self.core.text, dst),
            (MachineIntWidth::I64, MachineIntBinaryOp::ShrU) => enc::shr_cl_64(&mut self.core.text, dst),
            (MachineIntWidth::I32, MachineIntBinaryOp::ShrU) => enc::shr_cl_32(&mut self.core.text, dst),
            (MachineIntWidth::I64, MachineIntBinaryOp::Rotl) => enc::rol_cl_64(&mut self.core.text, dst),
            (MachineIntWidth::I32, MachineIntBinaryOp::Rotl) => enc::rol_cl_32(&mut self.core.text, dst),
            (MachineIntWidth::I64, MachineIntBinaryOp::Rotr) => enc::ror_cl_64(&mut self.core.text, dst),
            (MachineIntWidth::I32, MachineIntBinaryOp::Rotr) => enc::ror_cl_32(&mut self.core.text, dst),
            _ => unreachable!(),
        };
        if need_save_rcx {
            enc::mov_rr_64(&mut self.core.text, X86Reg::RCX, self.gp_scratch.reg(1));
        }
        Ok(())
    }

    /// Emit div/rem: x86_64 uses RAX:RDX for dividend, result in RAX (quot) / RDX (rem).
    fn lower_div_rem(
        &mut self,
        width: MachineIntWidth,
        op: MachineIntBinaryOp,
        dst: X86Reg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        // div/idiv implicitly uses RAX and RDX. RDX is a GP transient that
        // might hold a live value. Save it to self.gp_scratch.reg(1) (R11) and restore after.
        // (RAX = self.gp_scratch.reg(0), not in dynamic pool, so no save needed.)
        let need_save_rdx = dst != X86Reg::RDX;
        if need_save_rdx {
            enc::mov_rr_64(&mut self.core.text, self.gp_scratch.reg(1), X86Reg::RDX);
        }

        // Put dividend into RAX
        let lhs_gp = self.materialize_value(self.gp_scratch.reg(0), lhs)?;
        if lhs_gp != X86Reg::RAX {
            enc::mov_rr_64(&mut self.core.text, X86Reg::RAX, lhs_gp);
        }
        // Divisor must NOT be RAX or RDX. Use R10 as safe scratch for divisor.
        let rhs_gp = self.materialize_value(X86Reg::R10, rhs)?;
        if rhs_gp == X86Reg::RAX || rhs_gp == X86Reg::RDX {
            enc::mov_rr_64(&mut self.core.text, X86Reg::R10, rhs_gp);
        }
        let divisor = if rhs_gp == X86Reg::RAX || rhs_gp == X86Reg::RDX {
            X86Reg::R10
        } else {
            rhs_gp
        };

        // Division-by-zero check: divisor == 0 => trap
        enc::test_rr_64(&mut self.core.text, divisor, divisor);
        let div_zero_label = self.core.ensure_trap_label(MachineTrapKind::IntegerDivideByZero);
        self.emit_jcc(Cc::E, div_zero_label);

        match op {
            MachineIntBinaryOp::DivS => {
                // Signed overflow check: MIN / -1 => IntegerOverflow trap
                let not_min = self.core.new_label();
                match width {
                    MachineIntWidth::I32 => {
                        enc::cmp_ri_32(&mut self.core.text, X86Reg::RAX, i32::MIN);
                    }
                    MachineIntWidth::I64 => {
                        self.materialize_u64(X86Reg::RDX, i64::MIN as u64);
                        enc::cmp_rr_64(&mut self.core.text, X86Reg::RAX, X86Reg::RDX);
                    }
                };
                self.emit_jcc(Cc::NE, not_min);
                // Compare divisor against -1 using matching width
                match width {
                    MachineIntWidth::I32 => enc::cmp_ri_32(&mut self.core.text, divisor, -1),
                    MachineIntWidth::I64 => enc::cmp_ri_64(&mut self.core.text, divisor, -1),
                };
                let overflow_label = self.core.ensure_trap_label(MachineTrapKind::IntegerOverflow);
                self.emit_jcc(Cc::E, overflow_label);
                self.core.bind_label(not_min);
                // Sign-extend RAX → RDX:RAX, then IDIV
                match width {
                    MachineIntWidth::I64 => {
                        enc::cqo(&mut self.core.text);
                        enc::idiv_rm_64(&mut self.core.text, divisor);
                    }
                    MachineIntWidth::I32 => {
                        enc::cdq(&mut self.core.text);
                        enc::idiv_rm_32(&mut self.core.text, divisor);
                    }
                };
            }
            MachineIntBinaryOp::RemS => {
                // MIN % -1 = 0 (no trap, just skip the div)
                let not_min = self.core.new_label();
                let done = self.core.new_label();
                match width {
                    MachineIntWidth::I32 => {
                        enc::cmp_ri_32(&mut self.core.text, X86Reg::RAX, i32::MIN);
                    }
                    MachineIntWidth::I64 => {
                        self.materialize_u64(X86Reg::RDX, i64::MIN as u64);
                        enc::cmp_rr_64(&mut self.core.text, X86Reg::RAX, X86Reg::RDX);
                    }
                };
                self.emit_jcc(Cc::NE, not_min);
                match width {
                    MachineIntWidth::I32 => enc::cmp_ri_32(&mut self.core.text, divisor, -1),
                    MachineIntWidth::I64 => enc::cmp_ri_64(&mut self.core.text, divisor, -1),
                };
                self.emit_jcc(Cc::NE, not_min);
                // MIN % -1 = 0
                enc::xor_rr_32(&mut self.core.text, X86Reg::RDX, X86Reg::RDX);
                self.emit_jmp(done);
                self.core.bind_label(not_min);
                match width {
                    MachineIntWidth::I64 => {
                        enc::cqo(&mut self.core.text);
                        enc::idiv_rm_64(&mut self.core.text, divisor);
                    }
                    MachineIntWidth::I32 => {
                        enc::cdq(&mut self.core.text);
                        enc::idiv_rm_32(&mut self.core.text, divisor);
                    }
                };
                self.core.bind_label(done);
            }
            MachineIntBinaryOp::DivU | MachineIntBinaryOp::RemU => {
                // Zero-extend: XOR RDX, RDX
                enc::xor_rr_32(&mut self.core.text, X86Reg::RDX, X86Reg::RDX);
                match width {
                    MachineIntWidth::I64 => enc::div_rm_64(&mut self.core.text, divisor),
                    MachineIntWidth::I32 => enc::div_rm_32(&mut self.core.text, divisor),
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
            enc::mov_rr_64(&mut self.core.text, dst, result_reg);
        }
        // Restore RDX if it was saved
        if need_save_rdx {
            enc::mov_rr_64(&mut self.core.text, X86Reg::RDX, self.gp_scratch.reg(1));
        }
        Ok(())
    }

    // ── Integer compare ──────────────────────────────────────────────────────

    pub(super) fn lower_int_compare(
        &mut self,
        width: MachineIntWidth,
        kind: MachineCompareKind,
        sign: MachineSign,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        self.lower_cmp_values(width, lhs, rhs)?;
        let dst = self.map_gp_reg(dst)?;
        let cc = map_int_cond(kind, sign);
        enc::setcc(&mut self.core.text, cc, dst);
        enc::movzx_r32_r8(&mut self.core.text, dst, dst);
        Ok(())
    }

    pub(super) fn lower_cmp_values(
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
                let lhs_gp = self.materialize_value(self.gp_scratch.reg(0), lhs)?;
                match width {
                    MachineIntWidth::I64 => enc::cmp_ri_64(&mut self.core.text, lhs_gp, imm),
                    MachineIntWidth::I32 => enc::cmp_ri_32(&mut self.core.text, lhs_gp, imm),
                };
                return Ok(());
            }
        }
        let lhs_gp = self.materialize_value(self.gp_scratch.reg(0), lhs)?;
        let rhs_gp = self.materialize_value(self.gp_scratch.reg(1), rhs)?;
        match width {
            MachineIntWidth::I64 => enc::cmp_rr_64(&mut self.core.text, lhs_gp, rhs_gp),
            MachineIntWidth::I32 => enc::cmp_rr_32(&mut self.core.text, lhs_gp, rhs_gp),
        };
        Ok(())
    }

    // ── Select ───────────────────────────────────────────────────────────────

    pub(super) fn lower_select(
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
                    return self.lower_move(ty, dst, selected);
                }
                MachineValue::Reg(reg) => {
                    let cond_gp = self.map_gp_reg(reg)?;
                    let dst_fp = self.map_fp_reg(dst)? as u8;
                    let false_label = self.core.new_label();
                    let done = self.core.new_label();
                    enc::test_rr_64(&mut self.core.text, cond_gp, cond_gp);
                    self.emit_jcc(Cc::E, false_label);
                    let true_fp =
                        self.prepare_float_operand(width, on_true, self.gp_scratch.reg(0), self.fp_scratch.reg(0))?;
                    if dst_fp != true_fp as u8 {
                        match width {
                            MachineFloatWidth::F32 => {
                                enc::movss_rr(&mut self.core.text, dst_fp, true_fp as u8)
                            }
                            MachineFloatWidth::F64 => {
                                enc::movsd_rr(&mut self.core.text, dst_fp, true_fp as u8)
                            }
                        };
                    }
                    self.emit_jmp(done);
                    self.core.bind_label(false_label);
                    let false_fp =
                        self.prepare_float_operand(width, on_false, self.gp_scratch.reg(1), self.fp_scratch.reg(1))?;
                    if dst_fp != false_fp as u8 {
                        match width {
                            MachineFloatWidth::F32 => {
                                enc::movss_rr(&mut self.core.text, dst_fp, false_fp as u8)
                            }
                            MachineFloatWidth::F64 => {
                                enc::movsd_rr(&mut self.core.text, dst_fp, false_fp as u8)
                            }
                        };
                    }
                    self.core.bind_label(done);
                    self.core.set_fp_reg_width(dst, width)?;
                    Ok(())
                }
            }
        } else {
            if let MachineValue::Imm64(value) = cond {
                let selected = if value != 0 { on_true } else { on_false };
                return self.lower_move(ty, dst, selected);
            }
            let dst = self.map_gp_reg(dst)?;
            // Materialize operands BEFORE testing the condition, because
            // materialize_value may clobber flags (e.g. xor reg,reg for zero).
            let true_reg = self.materialize_value(self.gp_scratch.reg(0), on_true)?;
            let false_reg = self.materialize_value(self.gp_scratch.reg(1), on_false)?;
            let cond_gp = match cond {
                MachineValue::Reg(reg) => self.map_gp_reg(reg)?,
                _ => unreachable!(),
            };
            enc::test_rr_64(&mut self.core.text, cond_gp, cond_gp);
            if dst == true_reg && dst != false_reg {
                enc::cmovcc_rr_64(&mut self.core.text, Cc::E, dst, false_reg);
            } else if dst == false_reg {
                enc::cmovcc_rr_64(&mut self.core.text, Cc::NE, dst, true_reg);
            } else {
                enc::mov_rr_64(&mut self.core.text, dst, false_reg);
                enc::cmovcc_rr_64(&mut self.core.text, Cc::NE, dst, true_reg);
            }
            Ok(())
        }
    }

    // ── Convert (integer parts only — float conversions are Phase 4) ─────────

    pub(super) fn lower_convert(
        &mut self,
        op: MachineConvertOp,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let dst_float_width = convert_result_float_width(op);
        let src_gp = self.materialize_value(self.gp_scratch.reg(0), src)?;
        match op {
            MachineConvertOp::I32WrapI64 => {
                let dst_gp = self.map_gp_reg(dst)?;
                // MOV r32, r32 zeroes the upper 32 bits
                enc::mov_rr_32(&mut self.core.text, dst_gp, src_gp);
            }
            MachineConvertOp::I64ExtendI32S => {
                let dst_gp = self.map_gp_reg(dst)?;
                enc::movsxd_r64_r32(&mut self.core.text, dst_gp, src_gp);
            }
            MachineConvertOp::I64ExtendI32U => {
                let dst_gp = self.map_gp_reg(dst)?;
                // MOV r32, r32 zero-extends
                enc::mov_rr_32(&mut self.core.text, dst_gp, src_gp);
            }
            MachineConvertOp::I32ReinterpretF32 | MachineConvertOp::I64ReinterpretF64 => {
                let dst_gp = self.map_gp_reg(dst)?;
                if dst_gp != src_gp {
                    enc::mov_rr_64(&mut self.core.text, dst_gp, src_gp);
                }
            }
            MachineConvertOp::F32ReinterpretI32 | MachineConvertOp::F64ReinterpretI64 => {
                let width = dst_float_width.expect("float reinterpret width");
                if self.core.is_fp_reg(dst) {
                    let dst_fp = self.map_fp_reg(dst)? as u8;
                    match width {
                        MachineFloatWidth::F32 => enc::movd_xmm_r32(&mut self.core.text, dst_fp, src_gp),
                        MachineFloatWidth::F64 => enc::movq_xmm_r64(&mut self.core.text, dst_fp, src_gp),
                    };
                    self.core.set_fp_reg_width(dst, width)?;
                } else {
                    let dst_gp = self.map_gp_reg(dst)?;
                    if dst_gp != src_gp {
                        enc::mov_rr_64(&mut self.core.text, dst_gp, src_gp);
                    }
                }
            }
            // Float promotion / demotion
            MachineConvertOp::F64PromoteF32 => {
                let src_fp =
                    self.prepare_float_operand(MachineFloatWidth::F32, src, self.gp_scratch.reg(0), self.fp_scratch.reg(0))?;
                if self.core.is_fp_reg(dst) {
                    let dst_fp = self.map_fp_reg(dst)? as u8;
                    self.core.set_fp_reg_width(dst, MachineFloatWidth::F64)?;
                    enc::cvtss2sd(&mut self.core.text, dst_fp, src_fp as u8);
                } else {
                    enc::cvtss2sd(&mut self.core.text, self.fp_scratch.reg(1) as u8, src_fp as u8);
                    let dst_gp = self.map_gp_reg(dst)?;
                    enc::movq_r64_xmm(&mut self.core.text, dst_gp, self.fp_scratch.reg(1) as u8);
                }
            }
            MachineConvertOp::F32DemoteF64 => {
                let src_fp =
                    self.prepare_float_operand(MachineFloatWidth::F64, src, self.gp_scratch.reg(0), self.fp_scratch.reg(0))?;
                if self.core.is_fp_reg(dst) {
                    let dst_fp = self.map_fp_reg(dst)? as u8;
                    self.core.set_fp_reg_width(dst, MachineFloatWidth::F32)?;
                    enc::cvtsd2ss(&mut self.core.text, dst_fp, src_fp as u8);
                } else {
                    enc::cvtsd2ss(&mut self.core.text, self.fp_scratch.reg(1) as u8, src_fp as u8);
                    let dst_gp = self.map_gp_reg(dst)?;
                    enc::movd_r32_xmm(&mut self.core.text, dst_gp, self.fp_scratch.reg(1) as u8);
                }
            }
            // Int -> Float conversions
            MachineConvertOp::F32ConvertI32S => {
                // CVTSI2SS xmm, r32
                let dst_fp = self.dst_float_reg(dst, MachineFloatWidth::F32)?;
                enc::cvtsi2ss_r32(&mut self.core.text, dst_fp, src_gp);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F32, dst_fp)?;
            }
            MachineConvertOp::F32ConvertI32U => {
                // Zero-extend to 64-bit first for unsigned interpretation
                enc::mov_rr_32(&mut self.core.text, self.gp_scratch.reg(0), src_gp);
                let dst_fp = self.dst_float_reg(dst, MachineFloatWidth::F32)?;
                enc::cvtsi2ss_r64(&mut self.core.text, dst_fp, self.gp_scratch.reg(0));
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F32, dst_fp)?;
            }
            MachineConvertOp::F32ConvertI64S => {
                let dst_fp = self.dst_float_reg(dst, MachineFloatWidth::F32)?;
                enc::cvtsi2ss_r64(&mut self.core.text, dst_fp, src_gp);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F32, dst_fp)?;
            }
            MachineConvertOp::F32ConvertI64U => {
                // x86_64 has no unsigned int-to-float instruction.
                // For values that fit in i64 (bit 63 = 0), use signed conversion.
                // For values with bit 63 set, shift right by 1, convert, then double.
                let dst_fp = self.dst_float_reg(dst, MachineFloatWidth::F32)?;
                enc::test_rr_64(&mut self.core.text, src_gp, src_gp);
                let large = self.core.new_label();
                self.emit_jcc(Cc::S, large); // JS = sign flag set = bit 63 is 1
                                             // Small path: fits in i64
                enc::cvtsi2ss_r64(&mut self.core.text, dst_fp, src_gp);
                let done = self.core.new_label();
                self.emit_jmp(done);
                // Large path: bit 63 set
                self.core.bind_label(large);
                enc::mov_rr_64(&mut self.core.text, self.gp_scratch.reg(0), src_gp);
                enc::mov_rr_64(&mut self.core.text, self.gp_scratch.reg(1), src_gp);
                enc::shr_imm_64(&mut self.core.text, self.gp_scratch.reg(0), 1); // src >> 1
                enc::and_ri_32(&mut self.core.text, self.gp_scratch.reg(1), 1); // src & 1 (preserve LSB)
                enc::or_rr_64(&mut self.core.text, self.gp_scratch.reg(0), self.gp_scratch.reg(1)); // (src >> 1) | (src & 1)
                enc::cvtsi2ss_r64(&mut self.core.text, dst_fp, self.gp_scratch.reg(0));
                enc::addss(&mut self.core.text, dst_fp, dst_fp); // double it
                self.core.bind_label(done);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F32, dst_fp)?;
            }
            MachineConvertOp::F64ConvertI32S => {
                let dst_fp = self.dst_float_reg(dst, MachineFloatWidth::F64)?;
                enc::cvtsi2sd_r32(&mut self.core.text, dst_fp, src_gp);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F64, dst_fp)?;
            }
            MachineConvertOp::F64ConvertI32U => {
                enc::mov_rr_32(&mut self.core.text, self.gp_scratch.reg(0), src_gp);
                let dst_fp = self.dst_float_reg(dst, MachineFloatWidth::F64)?;
                enc::cvtsi2sd_r64(&mut self.core.text, dst_fp, self.gp_scratch.reg(0));
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F64, dst_fp)?;
            }
            MachineConvertOp::F64ConvertI64S => {
                let dst_fp = self.dst_float_reg(dst, MachineFloatWidth::F64)?;
                enc::cvtsi2sd_r64(&mut self.core.text, dst_fp, src_gp);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F64, dst_fp)?;
            }
            MachineConvertOp::F64ConvertI64U => {
                let dst_fp = self.dst_float_reg(dst, MachineFloatWidth::F64)?;
                enc::test_rr_64(&mut self.core.text, src_gp, src_gp);
                let large = self.core.new_label();
                self.emit_jcc(Cc::S, large);
                enc::cvtsi2sd_r64(&mut self.core.text, dst_fp, src_gp);
                let done = self.core.new_label();
                self.emit_jmp(done);
                self.core.bind_label(large);
                enc::mov_rr_64(&mut self.core.text, self.gp_scratch.reg(0), src_gp);
                enc::mov_rr_64(&mut self.core.text, self.gp_scratch.reg(1), src_gp);
                enc::shr_imm_64(&mut self.core.text, self.gp_scratch.reg(0), 1);
                enc::and_ri_32(&mut self.core.text, self.gp_scratch.reg(1), 1);
                enc::or_rr_64(&mut self.core.text, self.gp_scratch.reg(0), self.gp_scratch.reg(1));
                enc::cvtsi2sd_r64(&mut self.core.text, dst_fp, self.gp_scratch.reg(0));
                enc::addsd(&mut self.core.text, dst_fp, dst_fp);
                self.core.bind_label(done);
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
                self.lower_trapping_trunc(op, dst_gp, src_gp)?;
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
                self.lower_saturating_trunc(op, dst_gp, src_gp)?;
            }
        }
        Ok(())
    }

    // ── Float ops ─────────────────────────────────────────────────────────────

    pub(super) fn lower_float_unary(
        &mut self,
        width: MachineFloatWidth,
        op: MachineFloatUnaryOp,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let src_fp = self.prepare_float_operand(width, src, self.gp_scratch.reg(0), self.fp_scratch.reg(0))?;
        let result_fp = if self.core.is_fp_reg(dst) {
            let dst_fp = self.map_fp_reg(dst)? as u8;
            self.core.set_fp_reg_width(dst, width)?;
            dst_fp
        } else {
            self.fp_scratch.reg(2) as u8
        };
        match op {
            MachineFloatUnaryOp::Sqrt => {
                match width {
                    MachineFloatWidth::F32 => enc::sqrtss(&mut self.core.text, result_fp, src_fp as u8),
                    MachineFloatWidth::F64 => enc::sqrtsd(&mut self.core.text, result_fp, src_fp as u8),
                };
            }
            MachineFloatUnaryOp::Ceil => {
                match width {
                    MachineFloatWidth::F32 => {
                        enc::roundss(&mut self.core.text, result_fp, src_fp as u8, enc::ROUND_CEIL)
                    }
                    MachineFloatWidth::F64 => {
                        enc::roundsd(&mut self.core.text, result_fp, src_fp as u8, enc::ROUND_CEIL)
                    }
                };
            }
            MachineFloatUnaryOp::Floor => {
                match width {
                    MachineFloatWidth::F32 => {
                        enc::roundss(&mut self.core.text, result_fp, src_fp as u8, enc::ROUND_FLOOR)
                    }
                    MachineFloatWidth::F64 => {
                        enc::roundsd(&mut self.core.text, result_fp, src_fp as u8, enc::ROUND_FLOOR)
                    }
                };
            }
            MachineFloatUnaryOp::Trunc => {
                match width {
                    MachineFloatWidth::F32 => {
                        enc::roundss(&mut self.core.text, result_fp, src_fp as u8, enc::ROUND_TRUNC)
                    }
                    MachineFloatWidth::F64 => {
                        enc::roundsd(&mut self.core.text, result_fp, src_fp as u8, enc::ROUND_TRUNC)
                    }
                };
            }
            MachineFloatUnaryOp::Nearest => {
                match width {
                    MachineFloatWidth::F32 => {
                        enc::roundss(&mut self.core.text, result_fp, src_fp as u8, enc::ROUND_NEAREST)
                    }
                    MachineFloatWidth::F64 => {
                        enc::roundsd(&mut self.core.text, result_fp, src_fp as u8, enc::ROUND_NEAREST)
                    }
                };
            }
            MachineFloatUnaryOp::Abs => {
                // Clear sign bit: AND with mask.
                let mask_xmm = if result_fp != self.fp_scratch.reg(0) as u8 {
                    self.fp_scratch.reg(0) as u8
                } else {
                    self.fp_scratch.reg(2) as u8
                };
                if result_fp != src_fp as u8 {
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movss_rr(&mut self.core.text, result_fp, src_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.core.text, result_fp, src_fp as u8)
                        }
                    };
                }
                let mask = match width {
                    MachineFloatWidth::F32 => 0x7FFF_FFFFu64,
                    MachineFloatWidth::F64 => 0x7FFF_FFFF_FFFF_FFFFu64,
                };
                self.materialize_u64(self.gp_scratch.reg(0), mask);
                enc::movq_xmm_r64(&mut self.core.text, mask_xmm, self.gp_scratch.reg(0));
                enc::andpd(&mut self.core.text, result_fp, mask_xmm);
            }
            MachineFloatUnaryOp::Neg => {
                // Flip sign bit: XOR with mask.
                let mask_xmm = if result_fp != self.fp_scratch.reg(0) as u8 {
                    self.fp_scratch.reg(0) as u8
                } else {
                    self.fp_scratch.reg(2) as u8
                };
                if result_fp != src_fp as u8 {
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movss_rr(&mut self.core.text, result_fp, src_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.core.text, result_fp, src_fp as u8)
                        }
                    };
                }
                let mask = match width {
                    MachineFloatWidth::F32 => 0x8000_0000u64,
                    MachineFloatWidth::F64 => 0x8000_0000_0000_0000u64,
                };
                self.materialize_u64(self.gp_scratch.reg(0), mask);
                enc::movq_xmm_r64(&mut self.core.text, mask_xmm, self.gp_scratch.reg(0));
                enc::xorpd(&mut self.core.text, result_fp, mask_xmm);
            }
        }
        if !self.core.is_fp_reg(dst) {
            let dst_gp = self.map_gp_reg(dst)?;
            match width {
                MachineFloatWidth::F32 => enc::movd_r32_xmm(&mut self.core.text, dst_gp, result_fp),
                MachineFloatWidth::F64 => enc::movq_r64_xmm(&mut self.core.text, dst_gp, result_fp),
            };
        }
        Ok(())
    }

    pub(super) fn lower_float_binary(
        &mut self,
        width: MachineFloatWidth,
        op: MachineFloatBinaryOp,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        let lhs_fp = self.prepare_float_operand(width, lhs, self.gp_scratch.reg(0), self.fp_scratch.reg(0))?;
        let rhs_fp = self.prepare_float_operand(width, rhs, self.gp_scratch.reg(1), self.fp_scratch.reg(1))?;
        let result_fp = if self.core.is_fp_reg(dst) {
            let dst_fp = self.map_fp_reg(dst)? as u8;
            self.core.set_fp_reg_width(dst, width)?;
            dst_fp
        } else {
            self.fp_scratch.reg(2) as u8
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
                    let scratch = self.fp_scratch.reg(2) as u8;
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movss_rr(&mut self.core.text, scratch, rhs_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.core.text, scratch, rhs_fp as u8)
                        }
                    };
                    scratch
                } else {
                    rhs_fp as u8
                };
                if result_fp != lhs_fp as u8 {
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movss_rr(&mut self.core.text, result_fp, lhs_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.core.text, result_fp, lhs_fp as u8)
                        }
                    };
                }
                match (width, op) {
                    (MachineFloatWidth::F32, MachineFloatBinaryOp::Add) => {
                        enc::addss(&mut self.core.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F64, MachineFloatBinaryOp::Add) => {
                        enc::addsd(&mut self.core.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F32, MachineFloatBinaryOp::Sub) => {
                        enc::subss(&mut self.core.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F64, MachineFloatBinaryOp::Sub) => {
                        enc::subsd(&mut self.core.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F32, MachineFloatBinaryOp::Mul) => {
                        enc::mulss(&mut self.core.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F64, MachineFloatBinaryOp::Mul) => {
                        enc::mulsd(&mut self.core.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F32, MachineFloatBinaryOp::Div) => {
                        enc::divss(&mut self.core.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F64, MachineFloatBinaryOp::Div) => {
                        enc::divsd(&mut self.core.text, result_fp, actual_rhs)
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
                    let scratch = self.fp_scratch.reg(2) as u8;
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movss_rr(&mut self.core.text, scratch, rhs_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.core.text, scratch, rhs_fp as u8)
                        }
                    };
                    scratch
                } else {
                    rhs_fp as u8
                };
                if result_fp != lhs_fp as u8 {
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movss_rr(&mut self.core.text, result_fp, lhs_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.core.text, result_fp, lhs_fp as u8)
                        }
                    };
                }
                match width {
                    MachineFloatWidth::F32 => enc::minss(&mut self.core.text, result_fp, actual_rhs),
                    MachineFloatWidth::F64 => enc::minsd(&mut self.core.text, result_fp, actual_rhs),
                };
                // Compare for NaN: ucomisd lhs, rhs sets PF=1 if unordered (NaN)
                match width {
                    MachineFloatWidth::F32 => {
                        enc::ucomiss(&mut self.core.text, lhs_fp as u8, actual_rhs)
                    }
                    MachineFloatWidth::F64 => {
                        enc::ucomisd(&mut self.core.text, lhs_fp as u8, actual_rhs)
                    }
                };
                let done = self.core.new_label();
                self.emit_jcc(Cc::NP, done); // no NaN => minsd result is correct
                                             // NaN case: add propagates NaN
                if result_fp != lhs_fp as u8 {
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movss_rr(&mut self.core.text, result_fp, lhs_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.core.text, result_fp, lhs_fp as u8)
                        }
                    };
                }
                match width {
                    MachineFloatWidth::F32 => enc::addss(&mut self.core.text, result_fp, actual_rhs),
                    MachineFloatWidth::F64 => enc::addsd(&mut self.core.text, result_fp, actual_rhs),
                };
                self.core.bind_label(done);
            }
            MachineFloatBinaryOp::Max => {
                // Same NaN handling as Min but with maxsd/maxss.
                // Guard: if result == rhs and result != lhs, save rhs to scratch first.
                let actual_rhs = if result_fp == rhs_fp as u8 && result_fp != lhs_fp as u8 {
                    let scratch = self.fp_scratch.reg(2) as u8;
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movss_rr(&mut self.core.text, scratch, rhs_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.core.text, scratch, rhs_fp as u8)
                        }
                    };
                    scratch
                } else {
                    rhs_fp as u8
                };
                if result_fp != lhs_fp as u8 {
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movss_rr(&mut self.core.text, result_fp, lhs_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.core.text, result_fp, lhs_fp as u8)
                        }
                    };
                }
                match width {
                    MachineFloatWidth::F32 => enc::maxss(&mut self.core.text, result_fp, actual_rhs),
                    MachineFloatWidth::F64 => enc::maxsd(&mut self.core.text, result_fp, actual_rhs),
                };
                match width {
                    MachineFloatWidth::F32 => {
                        enc::ucomiss(&mut self.core.text, lhs_fp as u8, actual_rhs)
                    }
                    MachineFloatWidth::F64 => {
                        enc::ucomisd(&mut self.core.text, lhs_fp as u8, actual_rhs)
                    }
                };
                let done = self.core.new_label();
                self.emit_jcc(Cc::NP, done);
                if result_fp != lhs_fp as u8 {
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movss_rr(&mut self.core.text, result_fp, lhs_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.core.text, result_fp, lhs_fp as u8)
                        }
                    };
                }
                match width {
                    MachineFloatWidth::F32 => enc::addss(&mut self.core.text, result_fp, actual_rhs),
                    MachineFloatWidth::F64 => enc::addsd(&mut self.core.text, result_fp, actual_rhs),
                };
                self.core.bind_label(done);
            }
            MachineFloatBinaryOp::Copysign => {
                // magnitude of lhs, sign of rhs.
                // Strategy: clear sign of lhs (abs), extract sign of rhs, OR them.
                // Use a mask scratch that doesn't conflict with result_fp or rhs_fp.
                let mask_xmm =
                    if result_fp != self.fp_scratch.reg(0) as u8 && rhs_fp as u8 != self.fp_scratch.reg(0) as u8 {
                        self.fp_scratch.reg(0) as u8
                    } else {
                        self.fp_scratch.reg(2) as u8
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
                            enc::movss_rr(&mut self.core.text, result_fp, lhs_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.core.text, result_fp, lhs_fp as u8)
                        }
                    };
                }
                self.materialize_u64(self.gp_scratch.reg(0), abs_mask);
                enc::movq_xmm_r64(&mut self.core.text, mask_xmm, self.gp_scratch.reg(0));
                enc::andpd(&mut self.core.text, result_fp, mask_xmm);
                // mask_xmm = rhs & sign_mask (extract sign bit)
                self.materialize_u64(self.gp_scratch.reg(0), sign_mask);
                enc::movq_xmm_r64(&mut self.core.text, mask_xmm, self.gp_scratch.reg(0));
                enc::andpd(&mut self.core.text, mask_xmm, rhs_fp as u8);
                // result |= mask_xmm
                enc::orpd(&mut self.core.text, result_fp, mask_xmm);
            }
        };
        if !self.core.is_fp_reg(dst) {
            let dst_gp = self.map_gp_reg(dst)?;
            match width {
                MachineFloatWidth::F32 => enc::movd_r32_xmm(&mut self.core.text, dst_gp, result_fp),
                MachineFloatWidth::F64 => enc::movq_r64_xmm(&mut self.core.text, dst_gp, result_fp),
            };
        }
        Ok(())
    }

    pub(super) fn lower_float_compare(
        &mut self,
        width: MachineFloatWidth,
        kind: MachineCompareKind,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        let dst_gp = self.map_gp_reg(dst)?;
        let lhs_fp = self.prepare_float_operand(width, lhs, self.gp_scratch.reg(0), self.fp_scratch.reg(0))?;
        // Choose an rhs FP scratch that doesn't conflict with lhs. When lhs
        // already lives in a mapped FP register (not self.fp_scratch.reg(0)), reuse
        // self.fp_scratch.reg(0) for rhs to avoid clobbering live FP transients in
        // self.fp_scratch.reg(1)/self.fp_scratch.reg(2).
        let rhs_fp_scratch = if lhs_fp != self.fp_scratch.reg(0) as u32 {
            self.fp_scratch.reg(0)
        } else {
            self.fp_scratch.reg(2)
        };
        if matches!(rhs, MachineValue::Imm64(0)) {
            enc::xorpd(&mut self.core.text, rhs_fp_scratch as u8, rhs_fp_scratch as u8);
            match width {
                MachineFloatWidth::F32 => {
                    enc::ucomiss(&mut self.core.text, lhs_fp as u8, rhs_fp_scratch as u8)
                }
                MachineFloatWidth::F64 => {
                    enc::ucomisd(&mut self.core.text, lhs_fp as u8, rhs_fp_scratch as u8)
                }
            };
        } else {
            let rhs_fp = self.prepare_float_operand(width, rhs, self.gp_scratch.reg(1), rhs_fp_scratch)?;
            match width {
                MachineFloatWidth::F32 => enc::ucomiss(&mut self.core.text, lhs_fp as u8, rhs_fp as u8),
                MachineFloatWidth::F64 => enc::ucomisd(&mut self.core.text, lhs_fp as u8, rhs_fp as u8),
            };
        }
        // Wasm float comparisons: unordered (NaN) => 0 for all except Ne.
        // UCOMISD sets: ZF=1,PF=1,CF=1 for unordered; ZF=1,PF=0,CF=0 for equal;
        // ZF=0,PF=0,CF=1 for less-than; ZF=0,PF=0,CF=0 for greater-than.
        match kind {
            MachineCompareKind::Eq => {
                // Ordered and equal: ZF=1 AND PF=0
                // SETE sets dst to ZF, SETNP sets tmp to !PF, AND them.
                enc::setcc(&mut self.core.text, Cc::E, dst_gp);
                enc::setcc(&mut self.core.text, Cc::NP, self.gp_scratch.reg(0));
                enc::and_rr_32(&mut self.core.text, dst_gp, self.gp_scratch.reg(0));
                // Zero-extend the byte result to full register
                enc::movzx_r32_r8(&mut self.core.text, dst_gp, dst_gp);
            }
            MachineCompareKind::Ne => {
                // Unordered OR not-equal: NE=1 OR PF=1
                enc::setcc(&mut self.core.text, Cc::NE, dst_gp);
                enc::setcc(&mut self.core.text, Cc::P, self.gp_scratch.reg(0));
                enc::or_rr_32(&mut self.core.text, dst_gp, self.gp_scratch.reg(0));
                enc::movzx_r32_r8(&mut self.core.text, dst_gp, dst_gp);
            }
            MachineCompareKind::Lt => {
                // Ordered and less: CF=1 AND PF=0 → use JB (CF=1), but need !PF too
                // Actually, for UCOMISD: CF=1 for less-than AND for unordered.
                // So Lt = CF=1 AND PF=0 → SETB AND SETNP
                enc::setcc(&mut self.core.text, Cc::B, dst_gp);
                enc::setcc(&mut self.core.text, Cc::NP, self.gp_scratch.reg(0));
                enc::and_rr_32(&mut self.core.text, dst_gp, self.gp_scratch.reg(0));
                enc::movzx_r32_r8(&mut self.core.text, dst_gp, dst_gp);
            }
            MachineCompareKind::Gt => {
                // Ordered and greater: ZF=0, CF=0, PF=0 → JA (CF=0 AND ZF=0)
                // JA already excludes unordered (PF=1 implies CF=1), so SETA is correct.
                enc::setcc(&mut self.core.text, Cc::A, dst_gp);
                enc::movzx_r32_r8(&mut self.core.text, dst_gp, dst_gp);
            }
            MachineCompareKind::Le => {
                // Ordered and less-or-equal: (CF=1 OR ZF=1) AND PF=0
                enc::setcc(&mut self.core.text, Cc::BE, dst_gp);
                enc::setcc(&mut self.core.text, Cc::NP, self.gp_scratch.reg(0));
                enc::and_rr_32(&mut self.core.text, dst_gp, self.gp_scratch.reg(0));
                enc::movzx_r32_r8(&mut self.core.text, dst_gp, dst_gp);
            }
            MachineCompareKind::Ge => {
                // Ordered and greater-or-equal: CF=0 AND PF=0 → JAE excludes unordered already
                // Actually JAE = !CF. Unordered sets CF=1, so JAE is 0 for unordered. Correct.
                enc::setcc(&mut self.core.text, Cc::AE, dst_gp);
                enc::movzx_r32_r8(&mut self.core.text, dst_gp, dst_gp);
            }
        }
        Ok(())
    }

    // ── Helper calls ──────────────────────────────────────────────────────────

    pub(super) fn lower_call_helper(
        &mut self,
        extern_idx: usize,
        const_idx: usize,
    ) -> Result<(), WasmError> {
        let binding = self
            .core.compiled
            .module()
            .externs
            .get(extern_idx)
            .ok_or_else(|| WasmError::internal("x86_64 helper target is out of range".into()))?;
        let metadata = self
            .core.compiled
            .const_ptr(crate::vm::machine::machine_ir::MachineConstId(
                const_idx as u32,
            ))
            .ok_or_else(|| WasmError::internal("x86_64 helper metadata is out of range".into()))?;
        enc::mov_rr_64(&mut self.core.text, C_ARG0, map_fixed_reg(crate::vm::machine::machine_ir::MACHINE_CTX_REG));
        enc::mov_rr_64(&mut self.core.text, C_ARG1, map_fixed_reg(MACHINE_FP_REG));
        self.materialize_u64(C_ARG2, metadata as u64);
        self.materialize_u64(
            self.gp_scratch.reg(1),
            resolve_helper_entry(binding.symbol) as usize as u64,
        );
        enc::call_reg(&mut self.core.text, self.gp_scratch.reg(1));
        // Check return: RAX != 0 => error
        enc::test_rr_32(&mut self.core.text, X86Reg::RAX, X86Reg::RAX);
        self.emit_jcc(Cc::NE, self.core.return_error_label);
        Ok(())
    }

    // ── Float conversion helpers ────────────────────────────────────────────

    /// Get FP destination register: if dst is FP reg, use it directly; else use scratch.
    fn dst_float_reg(
        &mut self,
        dst: MachineReg,
        width: MachineFloatWidth,
    ) -> Result<u8, WasmError> {
        if self.core.is_fp_reg(dst) {
            let dst_fp = self.map_fp_reg(dst)? as u8;
            self.core.set_fp_reg_width(dst, width)?;
            Ok(dst_fp)
        } else {
            Ok(self.fp_scratch.reg(1) as u8)
        }
    }

    /// If dst is a GP register, move float result from XMM to GP.
    fn store_fp_result_if_gp(
        &mut self,
        dst: MachineReg,
        width: MachineFloatWidth,
        fp_reg: u8,
    ) -> Result<(), WasmError> {
        if !self.core.is_fp_reg(dst) {
            let dst_gp = self.map_gp_reg(dst)?;
            match width {
                MachineFloatWidth::F32 => enc::movd_r32_xmm(&mut self.core.text, dst_gp, fp_reg),
                MachineFloatWidth::F64 => enc::movq_r64_xmm(&mut self.core.text, dst_gp, fp_reg),
            };
        }
        Ok(())
    }

    /// Save GP transient registers to the system stack before a C helper call.
    /// Pushes 8 registers (7 transients + padding) for 16-byte alignment.
    fn save_gp_transients(&mut self) {
        // Push 7 GP transients + 1 padding for 16-byte alignment (8 * 8 = 64 bytes)
        enc::push(&mut self.core.text, X86Reg::RCX);
        enc::push(&mut self.core.text, X86Reg::RDX);
        enc::push(&mut self.core.text, X86Reg::RSI);
        enc::push(&mut self.core.text, X86Reg::RDI);
        enc::push(&mut self.core.text, X86Reg::R8);
        enc::push(&mut self.core.text, X86Reg::R9);
        enc::push(&mut self.core.text, X86Reg::R10);
        enc::push(&mut self.core.text, X86Reg::R10); // padding for 16-byte alignment
    }

    /// Restore GP transient registers from the system stack after a C helper call.
    fn restore_gp_transients(&mut self) {
        enc::pop(&mut self.core.text, X86Reg::R10); // padding
        enc::pop(&mut self.core.text, X86Reg::R10);
        enc::pop(&mut self.core.text, X86Reg::R9);
        enc::pop(&mut self.core.text, X86Reg::R8);
        enc::pop(&mut self.core.text, X86Reg::RDI);
        enc::pop(&mut self.core.text, X86Reg::RSI);
        enc::pop(&mut self.core.text, X86Reg::RDX);
        enc::pop(&mut self.core.text, X86Reg::RCX);
    }

    fn lower_trapping_trunc(
        &mut self,
        op: MachineConvertOp,
        dst: X86Reg,
        src: X86Reg,
    ) -> Result<(), WasmError> {
        // The C helper call clobbers all GP transient registers. Save them.
        self.save_gp_transients();
        #[cfg(not(target_os = "windows"))]
        {
            enc::mov_rr_64(&mut self.core.text, C_ARG0, map_fixed_reg(crate::vm::machine::machine_ir::MACHINE_CTX_REG));
            enc::mov_rr_64(&mut self.core.text, C_ARG1, src);
            self.materialize_u64(C_ARG2, convert_op_code(op));
            self.materialize_u64(self.gp_scratch.reg(1), x86_64_trapping_trunc as usize as u64);
            enc::call_reg(&mut self.core.text, self.gp_scratch.reg(1));
            enc::mov_rr_64(&mut self.core.text, self.gp_scratch.reg(1), X86Reg::RDX);
            self.restore_gp_transients();
            enc::test_rr_64(&mut self.core.text, X86Reg::RAX, X86Reg::RAX);
            self.emit_jcc(Cc::NE, self.core.return_error_label);
        }
        #[cfg(target_os = "windows")]
        {
            enc::mov_rr_64(&mut self.core.text, C_ARG0, map_fixed_reg(crate::vm::machine::machine_ir::MACHINE_CTX_REG));
            enc::mov_rr_64(&mut self.core.text, C_ARG1, src);
            self.materialize_u64(C_ARG2, convert_op_code(op));
            enc::mov_rr_64(&mut self.core.text, C_ARG3, X86Reg::RSP);
            self.materialize_u64(self.gp_scratch.reg(1), x86_64_trapping_trunc_win as usize as u64);
            enc::call_reg(&mut self.core.text, self.gp_scratch.reg(1));
            enc::load_64(&mut self.core.text, self.gp_scratch.reg(1), X86Reg::RSP, 0);
            self.restore_gp_transients();
            enc::test_rr_32(&mut self.core.text, X86Reg::RAX, X86Reg::RAX);
            self.emit_jcc(Cc::NE, self.core.return_error_label);
        }
        if dst != self.gp_scratch.reg(1) {
            enc::mov_rr_64(&mut self.core.text, dst, self.gp_scratch.reg(1));
        }
        Ok(())
    }

    fn lower_saturating_trunc(
        &mut self,
        op: MachineConvertOp,
        dst: X86Reg,
        src: X86Reg,
    ) -> Result<(), WasmError> {
        // The C helper call clobbers all GP transient registers. Save them.
        self.save_gp_transients();
        enc::mov_rr_64(&mut self.core.text, C_ARG0, src);
        self.materialize_u64(C_ARG1, convert_op_code(op));
        self.materialize_u64(self.gp_scratch.reg(1), x86_64_saturating_trunc as usize as u64);
        enc::call_reg(&mut self.core.text, self.gp_scratch.reg(1));
        // Save result before restoring transients
        enc::mov_rr_64(&mut self.core.text, self.gp_scratch.reg(1), X86Reg::RAX);
        self.restore_gp_transients();
        if dst != self.gp_scratch.reg(1) {
            enc::mov_rr_64(&mut self.core.text, dst, self.gp_scratch.reg(1));
        }
        Ok(())
    }

    /// Decomposed indexed load: extend(index) + offset into self.gp_scratch.reg(0), then
    /// load from [base + self.gp_scratch.reg(0)]. Stable-base form for store-forwarding.
    /// TODO: use x86_64 [base + index + disp] addressing for 1-2 instructions.
    pub(super) fn lower_indexed_load_decomposed(
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
        // Step 1: copy/extend index into self.gp_scratch.reg(0)
        if index_extend == MachineIndexExtend::ZeroExtend32 {
            enc::mov_rr_32(&mut self.core.text, self.gp_scratch.reg(0), index_x86);
        } else {
            enc::mov_rr_64(&mut self.core.text, self.gp_scratch.reg(0), index_x86);
        }
        // Step 2: add offset
        if offset != 0 {
            enc::add_ri_64(&mut self.core.text, self.gp_scratch.reg(0), offset);
        }
        // Step 3: add base → self.gp_scratch.reg(0) = base + extended_index + offset
        enc::add_rr_64(&mut self.core.text, self.gp_scratch.reg(0), base_x86);
        // Step 4: load from [scratch + 0]
        self.lower_load_from(dst, self.gp_scratch.reg(0), 0, width, extension)
    }

    /// Decomposed indexed store.
    pub(super) fn lower_indexed_store_decomposed(
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
            enc::mov_rr_32(&mut self.core.text, self.gp_scratch.reg(0), index_x86);
        } else {
            enc::mov_rr_64(&mut self.core.text, self.gp_scratch.reg(0), index_x86);
        }
        if offset != 0 {
            enc::add_ri_64(&mut self.core.text, self.gp_scratch.reg(0), offset);
        }
        enc::add_rr_64(&mut self.core.text, self.gp_scratch.reg(0), base_x86);
        self.lower_store_to(
            self.gp_scratch.reg(0),
            0,
            width,
            src,
        )
    }
}
