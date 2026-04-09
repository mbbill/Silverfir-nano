//! x86_64 backend: instruction lowering methods for X86_64Backend.

use crate::{
    error::WasmError,
    vm::machine::machine_ir::{
        MachineCompareKind, MachineConvertOp, MachineFloatBinaryOp, MachineFloatUnaryOp,
        MachineFloatWidth, MachineIndexExtend, MachineInst, MachineInstKind, MachineIntBinaryOp,
        MachineIntUnaryOp, MachineIntWidth, MachineLoadExtension, MachineMemWidth, MachineReg,
        MachineShiftOp, MachineSign, MachineStorageType, MachineTrapKind, MachineValue,
    },
};

use super::{
    abi::{map_fixed_reg, C_ARG0, C_ARG1, C_ARG2},
    backend::X86_64Backend,
    callconv,
    enc::{self, Cc},
    fusion::map_int_cond,
    helpers::x86_64_saturating_trunc,
    reg::X86Reg,
};
use crate::vm::arch::common::helpers::convert_result_float_width;

use crate::vm::machine::machine_ir::{MachineAddr, MACHINE_CTX_REG};

/// Map a `MachineConvertOp` to the u32 op code consumed by the runtime
/// trunc/saturating-trunc helpers. Returns `u32::MAX` for ops that do not
/// go through the helper path.
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

impl<'a> X86_64Backend<'a> {
    pub(super) fn lower_inst_dispatch(&mut self, inst: &MachineInst) -> Result<(), WasmError> {
        match &inst.kind {
            MachineInstKind::Move { dst, src, ty, .. } => self.lower_move(*ty, *dst, *src),
            MachineInstKind::FloatConst { width, dst, bits } => {
                self.lower_float_const(*width, *dst, *bits)
            }
            MachineInstKind::Load {
                dst,
                addr,
                width,
                extension,
                ..
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
            MachineInstKind::CallExternal(call) => {
                self.lower_call_external(call.metadata.0 as usize)
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
            MachineInstKind::BitfieldExtractU { width, dst, src, lsb, bits } => {
                self.lower_bitfield_extract_u(*width, *dst, *src, *lsb, *bits)
            }
            MachineInstKind::IntBinaryShifted { width, op, dst, lhs, rhs, shift, amount } => {
                self.lower_int_binary_shifted(*width, *op, *dst, *lhs, *rhs, *shift, *amount)
            }
            MachineInstKind::TestBits { width, kind, dst, src, mask } => {
                self.lower_test_bits(*width, *kind, *dst, *src, *mask)
            }
            MachineInstKind::MemoryGrow { mem_idx, dst, delta } => {
                self.lower_memory_grow(*mem_idx, *dst, *delta)
            }
            MachineInstKind::MemoryFill { mem_idx, dest, val, len } => {
                self.lower_memory_fill(*mem_idx, *dest, *val, *len)
            }
            MachineInstKind::MemoryCopy { dst_mem, src_mem, dest, src, len } => {
                self.lower_memory_copy(*dst_mem, *src_mem, *dest, *src, *len)
            }
            MachineInstKind::MemoryInit { mem_idx, data_idx, dest, src, len } => {
                self.lower_memory_init(*mem_idx, *data_idx, *dest, *src, *len)
            }
            MachineInstKind::DataDrop { data_idx } => self.lower_data_drop(*data_idx),
            MachineInstKind::TableGrow { table_idx, dst, init_val, delta } => {
                self.lower_table_grow(*table_idx, *dst, *init_val, *delta)
            }
            MachineInstKind::TableFill { table_idx, start, val, len } => {
                self.lower_table_fill(*table_idx, *start, *val, *len)
            }
            MachineInstKind::TableCopy { dst_tbl, src_tbl, dest, src, len } => {
                self.lower_table_copy(*dst_tbl, *src_tbl, *dest, *src, *len)
            }
            MachineInstKind::TableInit { table_idx, elem_idx, dest, src, len } => {
                self.lower_table_init(*table_idx, *elem_idx, *dest, *src, *len)
            }
            MachineInstKind::ElemDrop { elem_idx } => self.lower_elem_drop(*elem_idx),
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
                            MachineFloatWidth::F32 => {
                                enc::movss_rr(&mut self.core.text, dst_fp, src_fp)
                            }
                            MachineFloatWidth::F64 => {
                                enc::movsd_rr(&mut self.core.text, dst_fp, src_fp)
                            }
                        };
                    }
                    self.core.set_fp_reg_width(dst, width)?;
                    Ok(())
                }
                MachineValue::Reg(src_reg) => {
                    let src_gp = self.map_gp_reg(src_reg)?;
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movd_xmm_r32(&mut self.core.text, dst_fp, src_gp)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movq_xmm_r64(&mut self.core.text, dst_fp, src_gp)
                        }
                    };
                    self.core.set_fp_reg_width(dst, width)?;
                    Ok(())
                }
                MachineValue::Imm64(value) => {
                    let scratch = self.gp_scratch.scoped_alloc().detach();
                    self.materialize_u64(*scratch, value);
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movd_xmm_r32(&mut self.core.text, dst_fp, *scratch)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movq_xmm_r64(&mut self.core.text, dst_fp, *scratch)
                        }
                    };
                    self.core.set_fp_reg_width(dst, width)?;
                    Ok(())
                }
                MachineValue::ReservedReg(reg) => Err(WasmError::internal(alloc::format!(
                    "x86_64 Move cannot consume reserved cache register {}",
                    reg.0
                ))),
            }
        } else {
            let dst_gp = self.map_gp_reg(dst)?;
            match src {
                MachineValue::Reg(src_reg) if self.core.is_fp_reg(src_reg) => {
                    let src_fp = self.map_fp_reg(src_reg)? as u8;
                    match self.core.fp_reg_width(src_reg)? {
                        MachineFloatWidth::F32 => {
                            enc::movd_r32_xmm(&mut self.core.text, dst_gp, src_fp)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movq_r64_xmm(&mut self.core.text, dst_gp, src_fp)
                        }
                    };
                    Ok(())
                }
                MachineValue::Reg(src_reg) => {
                    let src_gp = self.map_gp_reg(src_reg)?;
                    if dst_gp != src_gp {
                        self.emit_gp_move_ty(ty, dst_gp, src_gp)?;
                    }
                    Ok(())
                }
                MachineValue::Imm64(value) => {
                    self.materialize_u64(dst_gp, value);
                    Ok(())
                }
                MachineValue::ReservedReg(reg) => Err(WasmError::internal(alloc::format!(
                    "x86_64 Move cannot consume reserved cache register {}",
                    reg.0
                ))),
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
            let scratch = self.gp_scratch.scoped_alloc().detach();
            self.materialize_u64(*scratch, imm);
            match width {
                MachineFloatWidth::F32 => enc::movd_xmm_r32(&mut self.core.text, dst_fp, *scratch),
                MachineFloatWidth::F64 => enc::movq_xmm_r64(&mut self.core.text, dst_fp, *scratch),
            };
        }
        self.core.set_fp_reg_width(dst, width)?;
        Ok(())
    }

    // ── Load / Store ─────────────────────────────────────────────────────────

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
        let scratch0 = self.gp_scratch.scoped_alloc().detach();
        let scratch1 = self.gp_scratch.scoped_alloc().detach();
        self.lower_store_to_with_scratch(base, disp, width, src, *scratch0, *scratch1)
    }

    fn lower_store_to_with_scratch(
        &mut self,
        base: X86Reg,
        disp: i32,
        width: MachineMemWidth,
        src: MachineValue,
        scratch0: X86Reg,
        scratch1: X86Reg,
    ) -> Result<(), WasmError> {
        // FP register source
        if let MachineValue::Reg(src_reg) = src {
            if self.core.is_fp_reg(src_reg) {
                let src_fp = self.map_fp_reg(src_reg)? as u8;
                match width {
                    MachineMemWidth::U32 => {
                        enc::movss_store(&mut self.core.text, base, disp, src_fp)
                    }
                    MachineMemWidth::U64 => {
                        enc::movsd_store(&mut self.core.text, base, disp, src_fp)
                    }
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

        // Indexed stores compute the effective address into gp_scratch[0]
        // before calling lower_store_to(). Do not reuse that register to
        // materialize the source or the base address will be lost.
        let materialize_scratch = if base == scratch0 { scratch1 } else { scratch0 };
        let src_gp = self.materialize_value(materialize_scratch, src)?;
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
        let scratch = self.gp_scratch.scoped_alloc().detach();
        let src = self.materialize_value(*scratch, src)?;
        match (width, op) {
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
                    enc::mov_rr_32(&mut self.core.text, dst, src);
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
                let scratch0 = self.gp_scratch.scoped_alloc().detach();
                let scratch1 = self.gp_scratch.scoped_alloc().detach();
                // Try immediate form: dst = lhs OP imm32
                if let MachineValue::Imm64(imm_val) = rhs {
                    let imm = imm_val as i64 as i32;
                    if imm as i64 == imm_val as i64
                        || (width == MachineIntWidth::I32 && imm_val as u32 as i32 == imm)
                    {
                        let lhs_gp = self.materialize_value(*scratch0, lhs)?;
                        if dst != lhs_gp {
                            self.emit_gp_move_width(width, dst, lhs_gp);
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
                let lhs_gp = self.materialize_value(*scratch0, lhs)?;
                let rhs_gp = self.materialize_value(*scratch1, rhs)?;
                // Handle aliasing: if dst == rhs_gp but dst != lhs_gp,
                // mov dst, lhs would clobber rhs before the operation.
                if dst == rhs_gp && dst != lhs_gp {
                    if op == MachineIntBinaryOp::Sub {
                        // Sub is not commutative: compute in scratch
                        self.emit_gp_move_width(width, *scratch0, lhs_gp);
                        match width {
                            MachineIntWidth::I64 => {
                                enc::sub_rr_64(&mut self.core.text, *scratch0, rhs_gp)
                            }
                            MachineIntWidth::I32 => {
                                enc::sub_rr_32(&mut self.core.text, *scratch0, rhs_gp)
                            }
                        };
                        self.emit_gp_move_width(width, dst, *scratch0);
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
                        self.emit_gp_move_width(width, dst, lhs_gp);
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
                let scratch0 = self.gp_scratch.scoped_alloc().detach();
                let scratch1 = self.gp_scratch.scoped_alloc().detach();
                let lhs_gp = self.materialize_value(*scratch0, lhs)?;
                let rhs_gp = self.materialize_value(*scratch1, rhs)?;
                if dst == rhs_gp && dst != lhs_gp {
                    // IMUL is commutative: dst already has rhs, just mul by lhs
                    match width {
                        MachineIntWidth::I64 => enc::imul_rr_64(&mut self.core.text, dst, lhs_gp),
                        MachineIntWidth::I32 => enc::imul_rr_32(&mut self.core.text, dst, lhs_gp),
                    };
                } else {
                    if dst != lhs_gp {
                        self.emit_gp_move_width(width, dst, lhs_gp);
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
        // For immediate shift amounts, use the imm8 form (no RCX needed).
        if let MachineValue::Imm64(amount) = rhs {
            let scratch0 = self.gp_scratch.scoped_alloc().detach();
            let lhs_gp = self.materialize_value(*scratch0, lhs)?;
            if dst != lhs_gp {
                self.emit_gp_move_width(width, dst, lhs_gp);
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
        // Variable shift: RCX is scratch-only on x86_64, so own it explicitly
        // rather than borrowing a dynamic register and restoring it later.
        let rcx = self.gp_scratch.claim_rcx().detach();
        let scratch0 = self.gp_scratch.scoped_alloc().detach();
        let lhs_gp = self.materialize_value(*scratch0, lhs)?;
        let rhs_gp = self.materialize_value(*rcx, rhs)?;
        if rhs_gp == dst && dst != lhs_gp {
            if rhs_gp != *rcx {
                enc::mov_rr_64(&mut self.core.text, *rcx, rhs_gp);
            }
            if dst != lhs_gp {
                self.emit_gp_move_width(width, dst, lhs_gp);
            }
        } else {
            if dst != lhs_gp {
                self.emit_gp_move_width(width, dst, lhs_gp);
            }
            if rhs_gp != *rcx {
                enc::mov_rr_64(&mut self.core.text, *rcx, rhs_gp);
            }
        }
        match (width, op) {
            (MachineIntWidth::I64, MachineIntBinaryOp::Shl) => {
                enc::shl_cl_64(&mut self.core.text, dst)
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Shl) => {
                enc::shl_cl_32(&mut self.core.text, dst)
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::ShrS) => {
                enc::sar_cl_64(&mut self.core.text, dst)
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::ShrS) => {
                enc::sar_cl_32(&mut self.core.text, dst)
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::ShrU) => {
                enc::shr_cl_64(&mut self.core.text, dst)
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::ShrU) => {
                enc::shr_cl_32(&mut self.core.text, dst)
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Rotl) => {
                enc::rol_cl_64(&mut self.core.text, dst)
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Rotl) => {
                enc::rol_cl_32(&mut self.core.text, dst)
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Rotr) => {
                enc::ror_cl_64(&mut self.core.text, dst)
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Rotr) => {
                enc::ror_cl_32(&mut self.core.text, dst)
            }
            _ => unreachable!(),
        };
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
        // div/idiv implicitly uses RAX and RDX. On x86_64 these are scratch-only
        // backend registers, so claim ownership explicitly before touching them.
        let rax = self.gp_scratch.claim_rax().detach();
        let rdx = self.gp_scratch.claim_rdx().detach();
        let scratch0 = self.gp_scratch.scoped_alloc().detach();

        let lhs_gp = self.materialize_value(*rax, lhs)?;
        if lhs_gp != *rax {
            enc::mov_rr_64(&mut self.core.text, *rax, lhs_gp);
        }
        let rhs_gp = self.materialize_value(*scratch0, rhs)?;
        let divisor = if rhs_gp == *rax || rhs_gp == *rdx {
            enc::mov_rr_64(&mut self.core.text, *scratch0, rhs_gp);
            *scratch0
        } else {
            rhs_gp
        };

        // Division-by-zero check: divisor == 0 => trap
        enc::test_rr_64(&mut self.core.text, divisor, divisor);
        let div_zero_label = self
            .core
            .ensure_trap_label(MachineTrapKind::IntegerDivideByZero);
        self.emit_jcc(Cc::E, div_zero_label);

        match op {
            MachineIntBinaryOp::DivS => {
                // Signed overflow check: MIN / -1 => IntegerOverflow trap
                let not_min = self.core.new_label();
                match width {
                    MachineIntWidth::I32 => {
                        enc::cmp_ri_32(&mut self.core.text, *rax, i32::MIN);
                    }
                    MachineIntWidth::I64 => {
                        self.materialize_u64(*rdx, i64::MIN as u64);
                        enc::cmp_rr_64(&mut self.core.text, *rax, *rdx);
                    }
                };
                self.emit_jcc(Cc::NE, not_min);
                // Compare divisor against -1 using matching width
                match width {
                    MachineIntWidth::I32 => enc::cmp_ri_32(&mut self.core.text, divisor, -1),
                    MachineIntWidth::I64 => enc::cmp_ri_64(&mut self.core.text, divisor, -1),
                };
                let overflow_label = self
                    .core
                    .ensure_trap_label(MachineTrapKind::IntegerOverflow);
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
                        enc::cmp_ri_32(&mut self.core.text, *rax, i32::MIN);
                    }
                    MachineIntWidth::I64 => {
                        self.materialize_u64(*rdx, i64::MIN as u64);
                        enc::cmp_rr_64(&mut self.core.text, *rax, *rdx);
                    }
                };
                self.emit_jcc(Cc::NE, not_min);
                match width {
                    MachineIntWidth::I32 => enc::cmp_ri_32(&mut self.core.text, divisor, -1),
                    MachineIntWidth::I64 => enc::cmp_ri_64(&mut self.core.text, divisor, -1),
                };
                self.emit_jcc(Cc::NE, not_min);
                // MIN % -1 = 0
                enc::xor_rr_32(&mut self.core.text, *rdx, *rdx);
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
                enc::xor_rr_32(&mut self.core.text, *rdx, *rdx);
                match width {
                    MachineIntWidth::I64 => enc::div_rm_64(&mut self.core.text, divisor),
                    MachineIntWidth::I32 => enc::div_rm_32(&mut self.core.text, divisor),
                };
            }
            _ => unreachable!(),
        }

        // Result: quotient in RAX, remainder in RDX
        let result_reg = match op {
            MachineIntBinaryOp::DivS | MachineIntBinaryOp::DivU => *rax,
            MachineIntBinaryOp::RemS | MachineIntBinaryOp::RemU => *rdx,
            _ => unreachable!(),
        };
        if dst != result_reg {
            self.emit_gp_move_width(width, dst, result_reg);
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
        let scratch0 = self.gp_scratch.scoped_alloc().detach();
        let scratch1 = self.gp_scratch.scoped_alloc().detach();
        // Immediate form
        if let MachineValue::Imm64(imm_val) = rhs {
            let imm = imm_val as i64 as i32;
            if imm as i64 == imm_val as i64
                || (width == MachineIntWidth::I32 && imm_val as u32 as i32 == imm)
            {
                let lhs_gp = self.materialize_value(*scratch0, lhs)?;
                match width {
                    MachineIntWidth::I64 => enc::cmp_ri_64(&mut self.core.text, lhs_gp, imm),
                    MachineIntWidth::I32 => enc::cmp_ri_32(&mut self.core.text, lhs_gp, imm),
                };
                return Ok(());
            }
        }
        let lhs_gp = self.materialize_value(*scratch0, lhs)?;
        let rhs_gp = self.materialize_value(*scratch1, rhs)?;
        match width {
            MachineIntWidth::I64 => enc::cmp_rr_64(&mut self.core.text, lhs_gp, rhs_gp),
            MachineIntWidth::I32 => enc::cmp_rr_32(&mut self.core.text, lhs_gp, rhs_gp),
        };
        Ok(())
    }

    /// Emit TEST (bitwise AND setting flags, discarding result).
    pub(super) fn lower_tst_values(
        &mut self,
        width: MachineIntWidth,
        src: MachineValue,
        mask: MachineValue,
    ) -> Result<(), WasmError> {
        let scratch0 = self.gp_scratch.scoped_alloc().detach();
        let scratch1 = self.gp_scratch.scoped_alloc().detach();
        let src_gp = self.materialize_value(*scratch0, src)?;
        match mask {
            MachineValue::Imm64(imm_val) => {
                let imm = imm_val as i64 as i32;
                if imm as i64 == imm_val as i64
                    || (width == MachineIntWidth::I32 && imm_val as u32 as i32 == imm)
                {
                    match width {
                        MachineIntWidth::I64 => enc::test_ri_64(&mut self.core.text, src_gp, imm),
                        MachineIntWidth::I32 => enc::test_ri_32(&mut self.core.text, src_gp, imm),
                    }
                } else {
                    let mask_gp =
                        self.materialize_value(*scratch1, MachineValue::Imm64(imm_val))?;
                    match width {
                        MachineIntWidth::I64 => {
                            enc::test_rr_64(&mut self.core.text, src_gp, mask_gp)
                        }
                        MachineIntWidth::I32 => {
                            enc::test_rr_32(&mut self.core.text, src_gp, mask_gp)
                        }
                    }
                }
            }
            MachineValue::Reg(_) => {
                let mask_gp = self.materialize_value(*scratch1, mask)?;
                match width {
                    MachineIntWidth::I64 => enc::test_rr_64(&mut self.core.text, src_gp, mask_gp),
                    MachineIntWidth::I32 => enc::test_rr_32(&mut self.core.text, src_gp, mask_gp),
                }
            }
            MachineValue::ReservedReg(reg) => {
                return Err(WasmError::internal(alloc::format!(
                    "x86_64 TestBits cannot read reserved cache register {}",
                    reg.0
                )));
            }
        }
        Ok(())
    }

    // ── Bitfield extract (decomposed to SHR + AND) ────────────────────────────

    fn lower_bitfield_extract_u(
        &mut self,
        width: MachineIntWidth,
        dst: MachineReg,
        src: MachineReg,
        lsb: u8,
        bits: u8,
    ) -> Result<(), WasmError> {
        let dst = self.map_gp_reg(dst)?;
        let src = self.map_gp_reg(src)?;
        // dst = (src >> lsb) & ((1 << bits) - 1)
        if dst != src {
            self.emit_gp_move_width(width, dst, src);
        }
        if lsb > 0 {
            match width {
                MachineIntWidth::I64 => enc::shr_imm_64(&mut self.core.text, dst, lsb),
                MachineIntWidth::I32 => enc::shr_imm_32(&mut self.core.text, dst, lsb),
            }
        }
        let mask = (1u64 << bits) - 1;
        match width {
            MachineIntWidth::I64 => {
                // `and r64, imm32` sign-extends its immediate, so only use the
                // immediate form when it encodes the exact 64-bit mask.
                let imm = mask as i64 as i32;
                if imm as i64 as u64 == mask {
                    enc::and_ri_64(&mut self.core.text, dst, imm);
                } else {
                    let scratch = self.gp_scratch.scoped_alloc().detach();
                    let mask_gp = self.materialize_value(*scratch, MachineValue::Imm64(mask))?;
                    enc::and_rr_64(&mut self.core.text, dst, mask_gp);
                }
            }
            MachineIntWidth::I32 => {
                let imm = mask as u32 as i32;
                enc::and_ri_32(&mut self.core.text, dst, imm);
            }
        }
        Ok(())
    }

    // ── Shifted-register binary (decomposed to shift + op) ──────────────────

    fn lower_int_binary_shifted(
        &mut self,
        width: MachineIntWidth,
        op: MachineIntBinaryOp,
        dst: MachineReg,
        lhs: MachineReg,
        rhs: MachineReg,
        shift: MachineShiftOp,
        amount: u8,
    ) -> Result<(), WasmError> {
        // Decompose: dst = lhs OP (rhs SHIFT amount)
        // Step 1: shift rhs into scratch
        let dst = self.map_gp_reg(dst)?;
        let lhs = self.map_gp_reg(lhs)?;
        let rhs = self.map_gp_reg(rhs)?;
        let scratch = self.gp_scratch.scoped_alloc().detach();
        if *scratch != rhs {
            self.emit_gp_move_width(width, *scratch, rhs);
        }
        match (width, shift) {
            (MachineIntWidth::I64, MachineShiftOp::Lsl) => {
                enc::shl_imm_64(&mut self.core.text, *scratch, amount)
            }
            (MachineIntWidth::I32, MachineShiftOp::Lsl) => {
                enc::shl_imm_32(&mut self.core.text, *scratch, amount)
            }
            (MachineIntWidth::I64, MachineShiftOp::Lsr) => {
                enc::shr_imm_64(&mut self.core.text, *scratch, amount)
            }
            (MachineIntWidth::I32, MachineShiftOp::Lsr) => {
                enc::shr_imm_32(&mut self.core.text, *scratch, amount)
            }
            (MachineIntWidth::I64, MachineShiftOp::Asr) => {
                enc::sar_imm_64(&mut self.core.text, *scratch, amount)
            }
            (MachineIntWidth::I32, MachineShiftOp::Asr) => {
                enc::sar_imm_32(&mut self.core.text, *scratch, amount)
            }
        }
        // Step 2: dst = lhs OP scratch
        if dst != lhs {
            self.emit_gp_move_width(width, dst, lhs);
        }
        match (width, op) {
            (MachineIntWidth::I64, MachineIntBinaryOp::Add) => {
                enc::add_rr_64(&mut self.core.text, dst, *scratch)
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Add) => {
                enc::add_rr_32(&mut self.core.text, dst, *scratch)
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Sub) => {
                enc::sub_rr_64(&mut self.core.text, dst, *scratch)
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Sub) => {
                enc::sub_rr_32(&mut self.core.text, dst, *scratch)
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::And) => {
                enc::and_rr_64(&mut self.core.text, dst, *scratch)
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::And) => {
                enc::and_rr_32(&mut self.core.text, dst, *scratch)
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Or) => {
                enc::or_rr_64(&mut self.core.text, dst, *scratch)
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Or) => {
                enc::or_rr_32(&mut self.core.text, dst, *scratch)
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Xor) => {
                enc::xor_rr_64(&mut self.core.text, dst, *scratch)
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Xor) => {
                enc::xor_rr_32(&mut self.core.text, dst, *scratch)
            }
            _ => {
                return Err(WasmError::internal(alloc::format!(
                    "IntBinaryShifted: unsupported op {:?}",
                    op
                )))
            }
        }
        Ok(())
    }

    // ── Test bits (TEST + SETCC) ────────────────────────────────────────────

    fn lower_test_bits(
        &mut self,
        width: MachineIntWidth,
        kind: MachineCompareKind,
        dst: MachineReg,
        src: MachineReg,
        mask: MachineValue,
    ) -> Result<(), WasmError> {
        let dst = self.map_gp_reg(dst)?;
        let src = self.map_gp_reg(src)?;
        let scratch = self.gp_scratch.scoped_alloc().detach();
        // Emit TEST to set flags.
        match mask {
            MachineValue::Imm64(imm_val) => {
                let imm = imm_val as i64 as i32;
                if imm as i64 == imm_val as i64
                    || (width == MachineIntWidth::I32 && imm_val as u32 as i32 == imm)
                {
                    match width {
                        MachineIntWidth::I64 => enc::test_ri_64(&mut self.core.text, src, imm),
                        MachineIntWidth::I32 => enc::test_ri_32(&mut self.core.text, src, imm),
                    }
                } else {
                    // Doesn't fit i32 — materialize and use register form.
                    self.materialize_u64(*scratch, imm_val);
                    match width {
                        MachineIntWidth::I64 => enc::test_rr_64(&mut self.core.text, src, *scratch),
                        MachineIntWidth::I32 => enc::test_rr_32(&mut self.core.text, src, *scratch),
                    }
                }
            }
            MachineValue::Reg(mask_reg) => {
                let mask_gp = self.map_gp_reg(mask_reg)?;
                match width {
                    MachineIntWidth::I64 => enc::test_rr_64(&mut self.core.text, src, mask_gp),
                    MachineIntWidth::I32 => enc::test_rr_32(&mut self.core.text, src, mask_gp),
                }
            }
            MachineValue::ReservedReg(reg) => {
                return Err(WasmError::internal(alloc::format!(
                    "x86_64 TestBits mask cannot be reserved cache register {}",
                    reg.0
                )));
            }
        }
        // TST sets Z flag: Eq → ZF=1 (test was zero), Ne → ZF=0 (test was nonzero).
        let cc = match kind {
            MachineCompareKind::Eq => Cc::E,
            MachineCompareKind::Ne => Cc::NE,
            _ => {
                return Err(WasmError::internal(alloc::format!(
                    "TestBits: unsupported compare kind {:?}",
                    kind
                )))
            }
        };
        enc::setcc(&mut self.core.text, cc, dst);
        enc::movzx_r32_r8(&mut self.core.text, dst, dst);
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
                    let gp0 = self.gp_scratch.scoped_alloc().detach();
                    let gp1 = self.gp_scratch.scoped_alloc().detach();
                    let fp0 = self.fp_scratch.scoped_alloc().detach();
                    let fp1 = self.fp_scratch.scoped_alloc().detach();
                    let false_label = self.core.new_label();
                    let done = self.core.new_label();
                    // Wasm select conditions are i32 values; ignore any stale
                    // upper half that may remain in a GpWord carrier.
                    enc::test_rr_32(&mut self.core.text, cond_gp, cond_gp);
                    self.emit_jcc(Cc::E, false_label);
                    let true_fp = self.prepare_float_operand(width, on_true, *gp0, *fp0)?;
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
                    let false_fp = self.prepare_float_operand(width, on_false, *gp1, *fp1)?;
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
                MachineValue::ReservedReg(reg) => Err(WasmError::internal(alloc::format!(
                    "x86_64 Select condition cannot be reserved cache register {}",
                    reg.0
                ))),
            }
        } else {
            if let MachineValue::Imm64(value) = cond {
                let selected = if value != 0 { on_true } else { on_false };
                return self.lower_move(ty, dst, selected);
            }
            let dst = self.map_gp_reg(dst)?;
            // Materialize operands BEFORE testing the condition, because
            // materialize_value may clobber flags (e.g. xor reg,reg for zero).
            let scratch0 = self.gp_scratch.scoped_alloc().detach();
            let scratch1 = self.gp_scratch.scoped_alloc().detach();
            let true_reg = self.materialize_value(*scratch0, on_true)?;
            let false_reg = self.materialize_value(*scratch1, on_false)?;
            let cond_gp = match cond {
                MachineValue::Reg(reg) => self.map_gp_reg(reg)?,
                _ => unreachable!(),
            };
            // Wasm select conditions are i32 values; ignore any stale
            // upper half that may remain in a GpWord carrier.
            enc::test_rr_32(&mut self.core.text, cond_gp, cond_gp);
            if dst == true_reg && dst != false_reg {
                self.emit_gp_cmov_ty(ty, Cc::E, dst, false_reg)?;
            } else if dst == false_reg {
                self.emit_gp_cmov_ty(ty, Cc::NE, dst, true_reg)?;
            } else {
                self.emit_gp_move_ty(ty, dst, false_reg)?;
                self.emit_gp_cmov_ty(ty, Cc::NE, dst, true_reg)?;
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
        let gp0 = self.gp_scratch.scoped_alloc().detach();
        let gp1 = self.gp_scratch.scoped_alloc().detach();
        let fp0 = self.fp_scratch.scoped_alloc().detach();
        let fp1 = self.fp_scratch.scoped_alloc().detach();
        let src_gp = self.materialize_value(*gp0, src)?;
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
                        MachineFloatWidth::F32 => {
                            enc::movd_xmm_r32(&mut self.core.text, dst_fp, src_gp)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movq_xmm_r64(&mut self.core.text, dst_fp, src_gp)
                        }
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
                let src_fp = self.prepare_float_operand(MachineFloatWidth::F32, src, *gp0, *fp0)?;
                if self.core.is_fp_reg(dst) {
                    let dst_fp = self.map_fp_reg(dst)? as u8;
                    self.core.set_fp_reg_width(dst, MachineFloatWidth::F64)?;
                    enc::cvtss2sd(&mut self.core.text, dst_fp, src_fp as u8);
                } else {
                    enc::cvtss2sd(&mut self.core.text, *fp1 as u8, src_fp as u8);
                    let dst_gp = self.map_gp_reg(dst)?;
                    enc::movq_r64_xmm(&mut self.core.text, dst_gp, *fp1 as u8);
                }
            }
            MachineConvertOp::F32DemoteF64 => {
                let src_fp = self.prepare_float_operand(MachineFloatWidth::F64, src, *gp0, *fp0)?;
                if self.core.is_fp_reg(dst) {
                    let dst_fp = self.map_fp_reg(dst)? as u8;
                    self.core.set_fp_reg_width(dst, MachineFloatWidth::F32)?;
                    enc::cvtsd2ss(&mut self.core.text, dst_fp, src_fp as u8);
                } else {
                    enc::cvtsd2ss(&mut self.core.text, *fp1 as u8, src_fp as u8);
                    let dst_gp = self.map_gp_reg(dst)?;
                    enc::movd_r32_xmm(&mut self.core.text, dst_gp, *fp1 as u8);
                }
            }
            // Int -> Float conversions
            MachineConvertOp::F32ConvertI32S => {
                // CVTSI2SS xmm, r32
                let dst_fp = self.dst_float_reg(dst, MachineFloatWidth::F32, *fp1 as u8)?;
                enc::cvtsi2ss_r32(&mut self.core.text, dst_fp, src_gp);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F32, dst_fp)?;
            }
            MachineConvertOp::F32ConvertI32U => {
                // Zero-extend to 64-bit first for unsigned interpretation
                enc::mov_rr_32(&mut self.core.text, *gp0, src_gp);
                let dst_fp = self.dst_float_reg(dst, MachineFloatWidth::F32, *fp1 as u8)?;
                enc::cvtsi2ss_r64(&mut self.core.text, dst_fp, *gp0);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F32, dst_fp)?;
            }
            MachineConvertOp::F32ConvertI64S => {
                let dst_fp = self.dst_float_reg(dst, MachineFloatWidth::F32, *fp1 as u8)?;
                enc::cvtsi2ss_r64(&mut self.core.text, dst_fp, src_gp);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F32, dst_fp)?;
            }
            MachineConvertOp::F32ConvertI64U => {
                // x86_64 has no unsigned int-to-float instruction.
                // For values that fit in i64 (bit 63 = 0), use signed conversion.
                // For values with bit 63 set, shift right by 1, convert, then double.
                let dst_fp = self.dst_float_reg(dst, MachineFloatWidth::F32, *fp1 as u8)?;
                enc::test_rr_64(&mut self.core.text, src_gp, src_gp);
                let large = self.core.new_label();
                self.emit_jcc(Cc::S, large); // JS = sign flag set = bit 63 is 1
                                             // Small path: fits in i64
                enc::cvtsi2ss_r64(&mut self.core.text, dst_fp, src_gp);
                let done = self.core.new_label();
                self.emit_jmp(done);
                // Large path: bit 63 set
                self.core.bind_label(large);
                enc::mov_rr_64(&mut self.core.text, *gp0, src_gp);
                enc::mov_rr_64(&mut self.core.text, *gp1, src_gp);
                enc::shr_imm_64(&mut self.core.text, *gp0, 1); // src >> 1
                enc::and_ri_32(&mut self.core.text, *gp1, 1); // src & 1 (preserve LSB)
                enc::or_rr_64(&mut self.core.text, *gp0, *gp1); // (src >> 1) | (src & 1)
                enc::cvtsi2ss_r64(&mut self.core.text, dst_fp, *gp0);
                enc::addss(&mut self.core.text, dst_fp, dst_fp); // double it
                self.core.bind_label(done);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F32, dst_fp)?;
            }
            MachineConvertOp::F64ConvertI32S => {
                let dst_fp = self.dst_float_reg(dst, MachineFloatWidth::F64, *fp1 as u8)?;
                enc::cvtsi2sd_r32(&mut self.core.text, dst_fp, src_gp);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F64, dst_fp)?;
            }
            MachineConvertOp::F64ConvertI32U => {
                enc::mov_rr_32(&mut self.core.text, *gp0, src_gp);
                let dst_fp = self.dst_float_reg(dst, MachineFloatWidth::F64, *fp1 as u8)?;
                enc::cvtsi2sd_r64(&mut self.core.text, dst_fp, *gp0);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F64, dst_fp)?;
            }
            MachineConvertOp::F64ConvertI64S => {
                let dst_fp = self.dst_float_reg(dst, MachineFloatWidth::F64, *fp1 as u8)?;
                enc::cvtsi2sd_r64(&mut self.core.text, dst_fp, src_gp);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F64, dst_fp)?;
            }
            MachineConvertOp::F64ConvertI64U => {
                let dst_fp = self.dst_float_reg(dst, MachineFloatWidth::F64, *fp1 as u8)?;
                enc::test_rr_64(&mut self.core.text, src_gp, src_gp);
                let large = self.core.new_label();
                self.emit_jcc(Cc::S, large);
                enc::cvtsi2sd_r64(&mut self.core.text, dst_fp, src_gp);
                let done = self.core.new_label();
                self.emit_jmp(done);
                self.core.bind_label(large);
                enc::mov_rr_64(&mut self.core.text, *gp0, src_gp);
                enc::mov_rr_64(&mut self.core.text, *gp1, src_gp);
                enc::shr_imm_64(&mut self.core.text, *gp0, 1);
                enc::and_ri_32(&mut self.core.text, *gp1, 1);
                enc::or_rr_64(&mut self.core.text, *gp0, *gp1);
                enc::cvtsi2sd_r64(&mut self.core.text, dst_fp, *gp0);
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
                if src_gp != *gp0 {
                    drop(gp0);
                }
                drop(gp1);
                drop(fp0);
                drop(fp1);
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
                if src_gp != *gp0 {
                    drop(gp0);
                }
                drop(gp1);
                drop(fp0);
                drop(fp1);
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
        let gp_scratch = self.gp_scratch.scoped_alloc().detach();
        let fp0 = self.fp_scratch.scoped_alloc().detach();
        let fp1 = self.fp_scratch.scoped_alloc().detach();
        let src_fp = self.prepare_float_operand(width, src, *gp_scratch, *fp0)?;
        let result_fp = if self.core.is_fp_reg(dst) {
            let dst_fp = self.map_fp_reg(dst)? as u8;
            self.core.set_fp_reg_width(dst, width)?;
            dst_fp
        } else {
            *fp1 as u8
        };
        match op {
            MachineFloatUnaryOp::Sqrt => {
                match width {
                    MachineFloatWidth::F32 => {
                        enc::sqrtss(&mut self.core.text, result_fp, src_fp as u8)
                    }
                    MachineFloatWidth::F64 => {
                        enc::sqrtsd(&mut self.core.text, result_fp, src_fp as u8)
                    }
                };
            }
            MachineFloatUnaryOp::Ceil => {
                match width {
                    MachineFloatWidth::F32 => enc::roundss(
                        &mut self.core.text,
                        result_fp,
                        src_fp as u8,
                        enc::ROUND_CEIL,
                    ),
                    MachineFloatWidth::F64 => enc::roundsd(
                        &mut self.core.text,
                        result_fp,
                        src_fp as u8,
                        enc::ROUND_CEIL,
                    ),
                };
            }
            MachineFloatUnaryOp::Floor => {
                match width {
                    MachineFloatWidth::F32 => enc::roundss(
                        &mut self.core.text,
                        result_fp,
                        src_fp as u8,
                        enc::ROUND_FLOOR,
                    ),
                    MachineFloatWidth::F64 => enc::roundsd(
                        &mut self.core.text,
                        result_fp,
                        src_fp as u8,
                        enc::ROUND_FLOOR,
                    ),
                };
            }
            MachineFloatUnaryOp::Trunc => {
                match width {
                    MachineFloatWidth::F32 => enc::roundss(
                        &mut self.core.text,
                        result_fp,
                        src_fp as u8,
                        enc::ROUND_TRUNC,
                    ),
                    MachineFloatWidth::F64 => enc::roundsd(
                        &mut self.core.text,
                        result_fp,
                        src_fp as u8,
                        enc::ROUND_TRUNC,
                    ),
                };
            }
            MachineFloatUnaryOp::Nearest => {
                match width {
                    MachineFloatWidth::F32 => enc::roundss(
                        &mut self.core.text,
                        result_fp,
                        src_fp as u8,
                        enc::ROUND_NEAREST,
                    ),
                    MachineFloatWidth::F64 => enc::roundsd(
                        &mut self.core.text,
                        result_fp,
                        src_fp as u8,
                        enc::ROUND_NEAREST,
                    ),
                };
            }
            MachineFloatUnaryOp::Abs => {
                // Clear sign bit: AND with mask.
                let mask_xmm = if result_fp != *fp0 as u8 {
                    *fp0 as u8
                } else {
                    *fp1 as u8
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
                self.materialize_u64(*gp_scratch, mask);
                enc::movq_xmm_r64(&mut self.core.text, mask_xmm, *gp_scratch);
                enc::andpd(&mut self.core.text, result_fp, mask_xmm);
            }
            MachineFloatUnaryOp::Neg => {
                // Flip sign bit: XOR with mask.
                let mask_xmm = if result_fp != *fp0 as u8 {
                    *fp0 as u8
                } else {
                    *fp1 as u8
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
                self.materialize_u64(*gp_scratch, mask);
                enc::movq_xmm_r64(&mut self.core.text, mask_xmm, *gp_scratch);
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
        let gp0 = self.gp_scratch.scoped_alloc().detach();
        let gp1 = self.gp_scratch.scoped_alloc().detach();
        let fp0 = self.fp_scratch.scoped_alloc().detach();
        let fp1 = self.fp_scratch.scoped_alloc().detach();
        let lhs_fp = self.prepare_float_operand(width, lhs, *gp0, *fp0)?;
        let rhs_fp = self.prepare_float_operand(width, rhs, *gp1, *fp1)?;
        let result_fp = if self.core.is_fp_reg(dst) {
            let dst_fp = self.map_fp_reg(dst)? as u8;
            self.core.set_fp_reg_width(dst, width)?;
            dst_fp
        } else {
            // Keep GP-targeted float ops in XMM0. XMM1 is already the rhs
            // materialization scratch, so using it as the destination can
            // overwrite the rhs before the binary op runs.
            *fp0 as u8
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
                    let scratch = *fp1 as u8;
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
            MachineFloatBinaryOp::Min | MachineFloatBinaryOp::Max => {
                // Wasm fmin/fmax: if either operand is NaN, result is NaN.
                // x86_64 minsd/minss/maxsd/maxss: if either is NaN, returns
                // the SECOND (source) operand — and destroys the first
                // (destination) operand. We therefore (a) check for NaN
                // BEFORE the minsd/maxsd so the compare still sees the
                // original lhs register unclobbered, and (b) fall through
                // to an `addsd` on the NaN path so any NaN propagates.
                //
                // Previous version did the ucomisd AFTER the minsd, which
                // is wrong when `result_fp == lhs_fp`: the minsd clobbers
                // lhs_fp with the min result, so the NaN check compares
                // the WRONG operand and incorrectly takes the fast path.
                let is_min = matches!(op, MachineFloatBinaryOp::Min);

                // Guard: if result == rhs and result != lhs, save rhs to
                // a scratch so the later `movss result_fp, lhs_fp` (or the
                // inline minsd/addsd which both clobber dst) doesn't
                // destroy the rhs value before we use it.
                let actual_rhs = if result_fp == rhs_fp as u8 && result_fp != lhs_fp as u8 {
                    let scratch = *fp1 as u8;
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
                // Move lhs into result_fp if needed. This must happen
                // BEFORE the ucomisd so the NaN check still sees the
                // original lhs_fp value (which we leave untouched in its
                // own register).
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
                // NaN check on the ORIGINAL operands. If `result_fp ==
                // lhs_fp`, lhs_fp still holds the original lhs at this
                // point (we have not yet run the clobbering min/max).
                match width {
                    MachineFloatWidth::F32 => {
                        enc::ucomiss(&mut self.core.text, lhs_fp as u8, actual_rhs)
                    }
                    MachineFloatWidth::F64 => {
                        enc::ucomisd(&mut self.core.text, lhs_fp as u8, actual_rhs)
                    }
                };
                let nan_path = self.core.new_label();
                let done = self.core.new_label();
                // PF=1 means unordered (NaN) — jump to NaN path.
                self.emit_jcc(Cc::P, nan_path);
                // Ordered fast path: minsd / maxsd directly.
                match (width, is_min) {
                    (MachineFloatWidth::F32, true) => {
                        enc::minss(&mut self.core.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F64, true) => {
                        enc::minsd(&mut self.core.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F32, false) => {
                        enc::maxss(&mut self.core.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F64, false) => {
                        enc::maxsd(&mut self.core.text, result_fp, actual_rhs)
                    }
                };
                self.emit_jmp(done);
                // NaN path: result_fp already holds lhs, so addsd
                // propagates any NaN from either operand.
                self.core.bind_label(nan_path);
                match width {
                    MachineFloatWidth::F32 => {
                        enc::addss(&mut self.core.text, result_fp, actual_rhs)
                    }
                    MachineFloatWidth::F64 => {
                        enc::addsd(&mut self.core.text, result_fp, actual_rhs)
                    }
                };
                self.core.bind_label(done);
            }
            MachineFloatBinaryOp::Copysign => {
                // magnitude of lhs, sign of rhs.
                // Strategy: clear sign of lhs (abs), extract sign of rhs, OR them.
                // Use a mask scratch that doesn't conflict with result_fp or rhs_fp.
                let mask_xmm = if result_fp != *fp0 as u8 && rhs_fp as u8 != *fp0 as u8 {
                    *fp0 as u8
                } else {
                    *fp1 as u8
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
                self.materialize_u64(*gp0, abs_mask);
                enc::movq_xmm_r64(&mut self.core.text, mask_xmm, *gp0);
                enc::andpd(&mut self.core.text, result_fp, mask_xmm);
                // mask_xmm = rhs & sign_mask (extract sign bit)
                self.materialize_u64(*gp0, sign_mask);
                enc::movq_xmm_r64(&mut self.core.text, mask_xmm, *gp0);
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
        let gp0 = self.gp_scratch.scoped_alloc().detach();
        let gp1 = self.gp_scratch.scoped_alloc().detach();
        let fp0 = self.fp_scratch.scoped_alloc().detach();
        let fp1 = self.fp_scratch.scoped_alloc().detach();
        let lhs_fp = self.prepare_float_operand(width, lhs, *gp0, *fp0)?;
        // Choose an rhs FP scratch that doesn't conflict with lhs. When lhs
        // already lives in a mapped FP register (not `fp0`), reuse `fp0` for
        // rhs to avoid clobbering a live FP SSA value.
        let rhs_fp_scratch = if lhs_fp != *fp0 as u32 { *fp0 } else { *fp1 };
        if matches!(rhs, MachineValue::Imm64(0)) {
            enc::xorpd(
                &mut self.core.text,
                rhs_fp_scratch as u8,
                rhs_fp_scratch as u8,
            );
            match width {
                MachineFloatWidth::F32 => {
                    enc::ucomiss(&mut self.core.text, lhs_fp as u8, rhs_fp_scratch as u8)
                }
                MachineFloatWidth::F64 => {
                    enc::ucomisd(&mut self.core.text, lhs_fp as u8, rhs_fp_scratch as u8)
                }
            };
        } else {
            let rhs_fp = self.prepare_float_operand(width, rhs, *gp1, rhs_fp_scratch)?;
            match width {
                MachineFloatWidth::F32 => {
                    enc::ucomiss(&mut self.core.text, lhs_fp as u8, rhs_fp as u8)
                }
                MachineFloatWidth::F64 => {
                    enc::ucomisd(&mut self.core.text, lhs_fp as u8, rhs_fp as u8)
                }
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
                enc::setcc(&mut self.core.text, Cc::NP, *gp0);
                enc::and_rr_32(&mut self.core.text, dst_gp, *gp0);
                // Zero-extend the byte result to full register
                enc::movzx_r32_r8(&mut self.core.text, dst_gp, dst_gp);
            }
            MachineCompareKind::Ne => {
                // Unordered OR not-equal: NE=1 OR PF=1
                enc::setcc(&mut self.core.text, Cc::NE, dst_gp);
                enc::setcc(&mut self.core.text, Cc::P, *gp0);
                enc::or_rr_32(&mut self.core.text, dst_gp, *gp0);
                enc::movzx_r32_r8(&mut self.core.text, dst_gp, dst_gp);
            }
            MachineCompareKind::Lt => {
                // Ordered and less: CF=1 AND PF=0 → use JB (CF=1), but need !PF too
                // Actually, for UCOMISD: CF=1 for less-than AND for unordered.
                // So Lt = CF=1 AND PF=0 → SETB AND SETNP
                enc::setcc(&mut self.core.text, Cc::B, dst_gp);
                enc::setcc(&mut self.core.text, Cc::NP, *gp0);
                enc::and_rr_32(&mut self.core.text, dst_gp, *gp0);
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
                enc::setcc(&mut self.core.text, Cc::NP, *gp0);
                enc::and_rr_32(&mut self.core.text, dst_gp, *gp0);
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

    pub(super) fn lower_call_external(&mut self, const_idx: usize) -> Result<(), WasmError> {
        // Delegated to the control-flow module, which owns the matching
        // body_local_error_label propagation path.
        self.lower_call_external_term(const_idx)
    }

    // ── Float conversion helpers ────────────────────────────────────────────

    /// Get FP destination register: if dst is FP reg, use it directly; else use scratch.
    fn dst_float_reg(
        &mut self,
        dst: MachineReg,
        width: MachineFloatWidth,
        scratch_fp: u8,
    ) -> Result<u8, WasmError> {
        if self.core.is_fp_reg(dst) {
            let dst_fp = self.map_fp_reg(dst)? as u8;
            self.core.set_fp_reg_width(dst, width)?;
            Ok(dst_fp)
        } else {
            Ok(scratch_fp)
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

    /// Save caller-clobbered GP dynamic registers to the system stack before a
    /// C helper call.
    pub(super) fn save_caller_clobbered_gp_dynamic(&mut self) {
        // Push the caller-clobbered dynamic GP subset. `RAX`, `RCX`, and `RDX`
        // are backend-owned on x86_64, not dynamic.
        enc::push(&mut self.core.text, X86Reg::RSI);
        enc::push(&mut self.core.text, X86Reg::RDI);
        enc::push(&mut self.core.text, X86Reg::R8);
        enc::push(&mut self.core.text, X86Reg::R9);
        enc::push(&mut self.core.text, X86Reg::R10);
        enc::push(&mut self.core.text, X86Reg::R11);
    }

    /// Restore caller-clobbered GP dynamic registers from the system stack
    /// after a C helper call.
    pub(super) fn restore_caller_clobbered_gp_dynamic(&mut self) {
        enc::pop(&mut self.core.text, X86Reg::R11);
        enc::pop(&mut self.core.text, X86Reg::R10);
        enc::pop(&mut self.core.text, X86Reg::R9);
        enc::pop(&mut self.core.text, X86Reg::R8);
        enc::pop(&mut self.core.text, X86Reg::RDI);
        enc::pop(&mut self.core.text, X86Reg::RSI);
    }

    fn lower_trapping_trunc(
        &mut self,
        op: MachineConvertOp,
        dst: X86Reg,
        src: X86Reg,
    ) -> Result<(), WasmError> {
        // The C trunc helper call is ABI-specific: SysV returns a two-field
        // `repr(C)` struct in RAX/RDX, while Win64 receives an out-pointer
        // as a fourth argument. `callconv::emit_trapping_trunc_call` owns
        // the whole save → arg-setup → call → restore → test → branch
        // sequence and leaves the 64-bit result in the supplied backend-owned
        // scratch register.
        #[cfg(sf_os_windows)]
        let result_scratch = X86Reg::RDX;
        #[cfg(not(sf_os_windows))]
        let result_scratch = *self
            .gp_scratch
            .try_claim_rcx()
            .or_else(|| self.gp_scratch.try_claim_rax())
            .expect("x86_64 trapping trunc needs RCX or RAX for the helper target")
            .detach();
        let error_label = self.core.body_local_error_label;
        callconv::emit_trapping_trunc_call(
            self,
            src,
            convert_op_code(op) as u64,
            result_scratch,
            error_label,
        );
        if dst != X86Reg::RDX {
            enc::mov_rr_64(&mut self.core.text, dst, X86Reg::RDX);
        }
        Ok(())
    }

    fn lower_saturating_trunc(
        &mut self,
        op: MachineConvertOp,
        dst: X86Reg,
        src: X86Reg,
    ) -> Result<(), WasmError> {
        // Keep the helper target out of ABI result registers. Win64 returns
        // the saturating result in RAX, so use backend-owned R11 for the
        // helper entry address.
        self.save_caller_clobbered_gp_dynamic();
        enc::mov_rr_64(&mut self.core.text, C_ARG0, src);
        self.materialize_u64(C_ARG1, convert_op_code(op) as u64);
        self.materialize_u64(X86Reg::R11, x86_64_saturating_trunc as usize as u64);
        enc::call_reg(&mut self.core.text, X86Reg::R11);
        self.restore_caller_clobbered_gp_dynamic();
        if dst != X86Reg::RAX {
            enc::mov_rr_64(&mut self.core.text, dst, X86Reg::RAX);
        }
        Ok(())
    }

    /// Decomposed indexed load: extend(index) + offset into a backend-owned GP
    /// scratch, then load from [base + scratch]. Stable-base form for
    /// store-forwarding.
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
        let scratch0 = self.gp_scratch.scoped_alloc().detach();
        let scratch1 = self.gp_scratch.scoped_alloc().detach();
        let addr_scratch = if base_x86 == *scratch0 {
            *scratch1
        } else {
            *scratch0
        };
        // Step 1: copy/extend index into the address scratch.
        if index_extend == MachineIndexExtend::ZeroExtend32 {
            enc::mov_rr_32(&mut self.core.text, addr_scratch, index_x86);
        } else {
            enc::mov_rr_64(&mut self.core.text, addr_scratch, index_x86);
        }
        // Step 2: add offset
        if offset != 0 {
            enc::add_ri_64(&mut self.core.text, addr_scratch, offset);
        }
        // Step 3: add base -> addr_scratch = base + extended_index + offset.
        enc::add_rr_64(&mut self.core.text, addr_scratch, base_x86);
        // Step 4: load from [scratch + 0]
        self.lower_load_from(dst, addr_scratch, 0, width, extension)
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
        let scratch0 = self.gp_scratch.scoped_alloc().detach();
        let scratch1 = self.gp_scratch.scoped_alloc().detach();
        let addr_scratch = if base_x86 == *scratch0 {
            *scratch1
        } else {
            *scratch0
        };
        if index_extend == MachineIndexExtend::ZeroExtend32 {
            enc::mov_rr_32(&mut self.core.text, addr_scratch, index_x86);
        } else {
            enc::mov_rr_64(&mut self.core.text, addr_scratch, index_x86);
        }
        if offset != 0 {
            enc::add_ri_64(&mut self.core.text, addr_scratch, offset);
        }
        enc::add_rr_64(&mut self.core.text, addr_scratch, base_x86);
        self.lower_store_to_with_scratch(addr_scratch, 0, width, src, *scratch0, *scratch1)
    }

    // ── Memory/table instruction lowering ────────────────────────────────────
    //
    // Memory/table ops on x86_64 route through the shared
    // `preserved_entry(ctx, op_code, io_ptr) -> u32` runtime helper. The
    // generated sequence is:
    //
    //   1. Save caller-clobbered GP dynamic regs (see save_caller_clobbered
    //      _gp_dynamic). Total 48 bytes → SP stays 16-aligned.
    //   2. sub rsp, PRESERVED_IO_BYTES                (64 bytes for 8 slots)
    //   3. Write IMM0/IMM1/ARG0..ARG2 slots on the stack.
    //   4. mov rdi, ctx ; mov rsi, op_code ; lea rdx, [rsp]
    //   5. mov r11, preserved_entry ; call r11
    //   6. Stash status (RAX) into a scratch; optionally read RET0 into
    //      another scratch.
    //   7. add rsp, PRESERVED_IO_BYTES
    //   8. Restore caller-clobbered GP dynamic regs.
    //   9. test status, status ; jne body_local_error_label
    //
    // FP dynamic registers are not saved around these calls. The MachineIR
    // pipeline is responsible for publishing cached state before helper calls.

    fn emit_preserved_io_open(&mut self) {
        self.save_caller_clobbered_gp_dynamic();
        // 8 slots × 8 bytes = 64 bytes, keeps RSP 16-byte aligned.
        enc::sub_rsp_imm8(&mut self.core.text, 64);
    }

    fn emit_io_store_imm(&mut self, slot: usize, value: u32) {
        let scratch = self.gp_scratch.scoped_alloc().detach();
        self.materialize_u64(*scratch, value as u64);
        enc::store_64(
            &mut self.core.text,
            X86Reg::RSP,
            (slot as i32) * 8,
            *scratch,
        );
    }

    fn emit_io_store_value(&mut self, slot: usize, value: MachineValue) -> Result<(), WasmError> {
        let scratch = self.gp_scratch.scoped_alloc().detach();
        let gp = self.materialize_value(*scratch, value)?;
        enc::store_64(&mut self.core.text, X86Reg::RSP, (slot as i32) * 8, gp);
        Ok(())
    }

    fn emit_io_store_u32_value(
        &mut self,
        slot: usize,
        value: MachineValue,
    ) -> Result<(), WasmError> {
        let scratch = self.gp_scratch.scoped_alloc().detach();
        match value {
            // Bulk-memory/table helper operands are Wasm i32 values.
            MachineValue::Imm64(imm) => self.materialize_u64(*scratch, imm as u32 as u64),
            MachineValue::Reg(reg) => {
                let src = self.map_gp_reg(reg)?;
                // Force a low-word move so the helper always sees a clean
                // zero-extended u32 even if the producer left stale high bits.
                enc::mov_rr_32(&mut self.core.text, *scratch, src);
            }
            MachineValue::ReservedReg(reg) => {
                return Err(WasmError::internal(alloc::format!(
                    "x86_64 cannot materialize reserved cache register {}",
                    reg.0
                )));
            }
        }
        enc::store_64(
            &mut self.core.text,
            X86Reg::RSP,
            (slot as i32) * 8,
            *scratch,
        );
        Ok(())
    }

    /// Call preserved_entry, handle status/result, tear down frame, check
    /// status. If `result_dst` is `Some`, the RET0 slot is loaded into
    /// that register *after* the caller-clobbered restore.
    fn emit_preserved_call_and_close(&mut self, op_code: u32, result_dst: Option<X86Reg>) {
        use crate::vm::runtime::preserved::{io as preserved_io, preserved_entry};
        // `R11` is volatile on both SysV and Win64 and is not used for any
        // argument slot here, so it can carry the helper target without
        // clobbering ABI inputs.
        let call_target = X86Reg::R11;
        // Keep the result in a non-restored caller-clobbered register so the
        // dynamic restore below cannot overwrite it.
        let result_scratch = X86Reg::RCX;

        enc::mov_rr_64(&mut self.core.text, C_ARG0, map_fixed_reg(MACHINE_CTX_REG));
        self.materialize_u64(C_ARG1, op_code as u64);
        enc::mov_rr_64(&mut self.core.text, C_ARG2, X86Reg::RSP);
        self.materialize_u64(call_target, preserved_entry as usize as u64);
        enc::call_reg(&mut self.core.text, call_target);

        // Read the result slot (if any) *before* restoring caller-clobbered
        // regs, because `result_dst` might alias one of the pushed regs.
        // We stash it into `result_scratch` and move it after restoration
        // below.
        if result_dst.is_some() {
            enc::load_64(
                &mut self.core.text,
                result_scratch,
                X86Reg::RSP,
                preserved_io::RET0 as i32 * 8,
            );
        }

        // Tear down the I/O area before popping caller-clobbered regs.
        enc::add_rsp_imm8(&mut self.core.text, 64);

        // Status lives in RAX and survives the dynamic restore because RAX is
        // backend-owned on x86_64, not part of the saved dynamic subset.

        self.restore_caller_clobbered_gp_dynamic();

        // If caller wanted the result, move it into the target dst now
        // that the stack is restored.
        if let Some(dst) = result_dst {
            if dst != result_scratch {
                enc::mov_rr_64(&mut self.core.text, dst, result_scratch);
            }
        }

        // Status check: non-zero means the helper trapped — branch to the
        // body-local error tail to propagate via the unified Return.
        enc::test_rr_64(&mut self.core.text, super::abi::C_RET0, super::abi::C_RET0);
        let body_local_error_label = self.core.body_local_error_label;
        self.emit_jcc(Cc::NE, body_local_error_label);
    }

    fn lower_memory_grow(
        &mut self,
        mem_idx: u32,
        dst: MachineReg,
        delta: MachineValue,
    ) -> Result<(), WasmError> {
        use crate::vm::runtime::preserved::{abi::op, io as preserved_io};
        let dst_gp = self.map_gp_reg(dst)?;
        self.emit_preserved_io_open();
        self.emit_io_store_imm(preserved_io::IMM0, mem_idx);
        self.emit_io_store_u32_value(preserved_io::ARG0, delta)?;
        self.emit_preserved_call_and_close(op::MEMORY_GROW, Some(dst_gp));
        Ok(())
    }

    fn lower_memory_fill(
        &mut self,
        mem_idx: u32,
        dest: MachineValue,
        val: MachineValue,
        len: MachineValue,
    ) -> Result<(), WasmError> {
        use crate::vm::runtime::preserved::{abi::op, io as preserved_io};
        self.emit_preserved_io_open();
        self.emit_io_store_imm(preserved_io::IMM0, mem_idx);
        self.emit_io_store_u32_value(preserved_io::ARG0, dest)?;
        self.emit_io_store_u32_value(preserved_io::ARG1, val)?;
        self.emit_io_store_u32_value(preserved_io::ARG2, len)?;
        self.emit_preserved_call_and_close(op::MEMORY_FILL, None);
        Ok(())
    }

    fn lower_memory_copy(
        &mut self,
        dst_mem: u32,
        src_mem: u32,
        dest: MachineValue,
        src: MachineValue,
        len: MachineValue,
    ) -> Result<(), WasmError> {
        use crate::vm::runtime::preserved::{abi::op, io as preserved_io};
        self.emit_preserved_io_open();
        self.emit_io_store_imm(preserved_io::IMM0, dst_mem);
        self.emit_io_store_imm(preserved_io::IMM1, src_mem);
        self.emit_io_store_u32_value(preserved_io::ARG0, dest)?;
        self.emit_io_store_u32_value(preserved_io::ARG1, src)?;
        self.emit_io_store_u32_value(preserved_io::ARG2, len)?;
        self.emit_preserved_call_and_close(op::MEMORY_COPY, None);
        Ok(())
    }

    fn lower_memory_init(
        &mut self,
        mem_idx: u32,
        data_idx: u32,
        dest: MachineValue,
        src: MachineValue,
        len: MachineValue,
    ) -> Result<(), WasmError> {
        use crate::vm::runtime::preserved::{abi::op, io as preserved_io};
        self.emit_preserved_io_open();
        self.emit_io_store_imm(preserved_io::IMM0, mem_idx);
        self.emit_io_store_imm(preserved_io::IMM1, data_idx);
        self.emit_io_store_u32_value(preserved_io::ARG0, dest)?;
        self.emit_io_store_u32_value(preserved_io::ARG1, src)?;
        self.emit_io_store_u32_value(preserved_io::ARG2, len)?;
        self.emit_preserved_call_and_close(op::MEMORY_INIT, None);
        Ok(())
    }

    fn lower_data_drop(&mut self, data_idx: u32) -> Result<(), WasmError> {
        use crate::vm::runtime::preserved::{abi::op, io as preserved_io};
        self.emit_preserved_io_open();
        self.emit_io_store_imm(preserved_io::IMM0, data_idx);
        self.emit_preserved_call_and_close(op::DATA_DROP, None);
        Ok(())
    }

    fn lower_table_grow(
        &mut self,
        table_idx: u32,
        dst: MachineReg,
        init_val: MachineValue,
        delta: MachineValue,
    ) -> Result<(), WasmError> {
        use crate::vm::runtime::preserved::{abi::op, io as preserved_io};
        let dst_gp = self.map_gp_reg(dst)?;
        self.emit_preserved_io_open();
        self.emit_io_store_imm(preserved_io::IMM0, table_idx);
        self.emit_io_store_value(preserved_io::ARG0, init_val)?;
        self.emit_io_store_u32_value(preserved_io::ARG1, delta)?;
        self.emit_preserved_call_and_close(op::TABLE_GROW, Some(dst_gp));
        Ok(())
    }

    fn lower_table_fill(
        &mut self,
        table_idx: u32,
        start: MachineValue,
        val: MachineValue,
        len: MachineValue,
    ) -> Result<(), WasmError> {
        use crate::vm::runtime::preserved::{abi::op, io as preserved_io};
        self.emit_preserved_io_open();
        self.emit_io_store_imm(preserved_io::IMM0, table_idx);
        self.emit_io_store_u32_value(preserved_io::ARG0, start)?;
        self.emit_io_store_value(preserved_io::ARG1, val)?;
        self.emit_io_store_u32_value(preserved_io::ARG2, len)?;
        self.emit_preserved_call_and_close(op::TABLE_FILL, None);
        Ok(())
    }

    fn lower_table_copy(
        &mut self,
        dst_tbl: u32,
        src_tbl: u32,
        dest: MachineValue,
        src: MachineValue,
        len: MachineValue,
    ) -> Result<(), WasmError> {
        use crate::vm::runtime::preserved::{abi::op, io as preserved_io};
        self.emit_preserved_io_open();
        self.emit_io_store_imm(preserved_io::IMM0, dst_tbl);
        self.emit_io_store_imm(preserved_io::IMM1, src_tbl);
        self.emit_io_store_u32_value(preserved_io::ARG0, dest)?;
        self.emit_io_store_u32_value(preserved_io::ARG1, src)?;
        self.emit_io_store_u32_value(preserved_io::ARG2, len)?;
        self.emit_preserved_call_and_close(op::TABLE_COPY, None);
        Ok(())
    }

    fn lower_table_init(
        &mut self,
        table_idx: u32,
        elem_idx: u32,
        dest: MachineValue,
        src: MachineValue,
        len: MachineValue,
    ) -> Result<(), WasmError> {
        use crate::vm::runtime::preserved::{abi::op, io as preserved_io};
        self.emit_preserved_io_open();
        self.emit_io_store_imm(preserved_io::IMM0, table_idx);
        self.emit_io_store_imm(preserved_io::IMM1, elem_idx);
        self.emit_io_store_u32_value(preserved_io::ARG0, dest)?;
        self.emit_io_store_u32_value(preserved_io::ARG1, src)?;
        self.emit_io_store_u32_value(preserved_io::ARG2, len)?;
        self.emit_preserved_call_and_close(op::TABLE_INIT, None);
        Ok(())
    }

    fn lower_elem_drop(&mut self, elem_idx: u32) -> Result<(), WasmError> {
        use crate::vm::runtime::preserved::{abi::op, io as preserved_io};
        self.emit_preserved_io_open();
        self.emit_io_store_imm(preserved_io::IMM0, elem_idx);
        self.emit_preserved_call_and_close(op::ELEM_DROP, None);
        Ok(())
    }
}
