//! Instruction emission methods for `FunctionCompiler`:
//! Move, float const, address computation, load/store, integer/float
//! arithmetic, compare, select, convert, and related helpers.

use crate::error::WasmError;
use crate::vm::machine::machine_ir::{
    MachineAddr, MachineCompareKind, MachineConvertOp, MachineFloatBinaryOp, MachineFloatUnaryOp,
    MachineFloatWidth, MachineIntBinaryOp, MachineIntUnaryOp, MachineIntWidth,
    MachineLoadExtension, MachineMemWidth, MachineReg, MachineSign, MachineStorageType,
    MachineTrapKind, MachineValue, MACHINE_CTX_REG,
};

use super::abi::{
    inv_map_reg, map_fixed_reg, FP_SCRATCH0, FP_SCRATCH1, FP_SCRATCH2, SCRATCH0,
    SCRATCH1,
};
use super::enc::{self, Cond};
use super::reg::Arm64Reg;
use super::compile::{FunctionCompiler, LabelKind};
use super::compile_fusion::{
    cmp_imm_inst, int_binary_imm_inst,
};
use super::compile_helpers::{
    arm64_trapping_trunc, arm64_saturating_trunc,
    convert_op_code, convert_result_float_width, map_float_cond, map_int_cond, mem_width_bytes,
};

impl<'a> FunctionCompiler<'a> {
    pub(super) fn emit_move(
        &mut self,
        ty: MachineStorageType,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        if let Some(width) = ty.float_width() {
            let dst_fp = self.map_fp_reg(dst)?;
            match src {
                MachineValue::Reg(src_reg) if self.is_fp_reg(src_reg) => {
                    let src_fp = self.map_fp_reg(src_reg)?;
                    let src_width = self.fp_reg_width(src_reg)?;
                    if src_width != width {
                        return Err(WasmError::invalid(alloc::format!(
                            "arm64 typed float move width mismatch: dst expects {:?}, src {} is {:?}",
                            width,
                            src_reg.0,
                            src_width,
                        )));
                    }
                    if dst_fp != src_fp {
                        self.text.emit_u32(match width {
                            MachineFloatWidth::F32 => enc::fmov_s(dst_fp, src_fp),
                            MachineFloatWidth::F64 => enc::fmov_d(dst_fp, src_fp),
                        });
                    }
                    self.set_fp_reg_width(dst, width)?;
                    Ok(())
                }
                MachineValue::Reg(src_reg) => {
                    let src_gp = self.map_gp_reg(src_reg)?;
                    self.text.emit_u32(match width {
                        MachineFloatWidth::F32 => enc::fmov_s_from_gp(dst_fp, src_gp),
                        MachineFloatWidth::F64 => enc::fmov_d_from_gp(dst_fp, src_gp),
                    });
                    self.set_fp_reg_width(dst, width)?;
                    Ok(())
                }
                MachineValue::Imm64(value) => {
                    self.materialize_u64(SCRATCH0, value);
                    self.text.emit_u32(match width {
                        MachineFloatWidth::F32 => enc::fmov_s_from_gp(dst_fp, SCRATCH0),
                        MachineFloatWidth::F64 => enc::fmov_d_from_gp(dst_fp, SCRATCH0),
                    });
                    self.set_fp_reg_width(dst, width)?;
                    Ok(())
                }
            }
        } else {
            let dst_gp = self.map_gp_reg(dst)?;
            match src {
                MachineValue::Reg(src_reg) if self.is_fp_reg(src_reg) => {
                    let src_fp = self.map_fp_reg(src_reg)?;
                    match self.fp_reg_width(src_reg)? {
                        MachineFloatWidth::F32 => {
                            self.text.emit_u32(enc::fmov_gp_from_s(dst_gp, src_fp));
                        }
                        MachineFloatWidth::F64 => {
                            self.text.emit_u32(enc::fmov_gp_from_d(dst_gp, src_fp));
                        }
                    }
                    Ok(())
                }
                MachineValue::Reg(src_reg) => {
                    let src_gp = self.map_gp_reg(src_reg)?;
                    if dst_gp != src_gp {
                        self.text.emit_u32(enc::mov_reg_64(dst_gp, src_gp));
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
                "arm64 FloatConst destination {} must be an FP register",
                dst.0
            )));
        }
        let dst_fp = self.map_fp_reg(dst)?;
        let imm = match width {
            MachineFloatWidth::F32 => u64::from(bits as u32),
            MachineFloatWidth::F64 => bits,
        };
        self.materialize_u64(SCRATCH0, imm);
        self.text.emit_u32(match width {
            MachineFloatWidth::F32 => enc::fmov_s_from_gp(dst_fp, SCRATCH0),
            MachineFloatWidth::F64 => enc::fmov_d_from_gp(dst_fp, SCRATCH0),
        });
        self.set_fp_reg_width(dst, width)?;
        Ok(())
    }

    pub(super) fn emit_addr_into(
        &mut self,
        dst: Arm64Reg,
        addr: MachineAddr,
    ) -> Result<(), WasmError> {
        let base = self.map_gp_reg(addr.base)?;
        let offset = addr.offset as i64;
        if offset == 0 {
            if dst != base {
                self.text.emit_u32(enc::mov_reg_64(dst, base));
            }
            return Ok(());
        }
        if offset > 0 && offset < 4096 {
            self.text
                .emit_u32(enc::add_imm_64(dst, base, offset as u32));
            return Ok(());
        }
        if offset < 0 && -offset < 4096 {
            self.text
                .emit_u32(enc::sub_imm_64(dst, base, (-offset) as u32));
            return Ok(());
        }
        self.materialize_u64(SCRATCH1, offset.unsigned_abs());
        if offset >= 0 {
            self.text.emit_u32(enc::add_reg_64(dst, base, SCRATCH1));
        } else {
            self.text.emit_u32(enc::sub_reg_64(dst, base, SCRATCH1));
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
        if self.is_fp_reg(dst) {
            let dst_fp = self.map_fp_reg(dst)?;
            // Derive width from the load, not from previously-tracked reg width.
            let tracked_width = match width {
                MachineMemWidth::U32 => MachineFloatWidth::F32,
                MachineMemWidth::U64 => MachineFloatWidth::F64,
                _ => {
                    return Err(WasmError::invalid(
                        "arm64 MachineIR backend does not support narrow integer loads into FP machine regs".into(),
                    ))
                }
            };
            let offset = addr.offset as i64;
            if offset >= 0
                && matches!(
                    (width, extension, tracked_width),
                    (
                        MachineMemWidth::U32,
                        MachineLoadExtension::None,
                        MachineFloatWidth::F32
                    ) | (
                        MachineMemWidth::U32,
                        MachineLoadExtension::ZeroExtend,
                        MachineFloatWidth::F32
                    ) | (
                        MachineMemWidth::U64,
                        MachineLoadExtension::None,
                        MachineFloatWidth::F64
                    ) | (
                        MachineMemWidth::U64,
                        MachineLoadExtension::ZeroExtend,
                        MachineFloatWidth::F64
                    )
                )
                && (offset / mem_width_bytes(width)) < 4096
                && (offset % mem_width_bytes(width)) == 0
            {
                self.text.emit_u32(match tracked_width {
                    MachineFloatWidth::F32 => enc::ldr_s(dst_fp, base, (offset / 4) as u32),
                    MachineFloatWidth::F64 => enc::ldr_d(dst_fp, base, (offset / 8) as u32),
                });
                self.set_fp_reg_width(dst, tracked_width)?;
                return Ok(());
            }
            self.emit_addr_into(SCRATCH0, addr)?;
            self.text.emit_u32(match (tracked_width, width, extension) {
                (MachineFloatWidth::F32, MachineMemWidth::U32, MachineLoadExtension::None)
                | (
                    MachineFloatWidth::F32,
                    MachineMemWidth::U32,
                    MachineLoadExtension::ZeroExtend,
                ) => enc::ldr_s_reg(dst_fp, SCRATCH0, Arm64Reg::Xzr, false),
                (MachineFloatWidth::F64, MachineMemWidth::U64, MachineLoadExtension::None)
                | (
                    MachineFloatWidth::F64,
                    MachineMemWidth::U64,
                    MachineLoadExtension::ZeroExtend,
                ) => enc::ldr_d_reg(dst_fp, SCRATCH0, Arm64Reg::Xzr, false),
                _ => return Err(WasmError::invalid(
                    "arm64 MachineIR backend does not support this load shape into FP machine regs"
                        .into(),
                )),
            });
            self.set_fp_reg_width(dst, tracked_width)?;
            return Ok(());
        }
        let dst = self.map_gp_reg(dst)?;
        // Fast path: U64 load with aligned immediate offset -> single ldr_64
        if matches!(
            (width, extension),
            (MachineMemWidth::U64, MachineLoadExtension::None)
                | (MachineMemWidth::U64, MachineLoadExtension::ZeroExtend)
        ) {
            let offset = addr.offset as i64;
            if offset >= 0 && (offset % 8) == 0 && (offset / 8) < 4096 {
                self.text
                    .emit_u32(enc::ldr_64(dst, base, (offset / 8) as u32));
                return Ok(());
            }
        }
        self.emit_addr_into(SCRATCH0, addr)?;
        let inst = match (width, extension) {
            (MachineMemWidth::U8, MachineLoadExtension::None)
            | (MachineMemWidth::U8, MachineLoadExtension::ZeroExtend) => {
                enc::ldrb_reg(dst, SCRATCH0, Arm64Reg::Xzr)
            }
            (MachineMemWidth::U8, MachineLoadExtension::SignExtend) => {
                enc::ldrsb_reg_64(dst, SCRATCH0, Arm64Reg::Xzr)
            }
            (MachineMemWidth::U16, MachineLoadExtension::None)
            | (MachineMemWidth::U16, MachineLoadExtension::ZeroExtend) => {
                enc::ldrh_reg(dst, SCRATCH0, Arm64Reg::Xzr)
            }
            (MachineMemWidth::U16, MachineLoadExtension::SignExtend) => {
                enc::ldrsh_reg_64(dst, SCRATCH0, Arm64Reg::Xzr)
            }
            (MachineMemWidth::U32, MachineLoadExtension::None)
            | (MachineMemWidth::U32, MachineLoadExtension::ZeroExtend) => {
                enc::ldr_reg_32(dst, SCRATCH0, Arm64Reg::Xzr)
            }
            (MachineMemWidth::U32, MachineLoadExtension::SignExtend) => {
                enc::ldrsw_reg(dst, SCRATCH0, Arm64Reg::Xzr)
            }
            (MachineMemWidth::U64, MachineLoadExtension::None)
            | (MachineMemWidth::U64, MachineLoadExtension::ZeroExtend) => {
                enc::ldr_reg_64(dst, SCRATCH0, Arm64Reg::Xzr)
            }
            (MachineMemWidth::U64, MachineLoadExtension::SignExtend) => {
                return Err(WasmError::invalid(
                    "arm64 MachineIR backend does not support sign-extending U64 loads".into(),
                ))
            }
        };
        self.text.emit_u32(inst);
        Ok(())
    }

    pub(super) fn emit_indexed_load(
        &mut self,
        dst: MachineReg,
        base: MachineReg,
        index: MachineReg,
        width: MachineMemWidth,
        extension: MachineLoadExtension,
        scaled: bool,
        uxtw: bool,
    ) -> Result<(), WasmError> {
        let base = self.map_gp_reg(base)?;
        let index = self.map_gp_reg(index)?;
        if self.is_fp_reg(dst) {
            let dst_fp = self.map_fp_reg(dst)?;
            let tracked_width = match width {
                MachineMemWidth::U32 => MachineFloatWidth::F32,
                MachineMemWidth::U64 => MachineFloatWidth::F64,
                _ => {
                    return Err(WasmError::invalid(
                        "arm64 MachineIR backend does not support narrow integer indexed loads into FP machine regs".into(),
                    ))
                }
            };
            let inst = match (tracked_width, width, extension) {
                (MachineFloatWidth::F32, MachineMemWidth::U32, MachineLoadExtension::None)
                | (MachineFloatWidth::F32, MachineMemWidth::U32, MachineLoadExtension::ZeroExtend) => {
                    if uxtw { enc::ldr_s_reg_uxtw(dst_fp, base, index) }
                    else { enc::ldr_s_reg(dst_fp, base, index, scaled) }
                }
                (MachineFloatWidth::F64, MachineMemWidth::U64, MachineLoadExtension::None)
                | (MachineFloatWidth::F64, MachineMemWidth::U64, MachineLoadExtension::ZeroExtend) => {
                    if uxtw { enc::ldr_d_reg_uxtw(dst_fp, base, index) }
                    else { enc::ldr_d_reg(dst_fp, base, index, scaled) }
                }
                _ => {
                    return Err(WasmError::invalid(
                        "arm64 MachineIR backend does not support this indexed load into FP machine regs".into(),
                    ))
                }
            };
            self.text.emit_u32(inst);
            self.set_fp_reg_width(dst, tracked_width)?;
            return Ok(());
        }
        let dst = self.map_gp_reg(dst)?;
        let inst = match (width, extension) {
            (MachineMemWidth::U8, MachineLoadExtension::None)
            | (MachineMemWidth::U8, MachineLoadExtension::ZeroExtend) => {
                if uxtw {
                    enc::ldrb_reg_uxtw(dst, base, index)
                } else {
                    enc::ldrb_reg(dst, base, index)
                }
            }
            (MachineMemWidth::U8, MachineLoadExtension::SignExtend) => {
                if uxtw {
                    enc::ldrsb_reg_64_uxtw(dst, base, index)
                } else {
                    enc::ldrsb_reg_64(dst, base, index)
                }
            }
            (MachineMemWidth::U16, MachineLoadExtension::None)
            | (MachineMemWidth::U16, MachineLoadExtension::ZeroExtend) => {
                if uxtw {
                    enc::ldrh_reg_uxtw(dst, base, index)
                } else if scaled {
                    enc::ldrh_reg_scaled(dst, base, index)
                } else {
                    enc::ldrh_reg(dst, base, index)
                }
            }
            (MachineMemWidth::U16, MachineLoadExtension::SignExtend) => {
                if uxtw {
                    enc::ldrsh_reg_64_uxtw(dst, base, index)
                } else if scaled {
                    enc::ldrsh_reg_64_scaled(dst, base, index)
                } else {
                    enc::ldrsh_reg_64(dst, base, index)
                }
            }
            (MachineMemWidth::U32, MachineLoadExtension::None)
            | (MachineMemWidth::U32, MachineLoadExtension::ZeroExtend) => {
                if uxtw {
                    enc::ldr_reg_32_uxtw(dst, base, index)
                } else if scaled {
                    enc::ldr_reg_32_scaled(dst, base, index)
                } else {
                    enc::ldr_reg_32(dst, base, index)
                }
            }
            (MachineMemWidth::U32, MachineLoadExtension::SignExtend) => {
                if uxtw {
                    enc::ldrsw_reg_uxtw(dst, base, index)
                } else if scaled {
                    enc::ldrsw_reg_scaled(dst, base, index)
                } else {
                    enc::ldrsw_reg(dst, base, index)
                }
            }
            (MachineMemWidth::U64, MachineLoadExtension::None)
            | (MachineMemWidth::U64, MachineLoadExtension::ZeroExtend) => {
                if uxtw {
                    enc::ldr_reg_64_uxtw(dst, base, index)
                } else if scaled {
                    enc::ldr_reg_64_scaled(dst, base, index)
                } else {
                    enc::ldr_reg_64(dst, base, index)
                }
            }
            (MachineMemWidth::U64, MachineLoadExtension::SignExtend) => {
                return Err(WasmError::invalid(
                    "arm64 MachineIR backend does not support sign-extending U64 loads".into(),
                ))
            }
        };
        self.text.emit_u32(inst);
        Ok(())
    }

    pub(super) fn emit_store(
        &mut self,
        addr: MachineAddr,
        width: MachineMemWidth,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let base = self.map_gp_reg(addr.base)?;
        if let MachineValue::Reg(src_reg) = src {
            if self.is_fp_reg(src_reg) {
                let src_fp = self.map_fp_reg(src_reg)?;
                let offset = addr.offset as i64;
                if offset >= 0
                    && (offset % mem_width_bytes(width)) == 0
                    && (offset / mem_width_bytes(width)) < 4096
                {
                    self.text.emit_u32(match width {
                        MachineMemWidth::U32 => enc::str_s(src_fp, base, (offset / 4) as u32),
                        MachineMemWidth::U64 => enc::str_d(src_fp, base, (offset / 8) as u32),
                        _ => {
                            return Err(WasmError::invalid(
                                "arm64 MachineIR backend does not support narrow FP stores".into(),
                            ))
                        }
                    });
                    return Ok(());
                }
                self.emit_addr_into(SCRATCH0, addr)?;
                self.text.emit_u32(match width {
                    MachineMemWidth::U32 => enc::str_s_reg(src_fp, SCRATCH0, Arm64Reg::Xzr, false),
                    MachineMemWidth::U64 => enc::str_d_reg(src_fp, SCRATCH0, Arm64Reg::Xzr, false),
                    _ => {
                        return Err(WasmError::invalid(
                            "arm64 MachineIR backend does not support narrow FP stores".into(),
                        ))
                    }
                });
                return Ok(());
            }
        }
        // Fast path: store zero -> use xzr directly (no materialization).
        if matches!(src, MachineValue::Imm64(0)) && width == MachineMemWidth::U64 {
            let offset = addr.offset as i64;
            if offset >= 0 && (offset % 8) == 0 && (offset / 8) < 4096 {
                self.text
                    .emit_u32(enc::str_64(Arm64Reg::Xzr, base, (offset / 8) as u32));
                return Ok(());
            }
        }
        // Fast path: U64 store with aligned immediate offset -> single str_64
        if width == MachineMemWidth::U64 {
            let offset = addr.offset as i64;
            if offset >= 0 && (offset % 8) == 0 && (offset / 8) < 4096 {
                let src_reg = self.materialize_value(SCRATCH1, src)?;
                self.text
                    .emit_u32(enc::str_64(src_reg, base, (offset / 8) as u32));
                return Ok(());
            }
        }
        self.emit_addr_into(SCRATCH0, addr)?;
        let src_reg = self.materialize_value(SCRATCH1, src)?;
        let inst = match width {
            MachineMemWidth::U8 => enc::strb_reg(src_reg, SCRATCH0, Arm64Reg::Xzr),
            MachineMemWidth::U16 => enc::strh_reg(src_reg, SCRATCH0, Arm64Reg::Xzr),
            MachineMemWidth::U32 => enc::str_reg_32(src_reg, SCRATCH0, Arm64Reg::Xzr),
            MachineMemWidth::U64 => enc::str_reg_64(src_reg, SCRATCH0, Arm64Reg::Xzr),
        };
        self.text.emit_u32(inst);
        Ok(())
    }

    pub(super) fn emit_indexed_store(
        &mut self,
        base: MachineReg,
        index: MachineReg,
        width: MachineMemWidth,
        src: MachineValue,
        scaled: bool,
    ) -> Result<(), WasmError> {
        let base = self.map_gp_reg(base)?;
        let index = self.map_gp_reg(index)?;
        if let MachineValue::Reg(src_reg) = src {
            if self.is_fp_reg(src_reg) {
                let src_fp = self.map_fp_reg(src_reg)?;
                self.text.emit_u32(match width {
                    MachineMemWidth::U32 => enc::str_s_reg(src_fp, base, index, scaled),
                    MachineMemWidth::U64 => enc::str_d_reg(src_fp, base, index, scaled),
                    _ => {
                        return Err(WasmError::invalid(
                            "arm64 MachineIR backend does not support narrow indexed FP stores"
                                .into(),
                        ))
                    }
                });
                return Ok(());
            }
        }
        let src_reg = self.materialize_value(SCRATCH1, src)?;
        let inst = match width {
            MachineMemWidth::U8 => enc::strb_reg(src_reg, base, index),
            MachineMemWidth::U16 => {
                if scaled {
                    enc::strh_reg_scaled(src_reg, base, index)
                } else {
                    enc::strh_reg(src_reg, base, index)
                }
            }
            MachineMemWidth::U32 => {
                if scaled {
                    enc::str_reg_32_scaled(src_reg, base, index)
                } else {
                    enc::str_reg_32(src_reg, base, index)
                }
            }
            MachineMemWidth::U64 => {
                if scaled {
                    enc::str_reg_64_scaled(src_reg, base, index)
                } else {
                    enc::str_reg_64(src_reg, base, index)
                }
            }
        };
        self.text.emit_u32(inst);
        Ok(())
    }

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
                self.text.emit_u32(enc::cmp_reg_32(src, Arm64Reg::Xzr));
                self.text.emit_u32(enc::cset_32(dst, Cond::Eq));
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Eqz) => {
                self.text.emit_u32(enc::cmp_reg_64(src, Arm64Reg::Xzr));
                self.text.emit_u32(enc::cset_64(dst, Cond::Eq));
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Clz) => {
                self.text.emit_u32(enc::clz_32(dst, src));
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Clz) => {
                self.text.emit_u32(enc::clz_64(dst, src));
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Extend8S) => {
                self.text.emit_u32(enc::sxtb_32(dst, src));
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Extend16S) => {
                self.text.emit_u32(enc::sxth_32(dst, src));
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Extend8S) => {
                self.text.emit_u32(enc::sxtb_64(dst, src));
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Extend16S) => {
                self.text.emit_u32(enc::sxth_64(dst, src));
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Extend32S) => {
                self.text.emit_u32(enc::sxtw(dst, src));
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Ctz) => {
                self.text.emit_u32(enc::rbit_32(dst, src));
                self.text.emit_u32(enc::clz_32(dst, dst));
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Ctz) => {
                self.text.emit_u32(enc::rbit_64(dst, src));
                self.text.emit_u32(enc::clz_64(dst, dst));
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Popcnt) => {
                // FMOV D0, X_src (move GP to FP); CNT V0.8B; ADDV B0; UMOV Wd, V0.B[0]
                self.text.emit_u32(enc::fmov_d_from_gp(FP_SCRATCH0, src));
                self.text.emit_u32(enc::cnt_8b(FP_SCRATCH0, FP_SCRATCH0));
                self.text.emit_u32(enc::addv_8b(FP_SCRATCH0, FP_SCRATCH0));
                self.text.emit_u32(enc::umov_b0(dst, FP_SCRATCH0));
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Popcnt) => {
                self.text.emit_u32(enc::fmov_d_from_gp(FP_SCRATCH0, src));
                self.text.emit_u32(enc::cnt_8b(FP_SCRATCH0, FP_SCRATCH0));
                self.text.emit_u32(enc::addv_8b(FP_SCRATCH0, FP_SCRATCH0));
                self.text.emit_u32(enc::umov_b0(dst, FP_SCRATCH0));
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Extend32S) => {
                // i32.extend32_s is a nop (already 32-bit)
                if dst != src {
                    self.text.emit_u32(enc::mov_reg_64(dst, src));
                }
            }
        }
        Ok(())
    }

    pub(super) fn emit_int_binary(
        &mut self,
        width: MachineIntWidth,
        op: MachineIntBinaryOp,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        let dst = self.map_gp_reg(dst)?;
        if let Some(inst) = int_binary_imm_inst(width, op, dst, lhs, rhs)? {
            self.text.emit_u32(inst);
            return Ok(());
        }
        let lhs = self.materialize_value(SCRATCH0, lhs)?;
        let rhs = self.materialize_value(SCRATCH1, rhs)?;
        match (width, op) {
            (MachineIntWidth::I32, MachineIntBinaryOp::Add) => {
                self.text.emit_u32(enc::add_reg_32(dst, lhs, rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Add) => {
                self.text.emit_u32(enc::add_reg_64(dst, lhs, rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Sub) => {
                self.text.emit_u32(enc::sub_reg_32(dst, lhs, rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Sub) => {
                self.text.emit_u32(enc::sub_reg_64(dst, lhs, rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Mul) => {
                self.text.emit_u32(enc::mul_32(dst, lhs, rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Mul) => {
                self.text.emit_u32(enc::mul_64(dst, lhs, rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::And) => {
                self.text.emit_u32(enc::and_reg_32(dst, lhs, rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::And) => {
                self.text.emit_u32(enc::and_reg_64(dst, lhs, rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Or) => {
                self.text.emit_u32(enc::orr_reg_32(dst, lhs, rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Or) => {
                self.text.emit_u32(enc::orr_reg_64(dst, lhs, rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Xor) => {
                self.text.emit_u32(enc::eor_reg_32(dst, lhs, rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Xor) => {
                self.text.emit_u32(enc::eor_reg_64(dst, lhs, rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Shl) => {
                self.text.emit_u32(enc::lslv_32(dst, lhs, rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Shl) => {
                self.text.emit_u32(enc::lslv_64(dst, lhs, rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::ShrS) => {
                self.text.emit_u32(enc::asrv_32(dst, lhs, rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::ShrS) => {
                self.text.emit_u32(enc::asrv_64(dst, lhs, rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::ShrU) => {
                self.text.emit_u32(enc::lsrv_32(dst, lhs, rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::ShrU) => {
                self.text.emit_u32(enc::lsrv_64(dst, lhs, rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Rotr) => {
                self.text.emit_u32(enc::rorv_32(dst, lhs, rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Rotr) => {
                self.text.emit_u32(enc::rorv_64(dst, lhs, rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Rotl) => {
                // rotl(x, n) = rotr(x, 32 - n)
                self.text.emit_u32(enc::neg_reg_32(SCRATCH0, rhs));
                self.text.emit_u32(enc::rorv_32(dst, lhs, SCRATCH0));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Rotl) => {
                self.text.emit_u32(enc::neg_reg_64(SCRATCH0, rhs));
                self.text.emit_u32(enc::rorv_64(dst, lhs, SCRATCH0));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::DivS) => {
                self.emit_div_s_32(dst, lhs, rhs);
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::DivS) => {
                self.emit_div_s_64(dst, lhs, rhs);
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::DivU) => {
                self.emit_div_u_check(lhs, rhs, MachineIntWidth::I32);
                self.text.emit_u32(enc::udiv_32(dst, lhs, rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::DivU) => {
                self.emit_div_u_check(lhs, rhs, MachineIntWidth::I64);
                self.text.emit_u32(enc::udiv_64(dst, lhs, rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::RemS) => {
                self.emit_rem_s_32(dst, lhs, rhs);
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::RemS) => {
                self.emit_rem_s_64(dst, lhs, rhs);
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::RemU) => {
                self.emit_div_u_check(lhs, rhs, MachineIntWidth::I32);
                // rem = lhs - (lhs / rhs) * rhs
                self.text.emit_u32(enc::udiv_32(SCRATCH0, lhs, rhs));
                self.text.emit_u32(enc::msub_32(dst, SCRATCH0, rhs, lhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::RemU) => {
                self.emit_div_u_check(lhs, rhs, MachineIntWidth::I64);
                self.text.emit_u32(enc::udiv_64(SCRATCH0, lhs, rhs));
                self.text.emit_u32(enc::msub_64(dst, SCRATCH0, rhs, lhs));
            }
        };
        Ok(())
    }

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
        let cond = map_int_cond(kind, sign);
        match width {
            MachineIntWidth::I32 => {
                self.text.emit_u32(enc::cset_32(dst, cond));
            }
            MachineIntWidth::I64 => {
                self.text.emit_u32(enc::cset_64(dst, cond));
            }
        };
        Ok(())
    }

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
                    let true_fp =
                        self.prepare_float_operand(width, on_true, SCRATCH0, FP_SCRATCH0)?;
                    let false_fp =
                        self.prepare_float_operand(width, on_false, SCRATCH1, FP_SCRATCH1)?;
                    let dst_fp = self.map_fp_reg(dst)?;
                    self.text
                        .emit_u32(enc::cmp_imm_64(self.map_gp_reg(reg)?, 0));
                    self.text.emit_u32(match width {
                        MachineFloatWidth::F32 => enc::fcsel_s(dst_fp, true_fp, false_fp, Cond::Ne),
                        MachineFloatWidth::F64 => enc::fcsel_d(dst_fp, true_fp, false_fp, Cond::Ne),
                    });
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
                MachineValue::Reg(reg) => {
                    self.text
                        .emit_u32(enc::cmp_imm_64(self.map_gp_reg(reg)?, 0));
                }
            }
            let true_reg = self.materialize_value(SCRATCH0, on_true)?;
            let false_reg = self.materialize_value(SCRATCH1, on_false)?;
            self.text
                .emit_u32(enc::csel_64(dst, true_reg, false_reg, Cond::Ne));
            Ok(())
        }
    }

    pub(super) fn emit_cmp_values(
        &mut self,
        width: MachineIntWidth,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        if let Some(inst) = cmp_imm_inst(width, lhs, rhs)? {
            self.text.emit_u32(inst);
            return Ok(());
        }
        let lhs = self.materialize_value(SCRATCH0, lhs)?;
        let rhs = self.materialize_value(SCRATCH1, rhs)?;
        match width {
            MachineIntWidth::I32 => {
                self.text.emit_u32(enc::cmp_reg_32(lhs, rhs));
            }
            MachineIntWidth::I64 => {
                self.text.emit_u32(enc::cmp_reg_64(lhs, rhs));
            }
        };
        Ok(())
    }

    // --- Division / remainder helpers with trap checks ---

    fn emit_div_u_check(&mut self, _lhs: Arm64Reg, rhs: Arm64Reg, width: MachineIntWidth) {
        // rhs == 0 => trap IntegerDivideByZero
        match width {
            MachineIntWidth::I32 => self.text.emit_u32(enc::cmp_reg_32(rhs, Arm64Reg::Xzr)),
            MachineIntWidth::I64 => self.text.emit_u32(enc::cmp_reg_64(rhs, Arm64Reg::Xzr)),
        };
        // Branch to a trap stub
        let trap_label = self.new_label(LabelKind::Edge);
        self.emit_b_cond(Cond::Eq, trap_label);
        // Emit the trap stub at the end via deferred_traps
        self.deferred_traps
            .push((trap_label, MachineTrapKind::IntegerDivideByZero));
    }

    fn emit_div_s_32(&mut self, dst: Arm64Reg, lhs: Arm64Reg, rhs: Arm64Reg) {
        // Check rhs == 0 => IntegerDivideByZero
        self.text.emit_u32(enc::cmp_reg_32(rhs, Arm64Reg::Xzr));
        let div_zero_label = self.new_label(LabelKind::Edge);
        self.emit_b_cond(Cond::Eq, div_zero_label);
        self.deferred_traps
            .push((div_zero_label, MachineTrapKind::IntegerDivideByZero));

        // Check lhs == i32::MIN && rhs == -1 => IntegerOverflow
        self.materialize_u64(SCRATCH0, i32::MIN as u32 as u64);
        self.text.emit_u32(enc::cmp_reg_32(lhs, SCRATCH0));
        let not_min = self.new_label(LabelKind::Edge);
        self.emit_b_cond(Cond::Ne, not_min);
        // lhs is MIN, check rhs == -1
        self.materialize_u64(SCRATCH0, (-1i32) as u32 as u64);
        self.text.emit_u32(enc::cmp_reg_32(rhs, SCRATCH0));
        let overflow_label = self.new_label(LabelKind::Edge);
        self.emit_b_cond(Cond::Eq, overflow_label);
        self.deferred_traps
            .push((overflow_label, MachineTrapKind::IntegerOverflow));

        self.bind_label(not_min);
        self.text.emit_u32(enc::sdiv_32(dst, lhs, rhs));
    }

    fn emit_div_s_64(&mut self, dst: Arm64Reg, lhs: Arm64Reg, rhs: Arm64Reg) {
        self.text.emit_u32(enc::cmp_reg_64(rhs, Arm64Reg::Xzr));
        let div_zero_label = self.new_label(LabelKind::Edge);
        self.emit_b_cond(Cond::Eq, div_zero_label);
        self.deferred_traps
            .push((div_zero_label, MachineTrapKind::IntegerDivideByZero));

        self.materialize_u64(SCRATCH0, i64::MIN as u64);
        self.text.emit_u32(enc::cmp_reg_64(lhs, SCRATCH0));
        let not_min = self.new_label(LabelKind::Edge);
        self.emit_b_cond(Cond::Ne, not_min);
        self.materialize_u64(SCRATCH0, (-1i64) as u64);
        self.text.emit_u32(enc::cmp_reg_64(rhs, SCRATCH0));
        let overflow_label = self.new_label(LabelKind::Edge);
        self.emit_b_cond(Cond::Eq, overflow_label);
        self.deferred_traps
            .push((overflow_label, MachineTrapKind::IntegerOverflow));

        self.bind_label(not_min);
        self.text.emit_u32(enc::sdiv_64(dst, lhs, rhs));
    }

    fn emit_rem_s_32(&mut self, dst: Arm64Reg, lhs: Arm64Reg, rhs: Arm64Reg) {
        // Check rhs == 0 => IntegerDivideByZero
        self.text.emit_u32(enc::cmp_reg_32(rhs, Arm64Reg::Xzr));
        let div_zero_label = self.new_label(LabelKind::Edge);
        self.emit_b_cond(Cond::Eq, div_zero_label);
        self.deferred_traps
            .push((div_zero_label, MachineTrapKind::IntegerDivideByZero));

        // rem = lhs - (lhs / rhs) * rhs  (wrapping, so MIN % -1 = 0, no trap)
        self.text.emit_u32(enc::sdiv_32(SCRATCH0, lhs, rhs));
        self.text.emit_u32(enc::msub_32(dst, SCRATCH0, rhs, lhs));
    }

    fn emit_rem_s_64(&mut self, dst: Arm64Reg, lhs: Arm64Reg, rhs: Arm64Reg) {
        self.text.emit_u32(enc::cmp_reg_64(rhs, Arm64Reg::Xzr));
        let div_zero_label = self.new_label(LabelKind::Edge);
        self.emit_b_cond(Cond::Eq, div_zero_label);
        self.deferred_traps
            .push((div_zero_label, MachineTrapKind::IntegerDivideByZero));

        self.text.emit_u32(enc::sdiv_64(SCRATCH0, lhs, rhs));
        self.text.emit_u32(enc::msub_64(dst, SCRATCH0, rhs, lhs));
    }

    // --- Float operations ---

    pub(super) fn emit_float_unary(
        &mut self,
        width: MachineFloatWidth,
        op: MachineFloatUnaryOp,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let src_fp = self.prepare_float_operand(width, src, SCRATCH0, FP_SCRATCH0)?;
        let result_fp = if self.is_fp_reg(dst) {
            let dst_fp = self.map_fp_reg(dst)?;
            self.set_fp_reg_width(dst, width)?;
            dst_fp
        } else {
            FP_SCRATCH2
        };
        // Perform the FP operation
        match (width, op) {
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Abs) => {
                self.text.emit_u32(enc::fabs_s(result_fp, src_fp))
            }
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Abs) => {
                self.text.emit_u32(enc::fabs_d(result_fp, src_fp))
            }
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Neg) => {
                self.text.emit_u32(enc::fneg_s(result_fp, src_fp))
            }
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Neg) => {
                self.text.emit_u32(enc::fneg_d(result_fp, src_fp))
            }
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Sqrt) => {
                self.text.emit_u32(enc::fsqrt_s(result_fp, src_fp))
            }
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Sqrt) => {
                self.text.emit_u32(enc::fsqrt_d(result_fp, src_fp))
            }
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Ceil) => {
                self.text.emit_u32(enc::frintp_s(result_fp, src_fp))
            }
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Ceil) => {
                self.text.emit_u32(enc::frintp_d(result_fp, src_fp))
            }
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Floor) => {
                self.text.emit_u32(enc::frintm_s(result_fp, src_fp))
            }
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Floor) => {
                self.text.emit_u32(enc::frintm_d(result_fp, src_fp))
            }
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Trunc) => {
                self.text.emit_u32(enc::frintz_s(result_fp, src_fp))
            }
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Trunc) => {
                self.text.emit_u32(enc::frintz_d(result_fp, src_fp))
            }
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Nearest) => {
                self.text.emit_u32(enc::frintn_s(result_fp, src_fp))
            }
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Nearest) => {
                self.text.emit_u32(enc::frintn_d(result_fp, src_fp))
            }
        };
        if !self.is_fp_reg(dst) {
            let dst_gp = self.map_gp_reg(dst)?;
            match width {
                MachineFloatWidth::F32 => {
                    self.text.emit_u32(enc::fmov_gp_from_s(dst_gp, result_fp))
                }
                MachineFloatWidth::F64 => {
                    self.text.emit_u32(enc::fmov_gp_from_d(dst_gp, result_fp))
                }
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
            let dst_fp = self.map_fp_reg(dst)?;
            self.set_fp_reg_width(dst, width)?;
            dst_fp
        } else {
            FP_SCRATCH2
        };
        match (width, op) {
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Add) => {
                self.text.emit_u32(enc::fadd_s(result_fp, lhs_fp, rhs_fp));
            }
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Add) => {
                self.text.emit_u32(enc::fadd_d(result_fp, lhs_fp, rhs_fp));
            }
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Sub) => {
                self.text.emit_u32(enc::fsub_s(result_fp, lhs_fp, rhs_fp));
            }
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Sub) => {
                self.text.emit_u32(enc::fsub_d(result_fp, lhs_fp, rhs_fp));
            }
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Mul) => {
                self.text.emit_u32(enc::fmul_s(result_fp, lhs_fp, rhs_fp));
            }
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Mul) => {
                self.text.emit_u32(enc::fmul_d(result_fp, lhs_fp, rhs_fp));
            }
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Div) => {
                self.text.emit_u32(enc::fdiv_s(result_fp, lhs_fp, rhs_fp));
            }
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Div) => {
                self.text.emit_u32(enc::fdiv_d(result_fp, lhs_fp, rhs_fp));
            }
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Min) => {
                // Wasm fmin: NaN if either is NaN. ARM64 FMIN returns non-NaN operand.
                self.text.emit_u32(enc::fmin_s(result_fp, lhs_fp, rhs_fp));
                self.text.emit_u32(enc::fcmp_s(lhs_fp, rhs_fp));
                let done = self.new_label(LabelKind::Edge);
                self.emit_b_cond(Cond::Vc, done); // no NaN => FMIN result is correct
                                                  // NaN case: FADD produces NaN from NaN input
                self.text.emit_u32(enc::fadd_s(result_fp, lhs_fp, rhs_fp));
                self.bind_label(done);
            }
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Min) => {
                self.text.emit_u32(enc::fmin_d(result_fp, lhs_fp, rhs_fp));
                self.text.emit_u32(enc::fcmp_d(lhs_fp, rhs_fp));
                let done = self.new_label(LabelKind::Edge);
                self.emit_b_cond(Cond::Vc, done);
                self.text.emit_u32(enc::fadd_d(result_fp, lhs_fp, rhs_fp));
                self.bind_label(done);
            }
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Max) => {
                self.text.emit_u32(enc::fmax_s(result_fp, lhs_fp, rhs_fp));
                self.text.emit_u32(enc::fcmp_s(lhs_fp, rhs_fp));
                let done = self.new_label(LabelKind::Edge);
                self.emit_b_cond(Cond::Vc, done);
                self.text.emit_u32(enc::fadd_s(result_fp, lhs_fp, rhs_fp));
                self.bind_label(done);
            }
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Max) => {
                self.text.emit_u32(enc::fmax_d(result_fp, lhs_fp, rhs_fp));
                self.text.emit_u32(enc::fcmp_d(lhs_fp, rhs_fp));
                let done = self.new_label(LabelKind::Edge);
                self.emit_b_cond(Cond::Vc, done);
                self.text.emit_u32(enc::fadd_d(result_fp, lhs_fp, rhs_fp));
                self.bind_label(done);
            }
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Copysign) => {
                // copysign: magnitude of lhs, sign of rhs
                self.text.emit_u32(enc::fabs_s(result_fp, lhs_fp)); // |lhs|
                self.text.emit_u32(enc::fneg_s(FP_SCRATCH0, result_fp)); // -|lhs|
                let rhs_gp = self.materialize_value(SCRATCH1, rhs)?;
                self.materialize_u64(SCRATCH0, 31);
                self.text.emit_u32(enc::lsrv_64(SCRATCH0, rhs_gp, SCRATCH0));
                self.text.emit_u32(enc::cmp_imm_64(SCRATCH0, 0));
                self.text
                    .emit_u32(enc::fcsel_s(result_fp, FP_SCRATCH0, result_fp, Cond::Ne));
            }
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Copysign) => {
                self.text.emit_u32(enc::fabs_d(result_fp, lhs_fp));
                self.text.emit_u32(enc::fneg_d(FP_SCRATCH0, result_fp));
                let rhs_gp = self.materialize_value(SCRATCH1, rhs)?;
                self.materialize_u64(SCRATCH0, 63);
                self.text.emit_u32(enc::lsrv_64(SCRATCH0, rhs_gp, SCRATCH0));
                self.text.emit_u32(enc::cmp_imm_64(SCRATCH0, 0));
                self.text
                    .emit_u32(enc::fcsel_d(result_fp, FP_SCRATCH0, result_fp, Cond::Ne));
            }
        };
        if !self.is_fp_reg(dst) {
            let dst_gp = self.map_gp_reg(dst)?;
            match width {
                MachineFloatWidth::F32 => {
                    self.text.emit_u32(enc::fmov_gp_from_s(dst_gp, result_fp))
                }
                MachineFloatWidth::F64 => {
                    self.text.emit_u32(enc::fmov_gp_from_d(dst_gp, result_fp))
                }
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
        // Compare against zero: use FCMP Dn, #0.0 when rhs is immediate zero.
        if matches!(rhs, MachineValue::Imm64(0)) {
            match width {
                MachineFloatWidth::F32 => self.text.emit_u32(enc::fcmp_s_zero(lhs_fp)),
                MachineFloatWidth::F64 => self.text.emit_u32(enc::fcmp_d_zero(lhs_fp)),
            };
        } else {
            let rhs_fp = self.prepare_float_operand(width, rhs, SCRATCH1, FP_SCRATCH1)?;
            match width {
                MachineFloatWidth::F32 => self.text.emit_u32(enc::fcmp_s(lhs_fp, rhs_fp)),
                MachineFloatWidth::F64 => self.text.emit_u32(enc::fcmp_d(lhs_fp, rhs_fp)),
            };
        }
        // Wasm float comparisons: unordered (NaN) => false for all except Ne
        let cond = map_float_cond(kind);
        self.text.emit_u32(enc::cset_32(dst_gp, cond));
        Ok(())
    }

    pub(super) fn emit_convert(
        &mut self,
        op: MachineConvertOp,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let dst_float_width = convert_result_float_width(op);
        let dst_float_reg = |this: &mut Self, width: MachineFloatWidth| -> Result<u32, WasmError> {
            if this.is_fp_reg(dst) {
                let dst_fp = this.map_fp_reg(dst)?;
                this.set_fp_reg_width(dst, width)?;
                Ok(dst_fp)
            } else {
                Ok(FP_SCRATCH1)
            }
        };
        match op {
            // Integer wrapping / extension (no FP involved)
            MachineConvertOp::I32WrapI64 => {
                let src_gp = self.materialize_value(SCRATCH0, src)?;
                let dst_gp = self.map_gp_reg(dst)?;
                // Just mask to 32 bits
                self.text.emit_u32(enc::mov_reg_32(dst_gp, src_gp));
            }
            MachineConvertOp::I64ExtendI32S => {
                let src_gp = self.materialize_value(SCRATCH0, src)?;
                let dst_gp = self.map_gp_reg(dst)?;
                self.text.emit_u32(enc::sxtw(dst_gp, src_gp));
            }
            MachineConvertOp::I64ExtendI32U => {
                let src_gp = self.materialize_value(SCRATCH0, src)?;
                let dst_gp = self.map_gp_reg(dst)?;
                self.text.emit_u32(enc::mov_reg_32(dst_gp, src_gp));
            }
            MachineConvertOp::I32ReinterpretF32 => {
                let dst_gp = self.map_gp_reg(dst)?;
                if let MachineValue::Reg(src_reg) = src {
                    if self.is_fp_reg(src_reg) {
                        let src_fp = self.map_fp_reg(src_reg)?;
                        self.text.emit_u32(enc::fmov_gp_from_s(dst_gp, src_fp));
                    } else {
                        let src_gp = self.map_gp_reg(src_reg)?;
                        if dst_gp != src_gp {
                            self.text.emit_u32(enc::mov_reg_32(dst_gp, src_gp));
                        }
                    }
                } else {
                    let src_gp = self.materialize_value(SCRATCH0, src)?;
                    if dst_gp != src_gp {
                        self.text.emit_u32(enc::mov_reg_32(dst_gp, src_gp));
                    }
                }
            }
            MachineConvertOp::I64ReinterpretF64 => {
                let dst_gp = self.map_gp_reg(dst)?;
                if let MachineValue::Reg(src_reg) = src {
                    if self.is_fp_reg(src_reg) {
                        let src_fp = self.map_fp_reg(src_reg)?;
                        self.text.emit_u32(enc::fmov_gp_from_d(dst_gp, src_fp));
                    } else {
                        let src_gp = self.map_gp_reg(src_reg)?;
                        if dst_gp != src_gp {
                            self.text.emit_u32(enc::mov_reg_64(dst_gp, src_gp));
                        }
                    }
                } else {
                    let src_gp = self.materialize_value(SCRATCH0, src)?;
                    if dst_gp != src_gp {
                        self.text.emit_u32(enc::mov_reg_64(dst_gp, src_gp));
                    }
                }
            }
            MachineConvertOp::F32ReinterpretI32 | MachineConvertOp::F64ReinterpretI64 => {
                let src_gp = self.materialize_value(SCRATCH0, src)?;
                let width = dst_float_width.expect("float reinterpret width");
                let dst_fp = dst_float_reg(self, width)?;
                self.text.emit_u32(match width {
                    MachineFloatWidth::F32 => enc::fmov_s_from_gp(dst_fp, src_gp),
                    MachineFloatWidth::F64 => enc::fmov_d_from_gp(dst_fp, src_gp),
                });
                if !self.is_fp_reg(dst) {
                    let dst_gp = self.map_gp_reg(dst)?;
                    self.text.emit_u32(enc::mov_reg_64(dst_gp, src_gp));
                }
            }
            // Float promotion / demotion
            MachineConvertOp::F64PromoteF32 => {
                let src_fp =
                    self.prepare_float_operand(MachineFloatWidth::F32, src, SCRATCH0, FP_SCRATCH0)?;
                let dst_fp = dst_float_reg(self, MachineFloatWidth::F64)?;
                self.text.emit_u32(enc::fcvt_d_from_s(dst_fp, src_fp));
                if !self.is_fp_reg(dst) {
                    let dst_gp = self.map_gp_reg(dst)?;
                    self.text.emit_u32(enc::fmov_gp_from_d(dst_gp, dst_fp));
                }
            }
            MachineConvertOp::F32DemoteF64 => {
                let src_fp =
                    self.prepare_float_operand(MachineFloatWidth::F64, src, SCRATCH0, FP_SCRATCH0)?;
                let dst_fp = dst_float_reg(self, MachineFloatWidth::F32)?;
                self.text.emit_u32(enc::fcvt_s_from_d(dst_fp, src_fp));
                if !self.is_fp_reg(dst) {
                    let dst_gp = self.map_gp_reg(dst)?;
                    self.text.emit_u32(enc::fmov_gp_from_s(dst_gp, dst_fp));
                }
            }
            // Int -> Float conversions
            MachineConvertOp::F32ConvertI32S => {
                let src_gp = self.materialize_value(SCRATCH0, src)?;
                let dst_fp = dst_float_reg(self, MachineFloatWidth::F32)?;
                self.text.emit_u32(enc::scvtf_s_32(dst_fp, src_gp));
                if !self.is_fp_reg(dst) {
                    let dst_gp = self.map_gp_reg(dst)?;
                    self.text.emit_u32(enc::fmov_gp_from_s(dst_gp, dst_fp));
                }
            }
            MachineConvertOp::F32ConvertI32U => {
                // Zero-extend to 64-bit first to ensure unsigned interpretation
                let src_gp = self.materialize_value(SCRATCH0, src)?;
                self.text.emit_u32(enc::mov_reg_32(SCRATCH0, src_gp));
                let dst_fp = dst_float_reg(self, MachineFloatWidth::F32)?;
                self.text.emit_u32(enc::ucvtf_s_64(dst_fp, SCRATCH0));
                if !self.is_fp_reg(dst) {
                    let dst_gp = self.map_gp_reg(dst)?;
                    self.text.emit_u32(enc::fmov_gp_from_s(dst_gp, dst_fp));
                }
            }
            MachineConvertOp::F32ConvertI64S => {
                let src_gp = self.materialize_value(SCRATCH0, src)?;
                let dst_fp = dst_float_reg(self, MachineFloatWidth::F32)?;
                self.text.emit_u32(enc::scvtf_s_64(dst_fp, src_gp));
                if !self.is_fp_reg(dst) {
                    let dst_gp = self.map_gp_reg(dst)?;
                    self.text.emit_u32(enc::fmov_gp_from_s(dst_gp, dst_fp));
                }
            }
            MachineConvertOp::F32ConvertI64U => {
                let src_gp = self.materialize_value(SCRATCH0, src)?;
                let dst_fp = dst_float_reg(self, MachineFloatWidth::F32)?;
                self.text.emit_u32(enc::ucvtf_s_64(dst_fp, src_gp));
                if !self.is_fp_reg(dst) {
                    let dst_gp = self.map_gp_reg(dst)?;
                    self.text.emit_u32(enc::fmov_gp_from_s(dst_gp, dst_fp));
                }
            }
            MachineConvertOp::F64ConvertI32S => {
                let src_gp = self.materialize_value(SCRATCH0, src)?;
                let dst_fp = dst_float_reg(self, MachineFloatWidth::F64)?;
                self.text.emit_u32(enc::scvtf_d_32(dst_fp, src_gp));
                if !self.is_fp_reg(dst) {
                    let dst_gp = self.map_gp_reg(dst)?;
                    self.text.emit_u32(enc::fmov_gp_from_d(dst_gp, dst_fp));
                }
            }
            MachineConvertOp::F64ConvertI32U => {
                let src_gp = self.materialize_value(SCRATCH0, src)?;
                self.text.emit_u32(enc::mov_reg_32(SCRATCH0, src_gp));
                let dst_fp = dst_float_reg(self, MachineFloatWidth::F64)?;
                self.text.emit_u32(enc::ucvtf_d_64(dst_fp, SCRATCH0));
                if !self.is_fp_reg(dst) {
                    let dst_gp = self.map_gp_reg(dst)?;
                    self.text.emit_u32(enc::fmov_gp_from_d(dst_gp, dst_fp));
                }
            }
            MachineConvertOp::F64ConvertI64S => {
                let src_gp = self.materialize_value(SCRATCH0, src)?;
                let dst_fp = dst_float_reg(self, MachineFloatWidth::F64)?;
                self.text.emit_u32(enc::scvtf_d_64(dst_fp, src_gp));
                if !self.is_fp_reg(dst) {
                    let dst_gp = self.map_gp_reg(dst)?;
                    self.text.emit_u32(enc::fmov_gp_from_d(dst_gp, dst_fp));
                }
            }
            MachineConvertOp::F64ConvertI64U => {
                let src_gp = self.materialize_value(SCRATCH0, src)?;
                let dst_fp = dst_float_reg(self, MachineFloatWidth::F64)?;
                self.text.emit_u32(enc::ucvtf_d_64(dst_fp, src_gp));
                if !self.is_fp_reg(dst) {
                    let dst_gp = self.map_gp_reg(dst)?;
                    self.text.emit_u32(enc::fmov_gp_from_d(dst_gp, dst_fp));
                }
            }
            // Trapping truncations: call Rust helpers
            MachineConvertOp::I32TruncF32S
            | MachineConvertOp::I32TruncF32U
            | MachineConvertOp::I32TruncF64S
            | MachineConvertOp::I32TruncF64U
            | MachineConvertOp::I64TruncF32S
            | MachineConvertOp::I64TruncF32U
            | MachineConvertOp::I64TruncF64S
            | MachineConvertOp::I64TruncF64U => {
                let dst_gp = self.map_gp_reg(dst)?;
                let src_gp = self.materialize_value(SCRATCH0, src)?;
                self.emit_trapping_trunc(op, dst_gp, src_gp)?;
            }
            // Saturating truncations
            MachineConvertOp::I32TruncSatF32S
            | MachineConvertOp::I32TruncSatF32U
            | MachineConvertOp::I32TruncSatF64S
            | MachineConvertOp::I32TruncSatF64U
            | MachineConvertOp::I64TruncSatF32S
            | MachineConvertOp::I64TruncSatF32U
            | MachineConvertOp::I64TruncSatF64S
            | MachineConvertOp::I64TruncSatF64U => {
                let dst_gp = self.map_gp_reg(dst)?;
                let src_gp = self.materialize_value(SCRATCH0, src)?;
                self.emit_saturating_trunc(op, dst_gp, src_gp)?;
            }
        }
        Ok(())
    }

    fn emit_trapping_trunc(
        &mut self,
        op: MachineConvertOp,
        dst: Arm64Reg,
        src: Arm64Reg,
    ) -> Result<(), WasmError> {
        // Call the helper: extern "C" fn(ctx, src_bits) -> status
        self.text.emit_u32(enc::mov_reg_64(
            Arm64Reg::X0,
            map_fixed_reg(MACHINE_CTX_REG),
        ));
        self.text.emit_u32(enc::mov_reg_64(Arm64Reg::X1, src));
        self.materialize_u64(Arm64Reg::X2, convert_op_code(op));
        self.materialize_u64(SCRATCH0, arm64_trapping_trunc as usize as u64);
        self.text.emit_u32(enc::blr(SCRATCH0));
        // X0 = status (0 = ok), X1 = result value
        self.emit_cbnz(Arm64Reg::X0, self.return_error_label);
        self.text.emit_u32(enc::mov_reg_64(dst, Arm64Reg::X1));
        Ok(())
    }

    fn emit_saturating_trunc(
        &mut self,
        op: MachineConvertOp,
        dst: Arm64Reg,
        src: Arm64Reg,
    ) -> Result<(), WasmError> {
        self.text.emit_u32(enc::mov_reg_64(Arm64Reg::X0, src));
        self.materialize_u64(Arm64Reg::X1, convert_op_code(op));
        self.materialize_u64(SCRATCH0, arm64_saturating_trunc as usize as u64);
        self.text.emit_u32(enc::blr(SCRATCH0));
        // X0 = result value (no error possible for sat)
        self.text.emit_u32(enc::mov_reg_64(dst, Arm64Reg::X0));
        Ok(())
    }
}
