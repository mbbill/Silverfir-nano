//! ARM64 instruction emission: emit_inst dispatch and individual instruction methods.

use crate::error::WasmError;
use crate::vm::machine::machine_ir::{
    MachineAddr, MachineBlockParam, MachineCompareKind, MachineConvertOp, MachineFloatBinaryOp,
    MachineFloatUnaryOp, MachineFloatWidth, MachineFuncId, MachineFunctionRuntime,
    MachineIntBinaryOp, MachineIntUnaryOp, MachineIntWidth,
    MachineInstKind, MachineInst, MachineIndexExtend, MachineLoadExtension, MachineMemWidth,
    MachineReg, MachineSign, MachineStorageType, MachineTrapKind, MachineValue,
    MACHINE_CTX_REG,
};

use super::{enc, reg::Arm64Reg};
use super::abi::{inv_map_reg, fp_machine_reg, map_fixed_reg, map_reg, SCRATCH0, SCRATCH1, FP_SCRATCH0, FP_SCRATCH1, FP_SCRATCH2};
use super::backend::BranchFixup;
use super::fusion::{cmp_imm_inst, int_binary_imm_inst, map_int_cond, map_float_cond};
use crate::vm::arch::common::helpers::{convert_op_code, convert_result_float_width, mem_width_bytes};
use crate::vm::arch::common::text_emitter::TextEmitter;
use crate::vm::arch::common::types::ParallelSource;

impl<'a> super::backend::Arm64Backend<'a> {

    // ── Register mapping ─────────────────────────────────────────────────

    pub(super) fn map_gp_reg(&self, reg: MachineReg) -> Result<Arm64Reg, WasmError> {
        self.core.validate_gp_reg(reg)?;
        map_reg(reg)
    }

    pub(super) fn map_fp_reg(&self, reg: MachineReg) -> Result<u32, WasmError> {
        let index = self.core.fp_reg_index(reg)?;
        fp_machine_reg(index).ok_or_else(|| {
            WasmError::invalid(alloc::format!(
                "arm64 has no physical FP mapping for machine reg {}", reg.0
            ))
        })
    }

    // ── Constant materialization ─────────────────────────────────────────

    pub(super) fn materialize_u64(&mut self, dst: Arm64Reg, value: u64) {
        materialize_u64_into(&mut self.core.text, dst, value);
    }

    // ── Branch emission ──────────────────────────────────────────────────

    pub(super) fn emit_b(&mut self, label: usize) {
        let inst_offset = self.core.text.emit_u32(enc::b(0));
        self.fixups.push(BranchFixup {
            inst_offset, label, kind: super::backend::BranchFixupKind::B,
        });
    }

    pub(super) fn emit_b_cond(&mut self, cond: enc::Cond, label: usize) {
        let inst_offset = self.core.text.emit_u32(enc::b_cond(cond, 0));
        self.fixups.push(BranchFixup {
            inst_offset, label, kind: super::backend::BranchFixupKind::BCond(cond),
        });
    }

    pub(super) fn emit_cbnz(&mut self, reg: Arm64Reg, label: usize) {
        let inst_offset = self.core.text.emit_u32(enc::cbnz_64(reg, 0));
        self.fixups.push(BranchFixup {
            inst_offset, label, kind: super::backend::BranchFixupKind::Cbnz(reg),
        });
    }

    pub(super) fn emit_cbz(&mut self, reg: Arm64Reg, label: usize) {
        let inst_offset = self.core.text.emit_u32(enc::cbz_64(reg, 0));
        self.fixups.push(BranchFixup {
            inst_offset, label, kind: super::backend::BranchFixupKind::Cbz(reg),
        });
    }

    // ── Value materialization ────────────────────────────────────────────

    /// Move a `MachineValue` into a GP register. If the value is already a
    /// GP register, returns the mapped physical register (scratch unused).
    /// Otherwise materializes into `scratch` and returns it.
    pub(super) fn materialize_value(
        &mut self, scratch: Arm64Reg, value: MachineValue,
    ) -> Result<Arm64Reg, WasmError> {
        match value {
            MachineValue::Reg(reg) if self.core.is_fp_reg(reg) => {
                let src_fp = self.map_fp_reg(reg)?;
                match self.core.fp_reg_width(reg)? {
                    MachineFloatWidth::F32 => {
                        self.core.text.emit_u32(enc::fmov_gp_from_s(scratch, src_fp));
                    }
                    MachineFloatWidth::F64 => {
                        self.core.text.emit_u32(enc::fmov_gp_from_d(scratch, src_fp));
                    }
                };
                Ok(scratch)
            }
            MachineValue::Reg(reg) => self.map_gp_reg(reg),
            MachineValue::Imm64(value) => {
                self.materialize_u64(scratch, value);
                Ok(scratch)
            }
        }
    }

    /// Emit a CMP/CMP-imm for two integer operands, setting flags.
    pub(super) fn emit_cmp_values(
        &mut self, width: MachineIntWidth, lhs: MachineValue, rhs: MachineValue,
    ) -> Result<(), WasmError> {
        if let Some(inst) = cmp_imm_inst(width, lhs, rhs)? {
            self.core.text.emit_u32(inst);
            return Ok(());
        }
        let lhs = self.materialize_value(SCRATCH0, lhs)?;
        let rhs = self.materialize_value(SCRATCH1, rhs)?;
        match width {
            MachineIntWidth::I32 => self.core.text.emit_u32(enc::cmp_reg_32(lhs, rhs)),
            MachineIntWidth::I64 => self.core.text.emit_u32(enc::cmp_reg_64(lhs, rhs)),
        };
        Ok(())
    }

    /// Prepare a float operand: if already in an FP register, return it.
    /// Otherwise materialize bits into `gp_scratch`, FMOV into `fp_scratch`.
    pub(super) fn prepare_float_operand(
        &mut self, width: MachineFloatWidth, value: MachineValue,
        gp_scratch: Arm64Reg, fp_scratch: u32,
    ) -> Result<u32, WasmError> {
        if let MachineValue::Reg(reg) = value {
            if self.core.is_fp_reg(reg) {
                return self.map_fp_reg(reg);
            }
        }
        let gp = self.materialize_value(gp_scratch, value)?;
        match width {
            MachineFloatWidth::F32 => self.core.text.emit_u32(enc::fmov_s_from_gp(fp_scratch, gp)),
            MachineFloatWidth::F64 => self.core.text.emit_u32(enc::fmov_d_from_gp(fp_scratch, gp)),
        };
        Ok(fp_scratch)
    }

    /// Look up runtime metadata for a machine function.
    pub(super) fn runtime_for(
        &self, func_id: MachineFuncId,
    ) -> Result<&MachineFunctionRuntime, WasmError> {
        self.core.runtime_for(func_id)
    }

    // ── Instruction dispatch ─────────────────────────────────────────────

pub(super) fn emit_inst_dispatch(&mut self,
inst: &MachineInst) -> Result<(), WasmError> {
    match &inst.kind {
        MachineInstKind::Move { dst, src, ty } => self.emit_move(*ty, *dst, *src),
        MachineInstKind::FloatConst { width, dst, bits } => self.emit_float_const(*width, *dst, *bits),
        MachineInstKind::Load { dst, addr, width, extension, .. } => {
            self.emit_load(*dst, *addr, *width, *extension)
        }
        MachineInstKind::Store { addr, width, src, .. } => self.emit_store(*addr, *width, *src),
        MachineInstKind::IntUnary { width, op, dst, src } => {
            self.emit_int_unary(*width, *op, *dst, *src)
        }
        MachineInstKind::IntBinary { width, op, dst, lhs, rhs } => {
            self.emit_int_binary(*width, *op, *dst, *lhs, *rhs)
        }
        MachineInstKind::IntCompare { width, kind, sign, dst, lhs, rhs } => {
            self.emit_int_compare(*width, *kind, *sign, *dst, *lhs, *rhs)
        }
        MachineInstKind::Select { ty, dst, on_true, on_false, cond, .. } => {
            self.emit_select(*ty, *dst, *on_true, *on_false, *cond)
        }
        MachineInstKind::TrapIf { kind, cond } => self.emit_trap_if(*kind, cond),
        MachineInstKind::CallHelper(call) => {
            self.emit_call_helper(call.target.0 as usize, call.metadata.0 as usize)
        }
        MachineInstKind::FloatUnary { width, op, dst, src } => {
            self.emit_float_unary(*width, *op, *dst, *src)
        }
        MachineInstKind::FloatBinary { width, op, dst, lhs, rhs } => {
            self.emit_float_binary(*width, *op, *dst, *lhs, *rhs)
        }
        MachineInstKind::FloatCompare { width, kind, dst, lhs, rhs } => {
            self.emit_float_compare(*width, *kind, *dst, *lhs, *rhs)
        }
        MachineInstKind::Convert { op, dst, src } => self.emit_convert(*op, *dst, *src),
        MachineInstKind::IndexedLoad { dst, base, index, index_extend, offset, width, extension } => {
            let uxtw = *index_extend == MachineIndexExtend::ZeroExtend32;
            if *offset == 0 {
                self.emit_indexed_load(*dst, *base, *index, *width, *extension, false, uxtw)
            } else {
                self.emit_indexed_load_with_offset(*dst, *base, *index, *offset, *width, *extension, uxtw)
            }
        }
        MachineInstKind::IndexedStore { base, index, index_extend, offset, width, src } => {
            let uxtw = *index_extend == MachineIndexExtend::ZeroExtend32;
            if *offset == 0 {
                self.emit_indexed_store(*base, *index, *width, *src, false, uxtw)
            } else {
                self.emit_indexed_store_with_offset(*base, *index, *offset, *width, *src, uxtw)
            }
        }
        // 32-bit legalized instructions -- should not reach arm64 codegen.
        MachineInstKind::Int64PairBinary { .. }
        | MachineInstKind::Int64PairUnary { .. }
        | MachineInstKind::Int64PairDivRem { .. }
        | MachineInstKind::Int64PairShift { .. }
        | MachineInstKind::ConvertI64PairToFloat { .. }
        | MachineInstKind::Int64PairCompare { .. }
        | MachineInstKind::ConvertFloatToI64Pair { .. }
        | MachineInstKind::ReinterpretF64ToI64Pair { .. }
        | MachineInstKind::ReinterpretI64PairToF64 { .. } => {
            Err(WasmError::internal(
                "arm64 backend received 32-bit legalized instruction".into(),
            ))
        }
    }
}

/// Emit a parallel-move source -> destination.
pub(super) fn emit_source_move_dispatch(&mut self,
dst: MachineBlockParam,
    src: ParallelSource,
) -> Result<(), WasmError> {
    match src {
        ParallelSource::Reg {
            reg: src_reg,
            float_width: src_float_width,
        } => {
            if let Some(width) = dst.ty.float_width() {
                let dst_fp = self.map_fp_reg(dst.reg)?;
                if self.core.is_fp_reg(src_reg) {
                    let src_fp = self.map_fp_reg(src_reg)?;
                    self.core.text.emit_u32(match width {
                        MachineFloatWidth::F32 => enc::fmov_s(dst_fp, src_fp),
                        MachineFloatWidth::F64 => enc::fmov_d(dst_fp, src_fp),
                    });
                } else {
                    let src_gp = self.map_gp_reg(src_reg)?;
                    self.core.text.emit_u32(match width {
                        MachineFloatWidth::F32 => enc::fmov_s_from_gp(dst_fp, src_gp),
                        MachineFloatWidth::F64 => enc::fmov_d_from_gp(dst_fp, src_gp),
                    });
                }
                self.core.set_fp_reg_width(dst.reg, width)?;
            } else {
                let dst_gp = self.map_gp_reg(dst.reg)?;
                if self.core.is_fp_reg(src_reg) {
                    let src_fp = self.map_fp_reg(src_reg)?;
                    match src_float_width.ok_or_else(|| {
                        WasmError::invalid(alloc::format!(
                            "arm64 edge move is missing float-width metadata for machine reg {}",
                            src_reg.0
                        ))
                    })? {
                        MachineFloatWidth::F32 => self.core.text.emit_u32(enc::fmov_gp_from_s(dst_gp, src_fp)),
                        MachineFloatWidth::F64 => self.core.text.emit_u32(enc::fmov_gp_from_d(dst_gp, src_fp)),
                    };
                } else {
                    let src_gp = self.map_gp_reg(src_reg)?;
                    self.core.text.emit_u32(enc::mov_reg_64(dst_gp, src_gp));
                }
            }
        }
        ParallelSource::Imm(value) => {
            if let Some(width) = dst.ty.float_width() {
                let dst_fp = self.map_fp_reg(dst.reg)?;
                let scratch = SCRATCH0;
                self.materialize_u64(scratch, value);
                self.core.text.emit_u32(match width {
                    MachineFloatWidth::F32 => enc::fmov_s_from_gp(dst_fp, scratch),
                    MachineFloatWidth::F64 => enc::fmov_d_from_gp(dst_fp, scratch),
                });
                self.core.set_fp_reg_width(dst.reg, width)?;
            } else {
                self.materialize_u64(self.map_gp_reg(dst.reg)?, value);
            }
        }
        ParallelSource::GpTemp => {
            // Well-known temp: SCRATCH1 (X17), set by emit_gp_cycle_break.
            self.core
                .text
                .emit_u32(enc::mov_reg_64(self.map_gp_reg(dst.reg)?, SCRATCH1));
        }
        ParallelSource::FpTemp(width) => {
            // Well-known temp: FP_SCRATCH2 (D2), set by emit_fp_cycle_break.
            let dst_fp = self.map_fp_reg(dst.reg)?;
            self.core.text.emit_u32(match width {
                MachineFloatWidth::F32 => enc::fmov_s(dst_fp, FP_SCRATCH2),
                MachineFloatWidth::F64 => enc::fmov_d(dst_fp, FP_SCRATCH2),
            });
            self.core.set_fp_reg_width(dst.reg, width)?;
        }
    }
    Ok(())
}

// ── Move / Float constant ────────────────────────────────────────────────────

fn emit_move(&mut self,
ty: MachineStorageType,
    dst: MachineReg,
    src: MachineValue,
) -> Result<(), WasmError> {
    if let Some(width) = ty.float_width() {
        let dst_fp = self.map_fp_reg(dst)?;
        match src {
            MachineValue::Reg(src_reg) if self.core.is_fp_reg(src_reg) => {
                let src_fp = self.map_fp_reg(src_reg)?;
                let src_width = self.core.fp_reg_width(src_reg)?;
                if src_width != width {
                    return Err(WasmError::invalid(alloc::format!(
                        "arm64 typed float move width mismatch: dst expects {:?}, src {} is {:?}",
                        width,
                        src_reg.0,
                        src_width,
                    )));
                }
                if dst_fp != src_fp {
                    self.core.text.emit_u32(match width {
                        MachineFloatWidth::F32 => enc::fmov_s(dst_fp, src_fp),
                        MachineFloatWidth::F64 => enc::fmov_d(dst_fp, src_fp),
                    });
                }
                self.core.set_fp_reg_width(dst, width)?;
                Ok(())
            }
            MachineValue::Reg(src_reg) => {
                let src_gp = self.map_gp_reg(src_reg)?;
                self.core.text.emit_u32(match width {
                    MachineFloatWidth::F32 => enc::fmov_s_from_gp(dst_fp, src_gp),
                    MachineFloatWidth::F64 => enc::fmov_d_from_gp(dst_fp, src_gp),
                });
                self.core.set_fp_reg_width(dst, width)?;
                Ok(())
            }
            MachineValue::Imm64(value) => {
                let scratch = SCRATCH0;
                self.materialize_u64(scratch, value);
                self.core.text.emit_u32(match width {
                    MachineFloatWidth::F32 => enc::fmov_s_from_gp(dst_fp, scratch),
                    MachineFloatWidth::F64 => enc::fmov_d_from_gp(dst_fp, scratch),
                });
                self.core.set_fp_reg_width(dst, width)?;
                Ok(())
            }
        }
    } else {
        let dst_gp = self.map_gp_reg(dst)?;
        match src {
            MachineValue::Reg(src_reg) if self.core.is_fp_reg(src_reg) => {
                let src_fp = self.map_fp_reg(src_reg)?;
                match self.core.fp_reg_width(src_reg)? {
                    MachineFloatWidth::F32 => {
                        self.core.text.emit_u32(enc::fmov_gp_from_s(dst_gp, src_fp));
                    }
                    MachineFloatWidth::F64 => {
                        self.core.text.emit_u32(enc::fmov_gp_from_d(dst_gp, src_fp));
                    }
                }
                Ok(())
            }
            MachineValue::Reg(src_reg) => {
                let src_gp = self.map_gp_reg(src_reg)?;
                if dst_gp != src_gp {
                    self.core.text.emit_u32(enc::mov_reg_64(dst_gp, src_gp));
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

fn emit_float_const(&mut self,
width: MachineFloatWidth,
    dst: MachineReg,
    bits: u64,
) -> Result<(), WasmError> {
    if !self.core.is_fp_reg(dst) {
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
    let scratch = SCRATCH0;
    self.materialize_u64(scratch, imm);
    self.core.text.emit_u32(match width {
        MachineFloatWidth::F32 => enc::fmov_s_from_gp(dst_fp, scratch),
        MachineFloatWidth::F64 => enc::fmov_d_from_gp(dst_fp, scratch),
    });
    self.core.set_fp_reg_width(dst, width)?;
    Ok(())
}

// ── Address computation ──────────────────────────────────────────────────────

/// Compute an effective address into `dst`. May use an additional scratch
/// register from the pool for large offsets.
fn emit_addr_into(&mut self,
dst: Arm64Reg,
    addr: MachineAddr,
) -> Result<(), WasmError> {
    let base = self.map_gp_reg(addr.base)?;
    let offset = addr.offset as i64;
    if offset == 0 {
        if dst != base {
            self.core.text.emit_u32(enc::mov_reg_64(dst, base));
        }
        return Ok(());
    }
    if offset > 0 && offset < 4096 {
        self.core
            .text
            .emit_u32(enc::add_imm_64(dst, base, offset as u32));
        return Ok(());
    }
    if offset < 0 && -offset < 4096 {
        self.core
            .text
            .emit_u32(enc::sub_imm_64(dst, base, (-offset) as u32));
        return Ok(());
    }
    // Large offset: use SCRATCH1 (dst may be SCRATCH0).
    let off_scratch = SCRATCH1;
    self.materialize_u64(off_scratch, offset.unsigned_abs());
    if offset >= 0 {
        self.core
            .text
            .emit_u32(enc::add_reg_64(dst, base, off_scratch));
    } else {
        self.core
            .text
            .emit_u32(enc::sub_reg_64(dst, base, off_scratch));
    }
    Ok(())
}

/// Add an immediate offset to an already-mapped ARM64 register in-place.
fn emit_add_imm_to_reg(&mut self,
reg: Arm64Reg, off: i64) {
    if off > 0 && off < 4096 {
        self.core.text.emit_u32(enc::add_imm_64(reg, reg, off as u32));
    } else if off < 0 && -off < 4096 {
        self.core
            .text
            .emit_u32(enc::sub_imm_64(reg, reg, (-off) as u32));
    } else {
        // Use SCRATCH1 because reg may be SCRATCH0.
        materialize_u64_into(&mut self.core.text, SCRATCH1, off as u64);
        self.core.text.emit_u32(enc::add_reg_64(reg, reg, SCRATCH1));
    }
}

// ── Load / Store ─────────────────────────────────────────────────────────────

fn emit_load(&mut self,
dst: MachineReg,
    addr: MachineAddr,
    width: MachineMemWidth,
    extension: MachineLoadExtension,
) -> Result<(), WasmError> {
    let base = self.map_gp_reg(addr.base)?;
    if self.core.is_fp_reg(dst) {
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
            self.core.text.emit_u32(match tracked_width {
                MachineFloatWidth::F32 => enc::ldr_s(dst_fp, base, (offset / 4) as u32),
                MachineFloatWidth::F64 => enc::ldr_d(dst_fp, base, (offset / 8) as u32),
            });
            self.core.set_fp_reg_width(dst, tracked_width)?;
            return Ok(());
        }
        let addr_scratch = SCRATCH0;
        self.emit_addr_into(addr_scratch, addr)?;
        self.core.text.emit_u32(match (tracked_width, width, extension) {
            (MachineFloatWidth::F32, MachineMemWidth::U32, MachineLoadExtension::None)
            | (
                MachineFloatWidth::F32,
                MachineMemWidth::U32,
                MachineLoadExtension::ZeroExtend,
            ) => enc::ldr_s_reg(dst_fp, addr_scratch, Arm64Reg::Xzr, false),
            (MachineFloatWidth::F64, MachineMemWidth::U64, MachineLoadExtension::None)
            | (
                MachineFloatWidth::F64,
                MachineMemWidth::U64,
                MachineLoadExtension::ZeroExtend,
            ) => enc::ldr_d_reg(dst_fp, addr_scratch, Arm64Reg::Xzr, false),
            _ => return Err(WasmError::invalid(
                "arm64 MachineIR backend does not support this load shape into FP machine regs"
                    .into(),
            )),
        });
        self.core.set_fp_reg_width(dst, tracked_width)?;
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
            self.core
                .text
                .emit_u32(enc::ldr_64(dst, base, (offset / 8) as u32));
            return Ok(());
        }
    }
    let addr_scratch = SCRATCH0;
    self.emit_addr_into(addr_scratch, addr)?;
    let inst = match (width, extension) {
        (MachineMemWidth::U8, MachineLoadExtension::None)
        | (MachineMemWidth::U8, MachineLoadExtension::ZeroExtend) => {
            enc::ldrb_reg(dst, addr_scratch, Arm64Reg::Xzr)
        }
        (MachineMemWidth::U8, MachineLoadExtension::SignExtend) => {
            enc::ldrsb_reg_64(dst, addr_scratch, Arm64Reg::Xzr)
        }
        (MachineMemWidth::U16, MachineLoadExtension::None)
        | (MachineMemWidth::U16, MachineLoadExtension::ZeroExtend) => {
            enc::ldrh_reg(dst, addr_scratch, Arm64Reg::Xzr)
        }
        (MachineMemWidth::U16, MachineLoadExtension::SignExtend) => {
            enc::ldrsh_reg_64(dst, addr_scratch, Arm64Reg::Xzr)
        }
        (MachineMemWidth::U32, MachineLoadExtension::None)
        | (MachineMemWidth::U32, MachineLoadExtension::ZeroExtend) => {
            enc::ldr_reg_32(dst, addr_scratch, Arm64Reg::Xzr)
        }
        (MachineMemWidth::U32, MachineLoadExtension::SignExtend) => {
            enc::ldrsw_reg(dst, addr_scratch, Arm64Reg::Xzr)
        }
        (MachineMemWidth::U64, MachineLoadExtension::None)
        | (MachineMemWidth::U64, MachineLoadExtension::ZeroExtend) => {
            enc::ldr_reg_64(dst, addr_scratch, Arm64Reg::Xzr)
        }
        (MachineMemWidth::U64, MachineLoadExtension::SignExtend) => {
            return Err(WasmError::invalid(
                "arm64 MachineIR backend does not support sign-extending U64 loads".into(),
            ))
        }
    };
    self.core.text.emit_u32(inst);
    Ok(())
}

fn emit_store(&mut self,
addr: MachineAddr,
    width: MachineMemWidth,
    src: MachineValue,
) -> Result<(), WasmError> {
    let base = self.map_gp_reg(addr.base)?;
    if let MachineValue::Reg(src_reg) = src {
        if self.core.is_fp_reg(src_reg) {
            let src_fp = self.map_fp_reg(src_reg)?;
            let offset = addr.offset as i64;
            if offset >= 0
                && (offset % mem_width_bytes(width)) == 0
                && (offset / mem_width_bytes(width)) < 4096
            {
                self.core.text.emit_u32(match width {
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
            let addr_scratch = SCRATCH0;
            self.emit_addr_into(addr_scratch, addr)?;
            self.core.text.emit_u32(match width {
                MachineMemWidth::U32 => enc::str_s_reg(src_fp, addr_scratch, Arm64Reg::Xzr, false),
                MachineMemWidth::U64 => enc::str_d_reg(src_fp, addr_scratch, Arm64Reg::Xzr, false),
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
            self.core
                .text
                .emit_u32(enc::str_64(Arm64Reg::Xzr, base, (offset / 8) as u32));
            return Ok(());
        }
    }
    // Fast path: U64 store with aligned immediate offset -> single str_64
    if width == MachineMemWidth::U64 {
        let offset = addr.offset as i64;
        if offset >= 0 && (offset % 8) == 0 && (offset / 8) < 4096 {
            let src_reg = self.materialize_value(SCRATCH0, src)?;
            self.core
                .text
                .emit_u32(enc::str_64(src_reg, base, (offset / 8) as u32));
            return Ok(());
        }
    }
    let addr_scratch = SCRATCH0;
    self.emit_addr_into(addr_scratch, addr)?;
    let src_reg = self.materialize_value(SCRATCH1, src)?;
    let inst = match width {
        MachineMemWidth::U8 => enc::strb_reg(src_reg, addr_scratch, Arm64Reg::Xzr),
        MachineMemWidth::U16 => enc::strh_reg(src_reg, addr_scratch, Arm64Reg::Xzr),
        MachineMemWidth::U32 => enc::str_reg_32(src_reg, addr_scratch, Arm64Reg::Xzr),
        MachineMemWidth::U64 => enc::str_reg_64(src_reg, addr_scratch, Arm64Reg::Xzr),
    };
    self.core.text.emit_u32(inst);
    Ok(())
}

// ── Indexed Load / Store ─────────────────────────────────────────────────────

fn emit_indexed_load(&mut self,
dst: MachineReg,
    base_reg: MachineReg,
    index_reg: MachineReg,
    width: MachineMemWidth,
    extension: MachineLoadExtension,
    scaled: bool,
    uxtw: bool,
) -> Result<(), WasmError> {
    let base = self.map_gp_reg(base_reg)?;
    let index = self.map_gp_reg(index_reg)?;
    if self.core.is_fp_reg(dst) {
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
        self.core.text.emit_u32(inst);
        self.core.set_fp_reg_width(dst, tracked_width)?;
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
    self.core.text.emit_u32(inst);
    Ok(())
}

/// Indexed load with a non-zero offset: fold index + offset into a scratch,
/// then register-indexed load with the ORIGINAL memory base.
fn emit_indexed_load_with_offset(&mut self,
dst: MachineReg,
    base_reg: MachineReg,
    index_reg: MachineReg,
    offset: i32,
    width: MachineMemWidth,
    extension: MachineLoadExtension,
    uxtw: bool,
) -> Result<(), WasmError> {
    let index_arm = self.map_gp_reg(index_reg)?;
    // For GP loads, use dst as the scratch to avoid false dependency chains
    // between consecutive loads. For FP loads, fall back to a pool scratch.
    let scratch = if self.core.is_fp_reg(dst) {
        SCRATCH0
    } else {
        self.map_gp_reg(dst)?
    };
    if uxtw {
        self.core.text.emit_u32(enc::mov_reg_32(scratch, index_arm));
    } else {
        self.core.text.emit_u32(enc::mov_reg_64(scratch, index_arm));
    }
    self.emit_add_imm_to_reg(scratch, offset as i64);
    // Register-indexed load using the pre-computed scratch as the index.
    let base_arm = self.map_gp_reg(base_reg)?;
    if self.core.is_fp_reg(dst) {
        let dst_fp = self.map_fp_reg(dst)?;
        let inst = match width {
            MachineMemWidth::U32 => enc::ldr_s_reg(dst_fp, base_arm, scratch, false),
            MachineMemWidth::U64 => enc::ldr_d_reg(dst_fp, base_arm, scratch, false),
            _ => return Err(WasmError::invalid(
                "arm64: narrow FP indexed load not supported".into())),
        };
        self.core.text.emit_u32(inst);
        let tracked = if width == MachineMemWidth::U32 { MachineFloatWidth::F32 } else { MachineFloatWidth::F64 };
        self.core.set_fp_reg_width(dst, tracked)?;
    } else {
        let dst_arm = self.map_gp_reg(dst)?;
        let inst = match (width, extension) {
            (MachineMemWidth::U8, MachineLoadExtension::None)
            | (MachineMemWidth::U8, MachineLoadExtension::ZeroExtend) => enc::ldrb_reg(dst_arm, base_arm, scratch),
            (MachineMemWidth::U8, MachineLoadExtension::SignExtend) => enc::ldrsb_reg_64(dst_arm, base_arm, scratch),
            (MachineMemWidth::U16, MachineLoadExtension::None)
            | (MachineMemWidth::U16, MachineLoadExtension::ZeroExtend) => enc::ldrh_reg(dst_arm, base_arm, scratch),
            (MachineMemWidth::U16, MachineLoadExtension::SignExtend) => enc::ldrsh_reg_64(dst_arm, base_arm, scratch),
            (MachineMemWidth::U32, MachineLoadExtension::None)
            | (MachineMemWidth::U32, MachineLoadExtension::ZeroExtend) => enc::ldr_reg_32(dst_arm, base_arm, scratch),
            (MachineMemWidth::U32, MachineLoadExtension::SignExtend) => enc::ldrsw_reg(dst_arm, base_arm, scratch),
            (MachineMemWidth::U64, MachineLoadExtension::None)
            | (MachineMemWidth::U64, MachineLoadExtension::ZeroExtend) => enc::ldr_reg_64(dst_arm, base_arm, scratch),
            _ => return Err(WasmError::invalid("arm64: unsupported indexed load extension".into())),
        };
        self.core.text.emit_u32(inst);
    }
    Ok(())
}

fn emit_indexed_store(&mut self,
base_reg: MachineReg,
    index_reg: MachineReg,
    width: MachineMemWidth,
    src: MachineValue,
    scaled: bool,
    uxtw: bool,
) -> Result<(), WasmError> {
    let base = self.map_gp_reg(base_reg)?;
    let index = self.map_gp_reg(index_reg)?;
    if let MachineValue::Reg(src_reg) = src {
        if self.core.is_fp_reg(src_reg) {
            let src_fp = self.map_fp_reg(src_reg)?;
            self.core.text.emit_u32(match width {
                MachineMemWidth::U32 => {
                    if uxtw { enc::str_s_reg_uxtw(src_fp, base, index) }
                    else { enc::str_s_reg(src_fp, base, index, scaled) }
                }
                MachineMemWidth::U64 => {
                    if uxtw { enc::str_d_reg_uxtw(src_fp, base, index) }
                    else { enc::str_d_reg(src_fp, base, index, scaled) }
                }
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
        MachineMemWidth::U8 => {
            if uxtw { enc::strb_reg_uxtw(src_reg, base, index) }
            else { enc::strb_reg(src_reg, base, index) }
        }
        MachineMemWidth::U16 => {
            if uxtw { enc::strh_reg_uxtw(src_reg, base, index) }
            else if scaled { enc::strh_reg_scaled(src_reg, base, index) }
            else { enc::strh_reg(src_reg, base, index) }
        }
        MachineMemWidth::U32 => {
            if uxtw { enc::str_reg_32_uxtw(src_reg, base, index) }
            else if scaled { enc::str_reg_32_scaled(src_reg, base, index) }
            else { enc::str_reg_32(src_reg, base, index) }
        }
        MachineMemWidth::U64 => {
            if uxtw { enc::str_reg_64_uxtw(src_reg, base, index) }
            else if scaled { enc::str_reg_64_scaled(src_reg, base, index) }
            else { enc::str_reg_64(src_reg, base, index) }
        }
    };
    self.core.text.emit_u32(inst);
    Ok(())
}

/// Indexed store with a non-zero offset: fold index + offset into a scratch,
/// then register-indexed store with the ORIGINAL memory base.
fn emit_indexed_store_with_offset(&mut self,
base_reg: MachineReg,
    index_reg: MachineReg,
    offset: i32,
    width: MachineMemWidth,
    src: MachineValue,
    uxtw: bool,
) -> Result<(), WasmError> {
    // Prepare SCRATCH = zext_or_copy(index) + offset
    let index_arm = self.map_gp_reg(index_reg)?;
    let idx_scratch = SCRATCH0;
    if uxtw {
        self.core.text.emit_u32(enc::mov_reg_32(idx_scratch, index_arm));
    } else {
        self.core.text.emit_u32(enc::mov_reg_64(idx_scratch, index_arm));
    }
    self.emit_add_imm_to_reg(idx_scratch, offset as i64);

    // Register-indexed store using pre-mapped index register.
    let base_arm = self.map_gp_reg(base_reg)?;
    if let MachineValue::Reg(src_reg) = src {
        if self.core.is_fp_reg(src_reg) {
            let src_fp = self.map_fp_reg(src_reg)?;
            let inst = match width {
                MachineMemWidth::U32 => enc::str_s_reg(src_fp, base_arm, idx_scratch, false),
                MachineMemWidth::U64 => enc::str_d_reg(src_fp, base_arm, idx_scratch, false),
                _ => return Err(WasmError::invalid(
                    "arm64: narrow FP indexed store not supported".into())),
            };
            self.core.text.emit_u32(inst);
            return Ok(());
        }
    }
    let src_arm = self.materialize_value(SCRATCH1, src)?;
    let inst = match width {
        MachineMemWidth::U8 => enc::strb_reg(src_arm, base_arm, idx_scratch),
        MachineMemWidth::U16 => enc::strh_reg(src_arm, base_arm, idx_scratch),
        MachineMemWidth::U32 => enc::str_reg_32(src_arm, base_arm, idx_scratch),
        MachineMemWidth::U64 => enc::str_reg_64(src_arm, base_arm, idx_scratch),
    };
    self.core.text.emit_u32(inst);
    Ok(())
}

// ── Integer unary ────────────────────────────────────────────────────────────

fn emit_int_unary(&mut self,
width: MachineIntWidth,
    op: MachineIntUnaryOp,
    dst: MachineReg,
    src: MachineValue,
) -> Result<(), WasmError> {
    let dst = self.map_gp_reg(dst)?;
    let src = self.materialize_value(SCRATCH0, src)?;
    match (width, op) {
        (MachineIntWidth::I32, MachineIntUnaryOp::Eqz) => {
            self.core.text.emit_u32(enc::cmp_reg_32(src, Arm64Reg::Xzr));
            self.core.text.emit_u32(enc::cset_32(dst, enc::Cond::Eq));
        }
        (MachineIntWidth::I64, MachineIntUnaryOp::Eqz) => {
            self.core.text.emit_u32(enc::cmp_reg_64(src, Arm64Reg::Xzr));
            self.core.text.emit_u32(enc::cset_64(dst, enc::Cond::Eq));
        }
        (MachineIntWidth::I32, MachineIntUnaryOp::Clz) => {
            self.core.text.emit_u32(enc::clz_32(dst, src));
        }
        (MachineIntWidth::I64, MachineIntUnaryOp::Clz) => {
            self.core.text.emit_u32(enc::clz_64(dst, src));
        }
        (MachineIntWidth::I32, MachineIntUnaryOp::Extend8S) => {
            self.core.text.emit_u32(enc::sxtb_32(dst, src));
        }
        (MachineIntWidth::I32, MachineIntUnaryOp::Extend16S) => {
            self.core.text.emit_u32(enc::sxth_32(dst, src));
        }
        (MachineIntWidth::I64, MachineIntUnaryOp::Extend8S) => {
            self.core.text.emit_u32(enc::sxtb_64(dst, src));
        }
        (MachineIntWidth::I64, MachineIntUnaryOp::Extend16S) => {
            self.core.text.emit_u32(enc::sxth_64(dst, src));
        }
        (MachineIntWidth::I64, MachineIntUnaryOp::Extend32S) => {
            self.core.text.emit_u32(enc::sxtw(dst, src));
        }
        (MachineIntWidth::I32, MachineIntUnaryOp::Ctz) => {
            self.core.text.emit_u32(enc::rbit_32(dst, src));
            self.core.text.emit_u32(enc::clz_32(dst, dst));
        }
        (MachineIntWidth::I64, MachineIntUnaryOp::Ctz) => {
            self.core.text.emit_u32(enc::rbit_64(dst, src));
            self.core.text.emit_u32(enc::clz_64(dst, dst));
        }
        (MachineIntWidth::I32, MachineIntUnaryOp::Popcnt) => {
            // FMOV D0, X_src (move GP to FP); CNT V0.8B; ADDV B0; UMOV Wd, V0.B[0]
            let fp_scratch = FP_SCRATCH0;
            self.core.text.emit_u32(enc::fmov_d_from_gp(fp_scratch, src));
            self.core.text.emit_u32(enc::cnt_8b(fp_scratch, fp_scratch));
            self.core.text.emit_u32(enc::addv_8b(fp_scratch, fp_scratch));
            self.core.text.emit_u32(enc::umov_b0(dst, fp_scratch));
        }
        (MachineIntWidth::I64, MachineIntUnaryOp::Popcnt) => {
            let fp_scratch = FP_SCRATCH0;
            self.core.text.emit_u32(enc::fmov_d_from_gp(fp_scratch, src));
            self.core.text.emit_u32(enc::cnt_8b(fp_scratch, fp_scratch));
            self.core.text.emit_u32(enc::addv_8b(fp_scratch, fp_scratch));
            self.core.text.emit_u32(enc::umov_b0(dst, fp_scratch));
        }
        (MachineIntWidth::I32, MachineIntUnaryOp::Extend32S) => {
            // i32.extend32_s is a nop (already 32-bit)
            if dst != src {
                self.core.text.emit_u32(enc::mov_reg_64(dst, src));
            }
        }
    }
    Ok(())
}

// ── Integer binary ───────────────────────────────────────────────────────────

fn emit_int_binary(&mut self,
width: MachineIntWidth,
    op: MachineIntBinaryOp,
    dst: MachineReg,
    lhs: MachineValue,
    rhs: MachineValue,
) -> Result<(), WasmError> {
    let dst = self.map_gp_reg(dst)?;
    if let Some(inst) = int_binary_imm_inst(width, op, dst, lhs, rhs)? {
        self.core.text.emit_u32(inst);
        return Ok(());
    }
    let lhs = self.materialize_value(SCRATCH0, lhs)?;
    let rhs = self.materialize_value(SCRATCH1, rhs)?;
    match (width, op) {
        (MachineIntWidth::I32, MachineIntBinaryOp::Add) => {
            self.core.text.emit_u32(enc::add_reg_32(dst, lhs, rhs));
        }
        (MachineIntWidth::I64, MachineIntBinaryOp::Add) => {
            self.core.text.emit_u32(enc::add_reg_64(dst, lhs, rhs));
        }
        (MachineIntWidth::I32, MachineIntBinaryOp::Sub) => {
            self.core.text.emit_u32(enc::sub_reg_32(dst, lhs, rhs));
        }
        (MachineIntWidth::I64, MachineIntBinaryOp::Sub) => {
            self.core.text.emit_u32(enc::sub_reg_64(dst, lhs, rhs));
        }
        (MachineIntWidth::I32, MachineIntBinaryOp::Mul) => {
            self.core.text.emit_u32(enc::mul_32(dst, lhs, rhs));
        }
        (MachineIntWidth::I64, MachineIntBinaryOp::Mul) => {
            self.core.text.emit_u32(enc::mul_64(dst, lhs, rhs));
        }
        (MachineIntWidth::I32, MachineIntBinaryOp::And) => {
            self.core.text.emit_u32(enc::and_reg_32(dst, lhs, rhs));
        }
        (MachineIntWidth::I64, MachineIntBinaryOp::And) => {
            self.core.text.emit_u32(enc::and_reg_64(dst, lhs, rhs));
        }
        (MachineIntWidth::I32, MachineIntBinaryOp::Or) => {
            self.core.text.emit_u32(enc::orr_reg_32(dst, lhs, rhs));
        }
        (MachineIntWidth::I64, MachineIntBinaryOp::Or) => {
            self.core.text.emit_u32(enc::orr_reg_64(dst, lhs, rhs));
        }
        (MachineIntWidth::I32, MachineIntBinaryOp::Xor) => {
            self.core.text.emit_u32(enc::eor_reg_32(dst, lhs, rhs));
        }
        (MachineIntWidth::I64, MachineIntBinaryOp::Xor) => {
            self.core.text.emit_u32(enc::eor_reg_64(dst, lhs, rhs));
        }
        (MachineIntWidth::I32, MachineIntBinaryOp::Shl) => {
            self.core.text.emit_u32(enc::lslv_32(dst, lhs, rhs));
        }
        (MachineIntWidth::I64, MachineIntBinaryOp::Shl) => {
            self.core.text.emit_u32(enc::lslv_64(dst, lhs, rhs));
        }
        (MachineIntWidth::I32, MachineIntBinaryOp::ShrS) => {
            self.core.text.emit_u32(enc::asrv_32(dst, lhs, rhs));
        }
        (MachineIntWidth::I64, MachineIntBinaryOp::ShrS) => {
            self.core.text.emit_u32(enc::asrv_64(dst, lhs, rhs));
        }
        (MachineIntWidth::I32, MachineIntBinaryOp::ShrU) => {
            self.core.text.emit_u32(enc::lsrv_32(dst, lhs, rhs));
        }
        (MachineIntWidth::I64, MachineIntBinaryOp::ShrU) => {
            self.core.text.emit_u32(enc::lsrv_64(dst, lhs, rhs));
        }
        (MachineIntWidth::I32, MachineIntBinaryOp::Rotr) => {
            self.core.text.emit_u32(enc::rorv_32(dst, lhs, rhs));
        }
        (MachineIntWidth::I64, MachineIntBinaryOp::Rotr) => {
            self.core.text.emit_u32(enc::rorv_64(dst, lhs, rhs));
        }
        (MachineIntWidth::I32, MachineIntBinaryOp::Rotl) => {
            // rotl(x, n) = rotr(x, -n). Pick a scratch that doesn't clobber lhs.
            let neg_dst = if lhs == SCRATCH0 { SCRATCH1 } else { SCRATCH0 };
            self.core.text.emit_u32(enc::neg_reg_32(neg_dst, rhs));
            self.core.text.emit_u32(enc::rorv_32(dst, lhs, neg_dst));
        }
        (MachineIntWidth::I64, MachineIntBinaryOp::Rotl) => {
            let neg_dst = if lhs == SCRATCH0 { SCRATCH1 } else { SCRATCH0 };
            self.core.text.emit_u32(enc::neg_reg_64(neg_dst, rhs));
            self.core.text.emit_u32(enc::rorv_64(dst, lhs, neg_dst));
        }
        (MachineIntWidth::I32, MachineIntBinaryOp::DivS) => {
            self.emit_div_s_32(dst, lhs, rhs);
        }
        (MachineIntWidth::I64, MachineIntBinaryOp::DivS) => {
            self.emit_div_s_64(dst, lhs, rhs);
        }
        (MachineIntWidth::I32, MachineIntBinaryOp::DivU) => {
            self.emit_div_u_check(lhs, rhs, MachineIntWidth::I32);
            self.core.text.emit_u32(enc::udiv_32(dst, lhs, rhs));
        }
        (MachineIntWidth::I64, MachineIntBinaryOp::DivU) => {
            self.emit_div_u_check(lhs, rhs, MachineIntWidth::I64);
            self.core.text.emit_u32(enc::udiv_64(dst, lhs, rhs));
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
            let tmp = SCRATCH0;
            self.core.text.emit_u32(enc::udiv_32(tmp, lhs, rhs));
            self.core.text.emit_u32(enc::msub_32(dst, tmp, rhs, lhs));
        }
        (MachineIntWidth::I64, MachineIntBinaryOp::RemU) => {
            self.emit_div_u_check(lhs, rhs, MachineIntWidth::I64);
            let tmp = SCRATCH0;
            self.core.text.emit_u32(enc::udiv_64(tmp, lhs, rhs));
            self.core.text.emit_u32(enc::msub_64(dst, tmp, rhs, lhs));
        }
    };
    Ok(())
}

// ── Division / remainder helpers with trap checks ────────────────────────────

fn emit_div_u_check(&mut self,
_lhs: Arm64Reg, rhs: Arm64Reg, width: MachineIntWidth) {
    // rhs == 0 => trap IntegerDivideByZero
    match width {
        MachineIntWidth::I32 => self.core.text.emit_u32(enc::cmp_reg_32(rhs, Arm64Reg::Xzr)),
        MachineIntWidth::I64 => self.core.text.emit_u32(enc::cmp_reg_64(rhs, Arm64Reg::Xzr)),
    };
    // Branch to a trap stub
    let trap_label = self.core.new_label();
    self.emit_b_cond(enc::Cond::Eq, trap_label);
    self.core
        .deferred_traps
        .push((trap_label, MachineTrapKind::IntegerDivideByZero));
}

fn emit_div_s_32(&mut self,
dst: Arm64Reg, lhs: Arm64Reg, rhs: Arm64Reg) {
    // Check rhs == 0 => IntegerDivideByZero
    self.core.text.emit_u32(enc::cmp_reg_32(rhs, Arm64Reg::Xzr));
    let div_zero_label = self.core.new_label();
    self.emit_b_cond(enc::Cond::Eq, div_zero_label);
    self.core
        .deferred_traps
        .push((div_zero_label, MachineTrapKind::IntegerDivideByZero));

    // Check lhs == i32::MIN && rhs == -1 => IntegerOverflow
    let scratch = SCRATCH0;
    self.materialize_u64(scratch, i32::MIN as u32 as u64);
    self.core.text.emit_u32(enc::cmp_reg_32(lhs, scratch));
    let not_min = self.core.new_label();
    self.emit_b_cond(enc::Cond::Ne, not_min);
    // lhs is MIN, check rhs == -1
    self.materialize_u64(scratch, (-1i32) as u32 as u64);
    self.core.text.emit_u32(enc::cmp_reg_32(rhs, scratch));
    let overflow_label = self.core.new_label();
    self.emit_b_cond(enc::Cond::Eq, overflow_label);
    self.core
        .deferred_traps
        .push((overflow_label, MachineTrapKind::IntegerOverflow));

    self.core.bind_label(not_min);
    self.core.text.emit_u32(enc::sdiv_32(dst, lhs, rhs));
}

fn emit_div_s_64(&mut self,
dst: Arm64Reg, lhs: Arm64Reg, rhs: Arm64Reg) {
    self.core.text.emit_u32(enc::cmp_reg_64(rhs, Arm64Reg::Xzr));
    let div_zero_label = self.core.new_label();
    self.emit_b_cond(enc::Cond::Eq, div_zero_label);
    self.core
        .deferred_traps
        .push((div_zero_label, MachineTrapKind::IntegerDivideByZero));
    let scratch = SCRATCH0;
    self.materialize_u64(scratch, i64::MIN as u64);
    self.core.text.emit_u32(enc::cmp_reg_64(lhs, scratch));
    let not_min = self.core.new_label();
    self.emit_b_cond(enc::Cond::Ne, not_min);
    self.materialize_u64(scratch, (-1i64) as u64);
    self.core.text.emit_u32(enc::cmp_reg_64(rhs, scratch));
    let overflow_label = self.core.new_label();
    self.emit_b_cond(enc::Cond::Eq, overflow_label);
    self.core
        .deferred_traps
        .push((overflow_label, MachineTrapKind::IntegerOverflow));

    self.core.bind_label(not_min);
    self.core.text.emit_u32(enc::sdiv_64(dst, lhs, rhs));
}

fn emit_rem_s_32(&mut self,
dst: Arm64Reg, lhs: Arm64Reg, rhs: Arm64Reg) {
    // Check rhs == 0 => IntegerDivideByZero
    self.core.text.emit_u32(enc::cmp_reg_32(rhs, Arm64Reg::Xzr));
    let div_zero_label = self.core.new_label();
    self.emit_b_cond(enc::Cond::Eq, div_zero_label);
    self.core
        .deferred_traps
        .push((div_zero_label, MachineTrapKind::IntegerDivideByZero));

    // rem = lhs - (lhs / rhs) * rhs  (wrapping, so MIN % -1 = 0, no trap)
    let scratch = SCRATCH0;
    self.core.text.emit_u32(enc::sdiv_32(scratch, lhs, rhs));
    self.core.text.emit_u32(enc::msub_32(dst, scratch, rhs, lhs));
}

fn emit_rem_s_64(&mut self,
dst: Arm64Reg, lhs: Arm64Reg, rhs: Arm64Reg) {
    self.core.text.emit_u32(enc::cmp_reg_64(rhs, Arm64Reg::Xzr));
    let div_zero_label = self.core.new_label();
    self.emit_b_cond(enc::Cond::Eq, div_zero_label);
    self.core
        .deferred_traps
        .push((div_zero_label, MachineTrapKind::IntegerDivideByZero));
    let scratch = SCRATCH0;
    self.core.text.emit_u32(enc::sdiv_64(scratch, lhs, rhs));
    self.core.text.emit_u32(enc::msub_64(dst, scratch, rhs, lhs));
}

// ── Integer compare ──────────────────────────────────────────────────────────

fn emit_int_compare(&mut self,
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
            self.core.text.emit_u32(enc::cset_32(dst, cond));
        }
        MachineIntWidth::I64 => {
            self.core.text.emit_u32(enc::cset_64(dst, cond));
        }
    };
    Ok(())
}

// ── Select ───────────────────────────────────────────────────────────────────

fn emit_select(&mut self,
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
                self.core
                    .text
                    .emit_u32(enc::cmp_imm_64(self.map_gp_reg(reg)?, 0));
                self.core.text.emit_u32(match width {
                    MachineFloatWidth::F32 => enc::fcsel_s(dst_fp, true_fp, false_fp, enc::Cond::Ne),
                    MachineFloatWidth::F64 => enc::fcsel_d(dst_fp, true_fp, false_fp, enc::Cond::Ne),
                });
                self.core.set_fp_reg_width(dst, width)?;
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
                self.core
                    .text
                    .emit_u32(enc::cmp_imm_64(self.map_gp_reg(reg)?, 0));
            }
        }
        let true_reg = self.materialize_value(SCRATCH0, on_true)?;
        let false_reg = self.materialize_value(SCRATCH1, on_false)?;
        // Always use csel_64: GpWord covers both i32 and reference types,
        // and refs need full 64-bit values preserved (e.g. null sentinel).
        self.core
            .text
            .emit_u32(enc::csel_64(dst, true_reg, false_reg, enc::Cond::Ne));
        Ok(())
    }
}

// ── Float operations ─────────────────────────────────────────────────────────

fn emit_float_unary(&mut self,
width: MachineFloatWidth,
    op: MachineFloatUnaryOp,
    dst: MachineReg,
    src: MachineValue,
) -> Result<(), WasmError> {
    let src_fp = self.prepare_float_operand(width, src, SCRATCH0, FP_SCRATCH0)?;
    let result_fp = if self.core.is_fp_reg(dst) {
        let dst_fp = self.map_fp_reg(dst)?;
        self.core.set_fp_reg_width(dst, width)?;
        dst_fp
    } else {
        FP_SCRATCH2
    };
    // Perform the FP operation
    match (width, op) {
        (MachineFloatWidth::F32, MachineFloatUnaryOp::Abs) => {
            self.core.text.emit_u32(enc::fabs_s(result_fp, src_fp))
        }
        (MachineFloatWidth::F64, MachineFloatUnaryOp::Abs) => {
            self.core.text.emit_u32(enc::fabs_d(result_fp, src_fp))
        }
        (MachineFloatWidth::F32, MachineFloatUnaryOp::Neg) => {
            self.core.text.emit_u32(enc::fneg_s(result_fp, src_fp))
        }
        (MachineFloatWidth::F64, MachineFloatUnaryOp::Neg) => {
            self.core.text.emit_u32(enc::fneg_d(result_fp, src_fp))
        }
        (MachineFloatWidth::F32, MachineFloatUnaryOp::Sqrt) => {
            self.core.text.emit_u32(enc::fsqrt_s(result_fp, src_fp))
        }
        (MachineFloatWidth::F64, MachineFloatUnaryOp::Sqrt) => {
            self.core.text.emit_u32(enc::fsqrt_d(result_fp, src_fp))
        }
        (MachineFloatWidth::F32, MachineFloatUnaryOp::Ceil) => {
            self.core.text.emit_u32(enc::frintp_s(result_fp, src_fp))
        }
        (MachineFloatWidth::F64, MachineFloatUnaryOp::Ceil) => {
            self.core.text.emit_u32(enc::frintp_d(result_fp, src_fp))
        }
        (MachineFloatWidth::F32, MachineFloatUnaryOp::Floor) => {
            self.core.text.emit_u32(enc::frintm_s(result_fp, src_fp))
        }
        (MachineFloatWidth::F64, MachineFloatUnaryOp::Floor) => {
            self.core.text.emit_u32(enc::frintm_d(result_fp, src_fp))
        }
        (MachineFloatWidth::F32, MachineFloatUnaryOp::Trunc) => {
            self.core.text.emit_u32(enc::frintz_s(result_fp, src_fp))
        }
        (MachineFloatWidth::F64, MachineFloatUnaryOp::Trunc) => {
            self.core.text.emit_u32(enc::frintz_d(result_fp, src_fp))
        }
        (MachineFloatWidth::F32, MachineFloatUnaryOp::Nearest) => {
            self.core.text.emit_u32(enc::frintn_s(result_fp, src_fp))
        }
        (MachineFloatWidth::F64, MachineFloatUnaryOp::Nearest) => {
            self.core.text.emit_u32(enc::frintn_d(result_fp, src_fp))
        }
    };
    if !self.core.is_fp_reg(dst) {
        let dst_gp = self.map_gp_reg(dst)?;
        match width {
            MachineFloatWidth::F32 => {
                self.core.text.emit_u32(enc::fmov_gp_from_s(dst_gp, result_fp))
            }
            MachineFloatWidth::F64 => {
                self.core.text.emit_u32(enc::fmov_gp_from_d(dst_gp, result_fp))
            }
        };
    }
    Ok(())
}

fn emit_float_binary(&mut self,
width: MachineFloatWidth,
    op: MachineFloatBinaryOp,
    dst: MachineReg,
    lhs: MachineValue,
    rhs: MachineValue,
) -> Result<(), WasmError> {
    let lhs_fp = self.prepare_float_operand(width, lhs, SCRATCH0, FP_SCRATCH0)?;
    let rhs_fp = self.prepare_float_operand(width, rhs, SCRATCH1, FP_SCRATCH1)?;
    let result_fp = if self.core.is_fp_reg(dst) {
        let dst_fp = self.map_fp_reg(dst)?;
        self.core.set_fp_reg_width(dst, width)?;
        dst_fp
    } else {
        FP_SCRATCH2
    };
    match (width, op) {
        (MachineFloatWidth::F32, MachineFloatBinaryOp::Add) => {
            self.core.text.emit_u32(enc::fadd_s(result_fp, lhs_fp, rhs_fp));
        }
        (MachineFloatWidth::F64, MachineFloatBinaryOp::Add) => {
            self.core.text.emit_u32(enc::fadd_d(result_fp, lhs_fp, rhs_fp));
        }
        (MachineFloatWidth::F32, MachineFloatBinaryOp::Sub) => {
            self.core.text.emit_u32(enc::fsub_s(result_fp, lhs_fp, rhs_fp));
        }
        (MachineFloatWidth::F64, MachineFloatBinaryOp::Sub) => {
            self.core.text.emit_u32(enc::fsub_d(result_fp, lhs_fp, rhs_fp));
        }
        (MachineFloatWidth::F32, MachineFloatBinaryOp::Mul) => {
            self.core.text.emit_u32(enc::fmul_s(result_fp, lhs_fp, rhs_fp));
        }
        (MachineFloatWidth::F64, MachineFloatBinaryOp::Mul) => {
            self.core.text.emit_u32(enc::fmul_d(result_fp, lhs_fp, rhs_fp));
        }
        (MachineFloatWidth::F32, MachineFloatBinaryOp::Div) => {
            self.core.text.emit_u32(enc::fdiv_s(result_fp, lhs_fp, rhs_fp));
        }
        (MachineFloatWidth::F64, MachineFloatBinaryOp::Div) => {
            self.core.text.emit_u32(enc::fdiv_d(result_fp, lhs_fp, rhs_fp));
        }
        (MachineFloatWidth::F32, MachineFloatBinaryOp::Min) => {
            // Wasm fmin: NaN if either is NaN. ARM64 FMIN returns non-NaN operand.
            self.core.text.emit_u32(enc::fmin_s(result_fp, lhs_fp, rhs_fp));
            self.core.text.emit_u32(enc::fcmp_s(lhs_fp, rhs_fp));
            let done = self.core.new_label();
            self.emit_b_cond(enc::Cond::Vc, done); // no NaN => FMIN result is correct
            // NaN case: FADD produces NaN from NaN input
            self.core.text.emit_u32(enc::fadd_s(result_fp, lhs_fp, rhs_fp));
            self.core.bind_label(done);
        }
        (MachineFloatWidth::F64, MachineFloatBinaryOp::Min) => {
            self.core.text.emit_u32(enc::fmin_d(result_fp, lhs_fp, rhs_fp));
            self.core.text.emit_u32(enc::fcmp_d(lhs_fp, rhs_fp));
            let done = self.core.new_label();
            self.emit_b_cond(enc::Cond::Vc, done);
            self.core.text.emit_u32(enc::fadd_d(result_fp, lhs_fp, rhs_fp));
            self.core.bind_label(done);
        }
        (MachineFloatWidth::F32, MachineFloatBinaryOp::Max) => {
            self.core.text.emit_u32(enc::fmax_s(result_fp, lhs_fp, rhs_fp));
            self.core.text.emit_u32(enc::fcmp_s(lhs_fp, rhs_fp));
            let done = self.core.new_label();
            self.emit_b_cond(enc::Cond::Vc, done);
            self.core.text.emit_u32(enc::fadd_s(result_fp, lhs_fp, rhs_fp));
            self.core.bind_label(done);
        }
        (MachineFloatWidth::F64, MachineFloatBinaryOp::Max) => {
            self.core.text.emit_u32(enc::fmax_d(result_fp, lhs_fp, rhs_fp));
            self.core.text.emit_u32(enc::fcmp_d(lhs_fp, rhs_fp));
            let done = self.core.new_label();
            self.emit_b_cond(enc::Cond::Vc, done);
            self.core.text.emit_u32(enc::fadd_d(result_fp, lhs_fp, rhs_fp));
            self.core.bind_label(done);
        }
        (MachineFloatWidth::F32, MachineFloatBinaryOp::Copysign) => {
            // copysign: magnitude of lhs, sign of rhs
            let neg_fp = FP_SCRATCH0;
            self.core.text.emit_u32(enc::fabs_s(result_fp, lhs_fp)); // |lhs|
            self.core.text.emit_u32(enc::fneg_s(neg_fp, result_fp)); // -|lhs|
            let rhs_gp = self.materialize_value(SCRATCH1, rhs)?;
            let shift_reg = SCRATCH0;
            self.materialize_u64(shift_reg, 31);
            self.core.text.emit_u32(enc::lsrv_64(shift_reg, rhs_gp, shift_reg));
            self.core.text.emit_u32(enc::cmp_imm_64(shift_reg, 0));
            self.core
                .text
                .emit_u32(enc::fcsel_s(result_fp, neg_fp, result_fp, enc::Cond::Ne));
        }
        (MachineFloatWidth::F64, MachineFloatBinaryOp::Copysign) => {
            let neg_fp = FP_SCRATCH0;
            self.core.text.emit_u32(enc::fabs_d(result_fp, lhs_fp));
            self.core.text.emit_u32(enc::fneg_d(neg_fp, result_fp));
            let rhs_gp = self.materialize_value(SCRATCH1, rhs)?;
            let shift_reg = SCRATCH0;
            self.materialize_u64(shift_reg, 63);
            self.core.text.emit_u32(enc::lsrv_64(shift_reg, rhs_gp, shift_reg));
            self.core.text.emit_u32(enc::cmp_imm_64(shift_reg, 0));
            self.core
                .text
                .emit_u32(enc::fcsel_d(result_fp, neg_fp, result_fp, enc::Cond::Ne));
        }
    };
    if !self.core.is_fp_reg(dst) {
        let dst_gp = self.map_gp_reg(dst)?;
        match width {
            MachineFloatWidth::F32 => {
                self.core.text.emit_u32(enc::fmov_gp_from_s(dst_gp, result_fp))
            }
            MachineFloatWidth::F64 => {
                self.core.text.emit_u32(enc::fmov_gp_from_d(dst_gp, result_fp))
            }
        };
    }
    Ok(())
}

fn emit_float_compare(&mut self,
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
            MachineFloatWidth::F32 => self.core.text.emit_u32(enc::fcmp_s_zero(lhs_fp)),
            MachineFloatWidth::F64 => self.core.text.emit_u32(enc::fcmp_d_zero(lhs_fp)),
        };
    } else {
        let rhs_fp = self.prepare_float_operand(width, rhs, SCRATCH1, FP_SCRATCH1)?;
        match width {
            MachineFloatWidth::F32 => self.core.text.emit_u32(enc::fcmp_s(lhs_fp, rhs_fp)),
            MachineFloatWidth::F64 => self.core.text.emit_u32(enc::fcmp_d(lhs_fp, rhs_fp)),
        };
    }
    // Wasm float comparisons: unordered (NaN) => false for all except Ne
    let cond = map_float_cond(kind);
    self.core.text.emit_u32(enc::cset_32(dst_gp, cond));
    Ok(())
}

/// Resolve FP destination register for convert ops: if dst is FP, map it;
/// otherwise use FP_SCRATCH1 as a temporary.
fn resolve_convert_fp_dst(
    &mut self,
    dst: MachineReg,
    width: MachineFloatWidth,
) -> Result<u32, WasmError> {
    if self.core.is_fp_reg(dst) {
        let dst_fp = self.map_fp_reg(dst)?;
        self.core.set_fp_reg_width(dst, width)?;
        Ok(dst_fp)
    } else {
        Ok(FP_SCRATCH1)
    }
}

// ── Convert ──────────────────────────────────────────────────────────────────

fn emit_convert(&mut self,
op: MachineConvertOp,
    dst: MachineReg,
    src: MachineValue,
) -> Result<(), WasmError> {
    let dst_float_width = convert_result_float_width(op);
    match op {
        // Integer wrapping / extension (no FP involved)
        MachineConvertOp::I32WrapI64 => {
            let src_gp = self.materialize_value(SCRATCH0, src)?;
            let dst_gp = self.map_gp_reg(dst)?;
            // Just mask to 32 bits
            self.core.text.emit_u32(enc::mov_reg_32(dst_gp, src_gp));
        }
        MachineConvertOp::I64ExtendI32S => {
            let src_gp = self.materialize_value(SCRATCH0, src)?;
            let dst_gp = self.map_gp_reg(dst)?;
            self.core.text.emit_u32(enc::sxtw(dst_gp, src_gp));
        }
        MachineConvertOp::I64ExtendI32U => {
            let src_gp = self.materialize_value(SCRATCH0, src)?;
            let dst_gp = self.map_gp_reg(dst)?;
            self.core.text.emit_u32(enc::mov_reg_32(dst_gp, src_gp));
        }
        MachineConvertOp::I32ReinterpretF32 => {
            let dst_gp = self.map_gp_reg(dst)?;
            if let MachineValue::Reg(src_reg) = src {
                if self.core.is_fp_reg(src_reg) {
                    let src_fp = self.map_fp_reg(src_reg)?;
                    self.core.text.emit_u32(enc::fmov_gp_from_s(dst_gp, src_fp));
                } else {
                    let src_gp = self.map_gp_reg(src_reg)?;
                    if dst_gp != src_gp {
                        self.core.text.emit_u32(enc::mov_reg_32(dst_gp, src_gp));
                    }
                }
            } else {
                let src_gp = self.materialize_value(SCRATCH0, src)?;
                if dst_gp != src_gp {
                    self.core.text.emit_u32(enc::mov_reg_32(dst_gp, src_gp));
                }
            }
        }
        MachineConvertOp::I64ReinterpretF64 => {
            let dst_gp = self.map_gp_reg(dst)?;
            if let MachineValue::Reg(src_reg) = src {
                if self.core.is_fp_reg(src_reg) {
                    let src_fp = self.map_fp_reg(src_reg)?;
                    self.core.text.emit_u32(enc::fmov_gp_from_d(dst_gp, src_fp));
                } else {
                    let src_gp = self.map_gp_reg(src_reg)?;
                    if dst_gp != src_gp {
                        self.core.text.emit_u32(enc::mov_reg_64(dst_gp, src_gp));
                    }
                }
            } else {
                let src_gp = self.materialize_value(SCRATCH0, src)?;
                if dst_gp != src_gp {
                    self.core.text.emit_u32(enc::mov_reg_64(dst_gp, src_gp));
                }
            }
        }
        MachineConvertOp::F32ReinterpretI32 | MachineConvertOp::F64ReinterpretI64 => {
            let src_gp = self.materialize_value(SCRATCH0, src)?;
            let width = dst_float_width.expect("float reinterpret width");
            let dst_fp = self.resolve_convert_fp_dst(dst, width)?;
            self.core.text.emit_u32(match width {
                MachineFloatWidth::F32 => enc::fmov_s_from_gp(dst_fp, src_gp),
                MachineFloatWidth::F64 => enc::fmov_d_from_gp(dst_fp, src_gp),
            });
            if !self.core.is_fp_reg(dst) {
                let dst_gp = self.map_gp_reg(dst)?;
                self.core.text.emit_u32(enc::mov_reg_64(dst_gp, src_gp));
            }
        }
        // Float promotion / demotion
        MachineConvertOp::F64PromoteF32 => {
            let src_fp =
                self.prepare_float_operand(MachineFloatWidth::F32, src, SCRATCH0, FP_SCRATCH0)?;
            let dst_fp = self.resolve_convert_fp_dst(dst, MachineFloatWidth::F64)?;
            self.core.text.emit_u32(enc::fcvt_d_from_s(dst_fp, src_fp));
            if !self.core.is_fp_reg(dst) {
                let dst_gp = self.map_gp_reg(dst)?;
                self.core.text.emit_u32(enc::fmov_gp_from_d(dst_gp, dst_fp));
            }
        }
        MachineConvertOp::F32DemoteF64 => {
            let src_fp =
                self.prepare_float_operand(MachineFloatWidth::F64, src, SCRATCH0, FP_SCRATCH0)?;
            let dst_fp = self.resolve_convert_fp_dst(dst, MachineFloatWidth::F32)?;
            self.core.text.emit_u32(enc::fcvt_s_from_d(dst_fp, src_fp));
            if !self.core.is_fp_reg(dst) {
                let dst_gp = self.map_gp_reg(dst)?;
                self.core.text.emit_u32(enc::fmov_gp_from_s(dst_gp, dst_fp));
            }
        }
        // Int -> Float conversions
        MachineConvertOp::F32ConvertI32S => {
            let src_gp = self.materialize_value(SCRATCH0, src)?;
            let dst_fp = self.resolve_convert_fp_dst(dst, MachineFloatWidth::F32)?;
            self.core.text.emit_u32(enc::scvtf_s_32(dst_fp, src_gp));
            if !self.core.is_fp_reg(dst) {
                let dst_gp = self.map_gp_reg(dst)?;
                self.core.text.emit_u32(enc::fmov_gp_from_s(dst_gp, dst_fp));
            }
        }
        MachineConvertOp::F32ConvertI32U => {
            let src_gp = self.materialize_value(SCRATCH0, src)?;
            let dst_fp = self.resolve_convert_fp_dst(dst, MachineFloatWidth::F32)?;
            self.core.text.emit_u32(enc::ucvtf_s_32(dst_fp, src_gp));
            if !self.core.is_fp_reg(dst) {
                let dst_gp = self.map_gp_reg(dst)?;
                self.core.text.emit_u32(enc::fmov_gp_from_s(dst_gp, dst_fp));
            }
        }
        MachineConvertOp::F32ConvertI64S => {
            let src_gp = self.materialize_value(SCRATCH0, src)?;
            let dst_fp = self.resolve_convert_fp_dst(dst, MachineFloatWidth::F32)?;
            self.core.text.emit_u32(enc::scvtf_s_64(dst_fp, src_gp));
            if !self.core.is_fp_reg(dst) {
                let dst_gp = self.map_gp_reg(dst)?;
                self.core.text.emit_u32(enc::fmov_gp_from_s(dst_gp, dst_fp));
            }
        }
        MachineConvertOp::F32ConvertI64U => {
            let src_gp = self.materialize_value(SCRATCH0, src)?;
            let dst_fp = self.resolve_convert_fp_dst(dst, MachineFloatWidth::F32)?;
            self.core.text.emit_u32(enc::ucvtf_s_64(dst_fp, src_gp));
            if !self.core.is_fp_reg(dst) {
                let dst_gp = self.map_gp_reg(dst)?;
                self.core.text.emit_u32(enc::fmov_gp_from_s(dst_gp, dst_fp));
            }
        }
        MachineConvertOp::F64ConvertI32S => {
            let src_gp = self.materialize_value(SCRATCH0, src)?;
            let dst_fp = self.resolve_convert_fp_dst(dst, MachineFloatWidth::F64)?;
            self.core.text.emit_u32(enc::scvtf_d_32(dst_fp, src_gp));
            if !self.core.is_fp_reg(dst) {
                let dst_gp = self.map_gp_reg(dst)?;
                self.core.text.emit_u32(enc::fmov_gp_from_d(dst_gp, dst_fp));
            }
        }
        MachineConvertOp::F64ConvertI32U => {
            let src_gp = self.materialize_value(SCRATCH0, src)?;
            let dst_fp = self.resolve_convert_fp_dst(dst, MachineFloatWidth::F64)?;
            self.core.text.emit_u32(enc::ucvtf_d_32(dst_fp, src_gp));
            if !self.core.is_fp_reg(dst) {
                let dst_gp = self.map_gp_reg(dst)?;
                self.core.text.emit_u32(enc::fmov_gp_from_d(dst_gp, dst_fp));
            }
        }
        MachineConvertOp::F64ConvertI64S => {
            let src_gp = self.materialize_value(SCRATCH0, src)?;
            let dst_fp = self.resolve_convert_fp_dst(dst, MachineFloatWidth::F64)?;
            self.core.text.emit_u32(enc::scvtf_d_64(dst_fp, src_gp));
            if !self.core.is_fp_reg(dst) {
                let dst_gp = self.map_gp_reg(dst)?;
                self.core.text.emit_u32(enc::fmov_gp_from_d(dst_gp, dst_fp));
            }
        }
        MachineConvertOp::F64ConvertI64U => {
            let src_gp = self.materialize_value(SCRATCH0, src)?;
            let dst_fp = self.resolve_convert_fp_dst(dst, MachineFloatWidth::F64)?;
            self.core.text.emit_u32(enc::ucvtf_d_64(dst_fp, src_gp));
            if !self.core.is_fp_reg(dst) {
                let dst_gp = self.map_gp_reg(dst)?;
                self.core.text.emit_u32(enc::fmov_gp_from_d(dst_gp, dst_fp));
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
        // Saturating truncations -- inline via native fcvtzs/fcvtzu
        // ARM64 fcvtzs/fcvtzu already matches Wasm saturating semantics:
        // NaN->0, overflow->clamp to min/max.
        MachineConvertOp::I32TruncSatF32S => {
            let src_gp = self.materialize_value(SCRATCH0, src)?;
            let fp_tmp = FP_SCRATCH0;
            self.core.text.emit_u32(enc::fmov_s_from_gp(fp_tmp, src_gp));
            let dst_gp = self.map_gp_reg(dst)?;
            self.core.text.emit_u32(enc::fcvtzs_32_s(dst_gp, fp_tmp));
        }
        MachineConvertOp::I32TruncSatF32U => {
            let src_gp = self.materialize_value(SCRATCH0, src)?;
            let fp_tmp = FP_SCRATCH0;
            self.core.text.emit_u32(enc::fmov_s_from_gp(fp_tmp, src_gp));
            let dst_gp = self.map_gp_reg(dst)?;
            self.core.text.emit_u32(enc::fcvtzu_32_s(dst_gp, fp_tmp));
        }
        MachineConvertOp::I32TruncSatF64S => {
            let src_gp = self.materialize_value(SCRATCH0, src)?;
            let fp_tmp = FP_SCRATCH0;
            self.core.text.emit_u32(enc::fmov_d_from_gp(fp_tmp, src_gp));
            let dst_gp = self.map_gp_reg(dst)?;
            self.core.text.emit_u32(enc::fcvtzs_32_d(dst_gp, fp_tmp));
        }
        MachineConvertOp::I32TruncSatF64U => {
            let src_gp = self.materialize_value(SCRATCH0, src)?;
            let fp_tmp = FP_SCRATCH0;
            self.core.text.emit_u32(enc::fmov_d_from_gp(fp_tmp, src_gp));
            let dst_gp = self.map_gp_reg(dst)?;
            self.core.text.emit_u32(enc::fcvtzu_32_d(dst_gp, fp_tmp));
        }
        MachineConvertOp::I64TruncSatF32S => {
            let src_gp = self.materialize_value(SCRATCH0, src)?;
            let fp_tmp = FP_SCRATCH0;
            self.core.text.emit_u32(enc::fmov_s_from_gp(fp_tmp, src_gp));
            let dst_gp = self.map_gp_reg(dst)?;
            self.core.text.emit_u32(enc::fcvtzs_64_s(dst_gp, fp_tmp));
        }
        MachineConvertOp::I64TruncSatF32U => {
            let src_gp = self.materialize_value(SCRATCH0, src)?;
            let fp_tmp = FP_SCRATCH0;
            self.core.text.emit_u32(enc::fmov_s_from_gp(fp_tmp, src_gp));
            let dst_gp = self.map_gp_reg(dst)?;
            self.core.text.emit_u32(enc::fcvtzu_64_s(dst_gp, fp_tmp));
        }
        MachineConvertOp::I64TruncSatF64S => {
            let src_gp = self.materialize_value(SCRATCH0, src)?;
            let fp_tmp = FP_SCRATCH0;
            self.core.text.emit_u32(enc::fmov_d_from_gp(fp_tmp, src_gp));
            let dst_gp = self.map_gp_reg(dst)?;
            self.core.text.emit_u32(enc::fcvtzs_64_d(dst_gp, fp_tmp));
        }
        MachineConvertOp::I64TruncSatF64U => {
            let src_gp = self.materialize_value(SCRATCH0, src)?;
            let fp_tmp = FP_SCRATCH0;
            self.core.text.emit_u32(enc::fmov_d_from_gp(fp_tmp, src_gp));
            let dst_gp = self.map_gp_reg(dst)?;
            self.core.text.emit_u32(enc::fcvtzu_64_d(dst_gp, fp_tmp));
        }
    }
    Ok(())
}

fn emit_trapping_trunc(&mut self,
op: MachineConvertOp,
    dst: Arm64Reg,
    src: Arm64Reg,
) -> Result<(), WasmError> {
    use super::helpers::arm64_trapping_trunc;

    // Call the helper: extern "C" fn(ctx, src_bits) -> status
    self.core.text.emit_u32(enc::mov_reg_64(
        Arm64Reg::X0,
        map_fixed_reg(MACHINE_CTX_REG),
    ));
    self.core.text.emit_u32(enc::mov_reg_64(Arm64Reg::X1, src));
    self.materialize_u64(Arm64Reg::X2, convert_op_code(op));
    let scratch = SCRATCH0;
    self.materialize_u64(scratch, arm64_trapping_trunc as usize as u64);
    self.core.text.emit_u32(enc::blr(scratch));
    // X0 = status (0 = ok), X1 = result value
    let return_error_label = self.core.return_error_label;
    self.emit_cbnz(Arm64Reg::X0, return_error_label);
    self.core.text.emit_u32(enc::mov_reg_64(dst, Arm64Reg::X1));
    Ok(())
}

} // impl Arm64Backend (inst.rs)

// ── Free helper (not a method — operates on TextEmitter directly) ────────────

pub(super) fn materialize_u64_into(text: &mut TextEmitter, dst: Arm64Reg, value: u64) {
    if value == 0 {
        text.emit_u32(enc::mov_reg_64(dst, Arm64Reg::Xzr));
        return;
    }
    let chunks = [
        (value & 0xffff) as u16,
        ((value >> 16) & 0xffff) as u16,
        ((value >> 32) & 0xffff) as u16,
        ((value >> 48) & 0xffff) as u16,
    ];
    let mut first = true;
    for (i, &chunk) in chunks.iter().enumerate() {
        if chunk != 0 || first && i == 3 {
            if first {
                text.emit_u32(enc::movz_64(dst, chunk, (i as u32) * 16));
                first = false;
            } else {
                text.emit_u32(enc::movk_64(dst, chunk, (i as u32) * 16));
            }
        }
    }
    if first {
        text.emit_u32(enc::movz_64(dst, 0, 0));
    }
}
