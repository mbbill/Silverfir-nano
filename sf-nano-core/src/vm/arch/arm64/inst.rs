//! ARM64 instruction emission: emit_inst dispatch and individual instruction methods.

use crate::error::WasmError;
use crate::vm::machine::machine_ir::{
    MachineAddr, MachineBlockParam, MachineCompareKind, MachineConvertOp, MachineFloatBinaryOp,
    MachineFloatUnaryOp, MachineFloatWidth, MachineFuncId, MachineFunctionAbi, MachineIndexExtend,
    MachineInst, MachineInstKind, MachineIntBinaryOp, MachineIntUnaryOp, MachineIntWidth,
    MachineLoadExtension, MachineMemWidth, MachineReg, MachineShiftOp, MachineSign,
    MachineStorageType, MachineTrapKind, MachineValue,
};

use super::abi::{fp_machine_reg, map_reg};
use super::operands::{PreparedFp, PreparedGp};
use super::{
    abi, enc,
    reg::{Arm64FpReg, Arm64Reg},
};

use super::backend::BranchFixup;
use super::fusion::{cmp_imm_inst, int_binary_imm_inst, map_float_cond, map_int_cond};
use crate::vm::arch::common::helpers::{convert_result_float_width, mem_width_bytes};
use crate::vm::arch::common::scratch_pool::ScratchPool;
use crate::vm::arch::common::text_emitter::TextEmitter;
use crate::vm::arch::common::types::ParallelSource;
use crate::vm::backend::BackendConfig;
use crate::vm::machine::machine_ir::{fp_reg_index, is_fp_reg};
use crate::vm::runtime::preserved::{io as preserved_io, op as preserved_op};

// ── Operand preparation (free functions) ─────────────────────────────────────
//
// `prepare_gp` and `prepare_fp` are **free functions** instead of methods
// on `Arm64Backend`. This is a deliberate Rust borrow-checker workaround:
//
// The returned `PreparedGp` holds an RAII guard that borrows `&ScratchPool`.
// If prepare were a `&mut self` method, the guard's lifetime would lock the
// entire backend, preventing a second prepare call or any `text.emit_u32()`.
//
// As free functions taking disjoint field references, Rust can see that the
// returned guard borrows only the pool (`&ScratchPool`, shared via `Cell`),
// while `&mut TextEmitter` is reborrowed only for the call's duration.
// This lets multiple `PreparedGp` values coexist — exactly what two-operand
// patterns like `lower_int_binary` require.

// materialize_u64_into is defined at the end of this file as a free function.

/// Map a MachineReg to a physical GP register, rejecting FP regs.
pub(super) fn map_gp(config: BackendConfig, reg: MachineReg) -> Result<Arm64Reg, WasmError> {
    if is_fp_reg(reg, config) {
        return Err(WasmError::invalid(alloc::format!(
            "expected GP register, got FP machine reg {}",
            reg.0
        )));
    }
    abi::map_reg(reg)
}

/// Map a MachineReg to a physical FP register.
fn map_fp(config: BackendConfig, reg: MachineReg) -> Result<Arm64FpReg, WasmError> {
    let index = fp_reg_index(reg, config).ok_or_else(|| {
        WasmError::invalid(alloc::format!(
            "expected FP register, got machine reg {}",
            reg.0
        ))
    })?;
    abi::fp_machine_reg(index).ok_or_else(|| {
        WasmError::invalid(alloc::format!(
            "arm64 has no FP mapping for machine reg {}",
            reg.0
        ))
    })
}

/// Prepare a MachineValue as a GP register.
///
/// - `Reg(gp)` → `Mapped(physical_gp)` — no scratch used.
/// - `Reg(fp)` → scratch alloc + fmov.
/// - `Imm64`   → scratch alloc + materialize.
pub(super) fn prepare_gp<'p>(
    config: BackendConfig,
    fp_widths: &[Option<MachineFloatWidth>],
    text: &mut TextEmitter,
    pool: &'p ScratchPool<Arm64Reg, 2>,
    value: MachineValue,
) -> Result<PreparedGp<'p>, WasmError> {
    match value {
        MachineValue::Reg(reg) if is_fp_reg(reg, config) => {
            let scratch = pool.scoped_alloc();
            let src_fp = map_fp(config, reg)?;
            let index = fp_reg_index(reg, config).unwrap();
            let width = fp_widths.get(index).and_then(|w| *w).ok_or_else(|| {
                WasmError::invalid(alloc::format!(
                    "missing float-width for machine reg {}",
                    reg.0
                ))
            })?;
            text.emit_u32(match width {
                MachineFloatWidth::F32 => enc::fmov_gp_from_s(*scratch, src_fp),
                MachineFloatWidth::F64 => enc::fmov_gp_from_d(*scratch, src_fp),
            });
            Ok(PreparedGp::Scratch(scratch))
        }
        MachineValue::Reg(reg) => Ok(PreparedGp::Mapped(map_gp(config, reg)?)),
        MachineValue::ReservedReg(reg) => Err(WasmError::internal(alloc::format!(
            "arm64 prepare_gp cannot consume reserved cache register {} as a real value",
            reg.0
        ))),
        MachineValue::Imm64(v) => {
            let scratch = pool.scoped_alloc();
            materialize_u64_into(text, *scratch, v);
            Ok(PreparedGp::Scratch(scratch))
        }
    }
}

/// Prepare a MachineValue as an FP register.
///
/// - `Reg(fp)` → `Mapped(physical_fp)` — no scratch used.
/// - otherwise → GP scratch + materialize + fmov into FP scratch.
///   The GP scratch is released before returning.
pub(super) fn prepare_fp<'p>(
    config: BackendConfig,
    fp_widths: &[Option<MachineFloatWidth>],
    text: &mut TextEmitter,
    gp_pool: &ScratchPool<Arm64Reg, 2>,
    fp_pool: &'p ScratchPool<Arm64FpReg, 3>,
    width: MachineFloatWidth,
    value: MachineValue,
) -> Result<PreparedFp<'p>, WasmError> {
    if let MachineValue::Reg(reg) = value {
        if is_fp_reg(reg, config) {
            return Ok(PreparedFp::Mapped(map_fp(config, reg)?));
        }
    }
    let gp = prepare_gp(config, fp_widths, text, gp_pool, value)?;
    let fp_scratch = fp_pool.scoped_alloc();
    text.emit_u32(match width {
        MachineFloatWidth::F32 => enc::fmov_s_from_gp(*fp_scratch, *gp),
        MachineFloatWidth::F64 => enc::fmov_d_from_gp(*fp_scratch, *gp),
    });
    // gp dropped here — GP scratch slot freed immediately
    Ok(PreparedFp::Scratch(fp_scratch))
}

impl<'a> super::backend::Arm64Backend<'a> {
    // ── Register mapping ─────────────────────────────────────────────────

    pub(super) fn map_gp_reg(&self, reg: MachineReg) -> Result<Arm64Reg, WasmError> {
        self.core.validate_gp_reg(reg)?;
        map_reg(reg)
    }

    pub(super) fn map_fp_reg(&self, reg: MachineReg) -> Result<Arm64FpReg, WasmError> {
        let index = self.core.fp_reg_index(reg)?;
        fp_machine_reg(index).ok_or_else(|| {
            WasmError::invalid(alloc::format!(
                "arm64 has no physical FP mapping for machine reg {}",
                reg.0
            ))
        })
    }

    // ── Constant materialization ─────────────────────────────────────────

    pub(super) fn materialize_u64(&mut self, dst: Arm64Reg, value: u64) {
        materialize_u64_into(&mut self.core.text, dst, value);
    }

    // ── Branch emission ──────────────────────────────────────────────────

    pub(super) fn lower_b(&mut self, label: usize) {
        let inst_offset = self.core.text.emit_u32(enc::b(0));
        self.fixups.push(BranchFixup {
            inst_offset,
            label,
            kind: super::backend::BranchFixupKind::B,
        });
    }

    /// `bl <label>` — branch with link, populates LR. Used by the public-
    /// entry caller stub and by `CallDirect` lowering. Resolved at
    /// `patch_fixups()` time.
    pub(super) fn lower_bl(&mut self, label: usize) {
        let inst_offset = self.core.text.emit_u32(enc::bl(0));
        self.fixups.push(BranchFixup {
            inst_offset,
            label,
            kind: super::backend::BranchFixupKind::Bl,
        });
    }

    pub(super) fn lower_b_cond(&mut self, cond: enc::Cond, label: usize) {
        let inst_offset = self.core.text.emit_u32(enc::b_cond(cond, 0));
        self.fixups.push(BranchFixup {
            inst_offset,
            label,
            kind: super::backend::BranchFixupKind::BCond(cond),
        });
    }

    pub(super) fn lower_cbnz(&mut self, reg: Arm64Reg, label: usize) {
        let inst_offset = self.core.text.emit_u32(enc::cbnz_64(reg, 0));
        self.fixups.push(BranchFixup {
            inst_offset,
            label,
            kind: super::backend::BranchFixupKind::Cbnz(reg),
        });
    }

    pub(super) fn lower_cbz(&mut self, reg: Arm64Reg, label: usize) {
        let inst_offset = self.core.text.emit_u32(enc::cbz_64(reg, 0));
        self.fixups.push(BranchFixup {
            inst_offset,
            label,
            kind: super::backend::BranchFixupKind::Cbz(reg),
        });
    }

    /// Emit a CMP/CMP-imm for two integer operands, setting flags.
    pub(super) fn lower_cmp_values(
        &mut self,
        width: MachineIntWidth,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        if let (MachineValue::Reg(lhs_reg), MachineValue::Imm64(imm)) = (lhs, rhs) {
            let lhs_phys = map_gp(self.core.compiled.backend(), lhs_reg)?;
            if let Some(inst) = cmp_imm_inst(width, lhs_phys, imm) {
                self.core.text.emit_u32(inst);
                return Ok(());
            }
        }
        let lhs = prepare_gp(
            self.core.compiled.backend(),
            &self.core.fp_reg_widths,
            &mut self.core.text,
            &self.gp_scratch,
            lhs,
        )?;
        let rhs = prepare_gp(
            self.core.compiled.backend(),
            &self.core.fp_reg_widths,
            &mut self.core.text,
            &self.gp_scratch,
            rhs,
        )?;
        match width {
            MachineIntWidth::I32 => self.core.text.emit_u32(enc::cmp_reg_32(*lhs, *rhs)),
            MachineIntWidth::I64 => self.core.text.emit_u32(enc::cmp_reg_64(*lhs, *rhs)),
        };
        Ok(())
    }

    /// Emit a TST (AND-then-set-flags) for src and mask operands.
    pub(super) fn lower_tst_values(
        &mut self,
        width: MachineIntWidth,
        src: MachineValue,
        mask: MachineValue,
    ) -> Result<(), WasmError> {
        let src_gp = match src {
            MachineValue::Reg(r) => PreparedGp::Mapped(map_gp(self.core.compiled.backend(), r)?),
            MachineValue::ReservedReg(reg) => {
                return Err(WasmError::internal(alloc::format!(
                    "arm64 lower_tst_values cannot consume reserved cache register {} as a source",
                    reg.0
                )));
            }
            MachineValue::Imm64(imm) => prepare_gp(
                self.core.compiled.backend(),
                &self.core.fp_reg_widths,
                &mut self.core.text,
                &self.gp_scratch,
                MachineValue::Imm64(imm),
            )?,
        };
        let src_phys = *src_gp;
        match mask {
            MachineValue::Imm64(imm) => {
                let inst = match width {
                    MachineIntWidth::I32 => enc::tst_imm_32(src_phys, imm as u32),
                    MachineIntWidth::I64 => enc::tst_imm_64(src_phys, imm),
                };
                if let Some(i) = inst {
                    self.core.text.emit_u32(i);
                } else {
                    let scratch = prepare_gp(
                        self.core.compiled.backend(),
                        &self.core.fp_reg_widths,
                        &mut self.core.text,
                        &self.gp_scratch,
                        MachineValue::Imm64(imm),
                    )?;
                    match width {
                        MachineIntWidth::I32 => {
                            self.core.text.emit_u32(enc::tst_reg_32(src_phys, *scratch));
                        }
                        MachineIntWidth::I64 => {
                            self.core.text.emit_u32(enc::tst_reg_64(src_phys, *scratch));
                        }
                    }
                }
            }
            MachineValue::Reg(mask_reg) => {
                let mask_phys = map_gp(self.core.compiled.backend(), mask_reg)?;
                match width {
                    MachineIntWidth::I32 => {
                        self.core
                            .text
                            .emit_u32(enc::tst_reg_32(src_phys, mask_phys));
                    }
                    MachineIntWidth::I64 => {
                        self.core
                            .text
                            .emit_u32(enc::tst_reg_64(src_phys, mask_phys));
                    }
                }
            }
            MachineValue::ReservedReg(reg) => {
                return Err(WasmError::internal(alloc::format!(
                    "arm64 lower_tst_values cannot consume reserved cache register {} as a mask",
                    reg.0
                )));
            }
        }
        Ok(())
    }

    /// Look up runtime metadata for a machine function.
    pub(super) fn runtime_for(
        &self,
        func_id: MachineFuncId,
    ) -> Result<&MachineFunctionAbi, WasmError> {
        self.core.runtime_for(func_id)
    }

    // ── Instruction dispatch ─────────────────────────────────────────────

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
                addr, width, src, ..
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
            MachineInstKind::TrapIf { kind, cond } => self.lower_trap_if(*kind, cond),
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
            MachineInstKind::IndexedLoad {
                dst,
                base,
                index,
                index_extend,
                offset,
                width,
                extension,
            } => {
                let uxtw = *index_extend == MachineIndexExtend::ZeroExtend32;
                if *offset == 0 {
                    self.lower_indexed_load(*dst, *base, *index, *width, *extension, false, uxtw)
                } else {
                    self.lower_indexed_load_with_offset(
                        *dst, *base, *index, *offset, *width, *extension, uxtw,
                    )
                }
            }
            MachineInstKind::IndexedStore {
                base,
                index,
                index_extend,
                offset,
                width,
                src,
            } => {
                let uxtw = *index_extend == MachineIndexExtend::ZeroExtend32;
                if *offset == 0 {
                    self.lower_indexed_store(*base, *index, *width, *src, false, uxtw)
                } else {
                    self.lower_indexed_store_with_offset(*base, *index, *offset, *width, *src, uxtw)
                }
            }
            MachineInstKind::BitfieldExtractU {
                width,
                dst,
                src,
                lsb,
                bits,
            } => self.lower_bitfield_extract_u(*width, *dst, *src, *lsb, *bits),
            MachineInstKind::IntBinaryShifted {
                width,
                op,
                dst,
                lhs,
                rhs,
                shift,
                amount,
            } => self.lower_int_binary_shifted(*width, *op, *dst, *lhs, *rhs, *shift, *amount),
            MachineInstKind::TestBits {
                width,
                kind,
                dst,
                src,
                mask,
            } => self.lower_test_bits(*width, *kind, *dst, *src, *mask),
            MachineInstKind::MemoryGrow {
                mem_idx,
                dst,
                delta,
            } => self.lower_memory_grow(*mem_idx, *dst, *delta),
            MachineInstKind::MemoryFill {
                mem_idx,
                dest,
                val,
                len,
            } => self.lower_memory_fill(*mem_idx, *dest, *val, *len),
            MachineInstKind::MemoryCopy {
                dst_mem,
                src_mem,
                dest,
                src,
                len,
            } => self.lower_memory_copy(*dst_mem, *src_mem, *dest, *src, *len),
            MachineInstKind::MemoryInit {
                mem_idx,
                data_idx,
                dest,
                src,
                len,
            } => self.lower_memory_init(*mem_idx, *data_idx, *dest, *src, *len),
            MachineInstKind::DataDrop { data_idx } => self.lower_data_drop(*data_idx),
            MachineInstKind::TableGrow {
                table_idx,
                dst,
                init_val,
                delta,
            } => self.lower_table_grow(*table_idx, *dst, *init_val, *delta),
            MachineInstKind::TableFill {
                table_idx,
                start,
                val,
                len,
            } => self.lower_table_fill(*table_idx, *start, *val, *len),
            MachineInstKind::TableCopy {
                dst_tbl,
                src_tbl,
                dest,
                src,
                len,
            } => self.lower_table_copy(*dst_tbl, *src_tbl, *dest, *src, *len),
            MachineInstKind::TableInit {
                table_idx,
                elem_idx,
                dest,
                src,
                len,
            } => self.lower_table_init(*table_idx, *elem_idx, *dest, *src, *len),
            MachineInstKind::ElemDrop { elem_idx } => self.lower_elem_drop(*elem_idx),
            // 32-bit legalized instructions -- should not reach arm64 codegen.
            MachineInstKind::Int64PairBinary { .. }
            | MachineInstKind::Int64PairUnary { .. }
            | MachineInstKind::Int64PairDivRem { .. }
            | MachineInstKind::Int64PairShift { .. }
            | MachineInstKind::ConvertI64PairToFloat { .. }
            | MachineInstKind::Int64PairCompare { .. }
            | MachineInstKind::ConvertFloatToI64Pair { .. }
            | MachineInstKind::ReinterpretF64ToI64Pair { .. }
            | MachineInstKind::ReinterpretI64PairToF64 { .. } => Err(WasmError::internal(
                "arm64 backend received 32-bit legalized instruction".into(),
            )),
        }
    }

    /// Emit a parallel-move source -> destination.
    pub(super) fn lower_source_move_dispatch(
        &mut self,
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
                            MachineFloatWidth::F32 => {
                                self.core.text.emit_u32(enc::fmov_gp_from_s(dst_gp, src_fp))
                            }
                            MachineFloatWidth::F64 => {
                                self.core.text.emit_u32(enc::fmov_gp_from_d(dst_gp, src_fp))
                            }
                        };
                    } else {
                        let src_gp = self.map_gp_reg(src_reg)?;
                        self.core.text.emit_u32(enc::mov_reg_64(dst_gp, src_gp));
                    }
                }
            }
            ParallelSource::ReservedReg(reg) => {
                return Err(WasmError::internal(alloc::format!(
                    "arm64 received non-identity reserved cache edge move into {} from {}",
                    dst.reg.0,
                    reg.0
                )));
            }
            ParallelSource::Imm(value) => {
                if let Some(width) = dst.ty.float_width() {
                    let dst_fp = self.map_fp_reg(dst.reg)?;
                    let scratch = *self.gp_scratch.scoped_alloc();
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
            ParallelSource::GpTemp(id) => {
                let temp = self.gp_scratch.reg(id);
                self.core
                    .text
                    .emit_u32(enc::mov_reg_64(self.map_gp_reg(dst.reg)?, temp));
            }
            ParallelSource::FpTemp(id, width) => {
                let temp = self.fp_scratch.reg(id);
                let dst_fp = self.map_fp_reg(dst.reg)?;
                self.core.text.emit_u32(match width {
                    MachineFloatWidth::F32 => enc::fmov_s(dst_fp, temp),
                    MachineFloatWidth::F64 => enc::fmov_d(dst_fp, temp),
                });
                self.core.set_fp_reg_width(dst.reg, width)?;
            }
        }
        Ok(())
    }

    // ── Move / Float constant ────────────────────────────────────────────────────

    fn lower_move(
        &mut self,
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
                MachineValue::ReservedReg(reg) => {
                    return Err(WasmError::internal(alloc::format!(
                        "arm64 lower_move cannot read reserved cache register {} as an FP source",
                        reg.0
                    )));
                }
                MachineValue::Imm64(value) => {
                    let scratch = *self.gp_scratch.scoped_alloc();
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
                MachineValue::ReservedReg(reg) => {
                    return Err(WasmError::internal(alloc::format!(
                        "arm64 lower_move cannot read reserved cache register {} as a GP source",
                        reg.0
                    )));
                }
                MachineValue::Imm64(value) => {
                    self.materialize_u64(dst_gp, value);
                    Ok(())
                }
            }
        }
    }

    fn lower_float_const(
        &mut self,
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
        let scratch = *self.gp_scratch.scoped_alloc();
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
    fn lower_addr_into(&mut self, dst: Arm64Reg, addr: MachineAddr) -> Result<(), WasmError> {
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
        // Large offset: use a second scratch (dst may already be a scratch).
        let off_scratch = *self.gp_scratch.scoped_alloc();
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
    fn lower_add_imm_to_reg(&mut self, reg: Arm64Reg, off: i64) {
        if off > 0 && off < 4096 {
            self.core
                .text
                .emit_u32(enc::add_imm_64(reg, reg, off as u32));
        } else if off < 0 && -off < 4096 {
            self.core
                .text
                .emit_u32(enc::sub_imm_64(reg, reg, (-off) as u32));
        } else {
            let tmp = *self.gp_scratch.scoped_alloc();
            materialize_u64_into(&mut self.core.text, tmp, off as u64);
            self.core.text.emit_u32(enc::add_reg_64(reg, reg, tmp));
        }
    }

    // ── Load / Store ─────────────────────────────────────────────────────────────

    fn lower_load(
        &mut self,
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
            let addr_scratch = *self.gp_scratch.scoped_alloc();
            self.lower_addr_into(addr_scratch, addr)?;
            self.core
                .text
                .emit_u32(match (tracked_width, width, extension) {
                (MachineFloatWidth::F32, MachineMemWidth::U32, MachineLoadExtension::None)
                | (
                    MachineFloatWidth::F32,
                    MachineMemWidth::U32,
                    MachineLoadExtension::ZeroExtend,
                ) => enc::ldr_s_base(dst_fp, addr_scratch),
                (MachineFloatWidth::F64, MachineMemWidth::U64, MachineLoadExtension::None)
                | (
                    MachineFloatWidth::F64,
                    MachineMemWidth::U64,
                    MachineLoadExtension::ZeroExtend,
                ) => enc::ldr_d_base(dst_fp, addr_scratch),
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
        let addr_scratch = *self.gp_scratch.scoped_alloc();
        self.lower_addr_into(addr_scratch, addr)?;
        let inst = match (width, extension) {
            (MachineMemWidth::U8, MachineLoadExtension::None)
            | (MachineMemWidth::U8, MachineLoadExtension::ZeroExtend) => {
                enc::ldrb_base(dst, addr_scratch)
            }
            (MachineMemWidth::U8, MachineLoadExtension::SignExtend) => {
                enc::ldrsb_64_base(dst, addr_scratch)
            }
            (MachineMemWidth::U16, MachineLoadExtension::None)
            | (MachineMemWidth::U16, MachineLoadExtension::ZeroExtend) => {
                enc::ldrh_base(dst, addr_scratch)
            }
            (MachineMemWidth::U16, MachineLoadExtension::SignExtend) => {
                enc::ldrsh_64_base(dst, addr_scratch)
            }
            (MachineMemWidth::U32, MachineLoadExtension::None)
            | (MachineMemWidth::U32, MachineLoadExtension::ZeroExtend) => {
                enc::ldr_reg_32_base(dst, addr_scratch)
            }
            (MachineMemWidth::U32, MachineLoadExtension::SignExtend) => {
                enc::ldrsw_base(dst, addr_scratch)
            }
            (MachineMemWidth::U64, MachineLoadExtension::None)
            | (MachineMemWidth::U64, MachineLoadExtension::ZeroExtend) => {
                enc::ldr_reg_64_base(dst, addr_scratch)
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

    fn lower_store(
        &mut self,
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
                let addr_scratch = *self.gp_scratch.scoped_alloc();
                self.lower_addr_into(addr_scratch, addr)?;
                self.core.text.emit_u32(match width {
                    MachineMemWidth::U32 => enc::str_s_base(src_fp, addr_scratch),
                    MachineMemWidth::U64 => enc::str_d_base(src_fp, addr_scratch),
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
                    .emit_u32(enc::str_zero_64(base, (offset / 8) as u32));
                return Ok(());
            }
        }
        // Fast path: U64 store with aligned immediate offset -> single str_64
        if width == MachineMemWidth::U64 {
            let offset = addr.offset as i64;
            if offset >= 0 && (offset % 8) == 0 && (offset / 8) < 4096 {
                let src_reg = prepare_gp(
                    self.core.compiled.backend(),
                    &self.core.fp_reg_widths,
                    &mut self.core.text,
                    &self.gp_scratch,
                    src,
                )?;
                self.core
                    .text
                    .emit_u32(enc::str_64(*src_reg, base, (offset / 8) as u32));
                return Ok(());
            }
        }
        let addr_scratch = *self.gp_scratch.scoped_alloc();
        self.lower_addr_into(addr_scratch, addr)?;
        let src_reg = prepare_gp(
            self.core.compiled.backend(),
            &self.core.fp_reg_widths,
            &mut self.core.text,
            &self.gp_scratch,
            src,
        )?;
        let inst = match width {
            MachineMemWidth::U8 => enc::strb_base(*src_reg, addr_scratch),
            MachineMemWidth::U16 => enc::strh_base(*src_reg, addr_scratch),
            MachineMemWidth::U32 => enc::str_reg_32_base(*src_reg, addr_scratch),
            MachineMemWidth::U64 => enc::str_reg_64_base(*src_reg, addr_scratch),
        };
        self.core.text.emit_u32(inst);
        Ok(())
    }

    // ── Indexed Load / Store ─────────────────────────────────────────────────────

    fn lower_indexed_load(
        &mut self,
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
    fn lower_indexed_load_with_offset(
        &mut self,
        dst: MachineReg,
        base_reg: MachineReg,
        index_reg: MachineReg,
        offset: i32,
        width: MachineMemWidth,
        extension: MachineLoadExtension,
        uxtw: bool,
    ) -> Result<(), WasmError> {
        let index_arm = self.map_gp_reg(index_reg)?;
        // Keep the base and destination mappings intact while materializing the
        // adjusted index. Reusing the GP destination register here is unsafe when
        // the load writes back into the same machine reg as the base.
        let scratch = *self.gp_scratch.scoped_alloc();
        if uxtw {
            self.core.text.emit_u32(enc::mov_reg_32(scratch, index_arm));
        } else {
            self.core.text.emit_u32(enc::mov_reg_64(scratch, index_arm));
        }
        self.lower_add_imm_to_reg(scratch, offset as i64);
        // Register-indexed load using the pre-computed scratch as the index.
        let base_arm = self.map_gp_reg(base_reg)?;
        if self.core.is_fp_reg(dst) {
            let dst_fp = self.map_fp_reg(dst)?;
            let inst = match width {
                MachineMemWidth::U32 => enc::ldr_s_reg(dst_fp, base_arm, scratch, false),
                MachineMemWidth::U64 => enc::ldr_d_reg(dst_fp, base_arm, scratch, false),
                _ => {
                    return Err(WasmError::invalid(
                        "arm64: narrow FP indexed load not supported".into(),
                    ))
                }
            };
            self.core.text.emit_u32(inst);
            let tracked = if width == MachineMemWidth::U32 {
                MachineFloatWidth::F32
            } else {
                MachineFloatWidth::F64
            };
            self.core.set_fp_reg_width(dst, tracked)?;
        } else {
            let dst_arm = self.map_gp_reg(dst)?;
            let inst = match (width, extension) {
                (MachineMemWidth::U8, MachineLoadExtension::None)
                | (MachineMemWidth::U8, MachineLoadExtension::ZeroExtend) => {
                    enc::ldrb_reg(dst_arm, base_arm, scratch)
                }
                (MachineMemWidth::U8, MachineLoadExtension::SignExtend) => {
                    enc::ldrsb_reg_64(dst_arm, base_arm, scratch)
                }
                (MachineMemWidth::U16, MachineLoadExtension::None)
                | (MachineMemWidth::U16, MachineLoadExtension::ZeroExtend) => {
                    enc::ldrh_reg(dst_arm, base_arm, scratch)
                }
                (MachineMemWidth::U16, MachineLoadExtension::SignExtend) => {
                    enc::ldrsh_reg_64(dst_arm, base_arm, scratch)
                }
                (MachineMemWidth::U32, MachineLoadExtension::None)
                | (MachineMemWidth::U32, MachineLoadExtension::ZeroExtend) => {
                    enc::ldr_reg_32(dst_arm, base_arm, scratch)
                }
                (MachineMemWidth::U32, MachineLoadExtension::SignExtend) => {
                    enc::ldrsw_reg(dst_arm, base_arm, scratch)
                }
                (MachineMemWidth::U64, MachineLoadExtension::None)
                | (MachineMemWidth::U64, MachineLoadExtension::ZeroExtend) => {
                    enc::ldr_reg_64(dst_arm, base_arm, scratch)
                }
                _ => {
                    return Err(WasmError::invalid(
                        "arm64: unsupported indexed load extension".into(),
                    ))
                }
            };
            self.core.text.emit_u32(inst);
        }
        Ok(())
    }

    fn lower_indexed_store(
        &mut self,
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
                        if uxtw {
                            enc::str_s_reg_uxtw(src_fp, base, index)
                        } else {
                            enc::str_s_reg(src_fp, base, index, scaled)
                        }
                    }
                    MachineMemWidth::U64 => {
                        if uxtw {
                            enc::str_d_reg_uxtw(src_fp, base, index)
                        } else {
                            enc::str_d_reg(src_fp, base, index, scaled)
                        }
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
        let src_reg = prepare_gp(
            self.core.compiled.backend(),
            &self.core.fp_reg_widths,
            &mut self.core.text,
            &self.gp_scratch,
            src,
        )?;
        let src_reg = *src_reg;
        let inst = match width {
            MachineMemWidth::U8 => {
                if uxtw {
                    enc::strb_reg_uxtw(src_reg, base, index)
                } else {
                    enc::strb_reg(src_reg, base, index)
                }
            }
            MachineMemWidth::U16 => {
                if uxtw {
                    enc::strh_reg_uxtw(src_reg, base, index)
                } else if scaled {
                    enc::strh_reg_scaled(src_reg, base, index)
                } else {
                    enc::strh_reg(src_reg, base, index)
                }
            }
            MachineMemWidth::U32 => {
                if uxtw {
                    enc::str_reg_32_uxtw(src_reg, base, index)
                } else if scaled {
                    enc::str_reg_32_scaled(src_reg, base, index)
                } else {
                    enc::str_reg_32(src_reg, base, index)
                }
            }
            MachineMemWidth::U64 => {
                if uxtw {
                    enc::str_reg_64_uxtw(src_reg, base, index)
                } else if scaled {
                    enc::str_reg_64_scaled(src_reg, base, index)
                } else {
                    enc::str_reg_64(src_reg, base, index)
                }
            }
        };
        self.core.text.emit_u32(inst);
        Ok(())
    }

    /// Indexed store with a non-zero offset: fold index + offset into a scratch,
    /// then register-indexed store with the ORIGINAL memory base.
    fn lower_indexed_store_with_offset(
        &mut self,
        base_reg: MachineReg,
        index_reg: MachineReg,
        offset: i32,
        width: MachineMemWidth,
        src: MachineValue,
        uxtw: bool,
    ) -> Result<(), WasmError> {
        let index_arm = self.map_gp_reg(index_reg)?;
        let idx_scratch = *self.gp_scratch.scoped_alloc();
        if uxtw {
            self.core
                .text
                .emit_u32(enc::mov_reg_32(idx_scratch, index_arm));
        } else {
            self.core
                .text
                .emit_u32(enc::mov_reg_64(idx_scratch, index_arm));
        }
        self.lower_add_imm_to_reg(idx_scratch, offset as i64);

        // Register-indexed store using pre-mapped index register.
        let base_arm = self.map_gp_reg(base_reg)?;
        if let MachineValue::Reg(src_reg) = src {
            if self.core.is_fp_reg(src_reg) {
                let src_fp = self.map_fp_reg(src_reg)?;
                let inst = match width {
                    MachineMemWidth::U32 => enc::str_s_reg(src_fp, base_arm, idx_scratch, false),
                    MachineMemWidth::U64 => enc::str_d_reg(src_fp, base_arm, idx_scratch, false),
                    _ => {
                        return Err(WasmError::invalid(
                            "arm64: narrow FP indexed store not supported".into(),
                        ))
                    }
                };
                self.core.text.emit_u32(inst);
                return Ok(());
            }
        }
        let src_arm = prepare_gp(
            self.core.compiled.backend(),
            &self.core.fp_reg_widths,
            &mut self.core.text,
            &self.gp_scratch,
            src,
        )?;
        let inst = match width {
            MachineMemWidth::U8 => enc::strb_reg(*src_arm, base_arm, idx_scratch),
            MachineMemWidth::U16 => enc::strh_reg(*src_arm, base_arm, idx_scratch),
            MachineMemWidth::U32 => enc::str_reg_32(*src_arm, base_arm, idx_scratch),
            MachineMemWidth::U64 => enc::str_reg_64(*src_arm, base_arm, idx_scratch),
        };
        self.core.text.emit_u32(inst);
        Ok(())
    }

    // ── Integer unary ────────────────────────────────────────────────────────────

    fn lower_int_unary(
        &mut self,
        width: MachineIntWidth,
        op: MachineIntUnaryOp,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let dst = self.map_gp_reg(dst)?;
        let src = prepare_gp(
            self.core.compiled.backend(),
            &self.core.fp_reg_widths,
            &mut self.core.text,
            &self.gp_scratch,
            src,
        )?;
        match (width, op) {
            (MachineIntWidth::I32, MachineIntUnaryOp::Clz) => {
                self.core.text.emit_u32(enc::clz_32(dst, *src));
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Clz) => {
                self.core.text.emit_u32(enc::clz_64(dst, *src));
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Extend8S) => {
                self.core.text.emit_u32(enc::sxtb_32(dst, *src));
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Extend16S) => {
                self.core.text.emit_u32(enc::sxth_32(dst, *src));
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Extend8S) => {
                self.core.text.emit_u32(enc::sxtb_64(dst, *src));
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Extend16S) => {
                self.core.text.emit_u32(enc::sxth_64(dst, *src));
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Extend32S) => {
                self.core.text.emit_u32(enc::sxtw(dst, *src));
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Ctz) => {
                self.core.text.emit_u32(enc::rbit_32(dst, *src));
                self.core.text.emit_u32(enc::clz_32(dst, dst));
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Ctz) => {
                self.core.text.emit_u32(enc::rbit_64(dst, *src));
                self.core.text.emit_u32(enc::clz_64(dst, dst));
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Popcnt) => {
                // FMOV D0, X_src (move GP to FP); CNT V0.8B; ADDV B0; UMOV Wd, V0.B[0]
                let fp_scratch = self.fp_scratch.scoped_alloc();
                self.core
                    .text
                    .emit_u32(enc::fmov_d_from_gp(*fp_scratch, *src));
                self.core
                    .text
                    .emit_u32(enc::cnt_8b(*fp_scratch, *fp_scratch));
                self.core
                    .text
                    .emit_u32(enc::addv_8b(*fp_scratch, *fp_scratch));
                self.core.text.emit_u32(enc::umov_b0(dst, *fp_scratch));
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Popcnt) => {
                let fp_scratch = self.fp_scratch.scoped_alloc();
                self.core
                    .text
                    .emit_u32(enc::fmov_d_from_gp(*fp_scratch, *src));
                self.core
                    .text
                    .emit_u32(enc::cnt_8b(*fp_scratch, *fp_scratch));
                self.core
                    .text
                    .emit_u32(enc::addv_8b(*fp_scratch, *fp_scratch));
                self.core.text.emit_u32(enc::umov_b0(dst, *fp_scratch));
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Extend32S) => {
                // i32.extend32_s is a nop (already 32-bit)
                if dst != *src {
                    self.core.text.emit_u32(enc::mov_reg_64(dst, *src));
                }
            }
        }
        Ok(())
    }

    // ── Integer binary ───────────────────────────────────────────────────────────

    fn lower_int_binary(
        &mut self,
        width: MachineIntWidth,
        op: MachineIntBinaryOp,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        let dst = self.map_gp_reg(dst)?;
        // Try immediate form: check reg+imm and imm+reg (for commutative ops).
        if let MachineValue::Imm64(imm) = rhs {
            if let MachineValue::Reg(lhs_reg) = lhs {
                let lhs_phys = map_gp(self.core.compiled.backend(), lhs_reg)?;
                if let Some(inst) = int_binary_imm_inst(width, op, dst, lhs_phys, imm) {
                    self.core.text.emit_u32(inst);
                    return Ok(());
                }
            }
        }
        if let MachineValue::Imm64(imm) = lhs {
            if let MachineValue::Reg(rhs_reg) = rhs {
                let rhs_phys = map_gp(self.core.compiled.backend(), rhs_reg)?;
                // Commutative ops: swap operands for imm selection.
                match op {
                    MachineIntBinaryOp::Add
                    | MachineIntBinaryOp::Mul
                    | MachineIntBinaryOp::And
                    | MachineIntBinaryOp::Or
                    | MachineIntBinaryOp::Xor => {
                        if let Some(inst) = int_binary_imm_inst(width, op, dst, rhs_phys, imm) {
                            self.core.text.emit_u32(inst);
                            return Ok(());
                        }
                    }
                    _ => {}
                }
            }
        }
        let lhs = prepare_gp(
            self.core.compiled.backend(),
            &self.core.fp_reg_widths,
            &mut self.core.text,
            &self.gp_scratch,
            lhs,
        )?
        .detach();
        let rhs = prepare_gp(
            self.core.compiled.backend(),
            &self.core.fp_reg_widths,
            &mut self.core.text,
            &self.gp_scratch,
            rhs,
        )?
        .detach();
        match (width, op) {
            (MachineIntWidth::I32, MachineIntBinaryOp::Add) => {
                self.core.text.emit_u32(enc::add_reg_32(dst, *lhs, *rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Add) => {
                self.core.text.emit_u32(enc::add_reg_64(dst, *lhs, *rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Sub) => {
                self.core.text.emit_u32(enc::sub_reg_32(dst, *lhs, *rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Sub) => {
                self.core.text.emit_u32(enc::sub_reg_64(dst, *lhs, *rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Mul) => {
                self.core.text.emit_u32(enc::mul_32(dst, *lhs, *rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Mul) => {
                self.core.text.emit_u32(enc::mul_64(dst, *lhs, *rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::And) => {
                self.core.text.emit_u32(enc::and_reg_32(dst, *lhs, *rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::And) => {
                self.core.text.emit_u32(enc::and_reg_64(dst, *lhs, *rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Or) => {
                self.core.text.emit_u32(enc::orr_reg_32(dst, *lhs, *rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Or) => {
                self.core.text.emit_u32(enc::orr_reg_64(dst, *lhs, *rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Xor) => {
                self.core.text.emit_u32(enc::eor_reg_32(dst, *lhs, *rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Xor) => {
                self.core.text.emit_u32(enc::eor_reg_64(dst, *lhs, *rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Shl) => {
                self.core.text.emit_u32(enc::lslv_32(dst, *lhs, *rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Shl) => {
                self.core.text.emit_u32(enc::lslv_64(dst, *lhs, *rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::ShrS) => {
                self.core.text.emit_u32(enc::asrv_32(dst, *lhs, *rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::ShrS) => {
                self.core.text.emit_u32(enc::asrv_64(dst, *lhs, *rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::ShrU) => {
                self.core.text.emit_u32(enc::lsrv_32(dst, *lhs, *rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::ShrU) => {
                self.core.text.emit_u32(enc::lsrv_64(dst, *lhs, *rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Rotr) => {
                self.core.text.emit_u32(enc::rorv_32(dst, *lhs, *rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Rotr) => {
                self.core.text.emit_u32(enc::rorv_64(dst, *lhs, *rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Rotl) => {
                // rotl(x, n) = rotr(x, -n)
                let neg_dst = *self.gp_scratch.scoped_alloc();
                self.core.text.emit_u32(enc::neg_reg_32(neg_dst, *rhs));
                self.core.text.emit_u32(enc::rorv_32(dst, *lhs, neg_dst));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Rotl) => {
                let neg_dst = *self.gp_scratch.scoped_alloc();
                self.core.text.emit_u32(enc::neg_reg_64(neg_dst, *rhs));
                self.core.text.emit_u32(enc::rorv_64(dst, *lhs, neg_dst));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::DivS) => {
                self.lower_div_s_32(dst, *lhs, *rhs);
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::DivS) => {
                self.lower_div_s_64(dst, *lhs, *rhs);
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::DivU) => {
                self.lower_div_u_check(*lhs, *rhs, MachineIntWidth::I32);
                self.core.text.emit_u32(enc::udiv_32(dst, *lhs, *rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::DivU) => {
                self.lower_div_u_check(*lhs, *rhs, MachineIntWidth::I64);
                self.core.text.emit_u32(enc::udiv_64(dst, *lhs, *rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::RemS) => {
                self.lower_rem_s_32(dst, *lhs, *rhs);
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::RemS) => {
                self.lower_rem_s_64(dst, *lhs, *rhs);
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::RemU) => {
                self.lower_div_u_check(*lhs, *rhs, MachineIntWidth::I32);
                // rem = lhs - (lhs / rhs) * rhs
                let tmp = *self.gp_scratch.scoped_alloc();
                self.core.text.emit_u32(enc::udiv_32(tmp, *lhs, *rhs));
                self.core.text.emit_u32(enc::msub_32(dst, tmp, *rhs, *lhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::RemU) => {
                self.lower_div_u_check(*lhs, *rhs, MachineIntWidth::I64);
                let tmp = *self.gp_scratch.scoped_alloc();
                self.core.text.emit_u32(enc::udiv_64(tmp, *lhs, *rhs));
                self.core.text.emit_u32(enc::msub_64(dst, tmp, *rhs, *lhs));
            }
        };
        Ok(())
    }

    // ── Division / remainder helpers with trap checks ────────────────────────────

    fn lower_div_u_check(&mut self, _lhs: Arm64Reg, rhs: Arm64Reg, width: MachineIntWidth) {
        // rhs == 0 => trap IntegerDivideByZero
        match width {
            MachineIntWidth::I32 => self.core.text.emit_u32(enc::cmp_zero_32(rhs)),
            MachineIntWidth::I64 => self.core.text.emit_u32(enc::cmp_zero_64(rhs)),
        };
        // Branch to a trap stub
        let trap_label = self.core.new_label();
        self.lower_b_cond(enc::Cond::Eq, trap_label);
        self.core
            .deferred_traps
            .push((trap_label, MachineTrapKind::IntegerDivideByZero));
    }

    fn lower_div_s_32(&mut self, dst: Arm64Reg, lhs: Arm64Reg, rhs: Arm64Reg) {
        // Check rhs == 0 => IntegerDivideByZero
        self.core.text.emit_u32(enc::cmp_zero_32(rhs));
        let div_zero_label = self.core.new_label();
        self.lower_b_cond(enc::Cond::Eq, div_zero_label);
        self.core
            .deferred_traps
            .push((div_zero_label, MachineTrapKind::IntegerDivideByZero));

        // Check lhs == i32::MIN && rhs == -1 => IntegerOverflow
        let scratch = *self.gp_scratch.scoped_alloc();
        self.materialize_u64(scratch, i32::MIN as u32 as u64);
        self.core.text.emit_u32(enc::cmp_reg_32(lhs, scratch));
        let not_min = self.core.new_label();
        self.lower_b_cond(enc::Cond::Ne, not_min);
        // lhs is MIN, check rhs == -1
        self.materialize_u64(scratch, (-1i32) as u32 as u64);
        self.core.text.emit_u32(enc::cmp_reg_32(rhs, scratch));
        let overflow_label = self.core.new_label();
        self.lower_b_cond(enc::Cond::Eq, overflow_label);
        self.core
            .deferred_traps
            .push((overflow_label, MachineTrapKind::IntegerOverflow));

        self.core.bind_label(not_min);
        self.core.text.emit_u32(enc::sdiv_32(dst, lhs, rhs));
    }

    fn lower_div_s_64(&mut self, dst: Arm64Reg, lhs: Arm64Reg, rhs: Arm64Reg) {
        self.core.text.emit_u32(enc::cmp_zero_64(rhs));
        let div_zero_label = self.core.new_label();
        self.lower_b_cond(enc::Cond::Eq, div_zero_label);
        self.core
            .deferred_traps
            .push((div_zero_label, MachineTrapKind::IntegerDivideByZero));
        let scratch = *self.gp_scratch.scoped_alloc();
        self.materialize_u64(scratch, i64::MIN as u64);
        self.core.text.emit_u32(enc::cmp_reg_64(lhs, scratch));
        let not_min = self.core.new_label();
        self.lower_b_cond(enc::Cond::Ne, not_min);
        self.materialize_u64(scratch, (-1i64) as u64);
        self.core.text.emit_u32(enc::cmp_reg_64(rhs, scratch));
        let overflow_label = self.core.new_label();
        self.lower_b_cond(enc::Cond::Eq, overflow_label);
        self.core
            .deferred_traps
            .push((overflow_label, MachineTrapKind::IntegerOverflow));

        self.core.bind_label(not_min);
        self.core.text.emit_u32(enc::sdiv_64(dst, lhs, rhs));
    }

    fn lower_rem_s_32(&mut self, dst: Arm64Reg, lhs: Arm64Reg, rhs: Arm64Reg) {
        // Check rhs == 0 => IntegerDivideByZero
        self.core.text.emit_u32(enc::cmp_zero_32(rhs));
        let div_zero_label = self.core.new_label();
        self.lower_b_cond(enc::Cond::Eq, div_zero_label);
        self.core
            .deferred_traps
            .push((div_zero_label, MachineTrapKind::IntegerDivideByZero));

        // rem = lhs - (lhs / rhs) * rhs  (wrapping, so MIN % -1 = 0, no trap)
        let scratch = *self.gp_scratch.scoped_alloc();
        self.core.text.emit_u32(enc::sdiv_32(scratch, lhs, rhs));
        self.core
            .text
            .emit_u32(enc::msub_32(dst, scratch, rhs, lhs));
    }

    fn lower_rem_s_64(&mut self, dst: Arm64Reg, lhs: Arm64Reg, rhs: Arm64Reg) {
        self.core.text.emit_u32(enc::cmp_zero_64(rhs));
        let div_zero_label = self.core.new_label();
        self.lower_b_cond(enc::Cond::Eq, div_zero_label);
        self.core
            .deferred_traps
            .push((div_zero_label, MachineTrapKind::IntegerDivideByZero));
        let scratch = *self.gp_scratch.scoped_alloc();
        self.core.text.emit_u32(enc::sdiv_64(scratch, lhs, rhs));
        self.core
            .text
            .emit_u32(enc::msub_64(dst, scratch, rhs, lhs));
    }

    // ── Integer compare ──────────────────────────────────────────────────────────

    fn lower_int_compare(
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

    // ── Bitfield extract (UBFX) ─────────────────────────────────────────────────

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
        match width {
            MachineIntWidth::I32 => {
                self.core
                    .text
                    .emit_u32(enc::ubfx_32(dst, src, lsb as u32, bits as u32));
            }
            MachineIntWidth::I64 => {
                self.core
                    .text
                    .emit_u32(enc::ubfx_64(dst, src, lsb as u32, bits as u32));
            }
        }
        Ok(())
    }

    // ── Shifted-register binary ─────────────────────────────────────────────────

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
        let dst = self.map_gp_reg(dst)?;
        let lhs = self.map_gp_reg(lhs)?;
        let rhs = self.map_gp_reg(rhs)?;
        let st = match shift {
            MachineShiftOp::Lsl => enc::ShiftType::Lsl,
            MachineShiftOp::Lsr => enc::ShiftType::Lsr,
            MachineShiftOp::Asr => enc::ShiftType::Asr,
        };
        let amt = amount as u32;
        let inst = match (width, op) {
            (MachineIntWidth::I32, MachineIntBinaryOp::Add) => {
                enc::add_reg_shifted_32(dst, lhs, rhs, st, amt)
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Add) => {
                enc::add_reg_shifted_64(dst, lhs, rhs, st, amt)
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Sub) => {
                enc::sub_reg_shifted_32(dst, lhs, rhs, st, amt)
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Sub) => {
                enc::sub_reg_shifted_64(dst, lhs, rhs, st, amt)
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::And) => {
                enc::and_reg_shifted_32(dst, lhs, rhs, st, amt)
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::And) => {
                enc::and_reg_shifted_64(dst, lhs, rhs, st, amt)
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Or) => {
                enc::orr_reg_shifted_32(dst, lhs, rhs, st, amt)
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Or) => {
                enc::orr_reg_shifted_64(dst, lhs, rhs, st, amt)
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Xor) => {
                enc::eor_reg_shifted_32(dst, lhs, rhs, st, amt)
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Xor) => {
                enc::eor_reg_shifted_64(dst, lhs, rhs, st, amt)
            }
            _ => {
                return Err(WasmError::internal(alloc::format!(
                    "IntBinaryShifted: unsupported op {:?}",
                    op
                )))
            }
        };
        self.core.text.emit_u32(inst);
        Ok(())
    }

    // ── Test bits (TST + CSET) ──────────────────────────────────────────────────

    fn lower_test_bits(
        &mut self,
        width: MachineIntWidth,
        kind: MachineCompareKind,
        dst: MachineReg,
        src: MachineReg,
        mask: MachineValue,
    ) -> Result<(), WasmError> {
        let dst_phys = self.map_gp_reg(dst)?;
        let src_phys = self.map_gp_reg(src)?;

        // Emit TST (ANDS with XZR destination, sets flags).
        match mask {
            MachineValue::Imm64(imm) => {
                let inst = match width {
                    MachineIntWidth::I32 => enc::tst_imm_32(src_phys, imm as u32),
                    MachineIntWidth::I64 => enc::tst_imm_64(src_phys, imm),
                };
                if let Some(i) = inst {
                    self.core.text.emit_u32(i);
                } else {
                    // Mask doesn't fit logical immediate — materialize into scratch.
                    let scratch = prepare_gp(
                        self.core.compiled.backend(),
                        &self.core.fp_reg_widths,
                        &mut self.core.text,
                        &self.gp_scratch,
                        MachineValue::Imm64(imm),
                    )?;
                    match width {
                        MachineIntWidth::I32 => {
                            self.core.text.emit_u32(enc::tst_reg_32(src_phys, *scratch));
                        }
                        MachineIntWidth::I64 => {
                            self.core.text.emit_u32(enc::tst_reg_64(src_phys, *scratch));
                        }
                    }
                }
            }
            MachineValue::Reg(mask_reg) => {
                let mask_phys = map_gp(self.core.compiled.backend(), mask_reg)?;
                match width {
                    MachineIntWidth::I32 => {
                        self.core
                            .text
                            .emit_u32(enc::tst_reg_32(src_phys, mask_phys));
                    }
                    MachineIntWidth::I64 => {
                        self.core
                            .text
                            .emit_u32(enc::tst_reg_64(src_phys, mask_phys));
                    }
                }
            }
            MachineValue::ReservedReg(reg) => {
                return Err(WasmError::internal(alloc::format!(
                    "arm64 TestBits cannot consume reserved cache register {} as a mask",
                    reg.0
                )));
            }
        }

        // TST sets Z flag. Eq → Z=1, Ne → Z=0.
        let cond = match kind {
            MachineCompareKind::Eq => enc::Cond::Eq,
            MachineCompareKind::Ne => enc::Cond::Ne,
            _ => {
                return Err(WasmError::internal(alloc::format!(
                    "TestBits: unsupported compare kind {:?}",
                    kind
                )))
            }
        };
        match width {
            MachineIntWidth::I32 => {
                self.core.text.emit_u32(enc::cset_32(dst_phys, cond));
            }
            MachineIntWidth::I64 => {
                self.core.text.emit_u32(enc::cset_64(dst_phys, cond));
            }
        }
        Ok(())
    }

    // ── Select ───────────────────────────────────────────────────────────────────

    fn lower_select(
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
                    let true_fp = prepare_fp(
                        self.core.compiled.backend(),
                        &self.core.fp_reg_widths,
                        &mut self.core.text,
                        &self.gp_scratch,
                        &self.fp_scratch,
                        width,
                        on_true,
                    )?;
                    let false_fp = prepare_fp(
                        self.core.compiled.backend(),
                        &self.core.fp_reg_widths,
                        &mut self.core.text,
                        &self.gp_scratch,
                        &self.fp_scratch,
                        width,
                        on_false,
                    )?;
                    let dst_fp = self.map_fp_reg(dst)?;
                    self.core
                        .text
                        .emit_u32(enc::cmp_imm_64(self.map_gp_reg(reg)?, 0));
                    self.core.text.emit_u32(match width {
                        MachineFloatWidth::F32 => {
                            enc::fcsel_s(dst_fp, *true_fp, *false_fp, enc::Cond::Ne)
                        }
                        MachineFloatWidth::F64 => {
                            enc::fcsel_d(dst_fp, *true_fp, *false_fp, enc::Cond::Ne)
                        }
                    });
                    self.core.set_fp_reg_width(dst, width)?;
                    Ok(())
                }
                MachineValue::ReservedReg(reg) => Err(WasmError::internal(alloc::format!(
                    "arm64 select cannot consume reserved cache register {} as a condition",
                    reg.0
                ))),
            }
        } else {
            let dst = self.map_gp_reg(dst)?;
            match cond {
                MachineValue::Imm64(value) => {
                    let selected = if value != 0 { on_true } else { on_false };
                    let src = prepare_gp(
                        self.core.compiled.backend(),
                        &self.core.fp_reg_widths,
                        &mut self.core.text,
                        &self.gp_scratch,
                        selected,
                    )?;
                    if dst != *src {
                        self.core.text.emit_u32(enc::mov_reg_64(dst, *src));
                    }
                    return Ok(());
                }
                MachineValue::Reg(reg) => {
                    self.core
                        .text
                        .emit_u32(enc::cmp_imm_64(self.map_gp_reg(reg)?, 0));
                }
                MachineValue::ReservedReg(reg) => {
                    return Err(WasmError::internal(alloc::format!(
                        "arm64 select cannot consume reserved cache register {} as a condition",
                        reg.0
                    )));
                }
            }
            let true_reg = prepare_gp(
                self.core.compiled.backend(),
                &self.core.fp_reg_widths,
                &mut self.core.text,
                &self.gp_scratch,
                on_true,
            )?;
            let false_reg = prepare_gp(
                self.core.compiled.backend(),
                &self.core.fp_reg_widths,
                &mut self.core.text,
                &self.gp_scratch,
                on_false,
            )?;
            // Always use csel_64: GpWord covers both i32 and reference types,
            // and refs need full 64-bit values preserved (e.g. null sentinel).
            self.core
                .text
                .emit_u32(enc::csel_64(dst, *true_reg, *false_reg, enc::Cond::Ne));
            Ok(())
        }
    }

    // ── Float operations ─────────────────────────────────────────────────────────

    fn lower_float_unary(
        &mut self,
        width: MachineFloatWidth,
        op: MachineFloatUnaryOp,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let src_fp = prepare_fp(
            self.core.compiled.backend(),
            &self.core.fp_reg_widths,
            &mut self.core.text,
            &self.gp_scratch,
            &self.fp_scratch,
            width,
            src,
        )?
        .detach();
        let result_fp = if self.core.is_fp_reg(dst) {
            let dst_fp = self.map_fp_reg(dst)?;
            self.core.set_fp_reg_width(dst, width)?;
            dst_fp
        } else {
            *self.fp_scratch.scoped_alloc()
        };
        // Perform the FP operation
        match (width, op) {
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Abs) => {
                self.core.text.emit_u32(enc::fabs_s(result_fp, *src_fp))
            }
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Abs) => {
                self.core.text.emit_u32(enc::fabs_d(result_fp, *src_fp))
            }
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Neg) => {
                self.core.text.emit_u32(enc::fneg_s(result_fp, *src_fp))
            }
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Neg) => {
                self.core.text.emit_u32(enc::fneg_d(result_fp, *src_fp))
            }
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Sqrt) => {
                self.core.text.emit_u32(enc::fsqrt_s(result_fp, *src_fp))
            }
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Sqrt) => {
                self.core.text.emit_u32(enc::fsqrt_d(result_fp, *src_fp))
            }
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Ceil) => {
                self.core.text.emit_u32(enc::frintp_s(result_fp, *src_fp))
            }
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Ceil) => {
                self.core.text.emit_u32(enc::frintp_d(result_fp, *src_fp))
            }
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Floor) => {
                self.core.text.emit_u32(enc::frintm_s(result_fp, *src_fp))
            }
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Floor) => {
                self.core.text.emit_u32(enc::frintm_d(result_fp, *src_fp))
            }
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Trunc) => {
                self.core.text.emit_u32(enc::frintz_s(result_fp, *src_fp))
            }
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Trunc) => {
                self.core.text.emit_u32(enc::frintz_d(result_fp, *src_fp))
            }
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Nearest) => {
                self.core.text.emit_u32(enc::frintn_s(result_fp, *src_fp))
            }
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Nearest) => {
                self.core.text.emit_u32(enc::frintn_d(result_fp, *src_fp))
            }
        };
        if !self.core.is_fp_reg(dst) {
            let dst_gp = self.map_gp_reg(dst)?;
            match width {
                MachineFloatWidth::F32 => self
                    .core
                    .text
                    .emit_u32(enc::fmov_gp_from_s(dst_gp, result_fp)),
                MachineFloatWidth::F64 => self
                    .core
                    .text
                    .emit_u32(enc::fmov_gp_from_d(dst_gp, result_fp)),
            };
        }
        Ok(())
    }

    fn lower_float_binary(
        &mut self,
        width: MachineFloatWidth,
        op: MachineFloatBinaryOp,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        let lhs_fp = prepare_fp(
            self.core.compiled.backend(),
            &self.core.fp_reg_widths,
            &mut self.core.text,
            &self.gp_scratch,
            &self.fp_scratch,
            width,
            lhs,
        )?
        .detach();
        let rhs_fp = prepare_fp(
            self.core.compiled.backend(),
            &self.core.fp_reg_widths,
            &mut self.core.text,
            &self.gp_scratch,
            &self.fp_scratch,
            width,
            rhs,
        )?
        .detach();
        let result_fp = if self.core.is_fp_reg(dst) {
            let dst_fp = self.map_fp_reg(dst)?;
            self.core.set_fp_reg_width(dst, width)?;
            dst_fp
        } else {
            *self.fp_scratch.scoped_alloc()
        };
        match (width, op) {
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Add) => {
                self.core
                    .text
                    .emit_u32(enc::fadd_s(result_fp, *lhs_fp, *rhs_fp));
            }
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Add) => {
                self.core
                    .text
                    .emit_u32(enc::fadd_d(result_fp, *lhs_fp, *rhs_fp));
            }
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Sub) => {
                self.core
                    .text
                    .emit_u32(enc::fsub_s(result_fp, *lhs_fp, *rhs_fp));
            }
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Sub) => {
                self.core
                    .text
                    .emit_u32(enc::fsub_d(result_fp, *lhs_fp, *rhs_fp));
            }
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Mul) => {
                self.core
                    .text
                    .emit_u32(enc::fmul_s(result_fp, *lhs_fp, *rhs_fp));
            }
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Mul) => {
                self.core
                    .text
                    .emit_u32(enc::fmul_d(result_fp, *lhs_fp, *rhs_fp));
            }
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Div) => {
                self.core
                    .text
                    .emit_u32(enc::fdiv_s(result_fp, *lhs_fp, *rhs_fp));
            }
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Div) => {
                self.core
                    .text
                    .emit_u32(enc::fdiv_d(result_fp, *lhs_fp, *rhs_fp));
            }
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Min) => {
                // Wasm fmin: NaN if either is NaN. ARM64 FMIN returns non-NaN operand.
                self.core
                    .text
                    .emit_u32(enc::fmin_s(result_fp, *lhs_fp, *rhs_fp));
                self.core.text.emit_u32(enc::fcmp_s(*lhs_fp, *rhs_fp));
                let done = self.core.new_label();
                self.lower_b_cond(enc::Cond::Vc, done); // no NaN => FMIN result is correct
                                                        // NaN case: FADD produces NaN from NaN input
                self.core
                    .text
                    .emit_u32(enc::fadd_s(result_fp, *lhs_fp, *rhs_fp));
                self.core.bind_label(done);
            }
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Min) => {
                self.core
                    .text
                    .emit_u32(enc::fmin_d(result_fp, *lhs_fp, *rhs_fp));
                self.core.text.emit_u32(enc::fcmp_d(*lhs_fp, *rhs_fp));
                let done = self.core.new_label();
                self.lower_b_cond(enc::Cond::Vc, done);
                self.core
                    .text
                    .emit_u32(enc::fadd_d(result_fp, *lhs_fp, *rhs_fp));
                self.core.bind_label(done);
            }
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Max) => {
                self.core
                    .text
                    .emit_u32(enc::fmax_s(result_fp, *lhs_fp, *rhs_fp));
                self.core.text.emit_u32(enc::fcmp_s(*lhs_fp, *rhs_fp));
                let done = self.core.new_label();
                self.lower_b_cond(enc::Cond::Vc, done);
                self.core
                    .text
                    .emit_u32(enc::fadd_s(result_fp, *lhs_fp, *rhs_fp));
                self.core.bind_label(done);
            }
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Max) => {
                self.core
                    .text
                    .emit_u32(enc::fmax_d(result_fp, *lhs_fp, *rhs_fp));
                self.core.text.emit_u32(enc::fcmp_d(*lhs_fp, *rhs_fp));
                let done = self.core.new_label();
                self.lower_b_cond(enc::Cond::Vc, done);
                self.core
                    .text
                    .emit_u32(enc::fadd_d(result_fp, *lhs_fp, *rhs_fp));
                self.core.bind_label(done);
            }
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Copysign) => {
                // copysign: magnitude of lhs, sign of rhs
                let neg_fp = *self.fp_scratch.scoped_alloc();
                self.core.text.emit_u32(enc::fabs_s(result_fp, *lhs_fp)); // |lhs|
                self.core.text.emit_u32(enc::fneg_s(neg_fp, result_fp)); // -|lhs|
                let rhs_gp = prepare_gp(
                    self.core.compiled.backend(),
                    &self.core.fp_reg_widths,
                    &mut self.core.text,
                    &self.gp_scratch,
                    rhs,
                )?
                .detach();
                let shift_reg = *self.gp_scratch.scoped_alloc();
                self.materialize_u64(shift_reg, 31);
                self.core
                    .text
                    .emit_u32(enc::lsrv_64(shift_reg, *rhs_gp, shift_reg));
                self.core.text.emit_u32(enc::cmp_imm_64(shift_reg, 0));
                self.core
                    .text
                    .emit_u32(enc::fcsel_s(result_fp, neg_fp, result_fp, enc::Cond::Ne));
            }
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Copysign) => {
                let neg_fp = *self.fp_scratch.scoped_alloc();
                self.core.text.emit_u32(enc::fabs_d(result_fp, *lhs_fp));
                self.core.text.emit_u32(enc::fneg_d(neg_fp, result_fp));
                let rhs_gp = prepare_gp(
                    self.core.compiled.backend(),
                    &self.core.fp_reg_widths,
                    &mut self.core.text,
                    &self.gp_scratch,
                    rhs,
                )?
                .detach();
                let shift_reg = *self.gp_scratch.scoped_alloc();
                self.materialize_u64(shift_reg, 63);
                self.core
                    .text
                    .emit_u32(enc::lsrv_64(shift_reg, *rhs_gp, shift_reg));
                self.core.text.emit_u32(enc::cmp_imm_64(shift_reg, 0));
                self.core
                    .text
                    .emit_u32(enc::fcsel_d(result_fp, neg_fp, result_fp, enc::Cond::Ne));
            }
        };
        if !self.core.is_fp_reg(dst) {
            let dst_gp = self.map_gp_reg(dst)?;
            match width {
                MachineFloatWidth::F32 => self
                    .core
                    .text
                    .emit_u32(enc::fmov_gp_from_s(dst_gp, result_fp)),
                MachineFloatWidth::F64 => self
                    .core
                    .text
                    .emit_u32(enc::fmov_gp_from_d(dst_gp, result_fp)),
            };
        }
        Ok(())
    }

    fn lower_float_compare(
        &mut self,
        width: MachineFloatWidth,
        kind: MachineCompareKind,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        let dst_gp = self.map_gp_reg(dst)?;
        let lhs_fp = prepare_fp(
            self.core.compiled.backend(),
            &self.core.fp_reg_widths,
            &mut self.core.text,
            &self.gp_scratch,
            &self.fp_scratch,
            width,
            lhs,
        )?;
        if matches!(rhs, MachineValue::Imm64(0)) {
            match width {
                MachineFloatWidth::F32 => self.core.text.emit_u32(enc::fcmp_s_zero(*lhs_fp)),
                MachineFloatWidth::F64 => self.core.text.emit_u32(enc::fcmp_d_zero(*lhs_fp)),
            };
        } else {
            let rhs_fp = prepare_fp(
                self.core.compiled.backend(),
                &self.core.fp_reg_widths,
                &mut self.core.text,
                &self.gp_scratch,
                &self.fp_scratch,
                width,
                rhs,
            )?;
            match width {
                MachineFloatWidth::F32 => self.core.text.emit_u32(enc::fcmp_s(*lhs_fp, *rhs_fp)),
                MachineFloatWidth::F64 => self.core.text.emit_u32(enc::fcmp_d(*lhs_fp, *rhs_fp)),
            };
        }
        // Wasm float comparisons: unordered (NaN) => false for all except Ne
        let cond = map_float_cond(kind);
        self.core.text.emit_u32(enc::cset_32(dst_gp, cond));
        Ok(())
    }

    /// Resolve FP destination register for convert ops: if dst is FP, map it;
    /// otherwise allocate an FP scratch.
    fn resolve_convert_fp_dst(
        &mut self,
        dst: MachineReg,
        width: MachineFloatWidth,
    ) -> Result<Arm64FpReg, WasmError> {
        if self.core.is_fp_reg(dst) {
            let dst_fp = self.map_fp_reg(dst)?;
            self.core.set_fp_reg_width(dst, width)?;
            Ok(dst_fp)
        } else {
            Ok(*self.fp_scratch.scoped_alloc())
        }
    }

    // ── Convert ──────────────────────────────────────────────────────────────────

    fn lower_convert(
        &mut self,
        op: MachineConvertOp,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let dst_float_width = convert_result_float_width(op);
        match op {
            // Integer wrapping / extension (no FP involved)
            MachineConvertOp::I32WrapI64 => {
                let src_gp = prepare_gp(
                    self.core.compiled.backend(),
                    &self.core.fp_reg_widths,
                    &mut self.core.text,
                    &self.gp_scratch,
                    src,
                )?
                .detach();
                let dst_gp = self.map_gp_reg(dst)?;
                self.core.text.emit_u32(enc::mov_reg_32(dst_gp, *src_gp));
            }
            MachineConvertOp::I64ExtendI32S => {
                let src_gp = prepare_gp(
                    self.core.compiled.backend(),
                    &self.core.fp_reg_widths,
                    &mut self.core.text,
                    &self.gp_scratch,
                    src,
                )?
                .detach();
                let dst_gp = self.map_gp_reg(dst)?;
                self.core.text.emit_u32(enc::sxtw(dst_gp, *src_gp));
            }
            MachineConvertOp::I64ExtendI32U => {
                let src_gp = prepare_gp(
                    self.core.compiled.backend(),
                    &self.core.fp_reg_widths,
                    &mut self.core.text,
                    &self.gp_scratch,
                    src,
                )?
                .detach();
                let dst_gp = self.map_gp_reg(dst)?;
                self.core.text.emit_u32(enc::mov_reg_32(dst_gp, *src_gp));
            }
            MachineConvertOp::I32ReinterpretF32 => {
                let dst_gp = self.map_gp_reg(dst)?;
                let src_gp = prepare_gp(
                    self.core.compiled.backend(),
                    &self.core.fp_reg_widths,
                    &mut self.core.text,
                    &self.gp_scratch,
                    src,
                )?
                .detach();
                if dst_gp != *src_gp {
                    self.core.text.emit_u32(enc::mov_reg_32(dst_gp, *src_gp));
                }
            }
            MachineConvertOp::I64ReinterpretF64 => {
                let dst_gp = self.map_gp_reg(dst)?;
                let src_gp = prepare_gp(
                    self.core.compiled.backend(),
                    &self.core.fp_reg_widths,
                    &mut self.core.text,
                    &self.gp_scratch,
                    src,
                )?
                .detach();
                if dst_gp != *src_gp {
                    self.core.text.emit_u32(enc::mov_reg_64(dst_gp, *src_gp));
                }
            }
            MachineConvertOp::F32ReinterpretI32 | MachineConvertOp::F64ReinterpretI64 => {
                let src_gp = prepare_gp(
                    self.core.compiled.backend(),
                    &self.core.fp_reg_widths,
                    &mut self.core.text,
                    &self.gp_scratch,
                    src,
                )?
                .detach();
                let width = dst_float_width.expect("float reinterpret width");
                let dst_fp = self.resolve_convert_fp_dst(dst, width)?;
                self.core.text.emit_u32(match width {
                    MachineFloatWidth::F32 => enc::fmov_s_from_gp(dst_fp, *src_gp),
                    MachineFloatWidth::F64 => enc::fmov_d_from_gp(dst_fp, *src_gp),
                });
                if !self.core.is_fp_reg(dst) {
                    let dst_gp = self.map_gp_reg(dst)?;
                    self.core.text.emit_u32(enc::mov_reg_64(dst_gp, *src_gp));
                }
            }
            // Float promotion / demotion
            MachineConvertOp::F64PromoteF32 => {
                let src_fp = prepare_fp(
                    self.core.compiled.backend(),
                    &self.core.fp_reg_widths,
                    &mut self.core.text,
                    &self.gp_scratch,
                    &self.fp_scratch,
                    MachineFloatWidth::F32,
                    src,
                )?
                .detach();
                let dst_fp = self.resolve_convert_fp_dst(dst, MachineFloatWidth::F64)?;
                self.core.text.emit_u32(enc::fcvt_d_from_s(dst_fp, *src_fp));
                if !self.core.is_fp_reg(dst) {
                    let dst_gp = self.map_gp_reg(dst)?;
                    self.core.text.emit_u32(enc::fmov_gp_from_d(dst_gp, dst_fp));
                }
            }
            MachineConvertOp::F32DemoteF64 => {
                let src_fp = prepare_fp(
                    self.core.compiled.backend(),
                    &self.core.fp_reg_widths,
                    &mut self.core.text,
                    &self.gp_scratch,
                    &self.fp_scratch,
                    MachineFloatWidth::F64,
                    src,
                )?
                .detach();
                let dst_fp = self.resolve_convert_fp_dst(dst, MachineFloatWidth::F32)?;
                self.core.text.emit_u32(enc::fcvt_s_from_d(dst_fp, *src_fp));
                if !self.core.is_fp_reg(dst) {
                    let dst_gp = self.map_gp_reg(dst)?;
                    self.core.text.emit_u32(enc::fmov_gp_from_s(dst_gp, dst_fp));
                }
            }
            // Int -> Float conversions
            MachineConvertOp::F32ConvertI32S => {
                let src_gp = prepare_gp(
                    self.core.compiled.backend(),
                    &self.core.fp_reg_widths,
                    &mut self.core.text,
                    &self.gp_scratch,
                    src,
                )?
                .detach();
                let dst_fp = self.resolve_convert_fp_dst(dst, MachineFloatWidth::F32)?;
                self.core.text.emit_u32(enc::scvtf_s_32(dst_fp, *src_gp));
                if !self.core.is_fp_reg(dst) {
                    let dst_gp = self.map_gp_reg(dst)?;
                    self.core.text.emit_u32(enc::fmov_gp_from_s(dst_gp, dst_fp));
                }
            }
            MachineConvertOp::F32ConvertI32U => {
                let src_gp = prepare_gp(
                    self.core.compiled.backend(),
                    &self.core.fp_reg_widths,
                    &mut self.core.text,
                    &self.gp_scratch,
                    src,
                )?
                .detach();
                let dst_fp = self.resolve_convert_fp_dst(dst, MachineFloatWidth::F32)?;
                self.core.text.emit_u32(enc::ucvtf_s_32(dst_fp, *src_gp));
                if !self.core.is_fp_reg(dst) {
                    let dst_gp = self.map_gp_reg(dst)?;
                    self.core.text.emit_u32(enc::fmov_gp_from_s(dst_gp, dst_fp));
                }
            }
            MachineConvertOp::F32ConvertI64S => {
                let src_gp = prepare_gp(
                    self.core.compiled.backend(),
                    &self.core.fp_reg_widths,
                    &mut self.core.text,
                    &self.gp_scratch,
                    src,
                )?
                .detach();
                let dst_fp = self.resolve_convert_fp_dst(dst, MachineFloatWidth::F32)?;
                self.core.text.emit_u32(enc::scvtf_s_64(dst_fp, *src_gp));
                if !self.core.is_fp_reg(dst) {
                    let dst_gp = self.map_gp_reg(dst)?;
                    self.core.text.emit_u32(enc::fmov_gp_from_s(dst_gp, dst_fp));
                }
            }
            MachineConvertOp::F32ConvertI64U => {
                let src_gp = prepare_gp(
                    self.core.compiled.backend(),
                    &self.core.fp_reg_widths,
                    &mut self.core.text,
                    &self.gp_scratch,
                    src,
                )?
                .detach();
                let dst_fp = self.resolve_convert_fp_dst(dst, MachineFloatWidth::F32)?;
                self.core.text.emit_u32(enc::ucvtf_s_64(dst_fp, *src_gp));
                if !self.core.is_fp_reg(dst) {
                    let dst_gp = self.map_gp_reg(dst)?;
                    self.core.text.emit_u32(enc::fmov_gp_from_s(dst_gp, dst_fp));
                }
            }
            MachineConvertOp::F64ConvertI32S => {
                let src_gp = prepare_gp(
                    self.core.compiled.backend(),
                    &self.core.fp_reg_widths,
                    &mut self.core.text,
                    &self.gp_scratch,
                    src,
                )?
                .detach();
                let dst_fp = self.resolve_convert_fp_dst(dst, MachineFloatWidth::F64)?;
                self.core.text.emit_u32(enc::scvtf_d_32(dst_fp, *src_gp));
                if !self.core.is_fp_reg(dst) {
                    let dst_gp = self.map_gp_reg(dst)?;
                    self.core.text.emit_u32(enc::fmov_gp_from_d(dst_gp, dst_fp));
                }
            }
            MachineConvertOp::F64ConvertI32U => {
                let src_gp = prepare_gp(
                    self.core.compiled.backend(),
                    &self.core.fp_reg_widths,
                    &mut self.core.text,
                    &self.gp_scratch,
                    src,
                )?
                .detach();
                let dst_fp = self.resolve_convert_fp_dst(dst, MachineFloatWidth::F64)?;
                self.core.text.emit_u32(enc::ucvtf_d_32(dst_fp, *src_gp));
                if !self.core.is_fp_reg(dst) {
                    let dst_gp = self.map_gp_reg(dst)?;
                    self.core.text.emit_u32(enc::fmov_gp_from_d(dst_gp, dst_fp));
                }
            }
            MachineConvertOp::F64ConvertI64S => {
                let src_gp = prepare_gp(
                    self.core.compiled.backend(),
                    &self.core.fp_reg_widths,
                    &mut self.core.text,
                    &self.gp_scratch,
                    src,
                )?
                .detach();
                let dst_fp = self.resolve_convert_fp_dst(dst, MachineFloatWidth::F64)?;
                self.core.text.emit_u32(enc::scvtf_d_64(dst_fp, *src_gp));
                if !self.core.is_fp_reg(dst) {
                    let dst_gp = self.map_gp_reg(dst)?;
                    self.core.text.emit_u32(enc::fmov_gp_from_d(dst_gp, dst_fp));
                }
            }
            MachineConvertOp::F64ConvertI64U => {
                let src_gp = prepare_gp(
                    self.core.compiled.backend(),
                    &self.core.fp_reg_widths,
                    &mut self.core.text,
                    &self.gp_scratch,
                    src,
                )?
                .detach();
                let dst_fp = self.resolve_convert_fp_dst(dst, MachineFloatWidth::F64)?;
                self.core.text.emit_u32(enc::ucvtf_d_64(dst_fp, *src_gp));
                if !self.core.is_fp_reg(dst) {
                    let dst_gp = self.map_gp_reg(dst)?;
                    self.core.text.emit_u32(enc::fmov_gp_from_d(dst_gp, dst_fp));
                }
            }
            // Trapping truncations: preserved-helper call
            MachineConvertOp::I32TruncF32S
            | MachineConvertOp::I32TruncF32U
            | MachineConvertOp::I32TruncF64S
            | MachineConvertOp::I32TruncF64U
            | MachineConvertOp::I64TruncF32S
            | MachineConvertOp::I64TruncF32U
            | MachineConvertOp::I64TruncF64S
            | MachineConvertOp::I64TruncF64U => {
                let dst_gp = self.map_gp_reg(dst)?;
                self.lower_trapping_trunc(op, dst_gp, src)?;
            }
            // Saturating truncations -- inline via native fcvtzs/fcvtzu
            // ARM64 fcvtzs/fcvtzu already matches Wasm saturating semantics:
            // NaN->0, overflow->clamp to min/max.
            MachineConvertOp::I32TruncSatF32S => {
                let src_gp = prepare_gp(
                    self.core.compiled.backend(),
                    &self.core.fp_reg_widths,
                    &mut self.core.text,
                    &self.gp_scratch,
                    src,
                )?
                .detach();
                let fp_tmp = *self.fp_scratch.scoped_alloc();
                self.core
                    .text
                    .emit_u32(enc::fmov_s_from_gp(fp_tmp, *src_gp));
                let dst_gp = self.map_gp_reg(dst)?;
                self.core.text.emit_u32(enc::fcvtzs_32_s(dst_gp, fp_tmp));
            }
            MachineConvertOp::I32TruncSatF32U => {
                let src_gp = prepare_gp(
                    self.core.compiled.backend(),
                    &self.core.fp_reg_widths,
                    &mut self.core.text,
                    &self.gp_scratch,
                    src,
                )?
                .detach();
                let fp_tmp = *self.fp_scratch.scoped_alloc();
                self.core
                    .text
                    .emit_u32(enc::fmov_s_from_gp(fp_tmp, *src_gp));
                let dst_gp = self.map_gp_reg(dst)?;
                self.core.text.emit_u32(enc::fcvtzu_32_s(dst_gp, fp_tmp));
            }
            MachineConvertOp::I32TruncSatF64S => {
                let src_gp = prepare_gp(
                    self.core.compiled.backend(),
                    &self.core.fp_reg_widths,
                    &mut self.core.text,
                    &self.gp_scratch,
                    src,
                )?
                .detach();
                let fp_tmp = *self.fp_scratch.scoped_alloc();
                self.core
                    .text
                    .emit_u32(enc::fmov_d_from_gp(fp_tmp, *src_gp));
                let dst_gp = self.map_gp_reg(dst)?;
                self.core.text.emit_u32(enc::fcvtzs_32_d(dst_gp, fp_tmp));
            }
            MachineConvertOp::I32TruncSatF64U => {
                let src_gp = prepare_gp(
                    self.core.compiled.backend(),
                    &self.core.fp_reg_widths,
                    &mut self.core.text,
                    &self.gp_scratch,
                    src,
                )?
                .detach();
                let fp_tmp = *self.fp_scratch.scoped_alloc();
                self.core
                    .text
                    .emit_u32(enc::fmov_d_from_gp(fp_tmp, *src_gp));
                let dst_gp = self.map_gp_reg(dst)?;
                self.core.text.emit_u32(enc::fcvtzu_32_d(dst_gp, fp_tmp));
            }
            MachineConvertOp::I64TruncSatF32S => {
                let src_gp = prepare_gp(
                    self.core.compiled.backend(),
                    &self.core.fp_reg_widths,
                    &mut self.core.text,
                    &self.gp_scratch,
                    src,
                )?
                .detach();
                let fp_tmp = *self.fp_scratch.scoped_alloc();
                self.core
                    .text
                    .emit_u32(enc::fmov_s_from_gp(fp_tmp, *src_gp));
                let dst_gp = self.map_gp_reg(dst)?;
                self.core.text.emit_u32(enc::fcvtzs_64_s(dst_gp, fp_tmp));
            }
            MachineConvertOp::I64TruncSatF32U => {
                let src_gp = prepare_gp(
                    self.core.compiled.backend(),
                    &self.core.fp_reg_widths,
                    &mut self.core.text,
                    &self.gp_scratch,
                    src,
                )?
                .detach();
                let fp_tmp = *self.fp_scratch.scoped_alloc();
                self.core
                    .text
                    .emit_u32(enc::fmov_s_from_gp(fp_tmp, *src_gp));
                let dst_gp = self.map_gp_reg(dst)?;
                self.core.text.emit_u32(enc::fcvtzu_64_s(dst_gp, fp_tmp));
            }
            MachineConvertOp::I64TruncSatF64S => {
                let src_gp = prepare_gp(
                    self.core.compiled.backend(),
                    &self.core.fp_reg_widths,
                    &mut self.core.text,
                    &self.gp_scratch,
                    src,
                )?
                .detach();
                let fp_tmp = *self.fp_scratch.scoped_alloc();
                self.core
                    .text
                    .emit_u32(enc::fmov_d_from_gp(fp_tmp, *src_gp));
                let dst_gp = self.map_gp_reg(dst)?;
                self.core.text.emit_u32(enc::fcvtzs_64_d(dst_gp, fp_tmp));
            }
            MachineConvertOp::I64TruncSatF64U => {
                let src_gp = prepare_gp(
                    self.core.compiled.backend(),
                    &self.core.fp_reg_widths,
                    &mut self.core.text,
                    &self.gp_scratch,
                    src,
                )?
                .detach();
                let fp_tmp = *self.fp_scratch.scoped_alloc();
                self.core
                    .text
                    .emit_u32(enc::fmov_d_from_gp(fp_tmp, *src_gp));
                let dst_gp = self.map_gp_reg(dst)?;
                self.core.text.emit_u32(enc::fcvtzu_64_d(dst_gp, fp_tmp));
            }
        }
        Ok(())
    }

    // ── Memory/table instruction lowering ────────────────────────────────────────

    fn lower_memory_grow(
        &mut self,
        mem_idx: u32,
        dst: MachineReg,
        delta: MachineValue,
    ) -> Result<(), WasmError> {
        let dst_gp = map_gp(self.core.compiled.backend(), dst)?;

        self.emit_preserved_frame_open();
        self.emit_io_store_imm(preserved_io::IMM0, mem_idx);
        self.emit_io_store_value(preserved_io::ARG0, delta)?;
        let result_scratch_idx = self.gp_scratch.alloc();
        let result_scratch = self.gp_scratch.reg(result_scratch_idx);
        self.emit_preserved_call_and_close(preserved_op::MEMORY_GROW, Some(result_scratch_idx));

        self.core
            .text
            .emit_u32(enc::mov_reg_64(dst_gp, result_scratch));
        self.gp_scratch.free_index(result_scratch_idx);
        Ok(())
    }

    fn lower_memory_fill(
        &mut self,
        mem_idx: u32,
        dest: MachineValue,
        val: MachineValue,
        len: MachineValue,
    ) -> Result<(), WasmError> {
        self.emit_preserved_frame_open();
        self.emit_io_store_imm(preserved_io::IMM0, mem_idx);
        self.emit_io_store_value(preserved_io::ARG0, dest)?;
        self.emit_io_store_value(preserved_io::ARG1, val)?;
        self.emit_io_store_value(preserved_io::ARG2, len)?;
        self.emit_preserved_call_and_close(preserved_op::MEMORY_FILL, None);
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
        self.emit_preserved_frame_open();
        self.emit_io_store_imm(preserved_io::IMM0, dst_mem);
        self.emit_io_store_imm(preserved_io::IMM1, src_mem);
        self.emit_io_store_value(preserved_io::ARG0, dest)?;
        self.emit_io_store_value(preserved_io::ARG1, src)?;
        self.emit_io_store_value(preserved_io::ARG2, len)?;
        self.emit_preserved_call_and_close(preserved_op::MEMORY_COPY, None);
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
        self.emit_preserved_frame_open();
        self.emit_io_store_imm(preserved_io::IMM0, mem_idx);
        self.emit_io_store_imm(preserved_io::IMM1, data_idx);
        self.emit_io_store_value(preserved_io::ARG0, dest)?;
        self.emit_io_store_value(preserved_io::ARG1, src)?;
        self.emit_io_store_value(preserved_io::ARG2, len)?;
        self.emit_preserved_call_and_close(preserved_op::MEMORY_INIT, None);
        Ok(())
    }

    fn lower_data_drop(&mut self, data_idx: u32) -> Result<(), WasmError> {
        self.emit_preserved_frame_open();
        self.emit_io_store_imm(preserved_io::IMM0, data_idx);
        self.emit_preserved_call_and_close(preserved_op::DATA_DROP, None);
        Ok(())
    }

    fn lower_table_grow(
        &mut self,
        table_idx: u32,
        dst: MachineReg,
        init_val: MachineValue,
        delta: MachineValue,
    ) -> Result<(), WasmError> {
        let dst_gp = map_gp(self.core.compiled.backend(), dst)?;

        self.emit_preserved_frame_open();
        self.emit_io_store_imm(preserved_io::IMM0, table_idx);
        self.emit_io_store_value(preserved_io::ARG0, init_val)?;
        self.emit_io_store_value(preserved_io::ARG1, delta)?;
        let result_scratch_idx = self.gp_scratch.alloc();
        let result_scratch = self.gp_scratch.reg(result_scratch_idx);
        self.emit_preserved_call_and_close(preserved_op::TABLE_GROW, Some(result_scratch_idx));

        self.core
            .text
            .emit_u32(enc::mov_reg_64(dst_gp, result_scratch));
        self.gp_scratch.free_index(result_scratch_idx);
        Ok(())
    }

    fn lower_table_fill(
        &mut self,
        table_idx: u32,
        start: MachineValue,
        val: MachineValue,
        len: MachineValue,
    ) -> Result<(), WasmError> {
        self.emit_preserved_frame_open();
        self.emit_io_store_imm(preserved_io::IMM0, table_idx);
        self.emit_io_store_value(preserved_io::ARG0, start)?;
        self.emit_io_store_value(preserved_io::ARG1, val)?;
        self.emit_io_store_value(preserved_io::ARG2, len)?;
        self.emit_preserved_call_and_close(preserved_op::TABLE_FILL, None);
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
        self.emit_preserved_frame_open();
        self.emit_io_store_imm(preserved_io::IMM0, dst_tbl);
        self.emit_io_store_imm(preserved_io::IMM1, src_tbl);
        self.emit_io_store_value(preserved_io::ARG0, dest)?;
        self.emit_io_store_value(preserved_io::ARG1, src)?;
        self.emit_io_store_value(preserved_io::ARG2, len)?;
        self.emit_preserved_call_and_close(preserved_op::TABLE_COPY, None);
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
        self.emit_preserved_frame_open();
        self.emit_io_store_imm(preserved_io::IMM0, table_idx);
        self.emit_io_store_imm(preserved_io::IMM1, elem_idx);
        self.emit_io_store_value(preserved_io::ARG0, dest)?;
        self.emit_io_store_value(preserved_io::ARG1, src)?;
        self.emit_io_store_value(preserved_io::ARG2, len)?;
        self.emit_preserved_call_and_close(preserved_op::TABLE_INIT, None);
        Ok(())
    }

    fn lower_elem_drop(&mut self, elem_idx: u32) -> Result<(), WasmError> {
        self.emit_preserved_frame_open();
        self.emit_io_store_imm(preserved_io::IMM0, elem_idx);
        self.emit_preserved_call_and_close(preserved_op::ELEM_DROP, None);
        Ok(())
    }

    // ── Inline trapping truncation ───────────────────────────────────────────────

    /// Inline trapping truncation: NaN check + bounds check + FCVTZS/FCVTZU.
    ///
    /// The error path is a one-way trap exit — no register preservation needed.
    /// The fast path (common case) is just a few inline instructions.
    fn lower_trapping_trunc(
        &mut self,
        op: MachineConvertOp,
        dst: Arm64Reg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let spec = trunc_spec(op);

        // Get source into an FP scratch register.
        let src_fp = *self.fp_scratch.scoped_alloc();
        let src_gp = prepare_gp(
            self.core.compiled.backend(),
            &self.core.fp_reg_widths,
            &mut self.core.text,
            &self.gp_scratch,
            src,
        )?
        .detach();
        self.core.text.emit_u32(if spec.src_f32 {
            enc::fmov_s_from_gp(src_fp, *src_gp)
        } else {
            enc::fmov_d_from_gp(src_fp, *src_gp)
        });

        // 1. NaN check: FCMP src, src — NaN ≠ NaN sets V flag.
        self.core.text.emit_u32(if spec.src_f32 {
            enc::fcmp_s(src_fp, src_fp)
        } else {
            enc::fcmp_d(src_fp, src_fp)
        });
        let trap_nan = self
            .core
            .ensure_trap_label(MachineTrapKind::InvalidConversion);
        self.lower_b_cond(enc::Cond::Vs, trap_nan);

        // 2. Upper bound check: FCMP src, upper — trap if src >= upper.
        let bound_fp = *self.fp_scratch.scoped_alloc();
        let bound_gp = *self.gp_scratch.scoped_alloc();
        materialize_u64_into(&mut self.core.text, bound_gp, spec.upper_bits);
        self.core.text.emit_u32(if spec.src_f32 {
            enc::fmov_s_from_gp(bound_fp, bound_gp)
        } else {
            enc::fmov_d_from_gp(bound_fp, bound_gp)
        });
        self.core.text.emit_u32(if spec.src_f32 {
            enc::fcmp_s(src_fp, bound_fp)
        } else {
            enc::fcmp_d(src_fp, bound_fp)
        });
        let trap_overflow = self
            .core
            .ensure_trap_label(MachineTrapKind::IntegerOverflow);
        self.lower_b_cond(enc::Cond::Ge, trap_overflow);

        // 3. Lower bound check: FCMP src, lower — trap condition depends on variant.
        materialize_u64_into(&mut self.core.text, bound_gp, spec.lower_bits);
        self.core.text.emit_u32(if spec.src_f32 {
            enc::fmov_s_from_gp(bound_fp, bound_gp)
        } else {
            enc::fmov_d_from_gp(bound_fp, bound_gp)
        });
        self.core.text.emit_u32(if spec.src_f32 {
            enc::fcmp_s(src_fp, bound_fp)
        } else {
            enc::fcmp_d(src_fp, bound_fp)
        });
        // Signed: trap if src < lower (Mi after NaN ruled out).
        // Unsigned: trap if src <= lower (Ls after NaN ruled out).
        self.lower_b_cond(spec.lower_cond, trap_overflow);

        // 4. Safe conversion — value is in range and not NaN.
        self.core.text.emit_u32((spec.fcvt)(dst, src_fp));
        Ok(())
    }
} // impl Arm64Backend (inst.rs)

// ── Trapping trunc specification ────────────────────────────────────────────

/// Compile-time specification for one trapping trunc variant.
struct TruncSpec {
    src_f32: bool,
    upper_bits: u64,
    lower_bits: u64,
    lower_cond: enc::Cond,
    fcvt: fn(Arm64Reg, Arm64FpReg) -> u32,
}

fn trunc_spec(op: MachineConvertOp) -> TruncSpec {
    match op {
        MachineConvertOp::I32TruncF32S => TruncSpec {
            src_f32: true,
            upper_bits: 2147483648.0_f32.to_bits() as u64, // 2^31
            lower_bits: (-2147483648.0_f32).to_bits() as u64, // -2^31
            lower_cond: enc::Cond::Mi,                     // trap if src < lower
            fcvt: enc::fcvtzs_32_s,
        },
        MachineConvertOp::I32TruncF32U => TruncSpec {
            src_f32: true,
            upper_bits: 4294967296.0_f32.to_bits() as u64, // 2^32
            lower_bits: (-1.0_f32).to_bits() as u64,       // -1.0
            lower_cond: enc::Cond::Ls,                     // trap if src <= lower
            fcvt: enc::fcvtzu_32_s,
        },
        MachineConvertOp::I32TruncF64S => TruncSpec {
            src_f32: false,
            upper_bits: 2147483648.0_f64.to_bits(),
            lower_bits: (-2147483649.0_f64).to_bits(),
            lower_cond: enc::Cond::Ls, // trap if src <= lower
            fcvt: enc::fcvtzs_32_d,
        },
        MachineConvertOp::I32TruncF64U => TruncSpec {
            src_f32: false,
            upper_bits: 4294967296.0_f64.to_bits(),
            lower_bits: (-1.0_f64).to_bits(),
            lower_cond: enc::Cond::Ls, // trap if src <= lower
            fcvt: enc::fcvtzu_32_d,
        },
        MachineConvertOp::I64TruncF32S => TruncSpec {
            src_f32: true,
            upper_bits: 9223372036854775808.0_f32.to_bits() as u64, // 2^63
            lower_bits: (-9223372036854775808.0_f32).to_bits() as u64, // -2^63
            lower_cond: enc::Cond::Mi,                              // trap if src < lower
            fcvt: enc::fcvtzs_64_s,
        },
        MachineConvertOp::I64TruncF32U => TruncSpec {
            src_f32: true,
            upper_bits: 18446744073709551616.0_f32.to_bits() as u64, // 2^64
            lower_bits: (-1.0_f32).to_bits() as u64,
            lower_cond: enc::Cond::Ls, // trap if src <= lower
            fcvt: enc::fcvtzu_64_s,
        },
        MachineConvertOp::I64TruncF64S => TruncSpec {
            src_f32: false,
            upper_bits: 9223372036854775808.0_f64.to_bits(),
            lower_bits: (-9223372036854775808.0_f64).to_bits(),
            lower_cond: enc::Cond::Mi, // trap if src < lower
            fcvt: enc::fcvtzs_64_d,
        },
        MachineConvertOp::I64TruncF64U => TruncSpec {
            src_f32: false,
            upper_bits: 18446744073709551616.0_f64.to_bits(),
            lower_bits: (-1.0_f64).to_bits(),
            lower_cond: enc::Cond::Ls, // trap if src <= lower
            fcvt: enc::fcvtzu_64_d,
        },
        _ => unreachable!("not a trapping trunc op"),
    }
}

// ── Free helper (not a method — operates on TextEmitter directly) ────────────

pub(super) fn materialize_u64_into(text: &mut TextEmitter, dst: Arm64Reg, value: u64) {
    if value == 0 {
        text.emit_u32(enc::mov_zero_64(dst));
        return;
    }

    // Values that fit in u32 can be materialized with the 32-bit move forms,
    // which zero the upper 32 bits of the X register. This is the case LLVM
    // takes for `mov w8, #-24575` (a single MOVN encoding 0xffffa001).
    if value <= u32::MAX as u64 {
        let lo = (value & 0xffff) as u16;
        let hi = ((value >> 16) & 0xffff) as u16;
        let movz_count = (lo != 0) as u32 + (hi != 0) as u32;
        let inv = !(value as u32);
        let inv_lo = (inv & 0xffff) as u16;
        let inv_hi = ((inv >> 16) & 0xffff) as u16;
        let movn_count = (inv_lo != 0) as u32 + (inv_hi != 0) as u32;

        if movn_count < movz_count {
            // MOVN seeds with a single inverted chunk and 1-fills the rest.
            // If the other chunk of the inverted value is non-zero, MOVK clears
            // it back to the right pattern. When both inverted chunks are zero
            // (i.e. value == 0xffffffff), MOVN with imm 0 produces all-ones.
            if inv_lo != 0 {
                text.emit_u32(enc::movn_32(dst, inv_lo, 0));
                if inv_hi != 0 {
                    text.emit_u32(enc::movk_32(dst, hi, 16));
                }
            } else if inv_hi != 0 {
                text.emit_u32(enc::movn_32(dst, inv_hi, 16));
            } else {
                // Both inverted chunks are zero → value is all-ones (0xffffffff).
                text.emit_u32(enc::movn_32(dst, 0, 0));
            }
        } else {
            // MOVZ seeds with a single chunk and 0-fills the rest, then MOVK
            // installs the other non-zero chunk.
            if lo != 0 {
                text.emit_u32(enc::movz_32(dst, lo, 0));
                if hi != 0 {
                    text.emit_u32(enc::movk_32(dst, hi, 16));
                }
            } else {
                // lo == 0 (we know value != 0, so hi != 0)
                text.emit_u32(enc::movz_32(dst, hi, 16));
            }
        }
        return;
    }

    // 64-bit value: use 64-bit move forms. Pick MOVZ chain or MOVN chain
    // based on which produces fewer instructions.
    let chunks = [
        (value & 0xffff) as u16,
        ((value >> 16) & 0xffff) as u16,
        ((value >> 32) & 0xffff) as u16,
        ((value >> 48) & 0xffff) as u16,
    ];
    let inv = !value;
    let inv_chunks = [
        (inv & 0xffff) as u16,
        ((inv >> 16) & 0xffff) as u16,
        ((inv >> 32) & 0xffff) as u16,
        ((inv >> 48) & 0xffff) as u16,
    ];
    let movz_count: u32 = chunks.iter().map(|&c| (c != 0) as u32).sum();
    let movn_count: u32 = inv_chunks.iter().map(|&c| (c != 0) as u32).sum();

    if movn_count < movz_count {
        // MOVN seeds one inverted chunk and 1-fills the rest, then MOVK
        // installs each remaining chunk that doesn't already match.
        let mut first = true;
        for (i, (&chunk, &inv_chunk)) in chunks.iter().zip(inv_chunks.iter()).enumerate() {
            let shift = (i as u32) * 16;
            if first {
                if inv_chunk != 0 {
                    text.emit_u32(enc::movn_64(dst, inv_chunk, shift));
                    first = false;
                }
            } else if chunk != 0xffff {
                text.emit_u32(enc::movk_64(dst, chunk, shift));
            }
        }
        if first {
            // All inverted chunks are zero → value is all-ones (-1).
            text.emit_u32(enc::movn_64(dst, 0, 0));
        }
    } else {
        // MOVZ chain.
        let mut first = true;
        for (i, &chunk) in chunks.iter().enumerate() {
            if chunk != 0 {
                let shift = (i as u32) * 16;
                if first {
                    text.emit_u32(enc::movz_64(dst, chunk, shift));
                    first = false;
                } else {
                    text.emit_u32(enc::movk_64(dst, chunk, shift));
                }
            }
        }
    }
}

#[cfg(test)]
mod materialize_tests {
    use super::*;
    use crate::vm::arch::arm64::reg::Arm64Reg;
    use crate::vm::arch::common::text_emitter::TextEmitter;
    use alloc::vec::Vec;

    fn materialize(value: u64) -> Vec<u32> {
        let mut text = TextEmitter::new();
        // Use x8 (a generic temp register) as the destination.
        materialize_u64_into(&mut text, Arm64Reg::from_raw(8), value);
        // Pull the encoded instructions back out as 32-bit words.
        let bytes = text.finish();
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn count_insns(value: u64) -> usize {
        materialize(value).len()
    }

    #[test]
    fn zero_uses_single_mov() {
        assert_eq!(count_insns(0), 1);
    }

    #[test]
    fn small_u32_uses_single_movz() {
        assert_eq!(count_insns(0x5), 1);
        assert_eq!(count_insns(0xa001), 1);
        assert_eq!(count_insns(0xffff), 1);
    }

    #[test]
    fn high_only_u32_uses_single_movz_shifted() {
        assert_eq!(count_insns(0x1234_0000), 1);
        assert_eq!(count_insns(0xffff_0000), 1);
    }

    #[test]
    fn high_ones_u32_uses_single_movn() {
        // 0xffffa001 — the coremark CRC constant. LLVM emits a single MOVN here.
        assert_eq!(count_insns(0xffff_a001), 1);
        // 0x80000000 (i32::MIN) — high bit only; MOVZ takes 1 insn anyway.
        assert_eq!(count_insns(0x8000_0000), 1);
    }

    #[test]
    fn all_ones_u32_uses_single_movn() {
        // 0xffffffff (= -1 as i32) — the coremark `(-1i32) as u32 as u64` path.
        assert_eq!(count_insns(0xffff_ffff), 1);
    }

    #[test]
    fn full_two_chunk_u32_uses_two_insns() {
        // No 0x0000 or 0xffff chunks → MOVZ chain wins, two insns.
        assert_eq!(count_insns(0x1234_5678), 2);
    }

    #[test]
    fn small_u64_with_three_chunks_uses_three_insns() {
        // The runtime helper trap address pattern: low 48 bits non-zero,
        // high 16 bits zero. Three insns either way.
        assert_eq!(count_insns(0x0000_0001_0276_d1ec), 3);
    }

    #[test]
    fn negative_one_u64_uses_single_movn() {
        // 0xffffffff_ffffffff — all chunks 0xffff. MOVN with imm 0 produces -1
        // in a single instruction.
        assert_eq!(count_insns(u64::MAX), 1);
    }

    #[test]
    fn high_ones_u64_uses_single_movn() {
        // 0xffffffff_ffff0000 — three chunks 0xffff and one 0x0000. MOVN with
        // the inverted 0x0000 chunk (= 0xffff_lsl_0) produces this in 1 insn.
        // Wait — that gives 0xffffffff_ffff0000? Let's check: ~(0xffff lsl 0)
        // = 0xffffffff_ffff0000. Yes.
        assert_eq!(count_insns(0xffff_ffff_ffff_0000), 1);
    }

    #[test]
    fn full_four_chunk_u64_uses_four_insns() {
        // 0x1234_5678_9abc_def0 — no chunk is 0 or 0xffff. Four insns.
        assert_eq!(count_insns(0x1234_5678_9abc_def0), 4);
    }
}
