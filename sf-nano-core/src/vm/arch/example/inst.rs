//! Ordinary instruction lowering and operand preparation.
//!
//! ## Naming convention
//!
//! - `lower_*` — translate a MachineIR node into encoded instructions.
//! - `enc::*`  — encode a single instruction into a u32 word.
//! - `text.emit_u32` — write a u32 into the code buffer.
//!
//! ## Split-borrow pattern
//!
//! `prepare_gp` and `prepare_fp` are **free functions** instead of methods
//! on `ExampleBackend`. This is a deliberate Rust borrow-checker workaround:
//!
//! The returned `PreparedGp` holds an RAII guard that borrows `&ScratchPool`.
//! If prepare were a `&mut self` method, the guard's lifetime would lock the
//! entire backend, preventing a second prepare call or any `text.emit_u32()`.
//!
//! As free functions taking disjoint field references, Rust can see that the
//! returned guard borrows only the pool (`&ScratchPool`, shared via `Cell`),
//! while `&mut TextEmitter` is reborrowed only for the call's duration.
//! This lets multiple `PreparedGp` values coexist — exactly what two-operand
//! patterns like `lower_int_binary` require.

use crate::{
    error::WasmError,
    vm::{
        arch::common::{scratch_pool::ScratchPool, text_emitter::TextEmitter},
        backend::BackendConfig,
        machine::machine_ir::{
            fp_reg_index, is_fp_reg, MachineAddr, MachineBlockParam, MachineFloatWidth,
            MachineInst, MachineInstKind, MachineIntBinaryOp, MachineIntWidth, MachineMemWidth,
            MachineReg, MachineStorageType, MachineValue,
        },
    },
};

use super::{
    abi, enc,
    operands::{PreparedFp, PreparedGp},
    reg::{FpReg, GpReg},
    select,
};
use crate::vm::arch::common::types::ParallelSource;

// ── Operand preparation (free functions) ─────────────────────────────────────

/// Write a u64 constant into `dst` via movz/movk.
pub(super) fn materialize_u64_into(text: &mut TextEmitter, dst: GpReg, value: u64) {
    text.emit_u32(enc::movz_64(dst, (value & 0xffff) as u16, 0));
    if value > 0xffff {
        text.emit_u32(enc::movk_64(dst, ((value >> 16) & 0xffff) as u16, 16));
    }
    if value > 0xffff_ffff {
        text.emit_u32(enc::movk_64(dst, ((value >> 32) & 0xffff) as u16, 32));
    }
    if value > 0xffff_ffff_ffff {
        text.emit_u32(enc::movk_64(dst, ((value >> 48) & 0xffff) as u16, 48));
    }
}

/// Map a MachineReg to a physical GP register, rejecting FP regs.
fn map_gp(config: BackendConfig, reg: MachineReg) -> Result<GpReg, WasmError> {
    if is_fp_reg(reg, config) {
        return Err(WasmError::invalid(alloc::format!(
            "expected GP register, got FP machine reg {}",
            reg.0
        )));
    }
    abi::map_gp(reg)
}

/// Map a MachineReg to a physical FP register.
fn map_fp(config: BackendConfig, reg: MachineReg) -> Result<FpReg, WasmError> {
    let index = fp_reg_index(reg, config).ok_or_else(|| {
        WasmError::invalid(alloc::format!(
            "expected FP register, got machine reg {}",
            reg.0
        ))
    })?;
    abi::map_fp(index).ok_or_else(|| {
        WasmError::invalid(alloc::format!(
            "example backend has no FP mapping for machine reg {}",
            reg.0
        ))
    })
}

/// Prepare a MachineValue as a GP register.
///
/// - `Reg(gp)` → `Mapped(physical_gp)` — no scratch used.
/// - `Reg(fp)` → scratch alloc + fmov.
/// - `Imm64`   → scratch alloc + materialize.
fn prepare_gp<'p>(
    config: BackendConfig,
    fp_widths: &[Option<MachineFloatWidth>],
    text: &mut TextEmitter,
    pool: &'p ScratchPool<GpReg, 2>,
    value: MachineValue,
) -> Result<PreparedGp<'p>, WasmError> {
    match value {
        MachineValue::Reg(reg) if is_fp_reg(reg, config) => {
            let scratch = pool.scoped_alloc();
            let src_fp = map_fp(config, reg)?;
            let index = fp_reg_index(reg, config).unwrap();
            let _width = fp_widths.get(index).and_then(|w| *w).ok_or_else(|| {
                WasmError::invalid(alloc::format!(
                    "missing float-width for machine reg {}",
                    reg.0
                ))
            })?;
            text.emit_u32(enc::fmov_gp_from_d(*scratch, src_fp));
            Ok(PreparedGp::Scratch(scratch))
        }
        MachineValue::Reg(reg) => Ok(PreparedGp::Mapped(map_gp(config, reg)?)),
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
fn prepare_fp<'p>(
    config: BackendConfig,
    fp_widths: &[Option<MachineFloatWidth>],
    text: &mut TextEmitter,
    gp_pool: &ScratchPool<GpReg, 2>,
    fp_pool: &'p ScratchPool<FpReg, 3>,
    _width: MachineFloatWidth,
    value: MachineValue,
) -> Result<PreparedFp<'p>, WasmError> {
    if let MachineValue::Reg(reg) = value {
        if is_fp_reg(reg, config) {
            return Ok(PreparedFp::Mapped(map_fp(config, reg)?));
        }
    }
    let gp = prepare_gp(config, fp_widths, text, gp_pool, value)?;
    let fp_scratch = fp_pool.scoped_alloc();
    text.emit_u32(enc::fmov_d_from_gp(*fp_scratch, *gp));
    // gp dropped here — GP scratch slot freed immediately
    Ok(PreparedFp::Scratch(fp_scratch))
}

// ── Backend methods ──────────────────────────────────────────────────────────

impl<'a> super::backend::ExampleBackend<'a> {
    // ── Register mapping (thin wrappers for convenience) ─────────────────

    pub(super) fn map_gp_reg(&self, reg: MachineReg) -> Result<GpReg, WasmError> {
        map_gp(self.core.compiled.backend(), reg)
    }

    pub(super) fn map_fp_reg(&self, reg: MachineReg) -> Result<FpReg, WasmError> {
        map_fp(self.core.compiled.backend(), reg)
    }

    pub(super) fn materialize_u64(&mut self, dst: GpReg, value: u64) {
        materialize_u64_into(&mut self.core.text, dst, value);
    }

    // ── Instruction dispatch ─────────────────────────────────────────────

    pub(super) fn lower_inst_dispatch(&mut self, inst: &MachineInst) -> Result<(), WasmError> {
        match &inst.kind {
            MachineInstKind::Move { dst, src, ty, .. } => self.lower_move(*ty, *dst, *src),
            MachineInstKind::IntBinary {
                width,
                op,
                dst,
                lhs,
                rhs,
            } => self.lower_int_binary(*width, *op, *dst, *lhs, *rhs),
            MachineInstKind::Load {
                dst, addr, width, ..
            } => self.lower_load(*dst, *addr, *width),
            MachineInstKind::Store {
                addr, width, src, ..
            } => self.lower_store(*addr, *width, *src),
            MachineInstKind::TrapIf { kind, cond } => self.lower_trap_if(*kind, cond),
            MachineInstKind::CallExternal(call) => {
                self.lower_call_external(call.metadata.0 as usize)
            }
            _ => todo!("example backend: unhandled instruction"),
        }
    }

    // ── Move ─────────────────────────────────────────────────────────────

    fn lower_move(
        &mut self,
        ty: MachineStorageType,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        if let Some(width) = ty.float_width() {
            let dst_fp = self.map_fp_reg(dst)?;
            let src_fp = prepare_fp(
                self.core.compiled.backend(),
                &self.core.fp_reg_widths,
                &mut self.core.text,
                &self.gp_scratch,
                &self.fp_scratch,
                width,
                src,
            )?;
            self.core.text.emit_u32(enc::fmov_d(dst_fp, *src_fp));
            self.core.set_fp_reg_width(dst, width)?;
            return Ok(());
        }

        let dst_gp = self.map_gp_reg(dst)?;
        let src_gp = prepare_gp(
            self.core.compiled.backend(),
            &self.core.fp_reg_widths,
            &mut self.core.text,
            &self.gp_scratch,
            src,
        )?;
        if dst_gp != *src_gp {
            self.core.text.emit_u32(enc::mov_reg_64(dst_gp, *src_gp));
        }
        Ok(())
    }

    // ── Integer binary (Add / Sub with immediate selection) ──────────────

    fn lower_int_binary(
        &mut self,
        width: MachineIntWidth,
        op: MachineIntBinaryOp,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        let dst = self.map_gp_reg(dst)?;

        // Try immediate form (selector is pure — takes physical regs only).
        if let Some(inst) = self.try_binary_imm(width, op, dst, lhs, rhs)? {
            self.core.text.emit_u32(inst);
            return Ok(());
        }

        // Register fallback: prepare both operands (may allocate scratches).
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
        let (l, r) = (*lhs, *rhs);
        let inst = match (width, op) {
            (MachineIntWidth::I32, MachineIntBinaryOp::Add) => enc::add_reg_32(dst, l, r),
            (MachineIntWidth::I64, MachineIntBinaryOp::Add) => enc::add_reg_64(dst, l, r),
            (MachineIntWidth::I32, MachineIntBinaryOp::Sub) => enc::sub_reg_32(dst, l, r),
            (MachineIntWidth::I64, MachineIntBinaryOp::Sub) => enc::sub_reg_64(dst, l, r),
            _ => todo!("example backend: unhandled int binary op"),
        };
        self.core.text.emit_u32(inst);
        Ok(())
    }

    /// Try to select a reg+imm immediate form for a binary op.
    /// The selector functions are pure — this method handles the MachineValue
    /// matching and maps the register operand.
    fn try_binary_imm(
        &self,
        _width: MachineIntWidth,
        op: MachineIntBinaryOp,
        dst: GpReg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<Option<u32>, WasmError> {
        match op {
            MachineIntBinaryOp::Add => match (lhs, rhs) {
                (MachineValue::Reg(r), MachineValue::Imm64(imm))
                | (MachineValue::Imm64(imm), MachineValue::Reg(r)) => Ok(select::add_imm_inst(
                    dst,
                    map_gp(self.core.compiled.backend(), r)?,
                    imm,
                )),
                _ => Ok(None),
            },
            MachineIntBinaryOp::Sub => match (lhs, rhs) {
                (MachineValue::Reg(r), MachineValue::Imm64(imm)) => Ok(select::sub_imm_inst(
                    dst,
                    map_gp(self.core.compiled.backend(), r)?,
                    imm,
                )),
                _ => Ok(None),
            },
            _ => Ok(None),
        }
    }

    // ── Compare ──────────────────────────────────────────────────────────

    pub(super) fn lower_cmp(
        &mut self,
        width: MachineIntWidth,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        // Try compare-immediate (pure selector).
        if let (MachineValue::Reg(r), MachineValue::Imm64(imm)) = (lhs, rhs) {
            let r = self.map_gp_reg(r)?;
            if let Some(inst) = select::cmp_imm_inst(width, r, imm) {
                self.core.text.emit_u32(inst);
                return Ok(());
            }
        }

        // Register fallback.
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
        self.core.text.emit_u32(enc::cmp_reg_64(*lhs, *rhs));
        Ok(())
    }

    // ── Load ─────────────────────────────────────────────────────────────

    fn lower_load(
        &mut self,
        dst: MachineReg,
        addr: MachineAddr,
        _width: MachineMemWidth,
    ) -> Result<(), WasmError> {
        let dst = self.map_gp_reg(dst)?;
        let base = self.map_gp_reg(addr.base)?;
        let slot = addr.offset as u64 / 8;

        if slot < 4096 {
            self.core.text.emit_u32(enc::ldr_64(dst, base, slot as u16));
        } else {
            // Large offset: materialize into scratch, add to base, load.
            let off = self.gp_scratch.scoped_alloc();
            materialize_u64_into(&mut self.core.text, *off, addr.offset as u64);
            self.core.text.emit_u32(enc::add_reg_64(*off, base, *off));
            self.core.text.emit_u32(enc::ldr_64(dst, *off, 0));
        }
        Ok(())
    }

    // ── Store ────────────────────────────────────────────────────────────

    fn lower_store(
        &mut self,
        addr: MachineAddr,
        _width: MachineMemWidth,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let base = self.map_gp_reg(addr.base)?;
        let slot = addr.offset as u64 / 8;

        // Prepare the source value (may allocate one scratch).
        let src = prepare_gp(
            self.core.compiled.backend(),
            &self.core.fp_reg_widths,
            &mut self.core.text,
            &self.gp_scratch,
            src,
        )?;

        if slot < 4096 {
            self.core
                .text
                .emit_u32(enc::str_64(*src, base, slot as u16));
        } else {
            // Large offset: allocate a second scratch for the address.
            let off = self.gp_scratch.scoped_alloc();
            materialize_u64_into(&mut self.core.text, *off, addr.offset as u64);
            self.core.text.emit_u32(enc::add_reg_64(*off, base, *off));
            self.core.text.emit_u32(enc::str_64(*src, *off, 0));
        }
        Ok(())
    }

    // ── Parallel-move source move ────────────────────────────────────────

    pub(super) fn lower_source_move_dispatch(
        &mut self,
        dst: MachineBlockParam,
        src: ParallelSource,
    ) -> Result<(), WasmError> {
        if let Some(width) = dst.ty.float_width() {
            let dst_fp = self.map_fp_reg(dst.reg)?;
            match src {
                ParallelSource::Reg { reg, .. } if self.core.is_fp_reg(reg) => {
                    let src_fp = self.map_fp_reg(reg)?;
                    self.core.text.emit_u32(enc::fmov_d(dst_fp, src_fp));
                }
                ParallelSource::Reg { reg, .. } => {
                    let src_gp = self.map_gp_reg(reg)?;
                    self.core.text.emit_u32(enc::fmov_d_from_gp(dst_fp, src_gp));
                }
                ParallelSource::FpTemp(id, _) => {
                    let temp = self.fp_scratch.reg(id);
                    self.core.text.emit_u32(enc::fmov_d(dst_fp, temp));
                }
                ParallelSource::Imm(bits) => {
                    let gp = self.gp_scratch.scoped_alloc();
                    materialize_u64_into(&mut self.core.text, *gp, bits);
                    self.core.text.emit_u32(enc::fmov_d_from_gp(dst_fp, *gp));
                }
                ParallelSource::GpTemp(id) => {
                    let temp = self.gp_scratch.reg(id);
                    self.core.text.emit_u32(enc::fmov_d_from_gp(dst_fp, temp));
                }
            }
            self.core.set_fp_reg_width(dst.reg, width)?;
        } else {
            let dst_gp = self.map_gp_reg(dst.reg)?;
            match src {
                ParallelSource::Reg { reg, .. } if self.core.is_fp_reg(reg) => {
                    let src_fp = self.map_fp_reg(reg)?;
                    self.core.text.emit_u32(enc::fmov_gp_from_d(dst_gp, src_fp));
                }
                ParallelSource::Reg { reg, .. } => {
                    let src_gp = self.map_gp_reg(reg)?;
                    if dst_gp != src_gp {
                        self.core.text.emit_u32(enc::mov_reg_64(dst_gp, src_gp));
                    }
                }
                ParallelSource::Imm(v) => {
                    materialize_u64_into(&mut self.core.text, dst_gp, v);
                }
                ParallelSource::GpTemp(id) => {
                    let temp = self.gp_scratch.reg(id);
                    self.core.text.emit_u32(enc::mov_reg_64(dst_gp, temp));
                }
                ParallelSource::FpTemp(id, _) => {
                    let temp = self.fp_scratch.reg(id);
                    self.core.text.emit_u32(enc::fmov_gp_from_d(dst_gp, temp));
                }
            }
        }
        Ok(())
    }
}
