//! Instruction emission for the ARMv7-A backend.
//!
//! ## Split-borrow pattern
//!
//! `prepare_gp` and `prepare_fp` are **free functions** instead of methods
//! on `Arm32Backend`. This is a deliberate Rust borrow-checker workaround:
//! the returned `PreparedGp`/`PreparedFp` holds an RAII guard that borrows
//! `&ScratchPool`. As free functions taking disjoint field references, Rust
//! can see that the guard borrows only the pool (via `Cell`), while
//! `&mut TextEmitter` is reborrowed only for the call's duration.

use crate::collections;

use crate::{
    error::WasmError,
    vm::{
        arch::common::{scratch_pool::ScratchPool, text_emitter::TextEmitter},
        machine::machine_ir::{
            MachineAddr, MachineBranchCond, MachineCallExternal, MachineCompareKind,
            MachineConvertOp, MachineFloatBinaryOp, MachineFloatUnaryOp, MachineFloatWidth,
            MachineInst, MachineInstKind, MachineIntBinaryOp, MachineIntUnaryOp, MachineIntWidth,
            MachineLoadExtension, MachineMemWidth, MachineReg, MachineShiftOp, MachineSign,
            MachineStorageType, MachineTrapKind, MachineValue, MACHINE_CTX_REG, MACHINE_FP_REG,
        },
    },
};

use super::{
    abi::{fp_machine_reg, map_fixed_reg, map_reg, FP_SCRATCH0},
    armv7a_f32_ceil, armv7a_f32_floor, armv7a_f32_nearest_bits, armv7a_f32_trunc, armv7a_f64_ceil,
    armv7a_f64_floor, armv7a_f64_nearest_bits, armv7a_f64_trunc, armv7a_i64_clz, armv7a_i64_ctz,
    armv7a_i64_div_s, armv7a_i64_div_u, armv7a_i64_mul, armv7a_i64_popcnt, armv7a_i64_rem_s,
    armv7a_i64_rem_u, armv7a_i64_rotl, armv7a_i64_rotr, armv7a_i64_shl, armv7a_i64_shr_s,
    armv7a_i64_shr_u, armv7a_i64s_to_f32, armv7a_i64s_to_f64, armv7a_i64u_to_f32,
    armv7a_i64u_to_f64, armv7a_saturating_trunc, armv7a_sdiv, armv7a_trapping_trunc, armv7a_udiv,
    backend::{Arm32Backend, BranchFixupKind},
    enc::{self, Cond},
    operands::{OwnedPreparedGp, PreparedFp, PreparedGp},
    reg::Arm32Reg,
    select,
};

// ── Operand preparation (free functions) ─────────────────────────────────────

/// Write a 32-bit constant into `dst` via MOV-imm or MOVW/MOVT.
pub(super) fn emit_load_u32_into(text: &mut TextEmitter, dst: Arm32Reg, value: u32) {
    if let Some((imm8, rot)) = enc::encode_arm_imm(value) {
        text.emit_u32(enc::mov_imm(dst, imm8, rot));
    } else {
        text.emit_u32(enc::movw(dst, value as u16));
        let hi = (value >> 16) as u16;
        if hi != 0 {
            text.emit_u32(enc::movt(dst, hi));
        }
    }
}

/// Prepare a MachineValue as a GP register.
///
/// - `Reg(gp)` → `Mapped(physical_gp)` — no scratch used.
/// - `Imm64`   → scratch alloc + materialize.
pub(super) fn prepare_gp<'p>(
    text: &mut TextEmitter,
    pool: &'p ScratchPool<Arm32Reg, 2>,
    value: MachineValue,
) -> Result<PreparedGp<'p>, WasmError> {
    match value {
        MachineValue::Reg(reg) => Ok(PreparedGp::Mapped(map_reg(reg)?)),
        MachineValue::ReservedReg(reg) => Err(WasmError::internal(alloc::format!(
            "armv7a prepare_gp cannot consume reserved cache register {} as a real value",
            reg.0
        ))),
        MachineValue::Imm64(v) => {
            let scratch = pool.scoped_alloc();
            emit_load_u32_into(text, *scratch, v as u32);
            Ok(PreparedGp::Scratch(scratch))
        }
    }
}

/// Prepare the rhs half of an `i64pair.(And|Or|Xor)` so it survives writing
/// the lhs into the destination pair. If the rhs physical register aliases
/// either destination half, snapshot it into owned scratch first.
fn prepare_pair_bitop_rhs(
    text: &mut TextEmitter,
    pool: &ScratchPool<Arm32Reg, 2>,
    value: MachineValue,
    dst_lo: Arm32Reg,
    dst_hi: Arm32Reg,
) -> Result<OwnedPreparedGp, WasmError> {
    match value {
        MachineValue::Reg(reg) => {
            let src = map_reg(reg)?;
            if src == dst_lo || src == dst_hi {
                let scratch = pool.scoped_alloc().detach();
                if *scratch != src {
                    text.emit_u32(enc::mov_reg(*scratch, src));
                }
                Ok(OwnedPreparedGp::Scratch(scratch))
            } else {
                Ok(OwnedPreparedGp::Mapped(src))
            }
        }
        MachineValue::Imm64(v) => {
            let scratch = pool.scoped_alloc().detach();
            emit_load_u32_into(text, *scratch, v as u32);
            Ok(OwnedPreparedGp::Scratch(scratch))
        }
        MachineValue::ReservedReg(reg) => Err(WasmError::internal(alloc::format!(
            "armv7a pair bitop rhs cannot consume reserved cache register {} as a real value",
            reg.0
        ))),
    }
}

/// Prepare a MachineValue as an FP D-register.
///
/// - `Reg(fp)` → `Mapped(d_reg)` — no scratch used.
/// - `Imm64`   → FP scratch alloc + GP scratch for materialization.
pub(super) fn prepare_fp<'p>(
    text: &mut TextEmitter,
    gp_pool: &ScratchPool<Arm32Reg, 2>,
    fp_pool: &'p ScratchPool<u32, 3>,
    width: MachineFloatWidth,
    value: MachineValue,
) -> Result<PreparedFp<'p>, WasmError> {
    match value {
        MachineValue::Reg(reg) => {
            let fp_idx = crate::vm::machine::machine_ir::fp_reg_index(
                reg,
                // We need the config but don't have it here; map_reg will fail
                // for FP regs so we detect via fp_machine_reg lookup.
                // Instead, we use the index computation from the backend.
                // This is a simplified path — callers pass FP regs only.
                crate::vm::backend::BackendConfig::new(
                    super::abi::GP_DYNAMIC.len() as u8,
                    super::abi::FP_DYNAMIC.len() as u8,
                    super::abi::GP_UNIT_BYTES,
                    8,
                ),
            )
            .ok_or_else(|| {
                WasmError::invalid(alloc::format!(
                    "armv7a prepare_fp: expected FP register, got machine reg {}",
                    reg.0
                ))
            })?;
            let d = fp_machine_reg(fp_idx).ok_or_else(|| {
                WasmError::invalid(alloc::format!(
                    "armv7a prepare_fp: FP index {} out of range",
                    fp_idx
                ))
            })?;
            Ok(PreparedFp::Mapped(d))
        }
        MachineValue::ReservedReg(reg) => Err(WasmError::internal(alloc::format!(
            "armv7a prepare_fp cannot consume reserved cache register {} as a real value",
            reg.0
        ))),
        MachineValue::Imm64(bits) => {
            let fp_scratch = fp_pool.scoped_alloc();
            match width {
                MachineFloatWidth::F64 => {
                    let gp_lo = gp_pool.scoped_alloc();
                    emit_load_u32_into(text, *gp_lo, bits as u32);
                    let gp_hi = gp_pool.scoped_alloc();
                    emit_load_u32_into(text, *gp_hi, (bits >> 32) as u32);
                    text.emit_u32(enc::vmov_d_rr(*fp_scratch, *gp_lo, *gp_hi));
                    // gp_lo, gp_hi dropped here — GP scratches freed
                }
                MachineFloatWidth::F32 => {
                    let gp = gp_pool.scoped_alloc();
                    emit_load_u32_into(text, *gp, bits as u32);
                    text.emit_u32(enc::vmov_s_r(*fp_scratch * 2, *gp));
                }
            }
            Ok(PreparedFp::Scratch(fp_scratch))
        }
    }
}

// ── Additional free-function helpers ─────────────────────────────────────────

/// Load a word from memory `[base + offset]` into `dst`.
///
/// For out-of-range offsets (`|offset| > 4095`) the function uses `dst`
/// itself as an address scratch.  This is safe because `dst` is about to
/// be overwritten and, when it comes from the scratch pool (R12 / R14),
/// it can never alias a mapped register that serves as `base`.
pub(super) fn emit_load_word_into(
    text: &mut TextEmitter,
    dst: Arm32Reg,
    base: Arm32Reg,
    offset: i32,
) {
    if (-4095..=4095).contains(&offset) {
        text.emit_u32(enc::ldr_imm(dst, base, offset));
    } else {
        emit_load_u32_into(text, dst, offset as u32);
        text.emit_u32(enc::add_reg(dst, base, dst));
        text.emit_u32(enc::ldr_imm(dst, dst, 0));
    }
}

/// Store a word from `src` to memory `[base + offset]`.
///
/// For out-of-range offsets the function allocates a scratch from `pool`
/// to hold the computed address.
pub(super) fn emit_store_word_to(
    text: &mut TextEmitter,
    pool: &ScratchPool<Arm32Reg, 2>,
    src: Arm32Reg,
    base: Arm32Reg,
    offset: i32,
) {
    if (-4095..=4095).contains(&offset) {
        text.emit_u32(enc::str_imm(src, base, offset));
    } else {
        let tmp = pool.scoped_alloc();
        emit_load_u32_into(text, *tmp, offset as u32);
        text.emit_u32(enc::add_reg(*tmp, base, *tmp));
        text.emit_u32(enc::str_imm(src, *tmp, 0));
    }
}

fn emit_load_byte_into(
    text: &mut TextEmitter,
    dst: Arm32Reg,
    base: Arm32Reg,
    offset: i32,
    sign_extend: bool,
) {
    if (-4095..=4095).contains(&offset) {
        let inst = if sign_extend {
            enc::ldrsb_imm(dst, base, offset)
        } else {
            enc::ldrb_imm(dst, base, offset)
        };
        text.emit_u32(inst);
    } else {
        emit_load_u32_into(text, dst, offset as u32);
        text.emit_u32(enc::add_reg(dst, base, dst));
        let inst = if sign_extend {
            enc::ldrsb_imm(dst, dst, 0)
        } else {
            enc::ldrb_imm(dst, dst, 0)
        };
        text.emit_u32(inst);
    }
}

fn emit_load_half_into(
    text: &mut TextEmitter,
    dst: Arm32Reg,
    base: Arm32Reg,
    offset: i32,
    sign_extend: bool,
) {
    if (-255..=255).contains(&offset) {
        let inst = if sign_extend {
            enc::ldrsh_imm(dst, base, offset)
        } else {
            enc::ldrh_imm(dst, base, offset)
        };
        text.emit_u32(inst);
    } else {
        emit_load_u32_into(text, dst, offset as u32);
        text.emit_u32(enc::add_reg(dst, base, dst));
        let inst = if sign_extend {
            enc::ldrsh_imm(dst, dst, 0)
        } else {
            enc::ldrh_imm(dst, dst, 0)
        };
        text.emit_u32(inst);
    }
}

fn emit_store_byte_to(
    text: &mut TextEmitter,
    pool: &ScratchPool<Arm32Reg, 2>,
    src: Arm32Reg,
    base: Arm32Reg,
    offset: i32,
) {
    if (-4095..=4095).contains(&offset) {
        text.emit_u32(enc::strb_imm(src, base, offset));
    } else {
        let tmp = pool.scoped_alloc();
        emit_load_u32_into(text, *tmp, offset as u32);
        text.emit_u32(enc::add_reg(*tmp, base, *tmp));
        text.emit_u32(enc::strb_imm(src, *tmp, 0));
    }
}

fn emit_store_half_to(
    text: &mut TextEmitter,
    pool: &ScratchPool<Arm32Reg, 2>,
    src: Arm32Reg,
    base: Arm32Reg,
    offset: i32,
) {
    if (-255..=255).contains(&offset) {
        text.emit_u32(enc::strh_imm(src, base, offset));
    } else {
        let tmp = pool.scoped_alloc();
        emit_load_u32_into(text, *tmp, offset as u32);
        text.emit_u32(enc::add_reg(*tmp, base, *tmp));
        text.emit_u32(enc::strh_imm(src, *tmp, 0));
    }
}

/// Emit a MOVW/MOVT pair with placeholder zeros — used for addresses that
/// will be patched later.  Returns the byte offset of the MOVW instruction.
pub(super) fn emit_patchable_addr_into(text: &mut TextEmitter, dst: Arm32Reg) -> usize {
    let offset = text.len();
    text.emit_u32(enc::movw(dst, 0));
    text.emit_u32(enc::movt(dst, 0));
    offset
}

// ─── Top-level instruction dispatch ─────────────────────────────────────────

impl<'a> Arm32Backend<'a> {
    pub(super) fn lower_inst_dispatch(&mut self, inst: &MachineInst) -> Result<(), WasmError> {
        match &inst.kind {
            MachineInstKind::Move { ty, dst, src, .. } => {
                let dst_is_fp = self.is_fp_machine_reg(*dst);
                let src_is_fp = match src {
                    MachineValue::Reg(r) => self.is_fp_machine_reg(*r),
                    MachineValue::ReservedReg(reg) => {
                        return Err(WasmError::internal(alloc::format!(
                            "armv7a Move cannot consume reserved cache register {} as source",
                            reg.0
                        )));
                    }
                    MachineValue::Imm64(_) => false,
                };

                if dst_is_fp && src_is_fp {
                    // FP → FP move (D-register)
                    let dd = self.map_fp_dreg(*dst)?;
                    let dm = self.map_fp_dreg(match src {
                        MachineValue::Reg(r) => *r,
                        _ => unreachable!(),
                    })?;
                    if dd != dm {
                        self.core.text.emit_u32(enc::vmov_d(dd, dm));
                    }
                    if let Some(w) = ty.float_width() {
                        self.core.set_fp_reg_width(*dst, w)?;
                    }
                } else if dst_is_fp {
                    // GP/Imm → FP: load to GP scratch then VMOV to D-reg
                    let dd = self.map_fp_dreg(*dst)?;
                    match src {
                        MachineValue::Reg(r) => {
                            let src_hw = map_reg(*r)?;
                            // Move GP value to low half of D-register
                            // (zero-extend the high half via a fresh GP
                            // scratch — never R1, which may hold live JIT
                            // state).
                            let zero_s = self.gp_scratch.scoped_alloc();
                            emit_load_u32_into(&mut self.core.text, *zero_s, 0);
                            self.core.text.emit_u32(enc::vmov_d_rr(dd, src_hw, *zero_s));
                        }
                        MachineValue::ReservedReg(reg) => {
                            return Err(WasmError::internal(alloc::format!(
                                "armv7a Move GP->FP cannot consume reserved cache register {} as source",
                                reg.0
                            )));
                        }
                        MachineValue::Imm64(imm) => {
                            // Materialize both halves into JIT GP
                            // scratches (R12/R14), not R0/R1, so live
                            // dynamic-bank values are preserved.
                            let lo = *imm as u32;
                            let hi = (*imm >> 32) as u32;
                            let lo_s = self.gp_scratch.scoped_alloc();
                            let hi_s = self.gp_scratch.scoped_alloc();
                            emit_load_u32_into(&mut self.core.text, *lo_s, lo);
                            emit_load_u32_into(&mut self.core.text, *hi_s, hi);
                            self.core.text.emit_u32(enc::vmov_d_rr(dd, *lo_s, *hi_s));
                        }
                    }
                    if let Some(w) = ty.float_width() {
                        self.core.set_fp_reg_width(*dst, w)?;
                    }
                } else if src_is_fp {
                    // FP → GP: VMOV from D-reg low word to GP. The
                    // `vmov_rr_d` instruction writes both halves, so we
                    // pass a fresh GP scratch as the high half so the
                    // existing JIT register file is not clobbered.
                    let dst_hw = map_reg(*dst)?;
                    let dm = self.map_fp_dreg(match src {
                        MachineValue::Reg(r) => *r,
                        _ => unreachable!(),
                    })?;
                    let hi_scratch = self.gp_scratch.scoped_alloc();
                    self.core
                        .text
                        .emit_u32(enc::vmov_rr_d(dst_hw, *hi_scratch, dm));
                    // dst_hw now has the low 32 bits.
                } else {
                    // GP → GP or Imm → GP
                    let dst_hw = map_reg(*dst)?;
                    match src {
                        MachineValue::Reg(r) => {
                            let src_hw = map_reg(*r)?;
                            if dst_hw != src_hw {
                                self.core.text.emit_u32(enc::mov_reg(dst_hw, src_hw));
                            }
                        }
                        MachineValue::ReservedReg(reg) => {
                            return Err(WasmError::internal(alloc::format!(
                                "armv7a Move GP->GP cannot consume reserved cache register {} as source",
                                reg.0
                            )));
                        }
                        MachineValue::Imm64(imm) => {
                            self.emit_load_u32(dst_hw, *imm as u32);
                        }
                    }
                }
            }

            MachineInstKind::FloatConst { width, dst, bits } => {
                // Load FP constant: put bits in GP scratch, then VMOV to FP reg
                let dd = self.map_fp_dreg(*dst)?;
                match width {
                    MachineFloatWidth::F32 => {
                        let lo = *bits as u32;
                        let s = self.gp_scratch.scoped_alloc();
                        emit_load_u32_into(&mut self.core.text, *s, lo);
                        self.core.text.emit_u32(enc::vmov_s_r(dd * 2, *s));
                    }
                    MachineFloatWidth::F64 => {
                        // Use two GP scratches (not R0/R1) so live JIT
                        // values in R0..R3 / R9 stay intact.
                        let lo = *bits as u32;
                        let hi = (*bits >> 32) as u32;
                        let lo_s = self.gp_scratch.scoped_alloc();
                        let hi_s = self.gp_scratch.scoped_alloc();
                        emit_load_u32_into(&mut self.core.text, *lo_s, lo);
                        emit_load_u32_into(&mut self.core.text, *hi_s, hi);
                        self.core.text.emit_u32(enc::vmov_d_rr(dd, *lo_s, *hi_s));
                    }
                }
                self.core.set_fp_reg_width(*dst, *width)?;
            }

            MachineInstKind::Load {
                dst,
                addr,
                width,
                extension,
                ..
            } => {
                self.compile_load(*dst, addr, *width, *extension)?;
            }

            MachineInstKind::Store {
                ty,
                addr,
                width,
                src,
            } => {
                self.compile_store(*ty, addr, *width, src)?;
            }

            MachineInstKind::IntBinary {
                width,
                op,
                dst,
                lhs,
                rhs,
            } => {
                self.compile_int_binary(*width, *op, *dst, lhs, rhs)?;
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
                self.compile_int64_pair_binary(
                    *op, *dst_lo, *dst_hi, lhs_lo, lhs_hi, rhs_lo, rhs_hi,
                )?;
            }

            MachineInstKind::Int64PairUnary {
                op,
                dst_lo,
                dst_hi,
                src_lo,
                src_hi,
            } => {
                self.compile_int64_pair_unary(*op, *dst_lo, *dst_hi, src_lo, src_hi)?;
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
                self.compile_int64_pair_div_rem(
                    *sign, *rem, *dst_lo, *dst_hi, lhs_lo, lhs_hi, rhs_lo, rhs_hi,
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
                self.compile_int64_pair_shift(*op, *dst_lo, *dst_hi, lhs_lo, lhs_hi, rhs)?;
            }

            MachineInstKind::IntUnary {
                width,
                op,
                dst,
                src,
            } => {
                self.compile_int_unary(*width, *op, *dst, src)?;
            }

            MachineInstKind::IntCompare {
                width,
                kind,
                sign,
                dst,
                lhs,
                rhs,
            } => {
                self.compile_int_compare(*width, *kind, *sign, *dst, lhs, rhs)?;
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
                self.compile_int64_pair_compare(
                    *kind, *sign, *dst, lhs_lo, lhs_hi, rhs_lo, rhs_hi,
                )?;
            }

            MachineInstKind::FloatBinary {
                width,
                op,
                dst,
                lhs,
                rhs,
            } => {
                self.compile_float_binary(*width, *op, *dst, lhs, rhs)?;
                self.core.set_fp_reg_width(*dst, *width)?;
            }

            MachineInstKind::FloatUnary {
                width,
                op,
                dst,
                src,
            } => {
                self.compile_float_unary(*width, *op, *dst, src)?;
                self.core.set_fp_reg_width(*dst, *width)?;
            }

            MachineInstKind::FloatCompare {
                width,
                kind,
                dst,
                lhs,
                rhs,
            } => {
                self.compile_float_compare(*width, *kind, *dst, lhs, rhs)?;
                // FloatCompare dst is GP (i32 boolean), not FP — no width tracking needed.
            }

            MachineInstKind::Convert { op, dst, src } => {
                self.compile_convert(*op, *dst, src)?;
                if let Some(w) = crate::vm::arch::common::helpers::convert_result_float_width(*op) {
                    self.core.set_fp_reg_width(*dst, w)?;
                }
            }

            MachineInstKind::ConvertI64PairToFloat {
                width,
                sign,
                dst,
                src_lo,
                src_hi,
            } => {
                self.compile_convert_i64_pair_to_float(*width, *sign, *dst, src_lo, src_hi)?;
                self.core.set_fp_reg_width(*dst, *width)?;
            }

            MachineInstKind::ConvertFloatToI64Pair {
                op,
                dst_lo,
                dst_hi,
                src,
            } => {
                self.compile_convert_float_to_i64_pair(*op, *dst_lo, *dst_hi, src)?;
            }

            MachineInstKind::ReinterpretF64ToI64Pair {
                dst_lo,
                dst_hi,
                src,
            } => {
                self.compile_reinterpret_f64_to_i64_pair(*dst_lo, *dst_hi, src)?;
            }

            MachineInstKind::ReinterpretI64PairToF64 {
                dst,
                src_lo,
                src_hi,
            } => {
                self.compile_reinterpret_i64_pair_to_f64(*dst, src_lo, src_hi)?;
                self.core.set_fp_reg_width(*dst, MachineFloatWidth::F64)?;
            }

            MachineInstKind::Select {
                ty,
                dst,
                on_true,
                on_false,
                cond,
            } => {
                self.compile_select(*dst, cond, on_true, on_false)?;
                if let Some(w) = ty.float_width() {
                    self.core.set_fp_reg_width(*dst, w)?;
                }
            }

            MachineInstKind::TrapIf { kind, cond } => {
                self.compile_trap_if(*kind, cond)?;
            }

            MachineInstKind::CallExternal(call) => {
                self.compile_call_external(call)?;
            }
            MachineInstKind::IndexedLoad {
                dst,
                base,
                index,
                offset,
                width,
                extension,
                ..
            } => {
                // Decompose: base + index + offset → scratch, then load from [scratch].
                // ARMv7 is 32-bit — no UXTW needed (index_extend is ignored).
                let base_hw = map_reg(*base)?;
                let index_hw = map_reg(*index)?;
                let s = self.gp_scratch.scoped_alloc().detach();
                self.core.text.emit_u32(enc::add_reg(*s, base_hw, index_hw));
                self.compile_load_from_base_hw(*dst, *s, *offset, *width, *extension)?;
            }
            MachineInstKind::IndexedStore {
                base,
                index,
                offset,
                width,
                src,
                ..
            } => {
                let base_hw = map_reg(*base)?;
                let index_hw = map_reg(*index)?;
                let s = self.gp_scratch.scoped_alloc().detach();
                self.core.text.emit_u32(enc::add_reg(*s, base_hw, index_hw));
                self.compile_store_from_base_hw(
                    MachineStorageType::GpWord,
                    *s,
                    *offset,
                    *width,
                    src,
                )?;
            }
            MachineInstKind::BitfieldExtractU {
                width,
                dst,
                src,
                lsb,
                bits,
            } => {
                self.compile_bitfield_extract_u(*width, *dst, *src, *lsb, *bits)?;
            }
            MachineInstKind::IntBinaryShifted {
                width,
                op,
                dst,
                lhs,
                rhs,
                shift,
                amount,
            } => {
                self.compile_int_binary_shifted(*width, *op, *dst, *lhs, *rhs, *shift, *amount)?;
            }
            MachineInstKind::TestBits {
                width,
                kind,
                dst,
                src,
                mask,
            } => {
                self.compile_test_bits(*width, *kind, *dst, *src, mask)?;
            }

            // ─── Bulk memory / table ops via preserved-helper bridge ──────
            // All these go through the shared `preserved_entry` runtime
            // helper. The bridge in backend.rs handles the I/O area, the
            // GP/FP register spill/restore, and the post-call status check.
            MachineInstKind::MemoryGrow {
                mem_idx,
                dst,
                delta,
            } => {
                self.emit_preserved_helper_call(
                    crate::vm::runtime::preserved::op::MEMORY_GROW,
                    &[(crate::vm::runtime::preserved::io::IMM0, *mem_idx)],
                    &[(crate::vm::runtime::preserved::io::ARG0, *delta)],
                    Some(*dst),
                )?;
            }
            MachineInstKind::MemoryFill {
                mem_idx,
                dest,
                val,
                len,
            } => {
                self.emit_preserved_helper_call(
                    crate::vm::runtime::preserved::op::MEMORY_FILL,
                    &[(crate::vm::runtime::preserved::io::IMM0, *mem_idx)],
                    &[
                        (crate::vm::runtime::preserved::io::ARG0, *dest),
                        (crate::vm::runtime::preserved::io::ARG1, *val),
                        (crate::vm::runtime::preserved::io::ARG2, *len),
                    ],
                    None,
                )?;
            }
            MachineInstKind::MemoryCopy {
                dst_mem,
                src_mem,
                dest,
                src,
                len,
            } => {
                self.emit_preserved_helper_call(
                    crate::vm::runtime::preserved::op::MEMORY_COPY,
                    &[
                        (crate::vm::runtime::preserved::io::IMM0, *dst_mem),
                        (crate::vm::runtime::preserved::io::IMM1, *src_mem),
                    ],
                    &[
                        (crate::vm::runtime::preserved::io::ARG0, *dest),
                        (crate::vm::runtime::preserved::io::ARG1, *src),
                        (crate::vm::runtime::preserved::io::ARG2, *len),
                    ],
                    None,
                )?;
            }
            MachineInstKind::MemoryInit {
                mem_idx,
                data_idx,
                dest,
                src,
                len,
            } => {
                self.emit_preserved_helper_call(
                    crate::vm::runtime::preserved::op::MEMORY_INIT,
                    &[
                        (crate::vm::runtime::preserved::io::IMM0, *mem_idx),
                        (crate::vm::runtime::preserved::io::IMM1, *data_idx),
                    ],
                    &[
                        (crate::vm::runtime::preserved::io::ARG0, *dest),
                        (crate::vm::runtime::preserved::io::ARG1, *src),
                        (crate::vm::runtime::preserved::io::ARG2, *len),
                    ],
                    None,
                )?;
            }
            MachineInstKind::DataDrop { data_idx } => {
                self.emit_preserved_helper_call(
                    crate::vm::runtime::preserved::op::DATA_DROP,
                    &[(crate::vm::runtime::preserved::io::IMM0, *data_idx)],
                    &[],
                    None,
                )?;
            }
            MachineInstKind::TableGrow {
                table_idx,
                dst,
                init_val,
                delta,
            } => {
                self.emit_preserved_helper_call(
                    crate::vm::runtime::preserved::op::TABLE_GROW,
                    &[(crate::vm::runtime::preserved::io::IMM0, *table_idx)],
                    &[
                        (crate::vm::runtime::preserved::io::ARG0, *init_val),
                        (crate::vm::runtime::preserved::io::ARG1, *delta),
                    ],
                    Some(*dst),
                )?;
            }
            MachineInstKind::TableFill {
                table_idx,
                start,
                val,
                len,
            } => {
                self.emit_preserved_helper_call(
                    crate::vm::runtime::preserved::op::TABLE_FILL,
                    &[(crate::vm::runtime::preserved::io::IMM0, *table_idx)],
                    &[
                        (crate::vm::runtime::preserved::io::ARG0, *start),
                        (crate::vm::runtime::preserved::io::ARG1, *val),
                        (crate::vm::runtime::preserved::io::ARG2, *len),
                    ],
                    None,
                )?;
            }
            MachineInstKind::TableCopy {
                dst_tbl,
                src_tbl,
                dest,
                src,
                len,
            } => {
                self.emit_preserved_helper_call(
                    crate::vm::runtime::preserved::op::TABLE_COPY,
                    &[
                        (crate::vm::runtime::preserved::io::IMM0, *dst_tbl),
                        (crate::vm::runtime::preserved::io::IMM1, *src_tbl),
                    ],
                    &[
                        (crate::vm::runtime::preserved::io::ARG0, *dest),
                        (crate::vm::runtime::preserved::io::ARG1, *src),
                        (crate::vm::runtime::preserved::io::ARG2, *len),
                    ],
                    None,
                )?;
            }
            MachineInstKind::TableInit {
                table_idx,
                elem_idx,
                dest,
                src,
                len,
            } => {
                self.emit_preserved_helper_call(
                    crate::vm::runtime::preserved::op::TABLE_INIT,
                    &[
                        (crate::vm::runtime::preserved::io::IMM0, *table_idx),
                        (crate::vm::runtime::preserved::io::IMM1, *elem_idx),
                    ],
                    &[
                        (crate::vm::runtime::preserved::io::ARG0, *dest),
                        (crate::vm::runtime::preserved::io::ARG1, *src),
                        (crate::vm::runtime::preserved::io::ARG2, *len),
                    ],
                    None,
                )?;
            }
            MachineInstKind::ElemDrop { elem_idx } => {
                self.emit_preserved_helper_call(
                    crate::vm::runtime::preserved::op::ELEM_DROP,
                    &[(crate::vm::runtime::preserved::io::IMM0, *elem_idx)],
                    &[],
                    None,
                )?;
            }
        }
        Ok(())
    }

    // ─── Load/Store helpers ─────────────────────────────────────────────────────

    fn compile_load(
        &mut self,
        dst: MachineReg,
        addr: &MachineAddr,
        width: MachineMemWidth,
        extension: MachineLoadExtension,
    ) -> Result<(), WasmError> {
        let base_hw = map_reg(addr.base)?;
        self.compile_load_from_base_hw(dst, base_hw, addr.offset, width, extension)
    }

    fn compile_load_from_base_hw(
        &mut self,
        dst: MachineReg,
        base_hw: Arm32Reg,
        offset: i32,
        width: MachineMemWidth,
        extension: MachineLoadExtension,
    ) -> Result<(), WasmError> {
        // ARMv7 VFP loads/stores require alignment that Wasm memory does not
        // guarantee. MachineAddr does not currently preserve enough provenance to
        // distinguish "provably aligned frame/context slot" from "possibly
        // unaligned Wasm memory address", so use a GP-word bridge here for
        // correctness.
        if self.is_fp_machine_reg(dst) {
            let dd = self.map_fp_dreg(dst)?;
            match width {
                MachineMemWidth::U64 => {
                    let s0 = self.gp_scratch.scoped_alloc();
                    let s1 = self.gp_scratch.scoped_alloc();
                    emit_load_word_into(&mut self.core.text, *s0, base_hw, offset);
                    emit_load_word_into(&mut self.core.text, *s1, base_hw, offset + 4);
                    self.core.text.emit_u32(enc::vmov_d_rr(dd, *s0, *s1));
                }
                MachineMemWidth::U32 => {
                    let s = self.gp_scratch.scoped_alloc();
                    emit_load_word_into(&mut self.core.text, *s, base_hw, offset);
                    self.core.text.emit_u32(enc::vmov_s_r(dd * 2, *s));
                }
                _ => {
                    return Err(WasmError::invalid(alloc::format!(
                        "armv7a: unsupported FP load width {:?}",
                        width
                    )));
                }
            }
            let tracked = if width == MachineMemWidth::U32 {
                MachineFloatWidth::F32
            } else {
                MachineFloatWidth::F64
            };
            self.core.set_fp_reg_width(dst, tracked)?;
            return Ok(());
        }

        // GP destination: use LDR/LDRB/LDRH etc.
        let dst_hw = map_reg(dst)?;
        match width {
            MachineMemWidth::U8 => match extension {
                MachineLoadExtension::SignExtend => {
                    emit_load_byte_into(&mut self.core.text, dst_hw, base_hw, offset, true);
                }
                _ => {
                    emit_load_byte_into(&mut self.core.text, dst_hw, base_hw, offset, false);
                }
            },
            MachineMemWidth::U16 => match extension {
                MachineLoadExtension::SignExtend => {
                    emit_load_half_into(&mut self.core.text, dst_hw, base_hw, offset, true);
                }
                _ => {
                    emit_load_half_into(&mut self.core.text, dst_hw, base_hw, offset, false);
                }
            },
            MachineMemWidth::U32 => {
                emit_load_word_into(&mut self.core.text, dst_hw, base_hw, offset);
            }
            MachineMemWidth::U64 => {
                // 64-bit load to GP: load low 32 bits only
                emit_load_word_into(&mut self.core.text, dst_hw, base_hw, offset);
            }
        }
        Ok(())
    }

    fn compile_store(
        &mut self,
        ty: MachineStorageType,
        addr: &MachineAddr,
        width: MachineMemWidth,
        src: &MachineValue,
    ) -> Result<(), WasmError> {
        let base_hw = map_reg(addr.base)?;
        self.compile_store_from_base_hw(ty, base_hw, addr.offset, width, src)
    }

    fn compile_store_from_base_hw(
        &mut self,
        ty: MachineStorageType,
        base_hw: Arm32Reg,
        offset: i32,
        width: MachineMemWidth,
        src: &MachineValue,
    ) -> Result<(), WasmError> {
        if matches!(ty, MachineStorageType::Fp32 | MachineStorageType::Fp64) {
            match src {
                MachineValue::Reg(r) if self.is_fp_machine_reg(*r) => {
                    let dd = self.map_fp_dreg(*r)?;
                    match width {
                        MachineMemWidth::U64 => {
                            {
                                let s = self.gp_scratch.scoped_alloc();
                                self.core.text.emit_u32(enc::vmov_r_s(*s, dd * 2));
                                emit_store_word_to(
                                    &mut self.core.text,
                                    &self.gp_scratch,
                                    *s,
                                    base_hw,
                                    offset,
                                );
                            }
                            {
                                let s = self.gp_scratch.scoped_alloc();
                                self.core.text.emit_u32(enc::vmov_r_s(*s, dd * 2 + 1));
                                emit_store_word_to(
                                    &mut self.core.text,
                                    &self.gp_scratch,
                                    *s,
                                    base_hw,
                                    offset + 4,
                                );
                            }
                        }
                        MachineMemWidth::U32 => {
                            let s = self.gp_scratch.scoped_alloc();
                            self.core.text.emit_u32(enc::vmov_r_s(*s, dd * 2));
                            emit_store_word_to(
                                &mut self.core.text,
                                &self.gp_scratch,
                                *s,
                                base_hw,
                                offset,
                            );
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
                        {
                            let s = self.gp_scratch.scoped_alloc();
                            emit_load_u32_into(&mut self.core.text, *s, *bits as u32);
                            emit_store_word_to(
                                &mut self.core.text,
                                &self.gp_scratch,
                                *s,
                                base_hw,
                                offset,
                            );
                        }
                        {
                            let s = self.gp_scratch.scoped_alloc();
                            emit_load_u32_into(&mut self.core.text, *s, (*bits >> 32) as u32);
                            emit_store_word_to(
                                &mut self.core.text,
                                &self.gp_scratch,
                                *s,
                                base_hw,
                                offset + 4,
                            );
                        }
                    }
                    MachineMemWidth::U32 => {
                        let s = self.gp_scratch.scoped_alloc();
                        emit_load_u32_into(&mut self.core.text, *s, *bits as u32);
                        emit_store_word_to(
                            &mut self.core.text,
                            &self.gp_scratch,
                            *s,
                            base_hw,
                            offset,
                        );
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
                MachineValue::ReservedReg(reg) => {
                    return Err(WasmError::internal(alloc::format!(
                        "armv7a FP store cannot consume reserved cache register {} as source value",
                        reg.0
                    )));
                }
            }
            return Ok(());
        }

        // GP source
        let src_hw = prepare_gp(&mut self.core.text, &self.gp_scratch, *src)?.detach();

        match width {
            MachineMemWidth::U8 => {
                emit_store_byte_to(
                    &mut self.core.text,
                    &self.gp_scratch,
                    *src_hw,
                    base_hw,
                    offset,
                );
            }
            MachineMemWidth::U16 => {
                emit_store_half_to(
                    &mut self.core.text,
                    &self.gp_scratch,
                    *src_hw,
                    base_hw,
                    offset,
                );
            }
            MachineMemWidth::U32 => {
                emit_store_word_to(
                    &mut self.core.text,
                    &self.gp_scratch,
                    *src_hw,
                    base_hw,
                    offset,
                );
            }
            MachineMemWidth::U64 => {
                // Store low word, then zero high word
                emit_store_word_to(
                    &mut self.core.text,
                    &self.gp_scratch,
                    *src_hw,
                    base_hw,
                    offset,
                );
                let s = self.gp_scratch.scoped_alloc();
                emit_load_u32_into(&mut self.core.text, *s, 0);
                emit_store_word_to(
                    &mut self.core.text,
                    &self.gp_scratch,
                    *s,
                    base_hw,
                    offset + 4,
                );
            }
        }
        Ok(())
    }

    // ─── Integer ALU ────────────────────────────────────────────────────────────

    fn compile_int_binary(
        &mut self,
        _width: MachineIntWidth,
        op: MachineIntBinaryOp,
        dst: MachineReg,
        lhs: &MachineValue,
        rhs: &MachineValue,
    ) -> Result<(), WasmError> {
        let dst_hw = map_reg(dst)?;

        let lhs_hw = match lhs {
            MachineValue::Reg(r) => OwnedPreparedGp::Mapped(map_reg(*r)?),
            MachineValue::ReservedReg(reg) => {
                return Err(WasmError::internal(alloc::format!(
                    "armv7a compile_int_binary cannot consume reserved cache register {} as lhs",
                    reg.0
                )));
            }
            MachineValue::Imm64(v) => {
                // Materialize into a scratch if rhs is a register that could
                // alias dst_hw.  Writing dst_hw first would clobber rhs.
                let rhs_aliases_dst =
                    matches!(rhs, MachineValue::Reg(r) if map_reg(*r).ok() == Some(dst_hw));
                if rhs_aliases_dst {
                    let s = self.gp_scratch.scoped_alloc().detach();
                    emit_load_u32_into(&mut self.core.text, *s, *v as u32);
                    OwnedPreparedGp::Scratch(s)
                } else {
                    emit_load_u32_into(&mut self.core.text, dst_hw, *v as u32);
                    OwnedPreparedGp::Mapped(dst_hw)
                }
            }
        };
        let lhs_hw = *lhs_hw;

        match op {
            MachineIntBinaryOp::Add => match rhs {
                MachineValue::Imm64(imm) => {
                    if let Some((imm8, rot)) = enc::encode_arm_imm(*imm as u32) {
                        self.core
                            .text
                            .emit_u32(enc::add_imm(dst_hw, lhs_hw, imm8, rot));
                    } else {
                        let s = self.gp_scratch.scoped_alloc();
                        emit_load_u32_into(&mut self.core.text, *s, *imm as u32);
                        self.core.text.emit_u32(enc::add_reg(dst_hw, lhs_hw, *s));
                    }
                }
                MachineValue::Reg(r) => {
                    self.core
                        .text
                        .emit_u32(enc::add_reg(dst_hw, lhs_hw, map_reg(*r)?));
                }
                MachineValue::ReservedReg(reg) => {
                    return Err(WasmError::internal(alloc::format!(
                        "armv7a int Add cannot consume reserved cache register {} as rhs",
                        reg.0
                    )));
                }
            },
            MachineIntBinaryOp::Sub => match rhs {
                MachineValue::Imm64(imm) => {
                    if let Some((imm8, rot)) = enc::encode_arm_imm(*imm as u32) {
                        self.core
                            .text
                            .emit_u32(enc::sub_imm(dst_hw, lhs_hw, imm8, rot));
                    } else {
                        let s = self.gp_scratch.scoped_alloc();
                        emit_load_u32_into(&mut self.core.text, *s, *imm as u32);
                        self.core.text.emit_u32(enc::sub_reg(dst_hw, lhs_hw, *s));
                    }
                }
                MachineValue::Reg(r) => {
                    self.core
                        .text
                        .emit_u32(enc::sub_reg(dst_hw, lhs_hw, map_reg(*r)?));
                }
                MachineValue::ReservedReg(reg) => {
                    return Err(WasmError::internal(alloc::format!(
                        "armv7a int Sub cannot consume reserved cache register {} as rhs",
                        reg.0
                    )));
                }
            },
            MachineIntBinaryOp::Mul => {
                let rhs_gp = prepare_gp(&mut self.core.text, &self.gp_scratch, *rhs)?;
                self.core.text.emit_u32(enc::mul(dst_hw, lhs_hw, *rhs_gp));
                drop(rhs_gp);
            }
            MachineIntBinaryOp::And => match rhs {
                MachineValue::Reg(r) => {
                    self.core
                        .text
                        .emit_u32(enc::and_reg(dst_hw, lhs_hw, map_reg(*r)?));
                }
                MachineValue::Imm64(v) => {
                    if let Some((imm8, rot)) = enc::encode_arm_imm(*v as u32) {
                        self.core
                            .text
                            .emit_u32(enc::and_imm(dst_hw, lhs_hw, imm8, rot));
                    } else {
                        let s = self.gp_scratch.scoped_alloc();
                        emit_load_u32_into(&mut self.core.text, *s, *v as u32);
                        self.core.text.emit_u32(enc::and_reg(dst_hw, lhs_hw, *s));
                    }
                }
                MachineValue::ReservedReg(reg) => {
                    return Err(WasmError::internal(alloc::format!(
                        "armv7a int And cannot consume reserved cache register {} as rhs",
                        reg.0
                    )));
                }
            },
            MachineIntBinaryOp::Or => match rhs {
                MachineValue::Reg(r) => {
                    self.core
                        .text
                        .emit_u32(enc::orr_reg(dst_hw, lhs_hw, map_reg(*r)?));
                }
                MachineValue::Imm64(v) => {
                    if let Some((imm8, rot)) = enc::encode_arm_imm(*v as u32) {
                        self.core
                            .text
                            .emit_u32(enc::orr_imm(dst_hw, lhs_hw, imm8, rot));
                    } else {
                        let s = self.gp_scratch.scoped_alloc();
                        emit_load_u32_into(&mut self.core.text, *s, *v as u32);
                        self.core.text.emit_u32(enc::orr_reg(dst_hw, lhs_hw, *s));
                    }
                }
                MachineValue::ReservedReg(reg) => {
                    return Err(WasmError::internal(alloc::format!(
                        "armv7a int Or cannot consume reserved cache register {} as rhs",
                        reg.0
                    )));
                }
            },
            MachineIntBinaryOp::Xor => match rhs {
                MachineValue::Reg(r) => {
                    self.core
                        .text
                        .emit_u32(enc::eor_reg(dst_hw, lhs_hw, map_reg(*r)?));
                }
                MachineValue::Imm64(v) => {
                    if let Some((imm8, rot)) = enc::encode_arm_imm(*v as u32) {
                        self.core
                            .text
                            .emit_u32(enc::eor_imm(dst_hw, lhs_hw, imm8, rot));
                    } else {
                        let s = self.gp_scratch.scoped_alloc();
                        emit_load_u32_into(&mut self.core.text, *s, *v as u32);
                        self.core.text.emit_u32(enc::eor_reg(dst_hw, lhs_hw, *s));
                    }
                }
                MachineValue::ReservedReg(reg) => {
                    return Err(WasmError::internal(alloc::format!(
                        "armv7a int Xor cannot consume reserved cache register {} as rhs",
                        reg.0
                    )));
                }
            },
            MachineIntBinaryOp::Shl => {
                let rhs_hw = match rhs {
                    MachineValue::Imm64(v) => {
                        let shift = (*v as u32) & 31;
                        self.core.text.emit_u32(enc::lsl_imm(dst_hw, lhs_hw, shift));
                        return Ok(());
                    }
                    MachineValue::Reg(r) => map_reg(*r)?,
                    MachineValue::ReservedReg(reg) => {
                        return Err(WasmError::internal(alloc::format!(
                            "armv7a int Shl cannot consume reserved cache register {} as rhs",
                            reg.0
                        )));
                    }
                };
                // Mask shift amount to 5 bits (wasm i32 semantics)
                {
                    let s = self.gp_scratch.scoped_alloc();
                    self.core.text.emit_u32(enc::and_imm(*s, rhs_hw, 31, 0));
                    self.core.text.emit_u32(enc::lsl_reg(dst_hw, lhs_hw, *s));
                }
            }
            MachineIntBinaryOp::ShrU => {
                let rhs_hw = match rhs {
                    MachineValue::Imm64(v) => {
                        let shift = (*v as u32) & 31;
                        self.core.text.emit_u32(enc::lsr_imm(dst_hw, lhs_hw, shift));
                        return Ok(());
                    }
                    MachineValue::Reg(r) => map_reg(*r)?,
                    MachineValue::ReservedReg(reg) => {
                        return Err(WasmError::internal(alloc::format!(
                            "armv7a int ShrU cannot consume reserved cache register {} as rhs",
                            reg.0
                        )));
                    }
                };
                {
                    let s = self.gp_scratch.scoped_alloc();
                    self.core.text.emit_u32(enc::and_imm(*s, rhs_hw, 31, 0));
                    self.core.text.emit_u32(enc::lsr_reg(dst_hw, lhs_hw, *s));
                }
            }
            MachineIntBinaryOp::ShrS => {
                let rhs_hw = match rhs {
                    MachineValue::Imm64(v) => {
                        let shift = (*v as u32) & 31;
                        self.core.text.emit_u32(enc::asr_imm(dst_hw, lhs_hw, shift));
                        return Ok(());
                    }
                    MachineValue::Reg(r) => map_reg(*r)?,
                    MachineValue::ReservedReg(reg) => {
                        return Err(WasmError::internal(alloc::format!(
                            "armv7a int ShrS cannot consume reserved cache register {} as rhs",
                            reg.0
                        )));
                    }
                };
                {
                    let s = self.gp_scratch.scoped_alloc();
                    self.core.text.emit_u32(enc::and_imm(*s, rhs_hw, 31, 0));
                    self.core.text.emit_u32(enc::asr_reg(dst_hw, lhs_hw, *s));
                }
            }
            MachineIntBinaryOp::Rotl => {
                // rotl(x, k) = rotr(x, 32-k)
                let rhs_hw = match rhs {
                    MachineValue::Imm64(v) => {
                        let shift = (32 - ((*v as u32) & 31)) & 31;
                        self.core.text.emit_u32(enc::ror_imm(dst_hw, lhs_hw, shift));
                        return Ok(());
                    }
                    MachineValue::Reg(r) => map_reg(*r)?,
                    MachineValue::ReservedReg(reg) => {
                        return Err(WasmError::internal(alloc::format!(
                            "armv7a int Rotl cannot consume reserved cache register {} as rhs",
                            reg.0
                        )));
                    }
                };
                {
                    let s = self.gp_scratch.scoped_alloc();
                    self.core.text.emit_u32(enc::and_imm(*s, rhs_hw, 31, 0));
                    self.core.text.emit_u32(enc::rsb_imm(*s, *s, 32, 0));
                    self.core.text.emit_u32(enc::ror_reg(dst_hw, lhs_hw, *s));
                }
            }
            MachineIntBinaryOp::Rotr => {
                let rhs_hw = match rhs {
                    MachineValue::Imm64(v) => {
                        let shift = (*v as u32) & 31;
                        self.core.text.emit_u32(enc::ror_imm(dst_hw, lhs_hw, shift));
                        return Ok(());
                    }
                    MachineValue::Reg(r) => map_reg(*r)?,
                    MachineValue::ReservedReg(reg) => {
                        return Err(WasmError::internal(alloc::format!(
                            "armv7a int Rotr cannot consume reserved cache register {} as rhs",
                            reg.0
                        )));
                    }
                };
                {
                    let s = self.gp_scratch.scoped_alloc();
                    self.core.text.emit_u32(enc::and_imm(*s, rhs_hw, 31, 0));
                    self.core.text.emit_u32(enc::ror_reg(dst_hw, lhs_hw, *s));
                }
            }
            MachineIntBinaryOp::DivU => {
                self.spill_caller_saved_gp_regs();
                self.emit_values_to_regs_via_stack(&[Arm32Reg::R0, Arm32Reg::R1], &[lhs, rhs])?;
                self.core.text.emit_u32(enc::cmp_imm(Arm32Reg::R1, 0, 0));
                let ok = self.core.new_label();
                let trap_div_zero = self.core.new_label();
                let done = self.core.new_label();
                self.emit_branch(BranchFixupKind::BCond(Cond::Ne), ok);
                self.emit_branch(BranchFixupKind::B, trap_div_zero);
                // Call armv7a_udiv(num, den) -> quotient in R0
                self.core.bind_label(ok);
                self.emit_host_call(armv7a_udiv as usize);
                if dst_hw != Arm32Reg::R0 {
                    self.core.text.emit_u32(enc::mov_reg(dst_hw, Arm32Reg::R0));
                }
                self.restore_caller_saved_gp_regs(&[dst_hw]);
                self.emit_branch(BranchFixupKind::B, done);
                self.core.bind_label(trap_div_zero);
                self.restore_caller_saved_gp_regs(&[]);
                let trap_label = self
                    .core
                    .ensure_trap_label(MachineTrapKind::IntegerDivideByZero);
                self.emit_branch(BranchFixupKind::B, trap_label);
                self.core.bind_label(done);
            }
            MachineIntBinaryOp::DivS => {
                self.spill_caller_saved_gp_regs();
                self.emit_values_to_regs_via_stack(&[Arm32Reg::R0, Arm32Reg::R1], &[lhs, rhs])?;
                // Trap on divide by zero
                self.core.text.emit_u32(enc::cmp_imm(Arm32Reg::R1, 0, 0));
                let not_zero = self.core.new_label();
                let trap_div_zero = self.core.new_label();
                let trap_overflow = self.core.new_label();
                let after_traps = self.core.new_label();
                self.emit_branch(BranchFixupKind::BCond(Cond::Ne), not_zero);
                self.emit_branch(BranchFixupKind::B, trap_div_zero);
                self.core.bind_label(not_zero);
                // Trap on INT_MIN / -1 (integer overflow)
                {
                    let s = self.gp_scratch.scoped_alloc();
                    emit_load_u32_into(&mut self.core.text, *s, 0x80000000u32);
                    self.core.text.emit_u32(enc::cmp_reg(Arm32Reg::R0, *s));
                }
                let not_overflow = self.core.new_label();
                self.emit_branch(BranchFixupKind::BCond(Cond::Ne), not_overflow);
                self.core.text.emit_u32(enc::cmn_imm(Arm32Reg::R1, 1, 0)); // CMN rhs, #1 == CMP rhs, #-1
                let not_overflow2 = self.core.new_label();
                self.emit_branch(BranchFixupKind::BCond(Cond::Ne), not_overflow2);
                self.emit_branch(BranchFixupKind::B, trap_overflow);
                self.core.bind_label(not_overflow);
                self.core.bind_label(not_overflow2);
                self.emit_branch(BranchFixupKind::B, after_traps);
                self.core.bind_label(trap_div_zero);
                self.restore_caller_saved_gp_regs(&[]);
                let trap_label = self
                    .core
                    .ensure_trap_label(MachineTrapKind::IntegerDivideByZero);
                self.emit_branch(BranchFixupKind::B, trap_label);
                self.core.bind_label(trap_overflow);
                self.restore_caller_saved_gp_regs(&[]);
                let trap_label = self
                    .core
                    .ensure_trap_label(MachineTrapKind::IntegerOverflow);
                self.emit_branch(BranchFixupKind::B, trap_label);
                self.core.bind_label(after_traps);
                // Call armv7a_sdiv(num, den)
                self.emit_host_call(armv7a_sdiv as usize);
                if dst_hw != Arm32Reg::R0 {
                    self.core.text.emit_u32(enc::mov_reg(dst_hw, Arm32Reg::R0));
                }
                self.restore_caller_saved_gp_regs(&[dst_hw]);
            }
            MachineIntBinaryOp::RemU => {
                self.spill_caller_saved_gp_regs();
                self.emit_values_to_regs_via_stack(&[Arm32Reg::R0, Arm32Reg::R1], &[lhs, rhs])?;
                // Trap on divide by zero
                self.core.text.emit_u32(enc::cmp_imm(Arm32Reg::R1, 0, 0));
                let ok = self.core.new_label();
                let trap_div_zero = self.core.new_label();
                let done = self.core.new_label();
                self.emit_branch(BranchFixupKind::BCond(Cond::Ne), ok);
                self.emit_branch(BranchFixupKind::B, trap_div_zero);
                self.core.bind_label(ok);
                self.emit_stack_temp_alloc(8);
                self.core
                    .text
                    .emit_u32(enc::str_imm(Arm32Reg::R0, Arm32Reg::SP, 0));
                self.core
                    .text
                    .emit_u32(enc::str_imm(Arm32Reg::R1, Arm32Reg::SP, 4));
                // rem = lhs - (lhs / rhs) * rhs
                self.emit_host_call(armv7a_udiv as usize);
                // R0 = quotient. Restore lhs, rhs
                self.core
                    .text
                    .emit_u32(enc::ldr_imm(Arm32Reg::R2, Arm32Reg::SP, 0));
                self.core
                    .text
                    .emit_u32(enc::ldr_imm(Arm32Reg::R3, Arm32Reg::SP, 4));
                self.emit_stack_temp_free(8);
                {
                    let s = self.gp_scratch.scoped_alloc();
                    self.core
                        .text
                        .emit_u32(enc::mul(*s, Arm32Reg::R0, Arm32Reg::R3));
                    self.core
                        .text
                        .emit_u32(enc::sub_reg(dst_hw, Arm32Reg::R2, *s));
                }
                self.restore_caller_saved_gp_regs(&[dst_hw]);
                self.emit_branch(BranchFixupKind::B, done);
                self.core.bind_label(trap_div_zero);
                self.restore_caller_saved_gp_regs(&[]);
                let trap_label = self
                    .core
                    .ensure_trap_label(MachineTrapKind::IntegerDivideByZero);
                self.emit_branch(BranchFixupKind::B, trap_label);
                self.core.bind_label(done);
            }
            MachineIntBinaryOp::RemS => {
                self.spill_caller_saved_gp_regs();
                self.emit_values_to_regs_via_stack(&[Arm32Reg::R0, Arm32Reg::R1], &[lhs, rhs])?;
                // Trap on divide by zero
                self.core.text.emit_u32(enc::cmp_imm(Arm32Reg::R1, 0, 0));
                let ok = self.core.new_label();
                let trap_div_zero = self.core.new_label();
                let done = self.core.new_label();
                self.emit_branch(BranchFixupKind::BCond(Cond::Ne), ok);
                self.emit_branch(BranchFixupKind::B, trap_div_zero);
                self.core.bind_label(ok);
                // INT_MIN % -1 == 0 in wasm (no trap, just returns 0)
                // rem = num - (num / den) * den — this naturally gives 0
                // Save lhs and rhs, call sdiv, compute remainder
                self.emit_stack_temp_alloc(8);
                self.core
                    .text
                    .emit_u32(enc::str_imm(Arm32Reg::R0, Arm32Reg::SP, 0));
                self.core
                    .text
                    .emit_u32(enc::str_imm(Arm32Reg::R1, Arm32Reg::SP, 4));
                self.emit_host_call(armv7a_sdiv as usize);
                // R0 = quotient. Restore lhs, rhs
                self.core
                    .text
                    .emit_u32(enc::ldr_imm(Arm32Reg::R2, Arm32Reg::SP, 0));
                self.core
                    .text
                    .emit_u32(enc::ldr_imm(Arm32Reg::R3, Arm32Reg::SP, 4));
                self.emit_stack_temp_free(8);
                // rem = lhs - quotient * rhs: MLS dst, R0, R3, R2
                {
                    let s = self.gp_scratch.scoped_alloc();
                    self.core
                        .text
                        .emit_u32(enc::mul(*s, Arm32Reg::R0, Arm32Reg::R3));
                    self.core
                        .text
                        .emit_u32(enc::sub_reg(dst_hw, Arm32Reg::R2, *s));
                }
                self.restore_caller_saved_gp_regs(&[dst_hw]);
                self.emit_branch(BranchFixupKind::B, done);
                self.core.bind_label(trap_div_zero);
                self.restore_caller_saved_gp_regs(&[]);
                let trap_label = self
                    .core
                    .ensure_trap_label(MachineTrapKind::IntegerDivideByZero);
                self.emit_branch(BranchFixupKind::B, trap_label);
                self.core.bind_label(done);
            }
        }

        // For I32 width, mask to 32 bits (already natural on ARM32)
        // For I64 width, we only handle low 32 bits currently
        Ok(())
    }

    fn compile_int_unary(
        &mut self,
        _width: MachineIntWidth,
        op: MachineIntUnaryOp,
        dst: MachineReg,
        src: &MachineValue,
    ) -> Result<(), WasmError> {
        let dst_hw = map_reg(dst)?;
        let src_hw = match src {
            MachineValue::Reg(r) => map_reg(*r)?,
            MachineValue::ReservedReg(reg) => {
                return Err(WasmError::internal(alloc::format!(
                    "armv7a compile_int_unary cannot consume reserved cache register {} as src",
                    reg.0
                )));
            }
            MachineValue::Imm64(v) => {
                self.emit_load_u32(dst_hw, *v as u32);
                dst_hw
            }
        };

        match op {
            MachineIntUnaryOp::Clz => {
                self.core.text.emit_u32(enc::clz(dst_hw, src_hw));
            }
            MachineIntUnaryOp::Ctz => {
                // ctz(x) = 31 - clz(x & -x) when x != 0, else 32
                // RBIT + CLZ on ARMv7
                // Actually ARMv7 has RBIT: reverse bits, then CLZ
                self.core.text.emit_u32(select::rbit(dst_hw, src_hw));
                self.core.text.emit_u32(enc::clz(dst_hw, dst_hw));
            }
            MachineIntUnaryOp::Popcnt => {
                let s0 = self.gp_scratch.scoped_alloc();
                let s1 = self.gp_scratch.scoped_alloc();
                // Hamming weight using parallel bit counting
                // x = x - ((x >> 1) & 0x55555555)
                self.core.text.emit_u32(enc::lsr_imm(*s0, src_hw, 1));
                emit_load_u32_into(&mut self.core.text, *s1, 0x55555555);
                self.core.text.emit_u32(enc::and_reg(*s0, *s0, *s1));
                self.core.text.emit_u32(enc::sub_reg(dst_hw, src_hw, *s0));
                // x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
                emit_load_u32_into(&mut self.core.text, *s1, 0x33333333);
                self.core.text.emit_u32(enc::lsr_imm(*s0, dst_hw, 2));
                self.core.text.emit_u32(enc::and_reg(*s0, *s0, *s1));
                self.core.text.emit_u32(enc::and_reg(dst_hw, dst_hw, *s1));
                self.core.text.emit_u32(enc::add_reg(dst_hw, dst_hw, *s0));
                // x = (x + (x >> 4)) & 0x0F0F0F0F
                self.core.text.emit_u32(enc::lsr_imm(*s0, dst_hw, 4));
                self.core.text.emit_u32(enc::add_reg(dst_hw, dst_hw, *s0));
                emit_load_u32_into(&mut self.core.text, *s1, 0x0F0F0F0F);
                self.core.text.emit_u32(enc::and_reg(dst_hw, dst_hw, *s1));
                // x = x * 0x01010101 >> 24
                emit_load_u32_into(&mut self.core.text, *s1, 0x01010101);
                self.core.text.emit_u32(enc::mul(dst_hw, dst_hw, *s1));
                self.core.text.emit_u32(enc::lsr_imm(dst_hw, dst_hw, 24));
            }
            MachineIntUnaryOp::Extend8S => {
                self.core.text.emit_u32(enc::sxtb(dst_hw, src_hw));
            }
            MachineIntUnaryOp::Extend16S => {
                self.core.text.emit_u32(enc::sxth(dst_hw, src_hw));
            }
            MachineIntUnaryOp::Extend32S => {
                // On 32-bit, this is a no-op (value is already 32 bits)
                if dst_hw != src_hw {
                    self.core.text.emit_u32(enc::mov_reg(dst_hw, src_hw));
                }
            }
        }
        Ok(())
    }

    // ─── I64 pair binary ────────────────────────────────────────────────────────

    fn compile_int64_pair_binary(
        &mut self,
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
                self.spill_caller_saved_gp_regs();
                self.emit_quad_args_to_r0_r3(lhs_lo, lhs_hi, rhs_lo, rhs_hi)?;
                self.core
                    .text
                    .emit_u32(enc::adds_reg(Arm32Reg::R0, Arm32Reg::R0, Arm32Reg::R2));
                self.core
                    .text
                    .emit_u32(enc::adc_reg(Arm32Reg::R1, Arm32Reg::R1, Arm32Reg::R3));
                self.emit_pair_results_from_r0_r1(dst_lo, dst_hi)?;
                self.restore_caller_saved_gp_regs(&[dst_lo_hw, dst_hi_hw]);
                Ok(())
            }
            MachineIntBinaryOp::Sub => {
                let dst_lo_hw = map_reg(dst_lo)?;
                let dst_hi_hw = map_reg(dst_hi)?;
                self.spill_caller_saved_gp_regs();
                self.emit_quad_args_to_r0_r3(lhs_lo, lhs_hi, rhs_lo, rhs_hi)?;
                self.core
                    .text
                    .emit_u32(enc::subs_reg(Arm32Reg::R0, Arm32Reg::R0, Arm32Reg::R2));
                self.core
                    .text
                    .emit_u32(enc::sbc_reg(Arm32Reg::R1, Arm32Reg::R1, Arm32Reg::R3));
                self.emit_pair_results_from_r0_r1(dst_lo, dst_hi)?;
                self.restore_caller_saved_gp_regs(&[dst_lo_hw, dst_hi_hw]);
                Ok(())
            }
            MachineIntBinaryOp::Mul => {
                let dst_lo_hw = map_reg(dst_lo)?;
                let dst_hi_hw = map_reg(dst_hi)?;
                self.spill_caller_saved_gp_regs();
                self.emit_quad_args_to_r0_r3(lhs_lo, lhs_hi, rhs_lo, rhs_hi)?;
                self.emit_host_call(armv7a_i64_mul as usize);
                self.emit_pair_results_from_r0_r1(dst_lo, dst_hi)?;
                self.restore_caller_saved_gp_regs(&[dst_lo_hw, dst_hi_hw]);
                Ok(())
            }
            MachineIntBinaryOp::And | MachineIntBinaryOp::Or | MachineIntBinaryOp::Xor => {
                let dst_lo_hw = map_reg(dst_lo)?;
                let dst_hi_hw = map_reg(dst_hi)?;
                // Snapshot the rhs halves into owned scratch *before* writing
                // the lhs into the destination pair, so a rhs that aliases
                // either destination half survives the materialize. Lua's
                // SWAR string hash exposes this pattern via i64 And/Or/Xor.
                let rhs_lo_gp = prepare_pair_bitop_rhs(
                    &mut self.core.text,
                    &self.gp_scratch,
                    *rhs_lo,
                    dst_lo_hw,
                    dst_hi_hw,
                )?;
                let rhs_hi_gp = prepare_pair_bitop_rhs(
                    &mut self.core.text,
                    &self.gp_scratch,
                    *rhs_hi,
                    dst_lo_hw,
                    dst_hi_hw,
                )?;
                self.materialize_gp_into(dst_lo_hw, lhs_lo)?;
                self.materialize_gp_into(dst_hi_hw, lhs_hi)?;
                let emit = match op {
                    MachineIntBinaryOp::And => enc::and_reg,
                    MachineIntBinaryOp::Or => enc::orr_reg,
                    MachineIntBinaryOp::Xor => enc::eor_reg,
                    _ => unreachable!(),
                };
                self.core
                    .text
                    .emit_u32(emit(dst_lo_hw, dst_lo_hw, *rhs_lo_gp));
                self.core
                    .text
                    .emit_u32(emit(dst_hi_hw, dst_hi_hw, *rhs_hi_gp));
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
        &mut self,
        op: MachineIntUnaryOp,
        dst_lo: MachineReg,
        dst_hi: MachineReg,
        src_lo: &MachineValue,
        src_hi: &MachineValue,
    ) -> Result<(), WasmError> {
        let dst_lo_hw = map_reg(dst_lo)?;
        let dst_hi_hw = map_reg(dst_hi)?;
        self.spill_caller_saved_gp_regs();
        match op {
            MachineIntUnaryOp::Clz | MachineIntUnaryOp::Ctz | MachineIntUnaryOp::Popcnt => {
                self.emit_pair_args_to_r0_r1(src_lo, src_hi)?;
                self.emit_host_call(match op {
                    MachineIntUnaryOp::Clz => armv7a_i64_clz as usize,
                    MachineIntUnaryOp::Ctz => armv7a_i64_ctz as usize,
                    MachineIntUnaryOp::Popcnt => armv7a_i64_popcnt as usize,
                    _ => unreachable!(),
                });
                self.emit_pair_results_from_r0_r1(dst_lo, dst_hi)?;
                self.restore_caller_saved_gp_regs(&[dst_lo_hw, dst_hi_hw]);
                Ok(())
            }
            MachineIntUnaryOp::Extend8S => {
                let src_lo_hw =
                    prepare_gp(&mut self.core.text, &self.gp_scratch, *src_lo)?.detach();
                self.core.text.emit_u32(enc::sxtb(dst_lo_hw, *src_lo_hw));
                self.core
                    .text
                    .emit_u32(enc::asr_imm(dst_hi_hw, dst_lo_hw, 31));
                self.restore_caller_saved_gp_regs(&[dst_lo_hw, dst_hi_hw]);
                Ok(())
            }
            MachineIntUnaryOp::Extend16S => {
                let src_lo_hw =
                    prepare_gp(&mut self.core.text, &self.gp_scratch, *src_lo)?.detach();
                self.core.text.emit_u32(enc::sxth(dst_lo_hw, *src_lo_hw));
                self.core
                    .text
                    .emit_u32(enc::asr_imm(dst_hi_hw, dst_lo_hw, 31));
                self.restore_caller_saved_gp_regs(&[dst_lo_hw, dst_hi_hw]);
                Ok(())
            }
            MachineIntUnaryOp::Extend32S => {
                let src_lo_hw =
                    prepare_gp(&mut self.core.text, &self.gp_scratch, *src_lo)?.detach();
                if dst_lo_hw != *src_lo_hw {
                    self.core.text.emit_u32(enc::mov_reg(dst_lo_hw, *src_lo_hw));
                }
                self.core
                    .text
                    .emit_u32(enc::asr_imm(dst_hi_hw, dst_lo_hw, 31));
                self.restore_caller_saved_gp_regs(&[dst_lo_hw, dst_hi_hw]);
                Ok(())
            }
        }
    }

    // ─── I64 pair div/rem ───────────────────────────────────────────────────────

    fn compile_int64_pair_div_rem(
        &mut self,
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
        self.spill_caller_saved_gp_regs();
        let trap_div_zero = self.core.new_label();
        let trap_overflow = self.core.new_label();
        let after_traps = self.core.new_label();
        self.emit_quad_args_to_r0_r3(lhs_lo, lhs_hi, rhs_lo, rhs_hi)?;

        {
            let s = self.gp_scratch.scoped_alloc();
            self.core
                .text
                .emit_u32(enc::orr_reg(*s, Arm32Reg::R2, Arm32Reg::R3));
            self.core.text.emit_u32(enc::cmp_imm(*s, 0, 0));
        }
        let non_zero = self.core.new_label();
        self.emit_branch(BranchFixupKind::BCond(Cond::Ne), non_zero);
        self.emit_branch(BranchFixupKind::B, trap_div_zero);
        self.core.bind_label(non_zero);

        if matches!(sign, MachineSign::Signed) && !rem {
            {
                let s = self.gp_scratch.scoped_alloc();
                emit_load_u32_into(&mut self.core.text, *s, 0x8000_0000);
                self.core.text.emit_u32(enc::cmp_reg(Arm32Reg::R1, *s));
            }
            let no_overflow = self.core.new_label();
            self.emit_branch(BranchFixupKind::BCond(Cond::Ne), no_overflow);
            self.core.text.emit_u32(enc::cmp_imm(Arm32Reg::R0, 0, 0));
            let no_overflow_lo = self.core.new_label();
            self.emit_branch(BranchFixupKind::BCond(Cond::Ne), no_overflow_lo);
            {
                let s = self.gp_scratch.scoped_alloc();
                emit_load_u32_into(&mut self.core.text, *s, u32::MAX);
                self.core.text.emit_u32(enc::cmp_reg(Arm32Reg::R2, *s));
            }
            let no_overflow_rhs_lo = self.core.new_label();
            self.emit_branch(BranchFixupKind::BCond(Cond::Ne), no_overflow_rhs_lo);
            {
                let s = self.gp_scratch.scoped_alloc();
                emit_load_u32_into(&mut self.core.text, *s, u32::MAX);
                self.core.text.emit_u32(enc::cmp_reg(Arm32Reg::R3, *s));
            }
            let no_overflow_rhs_hi = self.core.new_label();
            self.emit_branch(BranchFixupKind::BCond(Cond::Ne), no_overflow_rhs_hi);
            self.emit_branch(BranchFixupKind::B, trap_overflow);
            self.core.bind_label(no_overflow);
            self.core.bind_label(no_overflow_lo);
            self.core.bind_label(no_overflow_rhs_lo);
            self.core.bind_label(no_overflow_rhs_hi);
        }
        self.emit_branch(BranchFixupKind::B, after_traps);

        self.core.bind_label(trap_div_zero);
        self.restore_caller_saved_gp_regs(&[]);
        let trap_label = self
            .core
            .ensure_trap_label(MachineTrapKind::IntegerDivideByZero);
        self.emit_branch(BranchFixupKind::B, trap_label);

        self.core.bind_label(trap_overflow);
        self.restore_caller_saved_gp_regs(&[]);
        let trap_label = self
            .core
            .ensure_trap_label(MachineTrapKind::IntegerOverflow);
        self.emit_branch(BranchFixupKind::B, trap_label);

        self.core.bind_label(after_traps);

        self.emit_host_call(match (sign, rem) {
            (MachineSign::Signed, false) => armv7a_i64_div_s as usize,
            (MachineSign::Unsigned, false) => armv7a_i64_div_u as usize,
            (MachineSign::Signed, true) => armv7a_i64_rem_s as usize,
            (MachineSign::Unsigned, true) => armv7a_i64_rem_u as usize,
        });
        self.emit_pair_results_from_r0_r1(dst_lo, dst_hi)?;
        self.restore_caller_saved_gp_regs(&[dst_lo_hw, dst_hi_hw]);
        Ok(())
    }

    // ─── I64 pair shift ─────────────────────────────────────────────────────────

    fn compile_int64_pair_shift(
        &mut self,
        op: MachineIntBinaryOp,
        dst_lo: MachineReg,
        dst_hi: MachineReg,
        lhs_lo: &MachineValue,
        lhs_hi: &MachineValue,
        rhs: &MachineValue,
    ) -> Result<(), WasmError> {
        let dst_lo_hw = map_reg(dst_lo)?;
        let dst_hi_hw = map_reg(dst_hi)?;
        self.spill_caller_saved_gp_regs();
        self.emit_move_gp_value(Arm32Reg::R2, rhs)?;
        self.emit_pair_args_to_r0_r1(lhs_lo, lhs_hi)?;
        self.emit_host_call(match op {
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
        });
        self.emit_pair_results_from_r0_r1(dst_lo, dst_hi)?;
        self.restore_caller_saved_gp_regs(&[dst_lo_hw, dst_hi_hw]);
        Ok(())
    }

    // ─── I64 pair compare ───────────────────────────────────────────────────────

    fn compile_int64_pair_compare(
        &mut self,
        kind: MachineCompareKind,
        sign: MachineSign,
        dst: MachineReg,
        lhs_lo: &MachineValue,
        lhs_hi: &MachineValue,
        rhs_lo: &MachineValue,
        rhs_hi: &MachineValue,
    ) -> Result<(), WasmError> {
        let dst_hw = map_reg(dst)?;
        let set_true = self.core.new_label();
        let set_false = self.core.new_label();
        let done = self.core.new_label();

        let hi_lt = match sign {
            MachineSign::Signed => Cond::Lt,
            MachineSign::Unsigned => Cond::Cc,
        };
        let hi_gt = match sign {
            MachineSign::Signed => Cond::Gt,
            MachineSign::Unsigned => Cond::Hi,
        };

        // Inline CMP without spill/restore. We use pool-managed GP scratches
        // (R12, R14/LR — saved in prologue) as temporaries, so no live GP
        // dynamic state is clobbered.
        //
        // Compare hi words: prepare into scratch registers and CMP.
        let lhs_hi_gp = prepare_gp(&mut self.core.text, &self.gp_scratch, *lhs_hi)?;
        let rhs_hi_gp = prepare_gp(&mut self.core.text, &self.gp_scratch, *rhs_hi)?;
        self.core
            .text
            .emit_u32(enc::cmp_reg(*lhs_hi_gp, *rhs_hi_gp));
        drop(lhs_hi_gp);
        drop(rhs_hi_gp);

        match kind {
            MachineCompareKind::Eq => {
                self.emit_branch(BranchFixupKind::BCond(Cond::Ne), set_false);
            }
            MachineCompareKind::Ne => {
                self.emit_branch(BranchFixupKind::BCond(Cond::Ne), set_true);
            }
            MachineCompareKind::Lt | MachineCompareKind::Le => {
                self.emit_branch(BranchFixupKind::BCond(hi_lt), set_true);
                self.emit_branch(BranchFixupKind::BCond(hi_gt), set_false);
            }
            MachineCompareKind::Gt | MachineCompareKind::Ge => {
                self.emit_branch(BranchFixupKind::BCond(hi_gt), set_true);
                self.emit_branch(BranchFixupKind::BCond(hi_lt), set_false);
            }
        }

        // Hi words equal — compare lo words.
        let lhs_lo_gp = prepare_gp(&mut self.core.text, &self.gp_scratch, *lhs_lo)?;
        let rhs_lo_gp = prepare_gp(&mut self.core.text, &self.gp_scratch, *rhs_lo)?;
        self.core
            .text
            .emit_u32(enc::cmp_reg(*lhs_lo_gp, *rhs_lo_gp));
        drop(lhs_lo_gp);
        drop(rhs_lo_gp);

        match kind {
            MachineCompareKind::Eq => self.emit_branch(BranchFixupKind::BCond(Cond::Eq), set_true),
            MachineCompareKind::Ne => self.emit_branch(BranchFixupKind::BCond(Cond::Ne), set_true),
            MachineCompareKind::Lt => self.emit_branch(BranchFixupKind::BCond(Cond::Cc), set_true),
            MachineCompareKind::Le => self.emit_branch(BranchFixupKind::BCond(Cond::Ls), set_true),
            MachineCompareKind::Gt => self.emit_branch(BranchFixupKind::BCond(Cond::Hi), set_true),
            MachineCompareKind::Ge => self.emit_branch(BranchFixupKind::BCond(Cond::Cs), set_true),
        }

        self.emit_branch(BranchFixupKind::B, set_false);
        self.core.bind_label(set_true);
        self.emit_set_bool_immediate(dst_hw, true);
        self.emit_branch(BranchFixupKind::B, done);
        self.core.bind_label(set_false);
        self.emit_set_bool_immediate(dst_hw, false);
        self.core.bind_label(done);
        Ok(())
    }

    // ─── I64 pair → float conversion ────────────────────────────────────────────

    fn compile_convert_i64_pair_to_float(
        &mut self,
        width: MachineFloatWidth,
        sign: MachineSign,
        dst: MachineReg,
        src_lo: &MachineValue,
        src_hi: &MachineValue,
    ) -> Result<(), WasmError> {
        self.spill_caller_saved_gp_regs();
        self.emit_pair_args_to_r0_r1(src_lo, src_hi)?;
        self.emit_host_call(match (width, sign) {
            (MachineFloatWidth::F32, MachineSign::Signed) => armv7a_i64s_to_f32 as usize,
            (MachineFloatWidth::F32, MachineSign::Unsigned) => armv7a_i64u_to_f32 as usize,
            (MachineFloatWidth::F64, MachineSign::Signed) => armv7a_i64s_to_f64 as usize,
            (MachineFloatWidth::F64, MachineSign::Unsigned) => armv7a_i64u_to_f64 as usize,
        });

        match width {
            MachineFloatWidth::F32 => {
                let dst_s = self.map_fp_dreg(dst)? * 2;
                let s0 = FP_SCRATCH0 * 2;
                if dst_s != s0 {
                    self.core.text.emit_u32(enc::vmov_s(dst_s, s0));
                }
            }
            MachineFloatWidth::F64 => {
                let dst_d = self.map_fp_dreg(dst)?;
                if dst_d != FP_SCRATCH0 {
                    self.core.text.emit_u32(enc::vmov_d(dst_d, FP_SCRATCH0));
                }
            }
        }
        self.restore_caller_saved_gp_regs(&[]);
        Ok(())
    }

    // ─── Float → I64 pair conversion ────────────────────────────────────────────

    fn compile_convert_float_to_i64_pair(
        &mut self,
        op: MachineConvertOp,
        dst_lo: MachineReg,
        dst_hi: MachineReg,
        src: &MachineValue,
    ) -> Result<(), WasmError> {
        let dst_lo_hw = map_reg(dst_lo)?;
        let dst_hi_hw = map_reg(dst_hi)?;
        let src_is_f32 = matches!(
            op,
            MachineConvertOp::I64TruncF32S
                | MachineConvertOp::I64TruncF32U
                | MachineConvertOp::I64TruncSatF32S
                | MachineConvertOp::I64TruncSatF32U
        );
        let src_width = if src_is_f32 {
            MachineFloatWidth::F32
        } else {
            MachineFloatWidth::F64
        };
        let src_d = prepare_fp(
            &mut self.core.text,
            &self.gp_scratch,
            &self.fp_scratch,
            src_width,
            *src,
        )?
        .detach();
        self.spill_caller_saved_gp_regs();

        if src_is_f32 {
            let src_s = *src_d * 2;
            let s0 = FP_SCRATCH0 * 2;
            if src_s != s0 {
                self.core.text.emit_u32(enc::vmov_s(s0, src_s));
            }
            self.core.text.emit_u32(enc::vmov_r_s(Arm32Reg::R0, s0));
            self.emit_load_u32(Arm32Reg::R1, 0);
        } else {
            if *src_d != FP_SCRATCH0 {
                self.core.text.emit_u32(enc::vmov_d(FP_SCRATCH0, *src_d));
            }
            self.core
                .text
                .emit_u32(enc::vmov_rr_d(Arm32Reg::R0, Arm32Reg::R1, FP_SCRATCH0));
        }
        self.emit_load_u32(Arm32Reg::R2, select::convert_op_code(op));

        if matches!(
            op,
            MachineConvertOp::I64TruncSatF32S
                | MachineConvertOp::I64TruncSatF32U
                | MachineConvertOp::I64TruncSatF64S
                | MachineConvertOp::I64TruncSatF64U
        ) {
            self.emit_host_call(armv7a_saturating_trunc as usize);
            self.emit_pair_results_from_r0_r1(dst_lo, dst_hi)?;
            self.restore_caller_saved_gp_regs(&[dst_lo_hw, dst_hi_hw]);
            return Ok(());
        }

        self.emit_trunc_result_buffer_alloc();
        self.core
            .text
            .emit_u32(enc::mov_reg(Arm32Reg::R3, map_fixed_reg(MACHINE_CTX_REG)));
        {
            let s = self.gp_scratch.scoped_alloc();
            self.core
                .text
                .emit_u32(enc::add_imm(*s, Arm32Reg::SP, 8, 0));
            self.core.text.emit_u32(enc::str_imm(*s, Arm32Reg::SP, 0));
        }
        self.emit_host_call(armv7a_trapping_trunc as usize);
        self.core.text.emit_u32(enc::cmp_imm(Arm32Reg::R0, 0, 0));
        let ok = self.core.new_label();
        self.emit_branch(BranchFixupKind::BCond(Cond::Eq), ok);
        self.emit_trunc_result_buffer_free();
        self.restore_caller_saved_gp_regs(&[]);
        self.emit_load_u32(Arm32Reg::R0, 1);
        // R0 already holds NativeCallStatus::Error (= 1) per the trapping
        // helper's failure contract. Branch to the body-local error tail
        // which preserves R0 and propagates the trap upward.
        let body_local_error = self.core.body_local_error_label;
        self.emit_branch(BranchFixupKind::B, body_local_error);
        self.core.bind_label(ok);
        self.core
            .text
            .emit_u32(enc::ldr_imm(Arm32Reg::R0, Arm32Reg::SP, 8));
        self.core
            .text
            .emit_u32(enc::ldr_imm(Arm32Reg::R1, Arm32Reg::SP, 12));
        self.emit_trunc_result_buffer_free();
        self.emit_pair_results_from_r0_r1(dst_lo, dst_hi)?;
        self.restore_caller_saved_gp_regs(&[dst_lo_hw, dst_hi_hw]);
        Ok(())
    }

    // ─── Float → I32 conversion ────────────────────────────────────────────────

    fn compile_convert_float_to_i32(
        &mut self,
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
        let src_width = if src_is_f32 {
            MachineFloatWidth::F32
        } else {
            MachineFloatWidth::F64
        };
        let src_d = prepare_fp(
            &mut self.core.text,
            &self.gp_scratch,
            &self.fp_scratch,
            src_width,
            *src,
        )?
        .detach();
        let dst_hw = map_reg(dst)?;
        self.spill_caller_saved_gp_regs();

        if src_is_f32 {
            let src_s = *src_d * 2;
            let s0 = FP_SCRATCH0 * 2;
            if src_s != s0 {
                self.core.text.emit_u32(enc::vmov_s(s0, src_s));
            }
            self.core.text.emit_u32(enc::vmov_r_s(Arm32Reg::R0, s0));
            self.emit_load_u32(Arm32Reg::R1, 0);
        } else {
            if *src_d != FP_SCRATCH0 {
                self.core.text.emit_u32(enc::vmov_d(FP_SCRATCH0, *src_d));
            }
            self.core
                .text
                .emit_u32(enc::vmov_rr_d(Arm32Reg::R0, Arm32Reg::R1, FP_SCRATCH0));
        }
        self.emit_load_u32(Arm32Reg::R2, select::convert_op_code(op));

        if matches!(
            op,
            MachineConvertOp::I32TruncSatF32S
                | MachineConvertOp::I32TruncSatF32U
                | MachineConvertOp::I32TruncSatF64S
                | MachineConvertOp::I32TruncSatF64U
        ) {
            self.emit_host_call(armv7a_saturating_trunc as usize);
            if dst_hw != Arm32Reg::R0 {
                self.core.text.emit_u32(enc::mov_reg(dst_hw, Arm32Reg::R0));
            }
            self.restore_caller_saved_gp_regs(&[dst_hw]);
            return Ok(());
        }

        self.emit_trunc_result_buffer_alloc();
        self.core
            .text
            .emit_u32(enc::mov_reg(Arm32Reg::R3, map_fixed_reg(MACHINE_CTX_REG)));
        {
            let s = self.gp_scratch.scoped_alloc();
            self.core
                .text
                .emit_u32(enc::add_imm(*s, Arm32Reg::SP, 8, 0));
            self.core.text.emit_u32(enc::str_imm(*s, Arm32Reg::SP, 0));
        }
        self.emit_host_call(armv7a_trapping_trunc as usize);
        self.core.text.emit_u32(enc::cmp_imm(Arm32Reg::R0, 0, 0));
        let ok = self.core.new_label();
        self.emit_branch(BranchFixupKind::BCond(Cond::Eq), ok);
        self.emit_trunc_result_buffer_free();
        self.restore_caller_saved_gp_regs(&[]);
        self.emit_load_u32(Arm32Reg::R0, 1);
        // R0 already holds NativeCallStatus::Error (= 1) per the trapping
        // helper's failure contract. Branch to the body-local error tail
        // which preserves R0 and propagates the trap upward.
        let body_local_error = self.core.body_local_error_label;
        self.emit_branch(BranchFixupKind::B, body_local_error);
        self.core.bind_label(ok);
        self.core
            .text
            .emit_u32(enc::ldr_imm(Arm32Reg::R0, Arm32Reg::SP, 8));
        self.emit_trunc_result_buffer_free();
        if dst_hw != Arm32Reg::R0 {
            self.core.text.emit_u32(enc::mov_reg(dst_hw, Arm32Reg::R0));
        }
        self.restore_caller_saved_gp_regs(&[dst_hw]);
        Ok(())
    }

    // ─── Reinterpret F64 ↔ I64 pair ────────────────────────────────────────────

    fn compile_reinterpret_f64_to_i64_pair(
        &mut self,
        dst_lo: MachineReg,
        dst_hi: MachineReg,
        src: &MachineValue,
    ) -> Result<(), WasmError> {
        match src {
            MachineValue::Reg(reg) => {
                let dm = self.map_fp_dreg(*reg)?;
                let dst_lo_hw = map_reg(dst_lo)?;
                let dst_hi_hw = map_reg(dst_hi)?;
                self.core
                    .text
                    .emit_u32(enc::vmov_rr_d(dst_lo_hw, dst_hi_hw, dm));
            }
            MachineValue::ReservedReg(reg) => {
                return Err(WasmError::internal(alloc::format!(
                    "armv7a reinterpret_f64_to_i64_pair cannot consume reserved cache register {} as src",
                    reg.0
                )));
            }
            MachineValue::Imm64(bits) => {
                self.emit_move_gp_value(
                    map_reg(dst_lo)?,
                    &MachineValue::Imm64(u64::from(*bits as u32)),
                )?;
                self.emit_move_gp_value(
                    map_reg(dst_hi)?,
                    &MachineValue::Imm64(u64::from((*bits >> 32) as u32)),
                )?;
            }
        }
        Ok(())
    }

    fn compile_reinterpret_i64_pair_to_f64(
        &mut self,
        dst: MachineReg,
        src_lo: &MachineValue,
        src_hi: &MachineValue,
    ) -> Result<(), WasmError> {
        let dd = self.map_fp_dreg(dst)?;
        self.spill_caller_saved_gp_regs();
        self.emit_pair_args_to_r0_r1(src_lo, src_hi)?;
        self.core
            .text
            .emit_u32(enc::vmov_d_rr(dd, Arm32Reg::R0, Arm32Reg::R1));
        self.restore_caller_saved_gp_regs(&[]);
        Ok(())
    }

    // ─── Float ALU ──────────────────────────────────────────────────────────────

    fn compile_float_binary(
        &mut self,
        width: MachineFloatWidth,
        op: MachineFloatBinaryOp,
        dst: MachineReg,
        lhs: &MachineValue,
        rhs: &MachineValue,
    ) -> Result<(), WasmError> {
        let dn = prepare_fp(
            &mut self.core.text,
            &self.gp_scratch,
            &self.fp_scratch,
            width,
            *lhs,
        )?
        .detach();
        let dm = prepare_fp(
            &mut self.core.text,
            &self.gp_scratch,
            &self.fp_scratch,
            width,
            *rhs,
        )?
        .detach();

        let dd = self.map_fp_dreg(dst)?;

        match (width, op) {
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Add) => {
                self.core.text.emit_u32(enc::vadd_d(dd, *dn, *dm));
            }
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Sub) => {
                self.core.text.emit_u32(enc::vsub_d(dd, *dn, *dm));
            }
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Mul) => {
                self.core.text.emit_u32(enc::vmul_d(dd, *dn, *dm));
            }
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Div) => {
                self.core.text.emit_u32(enc::vdiv_d(dd, *dn, *dm));
            }
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Add) => {
                self.core
                    .text
                    .emit_u32(enc::vadd_s(dd * 2, *dn * 2, *dm * 2));
            }
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Sub) => {
                self.core
                    .text
                    .emit_u32(enc::vsub_s(dd * 2, *dn * 2, *dm * 2));
            }
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Mul) => {
                self.core
                    .text
                    .emit_u32(enc::vmul_s(dd * 2, *dn * 2, *dm * 2));
            }
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Div) => {
                self.core
                    .text
                    .emit_u32(enc::vdiv_s(dd * 2, *dn * 2, *dm * 2));
            }

            // Min/Max: compare, handle NaN, select. The destination may
            // alias `dn` due to dead-input reuse, so we compare *before*
            // any move into `dd` and never read from a clobbered register.
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Min) => {
                self.compile_float_min_max_d(dd, *dn, *dm, /* is_min */ true)?;
            }
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Max) => {
                self.compile_float_min_max_d(dd, *dn, *dm, /* is_min */ false)?;
            }
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Min) => {
                self.compile_float_min_max_s(dd, *dn, *dm, /* is_min */ true)?;
            }
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Max) => {
                self.compile_float_min_max_s(dd, *dn, *dm, /* is_min */ false)?;
            }

            // Copysign: take magnitude from lhs, sign from rhs
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Copysign) => {
                // Copysign uses R0..R3 as scratch GP regs to hold the
                // lo/hi halves of both operands while we mask the sign
                // bits. Spill the live JIT values in R0-R3, R9 first and
                // restore after the result is written into `dd`.
                let (sign_imm8, sign_rot) = enc::encode_arm_imm(0x8000_0000).unwrap();
                self.spill_caller_saved_gp_regs();
                // Extract rhs bits into R0/R1.
                self.core
                    .text
                    .emit_u32(enc::vmov_rr_d(Arm32Reg::R0, Arm32Reg::R1, *dm));
                // Extract lhs bits into R2/R3.
                self.core
                    .text
                    .emit_u32(enc::vmov_rr_d(Arm32Reg::R2, Arm32Reg::R3, *dn));
                // R3 = lhs_hi with sign cleared.
                self.core.text.emit_u32(enc::bic_imm(
                    Arm32Reg::R3,
                    Arm32Reg::R3,
                    sign_imm8,
                    sign_rot,
                ));
                // R1 = rhs_hi sign bit only.
                self.core.text.emit_u32(enc::and_imm(
                    Arm32Reg::R1,
                    Arm32Reg::R1,
                    sign_imm8,
                    sign_rot,
                ));
                self.core
                    .text
                    .emit_u32(enc::orr_reg(Arm32Reg::R3, Arm32Reg::R3, Arm32Reg::R1));
                self.core
                    .text
                    .emit_u32(enc::vmov_d_rr(dd, Arm32Reg::R2, Arm32Reg::R3));
                self.restore_caller_saved_gp_regs(&[]);
            }
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Copysign) => {
                // F32 Copysign uses R0/R1 as scratch GP regs. Spill the
                // live JIT values first and restore after.
                let (sign_imm8, sign_rot) = enc::encode_arm_imm(0x8000_0000).unwrap();
                let sdn = *dn * 2;
                let sdm = *dm * 2;
                let sdd = dd * 2;
                self.spill_caller_saved_gp_regs();
                self.core.text.emit_u32(enc::vmov_r_s(Arm32Reg::R0, sdn)); // lhs bits
                self.core.text.emit_u32(enc::vmov_r_s(Arm32Reg::R1, sdm)); // rhs bits
                self.core.text.emit_u32(enc::bic_imm(
                    Arm32Reg::R0,
                    Arm32Reg::R0,
                    sign_imm8,
                    sign_rot,
                ));
                self.core.text.emit_u32(enc::and_imm(
                    Arm32Reg::R1,
                    Arm32Reg::R1,
                    sign_imm8,
                    sign_rot,
                ));
                self.core
                    .text
                    .emit_u32(enc::orr_reg(Arm32Reg::R0, Arm32Reg::R0, Arm32Reg::R1));
                self.core.text.emit_u32(enc::vmov_s_r(sdd, Arm32Reg::R0));
                self.restore_caller_saved_gp_regs(&[]);
            }
        }
        Ok(())
    }

    // ─── Float min / max helpers ────────────────────────────────────────────────
    //
    // ARMv7-A has no IEEE-754 minimum/maximum instruction (`vminnm`/`vmaxnm`
    // are ARMv8). We synthesise the wasm semantics from `vcmp` + branches:
    //
    //   1. Compare lhs and rhs, transferring VFP flags to APSR.
    //   2. If unordered (NaN) → result = lhs + rhs (which is also NaN, with
    //      the canonical exception flag set).
    //   3. If ordered and `lhs < rhs`  → result = lhs (for min) / rhs (for max).
    //   4. If ordered and `lhs > rhs`  → result = rhs (for min) / lhs (for max).
    //   5. If ordered and equal       → fall through to a sign-of-zero
    //      resolution: OR the bit patterns for min, AND for max. Both
    //      operations are no-ops on equal non-zero values (the bit patterns
    //      are identical) but correctly pick -0 over +0 (or vice-versa for
    //      max), because the only differing bit between +0 and -0 is the
    //      sign bit.
    //
    // The compare always happens *before* any move into `dd` so that
    // dead-input reuse (where `dd` aliases `dn`) doesn't clobber the lhs
    // before we read it. Each tail then writes `dd` exactly once.

    fn compile_float_min_max_s(
        &mut self,
        dd: u32,
        dn: u32,
        dm: u32,
        is_min: bool,
    ) -> Result<(), WasmError> {
        let sdd = dd * 2;
        let sdn = dn * 2;
        let sdm = dm * 2;

        // 1. Compare and copy VFP flags into APSR.
        self.core.text.emit_u32(enc::vcmp_s(sdn, sdm));
        self.core.text.emit_u32(enc::vmrs_apsr());

        let nan_case = self.core.new_label();
        let lt_case = self.core.new_label();
        let gt_case = self.core.new_label();
        let eq_case = self.core.new_label();
        let done = self.core.new_label();

        // 2. Branch into the four ordered/unordered cases. The order is
        //    chosen so the hot ordered/non-equal path is one taken branch
        //    plus a fall-through.
        //    bvs → unordered (NaN)
        //    beq → equal (zero handling)
        //    bmi → lhs < rhs
        //    fall-through → lhs > rhs
        self.emit_branch(BranchFixupKind::BCond(Cond::Vs), nan_case);
        self.emit_branch(BranchFixupKind::BCond(Cond::Eq), eq_case);
        self.emit_branch(BranchFixupKind::BCond(Cond::Mi), lt_case);
        self.emit_branch(BranchFixupKind::B, gt_case);

        // ── NaN case ────────────────────────────────────────────────────
        // result = lhs + rhs (propagates NaN with the standard quietening).
        self.core.bind_label(nan_case);
        self.core.text.emit_u32(enc::vadd_s(sdd, sdn, sdm));
        self.emit_branch(BranchFixupKind::B, done);

        // ── Equal case (zero sign resolution) ──────────────────────────
        // Move both operands' bit patterns to GP regs, OR (min) or AND
        // (max) them, then move back to the destination.
        self.core.bind_label(eq_case);
        {
            let lo = self.gp_scratch.scoped_alloc();
            let hi = self.gp_scratch.scoped_alloc();
            self.core.text.emit_u32(enc::vmov_r_s(*lo, sdn));
            self.core.text.emit_u32(enc::vmov_r_s(*hi, sdm));
            if is_min {
                self.core.text.emit_u32(enc::orr_reg(*lo, *lo, *hi));
            } else {
                self.core.text.emit_u32(enc::and_reg(*lo, *lo, *hi));
            }
            self.core.text.emit_u32(enc::vmov_s_r(sdd, *lo));
        }
        self.emit_branch(BranchFixupKind::B, done);

        // ── lhs < rhs ───────────────────────────────────────────────────
        // For min: dst = lhs. For max: dst = rhs.
        self.core.bind_label(lt_case);
        {
            let src = if is_min { sdn } else { sdm };
            if sdd != src {
                self.core.text.emit_u32(enc::vmov_s(sdd, src));
            }
        }
        self.emit_branch(BranchFixupKind::B, done);

        // ── lhs > rhs ───────────────────────────────────────────────────
        // For min: dst = rhs. For max: dst = lhs.
        self.core.bind_label(gt_case);
        {
            let src = if is_min { sdm } else { sdn };
            if sdd != src {
                self.core.text.emit_u32(enc::vmov_s(sdd, src));
            }
        }

        self.core.bind_label(done);
        Ok(())
    }

    fn compile_float_min_max_d(
        &mut self,
        dd: u32,
        dn: u32,
        dm: u32,
        is_min: bool,
    ) -> Result<(), WasmError> {
        // 1. Compare and copy VFP flags into APSR.
        self.core.text.emit_u32(enc::vcmp_d(dn, dm));
        self.core.text.emit_u32(enc::vmrs_apsr());

        let nan_case = self.core.new_label();
        let lt_case = self.core.new_label();
        let gt_case = self.core.new_label();
        let eq_case = self.core.new_label();
        let done = self.core.new_label();

        self.emit_branch(BranchFixupKind::BCond(Cond::Vs), nan_case);
        self.emit_branch(BranchFixupKind::BCond(Cond::Eq), eq_case);
        self.emit_branch(BranchFixupKind::BCond(Cond::Mi), lt_case);
        self.emit_branch(BranchFixupKind::B, gt_case);

        // ── NaN case ────────────────────────────────────────────────────
        self.core.bind_label(nan_case);
        self.core.text.emit_u32(enc::vadd_d(dd, dn, dm));
        self.emit_branch(BranchFixupKind::B, done);

        // ── Equal case (zero sign resolution) ──────────────────────────
        // For F64 we need to combine both 32-bit halves of each operand.
        // Move dn to (dn_lo, dn_hi), dm to (dm_lo, dm_hi), then combine
        // each half with OR/AND, and re-pack into dd. This needs four GP
        // scratch slots; armv7a only has two in the pool, so we serialise
        // through R0/R1 with the spill/restore pattern used elsewhere.
        self.core.bind_label(eq_case);
        self.spill_caller_saved_gp_regs();
        self.core
            .text
            .emit_u32(enc::vmov_rr_d(Arm32Reg::R0, Arm32Reg::R1, dn));
        self.core
            .text
            .emit_u32(enc::vmov_rr_d(Arm32Reg::R2, Arm32Reg::R3, dm));
        if is_min {
            self.core
                .text
                .emit_u32(enc::orr_reg(Arm32Reg::R0, Arm32Reg::R0, Arm32Reg::R2));
            self.core
                .text
                .emit_u32(enc::orr_reg(Arm32Reg::R1, Arm32Reg::R1, Arm32Reg::R3));
        } else {
            self.core
                .text
                .emit_u32(enc::and_reg(Arm32Reg::R0, Arm32Reg::R0, Arm32Reg::R2));
            self.core
                .text
                .emit_u32(enc::and_reg(Arm32Reg::R1, Arm32Reg::R1, Arm32Reg::R3));
        }
        self.core
            .text
            .emit_u32(enc::vmov_d_rr(dd, Arm32Reg::R0, Arm32Reg::R1));
        self.restore_caller_saved_gp_regs(&[]);
        self.emit_branch(BranchFixupKind::B, done);

        // ── lhs < rhs ───────────────────────────────────────────────────
        self.core.bind_label(lt_case);
        {
            let src = if is_min { dn } else { dm };
            if dd != src {
                self.core.text.emit_u32(enc::vmov_d(dd, src));
            }
        }
        self.emit_branch(BranchFixupKind::B, done);

        // ── lhs > rhs ───────────────────────────────────────────────────
        self.core.bind_label(gt_case);
        {
            let src = if is_min { dm } else { dn };
            if dd != src {
                self.core.text.emit_u32(enc::vmov_d(dd, src));
            }
        }

        self.core.bind_label(done);
        Ok(())
    }

    // ─── Float unary ────────────────────────────────────────────────────────────

    fn compile_float_unary(
        &mut self,
        width: MachineFloatWidth,
        op: MachineFloatUnaryOp,
        dst: MachineReg,
        src: &MachineValue,
    ) -> Result<(), WasmError> {
        let dm = prepare_fp(
            &mut self.core.text,
            &self.gp_scratch,
            &self.fp_scratch,
            width,
            *src,
        )?
        .detach();
        let dd = self.map_fp_dreg(dst)?;

        match (width, op) {
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Abs) => {
                self.core.text.emit_u32(enc::vabs_d(dd, *dm));
            }
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Neg) => {
                self.core.text.emit_u32(enc::vneg_d(dd, *dm));
            }
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Ceil) => {
                self.spill_caller_saved_gp_regs();
                if *dm != FP_SCRATCH0 {
                    self.core.text.emit_u32(enc::vmov_d(FP_SCRATCH0, *dm));
                }
                self.emit_host_call(armv7a_f64_ceil as usize);
                if dd != FP_SCRATCH0 {
                    self.core.text.emit_u32(enc::vmov_d(dd, FP_SCRATCH0));
                }
                self.restore_caller_saved_gp_regs(&[]);
            }
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Floor) => {
                self.spill_caller_saved_gp_regs();
                if *dm != FP_SCRATCH0 {
                    self.core.text.emit_u32(enc::vmov_d(FP_SCRATCH0, *dm));
                }
                self.emit_host_call(armv7a_f64_floor as usize);
                if dd != FP_SCRATCH0 {
                    self.core.text.emit_u32(enc::vmov_d(dd, FP_SCRATCH0));
                }
                self.restore_caller_saved_gp_regs(&[]);
            }
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Trunc) => {
                self.spill_caller_saved_gp_regs();
                if *dm != FP_SCRATCH0 {
                    self.core.text.emit_u32(enc::vmov_d(FP_SCRATCH0, *dm));
                }
                self.emit_host_call(armv7a_f64_trunc as usize);
                if dd != FP_SCRATCH0 {
                    self.core.text.emit_u32(enc::vmov_d(dd, FP_SCRATCH0));
                }
                self.restore_caller_saved_gp_regs(&[]);
            }
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Nearest) => {
                self.spill_caller_saved_gp_regs();
                self.core
                    .text
                    .emit_u32(enc::vmov_rr_d(Arm32Reg::R0, Arm32Reg::R1, *dm));
                self.emit_host_call(armv7a_f64_nearest_bits as usize);
                self.core
                    .text
                    .emit_u32(enc::vmov_d_rr(dd, Arm32Reg::R0, Arm32Reg::R1));
                self.restore_caller_saved_gp_regs(&[]);
            }
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Sqrt) => {
                self.core.text.emit_u32(enc::vsqrt_d(dd, *dm));
            }
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Abs) => {
                self.core.text.emit_u32(enc::vabs_s(dd * 2, *dm * 2));
            }
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Neg) => {
                self.core.text.emit_u32(enc::vneg_s(dd * 2, *dm * 2));
            }
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Ceil) => {
                self.spill_caller_saved_gp_regs();
                let src_s = *dm * 2;
                let dst_s = dd * 2;
                let s0 = FP_SCRATCH0 * 2;
                if src_s != s0 {
                    self.core.text.emit_u32(enc::vmov_s(s0, src_s));
                }
                self.emit_host_call(armv7a_f32_ceil as usize);
                if dst_s != s0 {
                    self.core.text.emit_u32(enc::vmov_s(dst_s, s0));
                }
                self.restore_caller_saved_gp_regs(&[]);
            }
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Floor) => {
                self.spill_caller_saved_gp_regs();
                let src_s = *dm * 2;
                let dst_s = dd * 2;
                let s0 = FP_SCRATCH0 * 2;
                if src_s != s0 {
                    self.core.text.emit_u32(enc::vmov_s(s0, src_s));
                }
                self.emit_host_call(armv7a_f32_floor as usize);
                if dst_s != s0 {
                    self.core.text.emit_u32(enc::vmov_s(dst_s, s0));
                }
                self.restore_caller_saved_gp_regs(&[]);
            }
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Trunc) => {
                self.spill_caller_saved_gp_regs();
                let src_s = *dm * 2;
                let dst_s = dd * 2;
                let s0 = FP_SCRATCH0 * 2;
                if src_s != s0 {
                    self.core.text.emit_u32(enc::vmov_s(s0, src_s));
                }
                self.emit_host_call(armv7a_f32_trunc as usize);
                if dst_s != s0 {
                    self.core.text.emit_u32(enc::vmov_s(dst_s, s0));
                }
                self.restore_caller_saved_gp_regs(&[]);
            }
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Nearest) => {
                self.spill_caller_saved_gp_regs();
                let src_s = *dm * 2;
                let dst_s = dd * 2;
                self.core.text.emit_u32(enc::vmov_r_s(Arm32Reg::R0, src_s));
                self.emit_host_call(armv7a_f32_nearest_bits as usize);
                self.core.text.emit_u32(enc::vmov_s_r(dst_s, Arm32Reg::R0));
                self.restore_caller_saved_gp_regs(&[]);
            }
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Sqrt) => {
                self.core.text.emit_u32(enc::vsqrt_s(dd * 2, *dm * 2));
            }
        }
        Ok(())
    }

    // ─── Float compare ──────────────────────────────────────────────────────────

    fn compile_float_compare(
        &mut self,
        width: MachineFloatWidth,
        kind: MachineCompareKind,
        dst: MachineReg,
        lhs: &MachineValue,
        rhs: &MachineValue,
    ) -> Result<(), WasmError> {
        let lhs_d = prepare_fp(
            &mut self.core.text,
            &self.gp_scratch,
            &self.fp_scratch,
            width,
            *lhs,
        )?
        .detach();
        let rhs_d = prepare_fp(
            &mut self.core.text,
            &self.gp_scratch,
            &self.fp_scratch,
            width,
            *rhs,
        )?
        .detach();
        let dst_hw = map_reg(dst)?;

        match width {
            MachineFloatWidth::F64 => {
                self.core.text.emit_u32(enc::vcmp_d(*lhs_d, *rhs_d));
            }
            MachineFloatWidth::F32 => {
                self.core.text.emit_u32(enc::vcmp_s(*lhs_d * 2, *rhs_d * 2));
            }
        }
        self.core.text.emit_u32(enc::vmrs_apsr());

        // ARM VFP exposes unordered comparisons via V=1 after VMRS, while Wasm
        // requires all comparisons except `ne` to return false for NaNs.
        let ordered = self.core.new_label();
        let done = self.core.new_label();
        self.emit_branch(BranchFixupKind::BCond(Cond::Vc), ordered);
        self.emit_load_u32(
            dst_hw,
            u32::from(Self::float_compare_unordered_result(kind)),
        );
        self.emit_branch(BranchFixupKind::B, done);

        self.core.bind_label(ordered);
        self.emit_load_u32(dst_hw, 0);
        let (imm8, rot) = enc::encode_arm_imm(1).unwrap();
        self.core.text.emit_u32(enc::dp_imm_cond(
            Self::float_compare_ordered_cond(kind),
            0b1101,
            false,
            dst_hw,
            Arm32Reg::R0,
            imm8,
            rot,
        ));
        self.core.bind_label(done);
        Ok(())
    }

    #[inline]
    fn float_compare_ordered_cond(kind: MachineCompareKind) -> Cond {
        match kind {
            MachineCompareKind::Eq => Cond::Eq,
            MachineCompareKind::Ne => Cond::Ne,
            MachineCompareKind::Lt => Cond::Mi,
            MachineCompareKind::Gt => Cond::Gt,
            MachineCompareKind::Le => Cond::Ls,
            MachineCompareKind::Ge => Cond::Ge,
        }
    }

    #[inline]
    fn float_compare_unordered_result(kind: MachineCompareKind) -> bool {
        matches!(kind, MachineCompareKind::Ne)
    }

    // ─── Convert ────────────────────────────────────────────────────────────────

    fn compile_convert(
        &mut self,
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
                            self.core.text.emit_u32(enc::mov_reg(dst_hw, src_hw));
                        }
                    }
                    MachineValue::ReservedReg(reg) => {
                        return Err(WasmError::internal(alloc::format!(
                            "armv7a I32WrapI64 cannot consume reserved cache register {} as src",
                            reg.0
                        )));
                    }
                    MachineValue::Imm64(v) => self.emit_load_u32(dst_hw, *v as u32),
                }
            }
            MachineConvertOp::I64ExtendI32U | MachineConvertOp::I64ExtendI32S => {
                let dst_hw = map_reg(dst)?;
                match src {
                    MachineValue::Reg(r) => {
                        let src_hw = map_reg(*r)?;
                        if dst_hw != src_hw {
                            self.core.text.emit_u32(enc::mov_reg(dst_hw, src_hw));
                        }
                    }
                    MachineValue::ReservedReg(reg) => {
                        return Err(WasmError::internal(alloc::format!(
                            "armv7a I64ExtendI32 cannot consume reserved cache register {} as src",
                            reg.0
                        )));
                    }
                    MachineValue::Imm64(v) => self.emit_load_u32(dst_hw, *v as u32),
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
                self.compile_convert_float_to_i32(op, dst, src)?;
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
                let dd = self.map_fp_dreg(dst)?;
                let src_gp = prepare_gp(&mut self.core.text, &self.gp_scratch, *src)?;
                let sd_tmp = FP_SCRATCH0 * 2;
                self.core.text.emit_u32(enc::vmov_s_r(sd_tmp, *src_gp));
                self.core.text.emit_u32(enc::vcvt_d_s32(dd, sd_tmp));
            }
            MachineConvertOp::F64ConvertI32U => {
                let dd = self.map_fp_dreg(dst)?;
                let src_gp = prepare_gp(&mut self.core.text, &self.gp_scratch, *src)?;
                let sd_tmp = FP_SCRATCH0 * 2;
                self.core.text.emit_u32(enc::vmov_s_r(sd_tmp, *src_gp));
                self.core.text.emit_u32(enc::vcvt_d_u32(dd, sd_tmp));
            }

            // ─── I32 → F32 (GP src → FP dst) ────────────────────────────────
            MachineConvertOp::F32ConvertI32S => {
                let sd = self.map_fp_dreg(dst)? * 2; // S-register
                let src_gp = prepare_gp(&mut self.core.text, &self.gp_scratch, *src)?;
                let sd_tmp = FP_SCRATCH0 * 2;
                self.core.text.emit_u32(enc::vmov_s_r(sd_tmp, *src_gp));
                self.core.text.emit_u32(enc::vcvt_s_s32(sd, sd_tmp));
            }
            MachineConvertOp::F32ConvertI32U => {
                let sd = self.map_fp_dreg(dst)? * 2;
                let src_gp = prepare_gp(&mut self.core.text, &self.gp_scratch, *src)?;
                let sd_tmp = FP_SCRATCH0 * 2;
                self.core.text.emit_u32(enc::vmov_s_r(sd_tmp, *src_gp));
                self.core.text.emit_u32(enc::vcvt_s_u32(sd, sd_tmp));
            }

            // ─── F32 ↔ F64 (FP → FP) ───────────────────────────────────────
            MachineConvertOp::F64PromoteF32 => {
                let dd = self.map_fp_dreg(dst)?;
                let src_fp = prepare_fp(
                    &mut self.core.text,
                    &self.gp_scratch,
                    &self.fp_scratch,
                    MachineFloatWidth::F32,
                    *src,
                )?;
                let mut sm = *src_fp * 2;
                if *src_fp == dd {
                    sm = FP_SCRATCH0 * 2;
                    self.core.text.emit_u32(enc::vmov_s(sm, *src_fp * 2));
                }
                self.core.text.emit_u32(enc::vcvt_d_s(dd, sm));
            }
            MachineConvertOp::F32DemoteF64 => {
                let sd = self.map_fp_dreg(dst)? * 2;
                let src_fp = prepare_fp(
                    &mut self.core.text,
                    &self.gp_scratch,
                    &self.fp_scratch,
                    MachineFloatWidth::F64,
                    *src,
                )?;
                let mut dm = *src_fp;
                if dm * 2 == sd {
                    dm = FP_SCRATCH0;
                    self.core.text.emit_u32(enc::vmov_d(dm, *src_fp));
                }
                self.core.text.emit_u32(enc::vcvt_s_d(sd, dm));
            }

            // ─── I64 → F64/F32 (via helper call) ─────────────────────────────
            // On ARM32, the GP register holds the low 32 bits of the i64.
            // We sign/zero-extend from the 32-bit value to form the full i64,
            // then call a helper that does the conversion.
            MachineConvertOp::F64ConvertI64S => {
                let dd = self.map_fp_dreg(dst)?;
                let src_hw = prepare_gp(&mut self.core.text, &self.gp_scratch, *src)?.detach();
                self.spill_caller_saved_gp_regs();
                // R0 = lo, R1 = hi (sign-extend: hi = lo >> 31, arithmetic shift)
                self.core.text.emit_u32(enc::mov_reg(Arm32Reg::R0, *src_hw));
                self.core
                    .text
                    .emit_u32(enc::asr_imm(Arm32Reg::R1, *src_hw, 31));
                self.emit_host_call(armv7a_i64s_to_f64 as usize);
                // Result is in D0 (EABI: f64 returned in D0)
                if dd != FP_SCRATCH0 {
                    self.core.text.emit_u32(enc::vmov_d(dd, FP_SCRATCH0));
                }
                self.restore_caller_saved_gp_regs(&[]);
            }
            MachineConvertOp::F64ConvertI64U => {
                let dd = self.map_fp_dreg(dst)?;
                let src_hw = prepare_gp(&mut self.core.text, &self.gp_scratch, *src)?.detach();
                self.spill_caller_saved_gp_regs();
                // R0 = lo, R1 = 0 (zero-extend)
                self.core.text.emit_u32(enc::mov_reg(Arm32Reg::R0, *src_hw));
                self.emit_load_u32(Arm32Reg::R1, 0);
                self.emit_host_call(armv7a_i64u_to_f64 as usize);
                if dd != FP_SCRATCH0 {
                    self.core.text.emit_u32(enc::vmov_d(dd, FP_SCRATCH0));
                }
                self.restore_caller_saved_gp_regs(&[]);
            }
            MachineConvertOp::F32ConvertI64S => {
                let sd = self.map_fp_dreg(dst)? * 2;
                let src_hw = prepare_gp(&mut self.core.text, &self.gp_scratch, *src)?.detach();
                self.spill_caller_saved_gp_regs();
                self.core.text.emit_u32(enc::mov_reg(Arm32Reg::R0, *src_hw));
                self.core
                    .text
                    .emit_u32(enc::asr_imm(Arm32Reg::R1, *src_hw, 31));
                self.emit_host_call(armv7a_i64s_to_f32 as usize);
                // Result in S0 (EABI: f32 returned in S0)
                let s0 = FP_SCRATCH0 * 2;
                if sd != s0 {
                    self.core.text.emit_u32(enc::vmov_s(sd, s0));
                }
                self.restore_caller_saved_gp_regs(&[]);
            }
            MachineConvertOp::F32ConvertI64U => {
                let sd = self.map_fp_dreg(dst)? * 2;
                let src_hw = prepare_gp(&mut self.core.text, &self.gp_scratch, *src)?.detach();
                self.spill_caller_saved_gp_regs();
                self.core.text.emit_u32(enc::mov_reg(Arm32Reg::R0, *src_hw));
                self.emit_load_u32(Arm32Reg::R1, 0);
                self.emit_host_call(armv7a_i64u_to_f32 as usize);
                let s0 = FP_SCRATCH0 * 2;
                if sd != s0 {
                    self.core.text.emit_u32(enc::vmov_s(sd, s0));
                }
                self.restore_caller_saved_gp_regs(&[]);
            }

            // ─── Reinterpret (bit cast, no conversion) ──────────────────────
            MachineConvertOp::I32ReinterpretF32 => {
                let dst_hw = map_reg(dst)?;
                let src_fp = prepare_fp(
                    &mut self.core.text,
                    &self.gp_scratch,
                    &self.fp_scratch,
                    MachineFloatWidth::F32,
                    *src,
                )?;
                let sm = *src_fp * 2;
                self.core.text.emit_u32(enc::vmov_r_s(dst_hw, sm));
            }
            MachineConvertOp::F32ReinterpretI32 => {
                let sd = self.map_fp_dreg(dst)? * 2;
                let src_gp = prepare_gp(&mut self.core.text, &self.gp_scratch, *src)?;
                self.core.text.emit_u32(enc::vmov_s_r(sd, *src_gp));
            }
            MachineConvertOp::I64ReinterpretF64 => {
                // F64 (D-reg) → I64 (low half lives in `dst_hw`). The
                // `vmov_rr_d` writes both halves, so the high-half slot
                // must be a GP scratch — never a JIT-live register.
                let dst_hw = map_reg(dst)?;
                let src_fp = prepare_fp(
                    &mut self.core.text,
                    &self.gp_scratch,
                    &self.fp_scratch,
                    MachineFloatWidth::F64,
                    *src,
                )?;
                let dm = *src_fp;
                let hi_scratch = self.gp_scratch.scoped_alloc();
                self.core
                    .text
                    .emit_u32(enc::vmov_rr_d(dst_hw, *hi_scratch, dm));
            }
            MachineConvertOp::F64ReinterpretI64 => {
                // I64 (low half in GP, high half = 0 on 32-bit GP) → F64
                // (D-reg). Use JIT GP scratches (R12/R14) for any
                // intermediate values so live dynamic-bank registers are
                // not clobbered.
                let dd = self.map_fp_dreg(dst)?;
                match src {
                    MachineValue::Reg(r) => {
                        let src_hw = map_reg(*r)?;
                        let zero_s = self.gp_scratch.scoped_alloc();
                        emit_load_u32_into(&mut self.core.text, *zero_s, 0);
                        self.core.text.emit_u32(enc::vmov_d_rr(dd, src_hw, *zero_s));
                    }
                    MachineValue::ReservedReg(reg) => {
                        return Err(WasmError::internal(alloc::format!(
                            "armv7a F64ReinterpretI64 cannot consume reserved cache register {} as src",
                            reg.0
                        )));
                    }
                    MachineValue::Imm64(v) => {
                        let lo_s = self.gp_scratch.scoped_alloc();
                        let hi_s = self.gp_scratch.scoped_alloc();
                        emit_load_u32_into(&mut self.core.text, *lo_s, *v as u32);
                        emit_load_u32_into(&mut self.core.text, *hi_s, (*v >> 32) as u32);
                        self.core.text.emit_u32(enc::vmov_d_rr(dd, *lo_s, *hi_s));
                    }
                }
            }
        }
        Ok(())
    }

    // ─── Select ─────────────────────────────────────────────────────────────────

    fn compile_select(
        &mut self,
        dst: MachineReg,
        condition: &MachineValue,
        true_val: &MachineValue,
        false_val: &MachineValue,
    ) -> Result<(), WasmError> {
        if self.is_fp_machine_reg(dst) {
            // FP select: use branch-based approach since ARM32 has no conditional VMOV
            let dd = self.map_fp_dreg(dst)?;

            // Test condition first
            let cond_hw = prepare_gp(&mut self.core.text, &self.gp_scratch, *condition)?.detach();
            self.core.text.emit_u32(enc::cmp_imm(*cond_hw, 0, 0));

            let true_label = self.core.new_label();
            let done_label = self.core.new_label();
            self.emit_branch(BranchFixupKind::BCond(Cond::Ne), true_label);

            // False path: load false_val to dd
            match false_val {
                MachineValue::Reg(r) if self.is_fp_machine_reg(*r) => {
                    let sd = self.map_fp_dreg(*r)?;
                    if dd != sd {
                        self.core.text.emit_u32(enc::vmov_d(dd, sd));
                    }
                }
                MachineValue::Reg(r) => {
                    // Use a JIT GP scratch (R12/R14) for the high-half
                    // zero so we don't clobber live dynamic R0..R3, R9.
                    let src = map_reg(*r)?;
                    let zero_s = self.gp_scratch.scoped_alloc();
                    emit_load_u32_into(&mut self.core.text, *zero_s, 0);
                    self.core.text.emit_u32(enc::vmov_d_rr(dd, src, *zero_s));
                }
                MachineValue::ReservedReg(reg) => {
                    return Err(WasmError::internal(alloc::format!(
                        "armv7a FP select cannot consume reserved cache register {} as false_val",
                        reg.0
                    )));
                }
                MachineValue::Imm64(v) => {
                    let lo_s = self.gp_scratch.scoped_alloc();
                    let hi_s = self.gp_scratch.scoped_alloc();
                    emit_load_u32_into(&mut self.core.text, *lo_s, *v as u32);
                    emit_load_u32_into(&mut self.core.text, *hi_s, (*v >> 32) as u32);
                    self.core.text.emit_u32(enc::vmov_d_rr(dd, *lo_s, *hi_s));
                }
            }
            self.emit_branch(BranchFixupKind::B, done_label);

            // True path: load true_val to dd
            self.core.bind_label(true_label);
            match true_val {
                MachineValue::Reg(r) if self.is_fp_machine_reg(*r) => {
                    let sd = self.map_fp_dreg(*r)?;
                    if dd != sd {
                        self.core.text.emit_u32(enc::vmov_d(dd, sd));
                    }
                }
                MachineValue::Reg(r) => {
                    let src = map_reg(*r)?;
                    let zero_s = self.gp_scratch.scoped_alloc();
                    emit_load_u32_into(&mut self.core.text, *zero_s, 0);
                    self.core.text.emit_u32(enc::vmov_d_rr(dd, src, *zero_s));
                }
                MachineValue::ReservedReg(reg) => {
                    return Err(WasmError::internal(alloc::format!(
                        "armv7a FP select cannot consume reserved cache register {} as true_val",
                        reg.0
                    )));
                }
                MachineValue::Imm64(v) => {
                    let lo_s = self.gp_scratch.scoped_alloc();
                    let hi_s = self.gp_scratch.scoped_alloc();
                    emit_load_u32_into(&mut self.core.text, *lo_s, *v as u32);
                    emit_load_u32_into(&mut self.core.text, *hi_s, (*v >> 32) as u32);
                    self.core.text.emit_u32(enc::vmov_d_rr(dd, *lo_s, *hi_s));
                }
            }
            self.core.bind_label(done_label);
            return Ok(());
        }

        // GP select
        let dst_hw = map_reg(dst)?;

        // Test condition before touching dst so dst == cond is safe.
        let cond_hw = prepare_gp(&mut self.core.text, &self.gp_scratch, *condition)?.detach();
        self.core.text.emit_u32(enc::cmp_imm(*cond_hw, 0, 0));

        if Self::gp_value_aliases_dst(self, true_val, dst_hw)? {
            // Loading the false arm first would clobber the live true source when
            // `dst` reuses that register. Seed `dst` with the true arm, then
            // overwrite it on the false path.
            Self::emit_gp_select_value(self, dst_hw, true_val)?;
            Self::emit_gp_select_value_cond(self, dst_hw, false_val, Cond::Eq)?;
        } else {
            Self::emit_gp_select_value(self, dst_hw, false_val)?;
            Self::emit_gp_select_value_cond(self, dst_hw, true_val, Cond::Ne)?;
        }

        Ok(())
    }

    fn gp_value_aliases_dst(
        &self,
        value: &MachineValue,
        dst_hw: Arm32Reg,
    ) -> Result<bool, WasmError> {
        match value {
            MachineValue::Reg(r) if !self.is_fp_machine_reg(*r) => Ok(map_reg(*r)? == dst_hw),
            _ => Ok(false),
        }
    }

    fn emit_gp_select_value(
        &mut self,
        dst_hw: Arm32Reg,
        value: &MachineValue,
    ) -> Result<(), WasmError> {
        match value {
            MachineValue::Reg(r) if self.is_fp_machine_reg(*r) => {
                let sd = self.map_fp_dreg(*r)?;
                let hi_scratch = self.gp_scratch.scoped_alloc();
                self.core
                    .text
                    .emit_u32(enc::vmov_rr_d(dst_hw, *hi_scratch, sd));
            }
            MachineValue::Reg(r) => {
                let src = map_reg(*r)?;
                if dst_hw != src {
                    self.core.text.emit_u32(enc::mov_reg(dst_hw, src));
                }
            }
            MachineValue::ReservedReg(reg) => {
                return Err(WasmError::internal(alloc::format!(
                    "armv7a emit_gp_select_value cannot consume reserved cache register {} as value",
                    reg.0
                )));
            }
            MachineValue::Imm64(v) => {
                self.emit_load_u32(dst_hw, *v as u32);
            }
        }
        Ok(())
    }

    fn emit_gp_select_value_cond(
        &mut self,
        dst_hw: Arm32Reg,
        value: &MachineValue,
        cond: Cond,
    ) -> Result<(), WasmError> {
        match value {
            MachineValue::Reg(r) if self.is_fp_machine_reg(*r) => {
                let skip = self.core.new_label();
                self.emit_branch(BranchFixupKind::BCond(cond.invert()), skip);
                let sd = self.map_fp_dreg(*r)?;
                let hi_scratch = self.gp_scratch.scoped_alloc();
                self.core
                    .text
                    .emit_u32(enc::vmov_rr_d(dst_hw, *hi_scratch, sd));
                self.core.bind_label(skip);
            }
            MachineValue::Reg(r) => {
                let src = map_reg(*r)?;
                self.core
                    .text
                    .emit_u32(enc::mov_reg_cond(cond, dst_hw, src));
            }
            MachineValue::ReservedReg(reg) => {
                return Err(WasmError::internal(alloc::format!(
                    "armv7a emit_gp_select_value_cond cannot consume reserved cache register {} as value",
                    reg.0
                )));
            }
            MachineValue::Imm64(v) => {
                let s = self.gp_scratch.scoped_alloc();
                emit_load_u32_into(&mut self.core.text, *s, *v as u32);
                self.core.text.emit_u32(enc::mov_reg_cond(cond, dst_hw, *s));
            }
        }
        Ok(())
    }

    // ─── IntCompare ─────────────────────────────────────────────────────────

    fn compile_int_compare(
        &mut self,
        _width: MachineIntWidth,
        kind: MachineCompareKind,
        sign: MachineSign,
        dst: MachineReg,
        lhs: &MachineValue,
        rhs: &MachineValue,
    ) -> Result<(), WasmError> {
        let dst_hw = map_reg(dst)?;
        {
            let lhs_gp = prepare_gp(&mut self.core.text, &self.gp_scratch, *lhs)?;
            let lhs_hw = *lhs_gp;
            match rhs {
                MachineValue::Reg(r) => {
                    self.core.text.emit_u32(enc::cmp_reg(lhs_hw, map_reg(*r)?));
                }
                MachineValue::ReservedReg(reg) => {
                    return Err(WasmError::internal(alloc::format!(
                        "armv7a compile_int_compare cannot consume reserved cache register {} as rhs",
                        reg.0
                    )));
                }
                MachineValue::Imm64(v) => {
                    if let Some((imm8, rot)) = enc::encode_arm_imm(*v as u32) {
                        self.core.text.emit_u32(enc::cmp_imm(lhs_hw, imm8, rot));
                    } else {
                        let s = self.gp_scratch.scoped_alloc();
                        emit_load_u32_into(&mut self.core.text, *s, *v as u32);
                        self.core.text.emit_u32(enc::cmp_reg(lhs_hw, *s));
                    }
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

        self.emit_load_u32(dst_hw, 0);
        let (imm8, rot) = enc::encode_arm_imm(1).unwrap();
        self.core.text.emit_u32(enc::dp_imm_cond(
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

    // ─── Bitfield extract (UBFX) ───────────────────────────────────────────

    fn compile_bitfield_extract_u(
        &mut self,
        _width: MachineIntWidth,
        dst: MachineReg,
        src: MachineReg,
        lsb: u8,
        bits: u8,
    ) -> Result<(), WasmError> {
        let dst_hw = map_reg(dst)?;
        let src_hw = map_reg(src)?;
        self.core
            .text
            .emit_u32(enc::ubfx(dst_hw, src_hw, lsb as u32, bits as u32));
        Ok(())
    }

    // ─── Shifted-register binary ───────────────────────────────────────────

    fn compile_int_binary_shifted(
        &mut self,
        _width: MachineIntWidth,
        op: MachineIntBinaryOp,
        dst: MachineReg,
        lhs: MachineReg,
        rhs: MachineReg,
        shift: MachineShiftOp,
        amount: u8,
    ) -> Result<(), WasmError> {
        let dst_hw = map_reg(dst)?;
        let lhs_hw = map_reg(lhs)?;
        let rhs_hw = map_reg(rhs)?;
        let shift_type: u32 = match shift {
            MachineShiftOp::Lsl => 0b00,
            MachineShiftOp::Lsr => 0b01,
            MachineShiftOp::Asr => 0b10,
        };
        let amt = amount as u32;
        let inst = match op {
            MachineIntBinaryOp::Add => {
                enc::add_reg_shifted(dst_hw, lhs_hw, rhs_hw, shift_type, amt)
            }
            MachineIntBinaryOp::Sub => {
                enc::sub_reg_shifted(dst_hw, lhs_hw, rhs_hw, shift_type, amt)
            }
            MachineIntBinaryOp::And => {
                enc::and_reg_shifted(dst_hw, lhs_hw, rhs_hw, shift_type, amt)
            }
            MachineIntBinaryOp::Or => enc::orr_reg_shifted(dst_hw, lhs_hw, rhs_hw, shift_type, amt),
            MachineIntBinaryOp::Xor => {
                enc::eor_reg_shifted(dst_hw, lhs_hw, rhs_hw, shift_type, amt)
            }
            _ => {
                return Err(WasmError::internal(alloc::format!(
                    "IntBinaryShifted: unsupported op {:?}",
                    op
                )));
            }
        };
        self.core.text.emit_u32(inst);
        Ok(())
    }

    // ─── Test bits (TST + conditional MOV) ─────────────────────────────────

    fn compile_test_bits(
        &mut self,
        _width: MachineIntWidth,
        kind: MachineCompareKind,
        dst: MachineReg,
        src: MachineReg,
        mask: &MachineValue,
    ) -> Result<(), WasmError> {
        let dst_hw = map_reg(dst)?;
        let src_hw = map_reg(src)?;

        // Emit TST to set flags.
        match mask {
            MachineValue::Reg(r) => {
                self.core.text.emit_u32(enc::tst_reg(src_hw, map_reg(*r)?));
            }
            MachineValue::ReservedReg(reg) => {
                return Err(WasmError::internal(alloc::format!(
                    "armv7a compile_test_bits cannot consume reserved cache register {} as mask",
                    reg.0
                )));
            }
            MachineValue::Imm64(v) => {
                if let Some((imm8, rot)) = enc::encode_arm_imm(*v as u32) {
                    self.core.text.emit_u32(enc::tst_imm(src_hw, imm8, rot));
                } else {
                    let s = self.gp_scratch.scoped_alloc();
                    let tmp = *s;
                    emit_load_u32_into(&mut self.core.text, tmp, *v as u32);
                    self.core.text.emit_u32(enc::tst_reg(src_hw, tmp));
                }
            }
        }

        // Materialize boolean: load 0, then conditionally set to 1.
        let cond = match kind {
            MachineCompareKind::Eq => Cond::Eq,
            MachineCompareKind::Ne => Cond::Ne,
            _ => {
                return Err(WasmError::internal(alloc::format!(
                    "TestBits: unsupported compare kind {:?}",
                    kind
                )));
            }
        };
        emit_load_u32_into(&mut self.core.text, dst_hw, 0);
        let (imm8, rot) = enc::encode_arm_imm(1).unwrap();
        self.core.text.emit_u32(enc::dp_imm_cond(
            cond,
            0b1101, // MOV
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
        &mut self,
        kind: MachineTrapKind,
        cond: &MachineBranchCond,
    ) -> Result<(), WasmError> {
        let arm_cond = self.compile_branch_condition(cond)?;
        let trap_label = self.core.ensure_trap_label(kind);
        self.emit_branch(BranchFixupKind::BCond(arm_cond), trap_label);
        Ok(())
    }

    // ─── CallExternal ───────────────────────────────────────────────────────

    fn compile_call_external(&mut self, call: &MachineCallExternal) -> Result<(), WasmError> {
        let metadata = self.core.compiled.const_ptr(call.metadata).ok_or_else(|| {
            WasmError::internal("armv7a: external-call metadata is out of range".into())
        })?;

        let helper_ptr = crate::vm::runtime::external::call_external_entry_ptr() as usize;

        // Imported calls cross the foreign C ABI, so the caller-saved GP
        // dynamic subset must be spilled explicitly before we stage the ABI
        // arguments into R0-R2.
        self.spill_caller_saved_gp_regs();

        // EABI: fn(ctx: *mut NativeContext, frame: *mut u64, metadata: *const u8) -> u32
        self.core
            .text
            .emit_u32(enc::mov_reg(Arm32Reg::R0, map_fixed_reg(MACHINE_CTX_REG)));
        self.core
            .text
            .emit_u32(enc::mov_reg(Arm32Reg::R1, map_fixed_reg(MACHINE_FP_REG)));
        self.emit_load_addr(Arm32Reg::R2, metadata as usize);

        self.emit_host_call(helper_ptr);

        // Preserve the status code across the GP restore, then re-materialize
        // it in R0 for the post-call error check.
        self.core
            .text
            .emit_u32(enc::mov_reg(Arm32Reg::R12, Arm32Reg::R0));
        self.restore_caller_saved_gp_regs(&[]);
        self.core
            .text
            .emit_u32(enc::mov_reg(Arm32Reg::R0, Arm32Reg::R12));

        // Check return value: if non-zero, return error
        self.core.text.emit_u32(enc::cmp_imm(Arm32Reg::R0, 0, 0));
        let body_local_error = self.core.body_local_error_label;
        self.emit_branch(BranchFixupKind::BCond(Cond::Ne), body_local_error);

        Ok(())
    }
} // impl Arm32Backend

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::arch::common::text_emitter::TextEmitter;

    #[test]
    fn load_byte_helper_materializes_large_offsets() {
        let mut text = TextEmitter::new();
        emit_load_byte_into(&mut text, Arm32Reg::R3, Arm32Reg::R10, 0x1234, false);
        let words: collections::Vec<u32> = text
            .finish()
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(
            words,
            collections::vec![
                enc::movw(Arm32Reg::R3, 0x1234),
                enc::add_reg(Arm32Reg::R3, Arm32Reg::R10, Arm32Reg::R3),
                enc::ldrb_imm(Arm32Reg::R3, Arm32Reg::R3, 0),
            ]
        );
    }

    #[test]
    fn store_half_helper_materializes_large_offsets() {
        let mut text = TextEmitter::new();
        let pool = ScratchPool::new([Arm32Reg::R12, Arm32Reg::R14]);
        emit_store_half_to(&mut text, &pool, Arm32Reg::R5, Arm32Reg::R10, 0x2345);
        let words: collections::Vec<u32> = text
            .finish()
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(
            words,
            collections::vec![
                enc::movw(Arm32Reg::R12, 0x2345),
                enc::add_reg(Arm32Reg::R12, Arm32Reg::R10, Arm32Reg::R12),
                enc::strh_imm(Arm32Reg::R5, Arm32Reg::R12, 0),
            ]
        );
        pool.assert_all_free();
    }

    #[test]
    fn float_compare_helpers_match_wasm_unordered_rules() {
        assert_eq!(float_compare_ordered_cond(MachineCompareKind::Eq), Cond::Eq);
        assert_eq!(float_compare_ordered_cond(MachineCompareKind::Ne), Cond::Ne);
        assert_eq!(float_compare_ordered_cond(MachineCompareKind::Lt), Cond::Mi);
        assert_eq!(float_compare_ordered_cond(MachineCompareKind::Gt), Cond::Gt);
        assert_eq!(float_compare_ordered_cond(MachineCompareKind::Le), Cond::Ls);
        assert_eq!(float_compare_ordered_cond(MachineCompareKind::Ge), Cond::Ge);

        assert!(float_compare_unordered_result(MachineCompareKind::Ne));
        for kind in [
            MachineCompareKind::Eq,
            MachineCompareKind::Lt,
            MachineCompareKind::Gt,
            MachineCompareKind::Le,
            MachineCompareKind::Ge,
        ] {
            assert!(!float_compare_unordered_result(kind));
        }
    }
}
