//! Instruction emission for the 32-bit ARM backend.
//!
//! ## Split-borrow pattern
//!
//! `prepare_gp` and `prepare_fp` are **free functions** instead of methods
//! on `Arm32Backend`. This is a deliberate Rust borrow-checker workaround:
//! the returned `PreparedGp`/`PreparedFp` holds an RAII guard that borrows
//! `&ScratchPool`. As free functions taking disjoint field references, Rust
//! can see that the guard borrows only the pool (via `Cell`), while
//! `&mut TextEmitter` is reborrowed only for the call's duration.

use crate::{
    error::WasmError,
    vm::{
        arch::common::{scratch_pool::ScratchPool, text_emitter::TextEmitter},
        machine::machine_ir::{
            fp_reg_index, MachineAddr, MachineBranchCond, MachineCallRuntime, MachineCompareKind,
            MachineConvertOp, MachineFloatBinaryOp, MachineFloatUnaryOp, MachineFloatWidth,
            MachineInst, MachineInstKind, MachineIntBinaryOp, MachineIntUnaryOp, MachineIntWidth,
            MachineLoadExtension, MachineMemWidth, MachineReg, MachineShiftOp, MachineSign,
            MachineStorageType, MachineTrapKind, MachineValue, MACHINE_CTX_REG, MACHINE_FP_REG,
        },
        runtime::{
            preserved::{io as preserved_io, op as preserved_op},
            runtime_call::call_runtime_entry_ptr,
        },
    },
};

use super::{
    abi::{map_fixed_reg, map_reg, C_ARG0, C_ARG1, C_ARG2, C_RET0},
    arm32_f32_ceil, arm32_f32_floor, arm32_f32_nearest_bits, arm32_f32_trunc, arm32_f64_ceil,
    arm32_f64_floor, arm32_f64_nearest_bits, arm32_f64_trunc, arm32_i64_div_s, arm32_i64_div_u,
    arm32_i64_rem_s, arm32_i64_rem_u, arm32_i64_rotl, arm32_i64_rotr, arm32_i64s_to_f32,
    arm32_i64s_to_f64, arm32_i64u_to_f32, arm32_i64u_to_f64, arm32_saturating_trunc,
    arm32_trapping_trunc,
    backend::{Arm32Backend, BranchFixupKind},
    enc::{self, Cond},
    operands::{OwnedPreparedGp, PreparedFp, PreparedGp},
    reg::Arm32Reg,
    select,
};

use super::abi::{fp_machine_reg, C_FP_RET0};

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

/// Emit a conditional `MOV Rd, Rm` (16-bit T1 form). A32 uses the per-
/// instruction condition field; Thumb-2 emits `IT <cond>` followed by the
/// 16-bit unconditional MOV. The IT + 16-bit MOV pair is naturally
/// 4-byte-aligned, no trailing pad needed.
pub(super) fn emit_mov_reg_cond_into(
    text: &mut TextEmitter,
    cond: enc::Cond,
    dst: Arm32Reg,
    src: Arm32Reg,
) {
    #[cfg(not(sf_arm32_isa_thumb))]
    {
        text.emit_u32(enc::mov_reg_cond(cond, dst, src));
    }
    #[cfg(sf_arm32_isa_thumb)]
    {
        if matches!(cond, enc::Cond::Al) {
            // Unconditional: fall back to the plain (padded) mov_reg.
            text.emit_u32(enc::mov_reg(dst, src));
        } else {
            // IT <cond> + 16-bit MOV Rd, Rm (T1 high-reg form) = 4 bytes total.
            text.emit_u16(enc::it(cond));
            let rd = dst.idx() as u16;
            let rm = src.idx() as u16;
            let d = (rd >> 3) & 1;
            let rd_lo = rd & 0x7;
            let mov16: u16 = 0x4600 | (d << 7) | ((rm & 0xF) << 3) | rd_lo;
            text.emit_u16(mov16);
        }
    }
}

/// Emit a conditional DP-imm instruction. Bridges A32 (per-instruction
/// condition field) and Thumb-2 (`IT` prefix before a single trailing
/// unconditional instruction). Callers use this wherever they previously
/// wrote `emit_u32(enc::dp_imm_cond(...))` — that direct form would emit
/// only the DP, missing the `IT` prefix under Thumb-2.
pub(super) fn emit_dp_imm_cond_into(
    text: &mut TextEmitter,
    cond: enc::Cond,
    opcode: u32,
    s: bool,
    dst: Arm32Reg,
    lhs: Arm32Reg,
    imm8: u32,
    rot: u32,
) {
    #[cfg(not(sf_arm32_isa_thumb))]
    {
        text.emit_u32(enc::dp_imm_cond(cond, opcode, s, dst, lhs, imm8, rot));
    }
    #[cfg(sf_arm32_isa_thumb)]
    {
        if !matches!(cond, enc::Cond::Al) {
            // IT is 2 bytes; the very next instruction is the one under
            // the IT block. Emit raw 16-bit IT, then the 32-bit DP
            // (conditional), then a trailing 2-byte NOP so the whole
            // sequence stays 4-byte-aligned for the surrounding emitter
            // (emit_nop_padding assumes 4-byte slots; sub-word tails
            // would get silently truncated at page boundaries).
            text.emit_u16(enc::it(cond));
            text.emit_u32(enc::dp_imm_cond(
                enc::Cond::Al,
                opcode,
                s,
                dst,
                lhs,
                imm8,
                rot,
            ));
            text.emit_u16(0xBF00); // NOP — outside the IT block
        } else {
            text.emit_u32(enc::dp_imm_cond(
                enc::Cond::Al,
                opcode,
                s,
                dst,
                lhs,
                imm8,
                rot,
            ));
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
        MachineValue::ReservedReg(_reg) => Err(WasmError::internal(
            "arm32 prepare_gp cannot consume reserved cache register as a real value",
        )),
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
        MachineValue::ReservedReg(_reg) => Err(WasmError::internal(
            "arm32 pair bitop rhs cannot consume reserved cache register as a real value",
        )),
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
            let fp_idx = fp_reg_index(
                reg,
                // We need the config but don't have it here; map_reg will fail
                // for FP regs so we detect via fp_machine_reg lookup.
                // Instead, we use the index computation from the backend.
                // This is a simplified path — callers pass FP regs only.
                super::abi::compile_backend_config(),
            )
            .ok_or_else(|| {
                WasmError::invalid("arm32 prepare_fp: expected FP register, got machine reg")
            })?;
            let d = fp_machine_reg(fp_idx)
                .ok_or_else(|| WasmError::invalid("arm32 prepare_fp: FP index out of range"))?;
            Ok(PreparedFp::Mapped(d))
        }
        MachineValue::ReservedReg(_reg) => Err(WasmError::internal(
            "arm32 prepare_fp cannot consume reserved cache register as a real value",
        )),
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
            MachineInstKind::Move {
                ty: MachineStorageType::V128,
                ..
            } => {
                return Err(WasmError::internal(
                    "SIMD native codegen is not implemented yet",
                ));
            }
            #[cfg(sf_has_simd)]
            MachineInstKind::V128Const { .. }
            | MachineInstKind::V128FromRaw { .. }
            | MachineInstKind::V128ToRaw { .. }
            | MachineInstKind::SimdUnary { .. }
            | MachineInstKind::SimdBinary { .. }
            | MachineInstKind::SimdTernary { .. }
            | MachineInstKind::SimdShift { .. }
            | MachineInstKind::SimdExtractLane { .. }
            | MachineInstKind::SimdReplaceLane { .. }
            | MachineInstKind::SimdShuffle { .. }
            | MachineInstKind::SimdLoad { .. }
            | MachineInstKind::SimdStore { .. }
            | MachineInstKind::SimdLoadLane { .. }
            | MachineInstKind::SimdStoreLane { .. } => {
                return Err(WasmError::internal(
                    "SIMD native codegen is not implemented yet",
                ));
            }
            MachineInstKind::Move { ty, dst, src, .. } => {
                let dst_is_fp = self.is_fp_machine_reg(*dst);
                let src_is_fp = match src {
                    MachineValue::Reg(r) => self.is_fp_machine_reg(*r),
                    MachineValue::ReservedReg(_reg) => {
                        return Err(WasmError::internal(
                            "arm32 Move cannot consume reserved cache register as source",
                        ));
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
                        MachineValue::ReservedReg(_reg) => {
                            return Err(WasmError::internal("arm32 Move GP->FP cannot consume reserved cache register as source"));
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
                        MachineValue::ReservedReg(_reg) => {
                            return Err(WasmError::internal("arm32 Move GP->GP cannot consume reserved cache register as source"));
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

            MachineInstKind::Int64MulFromSignExt32 {
                dst_lo,
                dst_hi,
                lhs,
                rhs,
            } => {
                self.compile_int64_mul_from_sign_ext32(*dst_lo, *dst_hi, lhs, rhs)?;
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

            MachineInstKind::CallRuntime(call) => {
                self.compile_call_runtime(call)?;
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
                    preserved_op::MEMORY_GROW,
                    &[(preserved_io::IMM0, *mem_idx)],
                    &[(preserved_io::ARG0, *delta)],
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
                    preserved_op::MEMORY_FILL,
                    &[(preserved_io::IMM0, *mem_idx)],
                    &[
                        (preserved_io::ARG0, *dest),
                        (preserved_io::ARG1, *val),
                        (preserved_io::ARG2, *len),
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
                    preserved_op::MEMORY_COPY,
                    &[
                        (preserved_io::IMM0, *dst_mem),
                        (preserved_io::IMM1, *src_mem),
                    ],
                    &[
                        (preserved_io::ARG0, *dest),
                        (preserved_io::ARG1, *src),
                        (preserved_io::ARG2, *len),
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
                    preserved_op::MEMORY_INIT,
                    &[
                        (preserved_io::IMM0, *mem_idx),
                        (preserved_io::IMM1, *data_idx),
                    ],
                    &[
                        (preserved_io::ARG0, *dest),
                        (preserved_io::ARG1, *src),
                        (preserved_io::ARG2, *len),
                    ],
                    None,
                )?;
            }
            MachineInstKind::DataDrop { data_idx } => {
                self.emit_preserved_helper_call(
                    preserved_op::DATA_DROP,
                    &[(preserved_io::IMM0, *data_idx)],
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
                    preserved_op::TABLE_GROW,
                    &[(preserved_io::IMM0, *table_idx)],
                    &[
                        (preserved_io::ARG0, *init_val),
                        (preserved_io::ARG1, *delta),
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
                    preserved_op::TABLE_FILL,
                    &[(preserved_io::IMM0, *table_idx)],
                    &[
                        (preserved_io::ARG0, *start),
                        (preserved_io::ARG1, *val),
                        (preserved_io::ARG2, *len),
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
                    preserved_op::TABLE_COPY,
                    &[
                        (preserved_io::IMM0, *dst_tbl),
                        (preserved_io::IMM1, *src_tbl),
                    ],
                    &[
                        (preserved_io::ARG0, *dest),
                        (preserved_io::ARG1, *src),
                        (preserved_io::ARG2, *len),
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
                    preserved_op::TABLE_INIT,
                    &[
                        (preserved_io::IMM0, *table_idx),
                        (preserved_io::IMM1, *elem_idx),
                    ],
                    &[
                        (preserved_io::ARG0, *dest),
                        (preserved_io::ARG1, *src),
                        (preserved_io::ARG2, *len),
                    ],
                    None,
                )?;
            }
            MachineInstKind::ElemDrop { elem_idx } => {
                self.emit_preserved_helper_call(
                    preserved_op::ELEM_DROP,
                    &[(preserved_io::IMM0, *elem_idx)],
                    &[],
                    None,
                )?;
            }
            MachineInstKind::EhThrow { tag_idx, args } => {
                self.emit_preserved_helper_call(
                    preserved_op::EH_THROW,
                    &[(preserved_io::IMM0, *tag_idx)],
                    &[
                        (preserved_io::ARG0, MachineValue::Reg(MACHINE_FP_REG)),
                        (preserved_io::ARG1, MachineValue::Imm64(args.start.0 as u64)),
                        (preserved_io::ARG2, MachineValue::Imm64(args.count as u64)),
                    ],
                    None,
                )?;
            }
            MachineInstKind::EhThrowRef { exnref_slot } => {
                self.emit_preserved_helper_call(
                    preserved_op::EH_THROW_REF,
                    &[],
                    &[
                        (preserved_io::ARG0, MachineValue::Reg(MACHINE_FP_REG)),
                        (
                            preserved_io::ARG1,
                            MachineValue::Imm64(exnref_slot.0 as u64),
                        ),
                    ],
                    None,
                )?;
            }
            MachineInstKind::EhAllocExnRef { tag_idx, dst } => {
                self.compile_preserved_result(
                    preserved_op::EH_ALLOC_EXN_REF,
                    *tag_idx,
                    0,
                    MachineValue::Reg(MACHINE_FP_REG),
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    MachineStorageType::GpWord,
                    *dst,
                )?;
            }
            MachineInstKind::RefFunc { func_idx, dst } => {
                self.compile_preserved_result(
                    preserved_op::REF_FUNC,
                    *func_idx,
                    0,
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    MachineStorageType::GpWord,
                    *dst,
                )?;
            }
            MachineInstKind::RefAsNonNull { src, dst } => {
                self.compile_preserved_result(
                    preserved_op::REF_AS_NON_NULL,
                    0,
                    0,
                    *src,
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    MachineStorageType::GpWord,
                    *dst,
                )?;
            }
            MachineInstKind::RefEq { lhs, rhs, dst } => {
                self.compile_preserved_result(
                    preserved_op::REF_EQ,
                    0,
                    0,
                    *lhs,
                    *rhs,
                    MachineValue::Imm64(0),
                    MachineStorageType::GpWord,
                    *dst,
                )?;
            }
            MachineInstKind::RefI31 { src, dst } => {
                self.compile_preserved_result(
                    preserved_op::REF_I31,
                    0,
                    0,
                    *src,
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    MachineStorageType::GpWord,
                    *dst,
                )?;
            }
            MachineInstKind::I31GetS { src, dst } => {
                self.compile_preserved_result(
                    preserved_op::I31_GET_S,
                    0,
                    0,
                    *src,
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    MachineStorageType::GpWord,
                    *dst,
                )?;
            }
            MachineInstKind::I31GetU { src, dst } => {
                self.compile_preserved_result(
                    preserved_op::I31_GET_U,
                    0,
                    0,
                    *src,
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    MachineStorageType::GpWord,
                    *dst,
                )?;
            }
            MachineInstKind::AnyConvertExtern { src, dst } => {
                self.compile_preserved_result(
                    preserved_op::ANY_CONVERT_EXTERN,
                    0,
                    0,
                    *src,
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    MachineStorageType::GpWord,
                    *dst,
                )?;
            }
            MachineInstKind::ExternConvertAny { src, dst } => {
                self.compile_preserved_result(
                    preserved_op::EXTERN_CONVERT_ANY,
                    0,
                    0,
                    *src,
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    MachineStorageType::GpWord,
                    *dst,
                )?;
            }
            MachineInstKind::RefTest { ref_type, src, dst } => {
                let encoded = ref_type.encode_to_u64();
                self.compile_preserved_result(
                    preserved_op::REF_TEST,
                    encoded as u32,
                    (encoded >> 32) as u32,
                    *src,
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    MachineStorageType::GpWord,
                    *dst,
                )?;
            }
            MachineInstKind::RefCast { ref_type, src, dst } => {
                let encoded = ref_type.encode_to_u64();
                self.compile_preserved_result(
                    preserved_op::REF_CAST,
                    encoded as u32,
                    (encoded >> 32) as u32,
                    *src,
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    MachineStorageType::GpWord,
                    *dst,
                )?;
            }
            MachineInstKind::StructNew {
                type_idx,
                fields,
                dst,
            } => {
                self.compile_struct_new(*type_idx, fields, *dst)?;
            }
            MachineInstKind::StructNewDefault { type_idx, dst } => {
                self.compile_preserved_result(
                    preserved_op::STRUCT_NEW_DEFAULT,
                    *type_idx,
                    0,
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    MachineStorageType::GpWord,
                    *dst,
                )?;
            }
            MachineInstKind::StructGet {
                type_idx,
                field_idx,
                signed,
                ty,
                src,
                dst,
                dst_hi,
            } => {
                self.compile_struct_get(*type_idx, *field_idx, *signed, *ty, *src, *dst, *dst_hi)?;
            }
            MachineInstKind::StructSet {
                type_idx,
                field_idx,
                ref_src,
                value_lo,
                value_hi,
            } => {
                self.compile_struct_set(*type_idx, *field_idx, *ref_src, *value_lo, *value_hi)?;
            }
            MachineInstKind::ArrayNew {
                type_idx,
                init_lo,
                init_hi,
                length,
                dst,
            } => {
                self.compile_array_new(*type_idx, *init_lo, *init_hi, *length, *dst)?;
            }
            MachineInstKind::ArrayNewDefault {
                type_idx,
                length,
                dst,
            } => {
                self.compile_preserved_result(
                    preserved_op::ARRAY_NEW_DEFAULT,
                    *type_idx,
                    0,
                    *length,
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    MachineStorageType::GpWord,
                    *dst,
                )?;
            }
            MachineInstKind::ArrayNewFixed {
                type_idx,
                elements,
                dst,
            } => {
                self.compile_array_new_fixed(*type_idx, elements, *dst)?;
            }
            MachineInstKind::ArrayNewData {
                type_idx,
                data_idx,
                src,
                len,
                dst,
            } => {
                self.compile_array_new_data(*type_idx, *data_idx, *src, *len, *dst)?;
            }
            MachineInstKind::ArrayNewElem {
                type_idx,
                elem_idx,
                src,
                len,
                dst,
            } => {
                self.compile_array_new_elem(*type_idx, *elem_idx, *src, *len, *dst)?;
            }
            MachineInstKind::ArrayGet {
                type_idx,
                signed,
                ty,
                ref_src,
                index,
                dst,
                dst_hi,
            } => {
                self.compile_array_get(*type_idx, *signed, *ty, *ref_src, *index, *dst, *dst_hi)?;
            }
            MachineInstKind::ArraySet {
                type_idx,
                ref_src,
                index,
                value_lo,
                value_hi,
            } => {
                self.compile_array_set(*type_idx, *ref_src, *index, *value_lo, *value_hi)?;
            }
            MachineInstKind::ArrayFill {
                type_idx,
                ref_src,
                index,
                value_lo,
                value_hi,
                len,
            } => {
                self.compile_array_fill(*type_idx, *ref_src, *index, *value_lo, *value_hi, *len)?;
            }
            MachineInstKind::ArrayCopy {
                dst_type_idx,
                src_type_idx,
                dst_ref,
                dst_index,
                src_ref,
                src_index,
                len,
            } => {
                self.compile_array_copy(
                    *dst_type_idx,
                    *src_type_idx,
                    *dst_ref,
                    *dst_index,
                    *src_ref,
                    *src_index,
                    *len,
                )?;
            }
            MachineInstKind::ArrayInitData {
                type_idx,
                data_idx,
                ref_src,
                dst_index,
                src_index,
                len,
            } => {
                self.compile_array_init_data(
                    *type_idx, *data_idx, *ref_src, *dst_index, *src_index, *len,
                )?;
            }
            MachineInstKind::ArrayInitElem {
                type_idx,
                elem_idx,
                ref_src,
                dst_index,
                src_index,
                len,
            } => {
                self.compile_array_init_elem(
                    *type_idx, *elem_idx, *ref_src, *dst_index, *src_index, *len,
                )?;
            }
            MachineInstKind::ArrayLen { src, dst } => {
                self.compile_preserved_result(
                    preserved_op::ARRAY_LEN,
                    0,
                    0,
                    *src,
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    MachineStorageType::GpWord,
                    *dst,
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
                    return Err(WasmError::invalid("arm32: unsupported FP load width"));
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
                            return Err(WasmError::invalid("arm32: unsupported FP store width"));
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
                        return Err(WasmError::invalid("arm32: unsupported FP store width"));
                    }
                },
                MachineValue::Reg(_r) => {
                    return Err(WasmError::invalid(
                        "arm32: FP store expects an FP register source, got GP machine reg",
                    ));
                }
                MachineValue::ReservedReg(_reg) => {
                    return Err(WasmError::internal(
                        "arm32 FP store cannot consume reserved cache register as source value",
                    ));
                }
            }
            return Ok(());
        }

        if let MachineValue::Reg(r) = src {
            if self.is_fp_machine_reg(*r) {
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
                        return Err(WasmError::invalid("arm32: unsupported FP raw store width"));
                    }
                }
                return Ok(());
            }
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
            MachineValue::ReservedReg(_reg) => {
                return Err(WasmError::internal(
                    "arm32 compile_int_binary cannot consume reserved cache register as lhs",
                ));
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
                MachineValue::ReservedReg(_reg) => {
                    return Err(WasmError::internal(
                        "arm32 int Add cannot consume reserved cache register as rhs",
                    ));
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
                MachineValue::ReservedReg(_reg) => {
                    return Err(WasmError::internal(
                        "arm32 int Sub cannot consume reserved cache register as rhs",
                    ));
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
                MachineValue::ReservedReg(_reg) => {
                    return Err(WasmError::internal(
                        "arm32 int And cannot consume reserved cache register as rhs",
                    ));
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
                MachineValue::ReservedReg(_reg) => {
                    return Err(WasmError::internal(
                        "arm32 int Or cannot consume reserved cache register as rhs",
                    ));
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
                MachineValue::ReservedReg(_reg) => {
                    return Err(WasmError::internal(
                        "arm32 int Xor cannot consume reserved cache register as rhs",
                    ));
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
                    MachineValue::ReservedReg(_reg) => {
                        return Err(WasmError::internal(
                            "arm32 int Shl cannot consume reserved cache register as rhs",
                        ));
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
                    MachineValue::ReservedReg(_reg) => {
                        return Err(WasmError::internal(
                            "arm32 int ShrU cannot consume reserved cache register as rhs",
                        ));
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
                    MachineValue::ReservedReg(_reg) => {
                        return Err(WasmError::internal(
                            "arm32 int ShrS cannot consume reserved cache register as rhs",
                        ));
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
                    MachineValue::ReservedReg(_reg) => {
                        return Err(WasmError::internal(
                            "arm32 int Rotl cannot consume reserved cache register as rhs",
                        ));
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
                    MachineValue::ReservedReg(_reg) => {
                        return Err(WasmError::internal(
                            "arm32 int Rotr cannot consume reserved cache register as rhs",
                        ));
                    }
                };
                {
                    let s = self.gp_scratch.scoped_alloc();
                    self.core.text.emit_u32(enc::and_imm(*s, rhs_hw, 31, 0));
                    self.core.text.emit_u32(enc::ror_reg(dst_hw, lhs_hw, *s));
                }
            }
            MachineIntBinaryOp::DivU => {
                let rhs_gp = prepare_gp(&mut self.core.text, &self.gp_scratch, *rhs)?.detach();
                // Trap on divide by zero
                self.core.text.emit_u32(enc::cmp_imm(*rhs_gp, 0, 0));
                let ok = self.core.new_label();
                self.emit_branch(BranchFixupKind::BCond(Cond::Ne), ok);
                let trap = self
                    .core
                    .ensure_trap_label(MachineTrapKind::IntegerDivideByZero);
                self.emit_branch(BranchFixupKind::B, trap);
                self.core.bind_label(ok);
                self.core.text.emit_u32(enc::udiv(dst_hw, lhs_hw, *rhs_gp));
            }
            MachineIntBinaryOp::DivS => {
                let rhs_gp = prepare_gp(&mut self.core.text, &self.gp_scratch, *rhs)?.detach();
                // Trap on divide by zero
                self.core.text.emit_u32(enc::cmp_imm(*rhs_gp, 0, 0));
                let not_zero = self.core.new_label();
                self.emit_branch(BranchFixupKind::BCond(Cond::Ne), not_zero);
                let trap_dz = self
                    .core
                    .ensure_trap_label(MachineTrapKind::IntegerDivideByZero);
                self.emit_branch(BranchFixupKind::B, trap_dz);
                self.core.bind_label(not_zero);
                // Trap on INT_MIN / -1 (integer overflow)
                {
                    let s = self.gp_scratch.scoped_alloc();
                    emit_load_u32_into(&mut self.core.text, *s, 0x80000000u32);
                    self.core.text.emit_u32(enc::cmp_reg(lhs_hw, *s));
                }
                let not_min = self.core.new_label();
                self.emit_branch(BranchFixupKind::BCond(Cond::Ne), not_min);
                self.core.text.emit_u32(enc::cmn_imm(*rhs_gp, 1, 0));
                let not_neg1 = self.core.new_label();
                self.emit_branch(BranchFixupKind::BCond(Cond::Ne), not_neg1);
                let trap_ov = self
                    .core
                    .ensure_trap_label(MachineTrapKind::IntegerOverflow);
                self.emit_branch(BranchFixupKind::B, trap_ov);
                self.core.bind_label(not_min);
                self.core.bind_label(not_neg1);
                self.core.text.emit_u32(enc::sdiv(dst_hw, lhs_hw, *rhs_gp));
            }
            MachineIntBinaryOp::RemU => {
                let rhs_gp = prepare_gp(&mut self.core.text, &self.gp_scratch, *rhs)?.detach();
                // Trap on divide by zero
                self.core.text.emit_u32(enc::cmp_imm(*rhs_gp, 0, 0));
                let ok = self.core.new_label();
                self.emit_branch(BranchFixupKind::BCond(Cond::Ne), ok);
                let trap = self
                    .core
                    .ensure_trap_label(MachineTrapKind::IntegerDivideByZero);
                self.emit_branch(BranchFixupKind::B, trap);
                self.core.bind_label(ok);
                // rem = lhs - (lhs / rhs) * rhs
                // UDIV quotient, lhs, rhs; MLS dst, quotient, rhs, lhs
                let quotient = self.gp_scratch.scoped_alloc();
                self.core
                    .text
                    .emit_u32(enc::udiv(*quotient, lhs_hw, *rhs_gp));
                self.core
                    .text
                    .emit_u32(enc::mls(dst_hw, *quotient, *rhs_gp, lhs_hw));
            }
            MachineIntBinaryOp::RemS => {
                let rhs_gp = prepare_gp(&mut self.core.text, &self.gp_scratch, *rhs)?.detach();
                // Trap on divide by zero
                self.core.text.emit_u32(enc::cmp_imm(*rhs_gp, 0, 0));
                let ok = self.core.new_label();
                self.emit_branch(BranchFixupKind::BCond(Cond::Ne), ok);
                let trap = self
                    .core
                    .ensure_trap_label(MachineTrapKind::IntegerDivideByZero);
                self.emit_branch(BranchFixupKind::B, trap);
                self.core.bind_label(ok);
                // ARM SDIV returns 0 for INT_MIN/-1, so MLS would give
                // INT_MIN instead of the correct 0. Guard: any x % -1 = 0.
                self.core.text.emit_u32(enc::cmn_imm(*rhs_gp, 1, 0));
                let not_neg1 = self.core.new_label();
                let done = self.core.new_label();
                self.emit_branch(BranchFixupKind::BCond(Cond::Ne), not_neg1);
                self.core.text.emit_u32(enc::mov_imm(dst_hw, 0, 0));
                self.emit_branch(BranchFixupKind::B, done);
                self.core.bind_label(not_neg1);
                // SDIV quotient, lhs, rhs; MLS dst, quotient, rhs, lhs
                let quotient = self.gp_scratch.scoped_alloc();
                self.core
                    .text
                    .emit_u32(enc::sdiv(*quotient, lhs_hw, *rhs_gp));
                self.core
                    .text
                    .emit_u32(enc::mls(dst_hw, *quotient, *rhs_gp, lhs_hw));
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
            MachineValue::ReservedReg(_reg) => {
                return Err(WasmError::internal(
                    "arm32 compile_int_unary cannot consume reserved cache register as src",
                ));
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

    /// Inline `i64.add` / `i64.sub` directly as `adds/adc` (or `subs/sbc`) on
    /// the host pair of registers. Fast path: all four halves are `Reg` and
    /// `dst_lo` doesn't clobber an input hi-half needed by the carry step.
    /// Aliased fallback: stage `dst_lo` through a scratch. `Imm64` fallback:
    /// spill shell + R0..R3 staging like before.
    ///
    /// Aliasing rule derived from operand order:
    ///   `adds dst_lo, a_lo, b_lo`   reads a_lo, b_lo, writes dst_lo + flags
    ///   `adc  dst_hi, a_hi, b_hi`   reads a_hi, b_hi, consumes flags
    /// The only harmful alias is `dst_lo == a_hi_hw` or `dst_lo == b_hi_hw`
    /// (overwrites a hi-half before ADC can read it). All other
    /// dst↔input aliases are read-before-write within a single instruction.
    fn compile_i64_pair_addsub(
        &mut self,
        dst_lo: MachineReg,
        dst_hi: MachineReg,
        lhs_lo: &MachineValue,
        lhs_hi: &MachineValue,
        rhs_lo: &MachineValue,
        rhs_hi: &MachineValue,
        is_add: bool,
    ) -> Result<(), WasmError> {
        let dst_lo_hw = map_reg(dst_lo)?;
        let dst_hi_hw = map_reg(dst_hi)?;
        let emit_lo = if is_add { enc::adds_reg } else { enc::subs_reg };
        let emit_hi = if is_add { enc::adc_reg } else { enc::sbc_reg };
        if let (
            MachineValue::Reg(a_lo),
            MachineValue::Reg(a_hi),
            MachineValue::Reg(b_lo),
            MachineValue::Reg(b_hi),
        ) = (*lhs_lo, *lhs_hi, *rhs_lo, *rhs_hi)
        {
            let a_lo_hw = map_reg(a_lo)?;
            let a_hi_hw = map_reg(a_hi)?;
            let b_lo_hw = map_reg(b_lo)?;
            let b_hi_hw = map_reg(b_hi)?;
            let clobbers_hi_input = dst_lo_hw == a_hi_hw || dst_lo_hw == b_hi_hw;
            if !clobbers_hi_input {
                let text = &mut self.core.text;
                text.emit_u32(emit_lo(dst_lo_hw, a_lo_hw, b_lo_hw));
                text.emit_u32(emit_hi(dst_hi_hw, a_hi_hw, b_hi_hw));
                return Ok(());
            }
            // dst_lo aliases a hi-half that ADC/SBC still needs. Route
            // the low-half result through a scratch so the hi-half input
            // stays live until the carry step. One extra MOV.
            let s = self.gp_scratch.scoped_alloc();
            let text = &mut self.core.text;
            text.emit_u32(emit_lo(*s, a_lo_hw, b_lo_hw));
            text.emit_u32(emit_hi(dst_hi_hw, a_hi_hw, b_hi_hw));
            text.emit_u32(enc::mov_reg(dst_lo_hw, *s));
            return Ok(());
        }
        // Imm64 fallback. Materialize each half on demand via `prepare_gp`,
        // which keeps at most two scratches (one lo pair, then one hi pair)
        // live at a time. That fits the two-slot arm32 gp_scratch pool even
        // when every half is `Imm64`. Imm materialization uses movw/movt
        // (no flag side-effects), so it is safe between `adds` and `adc`.
        //
        // Aliasing: the `adc dst_hi, lhs_hi, rhs_hi` step still needs the
        // original `lhs_hi` / `rhs_hi` phys regs live. If either lands in
        // `dst_lo_hw`, route the low result through a scratch first. Only
        // `Reg(_)` halves can alias — `Imm64` halves land in fresh scratches.
        let hi_alias = matches!(*lhs_hi, MachineValue::Reg(r) if map_reg(r)? == dst_lo_hw)
            || matches!(*rhs_hi, MachineValue::Reg(r) if map_reg(r)? == dst_lo_hw);
        // ── Low-half step ────────────────────────────────────────────────
        if hi_alias {
            let s_lo = self.gp_scratch.scoped_alloc().detach();
            {
                let lhs_lo_gp = prepare_gp(&mut self.core.text, &self.gp_scratch, *lhs_lo)?;
                let rhs_lo_gp = prepare_gp(&mut self.core.text, &self.gp_scratch, *rhs_lo)?;
                self.core
                    .text
                    .emit_u32(emit_lo(*s_lo, *lhs_lo_gp, *rhs_lo_gp));
            }
            // ── High-half step ──────────────────────────────────────────
            {
                let lhs_hi_gp = prepare_gp(&mut self.core.text, &self.gp_scratch, *lhs_hi)?;
                let rhs_hi_gp = prepare_gp(&mut self.core.text, &self.gp_scratch, *rhs_hi)?;
                self.core
                    .text
                    .emit_u32(emit_hi(dst_hi_hw, *lhs_hi_gp, *rhs_hi_gp));
            }
            // Copy the staged low result into dst_lo now that the hi-half
            // read is done. `mov_reg` preserves flags (irrelevant here).
            self.core.text.emit_u32(enc::mov_reg(dst_lo_hw, *s_lo));
        } else {
            {
                let lhs_lo_gp = prepare_gp(&mut self.core.text, &self.gp_scratch, *lhs_lo)?;
                let rhs_lo_gp = prepare_gp(&mut self.core.text, &self.gp_scratch, *rhs_lo)?;
                self.core
                    .text
                    .emit_u32(emit_lo(dst_lo_hw, *lhs_lo_gp, *rhs_lo_gp));
            }
            // Imm materialization here uses movw/movt — no flag effects —
            // so the carry from `adds/subs` survives into `adc/sbc`.
            {
                let lhs_hi_gp = prepare_gp(&mut self.core.text, &self.gp_scratch, *lhs_hi)?;
                let rhs_hi_gp = prepare_gp(&mut self.core.text, &self.gp_scratch, *rhs_hi)?;
                self.core
                    .text
                    .emit_u32(emit_hi(dst_hi_hw, *lhs_hi_gp, *rhs_hi_gp));
            }
        }
        Ok(())
    }

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
        if self.core.current_pair_hi_dead()
            && matches!(
                op,
                MachineIntBinaryOp::Add | MachineIntBinaryOp::Sub | MachineIntBinaryOp::And
            )
        {
            self.compile_int_binary(MachineIntWidth::I32, op, dst_lo, lhs_lo, rhs_lo)?;
            let _ = dst_hi;
            let _ = lhs_hi;
            let _ = rhs_hi;
            return Ok(());
        }

        match op {
            MachineIntBinaryOp::Add => {
                self.compile_i64_pair_addsub(dst_lo, dst_hi, lhs_lo, lhs_hi, rhs_lo, rhs_hi, true)
            }
            MachineIntBinaryOp::Sub => {
                self.compile_i64_pair_addsub(dst_lo, dst_hi, lhs_lo, lhs_hi, rhs_lo, rhs_hi, false)
            }
            MachineIntBinaryOp::Mul => {
                // Native 64 x 64 → 64 (truncating). Three instructions:
                //
                //   UMULL s_lo, s_hi, a_lo, b_lo
                //   MLA   s_hi, a_lo, b_hi, s_hi
                //   MLA   s_hi, a_hi, b_lo, s_hi
                //
                // plus up to two `mov_reg` to copy the scratch pair into the
                // allocator-assigned destination regs.
                //
                // Fast path (all four inputs are `Reg`): compute in the two
                // GP scratches (R12, R14). They are disjoint from the dynamic
                // reg set, so no input can alias the UMULL destination, and
                // no shell spill is needed. This is the hot path; the body
                // matches what LLVM emits for `i64::wrapping_mul` on armv7.
                //
                // Fallback (any input is `Imm64`): the immediate materialize
                // needs a scratch that would conflict with the UMULL dst
                // scratches. Stage through the shell so R0..R3 hold the args
                // and the two scratches stay free for the 64-bit product.
                let dst_lo_hw = map_reg(dst_lo)?;
                let dst_hi_hw = map_reg(dst_hi)?;
                if let (
                    MachineValue::Reg(a_lo),
                    MachineValue::Reg(a_hi),
                    MachineValue::Reg(b_lo),
                    MachineValue::Reg(b_hi),
                ) = (*lhs_lo, *lhs_hi, *rhs_lo, *rhs_hi)
                {
                    let a_lo_hw = map_reg(a_lo)?;
                    let a_hi_hw = map_reg(a_hi)?;
                    let b_lo_hw = map_reg(b_lo)?;
                    let b_hi_hw = map_reg(b_hi)?;
                    let s_lo = self.gp_scratch.scoped_alloc();
                    let s_hi = self.gp_scratch.scoped_alloc();
                    let text = &mut self.core.text;
                    text.emit_u32(enc::umull(*s_lo, *s_hi, a_lo_hw, b_lo_hw));
                    text.emit_u32(enc::mla(*s_hi, a_lo_hw, b_hi_hw, *s_hi));
                    text.emit_u32(enc::mla(*s_hi, a_hi_hw, b_lo_hw, *s_hi));
                    if dst_lo_hw != *s_lo {
                        text.emit_u32(enc::mov_reg(dst_lo_hw, *s_lo));
                    }
                    if dst_hi_hw != *s_hi {
                        text.emit_u32(enc::mov_reg(dst_hi_hw, *s_hi));
                    }
                    return Ok(());
                }
                // Imm64 fallback.
                self.spill_caller_saved_gp_regs();
                self.emit_quad_args_to_r0_r3(lhs_lo, lhs_hi, rhs_lo, rhs_hi)?;
                {
                    let s_lo = self.gp_scratch.scoped_alloc();
                    let s_hi = self.gp_scratch.scoped_alloc();
                    self.core
                        .text
                        .emit_u32(enc::umull(*s_lo, *s_hi, Arm32Reg::R0, Arm32Reg::R2));
                    self.core
                        .text
                        .emit_u32(enc::mla(*s_hi, Arm32Reg::R0, Arm32Reg::R3, *s_hi));
                    self.core
                        .text
                        .emit_u32(enc::mla(*s_hi, Arm32Reg::R1, Arm32Reg::R2, *s_hi));
                    self.core.text.emit_u32(enc::mov_reg(Arm32Reg::R0, *s_lo));
                    self.core.text.emit_u32(enc::mov_reg(Arm32Reg::R1, *s_hi));
                }
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
            _other => Err(WasmError::invalid("arm32: unsupported i64 pair binary op")),
        }
    }

    /// Single-instruction signed 32×32 → 64 multiply, used when the peephole
    /// has proven both operands are i32 sign-extended into i64 (so the full
    /// `i64 * i64` truncated to 64 bits collapses to a plain SMULL).
    ///
    /// Saves the 2× MLA correction terms and the spill ping-pong that the
    /// generic `Int64PairBinary{Mul}` path emits.
    fn compile_int64_mul_from_sign_ext32(
        &mut self,
        dst_lo: MachineReg,
        dst_hi: MachineReg,
        lhs: &MachineValue,
        rhs: &MachineValue,
    ) -> Result<(), WasmError> {
        let dst_lo_hw = map_reg(dst_lo)?;
        let dst_hi_hw = map_reg(dst_hi)?;
        debug_assert_ne!(dst_lo_hw, dst_hi_hw, "SMULL forbids RdLo == RdHi");

        // Materialize each operand into a hardware register. Reg operands
        // route directly; Imm64 stages through a detached scratch (rare —
        // constant folding usually catches `(c1 as i64) * (c2 as i64)`
        // upstream). `detach()` decouples the scratch reservation from the
        // immutable borrow of `self.gp_scratch` so we can subsequently call
        // `&mut self` methods like `materialize_gp_into`.
        let lhs_scratch_guard;
        let lhs_hw = match *lhs {
            MachineValue::Reg(r) => {
                lhs_scratch_guard = None;
                map_reg(r)?
            }
            _ => {
                let s = self.gp_scratch.scoped_alloc().detach();
                let hw = *s;
                self.materialize_gp_into(hw, lhs)?;
                lhs_scratch_guard = Some(s);
                hw
            }
        };
        let rhs_scratch_guard;
        let rhs_hw = match *rhs {
            MachineValue::Reg(r) => {
                rhs_scratch_guard = None;
                map_reg(r)?
            }
            _ => {
                let s = self.gp_scratch.scoped_alloc().detach();
                let hw = *s;
                self.materialize_gp_into(hw, rhs)?;
                rhs_scratch_guard = Some(s);
                hw
            }
        };

        self.core
            .text
            .emit_u32(enc::smull(dst_lo_hw, dst_hi_hw, lhs_hw, rhs_hw));
        // Hold the scratch guards across the emit so the underlying slot is
        // not reused before SMULL reads its operands.
        drop(lhs_scratch_guard);
        drop(rhs_scratch_guard);
        Ok(())
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
        match op {
            MachineIntUnaryOp::Clz => {
                // CMP + CLZ + CLZ + ADD #32 + MOV #0. No call, no shell —
                // inputs stay in their current physical regs, result lands
                // directly in the allocator-assigned destination.
                let src_lo_hw =
                    prepare_gp(&mut self.core.text, &self.gp_scratch, *src_lo)?.detach();
                let src_hi_hw =
                    prepare_gp(&mut self.core.text, &self.gp_scratch, *src_hi)?.detach();
                let hi_is_zero = self.core.new_label();
                let done = self.core.new_label();
                self.core.text.emit_u32(enc::cmp_imm(*src_hi_hw, 0, 0));
                self.emit_branch(BranchFixupKind::BCond(Cond::Eq), hi_is_zero);
                self.core.text.emit_u32(enc::clz(dst_lo_hw, *src_hi_hw));
                self.emit_branch(BranchFixupKind::B, done);
                self.core.bind_label(hi_is_zero);
                self.core.text.emit_u32(enc::clz(dst_lo_hw, *src_lo_hw));
                self.core
                    .text
                    .emit_u32(enc::add_imm(dst_lo_hw, dst_lo_hw, 32, 0));
                self.core.bind_label(done);
                self.core.text.emit_u32(enc::mov_imm(dst_hi_hw, 0, 0));
                Ok(())
            }
            MachineIntUnaryOp::Ctz => {
                // ctz(x) = clz(rbit(x)) per half.
                let src_lo_hw =
                    prepare_gp(&mut self.core.text, &self.gp_scratch, *src_lo)?.detach();
                let src_hi_hw =
                    prepare_gp(&mut self.core.text, &self.gp_scratch, *src_hi)?.detach();
                let lo_is_zero = self.core.new_label();
                let done = self.core.new_label();
                self.core.text.emit_u32(enc::cmp_imm(*src_lo_hw, 0, 0));
                self.emit_branch(BranchFixupKind::BCond(Cond::Eq), lo_is_zero);
                self.core.text.emit_u32(enc::rbit(dst_lo_hw, *src_lo_hw));
                self.core.text.emit_u32(enc::clz(dst_lo_hw, dst_lo_hw));
                self.emit_branch(BranchFixupKind::B, done);
                self.core.bind_label(lo_is_zero);
                self.core.text.emit_u32(enc::rbit(dst_lo_hw, *src_hi_hw));
                self.core.text.emit_u32(enc::clz(dst_lo_hw, dst_lo_hw));
                self.core
                    .text
                    .emit_u32(enc::add_imm(dst_lo_hw, dst_lo_hw, 32, 0));
                self.core.bind_label(done);
                self.core.text.emit_u32(enc::mov_imm(dst_hi_hw, 0, 0));
                Ok(())
            }
            MachineIntUnaryOp::Popcnt => {
                // Popcnt keeps the spill shell: its SWAR body needs a temp
                // reg (R3) in addition to the two scratches (R12, R14) used
                // for masks. Without the shell R3 may hold a live dynamic
                // value. Popcnt is not on any current hot path, so carrying
                // the shell overhead is an intentional trade-off.
                self.spill_caller_saved_gp_regs();
                self.emit_pair_args_to_r0_r1(src_lo, src_hi)?;
                self.emit_inline_i64_popcnt_r0_r1();
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
                Ok(())
            }
            MachineIntUnaryOp::Extend16S => {
                let src_lo_hw =
                    prepare_gp(&mut self.core.text, &self.gp_scratch, *src_lo)?.detach();
                self.core.text.emit_u32(enc::sxth(dst_lo_hw, *src_lo_hw));
                self.core
                    .text
                    .emit_u32(enc::asr_imm(dst_hi_hw, dst_lo_hw, 31));
                Ok(())
            }
            MachineIntUnaryOp::Extend32S => {
                let src_lo_hw =
                    prepare_gp(&mut self.core.text, &self.gp_scratch, *src_lo)?.detach();
                if dst_lo_hw != *src_lo_hw {
                    self.core.text.emit_u32(enc::mov_reg(dst_lo_hw, *src_lo_hw));
                }
                if self.core.current_pair_hi_dead() {
                    return Ok(());
                }
                self.core
                    .text
                    .emit_u32(enc::asr_imm(dst_hi_hw, dst_lo_hw, 31));
                Ok(())
            }
        }
    }

    /// Inline 64-bit POPCNT using the classic SWAR sequence on each 32-bit
    /// half, then summing. Inputs in R0 (lo), R1 (hi); output in R0:R1.
    ///
    /// Masks are shared across the two halves to amortize the 32-bit constant
    /// loads (movw + movt each). `R12` holds `0x55555555`, `R14` holds
    /// `0x33333333` then is rewritten to `0x0F0F0F0F`. `R3` is the per-step
    /// scratch. The final byte-sum uses add-shifted-register forms, avoiding
    /// a `0x01010101` constant.
    fn emit_inline_i64_popcnt_r0_r1(&mut self) {
        const R0: Arm32Reg = Arm32Reg::R0;
        const R1: Arm32Reg = Arm32Reg::R1;
        const R3: Arm32Reg = Arm32Reg::R3;
        const SH_LSR: u32 = 0b01;
        let m_55 = self.gp_scratch.scoped_alloc().detach();
        let m_33 = self.gp_scratch.scoped_alloc().detach();
        let text = &mut self.core.text;
        emit_load_u32_into(text, *m_55, 0x5555_5555);
        emit_load_u32_into(text, *m_33, 0x3333_3333);

        // Per-half, in place: lo in R0, hi in R1.
        for w in [R0, R1] {
            // w = w - ((w >> 1) & 0x55555555)
            text.emit_u32(enc::lsr_imm(R3, w, 1));
            text.emit_u32(enc::and_reg(R3, R3, *m_55));
            text.emit_u32(enc::sub_reg(w, w, R3));
            // w = (w & 0x33333333) + ((w >> 2) & 0x33333333)
            text.emit_u32(enc::and_reg(R3, w, *m_33));
            text.emit_u32(enc::lsr_imm(w, w, 2));
            text.emit_u32(enc::and_reg(w, w, *m_33));
            text.emit_u32(enc::add_reg(w, w, R3));
        }

        // Reload m_33 as the 0x0F0F0F0F mask for step 3.
        emit_load_u32_into(text, *m_33, 0x0F0F_0F0F);

        for w in [R0, R1] {
            // w = (w + (w >> 4)) & 0x0F0F0F0F
            text.emit_u32(enc::lsr_imm(R3, w, 4));
            text.emit_u32(enc::add_reg(w, w, R3));
            text.emit_u32(enc::and_reg(w, w, *m_33));
            // w = low 8 of (w + (w >> 8) + (w >> 16) + (w >> 24))
            text.emit_u32(enc::add_reg_shifted(w, w, w, SH_LSR, 8));
            text.emit_u32(enc::add_reg_shifted(w, w, w, SH_LSR, 16));
            text.emit_u32(enc::and_imm(w, w, 0xFF, 0));
        }

        // Sum the two per-half counts into R0, zero out R1.
        text.emit_u32(enc::add_reg(R0, R0, R1));
        text.emit_u32(enc::mov_imm(R1, 0, 0));
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

        // Fast path: when both operands fit in 32 bits in the relevant sign
        // interpretation, we can do the whole div/rem natively with
        // UDIV/SDIV + MLS (for rem). Eligibility:
        //   Unsigned: lhs_hi == 0 && rhs_hi == 0       → both < 2^32
        //   Signed:   lhs_hi == (lhs_lo >>s 31) AND
        //             rhs_hi == (rhs_lo >>s 31)         → both fit in i32
        //
        // This covers the common case where a 64-bit local holds an i32-sized
        // value (most arithmetic over 64-bit locals with small counters or
        // pointers). The slow helper path still handles the genuinely-wide
        // case.
        let fast_done = self.core.new_label();
        let slow_path = self.core.new_label();
        match sign {
            MachineSign::Unsigned => {
                // (lhs_hi | rhs_hi) != 0  →  slow
                let s = self.gp_scratch.scoped_alloc();
                self.core
                    .text
                    .emit_u32(enc::orr_reg(*s, Arm32Reg::R1, Arm32Reg::R3));
                self.core.text.emit_u32(enc::cmp_imm(*s, 0, 0));
            }
            MachineSign::Signed => {
                // (lhs_hi ^ (lhs_lo >>s 31)) | (rhs_hi ^ (rhs_lo >>s 31)) != 0 → slow
                let t1 = self.gp_scratch.scoped_alloc();
                let t2 = self.gp_scratch.scoped_alloc();
                self.core.text.emit_u32(enc::asr_imm(*t1, Arm32Reg::R0, 31));
                self.core
                    .text
                    .emit_u32(enc::eor_reg(*t1, *t1, Arm32Reg::R1));
                self.core.text.emit_u32(enc::asr_imm(*t2, Arm32Reg::R2, 31));
                self.core
                    .text
                    .emit_u32(enc::eor_reg(*t2, *t2, Arm32Reg::R3));
                self.core.text.emit_u32(enc::orr_reg(*t1, *t1, *t2));
                self.core.text.emit_u32(enc::cmp_imm(*t1, 0, 0));
            }
        }
        self.emit_branch(BranchFixupKind::BCond(Cond::Ne), slow_path);

        // Native fast path: compute quot/rem in R0, fill R1 with sign/zero,
        // then fall through to the result-copy phase.
        {
            let saved_dividend = self.gp_scratch.scoped_alloc().detach();
            self.core
                .text
                .emit_u32(enc::mov_reg(*saved_dividend, Arm32Reg::R0));
            match sign {
                MachineSign::Unsigned => {
                    self.core
                        .text
                        .emit_u32(enc::udiv(Arm32Reg::R0, Arm32Reg::R0, Arm32Reg::R2));
                }
                MachineSign::Signed => {
                    self.core
                        .text
                        .emit_u32(enc::sdiv(Arm32Reg::R0, Arm32Reg::R0, Arm32Reg::R2));
                }
            }
            if rem {
                // R0 = saved - quotient * rhs_lo
                self.core.text.emit_u32(enc::mls(
                    Arm32Reg::R0,
                    Arm32Reg::R0,
                    Arm32Reg::R2,
                    *saved_dividend,
                ));
            }
            match sign {
                MachineSign::Unsigned => {
                    self.core.text.emit_u32(enc::mov_imm(Arm32Reg::R1, 0, 0));
                }
                MachineSign::Signed => {
                    // Sign-extend the 32-bit result into the high word.
                    self.core
                        .text
                        .emit_u32(enc::asr_imm(Arm32Reg::R1, Arm32Reg::R0, 31));
                }
            }
        }
        self.emit_pair_results_from_r0_r1(dst_lo, dst_hi)?;
        self.emit_branch(BranchFixupKind::B, fast_done);

        // Slow helper path.
        self.core.bind_label(slow_path);
        self.emit_host_call(match (sign, rem) {
            (MachineSign::Signed, false) => arm32_i64_div_s as *const () as usize,
            (MachineSign::Unsigned, false) => arm32_i64_div_u as *const () as usize,
            (MachineSign::Signed, true) => arm32_i64_rem_s as *const () as usize,
            (MachineSign::Unsigned, true) => arm32_i64_rem_u as *const () as usize,
        });
        self.emit_pair_results_from_r0_r1(dst_lo, dst_hi)?;

        self.core.bind_label(fast_done);
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

        // Fast path: constant shift count folds to a 2–4 instruction template
        // with no helper call, no spill, no stack staging. Wasm only observes
        // the low 6 bits for i64 shift counts, and rotl/rotr have the same
        // modular behavior, so mask early.
        if let MachineValue::Imm64(count) = *rhs {
            let n = (count as u32) & 63;
            return self.emit_i64_pair_shift_const(op, dst_lo_hw, dst_hi_hw, lhs_lo, lhs_hi, n);
        }

        // Variable-count native path for Shl / ShrS / ShrU. Leans on ARM's
        // "shift by register saturates at count ≥ 32" semantics for LSL/LSR
        // to cover the wide-shift case without a branch. ShrS is branched
        // because ASR replicates the sign instead of saturating to zero.
        if matches!(
            op,
            MachineIntBinaryOp::Shl | MachineIntBinaryOp::ShrS | MachineIntBinaryOp::ShrU
        ) {
            return self.emit_i64_pair_shift_variable(op, dst_lo, dst_hi, lhs_lo, lhs_hi, rhs);
        }

        // Remaining: Rotl / Rotr with a register-valued count fall back to
        // the existing helper call path. Register-count rotates are rare in
        // practice, and the inline sequence would not be a clear win over
        // the helper once icache pressure is accounted for.
        self.spill_caller_saved_gp_regs();
        self.emit_values_to_regs_via_stack(
            &[Arm32Reg::R0, Arm32Reg::R1, Arm32Reg::R2],
            &[lhs_lo, lhs_hi, rhs],
        )?;
        self.emit_host_call(match op {
            MachineIntBinaryOp::Rotl => arm32_i64_rotl as *const () as usize,
            MachineIntBinaryOp::Rotr => arm32_i64_rotr as *const () as usize,
            _other => {
                return Err(WasmError::invalid("arm32: unsupported i64 pair shift op"));
            }
        });
        self.emit_pair_results_from_r0_r1(dst_lo, dst_hi)?;
        self.restore_caller_saved_gp_regs(&[dst_lo_hw, dst_hi_hw]);
        Ok(())
    }

    /// Emit the register-count i64 pair shift for Shl / ShrS / ShrU directly
    /// as ARM instructions. Uses the existing spill + stack-stage shell to
    /// land `lhs_lo`, `lhs_hi`, `cnt` in R0, R1, R2, then emits the inline
    /// sequence into R0:R1 (result pair) and routes it back to the
    /// allocated destinations via `emit_pair_results_from_r0_r1`.
    ///
    /// The inline sequences exploit ARM shift-by-register semantics:
    /// LSL/LSR produce 0 when the count is ≥ 32, ASR replicates the sign bit.
    /// That lets Shl and ShrU be fully branch-free; ShrS needs a single
    /// compare-and-branch because ASR's saturating behavior disagrees with
    /// what the "wide shift" arm needs.
    fn emit_i64_pair_shift_variable(
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
        self.emit_values_to_regs_via_stack(
            &[Arm32Reg::R0, Arm32Reg::R1, Arm32Reg::R2],
            &[lhs_lo, lhs_hi, rhs],
        )?;
        // Post-stage register roles:
        //   R0 = lhs_lo, R1 = lhs_hi, R2 = count.
        //   R3 is caller-saved (now spilled) → available as a free temp.
        //   R12 / R14 are the gp_scratch pool — also free across this path.
        const R0: Arm32Reg = Arm32Reg::R0;
        const R1: Arm32Reg = Arm32Reg::R1;
        const R2: Arm32Reg = Arm32Reg::R2;
        const R3: Arm32Reg = Arm32Reg::R3;
        let s_nr = self.gp_scratch.scoped_alloc().detach(); // 32 - cnt
        let s_m = self.gp_scratch.scoped_alloc().detach(); // cnt - 32

        match op {
            MachineIntBinaryOp::Shl => {
                let text = &mut self.core.text;
                text.emit_u32(enc::and_imm(R2, R2, 63, 0));
                text.emit_u32(enc::rsb_imm(*s_nr, R2, 32, 0));
                text.emit_u32(enc::sub_imm(*s_m, R2, 32, 0));
                // R3 = hi << cnt (saturates to 0 when cnt ≥ 32)
                text.emit_u32(enc::lsl_reg(R3, R1, R2));
                // *s_nr = lo >> (32 - cnt) — contribution to hi for 0 < cnt < 32.
                // At cnt == 0: shift by 32 → 0. At cnt ≥ 32: shift by (negative
                // wrapped to 32+) → 0. So *s_nr is 0 outside [1..31].
                text.emit_u32(enc::lsr_reg(*s_nr, R0, *s_nr));
                // *s_m = lo << (cnt - 32) — contribution to hi for cnt ≥ 32.
                // At cnt < 32: shift by a negative-wrapped count ≥ 224 → 0.
                text.emit_u32(enc::lsl_reg(*s_m, R0, *s_m));
                text.emit_u32(enc::orr_reg(R3, R3, *s_nr));
                text.emit_u32(enc::orr_reg(R1, R3, *s_m));
                // result_lo = lo << cnt (saturates to 0 when cnt ≥ 32).
                text.emit_u32(enc::lsl_reg(R0, R0, R2));
            }
            MachineIntBinaryOp::ShrU => {
                let text = &mut self.core.text;
                text.emit_u32(enc::and_imm(R2, R2, 63, 0));
                text.emit_u32(enc::rsb_imm(*s_nr, R2, 32, 0));
                text.emit_u32(enc::sub_imm(*s_m, R2, 32, 0));
                // R3 = lo >> cnt (saturates to 0 when cnt ≥ 32).
                text.emit_u32(enc::lsr_reg(R3, R0, R2));
                // *s_nr = hi << (32 - cnt) — contribution to lo for 0 < cnt < 32.
                text.emit_u32(enc::lsl_reg(*s_nr, R1, *s_nr));
                // *s_m = hi >> (cnt - 32) — contribution to lo for cnt ≥ 32.
                text.emit_u32(enc::lsr_reg(*s_m, R1, *s_m));
                text.emit_u32(enc::orr_reg(R3, R3, *s_nr));
                text.emit_u32(enc::orr_reg(R0, R3, *s_m));
                // result_hi = hi >> cnt (saturates to 0 when cnt ≥ 32).
                text.emit_u32(enc::lsr_reg(R1, R1, R2));
            }
            MachineIntBinaryOp::ShrS => {
                let small = self.core.new_label();
                let done = self.core.new_label();
                {
                    let text = &mut self.core.text;
                    text.emit_u32(enc::and_imm(R2, R2, 63, 0));
                    text.emit_u32(enc::rsb_imm(*s_nr, R2, 32, 0));
                    text.emit_u32(enc::cmp_imm(R2, 32, 0));
                }
                // cnt < 32 → small path. CC (unsigned lower) matches cnt < 32
                // after CMP of an unsigned-masked count.
                self.emit_branch(BranchFixupKind::BCond(Cond::Cc), small);
                // Wide-shift path: cnt in 32..64.
                // result_lo = hi >>s (cnt - 32)
                // result_hi = hi >>s 31 (sign fill)
                {
                    let text = &mut self.core.text;
                    text.emit_u32(enc::sub_imm(*s_m, R2, 32, 0));
                    text.emit_u32(enc::asr_reg(R0, R1, *s_m));
                    text.emit_u32(enc::asr_imm(R1, R1, 31));
                }
                self.emit_branch(BranchFixupKind::B, done);
                // Narrow-shift path: cnt in 0..32.
                // result_lo = (lo >> cnt) | (hi << (32 - cnt))
                // result_hi = hi >>s cnt
                self.core.bind_label(small);
                {
                    let text = &mut self.core.text;
                    text.emit_u32(enc::lsr_reg(R3, R0, R2));
                    text.emit_u32(enc::lsl_reg(*s_nr, R1, *s_nr));
                    text.emit_u32(enc::orr_reg(R3, R3, *s_nr));
                    text.emit_u32(enc::asr_reg(R1, R1, R2));
                    text.emit_u32(enc::mov_reg(R0, R3));
                }
                self.core.bind_label(done);
            }
            _ => unreachable!("variable-count shift dispatch covered only Shl/ShrS/ShrU"),
        }

        self.emit_pair_results_from_r0_r1(dst_lo, dst_hi)?;
        self.restore_caller_saved_gp_regs(&[dst_lo_hw, dst_hi_hw]);
        Ok(())
    }

    /// Inline shift/rotate of a 64-bit pair by a compile-time constant count
    /// `n` in `0..64`. Emits straight-line code with no call, no spill, and
    /// at most one scratch reg per aliasing input. See header comments on each
    /// arm for the exact sequence.
    fn emit_i64_pair_shift_const(
        &mut self,
        op: MachineIntBinaryOp,
        dst_lo: Arm32Reg,
        dst_hi: Arm32Reg,
        lhs_lo: &MachineValue,
        lhs_hi: &MachineValue,
        n: u32,
    ) -> Result<(), WasmError> {
        debug_assert!(n < 64, "shift count must be pre-masked to 0..64");
        if self.core.current_pair_hi_dead()
            && matches!(op, MachineIntBinaryOp::ShrU | MachineIntBinaryOp::ShrS)
            && (1..32).contains(&n)
        {
            const SH_LSL: u32 = 0b00;
            let lhs_lo_gp = prepare_gp(&mut self.core.text, &self.gp_scratch, *lhs_lo)?.detach();
            let lhs_hi_gp = prepare_pair_bitop_rhs(
                &mut self.core.text,
                &self.gp_scratch,
                *lhs_hi,
                dst_lo,
                dst_lo,
            )?;
            self.core.text.emit_u32(enc::lsr_imm(dst_lo, *lhs_lo_gp, n));
            self.core.text.emit_u32(enc::orr_reg_shifted(
                dst_lo,
                dst_lo,
                *lhs_hi_gp,
                SH_LSL,
                32 - n,
            ));
            return Ok(());
        }

        // Snapshot each lhs half if it aliases either destination reg. The
        // shift sequences below read lhs_lo and lhs_hi after writing dst, so
        // an unsnapshotted alias would see the post-write value. Non-aliased
        // inputs stay in their current physical register with no move.
        let lhs_lo_gp = prepare_pair_bitop_rhs(
            &mut self.core.text,
            &self.gp_scratch,
            *lhs_lo,
            dst_lo,
            dst_hi,
        )?;
        let lhs_hi_gp = prepare_pair_bitop_rhs(
            &mut self.core.text,
            &self.gp_scratch,
            *lhs_hi,
            dst_lo,
            dst_hi,
        )?;
        let src_lo = *lhs_lo_gp;
        let src_hi = *lhs_hi_gp;

        const SH_LSL: u32 = 0b00;
        const SH_LSR: u32 = 0b01;
        const SH_ASR: u32 = 0b10;

        match op {
            MachineIntBinaryOp::Shl => {
                self.emit_i64_shl_const(dst_lo, dst_hi, src_lo, src_hi, n);
            }
            MachineIntBinaryOp::ShrS => {
                self.emit_i64_shr_const(dst_lo, dst_hi, src_lo, src_hi, n, SH_ASR);
            }
            MachineIntBinaryOp::ShrU => {
                self.emit_i64_shr_const(dst_lo, dst_hi, src_lo, src_hi, n, SH_LSR);
            }
            MachineIntBinaryOp::Rotl => {
                self.emit_i64_rot_const(dst_lo, dst_hi, src_lo, src_hi, n);
            }
            MachineIntBinaryOp::Rotr => {
                // rotr(n) == rotl(64 - n). The normalization is cheap and lets
                // us share one code path.
                let rotl_equiv = if n == 0 { 0 } else { 64 - n };
                self.emit_i64_rot_const(dst_lo, dst_hi, src_lo, src_hi, rotl_equiv);
            }
            _ => {
                return Err(WasmError::invalid(
                    "arm32: unsupported i64 pair shift op in const path",
                ));
            }
        }
        let _ = SH_LSL;
        Ok(())
    }

    /// Emit `dst_hi:dst_lo = (src_hi:src_lo) << n` for `n in 0..64`.
    fn emit_i64_shl_const(
        &mut self,
        dst_lo: Arm32Reg,
        dst_hi: Arm32Reg,
        src_lo: Arm32Reg,
        src_hi: Arm32Reg,
        n: u32,
    ) {
        const SH_LSL: u32 = 0b00;
        const SH_LSR: u32 = 0b01;
        let text = &mut self.core.text;
        if n == 0 {
            if dst_lo != src_lo {
                text.emit_u32(enc::mov_reg(dst_lo, src_lo));
            }
            if dst_hi != src_hi {
                text.emit_u32(enc::mov_reg(dst_hi, src_hi));
            }
            return;
        }
        if n == 32 {
            text.emit_u32(enc::mov_reg(dst_hi, src_lo));
            text.emit_u32(enc::mov_imm(dst_lo, 0, 0));
            return;
        }
        if n < 32 {
            // dst_hi = (src_hi << n) | (src_lo >> (32 - n))
            // dst_lo = src_lo << n
            // Compute dst_hi first so we can safely overwrite dst_lo = src_lo case.
            text.emit_u32(enc::lsl_imm(dst_hi, src_hi, n));
            text.emit_u32(enc::orr_reg_shifted(dst_hi, dst_hi, src_lo, SH_LSR, 32 - n));
            text.emit_u32(enc::lsl_imm(dst_lo, src_lo, n));
        } else {
            // n in 33..64
            text.emit_u32(enc::lsl_imm(dst_hi, src_lo, n - 32));
            text.emit_u32(enc::mov_imm(dst_lo, 0, 0));
        }
        let _ = SH_LSL;
    }

    /// Emit `dst_hi:dst_lo = (src_hi:src_lo) >> n` for `n in 0..64`, where
    /// `hi_shift` is the shift kind used on the high half (`SH_LSR` for shr_u,
    /// `SH_ASR` for shr_s).
    fn emit_i64_shr_const(
        &mut self,
        dst_lo: Arm32Reg,
        dst_hi: Arm32Reg,
        src_lo: Arm32Reg,
        src_hi: Arm32Reg,
        n: u32,
        hi_shift: u32,
    ) {
        const SH_LSL: u32 = 0b00;
        const SH_LSR: u32 = 0b01;
        const SH_ASR: u32 = 0b10;
        debug_assert!(hi_shift == SH_LSR || hi_shift == SH_ASR);
        let text = &mut self.core.text;
        if n == 0 {
            if dst_lo != src_lo {
                text.emit_u32(enc::mov_reg(dst_lo, src_lo));
            }
            if dst_hi != src_hi {
                text.emit_u32(enc::mov_reg(dst_hi, src_hi));
            }
            return;
        }
        if n == 32 {
            // dst_lo = src_hi; dst_hi = src_hi sign/zero fill.
            //
            // Write dst_hi first since dst_lo may alias src_hi — otherwise the
            // second instruction would read the just-clobbered src_hi.
            if hi_shift == SH_ASR {
                text.emit_u32(enc::asr_imm(dst_hi, src_hi, 31));
            } else {
                text.emit_u32(enc::mov_imm(dst_hi, 0, 0));
            }
            if dst_lo != src_hi {
                text.emit_u32(enc::mov_reg(dst_lo, src_hi));
            }
            return;
        }
        if n < 32 {
            // dst_lo = (src_lo >> n) | (src_hi << (32 - n))
            // dst_hi = src_hi >>{s|u} n
            // Compute dst_lo first since its sources are src_lo and src_hi, and
            // the dst_hi write only consumes src_hi which we've already read.
            text.emit_u32(enc::lsr_imm(dst_lo, src_lo, n));
            text.emit_u32(enc::orr_reg_shifted(dst_lo, dst_lo, src_hi, SH_LSL, 32 - n));
            if hi_shift == SH_ASR {
                text.emit_u32(enc::asr_imm(dst_hi, src_hi, n));
            } else {
                text.emit_u32(enc::lsr_imm(dst_hi, src_hi, n));
            }
        } else {
            // n in 33..64
            // dst_lo = src_hi >>{s|u} (n - 32)
            // dst_hi = src_hi sign/zero fill
            //
            // Write dst_hi first; it may alias src_hi and we still need src_hi
            // for dst_lo.
            let m = n - 32;
            if hi_shift == SH_ASR {
                // dst_hi = src_hi >>s 31 (sign fill)
                // For signed, dst_lo also needs ASR with (n-32), using src_hi.
                // Sequence: dst_lo = src_hi ASR m ; dst_hi = src_hi ASR 31.
                // But if dst_lo aliases src_hi, reading src_hi for dst_hi after
                // writing dst_lo would be wrong. prepare_pair_bitop_rhs already
                // snapshotted src_hi to an owned scratch if it aliased dst_lo,
                // so src_hi is safe after dst_lo is written.
                text.emit_u32(enc::asr_imm(dst_lo, src_hi, m));
                text.emit_u32(enc::asr_imm(dst_hi, src_hi, 31));
            } else {
                text.emit_u32(enc::lsr_imm(dst_lo, src_hi, m));
                text.emit_u32(enc::mov_imm(dst_hi, 0, 0));
            }
        }
    }

    /// Emit `dst_hi:dst_lo = rotate_left((src_hi:src_lo), n)` for `n in 0..64`.
    fn emit_i64_rot_const(
        &mut self,
        dst_lo: Arm32Reg,
        dst_hi: Arm32Reg,
        src_lo: Arm32Reg,
        src_hi: Arm32Reg,
        n: u32,
    ) {
        const SH_LSL: u32 = 0b00;
        const SH_LSR: u32 = 0b01;
        let text = &mut self.core.text;
        if n == 0 {
            if dst_lo != src_lo {
                text.emit_u32(enc::mov_reg(dst_lo, src_lo));
            }
            if dst_hi != src_hi {
                text.emit_u32(enc::mov_reg(dst_hi, src_hi));
            }
            return;
        }
        if n == 32 {
            // Swap halves. Handle all aliasing cases via the gp_scratch pool.
            if dst_lo == src_lo && dst_hi == src_hi {
                // dst_lo = src_hi, dst_hi = src_lo, but both dsts alias sources.
                let s = self.gp_scratch.scoped_alloc();
                text.emit_u32(enc::mov_reg(*s, src_lo));
                text.emit_u32(enc::mov_reg(dst_lo, src_hi));
                text.emit_u32(enc::mov_reg(dst_hi, *s));
                return;
            }
            if dst_lo == src_hi {
                // Must write dst_hi first (dst_hi = src_lo), then dst_lo = src_hi (stale);
                // so stage src_hi first.
                let s = self.gp_scratch.scoped_alloc();
                text.emit_u32(enc::mov_reg(*s, src_hi));
                text.emit_u32(enc::mov_reg(dst_hi, src_lo));
                text.emit_u32(enc::mov_reg(dst_lo, *s));
                return;
            }
            // dst_hi may alias src_lo; safe ordering: dst_lo = src_hi, then dst_hi = src_lo.
            text.emit_u32(enc::mov_reg(dst_lo, src_hi));
            if dst_hi != src_lo {
                text.emit_u32(enc::mov_reg(dst_hi, src_lo));
            }
            return;
        }
        // Normalize: if n >= 32, swap the source pair and reduce n by 32.
        let (src_lo, src_hi, m) = if n >= 32 {
            (src_hi, src_lo, n - 32)
        } else {
            (src_lo, src_hi, n)
        };
        // 1 <= m < 32
        // dst_lo = (src_lo << m) | (src_hi >> (32 - m))
        // dst_hi = (src_hi << m) | (src_lo >> (32 - m))
        // Both writes read src_lo and src_hi. If either dst aliases src, we
        // need to sequence carefully. prepare_pair_bitop_rhs already snapshotted
        // any src half that aliases a dst half, so src_lo and src_hi here refer
        // to values that are safe to read after either dst write.
        text.emit_u32(enc::lsl_imm(dst_lo, src_lo, m));
        text.emit_u32(enc::orr_reg_shifted(dst_lo, dst_lo, src_hi, SH_LSR, 32 - m));
        text.emit_u32(enc::lsl_imm(dst_hi, src_hi, m));
        text.emit_u32(enc::orr_reg_shifted(dst_hi, dst_hi, src_lo, SH_LSR, 32 - m));
        let _ = SH_LSL;
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
            (MachineFloatWidth::F32, MachineSign::Signed) => {
                arm32_i64s_to_f32 as *const () as usize
            }
            (MachineFloatWidth::F32, MachineSign::Unsigned) => {
                arm32_i64u_to_f32 as *const () as usize
            }
            (MachineFloatWidth::F64, MachineSign::Signed) => {
                arm32_i64s_to_f64 as *const () as usize
            }
            (MachineFloatWidth::F64, MachineSign::Unsigned) => {
                arm32_i64u_to_f64 as *const () as usize
            }
        });

        match width {
            MachineFloatWidth::F32 => {
                let dst_s = self.map_fp_dreg(dst)? * 2;
                let s0 = C_FP_RET0 * 2;
                if dst_s != s0 {
                    self.core.text.emit_u32(enc::vmov_s(dst_s, s0));
                }
            }
            MachineFloatWidth::F64 => {
                let dst_d = self.map_fp_dreg(dst)?;
                if dst_d != C_FP_RET0 {
                    self.core.text.emit_u32(enc::vmov_d(dst_d, C_FP_RET0));
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
            let s0 = C_FP_RET0 * 2;
            if src_s != s0 {
                self.core.text.emit_u32(enc::vmov_s(s0, src_s));
            }
            self.core.text.emit_u32(enc::vmov_r_s(Arm32Reg::R0, s0));
            emit_load_u32_into(&mut self.core.text, Arm32Reg::R1, 0);
        } else {
            if *src_d != C_FP_RET0 {
                self.core.text.emit_u32(enc::vmov_d(C_FP_RET0, *src_d));
            }
            self.core
                .text
                .emit_u32(enc::vmov_rr_d(Arm32Reg::R0, Arm32Reg::R1, C_FP_RET0));
        }
        self.emit_load_u32(Arm32Reg::R2, select::convert_op_code(op));

        if matches!(
            op,
            MachineConvertOp::I64TruncSatF32S
                | MachineConvertOp::I64TruncSatF32U
                | MachineConvertOp::I64TruncSatF64S
                | MachineConvertOp::I64TruncSatF64U
        ) {
            self.emit_host_call(arm32_saturating_trunc as *const () as usize);
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
        self.emit_host_call(arm32_trapping_trunc as *const () as usize);
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
            let s0 = C_FP_RET0 * 2;
            if src_s != s0 {
                self.core.text.emit_u32(enc::vmov_s(s0, src_s));
            }
            self.core.text.emit_u32(enc::vmov_r_s(Arm32Reg::R0, s0));
            emit_load_u32_into(&mut self.core.text, Arm32Reg::R1, 0);
        } else {
            if *src_d != C_FP_RET0 {
                self.core.text.emit_u32(enc::vmov_d(C_FP_RET0, *src_d));
            }
            self.core
                .text
                .emit_u32(enc::vmov_rr_d(Arm32Reg::R0, Arm32Reg::R1, C_FP_RET0));
        }
        self.emit_load_u32(Arm32Reg::R2, select::convert_op_code(op));

        if matches!(
            op,
            MachineConvertOp::I32TruncSatF32S
                | MachineConvertOp::I32TruncSatF32U
                | MachineConvertOp::I32TruncSatF64S
                | MachineConvertOp::I32TruncSatF64U
        ) {
            self.emit_host_call(arm32_saturating_trunc as *const () as usize);
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
        self.emit_host_call(arm32_trapping_trunc as *const () as usize);
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
            MachineValue::ReservedReg(_reg) => {
                return Err(WasmError::internal("arm32 reinterpret_f64_to_i64_pair cannot consume reserved cache register as src"));
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
        // `vmov_d_rr` accepts any two GP regs for the two source halves,
        // so we can feed it straight from `prepare_gp` output. Worst case
        // is both halves `Imm64`, which needs two scratches and matches the
        // two-slot gp_scratch pool exactly — no caller-saved shell needed.
        let dd = self.map_fp_dreg(dst)?;
        let src_lo_gp = prepare_gp(&mut self.core.text, &self.gp_scratch, *src_lo)?;
        let src_hi_gp = prepare_gp(&mut self.core.text, &self.gp_scratch, *src_hi)?;
        self.core
            .text
            .emit_u32(enc::vmov_d_rr(dd, *src_lo_gp, *src_hi_gp));
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
                // F32 Copysign extracts two 32-bit FP bit patterns into GP,
                // masks sign bits, ORs them, and writes back to the FP dst.
                // That only needs two GP temporaries — a perfect fit for the
                // two-slot gp_scratch pool, no caller-saved shell.
                let (sign_imm8, sign_rot) = enc::encode_arm_imm(0x8000_0000).unwrap();
                let sdn = *dn * 2;
                let sdm = *dm * 2;
                let sdd = dd * 2;
                let gp_lhs = self.gp_scratch.scoped_alloc();
                let gp_rhs = self.gp_scratch.scoped_alloc();
                let text = &mut self.core.text;
                text.emit_u32(enc::vmov_r_s(*gp_lhs, sdn)); // lhs bits
                text.emit_u32(enc::vmov_r_s(*gp_rhs, sdm)); // rhs bits
                text.emit_u32(enc::bic_imm(*gp_lhs, *gp_lhs, sign_imm8, sign_rot));
                text.emit_u32(enc::and_imm(*gp_rhs, *gp_rhs, sign_imm8, sign_rot));
                text.emit_u32(enc::orr_reg(*gp_lhs, *gp_lhs, *gp_rhs));
                text.emit_u32(enc::vmov_s_r(sdd, *gp_lhs));
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
        // scratch slots; arm32 only has two in the pool, so we serialise
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
                if *dm != C_FP_RET0 {
                    self.core.text.emit_u32(enc::vmov_d(C_FP_RET0, *dm));
                }
                self.emit_host_call(arm32_f64_ceil as *const () as usize);
                if dd != C_FP_RET0 {
                    self.core.text.emit_u32(enc::vmov_d(dd, C_FP_RET0));
                }
                self.restore_caller_saved_gp_regs(&[]);
            }
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Floor) => {
                self.spill_caller_saved_gp_regs();
                if *dm != C_FP_RET0 {
                    self.core.text.emit_u32(enc::vmov_d(C_FP_RET0, *dm));
                }
                self.emit_host_call(arm32_f64_floor as *const () as usize);
                if dd != C_FP_RET0 {
                    self.core.text.emit_u32(enc::vmov_d(dd, C_FP_RET0));
                }
                self.restore_caller_saved_gp_regs(&[]);
            }
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Trunc) => {
                self.spill_caller_saved_gp_regs();
                if *dm != C_FP_RET0 {
                    self.core.text.emit_u32(enc::vmov_d(C_FP_RET0, *dm));
                }
                self.emit_host_call(arm32_f64_trunc as *const () as usize);
                if dd != C_FP_RET0 {
                    self.core.text.emit_u32(enc::vmov_d(dd, C_FP_RET0));
                }
                self.restore_caller_saved_gp_regs(&[]);
            }
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Nearest) => {
                self.spill_caller_saved_gp_regs();
                self.core
                    .text
                    .emit_u32(enc::vmov_rr_d(Arm32Reg::R0, Arm32Reg::R1, *dm));
                self.emit_host_call(arm32_f64_nearest_bits as *const () as usize);
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
                let s0 = C_FP_RET0 * 2;
                if src_s != s0 {
                    self.core.text.emit_u32(enc::vmov_s(s0, src_s));
                }
                self.emit_host_call(arm32_f32_ceil as *const () as usize);
                if dst_s != s0 {
                    self.core.text.emit_u32(enc::vmov_s(dst_s, s0));
                }
                self.restore_caller_saved_gp_regs(&[]);
            }
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Floor) => {
                self.spill_caller_saved_gp_regs();
                let src_s = *dm * 2;
                let dst_s = dd * 2;
                let s0 = C_FP_RET0 * 2;
                if src_s != s0 {
                    self.core.text.emit_u32(enc::vmov_s(s0, src_s));
                }
                self.emit_host_call(arm32_f32_floor as *const () as usize);
                if dst_s != s0 {
                    self.core.text.emit_u32(enc::vmov_s(dst_s, s0));
                }
                self.restore_caller_saved_gp_regs(&[]);
            }
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Trunc) => {
                self.spill_caller_saved_gp_regs();
                let src_s = *dm * 2;
                let dst_s = dd * 2;
                let s0 = C_FP_RET0 * 2;
                if src_s != s0 {
                    self.core.text.emit_u32(enc::vmov_s(s0, src_s));
                }
                self.emit_host_call(arm32_f32_trunc as *const () as usize);
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
                self.emit_host_call(arm32_f32_nearest_bits as *const () as usize);
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
        emit_dp_imm_cond_into(
            &mut self.core.text,
            Self::float_compare_ordered_cond(kind),
            0b1101,
            false,
            dst_hw,
            Arm32Reg::R0,
            imm8,
            rot,
        );
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
                    MachineValue::ReservedReg(_reg) => {
                        return Err(WasmError::internal(
                            "arm32 I32WrapI64 cannot consume reserved cache register as src",
                        ));
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
                    MachineValue::ReservedReg(_reg) => {
                        return Err(WasmError::internal(
                            "arm32 I64ExtendI32 cannot consume reserved cache register as src",
                        ));
                    }
                    MachineValue::Imm64(v) => self.emit_load_u32(dst_hw, *v as u32),
                }
            }

            // All remaining Convert ops involve FP registers.
            op_fp => {
                self.compile_convert_fp(op_fp, dst, src)?;
            }
        }
        Ok(())
    }

    /// FP-involving Convert operations.
    fn compile_convert_fp(
        &mut self,
        op: MachineConvertOp,
        dst: MachineReg,
        src: &MachineValue,
    ) -> Result<(), WasmError> {
        match op {
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
                    "arm32 direct i64 trunc convert should be legalized to ConvertFloatToI64Pair"
                        .into(),
                ));
            }

            // ─── I32 → F64 (GP src → FP dst) ────────────────────────────────
            MachineConvertOp::F64ConvertI32S => {
                let dd = self.map_fp_dreg(dst)?;
                let src_gp = prepare_gp(&mut self.core.text, &self.gp_scratch, *src)?;
                let fp_s = self.fp_scratch.scoped_alloc();
                let sd_tmp = *fp_s * 2;
                self.core.text.emit_u32(enc::vmov_s_r(sd_tmp, *src_gp));
                self.core.text.emit_u32(enc::vcvt_d_s32(dd, sd_tmp));
            }
            MachineConvertOp::F64ConvertI32U => {
                let dd = self.map_fp_dreg(dst)?;
                let src_gp = prepare_gp(&mut self.core.text, &self.gp_scratch, *src)?;
                let fp_s = self.fp_scratch.scoped_alloc();
                let sd_tmp = *fp_s * 2;
                self.core.text.emit_u32(enc::vmov_s_r(sd_tmp, *src_gp));
                self.core.text.emit_u32(enc::vcvt_d_u32(dd, sd_tmp));
            }

            // ─── I32 → F32 (GP src → FP dst) ────────────────────────────────
            MachineConvertOp::F32ConvertI32S => {
                let sd = self.map_fp_dreg(dst)? * 2; // S-register
                let src_gp = prepare_gp(&mut self.core.text, &self.gp_scratch, *src)?;
                let fp_s = self.fp_scratch.scoped_alloc();
                let sd_tmp = *fp_s * 2;
                self.core.text.emit_u32(enc::vmov_s_r(sd_tmp, *src_gp));
                self.core.text.emit_u32(enc::vcvt_s_s32(sd, sd_tmp));
            }
            MachineConvertOp::F32ConvertI32U => {
                let sd = self.map_fp_dreg(dst)? * 2;
                let src_gp = prepare_gp(&mut self.core.text, &self.gp_scratch, *src)?;
                let fp_s = self.fp_scratch.scoped_alloc();
                let sd_tmp = *fp_s * 2;
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
                    let fp_s = self.fp_scratch.scoped_alloc();
                    sm = *fp_s * 2;
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
                    let fp_s = self.fp_scratch.scoped_alloc();
                    dm = *fp_s;
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
                self.emit_host_call(arm32_i64s_to_f64 as *const () as usize);
                // Result is in D0 (EABI: f64 returned in D0)
                if dd != C_FP_RET0 {
                    self.core.text.emit_u32(enc::vmov_d(dd, C_FP_RET0));
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
                self.emit_host_call(arm32_i64u_to_f64 as *const () as usize);
                if dd != C_FP_RET0 {
                    self.core.text.emit_u32(enc::vmov_d(dd, C_FP_RET0));
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
                self.emit_host_call(arm32_i64s_to_f32 as *const () as usize);
                // Result in S0 (EABI: f32 returned in S0)
                {
                    let s0 = C_FP_RET0 * 2;
                    if sd != s0 {
                        self.core.text.emit_u32(enc::vmov_s(sd, s0));
                    }
                }
                self.restore_caller_saved_gp_regs(&[]);
            }
            MachineConvertOp::F32ConvertI64U => {
                let sd = self.map_fp_dreg(dst)? * 2;
                let src_hw = prepare_gp(&mut self.core.text, &self.gp_scratch, *src)?.detach();
                self.spill_caller_saved_gp_regs();
                self.core.text.emit_u32(enc::mov_reg(Arm32Reg::R0, *src_hw));
                self.emit_load_u32(Arm32Reg::R1, 0);
                self.emit_host_call(arm32_i64u_to_f32 as *const () as usize);
                {
                    let s0 = C_FP_RET0 * 2;
                    if sd != s0 {
                        self.core.text.emit_u32(enc::vmov_s(sd, s0));
                    }
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
                    MachineValue::ReservedReg(_reg) => {
                        return Err(WasmError::internal(
                            "arm32 F64ReinterpretI64 cannot consume reserved cache register as src",
                        ));
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
            // GP-only ops are handled above; this arm should be unreachable.
            _ => {
                return Err(WasmError::internal(
                    "arm32 compile_convert_fp: unexpected GP-only convert op",
                ));
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
                MachineValue::ReservedReg(_reg) => {
                    return Err(WasmError::internal(
                        "arm32 FP select cannot consume reserved cache register as false_val",
                    ));
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
                MachineValue::ReservedReg(_reg) => {
                    return Err(WasmError::internal(
                        "arm32 FP select cannot consume reserved cache register as true_val",
                    ));
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
            MachineValue::ReservedReg(_reg) => {
                return Err(WasmError::internal(
                    "arm32 emit_gp_select_value cannot consume reserved cache register as value",
                ));
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
                emit_mov_reg_cond_into(&mut self.core.text, cond, dst_hw, src);
            }
            MachineValue::ReservedReg(_reg) => {
                return Err(WasmError::internal("arm32 emit_gp_select_value_cond cannot consume reserved cache register as value"));
            }
            MachineValue::Imm64(v) => {
                let s = self.gp_scratch.scoped_alloc();
                emit_load_u32_into(&mut self.core.text, *s, *v as u32);
                emit_mov_reg_cond_into(&mut self.core.text, cond, dst_hw, *s);
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
                MachineValue::ReservedReg(_reg) => {
                    return Err(WasmError::internal(
                        "arm32 compile_int_compare cannot consume reserved cache register as rhs",
                    ));
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
        emit_dp_imm_cond_into(
            &mut self.core.text,
            cond,
            0b1101,
            false,
            dst_hw,
            Arm32Reg::R0,
            imm8,
            rot,
        );
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
            MachineShiftOp::Ror => 0b11,
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
                return Err(WasmError::internal("IntBinaryShifted: unsupported op"));
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
            MachineValue::ReservedReg(_reg) => {
                return Err(WasmError::internal(
                    "arm32 compile_test_bits cannot consume reserved cache register as mask",
                ));
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
                return Err(WasmError::internal("TestBits: unsupported compare kind"));
            }
        };
        emit_load_u32_into(&mut self.core.text, dst_hw, 0);
        let (imm8, rot) = enc::encode_arm_imm(1).unwrap();
        emit_dp_imm_cond_into(
            &mut self.core.text,
            cond,
            0b1101, // MOV
            false,
            dst_hw,
            Arm32Reg::R0,
            imm8,
            rot,
        );
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

    // ─── CallRuntime ───────────────────────────────────────────────────────

    fn compile_call_runtime(&mut self, call: &MachineCallRuntime) -> Result<(), WasmError> {
        let metadata =
            self.core.compiled.const_ptr(call.metadata).ok_or_else(|| {
                WasmError::internal("arm32: runtime-call metadata is out of range")
            })?;

        let helper_ptr = call_runtime_entry_ptr() as usize;

        // Imported calls cross the foreign C ABI, so the caller-saved GP
        // dynamic subset must be spilled explicitly before we stage the ABI
        // arguments into R0-R2.
        self.spill_caller_saved_gp_regs();

        // EABI: fn(ctx: *mut NativeContext, frame: *mut u64, metadata: *const u8) -> u32
        self.core
            .text
            .emit_u32(enc::mov_reg(C_ARG0, map_fixed_reg(MACHINE_CTX_REG)));
        self.core
            .text
            .emit_u32(enc::mov_reg(C_ARG1, map_fixed_reg(MACHINE_FP_REG)));
        self.emit_load_addr(C_ARG2, metadata as usize);

        self.emit_host_call(helper_ptr);

        // Preserve the status code across the GP restore, then re-materialize
        // it in C_RET0 for the post-call error check.
        {
            let s = self.gp_scratch.scoped_alloc().detach();
            self.core.text.emit_u32(enc::mov_reg(*s, C_RET0));
            self.restore_caller_saved_gp_regs(&[]);
            self.core.text.emit_u32(enc::mov_reg(C_RET0, *s));
        }

        // Check return value: if non-zero, return error
        self.core.text.emit_u32(enc::cmp_imm(C_RET0, 0, 0));
        let body_local_error = self.core.body_local_error_label;
        self.emit_branch(BranchFixupKind::BCond(Cond::Ne), body_local_error);

        Ok(())
    }

    fn compile_preserved_result(
        &mut self,
        op_code: u32,
        imm0: u32,
        imm1: u32,
        arg0: MachineValue,
        arg1: MachineValue,
        arg2: MachineValue,
        ty: MachineStorageType,
        dst: MachineReg,
    ) -> Result<(), WasmError> {
        let result = if let Some(width) = ty.float_width() {
            super::preserved::PreservedResultTarget::Float { dst, width }
        } else {
            super::preserved::PreservedResultTarget::GpWord(dst)
        };
        self.emit_preserved_helper_call_extended(
            op_code,
            &[(preserved_io::IMM0, imm0), (preserved_io::IMM1, imm1)],
            &[
                (preserved_io::ARG0, arg0),
                (preserved_io::ARG1, arg1),
                (preserved_io::ARG2, arg2),
            ],
            &[],
            result,
        )
    }

    fn compile_struct_new(
        &mut self,
        type_idx: u32,
        fields: &[(MachineValue, Option<MachineValue>)],
        dst: MachineReg,
    ) -> Result<(), WasmError> {
        self.emit_preserved_helper_call_extended_with_payload(
            preserved_op::STRUCT_NEW,
            &[
                (preserved_io::IMM0, type_idx),
                (preserved_io::IMM1, fields.len() as u32),
            ],
            &[],
            &[],
            fields,
            Some(preserved_io::ARG0),
            super::preserved::PreservedResultTarget::GpWord(dst),
        )
    }

    fn compile_struct_get(
        &mut self,
        type_idx: u32,
        field_idx: u32,
        signed: Option<bool>,
        ty: MachineStorageType,
        src: MachineValue,
        dst: MachineReg,
        dst_hi: Option<MachineReg>,
    ) -> Result<(), WasmError> {
        let op_code = match signed {
            None => preserved_op::STRUCT_GET,
            Some(true) => preserved_op::STRUCT_GET_S,
            Some(false) => preserved_op::STRUCT_GET_U,
        };
        let result = if let Some(dst_hi) = dst_hi {
            super::preserved::PreservedResultTarget::GpPair {
                dst_lo: dst,
                dst_hi,
            }
        } else if let Some(width) = ty.float_width() {
            super::preserved::PreservedResultTarget::Float { dst, width }
        } else {
            super::preserved::PreservedResultTarget::GpWord(dst)
        };
        self.emit_preserved_helper_call_extended(
            op_code,
            &[
                (preserved_io::IMM0, type_idx),
                (preserved_io::IMM1, field_idx),
            ],
            &[(preserved_io::ARG0, src)],
            &[],
            result,
        )
    }

    fn compile_struct_set(
        &mut self,
        type_idx: u32,
        field_idx: u32,
        ref_src: MachineValue,
        value_lo: MachineValue,
        value_hi: Option<MachineValue>,
    ) -> Result<(), WasmError> {
        match value_hi {
            Some(value_hi) => self.emit_preserved_helper_call_extended(
                preserved_op::STRUCT_SET,
                &[
                    (preserved_io::IMM0, type_idx),
                    (preserved_io::IMM1, field_idx),
                ],
                &[(preserved_io::ARG0, ref_src)],
                &[(preserved_io::ARG1, value_lo, value_hi)],
                super::preserved::PreservedResultTarget::None,
            ),
            None => self.emit_preserved_helper_call_extended(
                preserved_op::STRUCT_SET,
                &[
                    (preserved_io::IMM0, type_idx),
                    (preserved_io::IMM1, field_idx),
                ],
                &[
                    (preserved_io::ARG0, ref_src),
                    (preserved_io::ARG1, value_lo),
                ],
                &[],
                super::preserved::PreservedResultTarget::None,
            ),
        }
    }

    fn compile_array_new(
        &mut self,
        type_idx: u32,
        init_lo: MachineValue,
        init_hi: Option<MachineValue>,
        length: MachineValue,
        dst: MachineReg,
    ) -> Result<(), WasmError> {
        let result = super::preserved::PreservedResultTarget::GpWord(dst);
        match init_hi {
            Some(init_hi) => self.emit_preserved_helper_call_extended(
                preserved_op::ARRAY_NEW,
                &[(preserved_io::IMM0, type_idx), (preserved_io::IMM1, 0)],
                &[(preserved_io::ARG1, length)],
                &[(preserved_io::ARG0, init_lo, init_hi)],
                result,
            ),
            None => self.emit_preserved_helper_call_extended(
                preserved_op::ARRAY_NEW,
                &[(preserved_io::IMM0, type_idx), (preserved_io::IMM1, 0)],
                &[(preserved_io::ARG0, init_lo), (preserved_io::ARG1, length)],
                &[],
                result,
            ),
        }
    }

    fn compile_array_new_fixed(
        &mut self,
        type_idx: u32,
        elements: &[(MachineValue, Option<MachineValue>)],
        dst: MachineReg,
    ) -> Result<(), WasmError> {
        self.emit_preserved_helper_call_extended_with_payload(
            preserved_op::ARRAY_NEW_FIXED,
            &[
                (preserved_io::IMM0, type_idx),
                (preserved_io::IMM1, elements.len() as u32),
            ],
            &[],
            &[],
            elements,
            Some(preserved_io::ARG0),
            super::preserved::PreservedResultTarget::GpWord(dst),
        )
    }

    fn compile_array_new_data(
        &mut self,
        type_idx: u32,
        data_idx: u32,
        src: MachineValue,
        len: MachineValue,
        dst: MachineReg,
    ) -> Result<(), WasmError> {
        self.emit_preserved_helper_call_extended(
            preserved_op::ARRAY_NEW_DATA,
            &[
                (preserved_io::IMM0, type_idx),
                (preserved_io::IMM1, data_idx),
            ],
            &[(preserved_io::ARG0, src), (preserved_io::ARG1, len)],
            &[],
            super::preserved::PreservedResultTarget::GpWord(dst),
        )
    }

    fn compile_array_new_elem(
        &mut self,
        type_idx: u32,
        elem_idx: u32,
        src: MachineValue,
        len: MachineValue,
        dst: MachineReg,
    ) -> Result<(), WasmError> {
        self.emit_preserved_helper_call_extended(
            preserved_op::ARRAY_NEW_ELEM,
            &[
                (preserved_io::IMM0, type_idx),
                (preserved_io::IMM1, elem_idx),
            ],
            &[(preserved_io::ARG0, src), (preserved_io::ARG1, len)],
            &[],
            super::preserved::PreservedResultTarget::GpWord(dst),
        )
    }

    fn compile_array_get(
        &mut self,
        type_idx: u32,
        signed: Option<bool>,
        ty: MachineStorageType,
        ref_src: MachineValue,
        index: MachineValue,
        dst: MachineReg,
        dst_hi: Option<MachineReg>,
    ) -> Result<(), WasmError> {
        let op_code = match signed {
            None => preserved_op::ARRAY_GET,
            Some(true) => preserved_op::ARRAY_GET_S,
            Some(false) => preserved_op::ARRAY_GET_U,
        };
        let result = if let Some(dst_hi) = dst_hi {
            super::preserved::PreservedResultTarget::GpPair {
                dst_lo: dst,
                dst_hi,
            }
        } else if let Some(width) = ty.float_width() {
            super::preserved::PreservedResultTarget::Float { dst, width }
        } else {
            super::preserved::PreservedResultTarget::GpWord(dst)
        };
        self.emit_preserved_helper_call_extended(
            op_code,
            &[(preserved_io::IMM0, type_idx), (preserved_io::IMM1, 0)],
            &[(preserved_io::ARG0, ref_src), (preserved_io::ARG1, index)],
            &[],
            result,
        )
    }

    fn compile_array_set(
        &mut self,
        type_idx: u32,
        ref_src: MachineValue,
        index: MachineValue,
        value_lo: MachineValue,
        value_hi: Option<MachineValue>,
    ) -> Result<(), WasmError> {
        match value_hi {
            Some(value_hi) => self.emit_preserved_helper_call_extended(
                preserved_op::ARRAY_SET,
                &[(preserved_io::IMM0, type_idx), (preserved_io::IMM1, 0)],
                &[(preserved_io::ARG0, ref_src), (preserved_io::ARG1, index)],
                &[(preserved_io::ARG2, value_lo, value_hi)],
                super::preserved::PreservedResultTarget::None,
            ),
            None => self.emit_preserved_helper_call_extended(
                preserved_op::ARRAY_SET,
                &[(preserved_io::IMM0, type_idx), (preserved_io::IMM1, 0)],
                &[
                    (preserved_io::ARG0, ref_src),
                    (preserved_io::ARG1, index),
                    (preserved_io::ARG2, value_lo),
                ],
                &[],
                super::preserved::PreservedResultTarget::None,
            ),
        }
    }

    fn compile_array_fill(
        &mut self,
        type_idx: u32,
        ref_src: MachineValue,
        index: MachineValue,
        value_lo: MachineValue,
        value_hi: Option<MachineValue>,
        len: MachineValue,
    ) -> Result<(), WasmError> {
        match value_hi {
            Some(value_hi) => self.emit_preserved_helper_call_extended(
                preserved_op::ARRAY_FILL,
                &[(preserved_io::IMM0, type_idx)],
                &[
                    (preserved_io::ARG0, ref_src),
                    (preserved_io::ARG1, index),
                    (preserved_io::ARG3, len),
                ],
                &[(preserved_io::ARG2, value_lo, value_hi)],
                super::preserved::PreservedResultTarget::None,
            ),
            None => self.emit_preserved_helper_call_extended(
                preserved_op::ARRAY_FILL,
                &[(preserved_io::IMM0, type_idx)],
                &[
                    (preserved_io::ARG0, ref_src),
                    (preserved_io::ARG1, index),
                    (preserved_io::ARG2, value_lo),
                    (preserved_io::ARG3, len),
                ],
                &[],
                super::preserved::PreservedResultTarget::None,
            ),
        }
    }

    fn compile_array_copy(
        &mut self,
        dst_type_idx: u32,
        src_type_idx: u32,
        dst_ref: MachineValue,
        dst_index: MachineValue,
        src_ref: MachineValue,
        src_index: MachineValue,
        len: MachineValue,
    ) -> Result<(), WasmError> {
        self.emit_preserved_helper_call_extended(
            preserved_op::ARRAY_COPY,
            &[
                (preserved_io::IMM0, dst_type_idx),
                (preserved_io::IMM1, src_type_idx),
            ],
            &[
                (preserved_io::ARG0, dst_ref),
                (preserved_io::ARG1, dst_index),
                (preserved_io::ARG2, src_ref),
                (preserved_io::ARG3, src_index),
                (preserved_io::ARG4, len),
            ],
            &[],
            super::preserved::PreservedResultTarget::None,
        )
    }

    fn compile_array_init_data(
        &mut self,
        type_idx: u32,
        data_idx: u32,
        ref_src: MachineValue,
        dst_index: MachineValue,
        src_index: MachineValue,
        len: MachineValue,
    ) -> Result<(), WasmError> {
        self.emit_preserved_helper_call_extended(
            preserved_op::ARRAY_INIT_DATA,
            &[
                (preserved_io::IMM0, type_idx),
                (preserved_io::IMM1, data_idx),
            ],
            &[
                (preserved_io::ARG0, ref_src),
                (preserved_io::ARG1, dst_index),
                (preserved_io::ARG2, src_index),
                (preserved_io::ARG3, len),
            ],
            &[],
            super::preserved::PreservedResultTarget::None,
        )
    }

    fn compile_array_init_elem(
        &mut self,
        type_idx: u32,
        elem_idx: u32,
        ref_src: MachineValue,
        dst_index: MachineValue,
        src_index: MachineValue,
        len: MachineValue,
    ) -> Result<(), WasmError> {
        self.emit_preserved_helper_call_extended(
            preserved_op::ARRAY_INIT_ELEM,
            &[
                (preserved_io::IMM0, type_idx),
                (preserved_io::IMM1, elem_idx),
            ],
            &[
                (preserved_io::ARG0, ref_src),
                (preserved_io::ARG1, dst_index),
                (preserved_io::ARG2, src_index),
                (preserved_io::ARG3, len),
            ],
            &[],
            super::preserved::PreservedResultTarget::None,
        )
    }
} // impl Arm32Backend

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collections;
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
