//! Inline float-to-integer truncation.
//!
//! CVTTSS2SI / CVTTSD2SI return the "integer indefinite" value (the most
//! negative integer of the destination width) for NaN and for every input
//! outside the destination range. The common in-range case is therefore one
//! conversion plus one check; only a result that might be the indefinite
//! value enters the slow classification of NaN, overflow, and the genuinely
//! representable minimum. Unsigned 32-bit results use the 64-bit conversion,
//! whose range covers every valid input exactly. Unsigned 64-bit results
//! split at 2^63 and rebias the upper half.
//!
//! The trapping forms branch to the shared `InvalidConversion` and
//! `IntegerOverflow` trap stubs; the saturating forms never trap.

use crate::{
    error::WasmError,
    vm::jit::machine::machine_ir::{
        MachineConvertOp, MachineFloatWidth, MachineReg, MachineTrapKind, MachineValue,
    },
};

use super::{
    backend::X86_64Backend,
    enc::{self, Cc, Xmm},
    reg::X86Reg,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum IntWidth {
    I32,
    I64,
}

#[derive(Clone, Copy)]
struct TruncShape {
    src: MachineFloatWidth,
    dst: IntWidth,
    signed: bool,
    saturating: bool,
}

fn trunc_shape(op: MachineConvertOp) -> Option<TruncShape> {
    use MachineConvertOp as Op;
    use MachineFloatWidth::{F32, F64};
    let (src, dst, signed, saturating) = match op {
        Op::I32TruncF32S => (F32, IntWidth::I32, true, false),
        Op::I32TruncF32U => (F32, IntWidth::I32, false, false),
        Op::I32TruncF64S => (F64, IntWidth::I32, true, false),
        Op::I32TruncF64U => (F64, IntWidth::I32, false, false),
        Op::I64TruncF32S => (F32, IntWidth::I64, true, false),
        Op::I64TruncF32U => (F32, IntWidth::I64, false, false),
        Op::I64TruncF64S => (F64, IntWidth::I64, true, false),
        Op::I64TruncF64U => (F64, IntWidth::I64, false, false),
        Op::I32TruncSatF32S => (F32, IntWidth::I32, true, true),
        Op::I32TruncSatF32U => (F32, IntWidth::I32, false, true),
        Op::I32TruncSatF64S => (F64, IntWidth::I32, true, true),
        Op::I32TruncSatF64U => (F64, IntWidth::I32, false, true),
        Op::I64TruncSatF32S => (F32, IntWidth::I64, true, true),
        Op::I64TruncSatF32U => (F32, IntWidth::I64, false, true),
        Op::I64TruncSatF64S => (F64, IntWidth::I64, true, true),
        Op::I64TruncSatF64U => (F64, IntWidth::I64, false, true),
        _ => return None,
    };
    Some(TruncShape {
        src,
        dst,
        signed,
        saturating,
    })
}

/// Whether `op` is a float-to-integer truncation lowered by this module.
pub(super) fn is_float_trunc(op: MachineConvertOp) -> bool {
    trunc_shape(op).is_some()
}

/// The bit pattern of `value` in the source float width.
fn float_bits(width: MachineFloatWidth, value: f64) -> u64 {
    match width {
        MachineFloatWidth::F32 => u64::from((value as f32).to_bits()),
        MachineFloatWidth::F64 => value.to_bits(),
    }
}

/// Registers and the join label shared by every trunc shape. `x` is either
/// the source's own FP register (live; never written) or `work` holding a
/// materialized copy. `constant` holds comparison constants; `tmp` is a GP
/// scratch distinct from `dst`.
#[derive(Clone, Copy)]
struct TruncRegs {
    dst: X86Reg,
    x: Xmm,
    constant: Xmm,
    work: Xmm,
    tmp: X86Reg,
    done: usize,
}

impl<'a> X86_64Backend<'a> {
    pub(super) fn lower_float_trunc(
        &mut self,
        op: MachineConvertOp,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let shape = trunc_shape(op)
            .ok_or_else(|| WasmError::internal("x86_64 trunc lowering got a non-trunc op"))?;
        let gp0 = self.gp_scratch.scoped_alloc().detach();
        let gp1 = self.gp_scratch.scoped_alloc().detach();
        let fp0 = self.fp_scratch.scoped_alloc().detach();
        let fp1 = self.fp_scratch.scoped_alloc().detach();
        let regs = TruncRegs {
            x: self.prepare_float_operand(shape.src, src, *gp0, *fp0)? as Xmm,
            dst: self.map_gp_reg(dst)?,
            constant: *fp1 as Xmm,
            work: *fp0 as Xmm,
            tmp: *gp1,
            done: self.core.new_label(),
        };
        match (shape.signed, shape.dst) {
            (true, width) => self.trunc_signed(shape, width, regs),
            (false, IntWidth::I32) => self.trunc_u32(shape, regs),
            (false, IntWidth::I64) => self.trunc_u64(shape, regs),
        }
        self.core.bind_label(regs.done);
        Ok(())
    }

    fn cvtt(&mut self, src_width: MachineFloatWidth, wide: bool, dst: X86Reg, src: Xmm) {
        let text = &mut self.core.text;
        match (src_width, wide) {
            (MachineFloatWidth::F32, false) => enc::cvttss2si_r32(text, dst, src),
            (MachineFloatWidth::F32, true) => enc::cvttss2si_r64(text, dst, src),
            (MachineFloatWidth::F64, false) => enc::cvttsd2si_r32(text, dst, src),
            (MachineFloatWidth::F64, true) => enc::cvttsd2si_r64(text, dst, src),
        }
    }

    /// Compare `x` against `value` rounded to `width`, via `constant`.
    fn ucomis_constant(&mut self, width: MachineFloatWidth, x: Xmm, value: f64, constant: Xmm) {
        self.load_float_immediate(width, float_bits(width, value), constant);
        self.ucomis(width, x, constant);
    }

    /// Branch to `nan` when `x` is unordered with itself.
    fn branch_if_nan(&mut self, width: MachineFloatWidth, x: Xmm, nan: usize) {
        self.ucomis(width, x, x);
        self.emit_jcc(Cc::P, nan);
    }

    fn trunc_signed(&mut self, shape: TruncShape, width: IntWidth, r: TruncRegs) {
        let wide = width == IntWidth::I64;
        self.cvtt(shape.src, wide, r.dst, r.x);
        // CMP r, 1 overflows exactly when r is the destination minimum.
        if wide {
            enc::cmp_ri_64(&mut self.core.text, r.dst, 1);
        } else {
            enc::cmp_ri_32(&mut self.core.text, r.dst, 1);
        }
        self.emit_jcc(Cc::NO, r.done);
        if shape.saturating {
            // NaN -> 0; non-negative overflow -> MAX; otherwise MIN stands.
            let nan = self.core.new_label();
            self.branch_if_nan(shape.src, r.x, nan);
            self.ucomis_constant(shape.src, r.x, 0.0, r.constant);
            self.emit_jcc(Cc::B, r.done);
            if wide {
                enc::mov_ri_64(&mut self.core.text, r.dst, i64::MAX as u64);
            } else {
                enc::mov_ri_32(&mut self.core.text, r.dst, i32::MAX as u32);
            }
            self.emit_jmp(r.done);
            self.core.bind_label(nan);
            enc::xor_rr_32(&mut self.core.text, r.dst, r.dst);
            return;
        }
        let invalid = self
            .core
            .ensure_trap_label(MachineTrapKind::InvalidConversion);
        let overflow = self
            .core
            .ensure_trap_label(MachineTrapKind::IntegerOverflow);
        self.branch_if_nan(shape.src, r.x, invalid);
        if shape.src == MachineFloatWidth::F64 && !wide {
            // Every f64 in (-2^31 - 1, -2^31] truncates to i32::MIN.
            self.ucomis_constant(shape.src, r.x, 0.0, r.constant);
            self.emit_jcc(Cc::AE, overflow);
            self.ucomis_constant(shape.src, r.x, -2_147_483_649.0, r.constant);
            self.emit_jcc(Cc::BE, overflow);
        } else {
            // The neighbours of -2^(N-1) in this source width are out of
            // range, so only the exact minimum is valid.
            let minimum = if wide {
                -9_223_372_036_854_775_808.0
            } else {
                -2_147_483_648.0
            };
            self.ucomis_constant(shape.src, r.x, minimum, r.constant);
            self.emit_jcc(Cc::NE, overflow);
        }
    }

    /// Unsigned 32-bit: the signed 64-bit conversion is exact for every
    /// valid input, and a valid result has a zero upper half. NaN and
    /// inputs outside the i64 range yield i64::MIN, which also fails.
    fn trunc_u32(&mut self, shape: TruncShape, r: TruncRegs) {
        self.cvtt(shape.src, true, r.dst, r.x);
        enc::mov_rr_64(&mut self.core.text, r.tmp, r.dst);
        enc::shr_imm_64(&mut self.core.text, r.tmp, 32);
        self.emit_jcc(Cc::E, r.done);
        if shape.saturating {
            // NaN or negative -> 0; positive overflow -> u32::MAX.
            let zero = self.core.new_label();
            self.branch_if_nan(shape.src, r.x, zero);
            self.ucomis_constant(shape.src, r.x, 0.0, r.constant);
            self.emit_jcc(Cc::B, zero);
            enc::mov_ri_32(&mut self.core.text, r.dst, u32::MAX);
            self.emit_jmp(r.done);
            self.core.bind_label(zero);
            enc::xor_rr_32(&mut self.core.text, r.dst, r.dst);
            return;
        }
        let invalid = self
            .core
            .ensure_trap_label(MachineTrapKind::InvalidConversion);
        let overflow = self
            .core
            .ensure_trap_label(MachineTrapKind::IntegerOverflow);
        self.branch_if_nan(shape.src, r.x, invalid);
        self.emit_jmp(overflow);
    }

    /// Unsigned 64-bit: inputs below 2^63 (and NaN, which compares
    /// unordered) use the signed conversion directly; inputs at or above
    /// 2^63 are rebiased by -2^63 and get the top bit restored.
    fn trunc_u64(&mut self, shape: TruncShape, r: TruncRegs) {
        let big = self.core.new_label();
        self.ucomis_constant(shape.src, r.x, 9_223_372_036_854_775_808.0, r.constant);
        self.emit_jcc(Cc::AE, big);
        self.cvtt(shape.src, true, r.dst, r.x);
        enc::test_rr_64(&mut self.core.text, r.dst, r.dst);
        self.emit_jcc(Cc::NS, r.done);
        // Negative: NaN, or an input at or below -1.
        if shape.saturating {
            enc::xor_rr_32(&mut self.core.text, r.dst, r.dst);
            self.emit_jmp(r.done);
        } else {
            let invalid = self
                .core
                .ensure_trap_label(MachineTrapKind::InvalidConversion);
            let overflow = self
                .core
                .ensure_trap_label(MachineTrapKind::IntegerOverflow);
            self.branch_if_nan(shape.src, r.x, invalid);
            self.emit_jmp(overflow);
        }
        self.core.bind_label(big);
        // `constant` still holds 2^63.
        if r.work != r.x {
            enc::movaps_rr(&mut self.core.text, r.work, r.x);
        }
        match shape.src {
            MachineFloatWidth::F32 => enc::subss(&mut self.core.text, r.work, r.constant),
            MachineFloatWidth::F64 => enc::subsd(&mut self.core.text, r.work, r.constant),
        }
        self.cvtt(shape.src, true, r.dst, r.work);
        enc::test_rr_64(&mut self.core.text, r.dst, r.dst);
        // A rebiased value still at or above 2^63 is out of range.
        let out_of_range = if shape.saturating {
            self.core.new_label()
        } else {
            self.core
                .ensure_trap_label(MachineTrapKind::IntegerOverflow)
        };
        self.emit_jcc(Cc::S, out_of_range);
        enc::mov_ri_64(&mut self.core.text, r.tmp, 1 << 63);
        enc::xor_rr_64(&mut self.core.text, r.dst, r.tmp);
        if shape.saturating {
            self.emit_jmp(r.done);
            self.core.bind_label(out_of_range);
            enc::mov_ri_64(&mut self.core.text, r.dst, u64::MAX);
        }
    }
}
