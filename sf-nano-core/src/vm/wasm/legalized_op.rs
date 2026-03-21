//! Legalized 32-bit semantic op vocabulary.
//!
//! These ops exist only in legalized semantic IR for 32-bit targets. They are
//! not canonical Wasm ops and do not belong in `primitive_op.rs`.
//!
//! The legalizer rewrites i64-producing Wasm ops into either:
//! - sequences of ordinary i32 `PrimitiveOpKind` ops (for simple cases), or
//! - `LegalizedOpKind` ops (for operations that need a target-deferred form)
//!
//! All operands and results are word-sized GP values (i32). Where an original
//! i64 value is involved, it appears as adjacent `(lo, hi)` word values on the
//! operand stack.

/// One legalized 32-bit semantic operation.
///
/// Stack effects are expressed in word-sized values. Operands that were
/// originally i64 appear as `(lo, hi)` pairs, with lo at the deeper stack
/// position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LegalizedOpKind {
    // ── Carry / borrow arithmetic ───────────────────────────────────────
    //
    // These are the core new primitives. They produce explicit carry/borrow
    // values as GP words (0 or 1).

    /// `(a_lo, b_lo) → (sum_lo, carry)`
    ///
    /// Used for the low half of a legalized i64 add.
    I32AddCarryOut,

    /// `(a_hi, b_hi, carry_in) → (sum_hi)`
    ///
    /// Used for the high half of a legalized i64 add.
    I32AddWithCarry,

    /// `(a_lo, b_lo) → (diff_lo, borrow)`
    ///
    /// Used for the low half of a legalized i64 sub.
    I32SubBorrowOut,

    /// `(a_hi, b_hi, borrow_in) → (diff_hi)`
    ///
    /// Used for the high half of a legalized i64 sub.
    I32SubWithBorrow,

    /// `(a, b) → (prod_lo, prod_hi)`
    ///
    /// Full-width 32×32→64 multiply. Sign does not matter for the low half;
    /// backends choose signed vs unsigned based on context.
    I32MulWideU,
    I32MulWideS,

    // ── Kept abstract for backend choice ────────────────────────────────
    //
    // These remain as explicit legalized ops because some backends can inline
    // them while others prefer helpers or target-specific sequences.

    /// `(a_lo, a_hi, b_lo, b_hi) → (bool_i32)`
    ///
    /// Ordered pair compare. The kind (Lt/Le/Gt/Ge) and sign are encoded.
    I64PairCompare {
        kind: PairCompareKind,
        sign: PairSign,
    },

    /// `(val_lo, val_hi, count_lo) → (res_lo, res_hi)`
    ///
    /// Variable shift/rotate on a legalized i64 pair. The shift count is
    /// already the low word only (Wasm i64 shifts use only the low 6 bits).
    I64PairShift { op: PairShiftOp },

    /// `(val_lo, val_hi) → (res_f32)` or `(res_f64)`
    I64PairToFloat {
        f64_result: bool,
        signed: bool,
    },

    /// `(val_f32)` or `(val_f64)` → `(res_lo, res_hi)`
    FloatToI64Pair {
        f64_source: bool,
        signed: bool,
        saturating: bool,
    },

    /// `(val_f64) → (lo, hi)` — raw bit reinterpret
    F64ReinterpretToI64Pair,

    /// `(lo, hi) → (val_f64)` — raw bit reinterpret
    I64PairReinterpretToF64,

    /// `(val_lo, val_hi) → (res_lo, res_hi)` — i64 div/rem as pair
    ///
    /// These are kept abstract because they are always helper-backed on
    /// 32-bit targets, but the legalizer preserves the structure so backends
    /// can choose the calling convention.
    I64PairDivRem {
        sign: PairSign,
        rem: bool,
    },

    /// `(val_lo, val_hi) → (res_lo, res_hi)` — i64 mul as pair
    ///
    /// Full 64×64→64 multiply over legalized pairs. Backends typically
    /// decompose this into wide-mul + cross products.
    I64PairMul,
}

/// Ordered compare relation for pair comparisons.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairCompareKind {
    Lt,
    Le,
    Gt,
    Ge,
}

/// Signedness for pair operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairSign {
    Signed,
    Unsigned,
}

/// Shift/rotate direction for pair shifts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairShiftOp {
    Shl,
    ShrS,
    ShrU,
    Rotl,
    Rotr,
}

/// Return the `ValueType` of the `n`-th argument (0-based) of a legalized op.
///
/// Most legalized ops consume only `I32` arguments, but float-consuming ops
/// (`FloatToI64Pair`, `F64ReinterpretToI64Pair`) take `F32` or `F64`.
pub fn legalized_arg_type(kind: &LegalizedOpKind, n: u8) -> crate::value_type::ValueType {
    use crate::value_type::ValueType;
    match kind {
        LegalizedOpKind::FloatToI64Pair { f64_source, .. } => {
            debug_assert_eq!(n, 0, "FloatToI64Pair has exactly 1 float arg");
            if *f64_source { ValueType::F64 } else { ValueType::F32 }
        }
        LegalizedOpKind::F64ReinterpretToI64Pair => {
            debug_assert_eq!(n, 0, "F64ReinterpretToI64Pair has exactly 1 f64 arg");
            ValueType::F64
        }
        _ => ValueType::I32,
    }
}

/// Return the `ValueType` of the `n`-th result (0-based) of a legalized op.
///
/// Most legalized ops produce only `I32` results, but float-producing ops
/// (`I64PairToFloat`, `I64PairReinterpretToF64`) return `F32` or `F64`.
pub fn legalized_result_type(kind: &LegalizedOpKind, n: u8) -> crate::value_type::ValueType {
    use crate::value_type::ValueType;
    match kind {
        LegalizedOpKind::I64PairToFloat { f64_result, .. } => {
            debug_assert_eq!(n, 0, "I64PairToFloat has exactly 1 result");
            if *f64_result { ValueType::F64 } else { ValueType::F32 }
        }
        LegalizedOpKind::I64PairReinterpretToF64 => {
            debug_assert_eq!(n, 0, "I64PairReinterpretToF64 has exactly 1 result");
            ValueType::F64
        }
        _ => ValueType::I32,
    }
}

/// Returns true if this legalized op must be lowered via a runtime helper call
/// rather than inline MachineIR.
/// Extra GP transient registers needed during inline lowering, beyond the
/// declared args+results. The planner must account for this so the transient
/// budget is never silently exceeded.
pub fn extra_inline_scratch(kind: &LegalizedOpKind) -> u8 {
    match kind {
        // Compare uses 1 scratch for intermediate boolean.
        LegalizedOpKind::I64PairCompare { .. } => 1,
        // Mul uses 1 scratch for cross-product accumulation.
        LegalizedOpKind::I64PairMul => 1,
        _ => 0,
    }
}

pub fn needs_helper(kind: &LegalizedOpKind) -> bool {
    matches!(
        kind,
        LegalizedOpKind::I64PairShift { .. }
        | LegalizedOpKind::I64PairDivRem { .. }
        | LegalizedOpKind::I64PairToFloat { .. }
        | LegalizedOpKind::FloatToI64Pair { .. }
        | LegalizedOpKind::F64ReinterpretToI64Pair
        | LegalizedOpKind::I64PairReinterpretToF64
    )
}

/// Encode the variant-specific discriminant for a helper-backed legalized op.
///
/// The helper entry function uses this to dispatch to the correct sub-operation.
pub fn helper_op_discriminant(kind: &LegalizedOpKind) -> u32 {
    match kind {
        LegalizedOpKind::I64PairShift { op } => *op as u32,
        LegalizedOpKind::I64PairDivRem { sign, rem } => {
            (*sign as u32) << 1 | (*rem as u32)
        }
        LegalizedOpKind::I64PairToFloat { f64_result, signed } => {
            (*f64_result as u32) << 1 | (*signed as u32)
        }
        LegalizedOpKind::FloatToI64Pair { f64_source, signed, saturating } => {
            (*f64_source as u32) << 2 | (*signed as u32) << 1 | (*saturating as u32)
        }
        LegalizedOpKind::F64ReinterpretToI64Pair => 0,
        LegalizedOpKind::I64PairReinterpretToF64 => 0,
        _ => 0,
    }
}

/// Stack effect of a legalized op, in word-sized values.
pub const fn legalized_stack_effect(kind: &LegalizedOpKind) -> (u8, u8) {
    match kind {
        // carry/borrow primitives
        LegalizedOpKind::I32AddCarryOut => (2, 2),      // (a, b) → (sum, carry)
        LegalizedOpKind::I32AddWithCarry => (3, 1),     // (a, b, carry) → (sum)
        LegalizedOpKind::I32SubBorrowOut => (2, 2),     // (a, b) → (diff, borrow)
        LegalizedOpKind::I32SubWithBorrow => (3, 1),    // (a, b, borrow) → (diff)
        LegalizedOpKind::I32MulWideU
        | LegalizedOpKind::I32MulWideS => (2, 2),       // (a, b) → (lo, hi)

        // pair compare → i32 bool
        LegalizedOpKind::I64PairCompare { .. } => (4, 1),

        // pair shift: (lo, hi, count) → (lo, hi)
        LegalizedOpKind::I64PairShift { .. } => (3, 2),

        // pair ↔ float conversions
        LegalizedOpKind::I64PairToFloat { .. } => (2, 1),
        LegalizedOpKind::FloatToI64Pair { .. } => (1, 2),
        LegalizedOpKind::F64ReinterpretToI64Pair => (1, 2),
        LegalizedOpKind::I64PairReinterpretToF64 => (2, 1),

        // pair div/rem and mul
        LegalizedOpKind::I64PairDivRem { .. } => (4, 2),
        LegalizedOpKind::I64PairMul => (4, 2),
    }
}
