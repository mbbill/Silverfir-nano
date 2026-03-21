//! LIR-level legalized-op vocabulary for 32-bit targets.
//!
//! This is the backend-facing representation of legalized i64 operations.
//! Unlike `wasm::legalized_op::LegalizedOpKind` (which is a semantic-layer
//! type), this enum lives entirely in the LIR layer and uses only LIR-level
//! discriminants. The planner converts from `LegalizedOpKind` when building
//! `LirInstKind::Legalized`.

use crate::value_type::ValueType;

/// One legalized 32-bit op in prepared LIR.
///
/// All operands and results are word-sized (I32) or float (F32/F64).
/// Where an original i64 value is involved, it appears as adjacent
/// `(lo, hi)` word values on the operand stack.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LirLegalizedOp {
    // ── Carry / borrow arithmetic (inline-lowered to MachineIR) ─────────
    /// `(a, b) → (sum, carry)`
    AddCarryOut,
    /// `(a, b, carry_in) → (sum)`
    AddWithCarry,
    /// `(a, b) → (diff, borrow)`
    SubBorrowOut,
    /// `(a, b, borrow_in) → (diff)`
    SubWithBorrow,
    /// `(a, b) → (lo, hi)` — full-width 32×32→64 multiply.
    MulWide { signed: bool },

    // ── Abstract pair ops (backend choice: inline or helper) ────────────
    /// `(a_lo, a_hi, b_lo, b_hi) → (bool_i32)`
    PairCompare { kind: CompareKind, signed: bool },
    /// `(a_lo, a_hi, b_lo, b_hi) → (res_lo, res_hi)`
    PairMul,

    // ── Helper-backed pair ops ──────────────────────────────────────────
    /// `(val_lo, val_hi, count) → (res_lo, res_hi)`
    PairShift { direction: ShiftDirection },
    /// `(a_lo, a_hi, b_lo, b_hi) → (res_lo, res_hi)`
    PairDivRem { signed: bool, rem: bool },
    /// `(val_lo, val_hi) → (res_f32)` or `(res_f64)`
    PairToFloat { f64_result: bool, signed: bool },
    /// `(val_f32)` or `(val_f64)` → `(res_lo, res_hi)`
    FloatToPair { f64_source: bool, signed: bool, saturating: bool },
    /// `(val_f64) → (lo, hi)` — raw bit reinterpret
    F64ReinterpretToPair,
    /// `(lo, hi) → (val_f64)` — raw bit reinterpret
    PairReinterpretToF64,
}

/// Ordered comparison relation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompareKind {
    Lt,
    Le,
    Gt,
    Ge,
}

/// Shift/rotate direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShiftDirection {
    Shl,
    ShrS,
    ShrU,
    Rotl,
    Rotr,
}

impl LirLegalizedOp {
    /// Stack effect in word-sized values: (pops, pushes).
    pub const fn stack_effect(&self) -> (u8, u8) {
        match self {
            Self::AddCarryOut => (2, 2),
            Self::AddWithCarry => (3, 1),
            Self::SubBorrowOut => (2, 2),
            Self::SubWithBorrow => (3, 1),
            Self::MulWide { .. } => (2, 2),
            Self::PairCompare { .. } => (4, 1),
            Self::PairMul => (4, 2),
            Self::PairShift { .. } => (3, 2),
            Self::PairDivRem { .. } => (4, 2),
            Self::PairToFloat { .. } => (2, 1),
            Self::FloatToPair { .. } => (1, 2),
            Self::F64ReinterpretToPair => (1, 2),
            Self::PairReinterpretToF64 => (2, 1),
        }
    }

    /// Type of the n-th result (0-based).
    pub fn result_type(&self, n: u8) -> ValueType {
        match self {
            Self::PairToFloat { f64_result, .. } => {
                debug_assert_eq!(n, 0);
                if *f64_result { ValueType::F64 } else { ValueType::F32 }
            }
            Self::PairReinterpretToF64 => {
                debug_assert_eq!(n, 0);
                ValueType::F64
            }
            _ => ValueType::I32,
        }
    }

    /// Type of the n-th argument (0-based).
    pub fn arg_type(&self, n: u8) -> ValueType {
        match self {
            Self::FloatToPair { f64_source, .. } => {
                debug_assert_eq!(n, 0);
                if *f64_source { ValueType::F64 } else { ValueType::F32 }
            }
            Self::F64ReinterpretToPair => {
                debug_assert_eq!(n, 0);
                ValueType::F64
            }
            _ => ValueType::I32,
        }
    }

    /// True if this op must be lowered via a runtime helper call.
    pub fn needs_helper(&self) -> bool {
        matches!(
            self,
            Self::PairShift { .. }
            | Self::PairDivRem { .. }
            | Self::PairToFloat { .. }
            | Self::FloatToPair { .. }
            | Self::F64ReinterpretToPair
            | Self::PairReinterpretToF64
        )
    }

    /// Extra GP transient registers needed during inline lowering.
    pub fn extra_inline_scratch(&self) -> u8 {
        match self {
            Self::PairCompare { .. } => 1,
            Self::PairMul => 1,
            _ => 0,
        }
    }

    /// Discriminant for the helper entry function.
    pub fn helper_discriminant(&self) -> u32 {
        match self {
            Self::PairShift { direction } => *direction as u32,
            Self::PairDivRem { signed, rem } => (*signed as u32) << 1 | (*rem as u32),
            Self::PairToFloat { f64_result, signed } => (*f64_result as u32) << 1 | (*signed as u32),
            Self::FloatToPair { f64_source, signed, saturating } => {
                (*f64_source as u32) << 2 | (*signed as u32) << 1 | (*saturating as u32)
            }
            Self::F64ReinterpretToPair | Self::PairReinterpretToF64 => 0,
            _ => 0,
        }
    }
}

/// Convert from the semantic-layer `LegalizedOpKind` to the LIR-layer `LirLegalizedOp`.
pub fn from_semantic(
    kind: &crate::vm::wasm::legalized_op::LegalizedOpKind,
) -> LirLegalizedOp {
    use crate::vm::wasm::legalized_op::{LegalizedOpKind, PairCompareKind, PairShiftOp, PairSign};

    match kind {
        LegalizedOpKind::I32AddCarryOut => LirLegalizedOp::AddCarryOut,
        LegalizedOpKind::I32AddWithCarry => LirLegalizedOp::AddWithCarry,
        LegalizedOpKind::I32SubBorrowOut => LirLegalizedOp::SubBorrowOut,
        LegalizedOpKind::I32SubWithBorrow => LirLegalizedOp::SubWithBorrow,
        LegalizedOpKind::I32MulWideU => LirLegalizedOp::MulWide { signed: false },
        LegalizedOpKind::I32MulWideS => LirLegalizedOp::MulWide { signed: true },
        LegalizedOpKind::I64PairCompare { kind, sign } => LirLegalizedOp::PairCompare {
            kind: match kind {
                PairCompareKind::Lt => CompareKind::Lt,
                PairCompareKind::Le => CompareKind::Le,
                PairCompareKind::Gt => CompareKind::Gt,
                PairCompareKind::Ge => CompareKind::Ge,
            },
            signed: matches!(sign, PairSign::Signed),
        },
        LegalizedOpKind::I64PairMul => LirLegalizedOp::PairMul,
        LegalizedOpKind::I64PairShift { op } => LirLegalizedOp::PairShift {
            direction: match op {
                PairShiftOp::Shl => ShiftDirection::Shl,
                PairShiftOp::ShrS => ShiftDirection::ShrS,
                PairShiftOp::ShrU => ShiftDirection::ShrU,
                PairShiftOp::Rotl => ShiftDirection::Rotl,
                PairShiftOp::Rotr => ShiftDirection::Rotr,
            },
        },
        LegalizedOpKind::I64PairToFloat { f64_result, signed } => LirLegalizedOp::PairToFloat {
            f64_result: *f64_result,
            signed: *signed,
        },
        LegalizedOpKind::FloatToI64Pair { f64_source, signed, saturating } => {
            LirLegalizedOp::FloatToPair {
                f64_source: *f64_source,
                signed: *signed,
                saturating: *saturating,
            }
        }
        LegalizedOpKind::F64ReinterpretToI64Pair => LirLegalizedOp::F64ReinterpretToPair,
        LegalizedOpKind::I64PairReinterpretToF64 => LirLegalizedOp::PairReinterpretToF64,
        LegalizedOpKind::I64PairDivRem { sign, rem } => LirLegalizedOp::PairDivRem {
            signed: matches!(sign, PairSign::Signed),
            rem: *rem,
        },
    }
}
