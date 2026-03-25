//! x86_64 condition-code mapping for integer and float comparisons.

use crate::vm::machine::machine_ir::{MachineCompareKind, MachineSign};
use super::enc::Cc;

pub(super) fn map_int_cond(kind: MachineCompareKind, sign: MachineSign) -> Cc {
    match (kind, sign) {
        (MachineCompareKind::Eq, _) => Cc::E,
        (MachineCompareKind::Ne, _) => Cc::NE,
        (MachineCompareKind::Lt, MachineSign::Signed) => Cc::L,
        (MachineCompareKind::Lt, MachineSign::Unsigned) => Cc::B,
        (MachineCompareKind::Gt, MachineSign::Signed) => Cc::G,
        (MachineCompareKind::Gt, MachineSign::Unsigned) => Cc::A,
        (MachineCompareKind::Le, MachineSign::Signed) => Cc::LE,
        (MachineCompareKind::Le, MachineSign::Unsigned) => Cc::BE,
        (MachineCompareKind::Ge, MachineSign::Signed) => Cc::GE,
        (MachineCompareKind::Ge, MachineSign::Unsigned) => Cc::AE,
    }
}

/// Map Wasm float comparison kind to x86_64 condition code.
///
/// x86_64 UCOMISD/UCOMISS set flags:
///   ordered & equal: ZF=1, PF=0, CF=0 → use E (but need to check PF for NaN)
///   unordered (NaN): ZF=1, PF=1, CF=1
///
/// Wasm semantics: NaN is false for all relations except Ne (Ne is true for NaN).
pub(super) fn map_float_cond(kind: MachineCompareKind) -> Cc {
    match kind {
        MachineCompareKind::Eq => Cc::E,
        MachineCompareKind::Ne => Cc::NE,
        MachineCompareKind::Lt => Cc::B,
        MachineCompareKind::Gt => Cc::A,
        MachineCompareKind::Le => Cc::BE,
        MachineCompareKind::Ge => Cc::AE,
    }
}
