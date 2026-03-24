//! Pure selector helpers.
//!
//! Every function here takes already-mapped physical registers and returns
//! an optional encoded instruction word. No MachineReg mapping, no scratch
//! allocation, no mutable state.

use crate::vm::machine::machine_ir::{MachineCompareKind, MachineIntWidth, MachineSign};

use super::{
    enc::{self, Cond},
    reg::GpReg,
};

// ── Integer immediate selection ──────────────────────────────────────────────

/// Try to encode `dst = src + imm` as a single add-immediate instruction.
#[inline]
pub(super) fn add_imm_inst(dst: GpReg, src: GpReg, imm: u64) -> Option<u32> {
    (imm < 4096).then(|| enc::add_imm_64(dst, src, imm as u16))
}

/// Try to encode `dst = src - imm` as a single sub-immediate instruction.
#[inline]
pub(super) fn sub_imm_inst(dst: GpReg, src: GpReg, imm: u64) -> Option<u32> {
    (imm < 4096).then(|| enc::sub_imm_64(dst, src, imm as u16))
}

/// Try to encode `CMP src, imm` as a single compare-immediate instruction.
#[inline]
pub(super) fn cmp_imm_inst(_width: MachineIntWidth, lhs: GpReg, rhs: u64) -> Option<u32> {
    (rhs < 4096).then(|| enc::cmp_imm_64(lhs, rhs as u16))
}

// ── Condition code mapping ───────────────────────────────────────────────────

pub(super) fn map_int_cond(kind: MachineCompareKind, sign: MachineSign) -> Cond {
    match (kind, sign) {
        (MachineCompareKind::Eq, _) => Cond::Eq,
        (MachineCompareKind::Ne, _) => Cond::Ne,
        (MachineCompareKind::Lt, MachineSign::Signed) => Cond::Lt,
        (MachineCompareKind::Lt, MachineSign::Unsigned) => Cond::Lo,
        (MachineCompareKind::Gt, MachineSign::Signed) => Cond::Gt,
        (MachineCompareKind::Gt, MachineSign::Unsigned) => Cond::Hi,
        (MachineCompareKind::Le, MachineSign::Signed) => Cond::Le,
        (MachineCompareKind::Le, MachineSign::Unsigned) => Cond::Ls,
        (MachineCompareKind::Ge, MachineSign::Signed) => Cond::Ge,
        (MachineCompareKind::Ge, MachineSign::Unsigned) => Cond::Hs,
    }
}
