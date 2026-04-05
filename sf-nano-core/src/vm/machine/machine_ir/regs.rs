use super::MachineReg;
use crate::vm::backend::BackendConfig;

/// Fixed machine-register roles shared by all MachineIR backends.
pub(crate) const MACHINE_CTX_REG: MachineReg = MachineReg(0);
pub(crate) const MACHINE_FP_REG: MachineReg = MachineReg(1);
pub(crate) const MACHINE_MEM0_BASE_REG: MachineReg = MachineReg(2);
pub(crate) const MACHINE_MEM0_SIZE_REG: MachineReg = MachineReg(3);

/// Number of fixed registers — one past the highest fixed ID.
pub(crate) const MACHINE_FIXED_REG_COUNT: u16 = MACHINE_MEM0_SIZE_REG.0 + 1;

// BackendConfig::FIXED must agree with the actual fixed-reg count here.
const _: () = assert!(
    BackendConfig::FIXED == MACHINE_FIXED_REG_COUNT,
    "BackendConfig::FIXED must equal MACHINE_FIXED_REG_COUNT"
);

// ── Register layout: [fixed | gp_dynamic | fp_dynamic] ─────
//
// The helpers below are the single source of truth for bank order.
// `MachineRegFile::new()` allocates IDs in this order, and architecture
// backends use these to map abstract IDs back to physical registers.

/// Returns `true` if `reg` belongs to the FP bank.
#[inline]
pub(crate) fn is_fp_reg(reg: MachineReg, config: BackendConfig) -> bool {
    reg.0 >= config.first_fp_reg() && reg.0 < config.total_reg_count()
}

/// Returns `true` if `reg` belongs to the GP bank (fixed or dynamic).
#[inline]
pub(crate) fn is_gp_reg(reg: MachineReg, config: BackendConfig) -> bool {
    reg.0 < config.first_fp_reg()
}

/// Returns `true` if `reg` belongs to either GP or FP dynamic bank.
#[inline]
pub(crate) fn is_dynamic_reg(reg: MachineReg, config: BackendConfig) -> bool {
    reg.0 >= MACHINE_FIXED_REG_COUNT && reg.0 < config.total_reg_count()
}

/// Returns `true` if both regs are in the same bank (both GP or both FP).
#[inline]
pub(crate) fn same_reg_bank(lhs: MachineReg, rhs: MachineReg, config: BackendConfig) -> bool {
    let fp_start = config.first_fp_reg();
    (lhs.0 < fp_start) == (rhs.0 < fp_start)
}

/// FP-bank index: `reg.0 - first_fp_reg`. Returns `None` if not an FP reg.
#[inline]
pub(crate) fn fp_reg_index(reg: MachineReg, config: BackendConfig) -> Option<usize> {
    let fp_start = config.first_fp_reg();
    if reg.0 >= fp_start && reg.0 < config.total_reg_count() {
        Some((reg.0 - fp_start) as usize)
    } else {
        None
    }
}

/// Returns the GP dynamic-bank index for `reg`.
///
/// Returns `None` for fixed or FP-bank regs.
#[inline]
pub(crate) fn gp_dynamic_index(reg: MachineReg, config: BackendConfig) -> Option<usize> {
    if reg.0 < MACHINE_FIXED_REG_COUNT {
        return None;
    }
    let dynamic = (reg.0 - MACHINE_FIXED_REG_COUNT) as usize;
    (dynamic < config.gp_dynamic_budget as usize).then_some(dynamic)
}
