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

// ── Register layout: [fixed | gp_cache | gp_trans | fp_trans | fp_cache] ─────
//
// The helpers below are the single source of truth for the partition order.
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

/// Returns `true` if `reg` is a GP transient (not fixed, not cache).
#[inline]
pub(crate) fn is_gp_transient(reg: MachineReg, config: BackendConfig) -> bool {
    reg.0 >= config.first_gp_transient() && reg.0 < config.first_fp_reg()
}

/// Returns `true` if `reg` is an FP transient (not cache).
#[inline]
pub(crate) fn is_fp_transient(reg: MachineReg, config: BackendConfig) -> bool {
    fp_reg_index(reg, config)
        .map(|index| index < config.fp_transient_budget as usize)
        .unwrap_or(false)
}

/// Returns `true` if `reg` is any transient register.
#[inline]
pub(crate) fn is_transient_reg(reg: MachineReg, config: BackendConfig) -> bool {
    is_gp_transient(reg, config) || is_fp_transient(reg, config)
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

/// Classify a GP MachineReg into its partition index.
///
/// Returns `None` for fixed regs (those use `map_fixed_reg`).
/// Returns `Some((index, is_cache))` where `index` is the offset within the
/// gp_local_cache or gp_transient array.
#[inline]
pub(crate) fn classify_gp_reg(reg: MachineReg, config: BackendConfig) -> Option<(usize, bool)> {
    if reg.0 < MACHINE_FIXED_REG_COUNT {
        return None;
    }
    let dynamic = (reg.0 - MACHINE_FIXED_REG_COUNT) as usize;
    let cache_count = config.gp_local_cache_budget as usize;
    if dynamic < cache_count {
        Some((dynamic, true))
    } else {
        Some((dynamic - cache_count, false))
    }
}

/// Classify an FP MachineReg into its partition index.
///
/// `fp_index` is `reg.0 - first_fp_reg` (the caller subtracts the GP span).
/// Returns `(index, is_cache)` where `index` is the offset within the
/// fp_transient or fp_local_cache array.
#[inline]
pub(crate) fn classify_fp_reg(fp_index: usize, config: BackendConfig) -> (usize, bool) {
    let trans_count = config.fp_transient_budget as usize;
    if fp_index < trans_count {
        (fp_index, false)
    } else {
        (fp_index - trans_count, true)
    }
}
