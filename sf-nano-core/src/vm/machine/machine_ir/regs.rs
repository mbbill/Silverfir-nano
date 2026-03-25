use super::MachineReg;
use crate::vm::backend::BackendConfig;

/// Fixed machine-register roles shared by all MachineIR backends.
pub(crate) const MACHINE_CTX_REG: MachineReg = MachineReg(0);
pub(crate) const MACHINE_FP_REG: MachineReg = MachineReg(1);
pub(crate) const MACHINE_MEM0_BASE_REG: MachineReg = MachineReg(2);
pub(crate) const MACHINE_MEM0_SIZE_REG: MachineReg = MachineReg(3);

/// Number of fixed registers — one past the highest fixed ID.
pub(crate) const MACHINE_FIXED_REG_COUNT: u16 = MACHINE_MEM0_SIZE_REG.0 + 1;

/// MachineIR register layout: `[fixed | gp_cache | gp_trans | fp_trans | fp_cache]`.
///
/// This is the single source of truth for the partition order.
/// `MachineRegFile::new()` allocates IDs in this order, and architecture
/// backends use `classify_gp_reg` / `classify_fp_reg` to map IDs back to
/// partition indices.

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
