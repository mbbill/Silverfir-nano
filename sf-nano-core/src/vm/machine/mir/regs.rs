use super::MachineReg;

/// Fixed machine-register roles shared by all MachineIR backends.
pub(crate) const MACHINE_CTX_REG: MachineReg = MachineReg(0);
pub(crate) const MACHINE_FP_REG: MachineReg = MachineReg(1);
pub(crate) const MACHINE_MEM0_BASE_REG: MachineReg = MachineReg(2);
pub(crate) const MACHINE_MEM0_SIZE_REG: MachineReg = MachineReg(3);
// Number of above fixed registers.
pub(crate) const MACHINE_FIXED_REG_COUNT: u16 = 4;
