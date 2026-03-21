use super::MachineReg;

/// Fixed machine-register roles shared by all MachineIR backends.
pub const MACHINE_CTX_REG: MachineReg = MachineReg(0);
pub const MACHINE_FP_REG: MachineReg = MachineReg(1);
pub const MACHINE_MEM0_BASE_REG: MachineReg = MachineReg(2);
pub const MACHINE_MEM0_SIZE_REG: MachineReg = MachineReg(3);
// Number of above fixed registers.
pub const MACHINE_FIXED_REG_COUNT: u16 = 4;
