use super::MachineReg;

/// Fixed machine-register roles shared by all MachineIR backends.
pub const MACHINE_CTX_REG: MachineReg = MachineReg(0);
pub const MACHINE_FP_REG: MachineReg = MachineReg(1);
pub const MACHINE_SCRATCH0_REG: MachineReg = MachineReg(2);
pub const MACHINE_SCRATCH1_REG: MachineReg = MachineReg(3);
pub const MACHINE_FIXED_REG_COUNT: u16 = 4;

#[inline]
pub const fn machine_scratch_reg(index: usize) -> Option<MachineReg> {
    match index {
        0 => Some(MACHINE_SCRATCH0_REG),
        1 => Some(MACHINE_SCRATCH1_REG),
        _ => None,
    }
}
