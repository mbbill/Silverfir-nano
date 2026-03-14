use alloc::vec::Vec;

use super::types::{MachineBlockId, MachineConstId, MachineFuncId};
use crate::vm::native::ir::runtime::MachineExternBinding;

/// One read-only sidecar constant record referenced from machine IR.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MachineConstData {
    pub id: MachineConstId,
    pub align: u32,
    pub bytes: Vec<u8>,
}

/// Full machine program for one function.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MachineProgram {
    pub entry: MachineBlockId,
    /// Registers `[first_fp_reg, reg_count)` belong to the FP-only transient bank.
    pub first_fp_reg: u16,
    pub reg_count: u16,
    pub blocks: Vec<super::cfg::MachineBlock>,
}

impl MachineProgram {
    #[inline]
    pub fn is_fp_reg(&self, reg: super::MachineReg) -> bool {
        reg.0 >= self.first_fp_reg && reg.0 < self.reg_count
    }

    #[inline]
    pub fn is_gp_reg(&self, reg: super::MachineReg) -> bool {
        reg.0 < self.first_fp_reg
    }
}

/// One machine function inside a machine module.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MachineFunction {
    pub id: MachineFuncId,
    pub program: MachineProgram,
}

/// One full machine module.
///
/// This is the shared allocation domain for function ids and sidecar constant
/// ids used by the machine IR, plus opaque external target ids used by helper
/// calls and other out-of-line native targets.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MachineModule {
    pub functions: Vec<MachineFunction>,
    pub consts: Vec<MachineConstData>,
    pub externs: Vec<MachineExternBinding>,
}

impl MachineModule {
    /// Run ISA-agnostic optimization passes on all functions.
    ///
    /// `first_transient` is the register index where transient (SSA-like)
    /// registers start. Registers below this are fixed or cached-local and
    /// must not be disturbed by peephole optimizations.
    pub fn optimize(&mut self, first_transient: u16) {
        for func in &mut self.functions {
            super::peephole::optimize(&mut func.program, first_transient);
        }
    }
}
