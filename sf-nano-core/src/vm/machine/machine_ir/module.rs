use super::types::MachineFloatWidth;
use alloc::vec::Vec;

use crate::vm::backend::BackendConfig;
use super::super::peephole;
use super::types::{MachineBlockId, MachineConstId, MachineFuncId};
use super::contract::MachineExternBinding;

/// One read-only sidecar constant record referenced from machine IR.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct MachineConstData {
    pub id: MachineConstId,
    pub align: u32,
    pub bytes: Vec<u8>,
}

/// Full machine program for one function.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct MachineProgram {
    pub entry: MachineBlockId,
    /// Initial semantic width for each FP-bank register, indexed by
    /// `reg - config.first_fp_reg()`. FP cached-local regs use `Some(width)`;
    /// transient regs start as `None`.
    pub fp_reg_init_widths: Vec<Option<MachineFloatWidth>>,
    pub blocks: Vec<super::cfg::MachineBlock>,
}

/// One machine function inside a machine module.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct MachineFunction {
    pub id: MachineFuncId,
    pub program: MachineProgram,
}

/// One full machine module.
///
/// This is the shared allocation domain for function ids and sidecar constant
/// ids used by the machine IR, plus opaque external target ids used by helper
/// calls and other out-of-line native targets.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct MachineModule {
    pub config: BackendConfig,
    pub functions: Vec<MachineFunction>,
    pub consts: Vec<MachineConstData>,
    pub externs: Vec<MachineExternBinding>,
}

impl MachineModule {
    /// Run ISA-agnostic optimization passes on all functions.
    pub(crate) fn optimize(&mut self) {
        let config = self.config;
        for func in &mut self.functions {
            peephole::optimize(&mut func.program, config);
        }
    }
}
