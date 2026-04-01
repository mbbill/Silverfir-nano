use super::types::MachineFloatWidth;
use alloc::vec::Vec;

use crate::vm::backend::BackendConfig;
use super::types::{MachineBlockId, MachineConstId, MachineFuncId};

/// One read-only constant-pool record referenced from machine IR.
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
/// This is the shared allocation domain for function ids and constant-pool ids
/// referenced by the machine IR.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MachineModule {
    pub config: BackendConfig,
    pub functions: Vec<MachineFunction>,
    pub consts: Vec<MachineConstData>,
}
