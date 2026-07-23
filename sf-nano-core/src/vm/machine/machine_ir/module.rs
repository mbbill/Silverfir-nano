use super::types::MachineFloatWidth;

use crate::collections;

use super::types::{MachineBlockId, MachineConstId, MachineFuncId, MachineReg};
use crate::vm::backend::BackendConfig;

/// One read-only constant-pool record referenced from machine IR.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct MachineConstData {
    pub id: MachineConstId,
    pub align: u32,
    pub bytes: collections::Vec<u8>,
}

/// Full machine program for one function.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct MachineProgram {
    pub entry: MachineBlockId,
    /// Initial semantic width for each FP-bank register, indexed by
    /// `reg - config.first_fp_reg()`. FP cached-local bindings use
    /// `Some(width)`; linear-value lanes start as `None`.
    pub fp_reg_init_widths: collections::Vec<Option<MachineFloatWidth>>,
    pub blocks: collections::Vec<super::cfg::MachineBlock>,
}

/// One machine function inside a machine module.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct MachineFunction {
    pub id: MachineFuncId,
    pub program: MachineProgram,
    /// Abstract dynamic registers in the preserved JIT-ABI class that this
    /// function body defines. Final MachineIR optimization populates this
    /// derived metadata before backends map the registers to physical lanes
    /// and save/restore exactly this set in the body host-stack frame.
    pub preserved_clobbers: collections::Vec<MachineReg>,
}

/// One full machine module.
///
/// This is the shared allocation domain for function ids and constant-pool ids
/// referenced by the machine IR.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MachineModule {
    pub config: BackendConfig,
    pub functions: collections::Vec<MachineFunction>,
    pub consts: collections::Vec<MachineConstData>,
}
